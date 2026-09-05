#!/usr/bin/env python3
"""Convert saved raw PICO/XRobot recordings to AMASS-style SMPL-X NPZ files.

Raw PICO recordings contain global XRobot joint poses, rather than SMPL
parameters.  This converter writes their local joint rotations in SMPL-X body
order.  PICO and SMPL-X use opposite pelvis local X/Z axes, so the global
root gets the matching local +180-degree Y basis change and children get its
conjugation.  The saved PICO pelvis position is converted to SMPL-X ``trans``
so that SMPL-X's rendered pelvis joint remains at that same world position.

For directory inputs, the source directory layout is preserved.  A source
name ending in ``_raw.pkl`` is saved without the ``_raw`` suffix as ``.npz``.

Example:
    python scripts/convert_pico_raw_to_amass_smplx.py \
      --raw_pico_dir pico/pico_0712/raw \
      --output_dir pico/pico_0712/amass_smplx \
      --motion_fps 100
"""

from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R


SMPLX_POSE_DIM = 165
SMPLX_BODY_POSE_DIM = 63
SMPLX_HAND_POSE_DIM = 90
SMPLX_JAW_POSE_DIM = 3
SMPLX_EYE_POSE_DIM = 6
SMPLX_NUM_BETAS = 16

XROBOT_BODY_JOINT_NAMES = (
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
)

# The first 22 SMPL-X joints are global orientation plus its body pose.  This
# order matches SMPL-X ``full_pose`` and the AMASS ``poses`` convention.
SMPLX_BODY_SOURCE_NAMES = (
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
)

# Parent indices in ``SMPLX_BODY_SOURCE_NAMES``.  PICO has two additional
# hand joints after the wrists; they are deliberately excluded because they
# are not part of SMPL-X's 21-joint body pose.
SMPLX_BODY_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
)

UNITY_TO_GMR_ROTATION_MATRIX = np.array(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)
UNITY_TO_GMR_ROTATION = R.from_matrix(UNITY_TO_GMR_ROTATION_MATRIX)
# PICO and SMPL-X have the same world frame, but their pelvis/body local bases
# differ by +pi about local Y.  For child joints this is a conjugation; at the
# root it is a right multiplication because the source quaternion is global.
PICO_TO_SMPLX_LOCAL_BASIS = R.from_rotvec(np.array([0.0, np.pi, 0.0]))
SMPLX_MODEL_ROOT = Path(__file__).resolve().parents[1] / "assets" / "body_models" / "smplx"


def safe_load_pickle(path: Path) -> Any:
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as handle:
            return pickle.load(handle)


def load_raw_pico_frames(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = safe_load_pickle(path)
    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        return payload["frames"], payload.get("meta", {})
    if isinstance(payload, list):
        return payload, {}
    raise ValueError(f"Unsupported raw PICO pickle schema: {path}")


def collect_raw_pico_files(raw_pico_dir: Path) -> list[Path]:
    raw_files = sorted(path for path in raw_pico_dir.rglob("*_raw.pkl") if path.is_file())
    return raw_files or sorted(path for path in raw_pico_dir.rglob("*.pkl") if path.is_file())


def output_path_for_raw_file(raw_path: Path, raw_pico_dir: Path, output_dir: Path) -> Path:
    relative_path = raw_path.relative_to(raw_pico_dir)
    stem = relative_path.stem.removesuffix("_raw")
    return output_dir / relative_path.with_name(f"{stem}.npz")


@lru_cache(maxsize=None)
def smplx_rest_pelvis_offset(gender: str) -> np.ndarray:
    """Return the zero-beta SMPL-X pelvis location relative to model origin."""
    model_path = SMPLX_MODEL_ROOT / f"SMPLX_{gender.upper()}.npz"
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL-X body model not found: {model_path}")
    with np.load(model_path, allow_pickle=True) as model_data:
        joints = (
            np.asarray(model_data["J_regressor"], dtype=np.float64)
            @ np.asarray(model_data["v_template"], dtype=np.float64)
        )
    return joints[0].astype(np.float32)


def normalize_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat_wxyz)
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Invalid zero-norm quaternion in raw PICO frame")
    return quat_wxyz / norm


def body_data_from_sdk_raw(sdk_raw_data: dict[str, Any] | None) -> dict[str, list[list[float]]]:
    if not isinstance(sdk_raw_data, dict):
        return {}
    raw_body = sdk_raw_data.get("body")
    if not isinstance(raw_body, dict) or raw_body.get("poses") is None:
        return {}

    body_data: dict[str, list[list[float]]] = {}
    for joint_name, raw_pose in zip(XROBOT_BODY_JOINT_NAMES, raw_body["poses"]):
        values = np.asarray(raw_pose, dtype=np.float64)
        if values.shape[0] < 7:
            continue
        position = values[:3] @ UNITY_TO_GMR_ROTATION_MATRIX.T
        orientation = (
            UNITY_TO_GMR_ROTATION * R.from_quat(normalize_quat_wxyz(values[[6, 3, 4, 5]])[[1, 2, 3, 0]])
        ).as_quat()
        body_data[joint_name] = [
            position.tolist(),
            orientation[[3, 0, 1, 2]].tolist(),
        ]
    return body_data


def body_data_from_raw_frame(frame: dict[str, Any]) -> dict[str, Any]:
    body_data = frame.get("body")
    if isinstance(body_data, dict) and body_data:
        return copy.deepcopy(body_data)
    return body_data_from_sdk_raw(frame.get("sdk_raw"))


def rotation_from_pose(pose: Any) -> R | None:
    if not isinstance(pose, (tuple, list)) or len(pose) < 2:
        return None
    quat_wxyz = np.asarray(pose[1], dtype=np.float64)
    if quat_wxyz.shape != (4,):
        return None
    try:
        quat_wxyz = normalize_quat_wxyz(quat_wxyz)
    except ValueError:
        return None
    return R.from_quat(quat_wxyz[[1, 2, 3, 0]])


def extract_pico_body_rotations(
    frame: dict[str, Any],
) -> tuple[np.ndarray, tuple[R | None, ...]] | None:
    """Return pelvis translation and SMPL-X-ordered PICO local rotations.

    The saved PICO orientations are global.  The root remains global here;
    every other body joint is converted to a local PICO rotation using the
    actual SMPL-X parent hierarchy above.
    """
    body_data = body_data_from_raw_frame(frame)
    pelvis = body_data.get("Pelvis")
    if not isinstance(pelvis, (tuple, list)) or len(pelvis) < 2:
        return None
    trans = np.asarray(pelvis[0], dtype=np.float32)
    if trans.shape != (3,) or not np.all(np.isfinite(trans)):
        return None

    world_rotations = {
        source_name: rotation_from_pose(body_data.get(source_name))
        for source_name in SMPLX_BODY_SOURCE_NAMES
    }
    if world_rotations["Pelvis"] is None:
        return None

    local_rotations: list[R | None] = []
    for joint_index, source_name in enumerate(SMPLX_BODY_SOURCE_NAMES):
        joint_world = world_rotations[source_name]
        parent_index = SMPLX_BODY_PARENTS[joint_index]
        if joint_world is None:
            local_rotations.append(None)
            continue
        if parent_index < 0:
            local_rotations.append(joint_world)
            continue
        parent_world = world_rotations[SMPLX_BODY_SOURCE_NAMES[parent_index]]
        local_rotations.append(
            parent_world.inv() * joint_world if parent_world is not None else None
        )
    return trans, tuple(local_rotations)


def pico_local_rotations_to_smplx_pose(
    local_rotations: tuple[R | None, ...],
) -> np.ndarray:
    """Write PICO local rotations in SMPL-X body-pose order."""
    pose_axis_angle = np.zeros((SMPLX_POSE_DIM // 3, 3), dtype=np.float32)
    for joint_index, local_rotation in enumerate(local_rotations):
        if local_rotation is None:
            continue
        # The raw PICO quaternion is wxyz and already in the GMR world frame.
        # Do not apply a world-space axis transform here.  Instead convert the
        # PICO *local* pelvis basis to SMPL-X by post-multiplying it by Y180.
        # A child rotation has both its parent and child bases changed, hence
        # the conjugation below.
        smplx_rotation = (
            local_rotation * PICO_TO_SMPLX_LOCAL_BASIS
            if joint_index == 0
            else PICO_TO_SMPLX_LOCAL_BASIS
            * local_rotation
            * PICO_TO_SMPLX_LOCAL_BASIS.inv()
        )
        pose_axis_angle[joint_index] = smplx_rotation.as_rotvec().astype(np.float32)
    return pose_axis_angle.reshape(-1)


def estimate_fps(frames: list[dict[str, Any]], meta: dict[str, Any], fps_override: float | None, fallback_fps: float) -> float:
    if fps_override is not None:
        return fps_override
    for key in ("saved_fps", "target_fps", "fps"):
        if key in meta:
            fps = float(meta[key])
            if np.isfinite(fps) and fps > 0:
                return fps

    timestamps = np.asarray(
        [frame.get("t_record_unix") for frame in frames if isinstance(frame, dict) and frame.get("t_record_unix") is not None],
        dtype=np.float64,
    )
    if len(timestamps) >= 2:
        time_steps = np.diff(timestamps)
        time_steps = time_steps[time_steps > 1e-6]
        if len(time_steps):
            return float(1.0 / np.mean(time_steps))
    return fallback_fps


def build_amass_smplx_payload(
    frames: list[dict[str, Any]],
    meta: dict[str, Any],
    source: Path,
    gender: str,
    motion_fps: float | None,
    fallback_fps: float,
) -> tuple[dict[str, np.ndarray], int]:
    translations: list[np.ndarray] = []
    source_rotations: list[tuple[R | None, ...]] = []
    dropped_frames = 0
    for frame in frames:
        if not isinstance(frame, dict):
            dropped_frames += 1
            continue
        extracted = extract_pico_body_rotations(frame)
        if extracted is None:
            dropped_frames += 1
            continue
        trans, local_rotations = extracted
        translations.append(trans)
        source_rotations.append(local_rotations)

    if not source_rotations:
        raise ValueError(f"No valid PICO body frames found in {source}")

    poses = np.asarray(
        [pico_local_rotations_to_smplx_pose(rotations) for rotations in source_rotations],
        dtype=np.float32,
    )
    if poses.shape != (len(source_rotations), SMPLX_POSE_DIM):
        raise RuntimeError(f"Unexpected SMPL-X pose shape: {poses.shape}")

    # SMPL-X's ``transl`` locates the model origin, not its pelvis joint.  In
    # the zero-beta neutral model the pelvis is about 35 cm from that origin.
    # Passing the raw PICO pelvis position directly as ``trans`` therefore
    # adds a root-orientation-dependent XY drift.  Choose ``trans`` so the
    # generated SMPL-X pelvis lands exactly on the recorded PICO pelvis.
    raw_pelvis_positions = np.asarray(translations, dtype=np.float32)
    rest_pelvis_offset = smplx_rest_pelvis_offset(gender)
    world_pelvis_offset = R.from_rotvec(poses[:, :3]).apply(rest_pelvis_offset)
    trans = raw_pelvis_positions - world_pelvis_offset.astype(np.float32)

    fps = estimate_fps(frames, meta, motion_fps, fallback_fps)
    num_frames = len(source_rotations)
    return {
        "gender": np.array(gender),
        "surface_model_type": np.array("smplx"),
        "mocap_frame_rate": np.array(fps, dtype=np.float32),
        "mocap_time_length": np.array(num_frames / fps, dtype=np.float32),
        "markers_latent": np.zeros((0, 3), dtype=np.float32),
        "latent_labels": np.array([], dtype="<U1"),
        "markers_latent_vids": np.array({}, dtype=object),
        "trans": trans,
        "poses": poses,
        "betas": np.zeros(SMPLX_NUM_BETAS, dtype=np.float32),
        "num_betas": np.array(SMPLX_NUM_BETAS, dtype=np.int64),
        "root_orient": poses[:, :3],
        "pose_body": poses[:, 3 : 3 + SMPLX_BODY_POSE_DIM],
        "pose_jaw": poses[:, 66 : 66 + SMPLX_JAW_POSE_DIM],
        "pose_eye": poses[:, 69 : 69 + SMPLX_EYE_POSE_DIM],
        "pose_hand": poses[:, 75 : 75 + SMPLX_HAND_POSE_DIM],
    }, dropped_frames


def convert_file(
    source: Path,
    output: Path,
    gender: str,
    motion_fps: float | None,
    fallback_fps: float,
    overwrite: bool,
) -> tuple[bool, int, int]:
    if output.exists() and not overwrite:
        print(f"[skip] {output}")
        return False, 0, 0

    frames, meta = load_raw_pico_frames(source)
    payload, dropped_frames = build_amass_smplx_payload(
        frames,
        meta,
        source,
        gender,
        motion_fps,
        fallback_fps,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    print(
        f"[ok] {source} -> {output} "
        f"({payload['poses'].shape[0]}/{len(frames)} frames, "
        f"{payload['mocap_frame_rate'].item():g} fps)"
    )
    return True, len(frames), dropped_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw PICO/XRobot directory to AMASS-style SMPL-X NPZ files."
    )
    parser.add_argument("--raw_pico_dir", type=Path, required=True, help="Input raw PICO directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output AMASS-style SMPL-X directory.")
    parser.add_argument("--gender", choices=("neutral", "male", "female"), default="neutral")
    parser.add_argument(
        "--motion_fps",
        type=float,
        default=None,
        help="Override output FPS. Defaults to raw metadata, timestamps, then --fallback_fps.",
    )
    parser.add_argument("--fallback_fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing NPZ files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_pico_dir = args.raw_pico_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.motion_fps is not None and args.motion_fps <= 0:
        raise ValueError("--motion_fps must be positive")
    if args.fallback_fps <= 0:
        raise ValueError("--fallback_fps must be positive")
    if not raw_pico_dir.is_dir():
        raise NotADirectoryError(f"Raw PICO directory not found: {raw_pico_dir}")

    raw_files = collect_raw_pico_files(raw_pico_dir)
    if not raw_files:
        raise FileNotFoundError(f"No .pkl files found under {raw_pico_dir}")

    converted = 0
    skipped = 0
    failed = 0
    total_dropped_frames = 0
    for source in raw_files:
        output = output_path_for_raw_file(source, raw_pico_dir, output_dir)
        try:
            was_converted, _, dropped_frames = convert_file(
                source,
                output,
                args.gender,
                args.motion_fps,
                args.fallback_fps,
                args.overwrite,
            )
            converted += was_converted
            skipped += not was_converted
            total_dropped_frames += dropped_frames
        except Exception as exc:
            failed += 1
            print(f"[error] {source}: {type(exc).__name__}: {exc}")

    print(
        f"Finished: {converted} converted, {skipped} skipped, {failed} failed, "
        f"{total_dropped_frames} invalid frames dropped. output={output_dir}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
