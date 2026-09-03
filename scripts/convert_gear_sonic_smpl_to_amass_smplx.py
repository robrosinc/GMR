#!/usr/bin/env python3
"""Convert GEAR-SONIC ``smpl_filtered`` joblib PKLs to AMASS-style SMPL-X NPZs.

The filtered GEAR-SONIC files store an SMPL pose in ``pose_aa`` with shape
``(frames, 72)`` (global orientation plus 23 SMPL joints), root translation
in ``transl``, and the frame rate in ``fps``.  AMASS SMPL-X uses the first 22
of those SMPL joints: global orientation plus 21 body joints.  SMPL's final
two hand joints have no direct body-pose slots in SMPL-X, so they are omitted;
the SMPL-X articulated hand and face poses are written as zeros.

The resulting NPZs contain the fields consumed by ``vis_raw_smplx_motion.py``
and by the GMR SMPL-X retargeting scripts.

Examples:
    # Convert one motion.
    python scripts/convert_gear_sonic_smpl_to_amass_smplx.py \
      /path/to/motion.pkl /path/to/motion.npz

    # Correct a coordinate-frame mismatch: +90 degrees around world Y.
    python scripts/convert_gear_sonic_smpl_to_amass_smplx.py \
      /path/to/motion.pkl /path/to/motion_y90.npz \
      --root-rotation-axis y --root-rotation-degrees 90

    # Convert a directory while preserving its directory layout.
    python scripts/convert_gear_sonic_smpl_to_amass_smplx.py \
      /path/to/smpl_filtered /path/to/amass_smplx --skip-existing
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


SMPL_POSE_DIM = 72
SMPLX_BODY_POSE_DIM = 63
SMPLX_HAND_POSE_DIM = 90
SMPLX_JAW_POSE_DIM = 3
SMPLX_EYE_POSE_DIM = 6
SMPLX_POSE_DIM = 165
SMPLX_NUM_BETAS = 16


def as_float32_array(value: Any, name: str, source: Path) -> np.ndarray:
    """Return a finite numeric array as float32 with a useful error message."""
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{source}: {name} contains NaN or Inf")
    return array


def load_gear_sonic_motion(source: Path) -> dict[str, Any]:
    """Load a compressed joblib PKL produced for GEAR-SONIC training."""
    try:
        payload = joblib.load(source)
    except Exception as exc:
        raise ValueError(f"Failed to load GEAR-SONIC joblib file: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: expected a dictionary, got {type(payload).__name__}")
    return payload


def get_fps(payload: dict[str, Any], source: Path, fallback_fps: float) -> float:
    value = payload.get("fps", payload.get("original_fps", fallback_fps))
    try:
        fps = float(np.asarray(value).item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid fps value {value!r}") from exc
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{source}: fps must be a positive finite number, got {fps}")
    return fps


def rotate_world_root(
    root_orient: np.ndarray,
    trans: np.ndarray,
    axis: str,
    degrees: float,
    pivot: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one world-space rotation to the root orientation and trajectory.

    ``pose_body`` remains local to the root and therefore must not be rotated.
    Translation is rotated with the same transform so the root orientation and
    its world-space trajectory stay consistent.  The default ``first`` pivot
    leaves the first root translation unchanged; ``origin`` is useful when the
    source and target coordinate systems share the same world origin.
    """
    if degrees == 0.0:
        return root_orient, trans

    world_rotation = Rotation.from_euler(axis, degrees, degrees=True)
    rotated_root_orient = (
        world_rotation * Rotation.from_rotvec(root_orient)
    ).as_rotvec().astype(np.float32, copy=False)
    translation_pivot = np.zeros(3, dtype=np.float32) if pivot == "origin" else trans[0]
    rotated_trans = (
        world_rotation.apply(trans - translation_pivot) + translation_pivot
    ).astype(np.float32, copy=False)
    return rotated_root_orient, rotated_trans


def build_amass_smplx_payload(
    payload: dict[str, Any],
    source: Path,
    gender: str,
    fallback_fps: float,
    root_rotation_axis: str,
    root_rotation_degrees: float,
    root_rotation_pivot: str,
) -> dict[str, np.ndarray]:
    if "pose_aa" not in payload:
        raise KeyError(f"{source}: missing required key 'pose_aa'")
    if "transl" not in payload:
        raise KeyError(f"{source}: missing required key 'transl'")

    smpl_pose = as_float32_array(payload["pose_aa"], "pose_aa", source)
    trans = as_float32_array(payload["transl"], "transl", source)
    if smpl_pose.ndim != 2 or smpl_pose.shape[1] != SMPL_POSE_DIM:
        raise ValueError(
            f"{source}: pose_aa must have shape (frames, {SMPL_POSE_DIM}), "
            f"got {smpl_pose.shape}"
        )
    if smpl_pose.shape[0] == 0:
        raise ValueError(f"{source}: motion contains no frames")
    if trans.shape != (smpl_pose.shape[0], 3):
        raise ValueError(
            f"{source}: transl must have shape ({smpl_pose.shape[0]}, 3), "
            f"got {trans.shape}"
        )

    num_frames = smpl_pose.shape[0]
    root_orient = smpl_pose[:, :3]
    root_orient, trans = rotate_world_root(
        root_orient,
        trans,
        axis=root_rotation_axis,
        degrees=root_rotation_degrees,
        pivot=root_rotation_pivot,
    )
    pose_body = smpl_pose[:, 3 : 3 + SMPLX_BODY_POSE_DIM]
    pose_hand = np.zeros((num_frames, SMPLX_HAND_POSE_DIM), dtype=np.float32)
    pose_jaw = np.zeros((num_frames, SMPLX_JAW_POSE_DIM), dtype=np.float32)
    pose_eye = np.zeros((num_frames, SMPLX_EYE_POSE_DIM), dtype=np.float32)
    poses = np.concatenate(
        [root_orient, pose_body, pose_jaw, pose_eye, pose_hand], axis=1
    ).astype(np.float32, copy=False)
    if poses.shape != (num_frames, SMPLX_POSE_DIM):
        raise RuntimeError(f"Internal error: unexpected SMPL-X pose shape {poses.shape}")

    fps = get_fps(payload, source, fallback_fps)
    return {
        "gender": np.array(gender),
        "surface_model_type": np.array("smplx"),
        "mocap_frame_rate": np.array(fps, dtype=np.float32),
        "mocap_time_length": np.array(num_frames / fps, dtype=np.float32),
        # These AMASS marker fields are empty because GEAR-SONIC's filtered
        # files contain only SMPL parameters, not marker trajectories.
        "markers_latent": np.zeros((0, 3), dtype=np.float32),
        "latent_labels": np.array([], dtype="<U1"),
        "markers_latent_vids": np.array({}, dtype=object),
        "trans": trans,
        "poses": poses,
        "betas": np.zeros(SMPLX_NUM_BETAS, dtype=np.float32),
        "num_betas": np.array(SMPLX_NUM_BETAS, dtype=np.int64),
        "root_orient": root_orient,
        "pose_body": pose_body,
        "pose_hand": pose_hand,
        "pose_jaw": pose_jaw,
        "pose_eye": pose_eye,
    }


def convert_file(
    source: Path,
    output: Path,
    gender: str,
    fallback_fps: float,
    root_rotation_axis: str,
    root_rotation_degrees: float,
    root_rotation_pivot: str,
    overwrite: bool,
) -> bool:
    if output.exists() and not overwrite:
        print(f"[skip] {output}")
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_amass_smplx_payload(
        load_gear_sonic_motion(source),
        source,
        gender,
        fallback_fps,
        root_rotation_axis,
        root_rotation_degrees,
        root_rotation_pivot,
    )
    np.savez(output, **payload)
    print(f"[ok] {source} -> {output} ({payload['pose_body'].shape[0]} frames)")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GEAR-SONIC smpl_filtered joblib PKLs to AMASS-style SMPL-X NPZs."
        )
    )
    parser.add_argument("input_path", type=Path, help="A .pkl file or directory of .pkl files.")
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output .npz file for a single input, or output directory for a directory input.",
    )
    parser.add_argument(
        "--gender",
        choices=("neutral", "male", "female"),
        default="neutral",
        help="SMPL-X model gender written to the output (default: neutral).",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=30.0,
        help="FPS used only if the source has neither fps nor original_fps (default: 30).",
    )
    parser.add_argument(
        "--root-rotation-axis",
        choices=("x", "y", "z"),
        default="y",
        help="World axis for the root coordinate correction (default: y).",
    )
    parser.add_argument(
        "--root-rotation-degrees",
        type=float,
        default=0.0,
        help=(
            "World-space root correction in degrees. This rotates root_orient "
            "and trans together; pose_body stays local and unchanged (default: 0)."
        ),
    )
    parser.add_argument(
        "--root-rotation-pivot",
        choices=("first", "origin"),
        default="first",
        help=(
            "Pivot used when rotating trans: first keeps the first root position fixed; "
            "origin rotates around world origin (default: first)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Explicitly skip existing files (this is the default unless --overwrite is set).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if args.fallback_fps <= 0:
        raise ValueError("--fallback-fps must be positive")
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".pkl":
            raise ValueError(f"Input file must be a .pkl: {input_path}")
        if output_path.suffix.lower() != ".npz":
            raise ValueError("For a single .pkl input, output_path must end in .npz")
        convert_file(
            input_path,
            output_path,
            args.gender,
            args.fallback_fps,
            args.root_rotation_axis,
            args.root_rotation_degrees,
            args.root_rotation_pivot,
            args.overwrite,
        )
        return

    source_paths = sorted(path for path in input_path.rglob("*.pkl") if path.is_file())
    if not source_paths:
        raise FileNotFoundError(f"No .pkl files found under: {input_path}")
    if output_path.suffix.lower() == ".npz":
        raise ValueError("For a directory input, output_path must be a directory, not an .npz file")

    converted = 0
    failed = 0
    for source in source_paths:
        output = output_path / source.relative_to(input_path).with_suffix(".npz")
        try:
            converted += convert_file(
                source,
                output,
                args.gender,
                args.fallback_fps,
                args.root_rotation_axis,
                args.root_rotation_degrees,
                args.root_rotation_pivot,
                args.overwrite,
            )
        except Exception as exc:
            failed += 1
            print(f"[error] {source}: {exc}")
    print(
        f"Finished: {converted} converted, {len(source_paths) - converted - failed} skipped, "
        f"{failed} failed."
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
