import argparse
import copy
import pickle
from pathlib import Path
from typing import Any

import joblib
import mujoco as mj
import numpy as np
from rich import print
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import IK_CONFIG_DICT
from general_motion_retargeting.rot_utils import quat_mul_np
from general_motion_retargeting.utils.motion_utils import (
    build_motion_data,
    get_default_keybody_names,
)


XROBOT_BODY_JOINT_NAMES = [
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
]

UNITY_TO_GMR_ROTATION_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
UNITY_TO_GMR_ROTATION_QUAT_WXYZ = R.from_matrix(
    UNITY_TO_GMR_ROTATION_MATRIX
).as_quat(scalar_first=True)


def safe_load_pickle(path: Path) -> Any:
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as handle:
            return pickle.load(handle)


def collect_raw_pico_files(raw_pico_dir: Path) -> list[Path]:
    raw_files = sorted(path for path in raw_pico_dir.rglob("*_raw.pkl") if path.is_file())
    if raw_files:
        return raw_files
    return sorted(path for path in raw_pico_dir.rglob("*.pkl") if path.is_file())


def output_path_for_raw_file(raw_path: Path, raw_pico_dir: Path, output_dir: Path) -> Path:
    relative_path = raw_path.relative_to(raw_pico_dir)
    stem = relative_path.stem
    if stem.endswith("_raw"):
        stem = stem[:-4]
    return output_dir / relative_path.with_name(f"{stem}.pkl")


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        raise ValueError("Invalid zero-norm quaternion in raw PICO frame")
    return quat / norm


def coordinate_transform_unity_pose(pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> tuple[list[float], list[float]]:
    orientation = quat_mul_np(
        UNITY_TO_GMR_ROTATION_QUAT_WXYZ,
        normalize_quat_wxyz(quat_wxyz),
        scalar_first=True,
    )
    position = pos_xyz @ UNITY_TO_GMR_ROTATION_MATRIX.T
    return position.tolist(), orientation.tolist()


def body_data_from_sdk_raw(sdk_raw_data: dict[str, Any] | None) -> dict[str, list[list[float]]]:
    if not isinstance(sdk_raw_data, dict):
        return {}
    raw_body = sdk_raw_data.get("body")
    if not isinstance(raw_body, dict):
        return {}
    poses = raw_body.get("poses")
    if poses is None:
        return {}

    body_pose_dict = {}
    for joint_name, raw_pose in zip(XROBOT_BODY_JOINT_NAMES, poses):
        values = np.asarray(raw_pose, dtype=np.float64)
        if values.shape[0] < 7:
            continue
        pos = values[:3]
        quat_wxyz = values[[6, 3, 4, 5]]
        body_pose_dict[joint_name] = list(coordinate_transform_unity_pose(pos, quat_wxyz))
    return body_pose_dict


def body_data_from_raw_frame(frame: dict[str, Any]) -> dict[str, Any]:
    body_data = frame.get("body")
    if isinstance(body_data, dict) and body_data:
        return copy.deepcopy(body_data)
    return body_data_from_sdk_raw(frame.get("sdk_raw"))


def load_raw_pico_frames(raw_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = safe_load_pickle(raw_path)
    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        return payload["frames"], payload.get("meta", {})
    if isinstance(payload, list):
        return payload, {}
    raise ValueError(f"Unsupported raw PICO pickle schema in {raw_path}")


def estimate_fps(frames: list[dict[str, Any]], meta: dict[str, Any], fps_override: float | None) -> float:
    if fps_override is not None:
        return float(fps_override)
    for key in ("saved_fps", "target_fps", "fps"):
        if key in meta:
            fps = float(meta[key])
            if fps > 0.0:
                return fps

    timestamps = np.asarray(
        [
            frame.get("t_record_unix")
            for frame in frames
            if isinstance(frame, dict) and frame.get("t_record_unix") is not None
        ],
        dtype=np.float64,
    )
    if len(timestamps) >= 2:
        dt = np.diff(timestamps)
        dt = dt[dt > 1e-6]
        if len(dt) > 0:
            return float(1.0 / np.mean(dt))
    return 30.0


def build_motion_data_from_qpos(
    aligned_fps: float,
    retargeter: GMR,
    qpos: np.ndarray,
    keybody_names: list[str],
) -> dict[str, Any]:
    if qpos.ndim != 2:
        raise ValueError(f"qpos must have shape [T, nq], got {qpos.shape}")

    keybody_pairs = [
        (name, retargeter.robot_body_names[name])
        for name in keybody_names
        if name in retargeter.robot_body_names
    ]
    keybody_names = [name for name, _ in keybody_pairs]
    keybody_ids = [idx for _, idx in keybody_pairs]

    mj_data = mj.MjData(retargeter.model)
    keybody_pos_samples = []
    keybody_rot_samples = []
    for frame_qpos in qpos:
        mj_data.qpos[:] = frame_qpos
        mj.mj_forward(retargeter.model, mj_data)
        if keybody_ids:
            keybody_pos_samples.append(mj_data.xpos[keybody_ids].copy())
            keybody_rot_samples.append(mj_data.xquat[keybody_ids].copy())

    num_frames = qpos.shape[0]
    if keybody_pos_samples:
        keybody_pos_world = np.stack(keybody_pos_samples)
        keybody_rot_world_wxyz = np.stack(keybody_rot_samples)
    else:
        keybody_pos_world = np.zeros((num_frames, 0, 3), dtype=np.float64)
        keybody_rot_world_wxyz = np.zeros((num_frames, 0, 4), dtype=np.float64)

    return build_motion_data(
        aligned_fps=aligned_fps,
        root_pos=qpos[:, :3],
        root_rot_wxyz=qpos[:, 3:7],
        dof_pos=qpos[:, 7:],
        keybody_pos_world=keybody_pos_world,
        keybody_rot_world_wxyz=keybody_rot_world_wxyz,
        keybody_names=keybody_names,
        local_body_pos=None,
        local_body_link_body_list=None,
    )


def retarget_raw_pico_file(
    raw_path: Path,
    raw_pico_dir: Path,
    output_dir: Path,
    robot: str,
    actual_human_height: float,
    fps_override: float | None,
    overwrite: bool,
) -> Path | None:
    output_path = output_path_for_raw_file(raw_path, raw_pico_dir, output_dir)
    if output_path.exists() and not overwrite:
        print(f"[yellow]Skip existing output:[/yellow] {output_path}")
        return None

    frames, meta = load_raw_pico_frames(raw_path)
    if not frames:
        raise ValueError(f"No raw PICO frames in {raw_path}")

    target_robot = robot
    if robot not in IK_CONFIG_DICT["xrobot"]:
        target_robot = "unitree_g1"
        print(f"[yellow]xrobot IK config for {robot} not found. Fallback to {target_robot}.[/yellow]")

    retargeter = GMR(
        src_human="xrobot",
        tgt_robot=target_robot,
        actual_human_height=actual_human_height,
        verbose=False,
    )

    qpos_list = []
    dropped_frames = 0
    for frame in tqdm(frames, desc=raw_path.name, leave=False):
        if not isinstance(frame, dict):
            dropped_frames += 1
            continue
        body_data = body_data_from_raw_frame(frame)
        if not body_data:
            dropped_frames += 1
            continue
        qpos = retargeter.retarget(body_data, offset_to_ground=True)
        qpos_list.append(qpos.copy())

    if not qpos_list:
        raise ValueError(f"No valid body frames found in {raw_path}")

    aligned_fps = estimate_fps(frames, meta, fps_override)
    motion_data = build_motion_data_from_qpos(
        aligned_fps=aligned_fps,
        retargeter=retargeter,
        qpos=np.asarray(qpos_list, dtype=np.float64),
        keybody_names=get_default_keybody_names(target_robot),
    )
    motion_data["retarget_meta"] = {
        "source": "pico_offline_to_robot",
        "source_raw_file": str(raw_path),
        "robot": target_robot,
        "requested_robot": robot,
        "actual_human_height": float(actual_human_height),
        "num_input_frames": len(frames),
        "num_output_frames": len(qpos_list),
        "dropped_frames": dropped_frames,
        "fps": float(aligned_fps),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(motion_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline retarget saved raw PICO/XRobot data to GMR robot motion pkl files."
    )
    parser.add_argument("--raw_pico_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "robros_igris_c_v2"],
        default="robros_igris_c_v2",
    )
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.5,
        help="Actual human height used by xrobot GMR retargeting.",
    )
    parser.add_argument(
        "--motion_fps",
        type=float,
        default=None,
        help="Override output fps. Defaults to raw metadata or timestamp estimate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_pico_dir = args.raw_pico_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not raw_pico_dir.is_dir():
        raise NotADirectoryError(f"Raw PICO directory not found: {raw_pico_dir}")

    raw_files = collect_raw_pico_files(raw_pico_dir)
    if not raw_files:
        raise FileNotFoundError(f"No .pkl files found under {raw_pico_dir}")

    saved_count = 0
    skipped_count = 0
    for raw_path in tqdm(raw_files, desc="PICO offline retarget"):
        output_path = retarget_raw_pico_file(
            raw_path=raw_path,
            raw_pico_dir=raw_pico_dir,
            output_dir=output_dir,
            robot=args.robot,
            actual_human_height=args.actual_human_height,
            fps_override=args.motion_fps,
            overwrite=args.overwrite,
        )
        if output_path is None:
            skipped_count += 1
        else:
            saved_count += 1
            print(f"Saved {output_path}")

    print(
        f"Done. saved={saved_count}, skipped={skipped_count}, "
        f"input={raw_pico_dir}, output={output_dir}"
    )


if __name__ == "__main__":
    main()
