#!/usr/bin/env python3
"""Lightweight MuJoCo viewer for raw SMPL-X npz files and skeleton pickles."""

from __future__ import annotations

import argparse
import copy
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich import print

from vis_controls import (
    create_control_state,
    log_controls_once,
    make_keyboard_callback,
)

mj = None
mjv = None


SKELETON_BODY_JOINT_NAMES = [
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

SKELETON_BODY_EDGES = (
    ("Pelvis", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Spine3"),
    ("Spine3", "Neck"),
    ("Neck", "Head"),
    ("Spine3", "Left_Collar"),
    ("Left_Collar", "Left_Shoulder"),
    ("Left_Shoulder", "Left_Elbow"),
    ("Left_Elbow", "Left_Wrist"),
    ("Left_Wrist", "Left_Hand"),
    ("Spine3", "Right_Collar"),
    ("Right_Collar", "Right_Shoulder"),
    ("Right_Shoulder", "Right_Elbow"),
    ("Right_Elbow", "Right_Wrist"),
    ("Right_Wrist", "Right_Hand"),
    ("Pelvis", "Left_Hip"),
    ("Left_Hip", "Left_Knee"),
    ("Left_Knee", "Left_Ankle"),
    ("Left_Ankle", "Left_Foot"),
    ("Pelvis", "Right_Hip"),
    ("Right_Hip", "Right_Knee"),
    ("Right_Knee", "Right_Ankle"),
    ("Right_Ankle", "Right_Foot"),
)

SKELETON_HAND_FINGER_CHAINS = (
    ("ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip"),
    ("IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip"),
    ("MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip"),
    ("RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip"),
    ("LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip"),
)

UNITY_TO_GMR_ROTATION_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

BODY_LINE_RGBA = np.array([0.1, 0.85, 1.0, 0.8])
HAND_LINE_RGBA = np.array([1.0, 0.55, 0.2, 0.75])
JOINT_RGBA = np.array([1.0, 0.9, 0.25, 0.9])

SCENE_XML = """
<mujoco model="raw_smplx_viewer">
  <visual>
    <global azimuth="135" elevation="-18"/>
    <headlight ambient="0.28 0.28 0.28" diffuse="0.32 0.32 0.32" specular="0.08 0.08 0.08"/>
    <rgba haze="0.9 0.93 0.96 1"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.72 0.8 0.88" rgb2="0.96 0.98 1" width="512" height="512"/>
    <texture name="checker" type="2d" builtin="checker" width="512" height="512" rgb1="0.82 0.83 0.8" rgb2="0.55 0.59 0.62"/>
    <material name="checker" texture="checker" texrepeat="16 16" reflectance="0.02"/>
  </asset>
  <worldbody>
    <light pos="0 -3 4" dir="0 1 -1" diffuse="0.45 0.45 0.45"/>
    <light pos="-3 2 3" dir="1 -1 -1" diffuse="0.22 0.22 0.22"/>
    <geom name="ground" type="plane" size="8 8 0.01" material="checker"/>
  </worldbody>
</mujoco>
"""


def build_hand_skeleton_edges(side: str) -> tuple[tuple[str, str], ...]:
    prefix = f"{side}Hand"
    wrist = f"{prefix}Wrist"
    palm = f"{prefix}Palm"
    edges = [(f"{side}_Hand", wrist), (wrist, palm)]
    for chain in SKELETON_HAND_FINGER_CHAINS:
        first_joint = f"{prefix}{chain[0]}"
        edges.append((palm, first_joint))
        for parent_name, child_name in zip(chain, chain[1:]):
            edges.append((f"{prefix}{parent_name}", f"{prefix}{child_name}"))
    return tuple(edges)


SKELETON_HAND_EDGES = (
    *build_hand_skeleton_edges("Left"),
    *build_hand_skeleton_edges("Right"),
)

SMPLX_TO_SKELETON_NAME = {
    "pelvis": "Pelvis",
    "left_hip": "Left_Hip",
    "right_hip": "Right_Hip",
    "spine1": "Spine1",
    "left_knee": "Left_Knee",
    "right_knee": "Right_Knee",
    "spine2": "Spine2",
    "left_ankle": "Left_Ankle",
    "right_ankle": "Right_Ankle",
    "spine3": "Spine3",
    "left_foot": "Left_Foot",
    "right_foot": "Right_Foot",
    "neck": "Neck",
    "left_collar": "Left_Collar",
    "right_collar": "Right_Collar",
    "head": "Head",
    "left_shoulder": "Left_Shoulder",
    "right_shoulder": "Right_Shoulder",
    "left_elbow": "Left_Elbow",
    "right_elbow": "Right_Elbow",
    "left_wrist": "Left_Wrist",
    "right_wrist": "Right_Wrist",
    "left_index1": "Left_Hand",
    "right_index1": "Right_Hand",
}


def collect_motion_paths(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Motion path not found: {path}")

    patterns = ("**/*_raw.pkl", "**/*.npz") if recursive else ("*_raw.pkl", "*.npz")
    motion_paths = []
    for pattern in patterns:
        motion_paths.extend(motion_path for motion_path in path.glob(pattern) if motion_path.is_file())
    if motion_paths:
        return sorted(motion_path for motion_path in set(motion_paths) if is_supported_motion_path(motion_path))

    fallback_pattern = "**/*.pkl" if recursive else "*.pkl"
    return sorted(motion_path for motion_path in path.glob(fallback_pattern) if motion_path.is_file())


def is_supported_motion_path(motion_path: Path) -> bool:
    if motion_path.suffix.lower() != ".npz":
        return True
    try:
        with np.load(motion_path, allow_pickle=True) as npz_data:
            return smplx_npz_num_frames(npz_data) > 0
    except Exception:
        return False


def load_raw_pkl_frames(raw_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with raw_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        frames = payload["frames"]
        meta = payload.get("meta", {})
    elif isinstance(payload, list):
        frames = payload
        meta = {}
    else:
        raise ValueError(f"Unsupported raw pickle schema: {raw_path}")

    if not all(isinstance(frame, dict) for frame in frames):
        raise ValueError(f"Raw pkl frames must be dictionaries: {raw_path}")
    return frames, meta if isinstance(meta, dict) else {}


def np_scalar_to_python(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return value


def smplx_npz_fps(npz_data: Any) -> float:
    if "mocap_frame_rate" in npz_data:
        fps = float(np_scalar_to_python(npz_data["mocap_frame_rate"]))
        if fps > 0.0:
            return fps
    return 30.0


def smplx_npz_num_frames(npz_data: Any) -> int:
    for key in ("trans", "pose_body", "root_orient", "poses"):
        if key in npz_data:
            return int(np.asarray(npz_data[key]).shape[0])
    return 0


def load_smplx_npz_frames(
    smplx_path: Path,
    smplx_body_model_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import smplx
        import torch
        from smplx.joint_names import JOINT_NAMES
    except ImportError as exc:
        raise ImportError(
            "SMPL-X npz playback requires `smplx` and `torch`. "
            "Install them in the environment used to run this viewer."
        ) from exc

    smplx_data = np.load(smplx_path, allow_pickle=True)
    num_frames = smplx_npz_num_frames(smplx_data)
    if num_frames <= 0:
        raise ValueError(f"No SMPL-X frames found in {smplx_path}")

    poses = np.asarray(smplx_data["poses"]) if "poses" in smplx_data else None
    root_orient = (
        np.asarray(smplx_data["root_orient"])
        if "root_orient" in smplx_data
        else poses[:, :3]
    )
    pose_body = (
        np.asarray(smplx_data["pose_body"])
        if "pose_body" in smplx_data
        else poses[:, 3:66]
    )
    trans = np.asarray(smplx_data["trans"]) if "trans" in smplx_data else np.zeros((num_frames, 3))
    betas = np.asarray(smplx_data["betas"]) if "betas" in smplx_data else np.zeros(16)
    pose_hand = (
        np.asarray(smplx_data["pose_hand"])
        if "pose_hand" in smplx_data
        else (
            poses[:, 75:165]
            if poses is not None and poses.shape[1] >= 165
            else np.zeros((num_frames, 90))
        )
    )
    pose_jaw = (
        np.asarray(smplx_data["pose_jaw"])
        if "pose_jaw" in smplx_data
        else (
            poses[:, 66:69]
            if poses is not None and poses.shape[1] >= 69
            else np.zeros((num_frames, 3))
        )
    )
    pose_eye = (
        np.asarray(smplx_data["pose_eye"])
        if "pose_eye" in smplx_data
        else (
            poses[:, 69:75]
            if poses is not None and poses.shape[1] >= 75
            else np.zeros((num_frames, 6))
        )
    )
    gender = str(np_scalar_to_python(smplx_data["gender"])) if "gender" in smplx_data else "neutral"

    body_model = smplx.create(
        smplx_body_model_dir,
        "smplx",
        gender=gender,
        use_pca=False,
    )

    with torch.no_grad():
        smplx_output = body_model(
            betas=torch.tensor(betas).float().view(1, -1),
            global_orient=torch.tensor(root_orient).float(),
            body_pose=torch.tensor(pose_body).float(),
            transl=torch.tensor(trans).float(),
            left_hand_pose=torch.tensor(pose_hand[:, :45]).float(),
            right_hand_pose=torch.tensor(pose_hand[:, 45:90]).float(),
            jaw_pose=torch.tensor(pose_jaw).float(),
            leye_pose=torch.tensor(pose_eye[:, :3]).float(),
            reye_pose=torch.tensor(pose_eye[:, 3:6]).float(),
        )

    joints = smplx_output.joints.detach().cpu().numpy()
    joint_names = JOINT_NAMES[: joints.shape[1]]
    mapped_joints = [
        (joint_idx, SMPLX_TO_SKELETON_NAME[joint_name])
        for joint_idx, joint_name in enumerate(joint_names)
        if joint_name in SMPLX_TO_SKELETON_NAME
    ]

    frames = []
    for frame_idx in range(joints.shape[0]):
        body_frame = {}
        for joint_idx, skeleton_name in mapped_joints:
            body_frame[skeleton_name] = [joints[frame_idx, joint_idx].tolist()]
        frames.append({"body": body_frame})

    meta = {
        "source": "smplx_npz",
        "surface_model_type": str(np_scalar_to_python(smplx_data["surface_model_type"]))
        if "surface_model_type" in smplx_data
        else "smplx",
        "gender": gender,
        "fps": smplx_npz_fps(smplx_data),
        "num_frames": num_frames,
    }
    return frames, meta


def load_motion_frames(
    motion_path: Path,
    smplx_body_model_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if motion_path.suffix.lower() == ".npz":
        return load_smplx_npz_frames(motion_path, smplx_body_model_dir)
    return load_raw_pkl_frames(motion_path)


def print_motion_dry_run(
    motion_path: Path,
    smplx_body_model_dir: Path,
    show_inactive_hands: bool,
    fps_override: float | None,
) -> None:
    if motion_path.suffix.lower() == ".npz":
        smplx_data = np.load(motion_path, allow_pickle=True)
        num_frames = smplx_npz_num_frames(smplx_data)
        fps = fps_override if fps_override is not None else smplx_npz_fps(smplx_data)
        gender = str(np_scalar_to_python(smplx_data["gender"])) if "gender" in smplx_data else "neutral"
        print(f"Loaded {motion_path}")
        print(f"Format: SMPL-X npz | Frames: {num_frames} | FPS: {fps:.3g}")
        print(f"Body model dir: {smplx_body_model_dir}")
        print(f"Gender: {gender} | First-frame mapped body points: {len(SMPLX_TO_SKELETON_NAME)}")
        return

    frames, meta = load_raw_pkl_frames(motion_path)
    if not frames:
        raise ValueError(f"No frames found in {motion_path}")
    fps = estimate_fps(frames, meta, fps_override)
    print(f"Loaded {motion_path}")
    print(f"Format: raw pkl | Frames: {len(frames)} | FPS: {fps:.3g}")
    if meta:
        print(f"Meta: source={meta.get('source')} robot={meta.get('robot')} chunk_idx={meta.get('chunk_idx')}")
    first_pose = build_pose_dict(frames[0], show_inactive_hands=show_inactive_hands)
    valid_points = sum(1 for name in first_pose if pose_position(first_pose, name) is not None)
    print(f"First-frame valid points: {valid_points}")


def estimate_fps(
    frames: list[dict[str, Any]],
    meta: dict[str, Any],
    fps_override: float | None,
) -> float:
    if fps_override is not None:
        return float(fps_override)

    for key in ("saved_fps", "target_fps", "fps", "mocap_frame_rate", "measured_fps"):
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


def body_data_from_sdk_raw(sdk_raw_data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(sdk_raw_data, dict):
        return {}
    raw_body = sdk_raw_data.get("body")
    if not isinstance(raw_body, dict):
        return {}
    poses = raw_body.get("poses")
    if poses is None:
        return {}

    body_pose_dict = {}
    for joint_name, raw_pose in zip(SKELETON_BODY_JOINT_NAMES, poses):
        values = np.asarray(raw_pose, dtype=np.float64)
        if values.shape[0] < 3:
            continue
        position = values[:3] @ UNITY_TO_GMR_ROTATION_MATRIX.T
        body_pose_dict[joint_name] = [position.tolist()]
    return body_pose_dict


def body_data_from_raw_frame(frame: dict[str, Any]) -> dict[str, Any]:
    body_data = frame.get("body")
    if isinstance(body_data, dict) and body_data:
        return copy.deepcopy(body_data)
    return body_data_from_sdk_raw(frame.get("sdk_raw"))


def extract_hand_pose_dict(hand_data: Any, *, show_inactive_hands: bool) -> dict[str, Any]:
    if isinstance(hand_data, dict):
        if "is_active" in hand_data and "data" in hand_data:
            if (bool(hand_data["is_active"]) or show_inactive_hands) and isinstance(hand_data["data"], dict):
                return hand_data["data"]
            return {}
        return hand_data
    if isinstance(hand_data, (tuple, list)) and len(hand_data) == 2:
        is_active, pose_dict = hand_data
        if (bool(is_active) or show_inactive_hands) and isinstance(pose_dict, dict):
            return pose_dict
    return {}


def build_pose_dict(frame: dict[str, Any], *, show_inactive_hands: bool) -> dict[str, Any]:
    pose_dict = body_data_from_raw_frame(frame)
    pose_dict.update(extract_hand_pose_dict(frame.get("left_hand"), show_inactive_hands=show_inactive_hands))
    pose_dict.update(extract_hand_pose_dict(frame.get("right_hand"), show_inactive_hands=show_inactive_hands))
    return pose_dict


def pose_position(pose_dict: dict[str, Any], body_name: str) -> np.ndarray | None:
    pose = pose_dict.get(body_name)
    if not isinstance(pose, (tuple, list)) or len(pose) < 1:
        return None
    pos = np.asarray(pose[0], dtype=float)
    if pos.shape != (3,) or not np.all(np.isfinite(pos)):
        return None
    return pos


def draw_capsule(viewer, start: np.ndarray, end: np.ndarray, width: float, rgba: np.ndarray) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    if np.linalg.norm(end - start) < 1e-6:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_connector(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        width=width,
        from_=start,
        to=end,
    )
    geom.rgba[:] = rgba
    viewer.user_scn.ngeom += 1


def draw_sphere(viewer, pos: np.ndarray, radius: float, rgba: np.ndarray, label: str | None = None) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0.0, 0.0],
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    if label is not None:
        geom.label = label
    viewer.user_scn.ngeom += 1


def draw_com_projection(viewer, com: np.ndarray, ground_z: float = 0.0, radius: float = 0.045) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        size=[radius, 0.004, 0.0],
        pos=np.array([com[0], com[1], ground_z + 0.004]),
        mat=np.eye(3).flatten(),
        rgba=[1.0, 0.75, 0.05, 0.9],
    )
    geom.label = "CoM"
    viewer.user_scn.ngeom += 1


def transformed_joint_positions(
    pose_dict: dict[str, Any],
    *,
    anchor_pelvis: bool,
    pelvis_height: float,
    offset: np.ndarray,
) -> dict[str, np.ndarray]:
    edges = SKELETON_BODY_EDGES + SKELETON_HAND_EDGES
    edge_names = {name for edge in edges for name in edge}

    placement_offset = offset
    if anchor_pelvis:
        pelvis_pos = pose_position(pose_dict, "Pelvis")
        if pelvis_pos is not None:
            placement_offset = np.array([0.0, 0.0, pelvis_height], dtype=np.float64) + offset - pelvis_pos

    joint_positions = {}
    for body_name in edge_names:
        pos = pose_position(pose_dict, body_name)
        if pos is not None:
            joint_positions[body_name] = pos + placement_offset
    return joint_positions


def estimate_ground_offset(
    frame: dict[str, Any],
    *,
    show_inactive_hands: bool,
    ground_z: float,
) -> np.ndarray:
    pose_dict = build_pose_dict(frame, show_inactive_hands=show_inactive_hands)
    positions = [
        pose_position(pose_dict, name)
        for name in pose_dict
    ]
    positions = [pos for pos in positions if pos is not None]
    if not positions:
        return np.zeros(3, dtype=np.float64)
    min_z = float(np.min(np.asarray(positions)[:, 2]))
    return np.array([0.0, 0.0, ground_z - min_z], dtype=np.float64)


def draw_skeleton(
    viewer,
    frame: dict[str, Any],
    *,
    anchor_pelvis: bool,
    pelvis_height: float,
    offset: np.ndarray,
    show_inactive_hands: bool,
    show_names: bool,
) -> dict[str, np.ndarray]:
    pose_dict = build_pose_dict(frame, show_inactive_hands=show_inactive_hands)
    joint_positions = transformed_joint_positions(
        pose_dict,
        anchor_pelvis=anchor_pelvis,
        pelvis_height=pelvis_height,
        offset=offset,
    )

    for parent_name, child_name in SKELETON_BODY_EDGES:
        parent_pos = joint_positions.get(parent_name)
        child_pos = joint_positions.get(child_name)
        if parent_pos is not None and child_pos is not None:
            draw_capsule(viewer, parent_pos, child_pos, width=0.014, rgba=BODY_LINE_RGBA)

    for parent_name, child_name in SKELETON_HAND_EDGES:
        parent_pos = joint_positions.get(parent_name)
        child_pos = joint_positions.get(child_name)
        if parent_pos is not None and child_pos is not None:
            draw_capsule(viewer, parent_pos, child_pos, width=0.006, rgba=HAND_LINE_RGBA)

    for name, pos in joint_positions.items():
        draw_sphere(viewer, pos, radius=0.023, rgba=JOINT_RGBA, label=name if show_names else None)
    return joint_positions


def install_keyboard_callback(viewer, callback) -> bool:
    if hasattr(viewer, "set_key_callback"):
        try:
            viewer.set_key_callback(callback)
            return True
        except TypeError:
            pass
    if hasattr(viewer, "set_keydown_callback"):
        try:
            viewer.set_keydown_callback(callback)
            return True
        except TypeError:
            pass
    if hasattr(viewer, "user_callbacks"):
        try:
            viewer.user_callbacks["keyboard"] = callback
            return True
        except Exception:
            pass
    if hasattr(viewer, "callbacks"):
        try:
            viewer.callbacks["keyboard"] = callback
            return True
        except Exception:
            pass
    return False


def viewer_is_running(viewer) -> bool:
    if hasattr(viewer, "is_running"):
        return viewer.is_running()
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View raw SMPL-X npz or skeleton pkl motions without retargeting."
    )
    parser.add_argument(
        "--motion_path",
        type=Path,
        required=True,
        help="Raw .pkl / SMPL-X .npz file, or a directory containing them.",
    )
    parser.add_argument(
        "--smplx_body_model_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "body_models",
        help="Body model root used for SMPL-X npz playback.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override playback fps.")
    parser.add_argument("--loop", action="store_true", help="Loop after the last frame.")
    parser.add_argument("--no_rate_limit", action="store_true")
    parser.add_argument("--non_recursive", action="store_true", help="Do not recurse when motion_path is a directory.")
    parser.add_argument(
        "--anchor_pelvis",
        action="store_true",
        help="Keep the Pelvis fixed near the origin instead of showing recorded global XY motion.",
    )
    parser.add_argument(
        "--keep_world_positions",
        action="store_true",
        help="Deprecated no-op; world positions are now the default.",
    )
    parser.add_argument(
        "--no_auto_ground",
        action="store_true",
        help="Do not add a constant Z offset that places the first frame on the checker board.",
    )
    parser.add_argument(
        "--pelvis_height",
        type=float,
        default=0.95,
        help="Pelvis Z height when anchoring the skeleton near the origin.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
    )
    parser.add_argument("--show_inactive_hands", action="store_true")
    parser.add_argument("--show_names", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Load and print metadata without launching the viewer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_input_path = args.motion_path.expanduser().resolve()
    smplx_body_model_dir = args.smplx_body_model_dir.expanduser().resolve()
    motion_paths = collect_motion_paths(motion_input_path, recursive=not args.non_recursive)
    if not motion_paths:
        raise FileNotFoundError(f"No supported .pkl/.npz motion files found under: {motion_input_path}")

    print(f"Found {len(motion_paths)} motion file(s) under {motion_input_path}")
    if args.dry_run:
        print_motion_dry_run(
            motion_paths[0],
            smplx_body_model_dir,
            show_inactive_hands=args.show_inactive_hands,
            fps_override=args.fps,
        )
        return

    global mj, mjv
    import mujoco as mj
    import mujoco.viewer as mjv

    model = mj.MjModel.from_xml_string(SCENE_XML)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    control_state = create_control_state(enable_curation=False)
    base_key_callback = make_keyboard_callback(control_state, enable_curation=False, log_fn=print)
    viewer_state = {"show_com_projection": False}

    def key_callback(keycode: int, *args, **kwargs) -> None:
        if keycode in (ord("v"), ord("V")):
            viewer_state["show_com_projection"] = not viewer_state["show_com_projection"]
            print(f"[cyan]CoM projection: {viewer_state['show_com_projection']}[/cyan]")
            return
        base_key_callback(keycode, *args, **kwargs)

    try:
        viewer = mjv.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=key_callback,
        )
    except TypeError:
        viewer = mjv.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False)
        if not install_keyboard_callback(viewer, key_callback):
            print("[yellow]Unable to register keyboard callback; key controls disabled.[/yellow]")

    viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.6])
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 145

    log_controls_once(include_curation=False, log_fn=print)
    print("Raw global XY positions are shown by default. Use --anchor_pelvis to fix Pelvis near the origin.")
    print("Close the MuJoCo window to exit.")

    clip_index = 0
    frame_idx = 0
    frames = []
    fps = 30.0
    ground_offset = np.zeros(3, dtype=np.float64)
    try:
        while viewer_is_running(viewer):
            if not frames:
                motion_path = motion_paths[clip_index]
                frames, meta = load_motion_frames(motion_path, smplx_body_model_dir)
                if not frames:
                    raise ValueError(f"No frames found in {motion_path}")
                fps = estimate_fps(frames, meta, args.fps)
                ground_offset = (
                    np.zeros(3, dtype=np.float64)
                    if args.anchor_pelvis or args.no_auto_ground
                    else estimate_ground_offset(
                        frames[0],
                        show_inactive_hands=args.show_inactive_hands,
                        ground_z=0.02,
                    )
                )
                print(f"[{clip_index + 1}/{len(motion_paths)}] {motion_path}")
                print(f"Frames: {len(frames)} | FPS: {fps:.3g}")
                if meta:
                    print(f"Meta: source={meta.get('source')} robot={meta.get('robot')} chunk_idx={meta.get('chunk_idx')}")

            if control_state["speed_dirty"]:
                control_state["speed_dirty"] = False

            if control_state["clip_delta"] != 0:
                clip_index = (clip_index + control_state["clip_delta"]) % len(motion_paths)
                control_state["clip_delta"] = 0
                frame_idx = 0
                frames = []
                continue

            viewer.user_scn.ngeom = 0
            joint_positions = draw_skeleton(
                viewer,
                frames[frame_idx],
                anchor_pelvis=args.anchor_pelvis and not args.keep_world_positions,
                pelvis_height=args.pelvis_height,
                offset=ground_offset + np.asarray(args.offset, dtype=np.float64),
                show_inactive_hands=args.show_inactive_hands,
                show_names=args.show_names,
            )
            drawn = len(joint_positions)
            if viewer_state["show_com_projection"] and joint_positions:
                com = np.mean(np.asarray(list(joint_positions.values())), axis=0)
                draw_com_projection(viewer, com)
            viewer.sync()

            if not args.no_rate_limit:
                time.sleep(max(0.0, 1.0 / max(fps * control_state["speed"], 1e-6)))

            if control_state["paused"]:
                if control_state["frame_step"] != 0:
                    frame_idx = (frame_idx + control_state["frame_step"]) % len(frames)
                    control_state["frame_step"] = 0
                continue

            frame_idx += 1
            if frame_idx >= len(frames):
                frame_idx = 0
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
