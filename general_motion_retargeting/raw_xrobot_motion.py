from dataclasses import dataclass
from pathlib import Path
import copy
import pickle
from typing import Any, Callable

import mujoco as mj
import numpy as np
from rich import print as rich_print
from scipy.spatial.transform import Rotation as R

from .rot_utils import quat_mul_np


RAW_XROBOT_EXTENSIONS = (".pkl",)

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

XROBOT_BODY_SKELETON_EDGES = (
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

XROBOT_HAND_FINGER_CHAINS = (
    ("ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip"),
    ("IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip"),
    ("MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip"),
    ("RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip"),
    ("LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip"),
)

XROBOT_SKELETON_LINE_RGBA = np.array([0.1, 0.85, 1.0, 0.75])
XROBOT_SKELETON_JOINT_RGBA = np.array([1.0, 0.9, 0.25, 0.9])

UNITY_TO_GMR_ROTATION_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
_UNITY_TO_GMR_ROTATION_QUAT_XYZW = R.from_matrix(UNITY_TO_GMR_ROTATION_MATRIX).as_quat()
UNITY_TO_GMR_ROTATION_QUAT_WXYZ = _UNITY_TO_GMR_ROTATION_QUAT_XYZW[[3, 0, 1, 2]]


def build_xrobot_hand_skeleton_edges(side: str) -> tuple[tuple[str, str], ...]:
    prefix = f"{side}Hand"
    wrist = f"{prefix}Wrist"
    palm = f"{prefix}Palm"
    edges = [(f"{side}_Hand", wrist), (wrist, palm)]
    for chain in XROBOT_HAND_FINGER_CHAINS:
        first_joint = f"{prefix}{chain[0]}"
        edges.append((palm, first_joint))
        for parent_name, child_name in zip(chain, chain[1:]):
            edges.append((f"{prefix}{parent_name}", f"{prefix}{child_name}"))
    return tuple(edges)


XROBOT_HAND_SKELETON_EDGES = (
    *build_xrobot_hand_skeleton_edges("Left"),
    *build_xrobot_hand_skeleton_edges("Right"),
)


@dataclass(frozen=True)
class RawXRobotMotion:
    path: Path
    frames: list[dict[str, Any]]

    def frame_at(self, frame_idx: int, reference_num_frames: int) -> dict[str, Any] | None:
        if len(self.frames) == 0:
            return None
        if len(self.frames) == reference_num_frames:
            return self.frames[frame_idx % len(self.frames)]
        if reference_num_frames <= 1:
            return self.frames[0]

        wrapped_frame = frame_idx % reference_num_frames
        raw_frame = round(
            wrapped_frame * (len(self.frames) - 1) / (reference_num_frames - 1)
        )
        return self.frames[int(np.clip(raw_frame, 0, len(self.frames) - 1))]


class RawXRobotMotionLoader:
    def __init__(
        self,
        raw_xrobot_dir: str | Path | None,
        *,
        motion_root_dir: str | Path | None = None,
        log_fn: Callable[[str], None] = rich_print,
    ):
        self.raw_xrobot_dir = Path(raw_xrobot_dir).expanduser() if raw_xrobot_dir else None
        self.motion_root_dir = (
            Path(motion_root_dir).expanduser() if motion_root_dir is not None else None
        )
        self.log_fn = log_fn
        self.enabled = self.raw_xrobot_dir is not None and self.raw_xrobot_dir.is_dir()

        if self.raw_xrobot_dir is not None and not self.enabled:
            self.log_fn(
                f"[yellow]Raw XRobot directory not found; ignoring: {self.raw_xrobot_dir}[/yellow]"
            )

    def find_path(self, motion_path: str | Path) -> Path | None:
        if not self.enabled or self.raw_xrobot_dir is None:
            return None

        motion_path = Path(motion_path)
        relative_stem = self._relative_motion_stem(motion_path)
        raw_stem = relative_stem.parent / f"{relative_stem.name}_raw"

        for extension in RAW_XROBOT_EXTENSIONS:
            candidate = self.raw_xrobot_dir / raw_stem.with_suffix(extension)
            if candidate.is_file():
                return candidate
        return None

    def load_for_motion(self, motion_path: str | Path) -> RawXRobotMotion | None:
        raw_path = self.find_path(motion_path)
        if raw_path is None:
            return None

        try:
            frames = _load_xrobot_frames(raw_path)
        except Exception:
            return None

        self.log_fn(f"[cyan]Raw XRobot: {raw_path}[/cyan]")
        return RawXRobotMotion(path=raw_path, frames=frames)

    def _relative_motion_stem(self, motion_path: Path) -> Path:
        if self.motion_root_dir is None:
            return Path(motion_path.stem)

        try:
            relative_path = motion_path.resolve().relative_to(self.motion_root_dir.resolve())
        except ValueError:
            relative_path = Path(motion_path.name)
        return relative_path.with_suffix("")


def draw_xrobot_skeleton(
    viewer,
    raw_xrobot_frame: dict[str, Any],
    root_pos: np.ndarray,
    skeleton_offset: np.ndarray,
) -> None:
    if not isinstance(raw_xrobot_frame, dict):
        return

    pose_dict = _build_xrobot_pose_dict(raw_xrobot_frame)
    pelvis_pos = _pose_position(pose_dict, "Pelvis")
    if pelvis_pos is None:
        return

    placement_offset = root_pos + skeleton_offset - pelvis_pos
    skeleton_edges = XROBOT_BODY_SKELETON_EDGES + XROBOT_HAND_SKELETON_EDGES
    edge_names = {name for edge in skeleton_edges for name in edge}

    joint_positions = {}
    for body_name in edge_names:
        pos = _pose_position(pose_dict, body_name)
        if pos is not None:
            joint_positions[body_name] = pos + placement_offset

    for parent_name, child_name in skeleton_edges:
        parent_pos = joint_positions.get(parent_name)
        child_pos = joint_positions.get(child_name)
        if parent_pos is None or child_pos is None:
            continue
        _draw_user_capsule(
            viewer,
            parent_pos,
            child_pos,
            width=0.012,
            rgba=XROBOT_SKELETON_LINE_RGBA,
        )

    for pos in joint_positions.values():
        _draw_user_sphere(
            viewer,
            pos,
            radius=0.025,
            rgba=XROBOT_SKELETON_JOINT_RGBA,
        )


def _load_xrobot_frames(raw_path: Path) -> list[dict[str, Any]]:
    with raw_path.open("rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        frames = payload["frames"]
    elif isinstance(payload, list):
        frames = payload
    else:
        raise ValueError(f"Unsupported raw XRobot pickle schema in {raw_path}")

    if not all(isinstance(frame, dict) for frame in frames):
        raise ValueError(f"Raw XRobot frames must be dictionaries: {raw_path}")
    if not any(_looks_like_xrobot_frame(frame) for frame in frames):
        raise ValueError(f"No raw XRobot body data found in {raw_path}")
    return frames


def _looks_like_xrobot_frame(frame: dict[str, Any]) -> bool:
    return isinstance(frame.get("body"), dict) or isinstance(frame.get("sdk_raw"), dict)


def _build_xrobot_pose_dict(frame: dict[str, Any]) -> dict[str, Any]:
    pose_dict = body_data_from_raw_frame(frame)
    pose_dict.update(_extract_active_hand_pose_dict(frame.get("left_hand")))
    pose_dict.update(_extract_active_hand_pose_dict(frame.get("right_hand")))
    return pose_dict


def body_data_from_raw_frame(frame: dict[str, Any]) -> dict[str, Any]:
    body_data = frame.get("body")
    if isinstance(body_data, dict) and body_data:
        return copy.deepcopy(body_data)
    return body_data_from_sdk_raw(frame.get("sdk_raw"))


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
        body_pose_dict[joint_name] = list(
            coordinate_transform_unity_pose(pos, quat_wxyz)
        )
    return body_pose_dict


def coordinate_transform_unity_pose(
    pos_xyz: np.ndarray,
    quat_wxyz: np.ndarray,
) -> tuple[list[float], list[float]]:
    orientation = quat_mul_np(
        UNITY_TO_GMR_ROTATION_QUAT_WXYZ,
        normalize_quat_wxyz(quat_wxyz),
        scalar_first=True,
    )
    position = pos_xyz @ UNITY_TO_GMR_ROTATION_MATRIX.T
    return position.tolist(), orientation.tolist()


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        raise ValueError("Invalid zero-norm quaternion in raw XRobot frame")
    return quat / norm


def _extract_active_hand_pose_dict(hand_data: Any) -> dict[str, Any]:
    if isinstance(hand_data, dict):
        if "is_active" in hand_data and "data" in hand_data:
            if bool(hand_data["is_active"]) and isinstance(hand_data["data"], dict):
                return hand_data["data"]
            return {}
        return hand_data
    if isinstance(hand_data, (tuple, list)) and len(hand_data) == 2:
        is_active, pose_dict = hand_data
        if bool(is_active) and isinstance(pose_dict, dict):
            return pose_dict
    return {}


def _pose_position(pose_dict: dict[str, Any], body_name: str) -> np.ndarray | None:
    if body_name not in pose_dict:
        return None
    pose = pose_dict[body_name]
    if not isinstance(pose, (tuple, list)) or len(pose) < 1:
        return None
    pos = np.asarray(pose[0], dtype=float)
    if pos.shape != (3,) or not np.all(np.isfinite(pos)):
        return None
    return pos


def _draw_user_capsule(viewer, start: np.ndarray, end: np.ndarray, width: float, rgba: np.ndarray) -> None:
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


def _draw_user_sphere(viewer, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
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
    viewer.user_scn.ngeom += 1
