import argparse
import os
import pathlib
import pickle
import time

import mujoco as mj
import numpy as np
from rich import print
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer, load_robot_motion
from general_motion_retargeting.robot_to_robot_retarget import RobotToRobotMotionRetargeting
from general_motion_retargeting.utils.motion_utils import (
    build_motion_data,
    get_default_keybody_names,
)


KIMODO_G1_BODY_INDEX = {
    "pelvis": 0,
    "left_hip_roll_link": 2,
    "left_knee_link": 4,
    "left_toe_link": 7,
    "right_hip_roll_link": 9,
    "right_knee_link": 11,
    "right_toe_link": 14,
    "torso_link": 17,
    "left_shoulder_yaw_link": 20,
    "left_elbow_link": 21,
    "left_wrist_yaw_link": 24,
    "right_shoulder_yaw_link": 28,
    "right_elbow_link": 29,
    "right_wrist_yaw_link": 32,
}

Y_UP_TO_Z_UP = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "y"):
        return True
    if lowered in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean string, got {value!r}.")


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("Root quaternion contains a zero-norm frame.")
    return quaternions / norms


def transform_y_up_position(position: np.ndarray) -> np.ndarray:
    return Y_UP_TO_Z_UP @ position


def transform_y_up_rotation(rotation_matrix: np.ndarray) -> np.ndarray:
    return Y_UP_TO_Z_UP @ rotation_matrix @ Y_UP_TO_Z_UP.T


def matrix_to_quat_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    return quat_xyzw[[3, 0, 1, 2]]


def quat_wxyz_to_rotation(quat_wxyz: np.ndarray) -> R:
    return R.from_quat(quat_wxyz[[1, 2, 3, 0]])


def load_source_qpos_from_pkl(
    g1_motion_path: str,
    root_quat_scalar_first: bool,
) -> tuple[dict, float, np.ndarray]:
    motion_data, motion_fps, root_pos, root_rot, dof_pos, _, _ = load_robot_motion(g1_motion_path)

    root_rot = np.asarray(root_rot, dtype=np.float64)
    if not root_quat_scalar_first:
        root_rot = root_rot[:, [3, 0, 1, 2]]
    root_rot = normalize_quaternions(root_rot)

    qpos = np.concatenate(
        [
            np.asarray(root_pos, dtype=np.float64),
            root_rot,
            np.asarray(dof_pos, dtype=np.float64),
        ],
        axis=1,
    )
    return motion_data, motion_fps, qpos


def load_kimodo_g1_npz(
    g1_motion_path: str,
    default_motion_fps: float,
) -> tuple[dict, float, list[dict[str, list[np.ndarray]]]]:
    with np.load(g1_motion_path, allow_pickle=True) as data:
        required_keys = {"posed_joints", "global_rot_mats"}
        missing_keys = required_keys - set(data.files)
        if missing_keys:
            raise ValueError(f"Missing required G1 npz keys: {sorted(missing_keys)}")

        posed_joints = np.asarray(data["posed_joints"], dtype=np.float64)
        global_rot_mats = np.asarray(data["global_rot_mats"], dtype=np.float64)
        motion_fps = float(data["fps"]) if "fps" in data.files else float(default_motion_fps)

    if posed_joints.ndim != 3 or posed_joints.shape[1:] != (34, 3):
        raise ValueError(f"posed_joints must have shape (T, 34, 3), got {posed_joints.shape}.")
    if global_rot_mats.shape != (posed_joints.shape[0], 34, 3, 3):
        raise ValueError(
            "global_rot_mats must have shape "
            f"{(posed_joints.shape[0], 34, 3, 3)}, got {global_rot_mats.shape}."
        )
    if posed_joints.shape[0] == 0:
        raise ValueError("G1 npz contains zero frames.")

    frames = []
    for frame_pos, frame_rot in zip(posed_joints, global_rot_mats):
        source_frame = {}
        for body_name, body_index in KIMODO_G1_BODY_INDEX.items():
            rot_matrix = transform_y_up_rotation(frame_rot[body_index])
            source_frame[body_name] = [
                transform_y_up_position(frame_pos[body_index]),
                matrix_to_quat_wxyz(rot_matrix),
            ]

        torso_pos, torso_quat = source_frame["torso_link"]
        torso_rot = quat_wxyz_to_rotation(torso_quat)
        source_frame["head_mocap"] = [
            torso_pos + torso_rot.apply(np.array([0.0, 0.0, 0.4])),
            torso_quat.copy(),
        ]
        frames.append(source_frame)

    return {"format": "kimodo_g1_npz"}, motion_fps, frames


def load_source_motion(
    g1_motion_path: str,
    root_quat_scalar_first: bool,
    default_motion_fps: float,
) -> tuple[dict, float, np.ndarray | list[dict[str, list[np.ndarray]]], str]:
    suffix = pathlib.Path(g1_motion_path).suffix.lower()
    if suffix == ".npz":
        motion_data, motion_fps, source_frames = load_kimodo_g1_npz(
            g1_motion_path,
            default_motion_fps=default_motion_fps,
        )
        return motion_data, motion_fps, source_frames, "source_frames"
    if suffix == ".pkl":
        motion_data, motion_fps, source_qpos = load_source_qpos_from_pkl(
            g1_motion_path,
            root_quat_scalar_first=root_quat_scalar_first,
        )
        return motion_data, motion_fps, source_qpos, "source_qpos"
    raise ValueError(f"Unsupported G1 motion format {suffix!r}. Expected .npz or .pkl.")


def collect_keybody_metadata(retargeter, target_robot: str):
    keybody_names = get_default_keybody_names(target_robot)
    keybody_pairs = [
        (name, retargeter.robot_body_names[name])
        for name in keybody_names
        if name in retargeter.robot_body_names
    ]
    missing_keybodies = [name for name in keybody_names if name not in retargeter.robot_body_names]
    if missing_keybodies:
        print(f"[warn] Missing keybodies in robot model: {missing_keybodies}")

    return [name for name, _ in keybody_pairs], [idx for _, idx in keybody_pairs]


def save_target_motion(
    save_path: str,
    target_robot: str,
    motion_fps: float,
    qpos_list: list[np.ndarray],
    retargeter: RobotToRobotMotionRetargeting,
) -> None:
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    keybody_names, keybody_ids = collect_keybody_metadata(retargeter, target_robot)
    mj_data = mj.MjData(retargeter.model)
    keybody_pos_samples = []
    keybody_rot_wxyz_samples = []

    for qpos in qpos_list:
        mj_data.qpos[:] = qpos
        mj.mj_forward(retargeter.model, mj_data)
        keybody_pos_samples.append(mj_data.xpos[keybody_ids].copy())
        keybody_rot_wxyz_samples.append(mj_data.xquat[keybody_ids].copy())

    num_frames = len(qpos_list)
    if keybody_pos_samples:
        keybody_pos_world = np.stack(keybody_pos_samples)
        keybody_rot_world_wxyz = np.stack(keybody_rot_wxyz_samples)
    else:
        keybody_pos_world = np.zeros((num_frames, 0, 3))
        keybody_rot_world_wxyz = np.zeros((num_frames, 0, 4))

    qpos = np.asarray(qpos_list)
    motion_data = build_motion_data(
        aligned_fps=motion_fps,
        root_pos=qpos[:, :3],
        root_rot_wxyz=qpos[:, 3:7],
        dof_pos=qpos[:, 7:],
        keybody_pos_world=keybody_pos_world,
        keybody_rot_world_wxyz=keybody_rot_world_wxyz,
        keybody_names=keybody_names,
        local_body_pos=None,
        local_body_link_body_list=None,
    )

    with open(save_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1_motion_path", required=True, help="Unitree G1 motion .npz or saved .pkl file.")
    parser.add_argument("--save_path", default=None, help="Path to save the Igris C motion .pkl.")
    parser.add_argument("--source_robot", default="unitree_g1", choices=["unitree_g1"])
    parser.add_argument("--target_robot", default="robros_igris_c_v2", choices=["robros_igris_c_v2"])
    parser.add_argument(
        "--root_quat_scalar_first",
        type=parse_bool,
        default=True,
        help="Whether the source root quaternion is stored as wxyz.",
    )
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--no_viewer", action="store_true", default=False)
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument("--offset_to_ground", action="store_true", default=False)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", default=None)
    parser.add_argument(
        "--motion_fps",
        type=float,
        default=30.0,
        help="FPS used when the source npz does not include an fps key.",
    )
    args = parser.parse_args()

    _, motion_fps, source_motion, source_motion_type = load_source_motion(
        args.g1_motion_path,
        root_quat_scalar_first=args.root_quat_scalar_first,
        default_motion_fps=args.motion_fps,
    )

    if source_motion_type == "source_qpos":
        retargeter = RobotToRobotMotionRetargeting(
            src_robot=args.source_robot,
            tgt_robot=args.target_robot,
        )

        if source_motion.shape[1] != retargeter.source_model.nq:
            raise ValueError(
                f"{args.source_robot} motion qpos width must be {retargeter.source_model.nq}, "
                f"got {source_motion.shape[1]}. Check source_robot and root quaternion ordering."
            )
    else:
        retargeter = GMR(
            src_human=args.source_robot,
            tgt_robot=args.target_robot,
        )

    viewer = None
    if not args.no_viewer:
        video_path = args.video_path
        if video_path is None:
            stem = pathlib.Path(args.g1_motion_path).stem
            video_path = f"videos/{args.target_robot}_{stem}.mp4"
        viewer = RobotMotionViewer(
            robot_type=args.target_robot,
            motion_fps=motion_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=video_path,
            root_quat_scalar_first=True,
        )

    qpos_list = []
    pbar = tqdm(total=len(source_motion), desc="Retargeting")
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0
    frame_idx = 0

    while True:
        if source_motion_type == "source_qpos":
            qpos = retargeter.retarget(
                source_motion[frame_idx],
                offset_to_ground=args.offset_to_ground,
            )
            source_visualization_data = retargeter.scaled_source_data
        else:
            qpos = retargeter.retarget(
                source_motion[frame_idx],
                offset_to_ground=args.offset_to_ground,
            )
            source_visualization_data = retargeter.scaled_human_data

        qpos_list.append(qpos.copy())

        if viewer is not None:
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=source_visualization_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )

            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time

        pbar.update(1)

        if args.loop:
            frame_idx = (frame_idx + 1) % len(source_motion)
        else:
            frame_idx += 1
            if frame_idx >= len(source_motion):
                break

    pbar.close()

    if args.save_path is not None:
        save_target_motion(
            save_path=args.save_path,
            target_robot=args.target_robot,
            motion_fps=motion_fps,
            qpos_list=qpos_list,
            retargeter=retargeter,
        )

    if viewer is not None:
        viewer.close()
