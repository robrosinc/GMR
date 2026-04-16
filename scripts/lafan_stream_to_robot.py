#!/usr/bin/env python3
"""
Realtime retargeting pipeline that consumes LaFAN-style BVH frames streamed via a FIFO.

Each line read from the FIFO must be a JSON object with the schema:
{
    "frame": 123,
    "bones": [
        {"p": [x, y, z], "r": [w, x, y, z]},
        ...
    ]
}

where the bones are ordered according to DEFAULT_BONE_NAMES unless overridden via --bone_names.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import mujoco as mj
import numpy as np
from rich import print

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
try:
    from smplx_to_robot import build_motion_data, get_default_keybody_names
except ModuleNotFoundError:
    from scripts.smplx_to_robot import build_motion_data, get_default_keybody_names


DEFAULT_BONE_NAMES: Tuple[str, ...] = (
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
)
POS_ROT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

ROT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
def _rotation_matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion (w, x, y, z)."""
    m = mat
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    return quat / np.linalg.norm(quat)


ROTATION_QUAT = _rotation_matrix_to_quaternion(ROT_MATRIX)


def _quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float32,
    )


def _rotate_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Apply the position rotation to a quaternion (w, x, y, z)."""
    return _quat_mul(ROTATION_QUAT, quat_wxyz)


def _rotate_pos(pos: np.ndarray) -> np.ndarray:
    """Rotate vector using the fixed permutation/rotation."""
    vec = np.asarray(pos, dtype=np.float32).reshape(3, 1)
    return (POS_ROT_MATRIX @ vec).reshape(3)


def _parse_bones(
    bones: Sequence[Dict[str, Sequence[float]]],
    bone_names: Sequence[str],
    foot_format: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if len(bones) - 1 < len(bone_names):
        raise ValueError(
            f"Incoming frame only has {len(bones)} bones (including root), "
            f"but at least {len(bone_names)} are required."
        )
    frame: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for idx, bone_name in enumerate(bone_names):
        bone = bones[idx + 1]
        pos = _rotate_pos(np.asarray(bone["p"], dtype=np.float32))
        quat_xyzw = np.asarray(bone["r"], dtype=np.float32)
        rot = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32
        )
        rot = _rotate_quat(rot)
        frame[bone_name] = (pos, rot)

    if foot_format == "lafan1":
        frame["LeftFootMod"] = (frame["LeftFoot"][0], frame["LeftToe"][1])
        frame["RightFootMod"] = (frame["RightFoot"][0], frame["RightToe"][1])
    elif foot_format == "nokov":
        missing = [name for name in ("LeftToeBase", "RightToeBase") if name not in frame]
        if missing:
            raise KeyError(
                f"Format 'nokov' requires toe base entries, but missing: {missing}"
            )
        frame["LeftFootMod"] = (frame["LeftFoot"][0], frame["LeftToeBase"][1])
        frame["RightFootMod"] = (frame["RightFoot"][0], frame["RightToeBase"][1])
    else:  # pragma: no cover - argument validation happens earlier
        raise ValueError(f"Unsupported format: {foot_format}")
    return frame


def _iter_fifo_lines(path: Path) -> Iterable[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        os.mkfifo(path)
    with open(path, "r", buffering=1) as fifo:
        while True:
            line = fifo.readline()
            if not line:
                continue
            yield line.strip()


def _save_motion_pickle(
    save_path: Path,
    motion_fps: float,
    qpos_list: list[np.ndarray],
    keybody_pos_samples: list[np.ndarray],
    keybody_rot_wxyz_samples: list[np.ndarray],
    keybody_names: list[str],
) -> None:
    save_dir = save_path.parent
    if str(save_dir) not in ("", "."):
        save_dir.mkdir(parents=True, exist_ok=True)

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    root_rot_wxyz = np.array([qpos[3:7] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    num_frames = len(qpos_list)

    if keybody_pos_samples:
        keybody_pos_world = np.stack(keybody_pos_samples)
        keybody_rot_world_wxyz = np.stack(keybody_rot_wxyz_samples)
    else:
        keybody_pos_world = np.zeros((num_frames, 0, 3))
        keybody_rot_world_wxyz = np.zeros((num_frames, 0, 4))

    motion_data = build_motion_data(
        aligned_fps=motion_fps,
        root_pos=root_pos,
        root_rot_wxyz=root_rot_wxyz,
        dof_pos=dof_pos,
        keybody_pos_world=keybody_pos_world,
        keybody_rot_world_wxyz=keybody_rot_world_wxyz,
        keybody_names=keybody_names,
        local_body_pos=None,
        local_body_link_body_list=None,
    )
    with save_path.open("wb") as f:
        pickle.dump(motion_data, f)
    print(f"[green]Saved motion pickle:[/green] {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream LaFAN BVH frames from FIFO into GMR.")
    parser.add_argument(
        "--fifo_path",
        type=Path,
        default=Path("/tmp/motion_frames.pipe"),
        help="FIFO path that provides JSON BVH frames.",
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "unitree_h1",
            "unitree_h1_2",
            "booster_t1",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "kuavo_s45",
            "hightorque_hi",
            "galaxea_r1pro",
            "berkeley_humanoid_lite",
            "booster_k1",
            "pnd_adam_lite",
            "openloong",
            "tienkung",
            "robros_igris_c_v2",
            "robros_igris_max",
        ],
        default="unitree_g1",
    )
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov"],
        default="lafan1",
        help="Controls how LeftFootMod/RightFootMod are synthesized.",
    )
    parser.add_argument(
        "--human_height",
        type=float,
        default=1.75,
        help="Estimated human height used for scaling.",
    )
    parser.add_argument(
        "--bone_names",
        type=str,
        default=",".join(DEFAULT_BONE_NAMES),
        help="Comma-separated bone order that the stream follows.",
    )
    parser.add_argument(
        "--motion_fps",
        type=float,
        default=30.0,
        help="Viewer playback FPS (used for rate limiting/logging).",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Record viewer output to videos/<robot>_lafan_stream.mp4.",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        help="Enable RobotMotionViewer rate limiting to motion_fps.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=None,
        help="Output .pkl path in standard GMR motion format.",
    )
    parser.add_argument(
        "--save_num_frames",
        type=int,
        default=None,
        help="Number of streamed frames to collect before auto-saving one .pkl and exiting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.save_path is None) != (args.save_num_frames is None):
        raise ValueError("Set both --save_path and --save_num_frames together, or set neither.")
    if args.save_num_frames is not None and args.save_num_frames <= 0:
        raise ValueError("--save_num_frames must be a positive integer.")

    bone_names = tuple(name.strip() for name in args.bone_names.split(",") if name.strip())
    if len(bone_names) < len(DEFAULT_BONE_NAMES):
        print(
            "[yellow]Warning:[/yellow] using fewer bone names than defaults may affect retargeting accuracy."
        )

    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=args.human_height,
    )
    video_path = None
    if args.record_video:
        video_dir = Path("videos")
        video_dir.mkdir(exist_ok=True)
        video_path = str(video_dir / f"{args.robot}_lafan_stream.mp4")
    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=args.motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=video_path,
    )

    print(
        f"[cyan]Listening for LaFAN BVH frames on {args.fifo_path} "
        f"(format={args.format}, bones={len(bone_names)})[/cyan]"
    )
    save_enabled = args.save_path is not None
    if save_enabled:
        keybody_names = get_default_keybody_names(args.robot)
        keybody_pairs = [
            (name, retargeter.robot_body_names[name])
            for name in keybody_names
            if name in retargeter.robot_body_names
        ]
        missing_keybodies = [name for name in keybody_names if name not in retargeter.robot_body_names]
        if missing_keybodies:
            print(f"[warn] Missing keybodies in robot model: {missing_keybodies}")
        keybody_names = [name for name, _ in keybody_pairs]
        keybody_ids = [idx for _, idx in keybody_pairs]
        qpos_list: list[np.ndarray] = []
        keybody_pos_samples: list[np.ndarray] = []
        keybody_rot_wxyz_samples: list[np.ndarray] = []
        mj_data_save = mj.MjData(retargeter.model)
        print(
            f"[cyan]Will auto-save after {args.save_num_frames} frames:[/cyan] "
            f"{args.save_path}"
        )

    fps_counter = 0
    fps_start = time.time()
    fps_interval = 2.0
    auto_saved = False
    try:
        for line in _iter_fifo_lines(args.fifo_path):
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[red]Failed to parse JSON line:[/red] {exc}")
                continue

            bones = payload.get("bones")
            if not isinstance(bones, list):
                print("[red]Invalid frame: missing 'bones' list[/red]")
                continue
            try:
                frame = _parse_bones(bones, bone_names, args.format)
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"[red]Skipping frame due to parse error:[/red] {exc}")
                continue

            qpos = retargeter.retarget(frame)
            # rotated_root = _rotate_quat(qpos[3:7])

            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7], #rotated_root,
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                human_pos_offset=np.zeros(3),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )

            if save_enabled:
                qpos_list.append(qpos.copy())
                mj_data_save.qpos[:] = qpos
                mj.mj_forward(retargeter.model, mj_data_save)
                keybody_pos_samples.append(mj_data_save.xpos[keybody_ids].copy())
                keybody_rot_wxyz_samples.append(mj_data_save.xquat[keybody_ids].copy())
                if len(qpos_list) >= args.save_num_frames:
                    _save_motion_pickle(
                        save_path=args.save_path,
                        motion_fps=args.motion_fps,
                        qpos_list=qpos_list,
                        keybody_pos_samples=keybody_pos_samples,
                        keybody_rot_wxyz_samples=keybody_rot_wxyz_samples,
                        keybody_names=keybody_names,
                    )
                    auto_saved = True
                    break

            fps_counter += 1
            now = time.time()
            if now - fps_start >= fps_interval:
                actual_fps = fps_counter / (now - fps_start)
                print(f"Streaming FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start = now
    except KeyboardInterrupt:
        print("[yellow]Interrupted, closing viewer.[/yellow]")
    finally:
        if save_enabled and (not auto_saved):
            print("[yellow]Stream ended before reaching --save_num_frames. No pickle was saved.[/yellow]")
        viewer.close()


if __name__ == "__main__":
    main()
