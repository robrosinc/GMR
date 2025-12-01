#!/usr/bin/env python3
"""
Realtime SMPL-X streaming retargeter.

This script listens to a ZeroMQ PUB/SUB endpoint (or a newline-delimited local
file stream such as a named pipe) for serialized SMPL-X frames and retargets them
to a robot in real-time using GMR, displaying the result via RobotMotionViewer.

Each incoming message must contain either:
    * A serialized .npz payload (i.e. bytes produced by numpy.savez) with keys such
      as 'root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye', 'trans',
      and optionally 'betas' and 'mocap_frame_rate'.
    * A JSON object (when --payload_format=json) carrying the same fields.
When reading from a file stream, emit one payload per line (e.g., write each JSON
object followed by '\n' into /tmp/smplx.pipe).

Minimal required fields per frame:
    - trans: (3,) translation
    - root_orient: (3,) axis-angle
    - pose_body: (63,) axis-angle for 21 body joints
Optional fields (recommended):
    - pose_hand: (90,) concatenated left/right hand axis-angle
      or left_hand_pose/right_hand_pose arrays of length 45 each.
    - pose_jaw: (3,), pose_eye: (6,)
    - betas: (16,) shape coefficients (if omitted, zeros are used)
    - mocap_frame_rate: scalar fps hint (viewer falls back to --default_fps)
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import time
from typing import Any, Callable, Dict, Optional, Tuple

import csv
import os

import numpy as np
import smplx
import torch
from rich import print
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

try:
    import zmq
except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "pyzmq is required for scripts/smplx_stream_to_robot.py. "
            "Install it via `pip install pyzmq`."
        ) from exc

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer

ROT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

POS_ROT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)

PELVIS_ROT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)

ROW_VALUE_SEQUENCE = (-1.0, 0.0, 1.0)
KEYCODE_RIGHT = 262  # GLFW
KEYCODE_LEFT = 263
KEYCODE_DOWN = 264
KEYCODE_UP = 265

def _parse_npz_payload(payload: bytes) -> Dict[str, Any]:
    with np.load(io.BytesIO(payload), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _parse_json_payload(payload: bytes) -> Dict[str, Any]:
    return json.loads(payload.decode("utf-8"))


def _reshape_pose_vectors(frame: Dict[str, Any]) -> Dict[str, np.ndarray]:
    def _as_np(key: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if key not in frame:
            return default
        arr = np.asarray(frame[key], dtype=np.float32)
        return arr

    if "poses" in frame and "pose_body" not in frame:
        flat = np.asarray(frame["poses"], dtype=np.float32).reshape(-1)
        if flat.shape[0] < 165:
            raise ValueError("poses vector must have at least 165 elements")
        frame = dict(frame)  # shallow copy
        frame["root_orient"] = flat[:3]
        frame["pose_body"] = flat[3 : 3 + 63]
        frame["pose_jaw"] = flat[66:69]
        frame["pose_eye"] = flat[69:75]
        frame["pose_hand"] = flat[75:165]

    pose_body = _as_np("pose_body")
    if pose_body is None:
        raise ValueError("Frame missing 'pose_body'")

    pose_hand = _as_np("pose_hand", np.zeros(90, dtype=np.float32))
    if pose_hand.shape[0] not in (0, 90):
        raise ValueError("pose_hand must have 90 elements (concatenated hands)")
    left_hand = pose_hand[:45] if pose_hand.shape[0] >= 45 else np.zeros(45, dtype=np.float32)
    right_hand = pose_hand[45:90] if pose_hand.shape[0] == 90 else np.zeros(45, dtype=np.float32)

    if "left_hand_pose" in frame or "right_hand_pose" in frame:
        left_hand = _as_np("left_hand_pose", left_hand).reshape(-1)
        right_hand = _as_np("right_hand_pose", right_hand).reshape(-1)

    parsed = {
        "trans": _as_np("trans"),
        "root_orient": _as_np("root_orient"),
        "pose_body": pose_body.reshape(-1),
        "left_hand_pose": left_hand.reshape(-1),
        "right_hand_pose": right_hand.reshape(-1),
        "pose_jaw": _as_np("pose_jaw", np.zeros(3, dtype=np.float32)).reshape(-1),
        "pose_eye": _as_np("pose_eye", np.zeros(6, dtype=np.float32)).reshape(-1),
        "betas": _as_np("betas"),
        "mocap_frame_rate": frame.get("mocap_frame_rate"),
    }
    if parsed["trans"] is None or parsed["root_orient"] is None:
        raise ValueError("Frame missing 'trans' or 'root_orient'")
    return parsed


def _joint_payload_to_gmr_frame(frame: Dict[str, Any]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if "joint_data" not in frame:
        raise ValueError("Joint payload missing 'joint_data'")

    joint_entries = frame["joint_data"]
    hand_joint_names = {"left_hand", "right_hand", "LeftHand", "RightHand"}
    gmr_frame: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for joint_name, payload in joint_entries.items():
        if joint_name in hand_joint_names:
            continue  # skip hand joints entirely
        if isinstance(payload, dict):
            pos = payload.get("pos")
            quat = payload.get("quat")
        else:
            pos, quat = payload
        if pos is None or quat is None:
            raise ValueError(f"Joint '{joint_name}' missing 'pos' or 'quat'")
        pos_arr = np.asarray(pos, dtype=np.float32).reshape(-1)
        quat_arr = np.asarray(quat, dtype=np.float32).reshape(-1)
        if pos_arr.shape[0] != 3:
            raise ValueError(f"Joint '{joint_name}' position must have 3 elements")
        if quat_arr.shape[0] != 4:
            raise ValueError(f"Joint '{joint_name}' quaternion must have 4 elements")
        norm = np.linalg.norm(quat_arr)
        if norm == 0:
            raise ValueError(f"Joint '{joint_name}' quaternion norm is zero")
        quat_arr = quat_arr / norm
        gmr_frame[joint_name] = (pos_arr, quat_arr)
    return gmr_frame


def _rotation_matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    m = np.asarray(mat, dtype=np.float32).reshape(3, 3)
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


def _smplx_output_to_gmr_frame(body_model, smplx_output) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    joints = smplx_output.joints[0].detach().cpu().numpy()
    full_pose = smplx_output.full_pose[0].detach().cpu().numpy().reshape(-1, 3)
    global_orient = smplx_output.global_orient[0].detach().cpu().numpy()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    gmr_frame = {}
    joint_rots = []
    for idx, name in enumerate(joint_names):
        if idx == 0:
            rot = R.from_rotvec(global_orient)
        else:
            rot = joint_rots[parents[idx]] * R.from_rotvec(full_pose[idx])
        joint_rots.append(rot)
        gmr_frame[name] = (joints[idx], rot.as_quat(scalar_first=True))
    return gmr_frame


class SMPLXStreamRetargeter:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        endpoint = args.stream_endpoint
        self.stream_path: Optional[pathlib.Path] = None
        if endpoint.startswith("file://"):
            self.stream_path = pathlib.Path(endpoint[len("file://") :]).expanduser()
        elif "://" not in endpoint:
            self.stream_path = pathlib.Path(endpoint).expanduser()

        self.payload_parser = (
            _parse_json_payload if args.payload_format == "json" else _parse_npz_payload
        )

        if self.stream_path is None:
            self.ctx = zmq.Context.instance()
            self.socket = self.ctx.socket(zmq.SUB)
            self.socket.connect(endpoint)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, args.topic)
        else:
            self.ctx = None
            self.socket = None

        self.body_model = smplx.create(
            model_path=args.smplx_model_path,
            model_type="smplx",
            gender=args.gender,
            use_pca=False,
        ).to(args.device)

        self.betas = None
        self.viewer = None
        self.retarget = None
        self.motion_fps = args.default_fps
        self.pos_rot_matrix = POS_ROT_MATRIX.copy()
        self.rot_matrix = ROT_MATRIX.copy()
        self.rot_quaternion = _rotation_matrix_to_quaternion(self.rot_matrix)
        self.pelvis_rot_matrix = PELVIS_ROT_MATRIX.copy()
        self.pelvis_rot_quaternion = _rotation_matrix_to_quaternion(self.pelvis_rot_matrix)
        self.cursor_row = 0
        self.cursor_col = 0
        self.csv_rows = []
        self.csv_joint_names: Optional[list[str]] = None
        self.csv_frame_count = 0

    def _init_session(
        self,
        betas: Optional[np.ndarray],
        fps_hint: Optional[float],
        human_height: Optional[float] = None,
    ) -> None:
        if betas is None:
            betas = np.zeros(16, dtype=np.float32)
        self.betas = torch.tensor(betas, dtype=torch.float32, device=self.args.device).view(1, -1)
        if human_height is None:
            human_height = 1.66 + 0.1 * float(betas[0])
        else:
            human_height = float(human_height)

        self.retarget = GMR(
            actual_human_height=human_height,
            src_human="smplx",
            tgt_robot=self.args.robot,
        )
        self.motion_fps = float(fps_hint) if fps_hint is not None else self.args.default_fps
        video_path = None
        if self.args.record_video:
            motion_name = self.args.video_tag or "smplx_stream"
            video_dir = pathlib.Path("videos")
            video_dir.mkdir(exist_ok=True)
            video_path = str(video_dir / f"{self.args.robot}_{motion_name}.mp4")
        self.viewer = RobotMotionViewer(
            robot_type=self.args.robot,
            camera_follow=not self.args.free_camera,
            motion_fps=self.motion_fps,
            transparent_robot=0,
            record_video=self.args.record_video,
            video_path=video_path,
            key_callback=self._handle_keyboard_event,
        )
        print(
            f"[green]Initialized retarget session[/green] | "
            f"height={human_height:.2f}m fps={self.motion_fps:.1f}"
        )

    def _build_smplx_output(self, frame: Dict[str, np.ndarray]):
        device = self.args.device
        betas_tensor = self.betas if self.betas is not None else torch.zeros((1, 16), device=device)
        output = self.body_model(
            betas=betas_tensor,
            global_orient=torch.tensor(frame["root_orient"], device=device).view(1, 3),
            body_pose=torch.tensor(frame["pose_body"], device=device).view(1, -1),
            transl=torch.tensor(frame["trans"], device=device).view(1, 3),
            left_hand_pose=torch.tensor(frame["left_hand_pose"], device=device).view(1, -1),
            right_hand_pose=torch.tensor(frame["right_hand_pose"], device=device).view(1, -1),
            jaw_pose=torch.tensor(frame["pose_jaw"], device=device).view(1, -1),
            leye_pose=torch.tensor(frame["pose_eye"][:3], device=device).view(1, -1),
            reye_pose=torch.tensor(frame["pose_eye"][3:], device=device).view(1, -1),
            return_full_pose=True,
        )
        return output

    def _apply_frame_rotations(self, frame: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        pos_rot_matrix = self.pos_rot_matrix
        rot_quaternion = self.rot_quaternion
        if pos_rot_matrix is None and rot_quaternion is None:
            return frame
        rotated = {}
        for joint_name, (pos, quat) in frame.items():
            pos_arr = np.asarray(pos, dtype=np.float32).reshape(3)
            quat_arr = np.asarray(quat, dtype=np.float32).reshape(4)
            if pos_rot_matrix is not None:
                pos_arr = (pos_rot_matrix @ pos_arr.reshape(3, 1)).reshape(3)
            if joint_name.lower() == "pelvis":
                apply_rot = self.pelvis_rot_quaternion
            else:
                apply_rot = rot_quaternion
            if apply_rot is not None:
                quat_arr = _quat_mul(apply_rot, quat_arr)
            rotated[joint_name] = (pos_arr, quat_arr)
        return rotated

    def _record_frame_for_csv(self, frame: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        if self.args.smplx_csv_path is None:
            return
        if self.csv_joint_names is None:
            self.csv_joint_names = list(frame.keys())
        else:
            for joint_name in frame.keys():
                if joint_name not in self.csv_joint_names:
                    self.csv_joint_names.append(joint_name)

        row_data = [self.csv_frame_count]
        for joint_name in self.csv_joint_names:
            if joint_name in frame:
                pos, quat = frame[joint_name]
            else:
                pos = np.zeros(3, dtype=np.float32)
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            values = np.concatenate(
                [
                    np.asarray(pos, dtype=np.float32).reshape(-1),
                    np.asarray(quat, dtype=np.float32).reshape(-1),
                ]
            )
            row_data.append(" ".join(f"{val:.6f}" for val in values))
        self.csv_rows.append(row_data)
        self.csv_frame_count += 1

    def _handle_keyboard_event(self, keycode: int, *args, **kwargs) -> None:
        if keycode in (KEYCODE_UP, KEYCODE_DOWN, KEYCODE_LEFT, KEYCODE_RIGHT):
            self._move_cursor(keycode)
        elif keycode == ord("4"):
            self._switch_matrix_value(
                matrix=self.pos_rot_matrix,
                matrix_name="POS_ROT_MATRIX",
            )
        elif keycode == ord("5"):
            self._switch_matrix_value(
                matrix=self.rot_matrix,
                matrix_name="ROT_MATRIX",
                post_update=self._refresh_rot_quaternion,
            )
        elif keycode == ord("6"):
            self._switch_matrix_value(
                matrix=self.pelvis_rot_matrix,
                matrix_name="PELVIS_ROT_MATRIX",
                post_update=self._refresh_pelvis_quaternion,
            )

    def _move_cursor(self, keycode: int) -> None:
        if keycode == KEYCODE_UP:
            self.cursor_row = (self.cursor_row - 1) % 3
        elif keycode == KEYCODE_DOWN:
            self.cursor_row = (self.cursor_row + 1) % 3
        elif keycode == KEYCODE_LEFT:
            self.cursor_col = (self.cursor_col - 1) % 3
        elif keycode == KEYCODE_RIGHT:
            self.cursor_col = (self.cursor_col + 1) % 3
        print(
            f"[cyan]Cursor -> row {self.cursor_row + 1}, col {self.cursor_col + 1}[/cyan]"
        )

    def _switch_matrix_value(
        self,
        matrix: np.ndarray,
        matrix_name: str,
        post_update: Optional[Callable[[], None]] = None,
    ) -> None:
        current_value = matrix[self.cursor_row, self.cursor_col]
        next_value = self._next_row_value(current_value)
        if np.isclose(current_value, next_value):
            return

        matrix[self.cursor_row, self.cursor_col] = next_value

        printable_value = int(next_value)
        if post_update is not None:
            post_update()
        self._log_matrix_state(matrix_name, printable_value)

    def _refresh_rot_quaternion(self) -> None:
        self.rot_quaternion = _rotation_matrix_to_quaternion(self.rot_matrix)

    def _refresh_pelvis_quaternion(self) -> None:
        self.pelvis_rot_quaternion = _rotation_matrix_to_quaternion(self.pelvis_rot_matrix)

    def _flush_csv(self) -> None:
        if self.args.smplx_csv_path is None or not self.csv_rows:
            return
        csv_dir = os.path.dirname(self.args.smplx_csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(self.args.smplx_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["frame"]
            if self.csv_joint_names is not None:
                header.extend(self.csv_joint_names)
            writer.writerow(header)
            writer.writerows(self.csv_rows)
        print(f"[green]Saved SMPL-X CSV to {self.args.smplx_csv_path}[/green]")

    def _log_matrix_state(
        self,
        matrix_name: str,
        printable_value: int,
    ) -> None:
        pos_repr = self._format_matrix(self.pos_rot_matrix)
        rot_repr = self._format_matrix(self.rot_matrix)
        pelvis_repr = self._format_matrix(self.pelvis_rot_matrix)
        print(
            "[cyan]"
            f"{matrix_name} [{self.cursor_row + 1}, {self.cursor_col + 1}] -> {printable_value}\n"
            f"POS_ROT_MATRIX =\n{pos_repr}\n"
            f"ROT_MATRIX =\n{rot_repr}\n"
            f"PELVIS_ROT_MATRIX =\n{pelvis_repr}"
            "[/cyan]"
        )

    @staticmethod
    def _next_row_value(current_value: float) -> float:
        for idx, value in enumerate(ROW_VALUE_SEQUENCE):
            if np.isclose(current_value, value):
                return ROW_VALUE_SEQUENCE[(idx + 1) % len(ROW_VALUE_SEQUENCE)]
        return ROW_VALUE_SEQUENCE[0]

    @staticmethod
    def _format_matrix(matrix: np.ndarray) -> str:
        def _fmt(val: float) -> str:
            return f"{int(round(val)):2d}"

        return np.array2string(matrix, formatter={"float_kind": _fmt})


    def _payload_iterator(self):
        if self.stream_path is None:
            while True:
                parts = self.socket.recv_multipart()
                yield parts[-1]
        else:
            path = self.stream_path
            print(f"[cyan]Reading frames from file stream: {path}[/cyan]")
            while True:
                try:
                    with open(path, "rb") as fp:
                        for raw_line in fp:
                            payload = raw_line.rstrip(b"\r\n")
                            if not payload:
                                continue
                            yield payload
                except FileNotFoundError:
                    print(f"[yellow]Waiting for stream file {path} to appear...[/yellow]")
                    time.sleep(0.5)
                except OSError as exc:
                    print(f"[red]File stream error:[/red] {exc}")
                    time.sleep(0.5)

    def run(self) -> None:
        if self.stream_path is None:
            print(
                f"[cyan]Listening for SMPL-X frames on {self.args.stream_endpoint} "
                f"(topic='{self.args.topic or '*'}', format={self.args.payload_format})[/cyan]"
            )
        else:
            print(
                f"[cyan]Streaming SMPL-X frames from file {self.stream_path} "
                f"(format={self.args.payload_format}, newline-delimited)[/cyan]"
            )
        try:
            for payload in self._payload_iterator():
                try:
                    frame_raw = self.payload_parser(payload)
                except Exception as exc:  # pragma: no cover - robustness
                    print(f"[red]Failed to parse frame:[/red] {exc}")
                    continue

                is_joint_payload = isinstance(frame_raw, dict) and "joint_data" in frame_raw
                if is_joint_payload:
                    try:
                        gmr_frame = _joint_payload_to_gmr_frame(frame_raw)
                    except Exception as exc:
                        print(f"[red]Invalid joint payload:[/red] {exc}")
                        continue
                    if self.retarget is None:
                        height_hint = frame_raw.get("human_height", self.args.default_human_height)
                        fps_hint = frame_raw.get("mocap_frame_rate")
                        self._init_session(frame_raw.get("betas"), fps_hint, height_hint)
                else:
                    try:
                        frame = _reshape_pose_vectors(frame_raw)
                    except Exception as exc:
                        print(f"[red]Failed to reshape SMPL-X frame:[/red] {exc}")
                        continue
                    if self.retarget is None:
                        self._init_session(frame.get("betas"), frame.get("mocap_frame_rate"))
                    smplx_output = self._build_smplx_output(frame)
                    gmr_frame = _smplx_output_to_gmr_frame(self.body_model, smplx_output)

                gmr_frame = self._apply_frame_rotations(gmr_frame)
                self._record_frame_for_csv(gmr_frame)
                qpos = self.retarget.retarget(gmr_frame)

                if self.viewer is not None:
                    self.viewer.step(
                        root_pos=qpos[:3],
                        root_rot=qpos[3:7],
                        dof_pos=qpos[7:],
                        human_motion_data=self.retarget.scaled_human_data,
                        human_pos_offset=np.zeros(3),
                        show_human_body_name=False,
                        rate_limit=self.args.rate_limit,
                    )
        except KeyboardInterrupt:
            print("[yellow]Interrupted, shutting down viewer.[/yellow]")
        finally:
            if self.viewer is not None:
                self.viewer.close()
            if self.socket is not None:
                self.socket.close()
            if self.ctx is not None:
                self.ctx.term()
            self._flush_csv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime SMPL-X stream retargeting viewer.")
    parser.add_argument(
        "--stream_endpoint",
        type=str,
        default="tcp://127.0.0.1:5555",
        help="ZeroMQ publisher endpoint (e.g., tcp://0.0.0.0:6006) or file path/file:// URI for a newline-delimited stream (e.g., /tmp/smplx.pipe).",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="",
        help="Optional ZeroMQ topic filter (empty string subscribes to all).",
    )
    parser.add_argument(
        "--payload_format",
        choices=["npz", "json"],
        default="npz",
        help="Message serialization format.",
    )
    parser.add_argument(
        "--smplx_model_path",
        type=str,
        default="assets/body_models/smplx/SMPLX_NEUTRAL.npz",
        help="Path to SMPL-X model (.npz).",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["neutral", "male", "female"],
        help="SMPL-X gender for the body model.",
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
        "--default_fps",
        type=float,
        default=30.0,
        help="Viewer FPS when the stream does not specify mocap_frame_rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device for SMPL-X body model (cpu or cuda).",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Record the viewer output to videos/<robot>_<tag>.mp4.",
    )
    parser.add_argument(
        "--smplx_csv_path",
        type=str,
        default=None,
        help="Optional CSV file to dump the SMPL-X data sent to retarget() for each frame.",
    )
    parser.add_argument(
        "--video_tag",
        type=str,
        default="smplx_stream",
        help="Filename tag used when --record_video is enabled.",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        help="Enable viewer rate limiting to align robot playback speed.",
    )
    parser.add_argument(
        "--free_camera",
        action="store_true",
        help="Disable automatic camera follow so you can tilt/rotate the viewer manually.",
    )
    parser.add_argument(
        "--default_human_height",
        type=float,
        default=1.7,
        help="Fallback human height (meters) when streaming joint_data without betas.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    streamer = SMPLXStreamRetargeter(cli_args)
    streamer.run()
