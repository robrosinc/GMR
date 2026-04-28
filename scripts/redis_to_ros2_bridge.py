"""
Bridge retargeted teleop data from Redis to ROS2 topics.

Design goal:
- Keep GMR (Python 3.10) untouched.
- Run this bridge with ROS Python (usually 3.12 for Jazzy).
- Avoid extra Python dependencies (no redis-py).
"""

from __future__ import annotations

import argparse
import ast
import json
import socket
import time
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int64, String


def _topic(prefix: str, name: str) -> str:
    p = prefix.strip() or "/gmr/teleop"
    if not p.startswith("/"):
        p = f"/{p}"
    return f"{p.rstrip('/')}/{name}"


def _to_multiarray(data: Any) -> Float64MultiArray:
    msg = Float64MultiArray()
    if data is None:
        msg.data = []
        return msg
    msg.data = [float(v) for v in data]
    return msg


def _preview(data: list[float] | None, n: int) -> str:
    if data is None:
        return "None"
    if len(data) == 0:
        return "[]"
    shown = ", ".join(f"{v:.3f}" for v in data[:n])
    suffix = ", ..." if len(data) > n else ""
    return f"[{shown}{suffix}] (len={len(data)})"


class RedisRespClient:
    """Tiny Redis RESP client for GET/MGET commands."""

    def __init__(self, host: str, port: int, timeout_sec: float):
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec
        self.sock: socket.socket | None = None
        self.reader = None

    def connect(self) -> None:
        self.close()
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_sec)
        self.sock.settimeout(self.timeout_sec)
        self.reader = self.sock.makefile("rb")

    def close(self) -> None:
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def command(self, *parts: str) -> Any:
        if self.sock is None or self.reader is None:
            self.connect()
        payload = self._encode(parts)
        assert self.sock is not None
        self.sock.sendall(payload)
        return self._read_reply()

    @staticmethod
    def _encode(parts: tuple[str, ...]) -> bytes:
        out = [f"*{len(parts)}\r\n".encode()]
        for part in parts:
            b = part.encode()
            out.append(f"${len(b)}\r\n".encode())
            out.append(b + b"\r\n")
        return b"".join(out)

    def _readline(self) -> bytes:
        if self.reader is None:
            raise ConnectionError("Redis reader is not initialized")
        line = self.reader.readline()
        if not line:
            raise ConnectionError("Redis connection closed")
        return line

    def _read_reply(self) -> Any:
        line = self._readline()
        prefix = line[:1]
        payload = line[1:-2]  # strip marker + CRLF

        if prefix == b"+":
            return payload.decode()
        if prefix == b":":
            return int(payload)
        if prefix == b"$":
            n = int(payload)
            if n == -1:
                return None
            if self.reader is None:
                raise ConnectionError("Redis reader is not initialized")
            data = self.reader.read(n + 2)
            if data is None or len(data) < n + 2:
                raise ConnectionError("Failed to read bulk string from Redis")
            return data[:-2]
        if prefix == b"*":
            n = int(payload)
            if n == -1:
                return None
            return [self._read_reply() for _ in range(n)]
        if prefix == b"-":
            raise RuntimeError(f"Redis error: {payload.decode(errors='replace')}")
        raise RuntimeError(f"Unknown Redis RESP prefix: {prefix!r}")


class RedisToRos2Bridge(Node):
    def __init__(self, args):
        super().__init__(args.ros_node_name)
        self.args = args
        self.client = RedisRespClient(args.redis_host, args.redis_port, args.redis_timeout_sec)

        suffix = self._resolve_action_suffix(args.robot)
        self.redis_keys = {
            "body": f"action_body_{suffix}",
            "hand_left": f"action_hand_left_{suffix}",
            "hand_right": f"action_hand_right_{suffix}",
            "neck": f"action_neck_{suffix}",
            "qpos": f"action_qpos_{suffix}",
            "dof_pos": f"action_dof_pos_{suffix}",
            "retarget_frame": f"action_retarget_frame_{suffix}",
            "controller": "controller_data",
            "t_action": "t_action",
        }

        qos_depth = max(1, int(args.ros_qos_depth))
        self.pub_body = self.create_publisher(
            Float64MultiArray, _topic(args.topic_prefix, "action_body"), qos_depth
        )
        self.pub_hand_left = self.create_publisher(
            Float64MultiArray, _topic(args.topic_prefix, "action_hand_left"), qos_depth
        )
        self.pub_hand_right = self.create_publisher(
            Float64MultiArray, _topic(args.topic_prefix, "action_hand_right"), qos_depth
        )
        self.pub_neck = self.create_publisher(
            Float64MultiArray, _topic(args.topic_prefix, "action_neck"), qos_depth
        )
        self.pub_qpos = self.create_publisher(
            Float64MultiArray, _topic(args.topic_prefix, "qpos"), qos_depth
        )
        self.pub_root_pos = self.create_publisher(
            Float64MultiArray, _topic(args.topic_prefix, "root_pos"), qos_depth
        )
        self.pub_root_rot = self.create_publisher(
            Float64MultiArray, _topic(args.topic_prefix, "root_rot"), qos_depth
        )
        self.pub_dof_pos = self.create_publisher(
            Float64MultiArray, _topic(args.topic_prefix, "dof_pos"), qos_depth
        )
        self.pub_retarget_frame = self.create_publisher(
            String, _topic(args.topic_prefix, "retarget_frame"), qos_depth
        )
        self.pub_controller = self.create_publisher(
            String, _topic(args.topic_prefix, "controller_data"), qos_depth
        )
        self.pub_t_action = self.create_publisher(
            Int64, _topic(args.topic_prefix, "t_action_ms"), qos_depth
        )

        self._last_error_log_time = 0.0
        self._last_stale_log_time = 0.0
        self._last_debug_log_time = 0.0
        period = 1.0 / max(1.0, float(args.poll_hz))
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            "Redis->ROS2 bridge started "
            f"(redis={args.redis_host}:{args.redis_port}, robot={args.robot}, "
            f"topic_prefix={args.topic_prefix}, poll_hz={args.poll_hz})"
        )
        self.get_logger().info(
            "Redis keys: "
            + ", ".join(f"{k}={v}" for k, v in self.redis_keys.items())
        )

    @staticmethod
    def _resolve_action_suffix(robot_name: str) -> str:
        if robot_name in ["unitree_g1", "unitree_g1_with_hands"]:
            return "unitree_g1_with_hands"
        return robot_name

    @staticmethod
    def _decode_bulk(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _tick(self) -> None:
        key_order = [
            self.redis_keys["body"],
            self.redis_keys["hand_left"],
            self.redis_keys["hand_right"],
            self.redis_keys["neck"],
            self.redis_keys["qpos"],
            self.redis_keys["dof_pos"],
            self.redis_keys["retarget_frame"],
            self.redis_keys["controller"],
            self.redis_keys["t_action"],
        ]
        try:
            values = self.client.command("MGET", *key_order)
            if values is None:
                return
            if not isinstance(values, list) or len(values) != len(key_order):
                raise RuntimeError(f"Unexpected MGET reply: {type(values)}")
            self._publish_values(values)
        except Exception as exc:
            self.client.close()
            now = time.time()
            if now - self._last_error_log_time > 2.0:
                self.get_logger().warning(f"Redis read failed: {exc}")
                self._last_error_log_time = now

    def _publish_values(self, values: list[Any]) -> None:
        body_raw, l_raw, r_raw, neck_raw, qpos_raw, dof_raw, frame_raw, ctrl_raw, t_raw = values
        body = self._parse_json_list(body_raw)
        left_hand = self._parse_json_list(l_raw)
        right_hand = self._parse_json_list(r_raw)
        neck = self._parse_json_list(neck_raw)
        qpos = self._parse_json_list(qpos_raw)
        dof_pos = self._parse_json_list(dof_raw)
        retarget_frame = self._parse_json_dict(frame_raw)
        controller = self._decode_bulk(ctrl_raw)
        t_action = self._parse_int(t_raw)

        if body is not None:
            self.pub_body.publish(_to_multiarray(body))
        if dof_pos is not None:
            self.pub_dof_pos.publish(_to_multiarray(dof_pos))
        elif body is not None and len(body) >= 6:
            # fallback layout: body obs = [base(6), dof...]
            self.pub_dof_pos.publish(_to_multiarray(body[6:]))

        if left_hand is not None:
            self.pub_hand_left.publish(_to_multiarray(left_hand))
        if right_hand is not None:
            self.pub_hand_right.publish(_to_multiarray(right_hand))
        if neck is not None:
            self.pub_neck.publish(_to_multiarray(neck))
        if controller is not None:
            msg = String()
            msg.data = controller
            self.pub_controller.publish(msg)
        if t_action is not None:
            tmsg = Int64()
            tmsg.data = t_action
            self.pub_t_action.publish(tmsg)
            self._check_staleness(t_action)

        if qpos is not None:
            self.pub_qpos.publish(_to_multiarray(qpos))
            if len(qpos) >= 7:
                self.pub_root_pos.publish(_to_multiarray(qpos[:3]))
                self.pub_root_rot.publish(_to_multiarray(qpos[3:7]))
                if dof_pos is None:
                    self.pub_dof_pos.publish(_to_multiarray(qpos[7:]))

        if retarget_frame is None and qpos is not None and len(qpos) >= 7:
            retarget_frame = self._build_fallback_retarget_frame(qpos)
        if retarget_frame is not None:
            msg = String()
            msg.data = self._to_dict_literal_string(retarget_frame)
            self.pub_retarget_frame.publish(msg)

        self._debug_log(body, dof_pos, qpos, left_hand, right_hand, neck, t_action)

    def _debug_log(
        self,
        body: list[float] | None,
        dof_pos: list[float] | None,
        qpos: list[float] | None,
        left_hand: list[float] | None,
        right_hand: list[float] | None,
        neck: list[float] | None,
        t_action_ms: int | None,
    ) -> None:
        if self.args.debug_log_sec <= 0:
            return
        now = time.time()
        if now - self._last_debug_log_time < self.args.debug_log_sec:
            return

        dof = dof_pos
        if dof is None and body is not None and len(body) >= 6:
            dof = body[6:]
        age_ms = None
        if t_action_ms is not None:
            age_ms = int(time.time() * 1000) - t_action_ms

        self.get_logger().info(
            "bridge_debug "
            f"body={_preview(body, self.args.debug_preview_dim)} "
            f"qpos={_preview(qpos, self.args.debug_preview_dim)} "
            f"dof={_preview(dof, self.args.debug_preview_dim)} "
            f"neck={_preview(neck, self.args.debug_preview_dim)} "
            f"hand_l={_preview(left_hand, self.args.debug_preview_dim)} "
            f"hand_r={_preview(right_hand, self.args.debug_preview_dim)} "
            f"t_action_age_ms={age_ms}"
        )
        self._last_debug_log_time = now

    def _check_staleness(self, t_action_ms: int) -> None:
        now_ms = int(time.time() * 1000)
        age = now_ms - t_action_ms
        if age <= self.args.stale_warn_ms:
            return
        now = time.time()
        if now - self._last_stale_log_time > 2.0:
            self.get_logger().warning(
                f"Stale teleop data detected: age_ms={age}, threshold_ms={self.args.stale_warn_ms}"
            )
            self._last_stale_log_time = now

    @staticmethod
    def _parse_json_list(raw: Any) -> list[float] | None:
        s = RedisToRos2Bridge._decode_bulk(raw)
        if s is None:
            return None
        try:
            val = json.loads(s)
        except Exception:
            return None
        if not isinstance(val, list):
            return None
        out = []
        for x in val:
            try:
                out.append(float(x))
            except Exception:
                return None
        return out

    @staticmethod
    def _parse_json_dict(raw: Any) -> dict[str, Any] | None:
        s = RedisToRos2Bridge._decode_bulk(raw)
        if s is None:
            return None
        try:
            val = json.loads(s)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
        try:
            val = ast.literal_eval(s)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
        return None

    @staticmethod
    def _to_dict_literal_string(payload: dict[str, Any]) -> str:
        # Publish as Python dict literal style:
        # data: {'fps': ..., 'root_pos': ...}
        return repr(payload)

    def _build_fallback_retarget_frame(self, qpos: list[float]) -> dict[str, Any]:
        dof = qpos[7:]
        zero_dof = [0.0] * len(dof)
        frame = {
            "fps": float(self.args.poll_hz),
            "root_pos": [qpos[:3]],
            "root_rot": [qpos[3:7]],
            "root_vel": [[0.0, 0.0, 0.0]],
            "root_angvel": [[0.0, 0.0, 0.0]],
            "dof_pos": [dof],
            "dof_vel": [zero_dof],
            "keybody_pos_world": [],
            "keybody_pos_local": [],
            "keybody_rot_world": [],
            "keybody_rot_local": [],
            "keybody_pos": [],
            "link_body_list": [],
            "local_body_link_body_list": None,
            "local_body_pos": None,
        }
        return frame

    @staticmethod
    def _parse_int(raw: Any) -> int | None:
        s = RedisToRos2Bridge._decode_bulk(raw)
        if s is None:
            return None
        try:
            return int(s)
        except Exception:
            return None

    def destroy_node(self) -> bool:
        self.client.close()
        return super().destroy_node()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "robros_igris_c_v2"],
        default="robros_igris_c_v2",
    )
    parser.add_argument("--redis_host", type=str, default="127.0.0.1")
    parser.add_argument("--redis_port", type=int, default=6379)
    parser.add_argument("--redis_timeout_sec", type=float, default=0.2)

    parser.add_argument("--topic_prefix", type=str, default="/gmr/teleop")
    parser.add_argument("--ros_node_name", type=str, default="gmr_redis_ros2_bridge")
    parser.add_argument("--ros_qos_depth", type=int, default=10)
    parser.add_argument("--poll_hz", type=float, default=100.0)
    parser.add_argument("--stale_warn_ms", type=int, default=150)
    parser.add_argument(
        "--debug_log_sec",
        type=float,
        default=2.0,
        help="Periodic debug print interval in seconds (<=0 to disable).",
    )
    parser.add_argument(
        "--debug_preview_dim",
        type=int,
        default=6,
        help="How many leading values to print for each topic payload preview.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = RedisToRos2Bridge(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
