#!/usr/bin/env python3
"""
Serve a lightweight browser debug view for GMR/PICO retarget_frame ROS2 data.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


def ensure_system_python_for_ros() -> None:
    if os.environ.get("PICO_WEB_DEBUG_NO_REEXEC") == "1":
        return
    system_python = "/usr/bin/python3"
    if sys.executable == system_python or not os.path.exists(system_python):
        return
    env = os.environ.copy()
    env["PICO_WEB_DEBUG_NO_REEXEC"] = "1"
    os.execve(system_python, [system_python, *sys.argv], env)


ensure_system_python_for_ros()

import numpy as np


IGRIS_C_EDGES = (
    ("Link_Waist_Yaw", "Link_Waist_Roll"),
    ("Link_Waist_Roll", "Link_Waist_Pitch"),
    ("Link_Waist_Pitch", "Link_Hip_Pitch_Left"),
    ("Link_Hip_Pitch_Left", "Link_Hip_Roll_Left"),
    ("Link_Hip_Roll_Left", "Link_Hip_Yaw_Left"),
    ("Link_Hip_Yaw_Left", "Link_Knee_Pitch_Left"),
    ("Link_Knee_Pitch_Left", "Link_Ankle_Pitch_Left"),
    ("Link_Ankle_Pitch_Left", "Link_Ankle_Roll_Left"),
    ("Link_Waist_Pitch", "Link_Hip_Pitch_Right"),
    ("Link_Hip_Pitch_Right", "Link_Hip_Roll_Right"),
    ("Link_Hip_Roll_Right", "Link_Hip_Yaw_Right"),
    ("Link_Hip_Yaw_Right", "Link_Knee_Pitch_Right"),
    ("Link_Knee_Pitch_Right", "Link_Ankle_Pitch_Right"),
    ("Link_Ankle_Pitch_Right", "Link_Ankle_Roll_Right"),
    ("Link_Waist_Pitch", "Link_Shoulder_Pitch_Left"),
    ("Link_Shoulder_Pitch_Left", "Link_Shoulder_Roll_Left"),
    ("Link_Shoulder_Roll_Left", "Link_Shoulder_Yaw_Left"),
    ("Link_Shoulder_Yaw_Left", "Link_Elbow_Pitch_Left"),
    ("Link_Elbow_Pitch_Left", "Link_Wrist_Yaw_Left"),
    ("Link_Wrist_Yaw_Left", "Link_Wrist_Roll_Left"),
    ("Link_Wrist_Roll_Left", "Link_Wrist_Pitch_Left"),
    ("Link_Wrist_Pitch_Left", "Left_Hand"),
    ("Link_Waist_Pitch", "Link_Shoulder_Pitch_Right"),
    ("Link_Shoulder_Pitch_Right", "Link_Shoulder_Roll_Right"),
    ("Link_Shoulder_Roll_Right", "Link_Shoulder_Yaw_Right"),
    ("Link_Shoulder_Yaw_Right", "Link_Elbow_Pitch_Right"),
    ("Link_Elbow_Pitch_Right", "Link_Wrist_Yaw_Right"),
    ("Link_Wrist_Yaw_Right", "Link_Wrist_Roll_Right"),
    ("Link_Wrist_Roll_Right", "Link_Wrist_Pitch_Right"),
    ("Link_Wrist_Pitch_Right", "Right_Hand"),
    ("Link_Waist_Pitch", "Link_Neck_Yaw"),
    ("Link_Neck_Yaw", "Link_Neck_Pitch"),
)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PICO Retarget Debug</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; background: #101316; color: #e6edf3; font-family: system-ui, sans-serif; }
    body { overflow: hidden; }
    #hud { position: fixed; left: 16px; top: 12px; z-index: 2; font-size: 14px; line-height: 1.45; color: #d5dde5; }
    #hud b { color: #ffffff; }
    #canvas { display: block; width: 100vw; height: 100vh; touch-action: none; }
    #empty { position: fixed; inset: 0; display: grid; place-items: center; color: #8b949e; font-size: 18px; pointer-events: none; }
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <div id="hud">
    <div><b>PICO Retarget Debug</b></div>
    <div id="status">connecting...</div>
    <div>drag: rotate, wheel/pinch: zoom</div>
  </div>
  <div id="empty">waiting for keybody positions</div>
  <script>
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const emptyEl = document.getElementById("empty");
    let latest = null;
    let yaw = -0.75;
    let pitch = -0.35;
    let zoom = 190;
    const pointers = new Map();
    let lastDrag = null;
    let lastPinchDistance = null;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(window.innerWidth * dpr);
      canvas.height = Math.floor(window.innerHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    window.addEventListener("resize", resize);
    resize();

    function clampZoom() {
      zoom = Math.max(60, Math.min(600, zoom));
    }

    function pointerDistance() {
      const pts = Array.from(pointers.values());
      if (pts.length < 2) return null;
      return Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
    }

    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 1) {
        lastDrag = { x: e.clientX, y: e.clientY };
        lastPinchDistance = null;
      } else if (pointers.size === 2) {
        lastDrag = null;
        lastPinchDistance = pointerDistance();
      }
    });

    canvas.addEventListener("pointermove", (e) => {
      if (!pointers.has(e.pointerId)) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 1 && lastDrag) {
        yaw += (e.clientX - lastDrag.x) * 0.008;
        pitch -= (e.clientY - lastDrag.y) * 0.008;
        pitch = Math.max(-1.45, Math.min(1.45, pitch));
        lastDrag = { x: e.clientX, y: e.clientY };
        return;
      }
      if (pointers.size === 2) {
        const distance = pointerDistance();
        if (distance !== null && lastPinchDistance !== null && lastPinchDistance > 0) {
          zoom *= distance / lastPinchDistance;
          clampZoom();
        }
        lastPinchDistance = distance;
      }
    });

    function releasePointer(e) {
      pointers.delete(e.pointerId);
      if (pointers.size === 1) {
        const p = Array.from(pointers.values())[0];
        lastDrag = { x: p.x, y: p.y };
        lastPinchDistance = null;
      } else {
        lastDrag = null;
        lastPinchDistance = null;
      }
    }

    canvas.addEventListener("pointerup", releasePointer);
    canvas.addEventListener("pointercancel", releasePointer);
    canvas.addEventListener("lostpointercapture", releasePointer);

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      zoom *= Math.exp(-e.deltaY * 0.001);
      clampZoom();
    }, { passive: false });

    const source = new EventSource("/events");
    source.onmessage = (event) => {
      latest = JSON.parse(event.data);
      const age = latest.age_ms == null ? "unknown" : `${latest.age_ms} ms`;
      const src = latest.source || "unknown";
      statusEl.textContent = `frames ${latest.frame_idx ?? "-"} | joints ${latest.points.length} | age ${age} | ${src}`;
    };
    source.onerror = () => {
      statusEl.textContent = "disconnected";
    };

    function rotatePoint(p, center) {
      let x = p[0] - center[0];
      let y = p[1] - center[1];
      let z = p[2] - center[2];
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy * x + sy * y;
      const y1 = -sy * x + cy * y;
      const z1 = z;
      const y2 = cp * y1 - sp * z1;
      const z2 = sp * y1 + cp * z1;
      return [x1, y2, z2];
    }

    function project(p) {
      return [window.innerWidth * 0.5 + p[0] * zoom, window.innerHeight * 0.57 - p[2] * zoom];
    }

    function draw() {
      requestAnimationFrame(draw);
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      if (!latest || latest.points.length === 0) {
        emptyEl.style.display = "grid";
        return;
      }
      emptyEl.style.display = "none";

      const pts = latest.points;
      const center = [0, 0, 0];
      for (const p of pts) {
        center[0] += p[0]; center[1] += p[1]; center[2] += p[2];
      }
      center[0] /= pts.length; center[1] /= pts.length; center[2] /= pts.length;

      const projected = pts.map((p) => project(rotatePoint(p, center)));
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#e6edf3";
      ctx.lineWidth = 2;
      for (const [a, b] of latest.edges) {
        if (!projected[a] || !projected[b]) continue;
        ctx.beginPath();
        ctx.moveTo(projected[a][0], projected[a][1]);
        ctx.lineTo(projected[b][0], projected[b][1]);
        ctx.stroke();
      }

      for (let i = 0; i < projected.length; i++) {
        const p = projected[i];
        ctx.fillStyle = i === 0 ? "#ffcf33" : "#40a6ff";
        ctx.beginPath();
        ctx.arc(p[0], p[1], 4.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    draw();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/gmr/teleop/retarget_frame")
    parser.add_argument("--ros_domain_id", "--ros-domain-id", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--update_hz", "--update-hz", type=float, default=20.0)
    return parser.parse_args()


def detect_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def parse_payload(raw: str) -> dict[str, Any] | None:
    for parser in (ast.literal_eval, json.loads):
        try:
            value = parser(raw)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def first_frame(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def payload_points(payload: dict[str, Any]) -> tuple[list[list[float]], list[str]]:
    names = payload.get("link_body_list")
    if not isinstance(names, list):
        names = []

    positions = first_frame(payload.get("keybody_pos_world"))
    if not positions:
        positions = first_frame(payload.get("keybody_pos"))
    if not positions:
        return [], names

    arr = np.asarray(positions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        return [], names
    return arr.tolist(), names


def named_edges(names: list[str], count: int) -> list[tuple[int, int]]:
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    edges = [
        (name_to_idx[a], name_to_idx[b])
        for a, b in IGRIS_C_EDGES
        if a in name_to_idx and b in name_to_idx
    ]
    if edges:
        return edges
    return [(idx, idx + 1) for idx in range(max(0, count - 1))]


class LatestFrame:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame: dict[str, Any] | None = None

    def set(self, payload: dict[str, Any]) -> None:
        points, names = payload_points(payload)
        meta = payload.get("record_meta") if isinstance(payload.get("record_meta"), dict) else {}
        unix_ms = meta.get("unix_ms") if isinstance(meta, dict) else None
        age_ms = None
        try:
            age_ms = int(time.time() * 1000) - int(unix_ms)
        except Exception:
            pass
        frame = {
            "points": points,
            "names": names,
            "edges": named_edges(names, len(points)),
            "age_ms": age_ms,
            "source": meta.get("source") if isinstance(meta, dict) else None,
            "frame_idx": meta.get("frame_idx") if isinstance(meta, dict) else None,
        }
        with self.lock:
            self.frame = frame

    def get(self) -> dict[str, Any]:
        with self.lock:
            if self.frame is None:
                return {"points": [], "names": [], "edges": [], "age_ms": None}
            return dict(self.frame)


def make_handler(latest: LatestFrame, update_hz: float):
    period = 1.0 / max(1.0, update_hz)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html()
            elif path == "/events":
                self._send_events()
            else:
                self.send_error(404)

        def _send_html(self) -> None:
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            while True:
                data = json.dumps(latest.get(), separators=(",", ":")).encode("utf-8")
                try:
                    self.wfile.write(b"data: " + data + b"\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                time.sleep(period)

    return Handler


def main() -> None:
    args = parse_args()
    host = args.host or detect_host_ip()
    if args.ros_domain_id is not None:
        os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    latest = LatestFrame()

    class RetargetFrameSubscriber(Node):
        def __init__(self):
            super().__init__("pico_web_debug")
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self.create_subscription(String, args.topic, self._callback, qos)
            self.get_logger().info(
                f"Subscribing to {args.topic} (ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', 'default')})"
            )

        def _callback(self, msg: String) -> None:
            payload = parse_payload(msg.data)
            if payload is None:
                self.get_logger().warning("Failed to parse retarget_frame payload")
                return
            latest.set(payload)

    rclpy.init()
    node = RetargetFrameSubscriber()
    server = ThreadingHTTPServer((host, args.port), make_handler(latest, args.update_hz))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Open http://{host}:{args.port}")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
