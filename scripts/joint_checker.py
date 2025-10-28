#!/usr/bin/env python3
"""Interactive MuJoCo viewer for the Unitree G1.

This tool loads the Unitree G1 model in MuJoCo, exposes basic keyboard controls
to step through joints, and prints quaternion information for the selected link.
It is designed to help inspect per-joint orientations while manually tweaking
joint positions in real time.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from general_motion_retargeting.params import ROBOT_XML_DICT


def _import_mujoco():
    """Lazy import so the script can provide a friendly hint if mujoco is absent."""
    try:
        mj = importlib.import_module("mujoco")
        mjv = importlib.import_module("mujoco.viewer")
    except ImportError as err:  # pragma: no cover - requires mujoco installation
        raise SystemExit(
            "This script needs the `mujoco` package (>=2.3). "
            "Install it first, e.g. `pip install mujoco`."
        ) from err
    return mj, mjv


def _resolve_model_path(robot: str, override_path: Optional[str]) -> str:
    if override_path:
        return override_path
    try:
        return str(ROBOT_XML_DICT[robot])
    except KeyError as exc:
        raise SystemExit(
            f"Unknown robot '{robot}'. Available options: {', '.join(sorted(ROBOT_XML_DICT))}"
        ) from exc


@dataclass
class JointInfo:
    index: int
    name: str
    joint_type: str
    body_id: int
    body_name: str
    qpos_indices: Iterable[int]
    qpos_values: np.ndarray
    qpos_range: Optional[np.ndarray]
    world_quat: np.ndarray
    local_quat: np.ndarray


class JointInspector:
    """Keeps track of the currently selected joint and handles keyboard actions."""

    def __init__(self, mj, model, data):
        self.mj = mj
        self.model = model
        self.data = data
        self.initial_qpos = data.qpos.copy()
        self.step_size = 0.05  # radians or meters depending on joint type

        # Keep only hinge/slide joints for direct control.
        controllable_types = {
            mj.mjtJoint.mjJNT_HINGE,
            mj.mjtJoint.mjJNT_SLIDE,
        }
        self.control_joint_ids: List[int] = [
            j_idx for j_idx in range(model.njnt) if model.jnt_type[j_idx] in controllable_types
        ]
        if not self.control_joint_ids:
            raise SystemExit("No hinge/slide joints found to control.")

        self.control_index = 0

        # Pre-compute a lookup table for readable joint type names.
        self.joint_type_name: dict[int, str] = {
            mj.mjtJoint.mjJNT_FREE: "free",
            mj.mjtJoint.mjJNT_BALL: "ball",
            mj.mjtJoint.mjJNT_SLIDE: "slide",
            mj.mjtJoint.mjJNT_HINGE: "hinge",
        }

    # ------------------------------------------------------------------
    # Helpers for joint/Jacobian bookkeeping
    # ------------------------------------------------------------------
    def _joint_qpos_slice(self, joint_id: int) -> slice:
        start = self.model.jnt_qposadr[joint_id]
        if joint_id == self.model.njnt - 1:
            stop = self.model.nq
        else:
            stop = self.model.jnt_qposadr[joint_id + 1]
        return slice(start, stop)

    def _joint_type_name(self, joint_id: int) -> str:
        return self.joint_type_name.get(self.model.jnt_type[joint_id], "unknown")

    def _joint_name(self, joint_id: int) -> str:
        if hasattr(self.model, "joint_id2name"):
            name = self.model.joint_id2name(joint_id)  # type: ignore[attr-defined]
            if name:
                return name
        try:
            return self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        except (TypeError, AttributeError, ValueError):
            return f"joint_{joint_id}"

    def _body_name(self, body_id: int) -> str:
        if hasattr(self.model, "body_id2name"):
            name = self.model.body_id2name(body_id)  # type: ignore[attr-defined]
            if name:
                return name
        try:
            return self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        except (TypeError, AttributeError, ValueError):
            return f"body_{body_id}"

    def _body_world_quat(self, body_id: int) -> np.ndarray:
        return np.array(self.data.xquat[body_id], copy=True)

    def _body_local_quat(self, body_id: int) -> np.ndarray:
        parent_id = self.model.body_parentid[body_id]
        local = np.zeros(4)
        world_quat = self._body_world_quat(body_id)
        if parent_id == -1:
            return world_quat
        parent_world = self._body_world_quat(parent_id)
        parent_conj = parent_world.copy()
        parent_conj[1:] *= -1
        self.mj.mju_mulQuat(local, parent_conj, world_quat)
        return local

    def _current_joint_id(self) -> int:
        return self.control_joint_ids[self.control_index]

    def current_joint_info(self) -> JointInfo:
        joint_id = self._current_joint_id()
        j_slice = self._joint_qpos_slice(joint_id)
        body_id = self.model.jnt_bodyid[joint_id]
        info = JointInfo(
            index=joint_id,
            name=self._joint_name(joint_id),
            joint_type=self._joint_type_name(joint_id),
            body_id=body_id,
            body_name=self._body_name(body_id),
            qpos_indices=range(j_slice.start, j_slice.stop),
            qpos_values=self.data.qpos[j_slice].copy(),
            qpos_range=self.model.jnt_range[joint_id].copy()
            if self.model.jnt_limited[joint_id]
            else None,
            world_quat=self._body_world_quat(body_id),
            local_quat=self._body_local_quat(body_id),
        )
        return info

    # ------------------------------------------------------------------
    # Keyboard interaction
    # ------------------------------------------------------------------
    def select_next(self):
        self.control_index = (self.control_index + 1) % len(self.control_joint_ids)

    def select_prev(self):
        self.control_index = (self.control_index - 1) % len(self.control_joint_ids)

    def adjust_current(self, delta: float):
        joint_id = self._current_joint_id()
        joint_type = self.model.jnt_type[joint_id]
        j_slice = self._joint_qpos_slice(joint_id)
        if joint_type == self.mj.mjtJoint.mjJNT_HINGE:
            self.data.qpos[j_slice.start] += delta
        elif joint_type == self.mj.mjtJoint.mjJNT_SLIDE:
            self.data.qpos[j_slice.start] += delta
        else:
            return  # Safety guard, though controllable list filters these out.

        if self.model.jnt_limited[joint_id]:
            lower, upper = self.model.jnt_range[joint_id]
            self.data.qpos[j_slice.start] = np.clip(self.data.qpos[j_slice.start], lower, upper)

        self.mj.mj_forward(self.model, self.data)

    def reset_pose(self):
        self.data.qpos[:] = self.initial_qpos
        self.mj.mj_forward(self.model, self.data)

    def modify_step(self, scale: float):
        self.step_size = float(np.clip(self.step_size * scale, 1e-4, 1.0))

    def print_current_joint(self):
        info = self.current_joint_info()
        qpos_val = float(info.qpos_values[0]) if info.qpos_values.size else 0.0
        deg = np.degrees(qpos_val)
        quat_local = np.array2string(info.local_quat, precision=4, suppress_small=True)
        quat_world = np.array2string(info.world_quat, precision=4, suppress_small=True)
        print(
            f"[{info.index}] {info.name} ({info.joint_type})\n"
            f"  body: {info.body_name}\n"
            f"  qpos: {qpos_val:.4f} rad ({deg:.2f} deg)\n"
            f"  local quaternion: {quat_local}\n"
            f"  world quaternion: {quat_world}",
            flush=True,
        )

    def print_all_quaternions(self):
        print("\n=== Body quaternions (world frame) ===")
        for body_id in range(self.model.nbody):
            name = self._body_name(body_id)
            quat = np.array2string(self._body_world_quat(body_id), precision=4, suppress_small=True)
            print(f"[{body_id:02d}] {name}: {quat}")
        print("=== End ===\n", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="unitree_g1", help="Robot key defined in ROBOT_XML_DICT.")
    parser.add_argument(
        "--xml-path", default=None, help="Override path to an MJCF/URDF file (skips ROBOT_XML_DICT lookup)."
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Initial adjustment step size in radians (hinge) or meters (slide).",
    )
    args = parser.parse_args(argv)

    xml_path = _resolve_model_path(args.robot, args.xml_path)
    mj, mjv = _import_mujoco()

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    inspector = JointInspector(mj, model, data)
    inspector.step_size = args.step
    keyboard_callback = _make_keyboard_callback(inspector)

    overlay_top_left, overlay_bottom_left = _resolve_overlay_slots(mj, mjv)

    try:
        viewer_ctx = mjv.launch_passive(
            model, data, key_callback=keyboard_callback
        )  # pragma: no cover - requires GUI
        callback_attached = True
    except TypeError:
        viewer_ctx = mjv.launch_passive(model, data)  # pragma: no cover - requires GUI
        callback_attached = False

    with contextlib.closing(viewer_ctx) as viewer:  # pragma: no cover - requires GUI
        if not callback_attached:
            if not _install_keyboard_callback(viewer, keyboard_callback):
                print(
                    "Warning: Unable to register MuJoCo keyboard callback; joint controls disabled.",
                    flush=True,
                )

        overlay_writer = _make_overlay_writer(viewer, overlay_top_left, overlay_bottom_left)

        print(
            "Interactive Unitree G1 viewer controls:\n"
            "  [ / ] : select previous / next controllable joint\n"
            "  , / . : decrease / increase the selected joint by the current step size\n"
            "  - / = : halve / double the step size\n"
            "  r     : reset pose to the default qpos\n"
            "  p     : print current joint quaternion to the console\n"
            "  a     : print every body quaternion (world frame) to the console\n",
            flush=True,
        )

        while viewer.is_running():  # pragma: no cover - requires GUI
            info = inspector.current_joint_info()
            local_quat = ", ".join(f"{x:+.3f}" for x in info.local_quat)
            world_quat = ", ".join(f"{x:+.3f}" for x in info.world_quat)
            qpos_val = float(info.qpos_values[0]) if info.qpos_values.size else 0.0
            qpos_deg = np.degrees(qpos_val)
            range_text = ""
            if info.qpos_range is not None:
                range_text = f" | range [{info.qpos_range[0]:+.2f}, {info.qpos_range[1]:+.2f}]"

            overlay_writer(
                instructions=(
                    "[ / ]: prev/next joint   , / .: adjust   - / =: change step   r: reset   "
                    "p: print joint   a: print all quats"
                ),
                selection=f"Selected #{info.index} '{info.name}' ({info.joint_type}) | step {inspector.step_size:.3f}",
                joint_state=f"qpos {qpos_val:+.3f} rad ({qpos_deg:+.1f} deg){range_text}",
                quats=f"local quat [{local_quat}]   world quat [{world_quat}]",
            )
            viewer.sync()

    return 0


def _make_keyboard_callback(inspector: JointInspector):
    def callback(keycode: int, *args, **kwargs):
        _handle_key_event(keycode, inspector)

    return callback


def _resolve_overlay_slots(mj, mjv):
    """Return overlay slots compatible with both new and old MuJoCo APIs."""
    if hasattr(mjv, "Overlay"):
        return mjv.Overlay.TopLeft, mjv.Overlay.BottomLeft

    # Fallback to C enum constants exposed on mujoco module.
    if hasattr(mj, "mjtGridPos"):
        return mj.mjtGridPos.mjGRID_TOPLEFT, mj.mjtGridPos.mjGRID_BOTTOMLEFT

    # Old viewer versions used simple integer keys (0..7): 0=TopLeft, 3=BottomLeft.
    return 0, 3


def _make_overlay_writer(viewer, top_slot, bottom_slot):
    """Create an overlay writer compatible with various viewer implementations."""
    if hasattr(viewer, "user_overlay"):
        def write(instructions, selection, joint_state, quats):
            viewer.user_overlay[top_slot][0] = instructions
            viewer.user_overlay[top_slot][1] = selection
            viewer.user_overlay[bottom_slot][0] = joint_state
            viewer.user_overlay[bottom_slot][1] = quats

        return write

    # Prefix-based helpers on some viewer handles.
    if hasattr(viewer, "add_overlay"):
        def write(instructions, selection, joint_state, quats):
            viewer.add_overlay(top_slot, instructions, selection)
            viewer.add_overlay(bottom_slot, joint_state, quats)

        return write

    # As a last resort, emit only console logging.
    def fallback(instructions, selection, joint_state, quats):
        # del top_slot, bottom_slot  # unused
        print(instructions)
        print(selection)
        print(joint_state)
        print(quats)

    return fallback


def _install_keyboard_callback(viewer, callback) -> bool:
    """Try a few viewer APIs to register keyboard callbacks across MuJoCo versions."""
    # Newer MuJoCo versions expose explicit setter helpers.
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

    # Legacy API storing callbacks in a dict.
    if hasattr(viewer, "user_callbacks"):
        try:
            viewer.user_callbacks["keyboard"] = callback
            return True
        except Exception:  # pragma: no cover - attribute optional
            pass

    if hasattr(viewer, "callbacks"):
        try:
            viewer.callbacks["keyboard"] = callback
            return True
        except Exception:  # pragma: no cover - attribute optional
            pass

    return False


def _handle_key_event(key: int, inspector: JointInspector):
    if key in (ord("["), ord("{")):
        inspector.select_prev()
    elif key in (ord("]"), ord("}")):
        inspector.select_next()
    elif key in (ord(","),):
        inspector.adjust_current(-inspector.step_size)
    elif key in (ord("."),):
        inspector.adjust_current(+inspector.step_size)
    elif key in (ord("-"),):
        inspector.modify_step(0.5)
    elif key in (ord("="), ord("+")):
        inspector.modify_step(2.0)
    elif key in (ord("r"), ord("R")):
        inspector.reset_pose()
    elif key in (ord("p"), ord("P")):
        inspector.print_current_joint()
    elif key in (ord("a"), ord("A")):
        inspector.print_all_quaternions()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
