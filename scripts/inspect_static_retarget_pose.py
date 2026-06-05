import argparse
import copy
import json
import pathlib
import time
from dataclasses import dataclass

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
try:
    import glfw
except ImportError:
    glfw = None
from rich import print
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import IK_CONFIG_DICT, ROBOT_BASE_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast


FRAME_COLORS = {
    "robot": (
        np.array([1.0, 0.25, 0.25, 1.0]),
        np.array([0.25, 1.0, 0.25, 1.0]),
        np.array([0.25, 0.25, 1.0, 1.0]),
    ),
    "target": (
        np.array([1.0, 0.8, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0, 1.0]),
        np.array([1.0, 0.0, 1.0, 1.0]),
    ),
}

EDITOR_KEYMAP = {
    "prev_link": (("KEY_LEFT", 263),),
    "next_link": (("KEY_RIGHT", 262),),
    "value_up": (("KEY_UP", 265),),
    "value_down": (("KEY_DOWN", 264),),
    "toggle_edit_mode": (("KEY_LEFT_CONTROL", 341), ("KEY_RIGHT_CONTROL", 345)),
    "toggle_frame_mode": (("KEY_LEFT_ALT", 342), ("KEY_RIGHT_ALT", 346)),
    "cycle_axis": (("KEY_LEFT_SHIFT", 340), ("KEY_RIGHT_SHIFT", 344)),
    "save": (("KEY_9", ord("9")),),
    "rollback": (("KEY_8", ord("8")),),
    "show_all": (("KEY_SPACE", ord(" ")),),
    "axis_x": (("KEY_X", ord("X")),),
    "axis_y": (("KEY_Y", ord("Y")),),
    "axis_z": (("KEY_Z", ord("Z")),),
}

TABLE_NAMES = {
    1: "ik_match_table1",
    2: "ik_match_table2",
}

AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


@dataclass(frozen=True)
class DisplayRow:
    table_index: int
    robot_body: str
    human_body: str
    pos_weight: float
    rot_weight: float


@dataclass
class TargetFrame:
    pos: np.ndarray
    quat: np.ndarray
    human_body: str


class DisplayState:
    def __init__(self):
        self.index = 0
        self.show_all = False
        self.frame_mode = "local"
        self.edit_mode = "translate"
        self.edit_axis = "x"
        self.dirty = True

    def move(self, delta, num_items):
        if num_items == 0:
            return
        self.index = (self.index + delta) % num_items
        self.dirty = True

    def toggle_all(self):
        self.show_all = not self.show_all
        self.dirty = True
        print(f"[cyan]Display mode:[/cyan] {'all' if self.show_all else 'single'}")

    def toggle_frame_mode(self):
        self.frame_mode = "world" if self.frame_mode == "local" else "local"
        self.dirty = True
        print(f"[cyan]Frame mode:[/cyan] {self.frame_mode}")

    def toggle_edit_mode(self):
        self.edit_mode = "rotate" if self.edit_mode == "translate" else "translate"
        self.dirty = True
        print(f"[cyan]Edit mode:[/cyan] {self.edit_mode}")

    def set_edit_axis(self, axis_name):
        self.edit_axis = axis_name
        self.dirty = True

    def cycle_edit_axis(self):
        axes = ("x", "y", "z")
        self.edit_axis = axes[(axes.index(self.edit_axis) + 1) % len(axes)]
        self.dirty = True
        print(f"[cyan]Edit axis:[/cyan] {self.edit_axis}")


class StaticRetargetPoseEditor:
    def __init__(self, args):
        self.args = args
        self.human_frame, self.src_human, self.actual_human_height, self.fps, self.num_frames = (
            self._load_reference_frame()
        )
        self.config_path = pathlib.Path(IK_CONFIG_DICT[self.src_human][self.args.robot])
        self.initial_config = self._load_config()
        self.config = copy.deepcopy(self.initial_config)

        self.retargeter = GMR(
            src_human=self.src_human,
            tgt_robot=self.args.robot,
            actual_human_height=self.actual_human_height,
            verbose=False,
        )
        self.qpos = self.retargeter.retarget(
            self.human_frame,
            offset_to_ground=self.args.offset_to_ground,
        )
        self.data = mj.MjData(self.retargeter.model)
        self.data.qpos[:] = self.qpos
        mj.mj_forward(self.retargeter.model, self.data)

        self.human_data = self._build_human_data()
        self.display_rows = self._build_display_rows()
        self.targets = self._build_targets()
        self.display_state = DisplayState()
        self.has_unsaved_changes = False

    def _load_reference_frame(self):
        motion_path = pathlib.Path(self.args.motion_file)
        if self.args.source == "smplx":
            smplx_folder = pathlib.Path(__file__).parent / ".." / "assets" / "body_models"
            smplx_data, body_model, smplx_output, detected_height = load_smplx_file(
                motion_path, smplx_folder
            )
            frames, fps = get_smplx_data_offline_fast(
                smplx_data, body_model, smplx_output, tgt_fps=self.args.tgt_fps
            )
            src_human = "smplx"
        else:
            frames, detected_height = load_bvh_file(motion_path, format="lafan1")
            fps = self.args.tgt_fps
            src_human = "bvh_lafan1"

        if not frames:
            raise ValueError(f"No frames loaded from {motion_path}")
        if self.args.frame_idx < 0 or self.args.frame_idx >= len(frames):
            raise ValueError(f"--frame_idx must be in [0, {len(frames) - 1}]")

        actual_human_height = (
            self.args.actual_human_height
            if self.args.actual_human_height is not None
            else float(detected_height)
        )
        return frames[self.args.frame_idx], src_human, actual_human_height, fps, len(frames)

    def _load_config(self):
        with self.config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_config(self, config):
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
            f.write("\n")

    def _clone_human_frame(self):
        return {
            body_name: [
                np.asarray(pos, dtype=float).copy(),
                np.asarray(quat, dtype=float).copy(),
            ]
            for body_name, (pos, quat) in self.human_frame.items()
        }

    def _build_human_data(self):
        human_data = self._clone_human_frame()
        human_data = self.retargeter.scale_human_data(
            human_data,
            self.retargeter.human_root_name,
            self.retargeter.human_scale_table,
        )
        human_data = {
            body_name: [
                np.asarray(pos, dtype=float).copy()
                - np.array([0.0, 0.0, self.retargeter.ground_offset]),
                np.asarray(quat, dtype=float).copy(),
            ]
            for body_name, (pos, quat) in human_data.items()
        }
        if self.args.offset_to_ground:
            human_data = self.retargeter.offset_human_data_to_ground(human_data)
        return human_data

    def _selected_table_indices(self):
        if self.args.table == "1":
            return (1,)
        if self.args.table == "2":
            return (2,)
        return (1, 2)

    def _build_display_rows(self):
        rows = []
        for table_index in self._selected_table_indices():
            table = self.config[TABLE_NAMES[table_index]]
            for robot_body, entry in table.items():
                human_body, pos_weight, rot_weight, _, _ = entry
                if self.args.show_inactive or pos_weight != 0 or rot_weight != 0:
                    rows.append(
                        DisplayRow(
                            table_index=table_index,
                            robot_body=robot_body,
                            human_body=human_body,
                            pos_weight=pos_weight,
                            rot_weight=rot_weight,
                        )
                    )
        return rows

    def _build_targets(self):
        targets = {}
        for row in self.display_rows:
            frame = self._target_from_config(row)
            if frame is not None:
                targets[self._row_key(row)] = frame
        return targets

    def _target_from_config(self, row):
        if row.human_body not in self.human_data:
            return None
        entry = self.config[TABLE_NAMES[row.table_index]][row.robot_body]
        cfg_pos_offset = np.asarray(entry[3], dtype=float)
        cfg_rot_offset = np.asarray(entry[4], dtype=float)
        human_pos, human_quat = self.human_data[row.human_body]

        target_quat = (
            R.from_quat(human_quat, scalar_first=True)
            * R.from_quat(cfg_rot_offset, scalar_first=True)
        ).as_quat(scalar_first=True)
        target_pos = human_pos + R.from_quat(target_quat, scalar_first=True).apply(
            cfg_pos_offset - self.retargeter.ground
        )
        return TargetFrame(pos=target_pos, quat=target_quat, human_body=row.human_body)

    def _row_key(self, row):
        return row.table_index, row.robot_body

    def _selected_row(self):
        if not self.display_rows:
            return None
        return self.display_rows[self.display_state.index]

    def _selected_target(self):
        row = self._selected_row()
        if row is None:
            return None, None
        return row, self.targets.get(self._row_key(row))

    def _robot_body_pose(self, robot_body):
        body_id = self.retargeter.robot_body_names[robot_body]
        return (
            self.data.xpos[body_id].copy(),
            self.data.xquat[body_id].copy(),
            self.data.xmat[body_id].reshape(3, 3).copy(),
        )

    def _sync_config_from_target(self, row, target):
        human_pos, human_quat = self.human_data[row.human_body]
        human_rot = R.from_quat(human_quat, scalar_first=True)
        target_rot = R.from_quat(target.quat, scalar_first=True)

        cfg_rot_offset = (human_rot.inv() * target_rot).as_quat(scalar_first=True)
        cfg_pos_offset = target_rot.inv().apply(target.pos - human_pos) + self.retargeter.ground

        entry = self.config[TABLE_NAMES[row.table_index]][row.robot_body]
        entry[3] = [float(v) for v in cfg_pos_offset]
        entry[4] = [float(v) for v in cfg_rot_offset]
        self.has_unsaved_changes = True

    def translate_selected(self, axis_name, sign):
        row, target = self._selected_target()
        if row is None or target is None:
            return
        axis = AXES[axis_name]
        if self.display_state.frame_mode == "local":
            axis = R.from_quat(target.quat, scalar_first=True).apply(axis)
        target.pos = target.pos + sign * self.args.pos_step * axis
        self._sync_config_from_target(row, target)
        self.display_state.dirty = True

    def rotate_selected(self, axis_name, sign):
        row, target = self._selected_target()
        if row is None or target is None:
            return
        axis = AXES[axis_name]
        delta = R.from_rotvec(sign * self.args.rot_step_deg * np.pi / 180.0 * axis)
        target_rot = R.from_quat(target.quat, scalar_first=True)
        if self.display_state.frame_mode == "local":
            target_rot = target_rot * delta
        else:
            target_rot = delta * target_rot
        target.quat = target_rot.as_quat(scalar_first=True)
        self._sync_config_from_target(row, target)
        self.display_state.dirty = True

    def adjust_selected_value(self, sign):
        if self.display_state.edit_mode == "translate":
            self.translate_selected(self.display_state.edit_axis, sign)
        else:
            self.rotate_selected(self.display_state.edit_axis, sign)

    def save_config(self):
        self._write_config(self.config)
        self.has_unsaved_changes = False
        print(f"[green]Saved cfg:[/green] {self.config_path}")

    def rollback_to_initial(self):
        self.config = copy.deepcopy(self.initial_config)
        self._write_config(self.initial_config)
        self.targets = self._build_targets()
        self.has_unsaved_changes = False
        self.display_state.dirty = True
        print(f"[yellow]Rolled back cfg to process start state:[/yellow] {self.config_path}")

    def print_header(self):
        print(
            f"[green]Loaded static frame[/green] source={self.src_human}, "
            f"robot={self.args.robot}, cfg={self.config_path}, "
            f"frame={self.args.frame_idx}/{self.num_frames - 1}, fps={self.fps}, "
            f"actual_human_height={self.actual_human_height:.3f}"
        )
        print(
            "[yellow]Note:[/yellow] table-2 offset edits are saved, but current GMR solve "
            "still applies table-1 offsets before both IK passes."
        )

    def print_error_report(self):
        print("\n[bold]Static IK target error report[/bold]")
        print("table | robot_body -> human_body | weights(pos, rot) | pos_err(m) | rot_err(deg)")
        for row in self.display_rows:
            target = self.targets.get(self._row_key(row))
            if row.robot_body not in self.retargeter.robot_body_names or target is None:
                print(
                    f"{row.table_index} | {row.robot_body} -> {row.human_body} | "
                    f"({row.pos_weight}, {row.rot_weight}) | missing body"
                )
                continue
            robot_pos, robot_quat, _ = self._robot_body_pose(row.robot_body)
            pos_err = np.linalg.norm(robot_pos - target.pos)
            rot_err = np.degrees(quat_angle_error_wxyz(robot_quat, target.quat))
            print(
                f"{row.table_index} | {row.robot_body} -> {row.human_body} | "
                f"({row.pos_weight:g}, {row.rot_weight:g}) | {pos_err:.4f} | {rot_err:.2f}"
            )

    def rows_to_draw(self):
        if self.display_state.show_all:
            return self.display_rows
        if not self.display_rows:
            return []
        return self.display_rows[self.display_state.index : self.display_state.index + 1]

    def launch(self):
        self.print_header()
        if self.args.print_errors:
            self.print_error_report()

        viewer = mjv.launch_passive(
            model=self.retargeter.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=make_key_callback(self),
        )
        viewer.cam.lookat = self.data.xpos[
            self.retargeter.model.body(ROBOT_BASE_DICT[self.args.robot]).id
        ]
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -12

        print("\n[green]Viewer opened.[/green] Robot frames are RGB, target frames are yellow/cyan/magenta.")
        print("Keys: Left/Right=link, Up/Down=value, Ctrl=translate/rotate, Alt=local/world.")
        print("Keys: Shift=cycle axis, X/Y/Z=axis, Space=single/all, 9=save cfg, 8=rollback cfg.")

        while viewer.is_running():
            viewer.user_scn.ngeom = 0
            self.display_state.dirty = False
            for row in self.rows_to_draw():
                self._draw_row(viewer, row)
            viewer.sync()
            time.sleep(1.0 / 60.0)

        viewer.close()

    def _draw_row(self, viewer, row):
        target = self.targets.get(self._row_key(row))
        if row.robot_body not in self.retargeter.robot_body_names or target is None:
            return
        robot_pos, _, robot_mat = self._robot_body_pose(row.robot_body)
        target_mat = R.from_quat(target.quat, scalar_first=True).as_matrix()

        draw_frame(
            viewer,
            robot_pos,
            robot_mat,
            self.args.frame_size,
            FRAME_COLORS["robot"],
            label=f"R:{row.robot_body}",
        )
        draw_frame(
            viewer,
            target.pos,
            target_mat,
            self.args.frame_size * 0.85,
            FRAME_COLORS["target"],
            label=f"T{row.table_index}:{target.human_body}->{row.robot_body}",
        )
        draw_error_line(viewer, robot_pos, target.pos, self.args.line_width)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and edit one static retargeting frame by drawing robot body frames "
            "and configured IK target frames in a MuJoCo viewer."
        )
    )
    parser.add_argument("--robot", default="robros_igris_c_v2")
    parser.add_argument("--source", choices=("smplx", "bvh_lafan1"), default="smplx")
    parser.add_argument("--motion_file", required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--tgt_fps", type=int, default=30)
    parser.add_argument("--actual_human_height", type=float, default=None)
    parser.add_argument("--table", choices=("1", "2", "both"), default="both")
    parser.add_argument("--offset_to_ground", action="store_true")
    parser.add_argument("--show_inactive", action="store_true")
    parser.add_argument("--frame_size", type=float, default=0.12)
    parser.add_argument("--line_width", type=float, default=0.004)
    parser.add_argument("--pos_step", type=float, default=0.01)
    parser.add_argument("--rot_step_deg", type=float, default=2.0)
    parser.add_argument("--print_errors", action="store_true")
    return parser.parse_args()


def make_key_callback(editor):
    key = key_codes()

    def key_callback(keycode):
        state = editor.display_state
        if is_key(keycode, key["next_link"]):
            state.move(1, len(editor.display_rows))
        elif is_key(keycode, key["prev_link"]):
            state.move(-1, len(editor.display_rows))
        elif is_key(keycode, key["value_up"]):
            editor.adjust_selected_value(1.0)
        elif is_key(keycode, key["value_down"]):
            editor.adjust_selected_value(-1.0)
        elif is_key(keycode, key["show_all"]):
            state.toggle_all()
        elif is_key(keycode, key["toggle_frame_mode"]):
            state.toggle_frame_mode()
        elif is_key(keycode, key["toggle_edit_mode"]):
            state.toggle_edit_mode()
        elif is_key(keycode, key["cycle_axis"]):
            state.cycle_edit_axis()
        elif is_key(keycode, key["save"]):
            editor.save_config()
            state.dirty = True
        elif is_key(keycode, key["rollback"]):
            editor.rollback_to_initial()
        elif is_key(keycode, key["axis_x"]):
            state.set_edit_axis("x")
        elif is_key(keycode, key["axis_y"]):
            state.set_edit_axis("y")
        elif is_key(keycode, key["axis_z"]):
            state.set_edit_axis("z")

    return key_callback


def key_codes():
    return {name: resolve_key_codes(specs) for name, specs in EDITOR_KEYMAP.items()}


def resolve_key_codes(specs):
    key_values = []
    for glfw_name, fallback in specs:
        if glfw is None:
            key_values.append(fallback)
        else:
            key_values.append(getattr(glfw, glfw_name, fallback))
    return tuple(key_values)


def is_key(keycode, key_values):
    return keycode in key_values


def draw_frame(viewer, pos, mat, size, colors, label=None):
    for axis_idx in range(3):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos,
            mat=mat.flatten(),
            rgba=colors[axis_idx],
        )
        if label is not None and axis_idx == 0:
            geom.label = label
        mj.mjv_connector(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.006,
            from_=pos,
            to=pos + size * mat[:, axis_idx],
        )
        viewer.user_scn.ngeom += 1


def draw_error_line(viewer, robot_pos, target_pos, width):
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    if np.linalg.norm(target_pos - robot_pos) < 1e-6:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_connector(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        width=width,
        from_=robot_pos,
        to=target_pos,
    )
    geom.rgba[:] = np.array([1.0, 1.0, 1.0, 0.55])
    viewer.user_scn.ngeom += 1


def quat_angle_error_wxyz(robot_quat, target_quat):
    robot_rot = R.from_quat(robot_quat, scalar_first=True)
    target_rot = R.from_quat(target_quat, scalar_first=True)
    return (robot_rot.inv() * target_rot).magnitude()


def main():
    args = parse_args()
    editor = StaticRetargetPoseEditor(args)
    editor.launch()


if __name__ == "__main__":
    main()
