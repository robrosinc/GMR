import argparse
from pathlib import Path

from natsort import natsorted
from rich import print
from loop_rate_limiters import RateLimiter

from general_motion_retargeting import (
    MotionCurationList,
    RawXRobotMotionLoader,
    RobotMotionViewer,
    load_robot_motion,
)
from vis_controls import (
    create_control_state,
    log_controls_once,
    make_keyboard_callback,
    viewer_alive,
)


def collect_motion_files(robot_motion_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.pkl" if recursive else "*.pkl"
    motion_files = [path for path in robot_motion_dir.glob(pattern) if path.is_file()]
    return natsorted(motion_files, key=lambda x: str(x))


def resolve_curation_path(curation_txt_path: str) -> Path:
    curation_path = Path(curation_txt_path)
    if curation_path.is_absolute():
        return curation_path
    return Path(__file__).resolve().parents[1] / curation_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_dir", type=str, required=True)
    parser.add_argument(
        "--raw_xrobot_dir",
        type=str,
        default=None,
        help="Optional directory containing raw PICO/XRobot '<motion_name>_raw.pkl' files.",
    )
    parser.add_argument(
        "--raw_xrobot_skeleton_offset",
        type=float,
        nargs=3,
        default=[0.0, 1.2, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="XYZ offset from retargeted root used to place the raw XRobot skeleton.",
    )
    parser.add_argument("--non_recursive", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--curation_txt_path", type=str, default="curation.txt")
    parser.add_argument("--no_rate_limit", action="store_true")
    parser.add_argument(
        "--root_quat_scalar_first",
        type=str,
        choices=("true", "false"),
        default="true",
        help="Whether input root quaternion is scalar-first (wxyz). true/false",
    )
    args = parser.parse_args()
    root_quat_scalar_first = args.root_quat_scalar_first == "true"

    robot_motion_dir = Path(args.robot_motion_dir)
    if not robot_motion_dir.exists() or not robot_motion_dir.is_dir():
        raise NotADirectoryError(f"Motion directory not found: {robot_motion_dir}")

    motion_files = collect_motion_files(robot_motion_dir, recursive=not args.non_recursive)
    if len(motion_files) == 0:
        raise FileNotFoundError(f"No .pkl files found under: {robot_motion_dir}")

    print(f"Found {len(motion_files)} motion files in {robot_motion_dir}")
    log_controls_once(include_curation=True, log_fn=print)
    print("Playback speed: 1x")
    print(f"Root quat scalar-first: {root_quat_scalar_first}")
    print("Close the MuJoCo window to exit.")
    curation = MotionCurationList(resolve_curation_path(args.curation_txt_path))
    print(f"Curation file: {curation.path} (loaded {len(curation)} clips)")
    raw_xrobot_loader = RawXRobotMotionLoader(
        args.raw_xrobot_dir,
        motion_root_dir=robot_motion_dir,
        log_fn=print,
    )

    viewer = None
    should_stop = False
    control_state = create_control_state(enable_curation=True)
    clip_index = 0
    frame_idx = 0

    try:
        while True:
            motion_path = motion_files[clip_index]
            (
                _motion_data,
                motion_fps,
                motion_root_pos,
                motion_root_rot,
                motion_dof_pos,
                _motion_local_body_pos,
                _motion_link_body_list,
            ) = load_robot_motion(motion_path)
            playback_fps = motion_fps * control_state["speed"]
            raw_xrobot_motion = raw_xrobot_loader.load_for_motion(motion_path)

            if viewer is None:
                viewer = RobotMotionViewer(
                    robot_type=args.robot,
                    motion_fps=playback_fps,
                    root_quat_scalar_first=root_quat_scalar_first,
                    camera_follow=False,
                    key_callback=make_keyboard_callback(
                        control_state, enable_curation=True, log_fn=print
                    ),
                )
            else:
                viewer.motion_fps = playback_fps
                viewer.rate_limiter = RateLimiter(frequency=playback_fps, warn=False)

            print(f"[{clip_index + 1}/{len(motion_files)}] {motion_path}")
            try:
                current_clip_name = str(motion_path.relative_to(robot_motion_dir))
            except ValueError:
                current_clip_name = str(motion_path)
            num_frames = len(motion_root_pos)
            while True:
                if not viewer_alive(viewer):
                    should_stop = True
                    break
                if control_state["speed_dirty"]:
                    viewer.motion_fps = motion_fps * control_state["speed"]
                    viewer.rate_limiter = RateLimiter(
                        frequency=viewer.motion_fps, warn=False
                    )
                    control_state["speed_dirty"] = False
                if control_state["curation_action"] is not None:
                    if control_state["curation_action"] == "add":
                        if curation.add(current_clip_name):
                            print(
                                f"[green]Curated (+): {current_clip_name} "
                                f"({len(curation)})[/green]"
                            )
                        else:
                            print(
                                f"[yellow]Already curated: {current_clip_name} "
                                f"({len(curation)})[/yellow]"
                            )
                    elif control_state["curation_action"] == "remove":
                        if curation.remove(current_clip_name):
                            print(
                                f"[green]Curated (-): {current_clip_name} "
                                f"({len(curation)})[/green]"
                            )
                        else:
                            print(
                                f"[yellow]Not in curation: {current_clip_name} "
                                f"({len(curation)})[/yellow]"
                            )
                    control_state["curation_action"] = None
                viewer.step(
                    motion_root_pos[frame_idx],
                    motion_root_rot[frame_idx],
                    motion_dof_pos[frame_idx],
                    xrobot_motion_data=(
                        raw_xrobot_motion.frame_at(frame_idx, num_frames)
                        if raw_xrobot_motion is not None
                        else None
                    ),
                    xrobot_skeleton_offset=args.raw_xrobot_skeleton_offset,
                    rate_limit=not args.no_rate_limit,
                )
                if control_state["clip_delta"] != 0:
                    break
                if control_state["paused"]:
                    if control_state["frame_step"] != 0:
                        frame_idx = (frame_idx + control_state["frame_step"]) % num_frames
                        control_state["frame_step"] = 0
                    continue
                frame_idx += 1
                if frame_idx >= num_frames:
                    frame_idx = 0

            if should_stop:
                break

            if control_state["clip_delta"] != 0:
                clip_index = (clip_index + control_state["clip_delta"]) % len(motion_files)
                control_state["clip_delta"] = 0
                frame_idx = 0
                continue
    finally:
        if viewer is not None:
            viewer.close()
