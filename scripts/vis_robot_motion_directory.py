import argparse
from pathlib import Path

from natsort import natsorted
from rich import print
from loop_rate_limiters import RateLimiter

from general_motion_retargeting import RobotMotionViewer, load_robot_motion
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_dir", type=str, required=True)
    parser.add_argument("--non_recursive", action="store_true")
    parser.add_argument("--loop", action="store_true")
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
    log_controls_once(log_fn=print)
    print("Playback speed: 1x")
    print(f"Root quat scalar-first: {root_quat_scalar_first}")
    print("Close the MuJoCo window to exit.")

    viewer = None
    should_stop = False
    control_state = create_control_state()
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

            if viewer is None:
                viewer = RobotMotionViewer(
                    robot_type=args.robot,
                    motion_fps=playback_fps,
                    root_quat_scalar_first=root_quat_scalar_first,
                    camera_follow=False,
                    key_callback=make_keyboard_callback(control_state, log_fn=print),
                )
            else:
                viewer.motion_fps = playback_fps
                viewer.rate_limiter = RateLimiter(frequency=playback_fps, warn=False)

            print(f"[{clip_index + 1}/{len(motion_files)}] {motion_path}")
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
                viewer.step(
                    motion_root_pos[frame_idx],
                    motion_root_rot[frame_idx],
                    motion_dof_pos[frame_idx],
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
