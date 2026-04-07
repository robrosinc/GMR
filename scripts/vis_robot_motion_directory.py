import argparse
from pathlib import Path

from natsort import natsorted
from rich import print
from loop_rate_limiters import RateLimiter

from general_motion_retargeting import RobotMotionViewer, load_robot_motion


def collect_motion_files(robot_motion_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.pkl" if recursive else "*.pkl"
    motion_files = [path for path in robot_motion_dir.glob(pattern) if path.is_file()]
    return natsorted(motion_files, key=lambda x: str(x))


def viewer_alive(viewer: RobotMotionViewer) -> bool:
    if hasattr(viewer.viewer, "is_running"):
        return viewer.viewer.is_running()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_dir", type=str, required=True)
    parser.add_argument("--non_recursive", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no_rate_limit", action="store_true")
    args = parser.parse_args()

    robot_motion_dir = Path(args.robot_motion_dir)
    if not robot_motion_dir.exists() or not robot_motion_dir.is_dir():
        raise NotADirectoryError(f"Motion directory not found: {robot_motion_dir}")

    motion_files = collect_motion_files(robot_motion_dir, recursive=not args.non_recursive)
    if len(motion_files) == 0:
        raise FileNotFoundError(f"No .pkl files found under: {robot_motion_dir}")

    print(f"Found {len(motion_files)} motion files in {robot_motion_dir}")
    print("Close the MuJoCo window to exit.")

    viewer = None
    should_stop = False
    pass_idx = 0

    try:
        while True:
            pass_idx += 1
            if args.loop:
                print(f"[bold cyan]Playlist pass #{pass_idx}[/bold cyan]")

            for index, motion_path in enumerate(motion_files, start=1):
                (
                    _motion_data,
                    motion_fps,
                    motion_root_pos,
                    motion_root_rot,
                    motion_dof_pos,
                    _motion_local_body_pos,
                    _motion_link_body_list,
                ) = load_robot_motion(motion_path)

                if viewer is None:
                    viewer = RobotMotionViewer(
                        robot_type=args.robot,
                        motion_fps=motion_fps,
                        camera_follow=False,
                    )
                else:
                    viewer.motion_fps = motion_fps
                    viewer.rate_limiter = RateLimiter(frequency=motion_fps, warn=False)

                print(f"[{index}/{len(motion_files)}] {motion_path}")
                for frame_idx in range(len(motion_root_pos)):
                    if not viewer_alive(viewer):
                        should_stop = True
                        break
                    viewer.step(
                        motion_root_pos[frame_idx],
                        motion_root_rot[frame_idx],
                        motion_dof_pos[frame_idx],
                        rate_limit=not args.no_rate_limit,
                    )

                if should_stop:
                    break

            if should_stop or not args.loop:
                break
    finally:
        if viewer is not None:
            viewer.close()
