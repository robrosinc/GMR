import argparse
import os

from general_motion_retargeting import (
    RawXRobotMotionLoader,
    RobotMotionViewer,
    load_robot_motion,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)
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
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
    parser.add_argument(
        "--root_quat_scalar_first",
        type=str,
        choices=("true", "false"),
        default="true",
        help="Whether input root quaternion is scalar-first (wxyz). true/false",
    )
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    root_quat_scalar_first = args.root_quat_scalar_first == "true"
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    (
        _motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        _motion_local_body_pos,
        _motion_link_body_list,
    ) = load_robot_motion(robot_motion_path)
    raw_xrobot_loader = RawXRobotMotionLoader(args.raw_xrobot_dir)
    raw_xrobot_motion = raw_xrobot_loader.load_for_motion(robot_motion_path)
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            root_quat_scalar_first=root_quat_scalar_first,
                            record_video=args.record_video, video_path=args.video_path)
    
    frame_idx = 0
    while True:
        env.step(motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                xrobot_motion_data=(
                    raw_xrobot_motion.frame_at(frame_idx, len(motion_root_pos))
                    if raw_xrobot_motion is not None
                    else None
                ),
                xrobot_skeleton_offset=args.raw_xrobot_skeleton_offset,
                rate_limit=True)
        frame_idx += 1
        if frame_idx >= len(motion_root_pos):
            frame_idx = 0
    env.close()
