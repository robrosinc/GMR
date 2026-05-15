import argparse
import os
import pathlib
import time
import pickle

import mujoco as mj
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.motion_utils import (
    build_motion_data,
    get_default_keybody_names,
)
from rich import print
from tqdm import tqdm


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos", "robros_igris_c_v2"],
        default="unitree_g1",
    )
    
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=30,
        type=int,
    )
    
    args = parser.parse_args()
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Load BVH trajectory
    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    motion_fps = args.motion_fps
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            root_quat_scalar_first=True
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0
    
    if args.save_path is not None:
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
        keybody_pos_samples = []
        keybody_rot_wxyz_samples = []
        mj_data_save = mj.MjData(retargeter.model)


    while True:
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data)
        

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            follow_camera=True,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        if args.save_path is not None:
            qpos_list.append(qpos.copy())
            mj_data_save.qpos[:] = qpos
            mj.mj_forward(retargeter.model, mj_data_save)
            keybody_pos_samples.append(mj_data_save.xpos[keybody_ids].copy())
            keybody_rot_wxyz_samples.append(mj_data_save.xquat[keybody_ids].copy())

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break

    if args.save_path is not None:
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
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
       
