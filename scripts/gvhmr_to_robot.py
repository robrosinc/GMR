import argparse
import csv
import os
import pathlib
import pickle
import time

import mujoco as mj
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.motion_utils import (
    build_motion_data,
    get_default_keybody_names,
)
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gvhmr_pred_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/GVHMR/outputs/demo/tennis/hmr4d_results.pt",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung", "robros_igris_c_v2", 'robros_igris_max'],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    parser.add_argument(
        "--smplx_csv_path",
        default=None,
        help="Optional CSV file to dump the SMPL-X data sent to retarget() for each frame.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        args.gvhmr_pred_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
    
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.gvhmr_pred_file.split('/')[-1].split('.')[0]}.mp4",)

    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    processed_frame_count = 0
    smplx_csv_rows = []
    csv_joint_names = None
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
        keybody_names = get_default_keybody_names(args.robot)
        keybody_pairs = [
            (name, retarget.robot_body_names[name])
            for name in keybody_names
            if name in retarget.robot_body_names
        ]
        missing_keybodies = [name for name in keybody_names if name not in retarget.robot_body_names]
        if missing_keybodies:
            print(f"[warn] Missing keybodies in robot model: {missing_keybodies}")
        keybody_names = [name for name, _ in keybody_pairs]
        keybody_ids = [idx for _, idx in keybody_pairs]
        keybody_pos_samples = []
        keybody_rot_wxyz_samples = []
        mj_data_save = mj.MjData(retarget.model)
    
    # Start the viewer
    i = 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Update task targets.
        smplx_data = smplx_data_frames[i]
        curr_frame_idx = processed_frame_count

        # retarget
        qpos = retarget.retarget(smplx_data)
        
        if args.smplx_csv_path is not None:
            if csv_joint_names is None:
                csv_joint_names = list(smplx_data.keys())
            row_data = [curr_frame_idx]
            for joint_name in csv_joint_names:
                joint_pos, joint_quat = smplx_data[joint_name]
                pos = np.asarray(joint_pos).reshape(-1)
                quat = np.asarray(joint_quat).reshape(-1)
                joint_values = np.concatenate([pos, quat])
                row_data.append(" ".join(f"{val:.6f}" for val in joint_values))
            smplx_csv_rows.append(row_data)
        processed_frame_count += 1

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            # human_motion_data=smplx_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
        )
        if args.save_path is not None:
            qpos_list.append(qpos.copy())
            mj_data_save.qpos[:] = qpos
            mj.mj_forward(retarget.model, mj_data_save)
            keybody_pos_samples.append(mj_data_save.xpos[keybody_ids].copy())
            keybody_rot_wxyz_samples.append(mj_data_save.xquat[keybody_ids].copy())
            
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
            aligned_fps=aligned_fps,
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

    if args.smplx_csv_path is not None:
        csv_dir = os.path.dirname(args.smplx_csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(args.smplx_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["frame"]
            if csv_joint_names is not None:
                header.extend(csv_joint_names)
            writer.writerow(header)
            writer.writerows(smplx_csv_rows)
        print(f"Saved SMPL-X data to {args.smplx_csv_path}")
            
      
    
    robot_motion_viewer.close()
