import argparse
import pathlib
import os
import time
import pickle

import mujoco as mj
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.utils.motion_utils import (
    build_motion_data,
    get_default_keybody_names,
)

from rich import print

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/TWIST-dev/motion_data/AMASS/KIT_572_dance_chacha11_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1Running_c3d/Run_C24_-_quick_side_step_left_stageii.npz",
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
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    # joint_names = JOINT_NAMES[: len(body_model.parents)]
    # print(f"[info] Loaded SMPL-X file with {len(joint_names)} joints. Order:")
    # print(", ".join(joint_names))
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
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
                                            video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)
    

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
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

        # retarget
        qpos = retarget.retarget(smplx_data)
        # print('smplx data:', qpos.shape)

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
            
      
    
    robot_motion_viewer.close()
