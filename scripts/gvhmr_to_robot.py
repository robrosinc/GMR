import argparse
import pathlib
import os
import time

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print

def finite_difference(data: np.ndarray, dt: float) -> np.ndarray:
    if len(data) < 2:
        return np.zeros_like(data)
    vel = np.zeros_like(data)
    vel[1:-1] = (data[2:] - data[:-2]) / (2.0 * dt)
    vel[0] = (data[1] - data[0]) / dt
    vel[-1] = (data[-1] - data[-2]) / dt
    return vel


def quaternion_angular_velocity(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    if len(quats_wxyz) < 2:
        return np.zeros((len(quats_wxyz), 3))
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    rotations = R.from_quat(quats_xyzw)
    ang_vel = np.zeros((len(quats_wxyz), 3))
    for i in range(1, len(quats_wxyz) - 1):
        rel = rotations[i - 1].inv() * rotations[i + 1]
        ang_vel[i] = rel.as_rotvec() / (2.0 * dt)
    ang_vel[0] = (rotations[0].inv() * rotations[1]).as_rotvec() / dt
    ang_vel[-1] = (rotations[-2].inv() * rotations[-1]).as_rotvec() / dt
    return ang_vel

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
        # keybody_names = ['Link_Wrist_Pitch_Left', 'Link_Wrist_Pitch_Right', 'Link_Ankle_Pitch_Left', 'Link_Ankle_Pitch_Right']
        if args.robot == 'robros_igris_c':
            keybody_names = ['Link_Wrist_Pitch_Left', 'Link_Wrist_Pitch_Right', 
                        'Link_Ankle_Pitch_Left', 'Link_Ankle_Pitch_Right',
                        'Link_Shoulder_Pitch_Left', 'Link_Shoulder_Pitch_Right',
                        'Link_Hip_Pitch_Left', 'Link_Hip_Pitch_Right',
                        'Link_Elbow_Pitch_Left', 'Link_Elbow_Pitch_Right',
                        'Link_Knee_Pitch_Left', 'Link_Knee_Pitch_Right',
                        'Link_Neck_Pitch']
        elif args.robot == 'robros_igris_max':
            keybody_names = ['Left_Arm_Wrist_Roll', 'Right_Arm_Wrist_Roll', 
                        'Left_Leg_Ankle_Roll_Foot', 'Right_Leg_Ankle_Roll_Foot',
                        'Left_Arm_Shoulder_Pitch', 'Right_Arm_Shoulder_Pitch',
                        'Left_Leg_Hip_Pitch', 'Right_Leg_Hip_Pitch',
                        'Left_Arm_Elbow', 'Right_Arm_Elbow',
                        'Left_Leg_Knee', 'Right_Leg_Knee',
                        'Waist_Yaw_Torso']
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
        keybody_samples = []
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
            keybody_samples.append(mj_data_save.xpos[keybody_ids].copy())
            
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        root_rot_wxyz = np.array([qpos[3:7] for qpos in qpos_list])
        root_rot = root_rot_wxyz[:, [1, 2, 3, 0]]
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])

        dt = 1.0 / aligned_fps
        root_vel = finite_difference(root_pos, dt)
        dof_vel = finite_difference(dof_pos, dt)
        root_angvel = quaternion_angular_velocity(root_rot_wxyz, dt)

        keybody_pos = np.stack(keybody_samples) if keybody_samples else np.zeros((0, len(keybody_ids), 3))
        print(keybody_ids)
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "root_vel": root_vel,
            "root_angvel": root_angvel,
            "dof_vel": dof_vel,
            "keybody_pos": keybody_pos,
            "local_body_pos": None,
            "link_body_list": keybody_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
            
      
    
    robot_motion_viewer.close()
