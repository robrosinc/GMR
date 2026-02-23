import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from rich import print
from tqdm import tqdm
import os
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R


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

    
    # Load SMPLX trajectory
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
    
    # for keybody
    if args.robot == 'robros_igris_c_v2':
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
                    'Left_Leg_Knee', 'Right_Leg_Knee']
    else:
        keybody_names = []
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
    keybody_samples = []
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

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break
   
        
        if args.save_path is not None:
            qpos_list.append(qpos)
            mj_data_save.qpos[:] = qpos
            mj.mj_forward(retargeter.model, mj_data_save)
            keybody_samples.append(mj_data_save.xpos[keybody_ids].copy())

    if args.save_path is not None:
        import pickle
        
        dt = 1.0 / motion_fps
        
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        root_rot_wxyz = np.array([qpos[3:7] for qpos in qpos_list])
        root_rot = root_rot_wxyz[:, [1, 2, 3, 0]]

        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        dof_vel = finite_difference(dof_pos, dt)
        root_vel = finite_difference(root_pos, dt)
        root_angvel = quaternion_angular_velocity(root_rot_wxyz, dt)
        keybody_pos = np.stack(keybody_samples) if keybody_samples else np.zeros((0, len(keybody_ids), 3))
        print(keybody_ids)

        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "dof_vel":dof_vel,
            "root_vel":root_vel,
            "root_angvel":root_angvel,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
            "keybody_pos": keybody_pos,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
       