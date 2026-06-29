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


def _build_bvh_motion_data_from_qpos(aligned_fps, model, qpos, keybody_names):
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"qpos must have shape (T, nq), got {qpos.shape}")
    if qpos.shape[1] != model.nq:
        raise ValueError(f"qpos width must match model.nq={model.nq}, got {qpos.shape[1]}")

    body_name_to_id = {
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id): body_id
        for body_id in range(model.nbody)
    }

    keybody_pairs = [(name, body_name_to_id[name]) for name in keybody_names if name in body_name_to_id]
    keybody_names = [name for name, _ in keybody_pairs]
    keybody_ids = [body_id for _, body_id in keybody_pairs]

    local_body_names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    ]
    local_body_ids = [body_name_to_id[name] for name in local_body_names]

    mj_data = mj.MjData(model)
    local_qpos = qpos.copy()
    local_qpos[:, :3] = 0.0
    local_qpos[:, 3:7] = np.array([1.0, 0.0, 0.0, 0.0])

    keybody_pos_samples = []
    keybody_rot_samples = []
    local_body_pos_samples = []
    for world_frame_qpos, local_frame_qpos in zip(qpos, local_qpos):
        mj_data.qpos[:] = world_frame_qpos
        mj.mj_forward(model, mj_data)
        if keybody_ids:
            keybody_pos_samples.append(mj_data.xpos[keybody_ids].copy())
            keybody_rot_samples.append(mj_data.xquat[keybody_ids].copy())

        mj_data.qpos[:] = local_frame_qpos
        mj.mj_forward(model, mj_data)
        local_body_pos_samples.append(mj_data.xpos[local_body_ids].copy())

    num_frames = qpos.shape[0]
    if keybody_pos_samples:
        keybody_pos_world = np.stack(keybody_pos_samples)
        keybody_rot_world_wxyz = np.stack(keybody_rot_samples)
    else:
        keybody_pos_world = np.zeros((num_frames, 0, 3), dtype=np.float64)
        keybody_rot_world_wxyz = np.zeros((num_frames, 0, 4), dtype=np.float64)

    return build_motion_data(
        aligned_fps=aligned_fps,
        root_pos=qpos[:, :3],
        root_rot_wxyz=qpos[:, 3:7],
        dof_pos=qpos[:, 7:],
        keybody_pos_world=keybody_pos_world,
        keybody_rot_world_wxyz=keybody_rot_world_wxyz,
        keybody_names=keybody_names,
        local_body_pos=np.stack(local_body_pos_samples),
        local_body_link_body_list=local_body_names,
    )


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
        keybody_candidates = get_default_keybody_names(args.robot)
        keybody_names = [
            name
            for name in keybody_candidates
            if name in retargeter.robot_body_names
        ]
        missing_keybodies = [name for name in keybody_candidates if name not in retargeter.robot_body_names]
        if missing_keybodies:
            print(f"[warn] Missing keybodies in robot model: {missing_keybodies}")


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

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break

    if args.save_path is not None:
        motion_data = _build_bvh_motion_data_from_qpos(
            aligned_fps=motion_fps,
            model=retargeter.model,
            qpos=np.asarray(qpos_list),
            keybody_names=keybody_names,
        )
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
       
