import os
import joblib
import numpy as np
import pickle


# these paths are from the original OMOMO dataset
motion_path1 = "/home/robros/workspace/motion_datas/OMOMO/train_diffusion_manip_seq_joints24.p"
motion_path2 = "/home/robros/workspace/motion_datas/OMOMO/test_diffusion_manip_seq_joints24.p"
motion_path3 = "/home/robros/workspace/motion_datas/OMOMO/train_diffusion_manip_window_120_cano_joints24.p"
motion_path4 = "/home/robros/workspace/motion_datas/OMOMO/test_diffusion_manip_window_120_processed_joints24.p"
all_motion_data1 = joblib.load(motion_path1)
all_motion_data2 = joblib.load(motion_path2)
all_motion_data3 = joblib.load(motion_path3)
all_motion_data4 = joblib.load(motion_path4)

# save as individual files
target_dir = "/home/robros/workspace/GMR/data/OMOMO_SMPLX"
os.makedirs(target_dir, exist_ok=True)
for motion_data in [all_motion_data1, all_motion_data2, all_motion_data3, all_motion_data4]:
    for data_name in motion_data.keys():
        try:
            smpl_data = motion_data[data_name]
            seq_name = smpl_data['seq_name']
            # save as npz
            num_frames = smpl_data["pose_body"].shape[0]
            mocap_frame_rate = 30
            poses = np.concatenate([smpl_data["pose_body"], 
                                    np.zeros((num_frames, 102))],
                                    axis=1)
            smpl_data["poses"] = poses
            smpl_data["mocap_frame_rate"] = np.array(mocap_frame_rate)
            # use pickle to save
            with open(f"{target_dir}/{seq_name}.pkl", "wb") as f:
                pickle.dump(smpl_data, f)
            print(f"saved {seq_name}")
        except:
            print('pass', data_name)