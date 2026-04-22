import argparse
import pathlib
import os
import mujoco as mj
import numpy as np
from tqdm import tqdm
import torch
import pickle

from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from rich import print


def _norm_rel(path_str):
    return path_str.replace("\\", "/").strip()


def load_allowed_motion_stems(filtered_motion_paths_file):
    if not filtered_motion_paths_file:
        return None
    allowed_stems = set()
    with open(filtered_motion_paths_file, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rel = line.split("|", 1)[0].strip()
            rel_norm = _norm_rel(rel)
            stem_norm = _norm_rel(str(pathlib.PurePosixPath(rel_norm).with_suffix("")))
            allowed_stems.add(stem_norm)
    print(f"Loaded allowed motions: {len(allowed_stems)} from {filtered_motion_paths_file}")
    return allowed_stems


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing BVH motion files to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        default="../../motion_data/LAFAN1_g1_gmr"
    )
    
    parser.add_argument(
        "--robot",
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
    )
    
    parser.add_argument(
        "--target_fps",
        default=30,
        type=int,
    )
    parser.add_argument(
        "--filtered_motion_paths",
        type=str,
        default=None,
        help="Optional txt of allowed motion paths. Motions not listed are skipped.",
    )

    args = parser.parse_args()
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder
    allowed_motion_stems = load_allowed_motion_stems(args.filtered_motion_paths)

   
    skipped_not_in_filtered = 0
        
    # walk over all files in src_folder
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in tqdm(sorted(filenames), desc="Retargeting files"):
            if not filename.endswith(".bvh"):
                continue
                
            # get the bvh file path
            bvh_file_path = os.path.join(dirpath, filename)
            if allowed_motion_stems is not None:
                rel_path = _norm_rel(os.path.relpath(bvh_file_path, src_folder))
                rel_stem = _norm_rel(str(pathlib.PurePosixPath(rel_path).with_suffix("")))
                if rel_stem not in allowed_motion_stems:
                    skipped_not_in_filtered += 1
                    continue
            
            # get the target file path
            tgt_file_path = bvh_file_path.replace(src_folder, tgt_folder).replace(".bvh", ".pkl")

            if os.path.exists(tgt_file_path) and not args.override:
                print(f"Skipping {bvh_file_path} because {tgt_file_path} exists")
                continue
            
            # Load LAFAN1 trajectory
            try:
                lafan1_data_frames, actual_human_height = load_lafan1_file(bvh_file_path)
                src_fps = 30  # LAFAN1 data is typically 30 FPS
            except Exception as e:
                print(f"Error loading {bvh_file_path}: {e}")
                continue

            
            # Initialize the retargeting system
            retarget = GMR(
                src_human="bvh_lafan1",
                tgt_robot=args.robot,
                actual_human_height=actual_human_height,
            )
            model = mj.MjModel.from_xml_path(retarget.xml_file)
            data = mj.MjData(model)

            

            # retarget to get all qpos
            qpos_list = []
            for curr_frame in range(len(lafan1_data_frames)):
                smplx_data = lafan1_data_frames[curr_frame]
                
                # Retarget till convergence
                qpos = retarget.retarget(smplx_data)
                
                qpos_list.append(qpos.copy())
            
            qpos_list = np.array(qpos_list)

            # Initialize the forward kinematics
            device = "cuda:0"
            kinematics_model = KinematicsModel(retarget.xml_file, device=device)
            
            root_pos = qpos_list[:, :3]
            root_rot = qpos_list[:, 3:7]
            root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
            dof_pos = qpos_list[:, 7:]
            num_frames = root_pos.shape[0]
            
            # obtain local body pos
            identity_root_pos = torch.zeros((num_frames, 3), device=device)
            identity_root_rot = torch.zeros((num_frames, 4), device=device)
            identity_root_rot[:, -1] = 1.0
            local_body_pos, _ = kinematics_model.forward_kinematics(
                identity_root_pos, 
                identity_root_rot, 
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
            )
            body_names = kinematics_model.body_names

            HEIGHT_ADJUST = False
            PERFRAME_ADJUST = False
            if HEIGHT_ADJUST:
                body_pos, _ = kinematics_model.forward_kinematics(
                    torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                    torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
                    torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
                )
                ground_offset = 0.00
                if not PERFRAME_ADJUST:
                    lowest_height = torch.min(body_pos[..., 2]).item()
                    root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
                else:
                    for i in range(root_pos.shape[0]):
                        lowest_body_part = torch.min(body_pos[i, :, 2])
                        root_pos[i, 2] = root_pos[i, 2] - lowest_body_part + ground_offset

            motion_data = {
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": local_body_pos.detach().cpu().numpy(),
                "fps": src_fps,
                "link_body_list": body_names,
            }
            

            os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
            with open(tgt_file_path, "wb") as f:
                pickle.dump(motion_data, f)

    if allowed_motion_stems is not None:
        print("skipped by filtered_motion_paths:", skipped_not_in_filtered)
    print("Done. saved to ", tgt_folder)
