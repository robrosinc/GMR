import argparse
import pathlib
import os
import mujoco as mj
import numpy as np
from tqdm import tqdm
import pickle

from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.motion_utils import (
    build_motion_data,
    get_default_keybody_names,
)
from rich import print


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

            # retarget to get all qpos
            qpos_list = []
            for curr_frame in range(len(lafan1_data_frames)):
                smplx_data = lafan1_data_frames[curr_frame]
                
                # Retarget till convergence
                qpos = retarget.retarget(smplx_data)
                
                qpos_list.append(qpos.copy())
            
            qpos_list = np.array(qpos_list)

            keybody_candidates = get_default_keybody_names(args.robot)
            keybody_names = [
                name
                for name in keybody_candidates
                if name in retarget.robot_body_names
            ]
            missing_keybodies = [name for name in keybody_candidates if name not in retarget.robot_body_names]
            if missing_keybodies:
                print(f"[warn] Missing keybodies in robot model: {missing_keybodies}")

            motion_data = _build_bvh_motion_data_from_qpos(
                aligned_fps=src_fps,
                model=retarget.model,
                qpos=qpos_list,
                keybody_names=keybody_names,
            )
            

            os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
            with open(tgt_file_path, "wb") as f:
                pickle.dump(motion_data, f)

    if allowed_motion_stems is not None:
        print("skipped by filtered_motion_paths:", skipped_not_in_filtered)
    print("Done. saved to ", tgt_folder)
