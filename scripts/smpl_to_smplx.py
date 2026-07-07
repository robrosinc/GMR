import os
import argparse
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

def rotate_root_orient(root_orient, axis, degrees):
    if degrees == 0.0:
        return root_orient

    root_rotation = R.from_rotvec(root_orient)
    rotation_offset = R.from_euler(axis, degrees, degrees=True)
    return (rotation_offset * root_rotation).as_rotvec().astype(root_orient.dtype, copy=False)


def rotate_root_translation(trans, axis, degrees):
    if degrees == 0.0:
        return trans

    rotation_offset = R.from_euler(axis, degrees, degrees=True)
    pivot = trans[0].copy()
    return (rotation_offset.apply(trans - pivot) + pivot).astype(trans.dtype, copy=False)


def offset_root_height(trans, axis, offset):
    if offset == 0.0:
        return trans

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    updated_trans = trans.copy()
    updated_trans[:, axis_index] += offset
    return updated_trans


def convert_smpl_to_smplx(input_path, output_path, gender='neutral', root_rotation_axis='y', root_rotation_degrees=0.0, rotate_translation=False, root_height_axis='y', root_height_offset=0.0):
    # Load SMPL data
    smpl_data = np.load(input_path, allow_pickle=True)
    data_dict = dict(smpl_data)  # Convert to dict for modification

    # Handle betas padding for SMPL-X (pad from 10 to 16 if necessary)
    if 'betas' in data_dict:
        betas = data_dict['betas']
        if betas.shape == (10,):
            data_dict['betas'] = np.concatenate([betas, np.zeros(6, dtype=betas.dtype)])
            print(f"Padded betas from 10 to 16 for {input_path}")
        elif betas.shape not in [(16,), (1, 16)]:
            raise ValueError(f"Unexpected betas shape: {betas.shape}. Expected (10,), (16,), or (1,16) for padding to SMPL-X.")

    # Handle mocap_frame_rate variations
    if 'mocap_framerate' in data_dict:
        data_dict['mocap_frame_rate'] = data_dict.pop('mocap_framerate')
        print(f"Renamed 'mocap_framerate' to 'mocap_frame_rate' for {input_path}")

    if 'poses' not in data_dict:
        raise ValueError("Input file does not contain 'poses' key. Is this an SMPL file?")

    poses = data_dict['poses']
    if poses.shape[1] > 72:
        poses = poses[:, :72]

    # Map to SMPL-X format
    data_dict['root_orient'] = rotate_root_orient(
        poses[:, :3],
        root_rotation_axis,
        root_rotation_degrees,
    )
    data_dict['pose_body'] = poses[:, 3:66]  # 21 joints x 3 = 63, ignoring SMPL hand poses
    if rotate_translation and 'trans' in data_dict:
        data_dict['trans'] = rotate_root_translation(
            data_dict['trans'],
            root_rotation_axis,
            root_rotation_degrees,
        )
    if 'trans' in data_dict:
        data_dict['trans'] = offset_root_height(
            data_dict['trans'],
            root_height_axis,
            root_height_offset,
        )

    # Ensure gender is set
    if 'gender' not in data_dict:
        data_dict['gender'] = np.array(gender)

    # Remove original poses key
    del data_dict['poses']

    # Save as SMPL-X npz
    np.savez(output_path, **data_dict)
    print(f"Converted {input_path} to {output_path}")

def _convert_task(task):
    (
        input_path,
        output_path,
        gender,
        root_rotation_axis,
        root_rotation_degrees,
        rotate_translation,
        root_height_axis,
        root_height_offset,
    ) = task
    try:
        convert_smpl_to_smplx(
            input_path,
            output_path,
            gender,
            root_rotation_axis,
            root_rotation_degrees,
            rotate_translation,
            root_height_axis,
            root_height_offset,
        )
        return input_path, output_path, None
    except Exception as exc:
        return input_path, output_path, repr(exc)


def collect_conversion_tasks(
    src_folder,
    tgt_folder,
    pattern="*.npz",
    overwrite=False,
    gender="neutral",
    root_rotation_axis="y",
    root_rotation_degrees=0.0,
    rotate_translation=False,
    root_height_axis="y",
    root_height_offset=0.0,
):
    src_root = Path(src_folder)
    tgt_root = Path(tgt_folder)
    input_paths = sorted(path for path in src_root.rglob(pattern) if path.is_file())
    tasks = []
    skipped = 0
    for input_path in input_paths:
        output_path = tgt_root / input_path.relative_to(src_root)
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tasks.append(
            (
                str(input_path),
                str(output_path),
                gender,
                root_rotation_axis,
                root_rotation_degrees,
                rotate_translation,
                root_height_axis,
                root_height_offset,
            )
        )
    return input_paths, tasks, skipped


def process_directory(
    src_folder,
    tgt_folder,
    gender='neutral',
    root_rotation_axis='y',
    root_rotation_degrees=0.0,
    rotate_translation=False,
    root_height_axis='y',
    root_height_offset=0.0,
    pattern="*.npz",
    num_workers=1,
    overwrite=False,
):
    src_root = Path(src_folder)
    if not src_root.is_dir():
        raise NotADirectoryError(f"Source directory not found: {src_folder}")
    num_workers = max(1, int(num_workers))
    os.makedirs(tgt_folder, exist_ok=True)

    input_paths, tasks, skipped = collect_conversion_tasks(
        src_folder,
        tgt_folder,
        pattern=pattern,
        overwrite=overwrite,
        gender=gender,
        root_rotation_axis=root_rotation_axis,
        root_rotation_degrees=root_rotation_degrees,
        rotate_translation=rotate_translation,
        root_height_axis=root_height_axis,
        root_height_offset=root_height_offset,
    )
    total = len(input_paths)
    if total == 0:
        raise FileNotFoundError(f"No files matched pattern {pattern!r} under {src_folder}")
    print(
        f"Found {total} files matching {pattern!r}. "
        f"Convert: {len(tasks)}, skipped: {skipped}, workers: {num_workers}."
    )
    if not tasks:
        print(f"Done. Converted: 0, skipped: {skipped}, total: {total}")
        return

    errors = []
    if num_workers == 1:
        for task in tqdm(tasks, total=len(tasks), desc="Converting SMPL to SMPL-X"):
            input_path, output_path, error = _convert_task(task)
            if error is not None:
                errors.append((input_path, output_path, error))
    else:
        chunksize = max(1, min(1000, len(tasks) // (num_workers * 4) or 1))
        with mp.Pool(processes=num_workers) as pool:
            results = pool.imap_unordered(_convert_task, tasks, chunksize=chunksize)
            for input_path, output_path, error in tqdm(
                results,
                total=len(tasks),
                desc=f"Converting SMPL to SMPL-X ({num_workers} workers)",
            ):
                if error is not None:
                    errors.append((input_path, output_path, error))

    if errors:
        print(f"Failed conversions: {len(errors)}")
        for input_path, output_path, error in errors[:20]:
            print(f"[ERROR] {input_path} -> {output_path}: {error}")
        raise RuntimeError(f"{len(errors)} SMPL files failed to convert.")

    print(f"Done. Converted: {len(tasks)}, skipped: {skipped}, total: {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SMPL motion data to SMPL-X format.")
    parser.add_argument("--src_folder", type=str, help="Source directory of SMPL .npz files")
    parser.add_argument("--tgt_folder", type=str, help="Target directory for SMPL-X .npz files")
    parser.add_argument("--input_file", type=str, help="Single input SMPL .npz file")
    parser.add_argument("--output_file", type=str, help="Single output SMPL-X .npz file")
    parser.add_argument("--pattern", type=str, default="*.npz",
                        help="Filename glob used in directory mode. Default: *.npz")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of worker processes used in directory mode. Use 1 to disable multiprocessing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs in directory mode.")
    parser.add_argument("--gender", type=str, default="neutral", choices=["male", "female", "neutral"],
                        help="Gender for SMPL-X model if not present in file.")
    parser.add_argument("--root_rotation_axis", type=str, default="y", choices=["x", "y", "z"],
                        help="World axis used to rotate the root orientation. Default: y.")
    parser.add_argument("--root_rotation_degrees", type=float, default=0.0,
                        help="Degrees to rotate the root orientation.")
    parser.add_argument("--rotate_translation", action="store_true",
                        help="Rotate the root translation trajectory around the first frame using the same rotation.")
    parser.add_argument("--root_height_axis", type=str, default="y", choices=["x", "y", "z"],
                        help="Translation axis used for root height offset. Default: y.")
    parser.add_argument("--root_height_offset", type=float, default=0.0,
                        help="Offset added to the root translation height axis after optional trajectory rotation.")
    args = parser.parse_args()

    if args.src_folder and args.tgt_folder:
        process_directory(
            args.src_folder,
            args.tgt_folder,
            args.gender,
            args.root_rotation_axis,
            args.root_rotation_degrees,
            args.rotate_translation,
            args.root_height_axis,
            args.root_height_offset,
            args.pattern,
            args.num_workers,
            args.overwrite,
        )
    elif args.input_file and args.output_file:
        convert_smpl_to_smplx(
            args.input_file,
            args.output_file,
            args.gender,
            args.root_rotation_axis,
            args.root_rotation_degrees,
            args.rotate_translation,
            args.root_height_axis,
            args.root_height_offset,
        )
    else:
        parser.print_help()
