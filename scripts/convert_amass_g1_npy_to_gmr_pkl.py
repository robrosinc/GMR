import argparse
import pickle
import re
from pathlib import Path

import numpy as np


FPS_PATTERN = re.compile(r"_(\d+)_jpos\.npy$")
EXPECTED_QPOS_WIDTH = 36
ROOT_HEIGHT_OFFSET = 0.793


def parse_fps(path: Path) -> int:
    match = FPS_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse FPS from file name: {path}")
    return int(match.group(1))


def convert_motion(src_path: Path, dst_path: Path, root_height_offset: float) -> None:
    qpos = np.load(src_path)
    if qpos.ndim != 2 or qpos.shape[1] != EXPECTED_QPOS_WIDTH:
        raise ValueError(
            f"Expected {src_path} to have shape [T, {EXPECTED_QPOS_WIDTH}], got {qpos.shape}"
        )
    if qpos.shape[0] == 0:
        raise ValueError(f"Motion contains no frames: {src_path}")

    root_pos = qpos[:, :3].copy()
    root_pos[:, 2] += root_height_offset

    motion_data = {
        "fps": parse_fps(src_path),
        "root_pos": root_pos,
        "root_rot": qpos[:, 3:7].copy(),
        "dof_pos": qpos[:, 7:].copy(),
        "local_body_pos": None,
        "source_file": str(src_path),
        "source_format": "retargeted_amass_g1_jpos_npy",
        "root_quat_order": "xyzw",
        "root_height_offset": root_height_offset,
    }

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("wb") as handle:
        pickle.dump(motion_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def iter_motion_files(src_dir: Path) -> list[Path]:
    return sorted(src_dir.rglob("*_jpos.npy"))


def make_dst_path(src_path: Path, src_dir: Path, dst_dir: Path) -> Path:
    relative_path = src_path.relative_to(src_dir)
    return dst_dir / relative_path.with_suffix(".pkl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, type=Path)
    parser.add_argument("--dst_dir", required=True, type=Path)
    parser.add_argument("--root_height_offset", type=float, default=ROOT_HEIGHT_OFFSET)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_dir = args.src_dir.expanduser().resolve()
    dst_dir = args.dst_dir.expanduser().resolve()

    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {src_dir}")

    motion_files = iter_motion_files(src_dir)
    if not motion_files:
        raise FileNotFoundError(f"No *_jpos.npy files found under: {src_dir}")

    converted = 0
    skipped = 0
    for src_path in motion_files:
        dst_path = make_dst_path(src_path, src_dir, dst_dir)
        if dst_path.exists() and not args.overwrite:
            skipped += 1
            continue
        convert_motion(src_path, dst_path, args.root_height_offset)
        converted += 1
        if converted % 500 == 0:
            print(f"converted={converted} skipped={skipped} latest={src_path.relative_to(src_dir)}")

    print(f"done converted={converted} skipped={skipped} total={len(motion_files)} dst={dst_dir}")


if __name__ == "__main__":
    main()
