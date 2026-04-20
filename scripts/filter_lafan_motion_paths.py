import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

LEFT_FOOT_JOINT_DEFAULT = "LeftFoot"
RIGHT_FOOT_JOINT_DEFAULT = "RightFoot"

_LAFAN_EXTRACT = None
_LAFAN_UTILS = None


def _load_lafan_vendor_modules():
    global _LAFAN_EXTRACT, _LAFAN_UTILS
    if _LAFAN_EXTRACT is not None and _LAFAN_UTILS is not None:
        return _LAFAN_EXTRACT, _LAFAN_UTILS

    vendor_dir = Path(__file__).resolve().parent.parent / "general_motion_retargeting" / "utils" / "lafan_vendor"
    if not vendor_dir.is_dir():
        raise FileNotFoundError(f"lafan_vendor directory not found: {vendor_dir}")

    pkg_name = "_lafan_vendor_local"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(vendor_dir)]
    sys.modules[pkg_name] = pkg

    utils_spec = importlib.util.spec_from_file_location(f"{pkg_name}.utils", vendor_dir / "utils.py")
    if utils_spec is None or utils_spec.loader is None:
        raise ImportError("Failed to load lafan_vendor utils.py")
    utils_mod = importlib.util.module_from_spec(utils_spec)
    sys.modules[f"{pkg_name}.utils"] = utils_mod
    utils_spec.loader.exec_module(utils_mod)

    extract_spec = importlib.util.spec_from_file_location(f"{pkg_name}.extract", vendor_dir / "extract.py")
    if extract_spec is None or extract_spec.loader is None:
        raise ImportError("Failed to load lafan_vendor extract.py")
    extract_mod = importlib.util.module_from_spec(extract_spec)
    sys.modules[f"{pkg_name}.extract"] = extract_mod
    extract_spec.loader.exec_module(extract_mod)

    _LAFAN_EXTRACT, _LAFAN_UTILS = extract_mod, utils_mod
    return _LAFAN_EXTRACT, _LAFAN_UTILS


def collect_bvh_files(input_path: Path) -> tuple[list[Path], Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".bvh":
            raise ValueError(f"Input file must be .bvh: {input_path}")
        return [input_path], input_path.parent
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.bvh") if path.is_file()), input_path
    raise FileNotFoundError(f"Input path not found: {input_path}")


def resolve_output_path(base_dir: Path, output_name: str) -> Path:
    output_path = Path(output_name).expanduser()
    return output_path if output_path.is_absolute() else base_dir / output_path


def to_output_path_str(base_dir: Path, file_path: Path, absolute_paths: bool) -> str:
    return str(file_path.resolve()) if absolute_paths else file_path.relative_to(base_dir).as_posix()


def has_sustained_true(mask: np.ndarray, min_frames: int) -> bool:
    count = 0
    for value in mask:
        count = count + 1 if bool(value) else 0
        if count >= min_frames:
            return True
    return False


def load_lafan_foot_positions(file_path: Path, left_foot_joint: str, right_foot_joint: str) -> tuple[np.ndarray, np.ndarray]:
    extract_mod, utils_mod = _load_lafan_vendor_modules()
    anim = extract_mod.read_bvh(str(file_path))
    _, global_pos = utils_mod.quat_fk(anim.quats, anim.pos, anim.parents)
    name_to_idx = {name: idx for idx, name in enumerate(anim.bones)}

    if left_foot_joint not in name_to_idx:
        raise KeyError(f"Missing left foot joint: {left_foot_joint}")
    if right_foot_joint not in name_to_idx:
        raise KeyError(f"Missing right foot joint: {right_foot_joint}")

    left_idx = name_to_idx[left_foot_joint]
    right_idx = name_to_idx[right_foot_joint]
    # BVH positions are in centimeters and Y-up.
    left_foot = np.asarray(global_pos[:, left_idx, :], dtype=np.float32) / 100.0
    right_foot = np.asarray(global_pos[:, right_idx, :], dtype=np.float32) / 100.0
    return left_foot, right_foot


def should_filter_airborne(
    left_foot: np.ndarray,
    right_foot: np.ndarray,
    foot_height_threshold: float,
    min_frames: int,
) -> bool:
    ground_y = float(np.min(np.concatenate([left_foot[:, 1], right_foot[:, 1]])))
    both_airborne = (left_foot[:, 1] - ground_y > foot_height_threshold) & (right_foot[:, 1] - ground_y > foot_height_threshold)
    return has_sustained_true(both_airborne, min_frames)


def evaluate_file(file_path: Path, cfg: dict) -> tuple[bool, str | None]:
    try:
        left_foot, right_foot = load_lafan_foot_positions(
            file_path=file_path,
            left_foot_joint=cfg["left_foot_joint"],
            right_foot_joint=cfg["right_foot_joint"],
        )
    except Exception as exc:
        if cfg["format_error_policy"] == "drop":
            return False, f"format_error:{exc.__class__.__name__}"
        if cfg["format_error_policy"] == "keep":
            return True, None
        raise

    if should_filter_airborne(
        left_foot=left_foot,
        right_foot=right_foot,
        foot_height_threshold=cfg["airborne_height_threshold"],
        min_frames=cfg["airborne_min_frames"],
    ):
        return False, (
            f"airborne:height>{cfg['airborne_height_threshold']},"
            f"min_frames>={cfg['airborne_min_frames']}"
        )
    return True, None


def _iter_with_progress(iterable, total: int, desc: str, enable_progress: bool):
    if not enable_progress:
        yield from iterable
        return
    try:
        from tqdm import tqdm  # type: ignore

        yield from tqdm(iterable, total=total, desc=desc, unit="file")
    except ModuleNotFoundError:
        for idx, item in enumerate(iterable, 1):
            if idx == 1 or idx == total or idx % max(1, total // 100) == 0:
                print(f"[progress] {desc}: {idx}/{total}", end="\n" if idx == total else "\r", flush=True)
            yield item


def _format_filtered_out_entry(path: str, reason: str | None) -> str:
    return f"{path} | reason={reason or 'unknown'}"


def save_paths_txt(paths: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if paths:
            handle.write("\n".join(paths) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter LAFAN BVH motions using airborne and format-error checks."
    )
    parser.add_argument("input_path", type=Path, help="Input .bvh file or directory (recursive scan).")
    parser.add_argument("--output_name", type=str, default="filtered_motion_paths.txt")
    parser.add_argument("--filtered_out_output_name", type=str, default="filtered_out_motion_paths.txt")
    parser.add_argument("--absolute_paths", action="store_true", help="Write absolute paths.")
    parser.add_argument("--airborne_height_threshold", type=float, default=0.20)
    parser.add_argument("--airborne_min_frames", type=int, default=24)
    parser.add_argument("--left_foot_joint", type=str, default=LEFT_FOOT_JOINT_DEFAULT)
    parser.add_argument("--right_foot_joint", type=str, default=RIGHT_FOOT_JOINT_DEFAULT)
    parser.add_argument(
        "--format_error_policy",
        type=str,
        choices=("drop", "keep", "error"),
        default="drop",
        help="When BVH format parsing fails: drop, keep, or error.",
    )
    parser.add_argument("--disable_progress", action="store_true")
    args = parser.parse_args()

    files, base_dir = collect_bvh_files(args.input_path)
    if not files:
        print("Scanned 0 .bvh files")
        return

    cfg = {
        "airborne_height_threshold": args.airborne_height_threshold,
        "airborne_min_frames": args.airborne_min_frames,
        "left_foot_joint": args.left_foot_joint,
        "right_foot_joint": args.right_foot_joint,
        "format_error_policy": args.format_error_policy,
    }

    kept_paths: list[str] = []
    filtered_out_paths: list[str] = []
    total = len(files)

    for file_path in _iter_with_progress(files, total, "Filtering LAFAN BVH", not args.disable_progress):
        out_path = to_output_path_str(base_dir, file_path, args.absolute_paths)
        passed, reason = evaluate_file(file_path, cfg)
        if passed:
            kept_paths.append(out_path)
        else:
            filtered_out_paths.append(_format_filtered_out_entry(out_path, reason))
            print(f"[filtered_out] {out_path} | reason={reason or 'unknown'}")

    output_path = resolve_output_path(base_dir, args.output_name)
    filtered_out_output_path = resolve_output_path(base_dir, args.filtered_out_output_name)
    save_paths_txt(kept_paths, output_path)
    save_paths_txt(filtered_out_paths, filtered_out_output_path)

    print(f"Scanned {total} .bvh files")
    print("Applied filters: airborne + format_error")
    print(f"Airborne threshold: {args.airborne_height_threshold}")
    print(f"Airborne min frames: {args.airborne_min_frames}")
    print(f"Format error policy: {args.format_error_policy}")
    print(f"Kept {len(kept_paths)} files")
    print(f"Filtered out {len(filtered_out_paths)} files")
    print(f"Saved -> {output_path.resolve()}")
    print(f"Saved filtered-out -> {filtered_out_output_path.resolve()}")


if __name__ == "__main__":
    main()
