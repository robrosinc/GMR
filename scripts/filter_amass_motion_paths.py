import argparse
import multiprocessing as mp
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_EXCLUDE_PATH_SUBSTRINGS: list[str] = ["sit", "crawl", "fall", "stair", "climb"]
COM_JOINT_NAME = "pelvis"
LEFT_FOOT_JOINT_CANDIDATES = ("left_foot", "left_ankle")
RIGHT_FOOT_JOINT_CANDIDATES = ("right_foot", "right_ankle")
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
DEFAULT_NUM_WORKERS = max(1, os.cpu_count() or 1)

_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_SMPLX_ANALYZER: "SmplxMotionAnalyzer | None" = None


def collect_motion_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".npz", ".pkl"})


def load_motion_data(file_path: Path) -> dict[str, Any]:
    suffix = file_path.suffix.lower()
    if suffix == ".npz":
        with np.load(file_path, allow_pickle=True) as npz:
            return {key: npz[key] for key in npz.files}
    if suffix == ".pkl":
        with file_path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, np.ndarray) and payload.shape == ():
            payload = payload.item()
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload in .pkl, got: {type(payload).__name__}")
        return dict(payload)
    raise ValueError(f"Unsupported file extension: {file_path.suffix}")


def _to_scalar(value: Any, default: Any) -> Any:
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    return default if value is None else value


def _ensure_2d_pose_array(poses_raw: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses_raw, dtype=np.float32)
    if poses.ndim == 3 and poses.shape[-1] == 3:
        return poses.reshape(poses.shape[0], -1)
    if poses.ndim == 2:
        return poses
    raise ValueError(f"Invalid poses shape: {poses.shape}")


def _extract_pose_blocks_from_npz(npz_data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "root_orient" in npz_data and "pose_body" in npz_data:
        root_orient = np.asarray(npz_data["root_orient"], dtype=np.float32)
        pose_body = np.asarray(npz_data["pose_body"], dtype=np.float32)
    elif "poses" in npz_data:
        poses = _ensure_2d_pose_array(npz_data["poses"])
        if poses.shape[1] < 66:
            raise ValueError(f"poses has too few dims: {poses.shape}")
        root_orient, pose_body = poses[:, :3], poses[:, 3:66]
    else:
        raise KeyError("Missing both (root_orient, pose_body) and poses.")

    pose_hand = np.asarray(npz_data.get("pose_hand", np.zeros((root_orient.shape[0], 90), dtype=np.float32)), dtype=np.float32)
    if pose_hand.ndim != 2 or pose_hand.shape[1] < 90:
        pose_hand = np.zeros((root_orient.shape[0], 90), dtype=np.float32)
    return root_orient, pose_body, pose_hand[:, :90]


def has_sustained_true(mask: np.ndarray, min_frames: int) -> bool:
    count = 0
    for value in mask:
        count = count + 1 if bool(value) else 0
        if count >= min_frames:
            return True
    return False


def point_to_segment_distance_xy(points: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    ab, ap = seg_b - seg_a, points - seg_a
    denom = np.sum(ab * ab, axis=1)
    valid = denom > 1e-8
    t = np.zeros(points.shape[0], dtype=np.float64)
    t[valid] = np.sum(ap[valid] * ab[valid], axis=1) / denom[valid]
    proj = seg_a + ab * np.clip(t, 0.0, 1.0)[:, None]
    dist = np.linalg.norm(points - proj, axis=1)
    if np.any(~valid):
        dist[~valid] = np.linalg.norm(points[~valid] - seg_a[~valid], axis=1)
    return dist


class SmplxMotionAnalyzer:
    def __init__(self, body_model_dir: Path, device: str) -> None:
        self.body_model_dir = body_model_dir
        self.requested_device = device
        self._runtime_ready = False
        self._torch = None
        self._smplx = None
        self._joint_names: list[str] = []
        self._name_to_idx: dict[str, int] = {}
        self._device = device
        self._models: dict[str, Any] = {}

    def _ensure_runtime(self) -> None:
        if self._runtime_ready:
            return
        try:
            import torch  # type: ignore
            import smplx  # type: ignore
            from smplx.joint_names import JOINT_NAMES  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Airborne/COM-footline filters require both `torch` and `smplx`. "
                "Install dependencies or disable these filters."
            ) from exc
        self._torch = torch
        self._smplx = smplx
        self._joint_names = list(JOINT_NAMES)
        self._name_to_idx = {name: idx for idx, name in enumerate(self._joint_names)}
        self._device = self.requested_device if not self.requested_device.startswith("cuda") or torch.cuda.is_available() else "cpu"
        self._runtime_ready = True

    def prepare(self) -> None:
        self._ensure_runtime()

    @staticmethod
    def _normalize_gender(value: Any) -> str:
        gender = str(_to_scalar(value, "neutral")).lower()
        return gender if gender in {"male", "female", "neutral"} else "neutral"

    def _get_model(self, gender: str):
        self._ensure_runtime()
        if gender not in self._models:
            self._models[gender] = self._smplx.create(
                str(self.body_model_dir),
                "smplx",
                gender=gender,
                use_pca=False,
            ).to(self._device)
        return self._models[gender]

    def _joint_index(self, candidates: tuple[str, ...], n_joints: int) -> int | None:
        for name in candidates:
            idx = self._name_to_idx.get(name)
            if idx is not None and idx < n_joints:
                return idx
        return None

    def compute_com_and_foot_points(self, npz_data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_runtime()
        torch = self._torch
        for key in ("trans", "betas"):
            if key not in npz_data:
                raise KeyError(f"Missing required key for SMPL-X analysis: {key}")

        trans = np.asarray(npz_data["trans"], dtype=np.float32)
        root_orient, pose_body, pose_hand = _extract_pose_blocks_from_npz(npz_data)
        betas = np.asarray(npz_data["betas"], dtype=np.float32).reshape(-1)
        if trans.ndim != 2 or trans.shape[1] < 3:
            raise ValueError(f"Invalid trans shape: {trans.shape}")
        if root_orient.ndim != 2 or root_orient.shape[1] < 3:
            raise ValueError(f"Invalid root_orient shape: {root_orient.shape}")
        if pose_body.ndim != 2 or pose_body.shape[1] < 63:
            raise ValueError(f"Invalid pose_body shape: {pose_body.shape}")

        num_frames = min(trans.shape[0], root_orient.shape[0], pose_body.shape[0], pose_hand.shape[0])
        if num_frames <= 0:
            raise ValueError("Empty sequence.")
        trans = trans[:num_frames, :3]
        root_orient = root_orient[:num_frames, :3]
        pose_body = pose_body[:num_frames, :63]
        pose_hand = pose_hand[:num_frames, :90]

        pose_jaw = np.asarray(npz_data.get("pose_jaw", np.zeros((num_frames, 3), dtype=np.float32)), dtype=np.float32)
        pose_eye = np.asarray(npz_data.get("pose_eye", np.zeros((num_frames, 6), dtype=np.float32)), dtype=np.float32)
        pose_jaw = pose_jaw[:num_frames, :3] if pose_jaw.ndim == 2 and pose_jaw.shape[0] >= num_frames and pose_jaw.shape[1] >= 3 else np.zeros((num_frames, 3), dtype=np.float32)
        pose_eye = pose_eye[:num_frames, :6] if pose_eye.ndim == 2 and pose_eye.shape[0] >= num_frames and pose_eye.shape[1] >= 6 else np.zeros((num_frames, 6), dtype=np.float32)

        model = self._get_model(self._normalize_gender(npz_data.get("gender", "neutral")))
        with torch.no_grad():
            output = model(
                betas=torch.tensor(betas, dtype=torch.float32, device=self._device).view(1, -1),
                global_orient=torch.tensor(root_orient, dtype=torch.float32, device=self._device),
                body_pose=torch.tensor(pose_body, dtype=torch.float32, device=self._device),
                transl=torch.tensor(trans, dtype=torch.float32, device=self._device),
                left_hand_pose=torch.tensor(pose_hand[:, :45], dtype=torch.float32, device=self._device),
                right_hand_pose=torch.tensor(pose_hand[:, 45:], dtype=torch.float32, device=self._device),
                jaw_pose=torch.tensor(pose_jaw, dtype=torch.float32, device=self._device),
                leye_pose=torch.tensor(pose_eye[:, :3], dtype=torch.float32, device=self._device),
                reye_pose=torch.tensor(pose_eye[:, 3:], dtype=torch.float32, device=self._device),
                return_full_pose=False,
            )

        joints = output.joints.detach().cpu().numpy()
        n_joints = joints.shape[1]
        com_idx = self._name_to_idx.get(COM_JOINT_NAME)
        com = joints[:, com_idx, :] if com_idx is not None and com_idx < n_joints else joints[:, 0, :]
        left_idx = self._joint_index(LEFT_FOOT_JOINT_CANDIDATES, n_joints)
        right_idx = self._joint_index(RIGHT_FOOT_JOINT_CANDIDATES, n_joints)
        if left_idx is None or right_idx is None:
            raise ValueError("Failed to find left/right foot joints in SMPL-X output.")
        return com, joints[:, left_idx, :], joints[:, right_idx, :]


def should_filter_airborne(left_foot: np.ndarray, right_foot: np.ndarray, foot_height_threshold: float, min_frames: int) -> bool:
    ground_z = float(np.min(np.concatenate([left_foot[:, 2], right_foot[:, 2]])))
    both_airborne = (left_foot[:, 2] - ground_z > foot_height_threshold) & (right_foot[:, 2] - ground_z > foot_height_threshold)
    return has_sustained_true(both_airborne, min_frames)


def should_filter_com_footline_distance(
    com: np.ndarray,
    left_foot: np.ndarray,
    right_foot: np.ndarray,
    distance_threshold: float,
    min_frames: int,
) -> bool:
    dist_xy = point_to_segment_distance_xy(points=com[:, :2], seg_a=left_foot[:, :2], seg_b=right_foot[:, :2])
    return has_sustained_true(dist_xy > distance_threshold, min_frames)


def _print_com_foot_xy_debug(
    file_path: Path,
    com: np.ndarray,
    left_foot: np.ndarray,
    right_foot: np.ndarray,
    distance_threshold: float,
) -> None:
    dist_xy = point_to_segment_distance_xy(points=com[:, :2], seg_a=left_foot[:, :2], seg_b=right_foot[:, :2])
    print(f"[debug_com_foot_xy] {file_path}")
    for i, d_line in enumerate(dist_xy):
        com_xy = com[i, :2]
        left_xy = left_foot[i, :2]
        right_xy = right_foot[i, :2]
        d_left = float(np.linalg.norm(com_xy - left_xy))
        d_right = float(np.linalg.norm(com_xy - right_xy))
        print(
            f"  frame={i} "
            f"com=[{float(com_xy[0]):.2f}, {float(com_xy[1]):.2f}] "
            f"left=[{float(left_xy[0]):.2f}, {float(left_xy[1]):.2f}] "
            f"right=[{float(right_xy[0]):.2f}, {float(right_xy[1]):.2f}] "
            f"d_left={d_left:.2f} d_right={d_right:.2f} d_line={float(d_line):.2f} "
            f"thr={float(distance_threshold):.2f}"
        )


def resolve_output_path(input_dir: Path, output_name: str) -> Path:
    output_path = Path(output_name).expanduser()
    return output_path if output_path.is_absolute() else input_dir / output_path


def to_output_path_str(base_dir: Path, file_path: Path, absolute_paths: bool) -> str:
    return str(file_path.resolve()) if absolute_paths else file_path.relative_to(base_dir).as_posix()


def _matched_exclude_token(input_dir: Path, file_path: Path, exclude_substrings: list[str]) -> str | None:
    path_text = file_path.relative_to(input_dir).as_posix().lower()
    for token in exclude_substrings:
        if token and token.lower() in path_text:
            return token
    return None


def evaluate_motion_filters(
    file_path: Path,
    cfg: dict[str, Any],
    smplx_analyzer: SmplxMotionAnalyzer | None,
) -> tuple[bool, str | None]:
    input_dir = Path(cfg["input_dir"])
    if cfg["enable_name_filter"]:
        matched = _matched_exclude_token(input_dir, file_path, cfg["exclude_substrings"])
        if matched is not None:
            return False, f"name_filter:{matched}"
    if not (cfg["enable_airborne_filter"] or cfg["enable_com_footline_filter"]):
        return True, None
    if smplx_analyzer is None:
        raise ValueError("SMPL-X analyzer is required for airborne/com-footline filters.")

    try:
        npz_data = load_motion_data(file_path)
        com, left_foot, right_foot = smplx_analyzer.compute_com_and_foot_points(npz_data)
    except Exception as exc:
        if cfg["analysis_error_policy"] == "drop":
            return False, f"analysis_error_drop:{exc.__class__.__name__}"
        if cfg["analysis_error_policy"] == "keep":
            return True, None
        raise

    if cfg.get("debug_com_foot_xy", False):
        _print_com_foot_xy_debug(
            file_path=file_path,
            com=com,
            left_foot=left_foot,
            right_foot=right_foot,
            distance_threshold=cfg["com_footline_distance_threshold"],
        )

    if cfg["enable_airborne_filter"] and should_filter_airborne(
        left_foot=left_foot,
        right_foot=right_foot,
        foot_height_threshold=cfg["airborne_height_threshold"],
        min_frames=cfg["airborne_min_frames"],
    ):
        return False, (
            f"airborne:height>{cfg['airborne_height_threshold']},"
            f"min_frames>={cfg['airborne_min_frames']}"
        )
    if cfg["enable_com_footline_filter"] and should_filter_com_footline_distance(
        com=com,
        left_foot=left_foot,
        right_foot=right_foot,
        distance_threshold=cfg["com_footline_distance_threshold"],
        min_frames=cfg["com_footline_min_frames"],
    ):
        return False, (
            f"com_footline:dist>{cfg['com_footline_distance_threshold']},"
            f"min_frames>={cfg['com_footline_min_frames']}"
        )
    return True, None


def _print_filtered_out(path: str, reason: str | None) -> None:
    print(f"[filtered_out] {path} | reason={reason or 'unknown'}")


def _format_filtered_out_entry(path: str, reason: str | None) -> str:
    return f"{path} | reason={reason or 'unknown'}"


def _iter_with_progress(iterable, total: int, desc: str, enable_progress: bool):
    if not enable_progress:
        yield from iterable
        return
    try:
        from tqdm import tqdm  # type: ignore
        yield from tqdm(iterable, total=total, desc=desc, unit="file")
    except ModuleNotFoundError:
        print_every = max(1, total // 100)
        for idx, item in enumerate(iterable, 1):
            if idx == 1 or idx % print_every == 0 or idx == total:
                print(f"[progress] {desc}: {idx}/{total}", end="\n" if idx == total else "\r", flush=True)
            yield item


def _set_thread_limits(num_threads: int, use_setdefault: bool = False) -> int:
    num_threads = max(1, int(num_threads))
    for var in THREAD_ENV_VARS:
        if use_setdefault:
            os.environ.setdefault(var, str(num_threads))
        else:
            os.environ[var] = str(num_threads)
    try:
        import torch  # type: ignore

        torch.set_num_threads(num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass
    return num_threads


def _apply_memory_limit_gb(max_memory_gb: float, context: str) -> None:
    if max_memory_gb <= 0:
        return
    try:
        import resource
    except Exception:
        print(f"[warn] Memory limit unsupported in this environment ({context}).")
        return

    limit_bytes = int(max_memory_gb * (1024**3))
    if limit_bytes <= 0:
        return
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        hard_is_unlimited = hard < 0 or hard >= (1 << 60)
        soft = limit_bytes if hard_is_unlimited else min(limit_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    except Exception as exc:
        print(f"[warn] Failed to apply memory limit in {context}: {exc}")


def _worker_memory_limit_gb(max_memory_gb: float, num_workers: int) -> float:
    return 0.0 if max_memory_gb <= 0 else max_memory_gb / max(1, int(num_workers))


def _init_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG, _WORKER_SMPLX_ANALYZER
    _WORKER_CONFIG, _WORKER_SMPLX_ANALYZER = config, None
    _set_thread_limits(config.get("worker_num_threads", 1))
    _apply_memory_limit_gb(float(config.get("max_memory_gb", 0.0)), "worker")
    if config["enable_airborne_filter"] or config["enable_com_footline_filter"]:
        _WORKER_SMPLX_ANALYZER = SmplxMotionAnalyzer(
            body_model_dir=Path(config["smplx_body_model_dir"]),
            device=config["smplx_device"],
        )
        _WORKER_SMPLX_ANALYZER.prepare()


def _process_single_file(file_path_str: str) -> tuple[str, bool, str | None]:
    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker configuration is not initialized.")
    cfg = _WORKER_CONFIG
    input_dir, file_path = Path(cfg["input_dir"]), Path(file_path_str)
    out_path_str = to_output_path_str(input_dir, file_path, cfg["absolute_paths"])
    passed, reason = evaluate_motion_filters(file_path, cfg, _WORKER_SMPLX_ANALYZER)
    return out_path_str, passed, reason


def filter_motion_files(
    input_dir: Path,
    absolute_paths: bool,
    filter_cfg: dict[str, Any],
    smplx_analyzer: SmplxMotionAnalyzer | None,
    smplx_body_model_dir: Path,
    smplx_device: str,
    num_workers: int,
    chunksize: int,
    show_progress: bool,
    mp_start_method: str,
    worker_num_threads: int,
    max_memory_gb: float,
) -> tuple[list[str], list[str], int]:
    motion_files = collect_motion_files(input_dir)
    kept_paths: list[str] = []
    filtered_out_paths: list[str] = []
    total = len(motion_files)
    if total == 0:
        return kept_paths, filtered_out_paths, 0

    num_workers, chunksize = max(1, int(num_workers)), max(1, int(chunksize))
    if num_workers == 1:
        for file_path in _iter_with_progress(motion_files, total, "Filtering motions", show_progress):
            out_path_str = to_output_path_str(input_dir, file_path, absolute_paths)
            passed, reason = evaluate_motion_filters(file_path, filter_cfg, smplx_analyzer)
            if passed:
                kept_paths.append(out_path_str)
            else:
                filtered_out_paths.append(_format_filtered_out_entry(out_path_str, reason))
                _print_filtered_out(out_path_str, reason)
        return kept_paths, filtered_out_paths, total

    worker_cfg = {
        **filter_cfg,
        "input_dir": str(input_dir),
        "absolute_paths": absolute_paths,
        "smplx_body_model_dir": str(smplx_body_model_dir),
        "smplx_device": smplx_device,
        "worker_num_threads": max(1, int(worker_num_threads)),
        "max_memory_gb": _worker_memory_limit_gb(max_memory_gb, num_workers),
    }
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp.get_context(mp_start_method),
        initializer=_init_worker,
        initargs=(worker_cfg,),
    ) as executor:
        results = executor.map(_process_single_file, (str(path) for path in motion_files), chunksize=chunksize)
        desc = f"Filtering motions ({num_workers} workers)"
        for out_path_str, passed, reason in _iter_with_progress(results, total, desc, show_progress):
            if passed:
                kept_paths.append(out_path_str)
            else:
                filtered_out_paths.append(_format_filtered_out_entry(out_path_str, reason))
                _print_filtered_out(out_path_str, reason)
    return kept_paths, filtered_out_paths, total


def _validate_mp_start_method(value: str) -> str:
    value = value.lower()
    if value not in set(mp.get_all_start_methods()):
        raise ValueError(f"Unsupported mp_start_method: {value}")
    return value


def _configure_main_threads(worker_num_threads: int) -> None:
    _set_thread_limits(worker_num_threads, use_setdefault=True)


def _should_init_main_analyzer(enable_airborne_filter: bool, enable_com_footline_filter: bool, num_workers: int) -> bool:
    return (enable_airborne_filter or enable_com_footline_filter) and max(1, int(num_workers)) == 1


def _print_runtime_summary(
    args,
    scanned_count: int,
    kept_count: int,
    filtered_count: int,
    output_path: Path,
    filtered_out_output_path: Path,
) -> None:
    print(f"Scanned {scanned_count} motion files (.npz/.pkl)")
    print(f"Name filter enabled: {args.enable_name_filter}")
    print(f"Airborne filter enabled: {args.enable_airborne_filter}")
    print(f"COM-footline filter enabled: {args.enable_com_footline_filter}")
    print(f"Analysis error policy: {args.analysis_error_policy}")
    print(f"Workers: {max(1, int(args.num_workers))}")
    print(f"MP start method: {args.mp_start_method}")
    print(f"Worker threads per process: {max(1, int(args.worker_num_threads))}")
    if args.max_memory_gb > 0:
        per_worker = _worker_memory_limit_gb(args.max_memory_gb, args.num_workers)
        print(f"Max memory limit: {args.max_memory_gb:.2f} GiB total ({per_worker:.2f} GiB per worker)")
    else:
        print("Max memory limit: disabled")
    print(f"Progress bar: {'disabled' if args.disable_progress else 'enabled (tqdm if installed)'}")
    print(f"Kept {kept_count} files")
    print(f"Filtered out {filtered_count} files")
    print(f"Saved -> {output_path.resolve()}")
    print(f"Saved filtered-out -> {filtered_out_output_path.resolve()}")


def save_paths_txt(paths: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if paths:
            handle.write("\n".join(paths) + "\n")


def _add_toggle_arg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    base_name = name.removeprefix("enable_")
    parser.add_argument(f"--enable_{base_name}", dest=name, action="store_true")
    parser.add_argument(f"--disable_{base_name}", dest=name, action="store_false")
    parser.set_defaults(**{name: default})


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter motion files recursively and save surviving paths to txt.")
    parser.add_argument("input_dir", type=Path, help="Directory to scan recursively for .npz/.pkl files.")
    parser.add_argument("--output_name", type=str, default="filtered_motion_paths.txt", help="Output txt filename or path.")
    parser.add_argument("--filtered_out_output_name", type=str, default="filtered_out_motion_paths.txt", help="Filtered-out txt filename or path.")
    parser.add_argument("--absolute_paths", action="store_true", help="Write absolute paths instead of paths relative to input_dir.")
    parser.add_argument("--exclude_substrings", type=str, nargs="*", default=None, help="Path substrings used by name filter.")

    _add_toggle_arg(parser, "enable_name_filter", True)
    _add_toggle_arg(parser, "enable_airborne_filter", False)
    _add_toggle_arg(parser, "enable_com_footline_filter", False)

    parser.add_argument("--airborne_height_threshold", type=float, default=0.15)
    parser.add_argument("--airborne_min_frames", type=int, default=20)
    parser.add_argument("--com_footline_distance_threshold", type=float, default=0.25)
    parser.add_argument("--com_footline_min_frames", type=int, default=20)
    parser.add_argument("--debug_com_foot_xy", action="store_true", help="Print simple COM/foot XY debug output.")
    parser.add_argument("--smplx_body_model_dir", type=Path, default=Path(__file__).resolve().parent.parent / "assets" / "body_models")
    parser.add_argument("--smplx_device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Use 1 to disable multiprocessing.")
    parser.add_argument("--worker_num_threads", type=int, default=1, help="Max CPU threads per worker process.")
    parser.add_argument(
        "--max_memory_gb",
        type=float,
        default=0.0,
        help="Max memory budget in GiB. 0 disables limit. In multiprocessing mode, split across workers.",
    )
    parser.add_argument("--mp_start_method", type=str, default="spawn", help="Multiprocessing start method.")
    parser.add_argument("--chunksize", type=int, default=8, help="Chunk size for worker map scheduling.")
    parser.add_argument("--disable_progress", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument(
        "--analysis_error_policy",
        type=str,
        choices=("drop", "keep", "error"),
        default="drop",
        help="When SMPL-X analysis fails: drop, keep, or error.",
    )

    args = parser.parse_args()
    args.mp_start_method = _validate_mp_start_method(args.mp_start_method)
    if args.max_memory_gb < 0:
        raise ValueError("--max_memory_gb must be >= 0")
    _configure_main_threads(args.worker_num_threads)
    if max(1, int(args.num_workers)) == 1:
        _apply_memory_limit_gb(args.max_memory_gb, "main")

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    filter_cfg: dict[str, Any] = {
        "input_dir": str(input_dir),
        "exclude_substrings": DEFAULT_EXCLUDE_PATH_SUBSTRINGS + (args.exclude_substrings or []),
        "enable_name_filter": args.enable_name_filter,
        "enable_airborne_filter": args.enable_airborne_filter,
        "airborne_height_threshold": args.airborne_height_threshold,
        "airborne_min_frames": args.airborne_min_frames,
        "enable_com_footline_filter": args.enable_com_footline_filter,
        "com_footline_distance_threshold": args.com_footline_distance_threshold,
        "com_footline_min_frames": args.com_footline_min_frames,
        "debug_com_foot_xy": args.debug_com_foot_xy,
        "analysis_error_policy": args.analysis_error_policy,
    }

    smplx_dir = args.smplx_body_model_dir.expanduser().resolve()
    smplx_analyzer: SmplxMotionAnalyzer | None = None
    if args.enable_airborne_filter or args.enable_com_footline_filter:
        if not smplx_dir.exists():
            raise FileNotFoundError(f"SMPL-X body model directory not found: {smplx_dir}")
        if _should_init_main_analyzer(args.enable_airborne_filter, args.enable_com_footline_filter, args.num_workers):
            smplx_analyzer = SmplxMotionAnalyzer(body_model_dir=smplx_dir, device=args.smplx_device)
            try:
                smplx_analyzer.prepare()
            except ModuleNotFoundError as exc:
                raise SystemExit(str(exc))

    output_path = resolve_output_path(input_dir, args.output_name)
    filtered_out_output_path = resolve_output_path(input_dir, args.filtered_out_output_name)
    kept_paths, filtered_out_paths, scanned_count = filter_motion_files(
        input_dir=input_dir,
        absolute_paths=args.absolute_paths,
        filter_cfg=filter_cfg,
        smplx_analyzer=smplx_analyzer,
        smplx_body_model_dir=smplx_dir,
        smplx_device=args.smplx_device,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
        show_progress=not args.disable_progress,
        mp_start_method=args.mp_start_method,
        worker_num_threads=args.worker_num_threads,
        max_memory_gb=args.max_memory_gb,
    )
    save_paths_txt(kept_paths, output_path)
    save_paths_txt(filtered_out_paths, filtered_out_output_path)
    _print_runtime_summary(
        args=args,
        scanned_count=scanned_count,
        kept_count=len(kept_paths),
        filtered_count=len(filtered_out_paths),
        output_path=output_path,
        filtered_out_output_path=filtered_out_output_path,
    )


if __name__ == "__main__":
    main()
