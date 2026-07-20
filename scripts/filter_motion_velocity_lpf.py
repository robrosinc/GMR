#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    import joblib
except ImportError:
    joblib = None


VELOCITY_KEYS = ("dof_vel", "root_vel", "root_angvel")


def safe_load_pickle(path: Path) -> Any:
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass

    with path.open("rb") as handle:
        class _NumpyCompatUnpickler(pickle.Unpickler):
            _module_map = {
                "numpy._core": "numpy.core",
                "numpy._core.multiarray": "numpy.core.multiarray",
                "numpy._core.numeric": "numpy.core.numeric",
                "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
            }

            def find_class(self, module, name):
                module = self._module_map.get(module, module)
                if module.startswith("numpy._core."):
                    module = "numpy.core." + module.split("numpy._core.", 1)[1]
                return super().find_class(module, name)

        return _NumpyCompatUnpickler(handle).load()


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def motion_fps(motion: dict, fallback_sample_rate: float | None) -> float:
    try:
        fps = float(motion.get("fps", 0.0))
    except (TypeError, ValueError):
        fps = 0.0
    if fps > 0.0:
        return fps
    if fallback_sample_rate is not None and fallback_sample_rate > 0.0:
        return float(fallback_sample_rate)
    raise ValueError("Motion pkl has no valid fps; provide --sample_rate.")


def lowpass_alpha(sample_rate: float, cutoff_hz: float) -> float:
    if sample_rate <= 0.0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if cutoff_hz <= 0.0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
    dt = 1.0 / sample_rate
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return float(dt / (tau + dt))


def first_order_lpf(values: np.ndarray, alpha: float) -> np.ndarray:
    filtered = np.empty_like(values, dtype=np.float64)
    filtered[0] = values[0]
    for idx in range(1, values.shape[0]):
        filtered[idx] = filtered[idx - 1] + alpha * (values[idx] - filtered[idx - 1])
    return filtered


def filter_velocity(values: np.ndarray, alpha: float, bidirectional: bool) -> np.ndarray:
    original_dtype = values.dtype
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim < 1 or arr.shape[0] == 0:
        return values
    filtered = first_order_lpf(arr, alpha)
    if bidirectional and filtered.shape[0] > 1:
        filtered = first_order_lpf(filtered[::-1], alpha)[::-1]
    if np.issubdtype(original_dtype, np.floating):
        return filtered.astype(original_dtype, copy=False)
    return filtered


def filter_motion(args: argparse.Namespace) -> tuple[dict, list[str]]:
    motion = safe_load_pickle(args.input_file)
    if not isinstance(motion, dict):
        raise ValueError(f"Expected dict motion pkl, got {type(motion).__name__}: {args.input_file}")

    sample_rate = motion_fps(motion, args.sample_rate)
    alpha = lowpass_alpha(sample_rate, args.cutoff_hz)
    filtered_keys: list[str] = []

    for key in VELOCITY_KEYS:
        if key not in motion:
            continue
        values = np.asarray(motion[key])
        if values.ndim < 1 or values.shape[0] == 0:
            continue
        motion[key] = filter_velocity(values, alpha, args.bidirectional)
        filtered_keys.append(key)

    motion["velocity_lpf_meta"] = {
        "cutoff_hz": float(args.cutoff_hz),
        "sample_rate": float(sample_rate),
        "alpha": float(alpha),
        "bidirectional": bool(args.bidirectional),
        "filtered_keys": filtered_keys,
    }
    return motion, filtered_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-pass filter velocity arrays in a GMR motion pkl.")
    parser.add_argument("--input_file", type=Path, required=True, help="Input motion pkl.")
    parser.add_argument("--output_file", type=Path, required=True, help="Output motion pkl.")
    parser.add_argument("--cutoff_hz", type=float, default=8.0, help="First-order LPF cutoff frequency.")
    parser.add_argument("--sample_rate", type=float, default=None, help="Fallback Hz when pkl has no fps key.")
    parser.add_argument("--bidirectional", action="store_true", help="Run LPF forward and backward to reduce lag.")
    parser.add_argument("--overwrite", action="store_true", help="Allow output_file to overwrite an existing file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_file.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists; pass --overwrite: {args.output_file}")
    motion, filtered_keys = filter_motion(args)
    if not filtered_keys:
        raise SystemExit("No velocity keys found to filter.")
    save_pickle(args.output_file, motion)
    print(f"Saved filtered motion: {args.output_file}")
    print(f"Filtered keys: {', '.join(filtered_keys)}")


if __name__ == "__main__":
    main()
