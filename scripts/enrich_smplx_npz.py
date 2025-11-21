#!/usr/bin/env python3
"""
Utility to augment a minimal SMPL-X .npz (e.g. generated from BVH) with all of the
fields expected by downstream consumers. The script copies static metadata such as
marker layouts from a reference SMPL-X sample (npz or pkl) and derives the standard
pose partitions (root, body, hands, jaw, eyes) from the provided poses array.
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


def _load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz_file:
        return {key: npz_file[key] for key in npz_file.files}


def _load_reference(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz(path)
    if suffix == ".pkl":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if not isinstance(data, Mapping):
            raise TypeError(f"Reference pickle must contain a mapping, got {type(data)}")
        return dict(data)
    raise ValueError(f"Unsupported reference file format: {path}")


def _ensure_pose_tensor(raw_pose: np.ndarray) -> np.ndarray:
    pose_array = np.asarray(raw_pose)
    if pose_array.ndim == 3 and pose_array.shape[-1] == 3:
        return pose_array
    if pose_array.ndim == 2 and pose_array.shape[1] % 3 == 0:
        return pose_array.reshape(pose_array.shape[0], -1, 3)
    raise ValueError(
        "Expected poses shaped as (frames, 55, 3) or (frames, 165) but "
        f"received array with shape {pose_array.shape}"
    )


def _extract_scalar(value: Any, *, dtype=None, default=None):
    if value is None:
        value = default
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.flatten()[0]
    if value is None:
        return None if dtype is None else np.array(None, dtype=dtype)
    return np.array(value, dtype=dtype) if dtype is not None else value


def _copy_optional(ref: Mapping[str, Any], key: str, fallback: Any) -> Any:
    if key not in ref:
        return fallback
    value = ref[key]
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def enrich_smplx_npz(
    src_path: Path, reference_path: Path | None, output_path: Path
) -> None:
    src = _load_npz(src_path)
    reference = _load_reference(reference_path)

    if "poses" not in src:
        raise KeyError(f"{src_path} must contain a 'poses' array.")
    if "trans" not in src:
        raise KeyError(f"{src_path} must contain a 'trans' array.")
    if "betas" not in src:
        raise KeyError(f"{src_path} must contain a 'betas' array.")

    pose_tensor = _ensure_pose_tensor(src["poses"]).astype(np.float32)
    n_frames = pose_tensor.shape[0]

    gender = _extract_scalar(
        src.get("gender", reference.get("gender", "neutral")), dtype=None
    )
    gender = str(gender) if not isinstance(gender, str) else gender

    fps_value = src.get("mocap_frame_rate", reference.get("mocap_frame_rate", 30.0))
    mocap_frame_rate = float(_extract_scalar(fps_value, dtype=float))
    mocap_time_length = np.array(n_frames / mocap_frame_rate, dtype=np.float32)

    full_pose = pose_tensor.reshape(n_frames, -1, 3)
    if full_pose.shape[1] < 55:
        raise ValueError(
            "The poses tensor does not contain enough joints to synthesize SMPL-X "
            f"poses. Expected 55 joints, got {full_pose.shape[1]}"
        )

    root_orient = full_pose[:, 0, :].astype(np.float32)
    pose_body = full_pose[:, 1:22, :].reshape(n_frames, -1).astype(np.float32)
    extra = full_pose[:, 22:, :]
    if extra.shape[1] < 3:
        raise ValueError(
            "Poses array does not contain jaw/eye entries. "
            f"Expected at least three extra joints, received {extra.shape[1]}"
        )
    pose_jaw = extra[:, 0, :].astype(np.float32)
    pose_eye = extra[:, 1:3, :].reshape(n_frames, 6).astype(np.float32)
    pose_hand = extra[:, 3:, :].reshape(n_frames, -1).astype(np.float32)

    poses_flat = full_pose.reshape(n_frames, -1).astype(np.float32)
    trans = np.asarray(src["trans"]).astype(np.float32)
    betas = np.asarray(src["betas"]).astype(np.float32).reshape(-1)
    num_betas = np.array(betas.shape[0], dtype=np.int64)

    enriched_payload: Dict[str, Any] = {
        "gender": np.array(gender),
        "surface_model_type": np.array(
            _copy_optional(reference, "surface_model_type", "smplx")
        ),
        "mocap_frame_rate": np.array(mocap_frame_rate, dtype=np.float32),
        "mocap_time_length": mocap_time_length,
        "markers_latent": _copy_optional(
            reference, "markers_latent", np.zeros((0, 3), dtype=np.float32)
        ),
        "latent_labels": _copy_optional(
            reference, "latent_labels", np.array([], dtype="<U1")
        ),
        "markers_latent_vids": _copy_optional(
            reference, "markers_latent_vids", np.array({}, dtype=object)
        ),
        "trans": trans,
        "poses": poses_flat,
        "betas": betas,
        "num_betas": num_betas,
        "root_orient": root_orient,
        "pose_body": pose_body,
        "pose_hand": pose_hand,
        "pose_jaw": pose_jaw,
        "pose_eye": pose_eye,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **enriched_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill in missing SMPL-X fields for an npz produced by lafan_to_smplx.py. "
            "Static metadata such as markers are copied from the provided reference."
        )
    )
    parser.add_argument(
        "--input_npz",
        required=True,
        type=Path,
        help="Path to the minimal SMPL-X npz (e.g. output/aiming1_subject1.npz).",
    )
    parser.add_argument(
        "--reference",
        required=False,
        type=Path,
        help="Reference SMPL-X sample (npz or pkl) with the desired metadata fields.",
    )
    parser.add_argument(
        "--output_npz",
        required=True,
        type=Path,
        help="Destination path for the enriched SMPL-X npz.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    enrich_smplx_npz(cli_args.input_npz, cli_args.reference, cli_args.output_npz)
