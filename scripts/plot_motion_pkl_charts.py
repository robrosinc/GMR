#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import joblib
except ImportError:
    joblib = None


ROOT_COMPONENTS = ("x", "y", "z")
QUAT_COMPONENTS = ("w", "x", "y", "z")


def safe_load_pickle(path: Path):
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


def load_motion(path: Path) -> dict:
    motion = safe_load_pickle(path)
    if not isinstance(motion, dict):
        raise ValueError(f"Expected dict motion pkl, got {type(motion).__name__}: {path}")
    return motion


def as_2d_array(motion: dict, key: str) -> np.ndarray | None:
    value = motion.get(key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    return arr


def as_3d_array(motion: dict, key: str) -> np.ndarray | None:
    value = motion.get(key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return None
    return arr


def motion_fps(motion: dict) -> float | None:
    try:
        fps = float(motion.get("fps", 0.0))
    except (TypeError, ValueError):
        return None
    return fps if fps > 0.0 else None


def x_axis(num_frames: int, fps: float | None, use_time_axis: bool) -> tuple[np.ndarray, str]:
    if use_time_axis and fps is not None:
        return np.arange(num_frames, dtype=np.float64) / fps, "time (s)"
    return np.arange(num_frames), "frame"


def import_pyplot():
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gmr")
    import matplotlib.pyplot as plt

    return plt


def names_from_motion(motion: dict, count: int, key: str, prefix: str) -> list[str]:
    names = motion.get(key)
    if isinstance(names, (list, tuple)) and names:
        out = [str(name) for name in names[:count]]
        if len(out) < count:
            out.extend(f"{prefix}_{idx:02d}" for idx in range(len(out), count))
        return out
    return [f"{prefix}_{idx:02d}" for idx in range(count)]


def save_line_group(
    plt,
    path: Path,
    title: str,
    x: np.ndarray,
    xlabel: str,
    series: Iterable[tuple[str, np.ndarray]],
    ylabel: str = "value",
    figsize: tuple[float, float] = (11.0, 6.0),
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for label, values in series:
        ax.plot(x, values, linewidth=1.0, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_component_grid(
    plt,
    path: Path,
    title: str,
    x: np.ndarray,
    xlabel: str,
    values: np.ndarray,
    names: list[str],
    columns: int,
    dpi: int,
) -> None:
    rows = int(np.ceil(values.shape[1] / columns))
    fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(5.5 * columns, 3.375 * rows), dpi=dpi)
    axes_flat = axes.reshape(-1)
    for idx, ax in enumerate(axes_flat):
        if idx >= values.shape[1]:
            ax.set_visible(False)
            continue
        ax.plot(x, values[:, idx], linewidth=0.9)
        ax.set_title(names[idx], fontsize="small")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_body_position_grid(
    plt,
    path: Path,
    title: str,
    x: np.ndarray,
    xlabel: str,
    body_pos: np.ndarray,
    body_names: list[str],
    columns: int,
    dpi: int,
) -> None:
    rows = int(np.ceil(body_pos.shape[1] / columns))
    fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(5.5 * columns, 3.75 * rows), dpi=dpi)
    axes_flat = axes.reshape(-1)
    for body_idx, ax in enumerate(axes_flat):
        if body_idx >= body_pos.shape[1]:
            ax.set_visible(False)
            continue
        for component_idx, component in enumerate(ROOT_COMPONENTS):
            ax.plot(x, body_pos[:, body_idx, component_idx], linewidth=0.9, label=component)
        ax.set_title(body_names[body_idx], fontsize="small")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize="x-small")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_charts(args: argparse.Namespace) -> list[Path]:
    motion = load_motion(args.motion_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt = import_pyplot()
    fps = motion_fps(motion)
    saved: list[Path] = []

    dof_pos = as_2d_array(motion, "dof_pos")
    if dof_pos is not None:
        x, xlabel = x_axis(dof_pos.shape[0], fps, args.time_axis)
        dof_names = names_from_motion(motion, dof_pos.shape[1], "dof_names", "dof")
        path = args.output_dir / "ref_joint_pos.png"
        save_component_grid(plt, path, "ref joint pos", x, xlabel, dof_pos, dof_names, args.joint_columns, args.dpi)
        saved.append(path)

    dof_vel = as_2d_array(motion, "dof_vel")
    if dof_vel is not None:
        x, xlabel = x_axis(dof_vel.shape[0], fps, args.time_axis)
        dof_names = names_from_motion(motion, dof_vel.shape[1], "dof_names", "dof")
        path = args.output_dir / "ref_joint_vel.png"
        save_component_grid(plt, path, "ref joint vel", x, xlabel, dof_vel, dof_names, args.joint_columns, args.dpi)
        saved.append(path)

    root_pos = as_2d_array(motion, "root_pos")
    if root_pos is not None and root_pos.shape[1] >= 3:
        x, xlabel = x_axis(root_pos.shape[0], fps, args.time_axis)
        path = args.output_dir / "ref_root_position_z.png"
        save_line_group(
            plt,
            path,
            "ref root position z",
            x,
            xlabel,
            [("z", root_pos[:, 2])],
            ylabel="z",
            figsize=(5.5, 3.0),
            dpi=args.dpi,
        )
        saved.append(path)

    root_vel = as_2d_array(motion, "root_vel")
    if root_vel is not None and root_vel.shape[1] >= 3:
        x, xlabel = x_axis(root_vel.shape[0], fps, args.time_axis)
        path = args.output_dir / "ref_root_linear_velocity.png"
        save_line_group(
            plt,
            path,
            "ref root linear velocity",
            x,
            xlabel,
            [(component, root_vel[:, idx]) for idx, component in enumerate(ROOT_COMPONENTS)],
            dpi=args.dpi,
        )
        saved.append(path)

    root_angvel = as_2d_array(motion, "root_angvel")
    if root_angvel is not None and root_angvel.shape[1] >= 3:
        x, xlabel = x_axis(root_angvel.shape[0], fps, args.time_axis)
        path = args.output_dir / "ref_root_angular_velocity.png"
        save_line_group(
            plt,
            path,
            "ref root angular velocity",
            x,
            xlabel,
            [(component, root_angvel[:, idx]) for idx, component in enumerate(ROOT_COMPONENTS)],
            dpi=args.dpi,
        )
        saved.append(path)

    root_rot = as_2d_array(motion, "root_rot")
    if root_pos is not None or root_rot is not None:
        num_frames = (root_pos if root_pos is not None else root_rot).shape[0]
        x, xlabel = x_axis(num_frames, fps, args.time_axis)
        root_state_path = args.output_dir / "ref_root_state.png"
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), dpi=args.dpi)
        if root_pos is not None:
            for idx, component in enumerate(ROOT_COMPONENTS[: root_pos.shape[1]]):
                axes[0].plot(x, root_pos[:, idx], linewidth=1.0, label=component)
            axes[0].legend(loc="best", fontsize="small")
        axes[0].set_title("root_pos")
        if root_rot is not None:
            components = QUAT_COMPONENTS[: root_rot.shape[1]]
            for idx, component in enumerate(components):
                axes[1].plot(x, root_rot[:, idx], linewidth=1.0, label=component)
            axes[1].legend(loc="best", fontsize="small")
        axes[1].set_title("root_rot")
        if root_vel is not None:
            for idx, component in enumerate(ROOT_COMPONENTS[: root_vel.shape[1]]):
                axes[2].plot(x, root_vel[:, idx], linewidth=1.0, label=component)
            axes[2].legend(loc="best", fontsize="small")
        axes[2].set_title("root_vel")
        for ax in axes:
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.25)
        fig.suptitle("ref root state")
        fig.tight_layout()
        fig.savefig(root_state_path)
        plt.close(fig)
        saved.append(root_state_path)

    body_pos = as_3d_array(motion, args.body_position_key)
    if body_pos is not None and body_pos.shape[2] >= 3:
        x, xlabel = x_axis(body_pos.shape[0], fps, args.time_axis)
        body_names = names_from_motion(motion, body_pos.shape[1], "link_body_list", "body")
        path = args.output_dir / "ref_body_position.png"
        save_body_position_grid(
            plt,
            path,
            f"ref body position ({args.body_position_key})",
            x,
            xlabel,
            body_pos[:, :, :3],
            body_names,
            args.body_columns,
            args.dpi,
        )
        saved.append(path)

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save chart PNGs from one GMR motion pkl.")
    parser.add_argument("--motion_file", type=Path, required=True, help="Input GMR motion pkl.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for output PNG charts.")
    parser.add_argument("--body_position_key", default="keybody_pos_world", help="3D body position key to plot.")
    parser.add_argument("--joint_columns", type=int, default=4, help="Columns for joint position/velocity charts.")
    parser.add_argument("--body_columns", type=int, default=4, help="Columns for body position chart.")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI.")
    parser.add_argument("--time_axis", action="store_true", help="Use seconds on x-axis instead of frame index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = write_charts(args)
    if not saved:
        raise SystemExit("No supported motion arrays found; no charts were written.")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
