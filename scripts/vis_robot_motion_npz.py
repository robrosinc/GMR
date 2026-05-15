import argparse
from pathlib import Path

import numpy as np
from rich import print
from loop_rate_limiters import RateLimiter

from general_motion_retargeting import RobotMotionViewer
from vis_controls import (
    create_control_state,
    log_controls_once,
    make_keyboard_callback,
    viewer_alive,
)


def load_curation_list(curation_path: Path) -> list[str]:
    if not curation_path.exists():
        return []
    seen = set()
    items = []
    for line in curation_path.read_text(encoding="utf-8").splitlines():
        clip_name = line.strip()
        if not clip_name or clip_name in seen:
            continue
        seen.add(clip_name)
        items.append(clip_name)
    return items


def save_curation_list(curation_path: Path, clip_names: list[str]) -> None:
    curation_path.parent.mkdir(parents=True, exist_ok=True)
    with curation_path.open("w", encoding="utf-8") as f:
        for clip_name in clip_names:
            f.write(f"{clip_name}\n")


def load_motion_dataset(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    required_keys = [
        "fps",
        "body_names",
        "local_frame_body_name",
        "body_pos_w",
        "body_quat_w",
        "joint_pos",
        "joint_names",
        "clip_offsets",
        "clip_names",
    ]
    missing = [key for key in required_keys if key not in data]
    if missing:
        data.close()
        raise KeyError(f"Missing required keys in {npz_path}: {missing}")

    fps = float(data["fps"])
    body_names = data["body_names"]
    local_frame_body_name = str(data["local_frame_body_name"])
    root_body_indices = np.where(body_names == local_frame_body_name)[0]
    if len(root_body_indices) == 0:
        data.close()
        raise ValueError(
            f"local_frame_body_name '{local_frame_body_name}' not found in body_names."
        )
    root_body_idx = int(root_body_indices[0])

    body_pos_w = data["body_pos_w"]
    body_quat_w = data["body_quat_w"]
    joint_pos = data["joint_pos"]
    joint_names = data["joint_names"]
    clip_offsets = data["clip_offsets"].astype(np.int64, copy=False)
    clip_names = data["clip_names"]

    num_frames = int(joint_pos.shape[0])
    if len(clip_offsets) != len(clip_names) + 1:
        data.close()
        raise ValueError(
            "Invalid clip metadata: len(clip_offsets) must be len(clip_names) + 1."
        )
    if int(clip_offsets[0]) != 0 or int(clip_offsets[-1]) != num_frames:
        data.close()
        raise ValueError(
            "Invalid clip metadata: clip_offsets must start at 0 and end at total frames."
        )

    return {
        "data": data,
        "fps": fps,
        "root_body_idx": root_body_idx,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "joint_pos": joint_pos,
        "joint_names": joint_names,
        "clip_offsets": clip_offsets,
        "clip_names": clip_names,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="robros_igris_c_v2")
    parser.add_argument("--motion_npz_path", type=str, required=True)
    parser.add_argument("--no_rate_limit", action="store_true")
    parser.add_argument(
        "--root_quat_scalar_first",
        type=str,
        choices=("true", "false"),
        default="true",
        help="Whether input root quaternion is scalar-first (wxyz). true/false",
    )
    args = parser.parse_args()
    root_quat_scalar_first = args.root_quat_scalar_first == "true"

    motion_npz_path = Path(args.motion_npz_path)
    if not motion_npz_path.exists() or not motion_npz_path.is_file():
        raise FileNotFoundError(f"Motion npz file not found: {motion_npz_path}")

    dataset = load_motion_dataset(motion_npz_path)
    fps = dataset["fps"]
    root_body_idx = dataset["root_body_idx"]
    body_pos_w = dataset["body_pos_w"]
    body_quat_w = dataset["body_quat_w"]
    joint_pos = dataset["joint_pos"]
    joint_names = [str(name) for name in dataset["joint_names"].tolist()]
    clip_offsets = dataset["clip_offsets"]
    clip_names = dataset["clip_names"]
    n_clips = len(clip_names)

    print(f"Loaded {motion_npz_path}")
    print(f"Frames: {joint_pos.shape[0]}, clips: {n_clips}, fps: {fps}")
    log_controls_once(include_curation=True, log_fn=print)
    print(f"Root quat scalar-first: {root_quat_scalar_first}")
    print("Close the MuJoCo window to exit.")
    curation_path = Path(__file__).resolve().parents[1] / "curation.txt"
    if not curation_path.exists():
        curation_path.write_text("", encoding="utf-8")
    curated_clips = load_curation_list(curation_path)
    curated_set = set(curated_clips)
    print(f"Curation file: {curation_path} (loaded {len(curated_clips)} clips)")

    viewer = None
    control_state = create_control_state(enable_curation=True)
    clip_index = 0
    active_clip_index = -1
    frame_idx = 0
    clip_start = 0
    clip_len = 0
    dof_buffer = None
    motion_to_dof_indices = None

    try:
        while True:
            if clip_index != active_clip_index:
                active_clip_index = clip_index
                clip_start = int(clip_offsets[active_clip_index])
                clip_end = int(clip_offsets[active_clip_index + 1])
                clip_len = clip_end - clip_start
                if clip_len <= 0:
                    print(
                        f"[yellow]Skipping empty clip {active_clip_index}: "
                        f"{clip_names[active_clip_index]}[/yellow]"
                    )
                    clip_index = (clip_index + 1) % n_clips
                    active_clip_index = -1
                    continue
                frame_idx = 0
                print(
                    f"[{active_clip_index + 1}/{n_clips}] {clip_names[active_clip_index]} "
                    f"(frames: {clip_len})"
                )

            if viewer is None:
                viewer = RobotMotionViewer(
                    robot_type=args.robot,
                    motion_fps=fps * control_state["speed"],
                    root_quat_scalar_first=root_quat_scalar_first,
                    camera_follow=False,
                    key_callback=make_keyboard_callback(
                        control_state, enable_curation=True, log_fn=print
                    ),
                )
                expected_dof = viewer.model.nq - 7
                model_dof_name_to_idx = {}
                for joint_idx in range(viewer.model.njnt):
                    qpos_adr = int(viewer.model.jnt_qposadr[joint_idx])
                    if qpos_adr < 7:
                        continue
                    model_dof_name_to_idx[viewer.model.joint(joint_idx).name] = qpos_adr - 7

                motion_to_dof_indices = np.array(
                    [model_dof_name_to_idx[name] for name in joint_names],
                    dtype=np.int64,
                )
                if joint_pos.shape[1] != len(motion_to_dof_indices):
                    raise ValueError(
                        f"joint_pos dim ({joint_pos.shape[1]}) and joint_names "
                        f"len ({len(motion_to_dof_indices)}) mismatch."
                    )
                dof_buffer = np.zeros(expected_dof, dtype=joint_pos.dtype)

                zero_fill_count = expected_dof - len(motion_to_dof_indices)
                print(f"[yellow]Zero-filled joints: {zero_fill_count}[/yellow]")
            else:
                if control_state["speed_dirty"]:
                    viewer.motion_fps = fps * control_state["speed"]
                    viewer.rate_limiter = RateLimiter(
                        frequency=viewer.motion_fps, warn=False
                    )
                    control_state["speed_dirty"] = False

            if not viewer_alive(viewer):
                break

            frame_abs = clip_start + frame_idx
            dof_buffer.fill(0)
            dof_buffer[motion_to_dof_indices] = joint_pos[frame_abs]
            viewer.step(
                body_pos_w[frame_abs, root_body_idx],
                body_quat_w[frame_abs, root_body_idx],
                dof_buffer,
                rate_limit=not args.no_rate_limit,
            )

            if control_state["curation_action"] is not None:
                current_clip_name = str(clip_names[active_clip_index])
                if control_state["curation_action"] == "add":
                    if current_clip_name in curated_set:
                        print(
                            f"[yellow]Already curated: {current_clip_name} "
                            f"({len(curated_clips)})[/yellow]"
                        )
                    else:
                        curated_clips.append(current_clip_name)
                        curated_set.add(current_clip_name)
                        save_curation_list(curation_path, curated_clips)
                        print(
                            f"[green]Curated (+): {current_clip_name} "
                            f"({len(curated_clips)})[/green]"
                        )
                elif control_state["curation_action"] == "remove":
                    if current_clip_name in curated_set:
                        curated_set.remove(current_clip_name)
                        curated_clips = [
                            name for name in curated_clips if name != current_clip_name
                        ]
                        save_curation_list(curation_path, curated_clips)
                        print(
                            f"[green]Curated (-): {current_clip_name} "
                            f"({len(curated_clips)})[/green]"
                        )
                    else:
                        print(
                            f"[yellow]Not in curation: {current_clip_name} "
                            f"({len(curated_clips)})[/yellow]"
                        )
                control_state["curation_action"] = None

            if control_state["clip_delta"] != 0:
                clip_index = (clip_index + control_state["clip_delta"]) % n_clips
                control_state["clip_delta"] = 0
                continue

            if control_state["paused"]:
                if control_state["frame_step"] != 0:
                    frame_idx = (frame_idx + control_state["frame_step"]) % clip_len
                    control_state["frame_step"] = 0
                continue

            frame_idx += 1
            if frame_idx >= clip_len:
                frame_idx = 0
    finally:
        if viewer is not None:
            viewer.close()
        dataset["data"].close()
