#!/usr/bin/env bash
set -euo pipefail

# Connection and retargeting
redis_ip="localhost"
actual_human_height=1.5  # PICO height estimate tends to run high.
target_fps=100

# Viewer: offsets are in the MuJoCo world frame (meters).
show_raw_pico_devices=true
raw_pico_devices_offset=(0.0 -1.2 0.0)
show_raw_xrobot_skeleton=true
raw_xrobot_skeleton_offset=(0.0 1.2 0.0)

# Each recorded clip uses the same directory and index for motion, raw XRobot, and raw device files.
save_pkl_enabled=true
save_pkl_dir="pico/debug/"
save_pkl_every_n_steps=1000
save_pkl_fps=100
save_pkl_prefix="pico"
save_pkl_resume_discard_frames=30
save_raw_xrobot_data=true

# Original SDK device poses: one CSV per recorded motion clip.
save_raw_device_poses=true

python scripts/xrobot_teleop_to_robot_w_hand.py \
    --robot robros_igris_c_v2 \
    --actual_human_height "$actual_human_height" \
    --redis_ip "$redis_ip" \
    --target_fps "$target_fps" \
    --measure_fps 1 \
    --retarget_velocity_lpf_enabled true \
    --retarget_velocity_lpf_cutoff_hz 8.0 \
    --pinch_mode \
    --show_raw_pico_devices "$show_raw_pico_devices" \
    --raw_pico_devices_offset "${raw_pico_devices_offset[@]}" \
    --show_raw_xrobot_skeleton "$show_raw_xrobot_skeleton" \
    --raw_xrobot_skeleton_offset "${raw_xrobot_skeleton_offset[@]}" \
    --save_pkl_enabled "$save_pkl_enabled" \
    --save_pkl_dir "$save_pkl_dir" \
    --save_pkl_every_n_steps "$save_pkl_every_n_steps" \
    --save_pkl_fps "$save_pkl_fps" \
    --save_pkl_prefix "$save_pkl_prefix" \
    --save_pkl_resume_discard_frames "$save_pkl_resume_discard_frames" \
    --save_raw_xrobot_data "$save_raw_xrobot_data" \
    --save_raw_device_poses "$save_raw_device_poses" \
    "$@"
