#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
G1_MOTION_PATH="${G1_MOTION_PATH:-/home/robros/workspace/kimodo/rolling.npz}"
SAVE_PATH="${SAVE_PATH:-/home/robros/workspace/kimodo/rolling.pkl}"
ROOT_QUAT_SCALAR_FIRST="${ROOT_QUAT_SCALAR_FIRST:-true}"
MOTION_FPS="${MOTION_FPS:-30}"
NO_VIEWER="${NO_VIEWER:-0}"
VISUALIZE="${VISUALIZE:-1}"

EXTRA_ARGS=()
if [[ "$NO_VIEWER" == "1" ]]; then
  EXTRA_ARGS+=(--no_viewer)
fi

"$PYTHON_BIN" scripts/g1_to_igris_c.py \
  --g1_motion_path "$G1_MOTION_PATH" \
  --save_path "$SAVE_PATH" \
  --root_quat_scalar_first "$ROOT_QUAT_SCALAR_FIRST" \
  --motion_fps "$MOTION_FPS" \
  "${EXTRA_ARGS[@]}" \
  "$@"

if [[ "$VISUALIZE" == "1" ]]; then
  "$PYTHON_BIN" scripts/vis_robot_motion.py \
    --robot robros_igris_c_v2 \
    --robot_motion_path "$SAVE_PATH" \
    --root_quat_scalar_first true
fi
