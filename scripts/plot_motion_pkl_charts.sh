#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MOTION_FILE="${MOTION_FILE:-/home/robros/workspace/GMR/pico/recorded_arms_lpf.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/recorded_arms_lpf}"
BODY_POSITION_KEY="${BODY_POSITION_KEY:-keybody_pos_world}"
JOINT_COLUMNS="${JOINT_COLUMNS:-4}"
BODY_COLUMNS="${BODY_COLUMNS:-4}"
DPI="${DPI:-150}"
TIME_AXIS="${TIME_AXIS:-true}"

ARGS=(
  --motion_file "$MOTION_FILE"
  --output_dir "$OUTPUT_DIR"
  --body_position_key "$BODY_POSITION_KEY"
  --joint_columns "$JOINT_COLUMNS"
  --body_columns "$BODY_COLUMNS"
  --dpi "$DPI"
)

if [[ "$TIME_AXIS" == "true" ]]; then
  ARGS+=(--time_axis)
fi

cd "$REPO_ROOT"
python3 "$SCRIPT_DIR/plot_motion_pkl_charts.py" "${ARGS[@]}" "$@"
