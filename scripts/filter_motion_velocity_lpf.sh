#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_FILE="${INPUT_FILE:-/home/robros/workspace/GMR/pico/recorded_arms.pkl}"
OUTPUT_FILE="${OUTPUT_FILE:-/home/robros/workspace/GMR/pico/recorded_arms_lpf.pkl}"
CUTOFF_HZ="${CUTOFF_HZ:-8.0}"
SAMPLE_RATE="${SAMPLE_RATE:-}"
BIDIRECTIONAL="${BIDIRECTIONAL:-true}"
OVERWRITE="${OVERWRITE:-true}"

ARGS=(
  --input_file "$INPUT_FILE"
  --output_file "$OUTPUT_FILE"
  --cutoff_hz "$CUTOFF_HZ"
)

if [[ -n "$SAMPLE_RATE" ]]; then
  ARGS+=(--sample_rate "$SAMPLE_RATE")
fi

if [[ "$BIDIRECTIONAL" == "true" ]]; then
  ARGS+=(--bidirectional)
fi

if [[ "$OVERWRITE" == "true" ]]; then
  ARGS+=(--overwrite)
fi

cd "$REPO_ROOT"
python3 "$SCRIPT_DIR/filter_motion_velocity_lpf.py" "${ARGS[@]}" "$@"
