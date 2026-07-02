#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_DIR="${INPUT_DIR:-/home/robros/workspace/motion_datas/bones_seed_g1_npz/}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/robros/workspace/motion_datas/retargeted/bones_seed_retargeted/}"
PATTERN="${PATTERN:-*.npz}"
ROOT_QUAT_SCALAR_FIRST="${ROOT_QUAT_SCALAR_FIRST:-true}"
MOTION_FPS="${MOTION_FPS:-30}"
OFFSET_TO_GROUND="${OFFSET_TO_GROUND:-0}"
OVERWRITE="${OVERWRITE:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/retarget_g1_to_igris_c_batch.sh --input_dir DIR --output_dir DIR [options] [-- extra g1_to_igris_c.py args]

Options:
  -i, --input_dir DIR              Source directory containing G1 .npz or .pkl files.
  -o, --output_dir DIR             Destination directory for Igris C .pkl files.
  -p, --pattern GLOB               Input filename glob relative to input_dir. Default: *.npz
      --root_quat_scalar_first V   Whether source .pkl root quat is wxyz. Default: true
      --motion_fps FPS             FPS used when source .npz has no fps key. Default: 30
      --offset_to_ground           Pass --offset_to_ground to retargeter.
      --overwrite                  Retarget even when the output already exists.
  -h, --help                       Show this help.

Environment overrides:
  PYTHON_BIN, INPUT_DIR, OUTPUT_DIR, PATTERN, ROOT_QUAT_SCALAR_FIRST, MOTION_FPS,
  OFFSET_TO_GROUND, OVERWRITE
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input_dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    -o|--output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -p|--pattern)
      PATTERN="$2"
      shift 2
      ;;
    --root_quat_scalar_first)
      ROOT_QUAT_SCALAR_FIRST="$2"
      shift 2
      ;;
    --motion_fps)
      MOTION_FPS="$2"
      shift 2
      ;;
    --offset_to_ground)
      OFFSET_TO_GROUND="1"
      shift
      ;;
    --overwrite)
      OVERWRITE="1"
      shift
      ;;
    --no-overwrite)
      OVERWRITE="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Both --input_dir and --output_dir are required." >&2
  usage >&2
  exit 2
fi

INPUT_DIR="${INPUT_DIR%/}"
OUTPUT_DIR="${OUTPUT_DIR%/}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory does not exist: $INPUT_DIR" >&2
  exit 1
fi

RETARGET_ARGS=(--no_viewer)
if [[ "$OFFSET_TO_GROUND" == "1" ]]; then
  RETARGET_ARGS+=(--offset_to_ground)
fi

mapfile -d '' MOTION_FILES < <(find "$INPUT_DIR" -type f -name "$PATTERN" -print0 | sort -z)
if [[ "${#MOTION_FILES[@]}" -eq 0 ]]; then
  echo "No files matched pattern '$PATTERN' under $INPUT_DIR" >&2
  exit 1
fi

total="${#MOTION_FILES[@]}"
done_count=0
skip_count=0

for motion_path in "${MOTION_FILES[@]}"; do
  rel_path="${motion_path#"$INPUT_DIR"/}"
  rel_without_ext="${rel_path%.*}"
  save_path="$OUTPUT_DIR/$rel_without_ext.pkl"

  if [[ "$OVERWRITE" != "1" && -e "$save_path" ]]; then
    echo "[skip] $save_path already exists"
    skip_count=$((skip_count + 1))
    continue
  fi

  mkdir -p "$(dirname "$save_path")"
  echo "[$((done_count + skip_count + 1))/$total] $motion_path -> $save_path"

  "$PYTHON_BIN" scripts/g1_to_igris_c.py \
    --g1_motion_path "$motion_path" \
    --save_path "$save_path" \
    --root_quat_scalar_first "$ROOT_QUAT_SCALAR_FIRST" \
    --motion_fps "$MOTION_FPS" \
    "${RETARGET_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

  done_count=$((done_count + 1))
done

echo "Done. Retargeted: $done_count, skipped: $skip_count, total: $total"
