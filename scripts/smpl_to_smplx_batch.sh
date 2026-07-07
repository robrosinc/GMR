#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PATTERN="${PATTERN:-*.npz}"
GENDER="${GENDER:-neutral}"
ROOT_ROTATION_AXIS="${ROOT_ROTATION_AXIS:-x}"
ROOT_ROTATION_DEGREES="${ROOT_ROTATION_DEGREES:-90}"
ROTATE_TRANSLATION="${ROTATE_TRANSLATION:-1}"
ROOT_HEIGHT_AXIS="${ROOT_HEIGHT_AXIS:-z}"
ROOT_HEIGHT_OFFSET="${ROOT_HEIGHT_OFFSET:-0.97}"
OVERWRITE="${OVERWRITE:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/smpl_to_smplx_batch.sh --input_dir DIR --output_dir DIR [options]

Options:
  -i, --input_dir DIR              Source directory containing SMPL .npz files.
  -o, --output_dir DIR             Destination directory for SMPL-X .npz files.
  -p, --pattern GLOB               Input filename glob. Default: *.npz
      --gender GENDER              Gender to write when missing. Default: neutral
      --root_rotation_axis AXIS    Root rotation axis: x, y, or z. Default: x
      --root_rotation_degrees DEG  Root rotation degrees. Default: 90
      --rotate_translation         Rotate translation trajectory. Default enabled
      --no-rotate_translation      Do not rotate translation trajectory.
      --root_height_axis AXIS      Root height offset axis: x, y, or z. Default: z
      --root_height_offset VALUE   Root height offset. Default: 0.97
      --overwrite                  Convert even when output already exists.
  -h, --help                       Show this help.

Environment overrides:
  PYTHON_BIN, INPUT_DIR, OUTPUT_DIR, PATTERN, GENDER, ROOT_ROTATION_AXIS,
  ROOT_ROTATION_DEGREES, ROTATE_TRANSLATION, ROOT_HEIGHT_AXIS,
  ROOT_HEIGHT_OFFSET, OVERWRITE
EOF
}

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
    --gender)
      GENDER="$2"
      shift 2
      ;;
    --root_rotation_axis)
      ROOT_ROTATION_AXIS="$2"
      shift 2
      ;;
    --root_rotation_degrees)
      ROOT_ROTATION_DEGREES="$2"
      shift 2
      ;;
    --rotate_translation)
      ROTATE_TRANSLATION="1"
      shift
      ;;
    --no-rotate_translation)
      ROTATE_TRANSLATION="0"
      shift
      ;;
    --root_height_axis)
      ROOT_HEIGHT_AXIS="$2"
      shift 2
      ;;
    --root_height_offset)
      ROOT_HEIGHT_OFFSET="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
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

case "$GENDER" in
  male|female|neutral) ;;
  *)
    echo "Invalid --gender '$GENDER'. Expected male, female, or neutral." >&2
    exit 2
    ;;
esac

case "$ROOT_ROTATION_AXIS" in
  x|y|z) ;;
  *)
    echo "Invalid --root_rotation_axis '$ROOT_ROTATION_AXIS'. Expected x, y, or z." >&2
    exit 2
    ;;
esac

case "$ROOT_HEIGHT_AXIS" in
  x|y|z) ;;
  *)
    echo "Invalid --root_height_axis '$ROOT_HEIGHT_AXIS'. Expected x, y, or z." >&2
    exit 2
    ;;
esac

CONVERT_ARGS=(
  --gender "$GENDER"
  --root_rotation_axis "$ROOT_ROTATION_AXIS"
  --root_rotation_degrees "$ROOT_ROTATION_DEGREES"
  --root_height_axis "$ROOT_HEIGHT_AXIS"
  --root_height_offset "$ROOT_HEIGHT_OFFSET"
)

if [[ "$ROTATE_TRANSLATION" == "1" ]]; then
  CONVERT_ARGS+=(--rotate_translation)
fi

mapfile -d '' INPUT_FILES < <(find "$INPUT_DIR" -type f -name "$PATTERN" -print0 | sort -z)
if [[ "${#INPUT_FILES[@]}" -eq 0 ]]; then
  echo "No files matched pattern '$PATTERN' under $INPUT_DIR" >&2
  exit 1
fi

total="${#INPUT_FILES[@]}"
converted_count=0
skip_count=0

for input_file in "${INPUT_FILES[@]}"; do
  rel_path="${input_file#"$INPUT_DIR"/}"
  output_file="$OUTPUT_DIR/$rel_path"

  if [[ "$OVERWRITE" != "1" && -e "$output_file" ]]; then
    echo "[skip] $output_file already exists"
    skip_count=$((skip_count + 1))
    continue
  fi

  mkdir -p "$(dirname "$output_file")"
  echo "[$((converted_count + skip_count + 1))/$total] $input_file -> $output_file"

  "$PYTHON_BIN" scripts/smpl_to_smplx.py \
    --input_file "$input_file" \
    --output_file "$output_file" \
    "${CONVERT_ARGS[@]}"

  converted_count=$((converted_count + 1))
done

echo "Done. Converted: $converted_count, skipped: $skip_count, total: $total"
