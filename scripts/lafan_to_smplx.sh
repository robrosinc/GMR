#!/usr/bin/env bash

set -euo pipefail

TARGET_MOTION_NAME="${TARGET_MOTION_NAME:-obstacles1_subject1}"
BVH_ROOT=${BVH_ROOT:-/home/robros/workspace/motion_datas/lafan1}
SMPLX_MODEL=${SMPLX_MODEL:-/home/robros/workspace/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz}
OUTPUT_DIR=${OUTPUT_DIR:-output}
REFERENCE_SAMPLE=${REFERENCE_SAMPLE:-data/ACCAD/Female1General_c3d/A1_-_Stand_stageii.npz}

BVH_FILE="${BVH_ROOT}/${TARGET_MOTION_NAME}.bvh"
RAW_OUTPUT="${OUTPUT_DIR}/${TARGET_MOTION_NAME}.npz"
ENRICHED_OUTPUT="${OUTPUT_DIR}/${TARGET_MOTION_NAME}_enriched.npz"

python scripts/lafan_to_smplx.py \
  --bvh_file "${BVH_FILE}" \
  --smplx_model_path "${SMPLX_MODEL}" \
  --output "${RAW_OUTPUT}"

python scripts/enrich_smplx_npz.py \
  --input_npz "${RAW_OUTPUT}" \
  --reference "${REFERENCE_SAMPLE}" \
  --output_npz "${ENRICHED_OUTPUT}"
