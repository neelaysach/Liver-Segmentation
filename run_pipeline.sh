#!/usr/bin/env bash

set -euo pipefail

IMAGES_DIR="batch7_usable"
YOLO_WEIGHTS="pipeline_blocks/runs/detect_from_labels/weights/best.pt"
MEDSAM_REPO="MedSAM"
MEDSAM_CHECKPOINT="MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
OUTPUT_DIR="batch7_outputs"

python pipeline_blocks/medsam_run.py \
  --images-dir "$IMAGES_DIR" \
  --yolo-weights "$YOLO_WEIGHTS" \
  --medsam-repo "$MEDSAM_REPO" \
  --medsam-checkpoint "$MEDSAM_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR"