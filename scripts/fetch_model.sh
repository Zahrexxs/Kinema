#!/usr/bin/env bash
# Fetch the MediaPipe Pose Landmarker weights that MediaPipeEstimator loads at
# runtime. They are gitignored because the "heavy" variant alone is ~30 MB.
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ahma/core/models"
BASE="https://storage.googleapis.com/mediapipe-models/pose_landmarker"

mkdir -p "$MODEL_DIR"

for variant in lite full heavy; do
  dest="$MODEL_DIR/pose_landmarker_${variant}.task"
  if [ -f "$dest" ]; then
    echo "==> ${variant}: already present, skipping"
    continue
  fi
  echo "==> ${variant}: downloading"
  curl -fsSL -o "$dest" \
    "${BASE}/pose_landmarker_${variant}/float16/latest/pose_landmarker_${variant}.task"
done

echo "Models ready in $MODEL_DIR"
echo
echo "MediaPipeEstimator selects pose_landmarker_{lite,full,heavy}.task"
echo "from model_complexity 0, 1 and 2 respectively."
