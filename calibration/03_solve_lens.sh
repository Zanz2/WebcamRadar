#!/usr/bin/env bash
# Step 3: solve the lens for each camera (or the ones given as arguments).
source "$(dirname "$0")/_common.sh"
mkdir -p "$RESULTS"
for CAM in ${@:-$CAMERAS}; do
  need "$WORK/captures/cam$CAM" "run 02_capture_lens.sh $CAM first"
  echo "=== camera $CAM ==="
  cal intrinsics "$WORK/captures/cam$CAM" --board "$BOARD" \
      --out "$RESULTS/cam${CAM}_intrinsics.json" --preview "$RESULTS/cam${CAM}_undistort_preview.png"
  echo
done
