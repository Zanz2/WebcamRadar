#!/usr/bin/env bash
# Step 5: solve each camera's position and angles from the floor stills.
# Optional: ORIGIN="x y z" puts the board corner somewhere else in your world (metres).
source "$(dirname "$0")/_common.sh"
ORIGIN_ARG=()
if [ -n "${ORIGIN:-}" ]; then ORIGIN_ARG=(--origin $ORIGIN); fi
for CAM in ${@:-$CAMERAS}; do
  need "$RESULTS/cam${CAM}_intrinsics.json" "run 03_solve_lens.sh first"
  need "$WORK/poses/cam$CAM.png" "run 04_capture_placement.sh first"
  echo "=== camera $CAM ==="
  cal pose-board --image "$WORK/poses/cam$CAM.png" --intrinsics "$RESULTS/cam${CAM}_intrinsics.json" \
      --board "$BOARD" ${SQ_ARG[@]+"${SQ_ARG[@]}"} ${ORIGIN_ARG[@]+"${ORIGIN_ARG[@]}"} --out "$RESULTS/cam${CAM}_pose.json"
done
