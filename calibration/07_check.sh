#!/usr/bin/env bash
# Step 7: MOVE the board to a new spot (e.g. on a box, turned) where 2+ cameras see it, then run this.
# Reports how closely the cameras' rays meet, in mm and in voxels of your grid.
source "$(dirname "$0")/_common.sh"
need "$CALIBRATED" "run 06_apply.sh first"
IMAGES=()
for CAM in ${@:-$CAMERAS}; do
  cal snapshot --config "$CONFIG" --camera "$CAM" --board "$BOARD" --out "$WORK/check/cam$CAM.png"
  IMAGES+=(--image "$CAM=$WORK/check/cam$CAM.png")
done
cal check --config "$CALIBRATED" --board "$BOARD" ${SQ_ARG[@]+"${SQ_ARG[@]}"} "${IMAGES[@]}"
