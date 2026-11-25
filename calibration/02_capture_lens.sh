#!/usr/bin/env bash
# Step 2: collect ~25 lens-calibration images per camera.
#   02_capture_lens.sh            -> every camera in CAMERAS (settings.sh), one after another,
#                                    pausing so you can move to the next camera
#   02_capture_lens.sh <index>    -> just that camera  (extra args go to `p2v-calibrate capture`)
# Hold the board still at many spots, distances and tilts -- including the image corners.
# A window shows detection + coverage; frames save automatically when the board is still.
# It stops after the target count (or press q) and moves on.
source "$(dirname "$0")/_common.sh"
need "$BOARD" "run 01_make_board.sh first"

capture_one() {
  echo "=== camera $1: capturing into $WORK/captures/cam$1 ==="
  cal capture --config "$CONFIG" --camera "$1" --board "$BOARD" --out "$WORK/captures/cam$1" "${@:2}"
}

if [ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  capture_one "$@"
  exit 0
fi

set -- $CAMERAS
TOTAL=$#
N=0
for CAM in $CAMERAS; do
  N=$((N + 1))
  read -r -p "Camera $CAM ($N of $TOTAL): get in front of it with the board, then press Enter (s = skip) " ANSWER
  if [ "$ANSWER" = "s" ]; then
    echo "skipped camera $CAM"
    continue
  fi
  capture_one "$CAM"
done
echo "All cameras done. Next: calibration/03_solve_lens.sh"
