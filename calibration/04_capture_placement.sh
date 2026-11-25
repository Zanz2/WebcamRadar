#!/usr/bin/env bash
# Step 4: lay the board FLAT on the floor where every camera sees it, then run this.
# It saves one averaged still per camera. Don't move the board until all are done.
source "$(dirname "$0")/_common.sh"
for CAM in ${@:-$CAMERAS}; do
  cal snapshot --config "$CONFIG" --camera "$CAM" --board "$BOARD" --out "$WORK/poses/cam$CAM.png"
done
echo "The board's printed origin corner is now the world origin (x along its bottom edge, y up the page, z up)."
