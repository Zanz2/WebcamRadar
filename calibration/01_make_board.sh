#!/usr/bin/env bash
# Step 1: generate the printable board (PDF + PNG + board.json).
source "$(dirname "$0")/_common.sh"
cal board --paper "$PAPER" --squares $SQUARES --out "$WORK/board"
echo
echo "Next: print $WORK/board/board.pdf at 100% (actual size), glue it to something flat and rigid,"
echo "measure the 100 mm ruler and one square. If the square differs, put it in SQUARE_MM in settings.sh."
