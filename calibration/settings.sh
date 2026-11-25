# Calibration settings -- edit these, the numbered scripts read them.
# Paths are relative to the repository root. Environment variables of the same name override these.
# Optional: put your own values in calibration/settings.local.sh (git-ignored).
if [ -f calibration/settings.local.sh ]; then source calibration/settings.local.sh; fi

CONFIG="${CONFIG:-config/my_cameras.json}"   # your camera config (copy one of the examples)
CAMERAS="${CAMERAS:-0 1 2}"       # the camera_index values in that config
PAPER="${PAPER:-A3}"              # A4 A3 A2 A1 A0 letter tabloid -- bigger is better for placement
SQUARES="7 5"                     # board squares (x y)
SQUARE_MM="${SQUARE_MM:-}"        # MEASURED square size if it differs from the print (e.g. printed scaled)
WORK="${WORK:-calibration/work}"  # everything generated goes here (git-ignored: photos of your space)

BOARD="$WORK/board/board.json"
RESULTS="$WORK/results"
CALIBRATED="${CONFIG%.json}.calibrated.json"
SQ_ARG=()
if [ -n "$SQUARE_MM" ]; then SQ_ARG=(--square-mm "$SQUARE_MM"); fi
