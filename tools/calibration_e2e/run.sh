#!/usr/bin/env bash
# End-to-end test of the calibration/ step scripts, run exactly as a user would, with three synthetic
# cameras (2 standard ~87-108 deg, 1 fisheye ~130 deg) posing as RTSP streams via video files.
# The URLs change between steps only because the synthetic scene changes (board held up / on the floor /
# raised); with real cameras they never change.  Usage: tools/calibration_e2e/run.sh   (~30 s)
set -euo pipefail
cd "$(dirname "$0")/../.."
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
export WORK="$T/work" PAPER=A3 CAMERAS="0 1 2"
calibration/01_make_board.sh > /dev/null
uv run python tools/calibration_e2e/make_scene.py "$T"
for c in 0 1 2; do CONFIG="$T/lens.json" calibration/02_capture_lens.sh "$c" --no-gui --target 22 | tail -1; done
CONFIG="$T/lens.json"  calibration/03_solve_lens.sh | grep -E "===|Chosen|fx"
CONFIG="$T/floor.json" calibration/04_capture_placement.sh > /dev/null
CONFIG="$T/floor.json" calibration/05_solve_placement.sh | grep -E "===|Camera at|WARNING"
CONFIG="$T/rough.json" calibration/06_apply.sh
cp "$T/rough.calibrated.json" "$T/check.calibrated.json"
OUT="$(CONFIG="$T/check.json" calibration/07_check.sh | grep -v '^Saved')"
echo "$OUT"
uv run python - "$T" <<'PY'
import json, sys, numpy as np
T = sys.argv[1]; truth = json.load(open(f"{T}/truth.json"))
for ci, t in truth.items():
    est = json.load(open(f"{T}/work/results/cam{ci}_pose.json"))["camera_position"]
    print(f"camera {ci}: position error vs ground truth {np.linalg.norm(np.subtract(est, t['C'])) * 1000:.1f} mm")
PY
echo "$OUT" | grep -q "^GOOD" && echo "PASS" || { echo "FAIL: check verdict was not GOOD"; exit 1; }
