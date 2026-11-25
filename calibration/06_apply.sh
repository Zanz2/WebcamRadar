#!/usr/bin/env bash
# Step 6: write the results into a NEW config (your original is kept).
source "$(dirname "$0")/_common.sh"
cal apply --config "$CONFIG" --results "$RESULTS" --out "$CALIBRATED"
