#!/usr/bin/env bash
# Step 5 (alternative, for big/outdoor spaces): placement from measured landmarks.
# 1. Fill calibration/landmarks.csv (copy landmarks.example.csv) with 6+ points you measured, in metres.
# 2. Usage: 05b_placement_from_landmarks.sh <camera_index>
#    It grabs a still, lets you click each landmark, then solves the pose.
source "$(dirname "$0")/_common.sh"
CAM="${1:?usage: $0 <camera_index>}"
LANDMARKS="${LANDMARKS:-calibration/landmarks.csv}"
need "$LANDMARKS" "copy calibration/landmarks.example.csv and fill in your measurements"
need "$RESULTS/cam${CAM}_intrinsics.json" "run 03_solve_lens.sh first"
cal snapshot --config "$CONFIG" --camera "$CAM" --out "$WORK/landmarks/cam$CAM.png"
cal click-landmarks --image "$WORK/landmarks/cam$CAM.png" --landmarks "$LANDMARKS" --out "$WORK/landmarks/cam${CAM}_points.csv"
cal pose-landmarks --intrinsics "$RESULTS/cam${CAM}_intrinsics.json" --landmarks "$LANDMARKS" \
    --points "$WORK/landmarks/cam${CAM}_points.csv" --out "$RESULTS/cam${CAM}_pose.json"
