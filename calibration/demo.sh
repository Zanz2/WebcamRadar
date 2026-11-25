#!/usr/bin/env bash
# Try the whole workflow on synthetic cameras first (no hardware, ~10 s).
source "$(dirname "$0")/_common.sh"
cal demo --out calibration/demo
