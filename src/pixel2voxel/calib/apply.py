"""Write calibration results into a camera config (a new file; the original is kept)."""

from __future__ import annotations

import json
from pathlib import Path

LENS_KEYS = ("model", "width", "height", "K", "dist", "balance")


def apply(config_in: str | Path, results_dir: str | Path, config_out: str | Path) -> list[str]:
    """For each camera N: results/camN_intrinsics.json -> "intrinsics",
    results/camN_pose.json -> camera_position + yaw/pitch/roll. Returns a log."""
    data = json.loads(Path(config_in).read_text())
    cams = data if isinstance(data, list) else data.get("cameras", [data])
    results = Path(results_dir)
    log = []
    for cam in cams:
        n = int(cam.get("camera_index", 0))
        lens_f, pose_f = results / f"cam{n}_intrinsics.json", results / f"cam{n}_pose.json"
        done = []
        if lens_f.exists():
            lens = json.loads(lens_f.read_text())
            cam["intrinsics"] = {k: lens[k] for k in LENS_KEYS}
            done.append(f"lens ({lens['model']}, {lens['width']}x{lens['height']}, rms {lens.get('rms_px', '?')} px)")
        if pose_f.exists():
            pose = json.loads(pose_f.read_text())
            cam["camera_position"] = pose["camera_position"]
            cam["yaw"], cam["pitch"], cam["roll"] = pose["yaw"], pose["pitch"], pose["roll"]
            done.append(f"pose ({pose['method']}, pos {[round(v, 3) for v in pose['camera_position']]}, "
                        f"rms {pose['rms_px']} px)")
        missing = [name for name, f in (("lens", lens_f), ("placement", pose_f)) if not f.exists()]
        line = f"camera {n}: " + (", ".join(done) if done else "no results found (unchanged)")
        if missing:
            line += f"\n  WARNING: camera {n} has no {' or '.join(missing)} result: calibrate it or delete it from the config"
        log.append(line)
    grid = data.get("grid") if isinstance(data, dict) else None
    if isinstance(grid, dict) and "center" in grid:
        size = grid.get("size", 500)
        n = size if isinstance(size, list) else [size] * 3
        half = [v * grid.get("voxel_size", 6.0) / 2 for v in n]
        lo = [round(c - h, 3) for c, h in zip(grid["center"], half)]
        hi = [round(c + h, 3) for c, h in zip(grid["center"], half)]
        log.append(f"grid covers x {lo[0]}..{hi[0]}, y {lo[1]}..{hi[1]}, z {lo[2]}..{hi[2]} -- positions are now in the "
                   "calibration's world frame (board origin corner); move the grid if it no longer covers the space")
    Path(config_out).write_text(json.dumps(data, indent=2) + "\n")
    return log
