"""Offline reconstruction from pre-recorded images.

Reads a metadata JSON (a list of frames),
groups frames by camera, diffs consecutive frames, casts rays for changed
pixels and writes the accumulated grid.

Behaviour:
  * the batch rotation convention, Rz(yaw) Ry(roll) Rx(pitch), which differs
    from the RTSP tool's (see geometry.rotation_batch)
  * uniform noise in [-1, 1] added to every loaded pixel (use --noise 0 to disable)
  * no distance attenuation
  * output in the cube .bin format (int32 N, float32 voxel_size, data)
The grid constants are flags.

Usage:  p2v-batch <metadata.json> <image_folder> <output_voxel.bin>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import geometry, gridio, raycast


def load_gray(path: Path, rng: np.random.Generator, noise: float) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    out = img.astype(np.float32)
    if noise > 0:
        out += rng.uniform(-noise, noise, size=out.shape).astype(np.float32)
        np.clip(out, 0.0, 255.0, out=out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p2v-batch", description=__doc__.split("\n\n")[0])
    ap.add_argument("metadata")
    ap.add_argument("image_folder")
    ap.add_argument("output_bin")
    ap.add_argument("--size", type=int, default=500, help="grid is N x N x N (default 500)")
    ap.add_argument("--voxel-size", type=float, default=6.0)
    ap.add_argument("--center", type=float, nargs=3, default=(0.0, 0.0, 500.0), metavar=("X", "Y", "Z"))
    ap.add_argument("--threshold", type=float, default=2.0, help="motion threshold (default 2.0)")
    ap.add_argument("--noise", type=float, default=1.0, help="uniform noise amplitude added per pixel (default 1.0)")
    ap.add_argument("--seed", type=int, default=None, help="seed for the noise (default: random)")
    args = ap.parse_args(argv)

    frames = json.loads(Path(args.metadata).read_text())
    if not isinstance(frames, list) or not frames:
        print("No frames loaded.", file=sys.stderr)
        return 1

    by_cam: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by_cam[int(f.get("camera_index", 0))].append(f)

    n = args.size
    grid = np.zeros((n, n, n), dtype=np.float32)
    center = np.asarray(args.center, dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    folder = Path(args.image_folder)

    for cam_id in sorted(by_cam):
        cam_frames = sorted(by_cam[cam_id], key=lambda f: int(f.get("frame_index", 0)))
        if len(cam_frames) < 2:
            continue
        prev = None
        for info in cam_frames:
            curr = load_gray(folder / info.get("image_file", ""), rng, args.noise)
            if curr is None:
                print(f"Failed to load image: {folder / info.get('image_file', '')}", file=sys.stderr)
                continue
            if prev is None:
                prev = curr
                continue
            if prev.shape != curr.shape:
                print("Images differ in size. Can't do motion detection!", file=sys.stderr)
                prev = curr
                continue

            # Ray-cast with the *current* frame's camera pose.
            pos = info.get("camera_position", [0.0, 0.0, 0.0])
            rot = geometry.rotation_batch(info.get("yaw", 0.0), info.get("pitch", 0.0), info.get("roll", 0.0))
            h, w = curr.shape
            focal, _ = geometry.focal_lengths(w, h, info.get("fov_degrees", 60.0))
            raycast.cast_frame(grid, np.abs(prev - curr), args.threshold, pos[:3], rot,
                               focal, focal, center, args.voxel_size, 0.0, raycast.RAY_BATCH)
            prev = curr
        print(f"Camera {cam_id}: {len(cam_frames)} frames processed")

    gridio.write_cube_bin(args.output_bin, grid, args.voxel_size)
    print(f"Saved voxel grid to {args.output_bin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
