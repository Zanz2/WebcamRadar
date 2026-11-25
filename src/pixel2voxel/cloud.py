"""Point-cloud snapshot of a voxel grid.

Keeps the top percentile of voxels, applies an optional Euler rotation, saves a
numbered high-resolution screenshot into screenshots/voxel_####.png (a history
of runs), then opens an interactive window.

The array is interpreted as (Z, Y, X) -- the
opposite of how p2v-batch indexes it ([ix, iy, iz]) -- and the default
centre (30, 0, 14000) and rotation (90, 270, 0) can be changed
with flags. Either .bin layout can be read.

Usage:  p2v-cloud [voxel_grid.bin]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from . import gridio


def extract_top_percentile_z_up(grid, voxel_size, grid_center, percentile=99.5, hard_thresh=None):
    """(points[N, 3], intensities[N]) for voxels above the percentile (or hard threshold)."""
    n = grid.shape[0]
    grid_min = np.asarray(grid_center, np.float32) - (n * voxel_size) * 0.5
    thresh = hard_thresh if hard_thresh is not None else np.percentile(grid, percentile)
    coords = np.argwhere(grid > thresh)
    if coords.size == 0:
        print(f"No voxels above threshold {thresh}. Nothing to display.")
        return None, None
    intensities = grid[coords[:, 0], coords[:, 1], coords[:, 2]]
    z_idx, y_idx, x_idx = (coords[:, k] + 0.5 for k in range(3))  # index 0 -> z, 1 -> y, 2 -> x
    points = np.column_stack((grid_min[0] + x_idx * voxel_size,
                              grid_min[1] + y_idx * voxel_size,
                              grid_min[2] + z_idx * voxel_size))
    return points, intensities


def rotation_matrix_xyz(rx_deg, ry_deg, rz_deg):
    """R = Rz @ Ry @ Rx (rotate about X first, then Y, then Z)."""
    rx, ry, rz = (math.radians(a) for a in (rx_deg, ry_deg, rz_deg))
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
    return Rz @ Ry @ Rx


def next_image_index(folder: Path, prefix="voxel_", suffix=".png") -> int:
    if not folder.exists():
        return 1
    pat = re.compile(rf"^{prefix}(\d+){suffix}$")
    nums = [int(m.group(1)) for p in folder.iterdir() if (m := pat.match(p.name))]
    return max(nums, default=0) + 1


def _plotter(points, intensities, off_screen, opacity, scalar_bar):
    p = pv.Plotter(off_screen=off_screen)
    p.set_background("white")
    p.enable_terrain_style()
    cloud = pv.PolyData(points)
    cloud["intensity"] = intensities
    p.add_points(cloud, scalars="intensity", cmap="hot", point_size=4.0,
                 render_points_as_spheres=True, opacity=opacity, show_scalar_bar=False)
    if scalar_bar:
        p.add_scalar_bar(title="Brightness", n_labels=5)
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p2v-cloud", description=__doc__.split("\n\n")[0])
    ap.add_argument("file", nargs="?", default="voxel_grid.bin")
    ap.add_argument("--center", type=float, nargs=3, default=(30.0, 0.0, 14000.0), metavar=("X", "Y", "Z"))
    ap.add_argument("--rotate", type=float, nargs=3, default=(90.0, 270.0, 0.0), metavar=("RX", "RY", "RZ"))
    ap.add_argument("--percentile", type=float, default=99.9)
    ap.add_argument("--hard-threshold", type=float, default=None, help="use a fixed threshold instead")
    ap.add_argument("--screenshots", default="screenshots", help="folder for the numbered screenshots")
    ap.add_argument("--no-window", action="store_true", help="only save the screenshot")
    args = ap.parse_args(argv)

    if not Path(args.file).is_file():
        print(f"Usage: p2v-cloud <voxel_grid.bin>   ({args.file} not found)")
        return 1
    grid, vs = gridio.read_bin(args.file)
    print("Loaded", args.file, "shape", grid.shape, "voxel_size=", vs)
    print("Max voxel value:", grid.max())

    points, intensities = extract_top_percentile_z_up(grid, vs, args.center, args.percentile, args.hard_threshold)
    if points is None:
        return 0
    points = points @ rotation_matrix_xyz(*args.rotate).T

    folder = Path(args.screenshots)
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"voxel_{next_image_index(folder):04d}.png"
    p = _plotter(points, intensities, off_screen=True, opacity=0.1, scalar_bar=True)
    p.show(window_size=[3840, 2160], auto_close=False, screenshot=str(out))
    p.close()
    print(f"[Info] Saved screenshot to {out}")

    if not args.no_window:
        _plotter(points, intensities, off_screen=False, opacity=0.05, scalar_bar=False).show(window_size=[1920, 1080])
    return 0


if __name__ == "__main__":
    sys.exit(main())
