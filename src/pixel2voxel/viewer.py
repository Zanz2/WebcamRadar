"""Live 3D voxel grid viewer.

Thresholds the grid with numpy and renders the surviving voxels as a PyVista point cloud.

Sources (pick one; the default is the live ZeroMQ stream from p2v-rtsp):
  p2v-view                          live stream on tcp://127.0.0.1:$ZMQ_PORT (5556)
  p2v-view --watch output_dir       reload the newest .bin in a directory
  p2v-view voxel_grid_123.bin       a single file (either .bin layout)

Keys: mouse = orbit/zoom/pan, + / - = lower / raise threshold, q = quit.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import zmq

from . import gridio


def voxel_points(grid: np.ndarray, voxel_size: float, threshold: float,
                 center=(0.0, 0.0, 0.0), max_points: int = 2_000_000):
    """World-space centres and values of voxels above `threshold`.

    Coordinates are relative to the grid centre (+ `center`). If more than `max_points` voxels pass, the brightest are kept.
    """
    idx = np.flatnonzero(grid > threshold)
    if idx.size == 0:
        return np.empty((0, 3), np.float32), np.empty(0, np.float32)
    values = grid.ravel()[idx]
    if idx.size > max_points:
        keep = np.argpartition(values, -max_points)[-max_points:]
        idx, values = idx[keep], values[keep]
    ijk = np.stack(np.unravel_index(idx, grid.shape), axis=1).astype(np.float32)
    half = np.asarray(grid.shape, np.float32) * voxel_size / 2.0
    pts = (ijk + 0.5) * voxel_size - half + np.asarray(center, np.float32)
    return pts, values


class Viewer:
    def __init__(self, threshold: float, center, max_points: int, off_screen: bool = False):
        self.threshold = threshold
        self.center = center
        self.max_points = max_points
        self.grid: np.ndarray | None = None
        self.voxel_size = 1.0
        self.dirty = False
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=(1200, 800), title="3D Voxel Grid Viewer")
        self.plotter.set_background("black")
        self.plotter.add_axes()
        self._text = self.plotter.add_text("", position="upper_left", font_size=10, color="white")
        self._camera_set = False

    def set_grid(self, grid: np.ndarray, voxel_size: float) -> None:
        self.grid, self.voxel_size, self.dirty = grid, voxel_size, True

    def adjust(self, factor: float) -> None:
        self.threshold *= factor
        self.dirty = True
        print(f"Threshold: {self.threshold:g}")

    def redraw(self) -> None:
        if not self.dirty or self.grid is None:
            return
        self.dirty = False
        pts, vals = voxel_points(self.grid, self.voxel_size, self.threshold, self.center, self.max_points)
        if len(pts) == 0:
            self.plotter.remove_actor("voxels", render=False)
        else:
            cloud = pv.PolyData(pts)
            cloud["value"] = vals
            self.plotter.add_points(
                cloud, name="voxels", scalars="value", cmap="hot",
                clim=(self.threshold, self.threshold * 5.0), point_size=4,
                render_points_as_spheres=False, show_scalar_bar=False, reset_camera=False,
            )
        if not self._camera_set:
            # Frame the whole grid volume once; later updates keep the user's view.
            half = np.asarray(self.grid.shape) * self.voxel_size / 2.0
            c = np.asarray(self.center)
            self.plotter.add_mesh(pv.Box(bounds=(c[0] - half[0], c[0] + half[0], c[1] - half[1],
                                                 c[1] + half[1], c[2] - half[2], c[2] + half[2])),
                                  style="wireframe", color="gray", name="bounds")
            self.plotter.reset_camera()
            self._camera_set = True
        nx, ny, nz = self.grid.shape
        self._text.SetText(2, f"{nx}x{ny}x{nz} @ {self.voxel_size:g}   threshold {self.threshold:g}   "
                              f"{len(pts):,} voxels shown   (+/- threshold, q quit)")
        self.plotter.render()


def _newest_bin(folder: Path) -> Path | None:
    bins = [p for p in folder.glob("*.bin") if p.is_file()]
    return max(bins, key=lambda p: p.stat().st_mtime) if bins else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p2v-view", description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("file", nargs="?", help="a single .bin file")
    src.add_argument("--watch", metavar="DIR", help="reload the newest .bin in DIR")
    src.add_argument("--zmq", metavar="ADDR", help="ZeroMQ address (default tcp://127.0.0.1:$ZMQ_PORT or 5556)")
    ap.add_argument("--threshold", type=float, default=0.5, help="initial brightness threshold (default 0.5)")
    ap.add_argument("--center", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                    help="offset added to coordinates (default: grid centred on the origin)")
    ap.add_argument("--max-points", type=int, default=2_000_000, help="cap on rendered voxels (brightest kept)")
    ap.add_argument("--screenshot", metavar="PNG", help="render FILE off-screen to PNG and exit")
    args = ap.parse_args(argv)

    if args.screenshot:
        if not args.file:
            ap.error("--screenshot needs a FILE")
        viewer = Viewer(args.threshold, args.center, args.max_points, off_screen=True)
        viewer.set_grid(*gridio.read_bin(args.file))
        viewer.redraw()
        viewer.plotter.screenshot(args.screenshot)
        print(f"Saved {args.screenshot}")
        return 0

    viewer = Viewer(args.threshold, args.center, args.max_points)
    viewer.plotter.add_key_event("plus", lambda: viewer.adjust(0.8))
    viewer.plotter.add_key_event("equal", lambda: viewer.adjust(0.8))
    viewer.plotter.add_key_event("minus", lambda: viewer.adjust(1.2))

    sub = None
    watch_dir, last_mtime = None, 0.0
    if args.file:
        viewer.set_grid(*gridio.read_bin(args.file))
        print(f"Loaded {args.file}: shape {viewer.grid.shape}, voxel size {viewer.voxel_size:g}, "
              f"max value {viewer.grid.max():g}")
    elif args.watch:
        watch_dir = Path(args.watch)
        print(f"Watching {watch_dir} for new .bin files")
    else:
        address = args.zmq or f"tcp://127.0.0.1:{os.environ.get('ZMQ_PORT', '5556')}"
        sub = zmq.Context.instance().socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.setsockopt(zmq.RCVHWM, 2)
        sub.connect(address)
        print(f"Listening for live grids on {address}")

    def tick(_step=None):
        nonlocal last_mtime
        if sub is not None:
            latest = None
            while True:  # drain, keep only the newest grid
                try:
                    latest = sub.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
            if latest is not None and len(latest) == 2:
                first = viewer.grid is None
                viewer.set_grid(*gridio.decode_zmq(*latest))
                if first:
                    print(f"Receiving live grid {viewer.grid.shape}, voxel size {viewer.voxel_size:g}")
        elif watch_dir is not None:
            newest = _newest_bin(watch_dir)
            if newest is not None and newest.stat().st_mtime > last_mtime:
                try:
                    viewer.set_grid(*gridio.read_bin(newest))
                    last_mtime = newest.stat().st_mtime
                    print(f"Loaded {newest.name}")
                except ValueError:
                    pass  # file still being written; retry next tick
        viewer.redraw()

    tick()
    viewer.plotter.add_timer_event(max_steps=10**9, duration=100, callback=tick)
    viewer.plotter.show()
    if sub is not None:
        sub.close(linger=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
