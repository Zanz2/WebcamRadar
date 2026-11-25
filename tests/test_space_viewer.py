import math
import subprocess
import sys

import numpy as np

from pixel2voxel import gridio, viewer
from pixel2voxel.space import process_image


def test_space_kernel_single_pixel_ray():
    # One bright centre pixel, telescope pointing +x from the origin, grid on the +x axis.
    img = np.zeros((10, 10))  # even size: pixel (5, 5) sits exactly on the optical axis (cx = cy = 5)
    img[5, 5] = 2.0
    grid = np.zeros((10, 4, 4))
    ext = np.array([[100.0, 200.0], [-20.0, 20.0], [-20.0, 20.0]])
    tex = np.zeros((32, 32))
    process_image(img, np.zeros(3), np.array([1.0, 0, 0]), math.radians(5), grid, ext, 1000.0, 1000, tex,
                  0.0, 0.0, math.radians(10), math.radians(10), True, False)
    # step = 1.0 -> samples at x = 100..200 inclusive -> 101 samples, all on the axis (y, z index 2)
    assert grid.sum() == 2.0 * 101
    assert np.count_nonzero(grid[:, 2, 2]) == 10 and grid[:, :2].sum() == 0
    assert tex.sum() == 2.0  # pixel direction lies inside the sky patch


def test_voxel_points_centres_and_cap():
    g = np.zeros((4, 4, 4), np.float32)
    g[0, 0, 0], g[3, 3, 3], g[1, 2, 3] = 5, 9, 1
    pts, vals = viewer.voxel_points(g, 2.0, 0.5)
    assert sorted(vals) == [1, 5, 9]
    assert np.allclose(pts[np.argmax(vals)], [3, 3, 3])  # (3 + 0.5) * 2 - 4
    pts, vals = viewer.voxel_points(g, 2.0, 0.5, max_points=1)
    assert list(vals) == [9]


def test_viewer_offscreen_screenshot(tmp_path):
    g = np.zeros((20, 20, 20), np.float32)
    g[5:15, 10, 10] = 10
    gridio.write_grid_bin(tmp_path / "g.bin", g, 1.0)
    png = tmp_path / "shot.png"
    r = subprocess.run([sys.executable, "-m", "pixel2voxel.viewer", str(tmp_path / "g.bin"),
                        "--screenshot", str(png)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert png.stat().st_size > 1000
