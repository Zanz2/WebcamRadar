"""End-to-end geometry: several cameras see a moving blob; the voxel peak must land on it."""

import numpy as np

from pixel2voxel import geometry, raycast

W, H = 320, 240


def project(point, cam_pos, rot, fx, fy):
    """Inverse of the RTSP ray model: world point -> pixel (u, v)."""
    d = rot.T.astype(np.float64) @ (np.asarray(point, float) - np.asarray(cam_pos, float))
    assert d[2] < 0, "point must be in front of the camera (camera looks along -z)"
    return fx * d[0] / -d[2] + W / 2, -fy * d[1] / -d[2] + H / 2


def blob_image(u, v, radius=3):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W), np.float32)
    img[(xx - u) ** 2 + (yy - v) ** 2 <= radius**2] = 200.0
    return img


def test_three_cameras_triangulate_moving_blob():
    shape, vs, center = (48, 48, 48), 0.25, np.array([0.0, 0.0, 0.0])
    grid = np.zeros(shape, np.float32)
    a, b = np.array([0.6, -0.9, 0.4]), np.array([-1.1, 0.7, -0.5])  # blob positions in frame 1 / frame 2
    cams = [  # (position, yaw, pitch, roll) -- each looking roughly at the grid
        ((0.0, 0.0, 12.0), 0.0, 0.0, 0.0),
        ((12.0, 0.0, 0.0), 0.0, 90.0, 0.0),
        ((0.0, -12.0, 0.0), 0.0, 0.0, 90.0),
    ]
    for pos, yaw, pitch, roll in cams:
        rot = geometry.rotation_rtsp(yaw, pitch, roll)
        fx, fy = geometry.focal_lengths(W, H, 60.0)
        diff = np.abs(blob_image(*project(a, pos, rot, fx, fy)) - blob_image(*project(b, pos, rot, fx, fy)))
        raycast.cast_frame(grid, diff, 5.0, pos, rot, fx, fy, center, vs, 0.0)

    def world(ijk):
        return center - np.array(shape) * vs / 2 + (np.array(ijk) + 0.5) * vs

    # The two brightest well-separated peaks are the two blob positions.
    peak1 = np.unravel_index(np.argmax(grid), shape)
    masked = grid.copy()
    i, j, k = peak1
    masked[max(i - 4, 0):i + 5, max(j - 4, 0):j + 5, max(k - 4, 0):k + 5] = 0
    peak2 = np.unravel_index(np.argmax(masked), shape)
    found = sorted([world(peak1), world(peak2)], key=lambda p: p[0])
    expected = sorted([a, b], key=lambda p: p[0])
    for f, e in zip(found, expected):
        assert np.linalg.norm(f - e) < 2 * vs, (f, e)
