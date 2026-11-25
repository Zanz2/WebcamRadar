"""Render synthetic views of the ChArUco board through a known lens and pose.

Used by the tests and by `p2v-calibrate demo`, so the whole workflow can be
tried (and verified against ground truth) without any camera.

Rendering is done backwards: for every output pixel, take its ray through the
lens (with distortion), intersect it with the board plane, and sample the
board pattern there. This works for any field of view, including fisheye.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .board import BoardSpec
from .model import Lens


def render_board_view(spec: BoardSpec, lens: Lens, rvec, tvec, supersample: int = 2,
                      px_per_square: int = 160, background: int = 128, noise: float = 0.0,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Grayscale image of the board (OpenCV board frame, metres) seen by `lens` at pose (rvec, tvec)."""
    sq = spec.square_mm / 1000.0
    margin = px_per_square // 2  # white quiet zone around the pattern
    pattern = spec.cv_board().generateImage((spec.squares_x * px_per_square + 2 * margin,
                                             spec.squares_y * px_per_square + 2 * margin), marginSize=margin)
    w, h, ss = lens.width, lens.height, supersample
    # Supersampled output pixel centres, expressed in 1x pixel coordinates.
    u = (np.arange(w * ss) + 0.5) / ss - 0.5
    v = (np.arange(h * ss) + 0.5) / ss - 0.5
    uu, vv = np.meshgrid(u, v)
    n = lens.normalized(np.stack([uu.ravel(), vv.ravel()], axis=1))
    rays = np.concatenate([n, np.ones((len(n), 1))], axis=1)  # camera-frame directions (x/z, y/z, 1)

    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    H = np.column_stack([R[:, 0], R[:, 1], np.asarray(tvec, dtype=np.float64).reshape(3)])
    b = rays @ np.linalg.inv(H).T  # (x, y, 1) * lambda on the board plane
    lam = b[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        bx, by = b[:, 0] / lam, b[:, 1] / lam
    valid = lam > 0  # plane is in front of the camera
    px = np.where(valid, bx / sq * px_per_square + margin - 0.5, -1e6).astype(np.float32)
    py = np.where(valid, by / sq * px_per_square + margin - 0.5, -1e6).astype(np.float32)
    img = cv2.remap(pattern, px.reshape(uu.shape), py.reshape(uu.shape), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    if ss > 1:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if noise > 0:
        rng = rng or np.random.default_rng()
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def random_calibration_poses(spec: BoardSpec, lens: Lens, n: int, rng: np.random.Generator,
                             fill=(0.25, 0.5), max_tilt=40.0, edge=0.1):
    """Board poses (rvec, tvec) spread over the image, at varied tilts and distances.

    `fill` is the board's width as a fraction of the image width -- what a person
    should aim for when holding the board (a wide lens needs the board close).
    """
    c_obj = np.array([spec.squares_x, spec.squares_y, 0.0]) * spec.square_mm / 2000.0
    board_w = spec.squares_x * spec.square_mm / 1000.0
    poses = []
    while len(poses) < n:
        target = rng.uniform([edge * lens.width, edge * lens.height],
                             [(1 - edge) * lens.width, (1 - edge) * lens.height])
        nrm = lens.normalized(target[None])[0]
        dist = board_w * lens.K[0, 0] / (rng.uniform(*fill) * lens.width)
        center = np.array([nrm[0], nrm[1], 1.0]) / math.sqrt(1 + nrm[0] ** 2 + nrm[1] ** 2) * dist
        ax, ay, az = (math.radians(a) for a in rng.uniform([-max_tilt, -max_tilt, -20], [max_tilt, max_tilt, 20]))
        Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
        Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
        Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx  # identity = board facing the camera (its z points away from the lens)
        t = center - R @ c_obj
        rvec, _ = cv2.Rodrigues(R)
        poses.append((rvec.ravel(), t))
    return poses
