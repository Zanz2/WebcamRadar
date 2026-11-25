"""Measure what actually matters for reconstruction: do the cameras' rays meet?

Put the board somewhere NEW (move it, raise it on a box, hold it up) and take a
still from every camera. Poses solved from a board are consistent with that same
placement almost by construction, so checking against it would only confirm
itself.

For every board corner seen by 2+ cameras, each camera's ray is built with the
pipeline's own ray model (undistort -> new_K pinhole -> yaw/pitch/roll), the
corner is triangulated, and we report:
  * ray miss -- distance from the triangulated point to each camera's ray. Rays
    from different cameras must pass within about one voxel of each other for
    the voxel grid to form a peak.
  * spacing  -- distance between neighbouring triangulated corners vs. the real
    square size; this measures scale/shape and doesn't depend on the world origin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import geometry
from ..config import CameraInfo
from .board import BoardDetector
from .model import Lens, matrix_from_euler, world_rays


def camera_rays(cam: CameraInfo, width: int, height: int, pixels) -> tuple[np.ndarray, np.ndarray]:
    """World rays for distorted pixels, exactly as p2v-rtsp would cast them."""
    R = matrix_from_euler(cam.yaw, cam.pitch, cam.roll)
    if cam.intrinsics:
        return world_rays(Lens.from_dict(cam.intrinsics).for_size(width, height), cam.camera_position, R, pixels)
    fx, fy = (float(v) for v in geometry.focal_lengths(width, height, cam.fov_degrees))
    p = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    d = np.stack([(p[:, 0] - width / 2) / fx, -(p[:, 1] - height / 2) / fy, -np.ones(len(p))], axis=1) @ R.T
    return np.asarray(cam.camera_position, dtype=np.float64), d / np.linalg.norm(d, axis=1, keepdims=True)


def triangulate(origins: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares point closest to all rays; returns (point, distance to each ray)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    x = np.linalg.solve(A, b)
    miss = np.array([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (x - o)) for o, d in zip(origins, dirs)])
    return x, miss


@dataclass
class CheckReport:
    corners: int  # corners seen by >= 2 cameras
    per_camera: dict[int, int]  # corners detected per camera
    miss_median_m: float
    miss_p95_m: float
    miss_max_m: float
    spacing_err_median_pct: float
    spacing_err_max_pct: float
    points: dict[int, np.ndarray]

    def summary(self, voxel_size: float | None = None) -> str:
        def fmt(m):
            s = f"{m * 1000:.1f} mm"
            return s + (f" ({m / voxel_size:.2f} voxels)" if voxel_size else "")
        lines = [f"corners seen by 2+ cameras: {self.corners}   (per camera: {self.per_camera})",
                 f"ray miss  median {fmt(self.miss_median_m)}   95% {fmt(self.miss_p95_m)}   max {fmt(self.miss_max_m)}",
                 f"square spacing error  median {self.spacing_err_median_pct:.2f}%   max {self.spacing_err_max_pct:.2f}%"]
        if voxel_size:
            verdict = ("GOOD: rays meet within a voxel" if self.miss_p95_m <= voxel_size else
                       "MARGINAL: rays miss by 1-2 voxels" if self.miss_p95_m <= 2 * voxel_size else
                       "POOR: rays miss by several voxels; recalibrate (or use bigger voxels)")
            lines.append(verdict)
        return "\n".join(lines)


def check(cameras: list[CameraInfo], images: dict[int, np.ndarray], det: BoardDetector) -> CheckReport:
    by_id: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    per_cam = {}
    cams = {c.camera_index: c for c in cameras}
    for ci, img in images.items():
        corners, ids = det.detect_upscaled(img)
        per_cam[ci] = len(ids)
        if len(ids) == 0:
            continue
        o, d = camera_rays(cams[ci], img.shape[1], img.shape[0], corners)
        for k, cid in enumerate(ids):
            by_id.setdefault(int(cid), []).append((o, d[k]))
    points, misses = {}, []
    for cid, rays in by_id.items():
        if len(rays) < 2:
            continue
        x, miss = triangulate(np.array([r[0] for r in rays]), np.array([r[1] for r in rays]))
        points[cid] = x
        misses.extend(miss.tolist())
    if not points:
        raise RuntimeError(f"no board corner was seen by two cameras (corners per camera: {per_cam})")
    # Neighbouring corners along rows and columns are exactly one square apart.
    nx = det.spec.squares_x - 1
    sq = det.board.getSquareLength()
    errs = []
    for cid, x in points.items():
        for nb in ((cid + 1) if (cid + 1) % nx else None, cid + nx):
            if nb is not None and nb in points:
                errs.append(abs(np.linalg.norm(points[nb] - x) - sq) / sq * 100)
    m = np.array(misses)
    e = np.array(errs) if errs else np.array([np.nan])
    return CheckReport(len(points), per_cam, float(np.median(m)), float(np.percentile(m, 95)), float(m.max()),
                       float(np.median(e)), float(np.max(e)), points)
