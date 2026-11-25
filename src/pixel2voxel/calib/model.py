"""Lens model + conversions between OpenCV's camera frame and the pipeline's.

Coordinate frames
-----------------
OpenCV camera:    x right, y DOWN, z FORWARD (out of the lens).
Pipeline camera:  x right, y UP,   looks along -z (the ray model).
They differ by F = diag(1, -1, -1).

solvePnP gives  X_cv = R_cv @ X_world + t.  The pipeline needs the world-from-camera
rotation R (so that d_world = R @ d_cam) and the camera centre C:

    R = R_cv.T @ F        C = -R_cv.T @ t

The pipeline stores R as yaw/pitch/roll with R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
(geometry.rotation_rtsp), so calibrated configs stay hand-editable.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

F = np.diag([1.0, -1.0, -1.0])
MODELS = ("standard", "fisheye")


# ----------------------------------------------------------------------------- rotations


def euler_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    R = np.asarray(R, dtype=np.float64)
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    if abs(R[2, 0]) < 1.0 - 1e-9:
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
    else:  # gimbal lock (pitch = +-90): only yaw - roll / yaw + roll is defined; put it all in yaw
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def matrix_from_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    cy, sy, cp, sp, cr, sr = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def pose_cv_to_pipeline(rvec, tvec) -> tuple[np.ndarray, np.ndarray]:
    """solvePnP (rvec, tvec) -> (camera centre C, world-from-camera R in pipeline axes)."""
    R_cv, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    return -R_cv.T @ t, R_cv.T @ F


def pose_pipeline_to_cv(C, R) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pose_cv_to_pipeline: (C, R) -> (rvec, tvec) for cv2.projectPoints."""
    R_cv = (np.asarray(R, dtype=np.float64) @ F).T
    t = -R_cv @ np.asarray(C, dtype=np.float64).reshape(3)
    rvec, _ = cv2.Rodrigues(R_cv)
    return rvec.reshape(3), t


# ----------------------------------------------------------------------------- lens


@dataclass
class Lens:
    """A calibrated lens and the undistorted ("rectified") pinhole camera it maps to.

    `balance` (0..1) trades field of view against centre resolution when
    undistorting: 0 crops to valid pixels only, 1 keeps every source pixel.
    Rays must use `new_K` -- the matrix frames are undistorted INTO -- not `K`.
    """

    model: str
    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray
    balance: float = 0.5
    new_K: np.ndarray = field(init=False)
    _maps: tuple | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"unknown lens model {self.model!r} (use one of {MODELS})")
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).ravel()
        size = (self.width, self.height)
        if self.model == "fisheye":
            self.dist = self.dist[:4]
            self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.dist.reshape(4, 1), size, np.eye(3), balance=self.balance)
        else:
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, size, self.balance, size)

    # -- serialisation ----------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Lens":
        return cls(d["model"], int(d["width"]), int(d["height"]), d["K"], d["dist"], float(d.get("balance", 0.5)))

    def to_dict(self) -> dict:
        return {"model": self.model, "width": self.width, "height": self.height,
                "K": self.K.round(6).tolist(), "dist": self.dist.round(9).tolist(), "balance": self.balance}

    # -- resolution checks --------------------------------------------------------
    def for_size(self, width: int, height: int) -> "Lens":
        """This lens for a stream of another resolution.

        Same size: itself. Same aspect ratio: K scaled, with a warning (only safe
        if the stream is a pure rescale, not a different sensor crop). Different
        aspect ratio: refused -- calibrate on the stream you will run.
        """
        if (width, height) == (self.width, self.height):
            return self
        if abs(width / height - self.width / self.height) > 1e-3:
            raise ValueError(
                f"stream is {width}x{height} but the lens was calibrated at {self.width}x{self.height} "
                "(different aspect ratio). Calibrate on the same stream (main/sub) you run with.")
        s = width / self.width
        print(f"WARNING: stream is {width}x{height}, lens calibrated at {self.width}x{self.height}; "
              "scaling the calibration. Recalibrate on this stream if the sub-stream crops the sensor.",
              file=sys.stderr)
        K = self.K.copy()
        K[:2] *= s
        return Lens(self.model, width, height, K, self.dist, self.balance)

    # -- images ---------------------------------------------------------------------
    def undistort(self, img: np.ndarray) -> np.ndarray:
        """Undistort a frame into the new_K pinhole image (same size)."""
        if self._maps is None:
            size = (self.width, self.height)
            if self.model == "fisheye":
                self._maps = cv2.fisheye.initUndistortRectifyMap(
                    self.K, self.dist.reshape(4, 1), np.eye(3), self.new_K, size, cv2.CV_32FC1)
            else:
                self._maps = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.new_K, size, cv2.CV_32FC1)
            # Fixed-point maps (1/32 px) remap roughly twice as fast as float maps.
            self._maps = cv2.convertMaps(self._maps[0], self._maps[1], cv2.CV_16SC2)
        return cv2.remap(img, self._maps[0], self._maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # -- points ---------------------------------------------------------------------
    def normalized(self, pts) -> np.ndarray:
        """Distorted pixel coords -> ideal normalized OpenCV coords (x/z, y/z), shape (N, 2)."""
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if self.model == "fisheye":
            out = cv2.fisheye.undistortPoints(p, self.K, self.dist.reshape(4, 1))
        else:
            out = cv2.undistortPoints(p, self.K, self.dist)
        return out.reshape(-1, 2)

    def undistort_points(self, pts) -> np.ndarray:
        """Distorted pixel coords -> pixel coords in the undistorted (new_K) image."""
        n = self.normalized(pts)
        return n * [self.new_K[0, 0], self.new_K[1, 1]] + [self.new_K[0, 2], self.new_K[1, 2]]

    def project(self, X_cv: np.ndarray) -> np.ndarray:
        """Points in the OpenCV camera frame -> distorted pixel coords, shape (N, 2)."""
        X = np.asarray(X_cv, dtype=np.float64).reshape(-1, 1, 3)
        z = np.zeros(3)
        if self.model == "fisheye":
            out, _ = cv2.fisheye.projectPoints(X, z, z, self.K, self.dist.reshape(4, 1))
        else:
            out, _ = cv2.projectPoints(X, z, z, self.K, self.dist)
        return out.reshape(-1, 2)

    @property
    def ray_intrinsics(self) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy) for the kernel -- of new_K, since rays are cast in the undistorted image."""
        k = self.new_K
        return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])


def world_rays(lens: Lens, C, R, pixels) -> tuple[np.ndarray, np.ndarray]:
    """Distorted pixels -> (origin, unit directions[N, 3]) in world coordinates.

    Uses the pipeline's own ray model ((u-cx)/fx, -(v-cy)/fy, -1) in the
    undistorted image, then R -- i.e. exactly what p2v-rtsp casts.
    """
    fx, fy, cx, cy = lens.ray_intrinsics
    uv = lens.undistort_points(pixels)
    d_cam = np.stack([(uv[:, 0] - cx) / fx, -(uv[:, 1] - cy) / fy, -np.ones(len(uv))], axis=1)
    d = d_cam @ np.asarray(R, dtype=np.float64).T
    return np.asarray(C, dtype=np.float64), d / np.linalg.norm(d, axis=1, keepdims=True)
