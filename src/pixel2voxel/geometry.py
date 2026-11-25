"""Camera rotation matrices and intrinsics.

All math is done in float32 because
float64 changes DDA tie-breaks and therefore which voxels a ray hits.
"""

from __future__ import annotations

import math

import numpy as np

F32 = np.float32
_PI = F32(3.14159265358979323846)


# Trig is evaluated in double and rounded to float32, which is portable and deterministic.
def _f32(fn):
    return lambda x: F32(fn(float(x)))


_cosf, _sinf, _tanf, _atanf = (_f32(f) for f in (math.cos, math.sin, math.tan, math.atan))


def deg2rad(deg: float) -> np.float32:
    return F32(deg) * _PI / F32(180.0)


def _rz(a: np.float32) -> np.ndarray:
    c, s = _cosf(a), _sinf(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=F32)


def _ry(a: np.float32) -> np.ndarray:
    c, s = _cosf(a), _sinf(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=F32)


def _rx(a: np.float32) -> np.ndarray:
    c, s = _cosf(a), _sinf(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=F32)


def _matmul3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Fixed summation order, in float32.
    out = np.empty((3, 3), dtype=F32)
    for r in range(3):
        for c in range(3):
            out[r, c] = a[r, 0] * b[0, c] + a[r, 1] * b[1, c] + a[r, 2] * b[2, c]
    return out


def rotation_rtsp(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll) -- convention of the live tool."""
    return _matmul3(_matmul3(_rz(deg2rad(yaw_deg)), _ry(deg2rad(pitch_deg))), _rx(deg2rad(roll_deg)))


def rotation_batch(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(roll) @ Rx(pitch) -- convention of the batch tool.

    Note pitch and roll are about swapped axes compared to `rotation_rtsp`. The two
    tools use different conventions on purpose, so existing data keeps its meaning.
    """
    return _matmul3(_matmul3(_rz(deg2rad(yaw_deg)), _ry(deg2rad(roll_deg))), _rx(deg2rad(pitch_deg)))


def focal_lengths(width: int, height: int, fov_h_deg: float) -> tuple[np.float32, np.float32]:
    """(fx, fy) in pixels from a horizontal FOV."""
    fov_h = deg2rad(fov_h_deg)
    half_tan = _tanf(fov_h * F32(0.5))
    fx = (F32(width) * F32(0.5)) / half_tan
    aspect = F32(width) / F32(height)
    fov_v = F32(2.0) * _atanf((F32(1.0) / aspect) * half_tan)
    fy = (F32(height) * F32(0.5)) / _tanf(fov_v * F32(0.5))
    return F32(fx), F32(fy)
