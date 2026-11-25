"""Voxel DDA ray casting (numba kernels: one ray, and the per-pixel loops).

Every float constant is a float32 global and every int*float product casts the int
first; letting numba promote to float64 would change DDA tie-breaks and therefore
which voxels a ray hits.

The kernels are serial on purpose: serial accumulation keeps the summation order
(and therefore the result) deterministic, and numba's default `workqueue`
threading layer is not safe to call from several Python threads. `nogil=True`
lets the camera-capture threads keep decoding while a frame is being cast.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

F32 = np.float32
_ZERO = F32(0.0)
_ONE = F32(1.0)
_HALF = F32(0.5)
_EPS = F32(1e-12)
_MIN_PIX = F32(1e-3)
_INF = F32(np.inf)

# Ray construction modes for accumulate_frame().
RAY_RTSP = 0  # live: ((u - w/2)/fx, -(v - h/2)/fy, -1), attenuated
RAY_BATCH = 1  # batch: (u - w/2, -(v - h/2), -f), NOT attenuated


@njit(cache=True, inline="always")
def _safe_div(num, den):
    if abs(den) < _EPS:
        return _INF
    return F32(num / den)


@njit(cache=True, nogil=True)
def cast_ray_steps(cam, d, nx, ny, nz, voxel_size, center, out_idx, out_dist):
    """Walk one ray through the grid; write visited voxels into out_idx/out_dist.

    cam, d, center: float32[3] (d already normalized). Returns the step count.
    out_idx must hold at least nx + ny + nz + 3 rows.
    """
    half_x = _HALF * (F32(nx) * voxel_size)
    half_y = _HALF * (F32(ny) * voxel_size)
    half_z = _HALF * (F32(nz) * voxel_size)
    gmin_x = center[0] - half_x
    gmin_y = center[1] - half_y
    gmin_z = center[2] - half_z
    gmax_x = center[0] + half_x
    gmax_y = center[1] + half_y
    gmax_z = center[2] + half_z

    t_min = _ZERO
    t_max = _INF

    # 1) Ray-box intersection (slab method), axis by axis.
    for i in range(3):
        if i == 0:
            origin, dd, mn, mx = cam[0], d[0], gmin_x, gmax_x
        elif i == 1:
            origin, dd, mn, mx = cam[1], d[1], gmin_y, gmax_y
        else:
            origin, dd, mn, mx = cam[2], d[2], gmin_z, gmax_z
        if abs(dd) < _EPS:
            if origin < mn or origin > mx:
                return 0
        else:
            t1 = F32((mn - origin) / dd)
            t2 = F32((mx - origin) / dd)
            t_near = min(t1, t2)
            t_far = max(t1, t2)
            if t_near > t_min:
                t_min = t_near
            if t_far < t_max:
                t_max = t_far
            if t_min > t_max:
                return 0

    if t_min < _ZERO:
        t_min = _ZERO

    # 2) Start voxel. int() truncates toward zero (not floor).
    sx = cam[0] + t_min * d[0]
    sy = cam[1] + t_min * d[1]
    sz = cam[2] + t_min * d[2]
    ix = int(F32((sx - gmin_x) / voxel_size))
    iy = int(F32((sy - gmin_y) / voxel_size))
    iz = int(F32((sz - gmin_z) / voxel_size))
    # A ray entering through a
    # +x/+y/+z face starts exactly on that face, which maps to index N, one past
    # the end. The voxel actually entered is N - 1.
    ix = min(ix, nx - 1)
    iy = min(iy, ny - 1)
    iz = min(iz, nz - 1)
    if ix < 0 or iy < 0 or iz < 0:
        return 0

    # 3) Step direction and initial boundary crossings.
    step_x = 1 if d[0] >= _ZERO else -1
    step_y = 1 if d[1] >= _ZERO else -1
    step_z = 1 if d[2] >= _ZERO else -1

    next_bx = gmin_x + F32(ix + (1 if step_x > 0 else 0)) * voxel_size
    next_by = gmin_y + F32(iy + (1 if step_y > 0 else 0)) * voxel_size
    next_bz = gmin_z + F32(iz + (1 if step_z > 0 else 0)) * voxel_size

    t_max_x = _safe_div(next_bx - cam[0], d[0])
    t_max_y = _safe_div(next_by - cam[1], d[1])
    t_max_z = _safe_div(next_bz - cam[2], d[2])
    t_delta_x = _safe_div(voxel_size, abs(d[0]))
    t_delta_y = _safe_div(voxel_size, abs(d[1]))
    t_delta_z = _safe_div(voxel_size, abs(d[2]))

    t_current = t_min
    n = 0
    cap = out_idx.shape[0]

    # 4) Walk.
    while t_current <= t_max and n < cap:
        out_idx[n, 0] = ix
        out_idx[n, 1] = iy
        out_idx[n, 2] = iz
        out_dist[n] = t_current
        n += 1

        if t_max_x < t_max_y and t_max_x < t_max_z:
            ix += step_x
            t_current = t_max_x
            t_max_x = F32(t_max_x + t_delta_x)
        elif t_max_y < t_max_z:
            iy += step_y
            t_current = t_max_y
            t_max_y = F32(t_max_y + t_delta_y)
        else:
            iz += step_z
            t_current = t_max_z
            t_max_z = F32(t_max_z + t_delta_z)
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
            break
    return n


@njit(cache=True, inline="always")
def _normalize(x, y, z):
    length = F32(math.sqrt(F32(x * x + y * y + z * z)))
    if length < _EPS:
        return _ZERO, _ZERO, _ZERO
    return F32(x / length), F32(y / length), F32(z / length)


@njit(cache=True, nogil=True)
def accumulate_frame(grid, diff, threshold, cam, rot, fx, fy, cx, cy, center, voxel_size, alpha, mode):
    """Cast a ray for every changed pixel of one frame and add into `grid`.

    grid:  float32[Nx, Ny, Nz], C-contiguous, modified in place
    diff:  float32[H, W], absolute frame difference
    cam, center: float32[3];  rot: float32[3, 3]
    fx, fy, cx, cy: pinhole intrinsics in pixels
    mode:  RAY_RTSP or RAY_BATCH (see module constants)
    Returns the number of voxel updates (0 means nothing was written).
    """
    nx, ny, nz = grid.shape
    h, w = diff.shape
    out_idx = np.empty((nx + ny + nz + 3, 3), dtype=np.int64)
    out_dist = np.empty(nx + ny + nz + 3, dtype=np.float32)
    d = np.empty(3, dtype=np.float32)
    writes = 0

    for v in range(h):
        for u in range(w):
            pix_val = diff[v, u]
            if not (pix_val > threshold):
                continue
            if pix_val < _MIN_PIX:
                continue

            if mode == RAY_RTSP:
                x = F32((F32(u) - cx) / fx)
                y = F32(-(F32(v) - cy) / fy)
                z = -_ONE
            else:
                x = F32(u) - cx
                y = -(F32(v) - cy)
                z = -fx
            x, y, z = _normalize(x, y, z)

            wx = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z
            wy = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z
            wz = rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z
            d[0], d[1], d[2] = _normalize(wx, wy, wz)

            n = cast_ray_steps(cam, d, nx, ny, nz, voxel_size, center, out_idx, out_dist)
            for s in range(n):
                if mode == RAY_RTSP:
                    val = pix_val * F32(_ONE / (_ONE + alpha * out_dist[s]))
                else:
                    val = pix_val
                grid[out_idx[s, 0], out_idx[s, 1], out_idx[s, 2]] += val
            writes += n
    return writes


def cast_frame(grid, diff, threshold, cam_pos, rot, fx, fy, center, voxel_size, alpha=0.0, mode=RAY_RTSP,
               cx=None, cy=None) -> int:
    """Typed entry point for accumulate_frame().

    Coerces every scalar/array to float32 first. Without this, passing a Python
    float makes numba compile a float64 specialization with different results.
    cx/cy default to the image centre.
    """
    if grid.dtype != np.float32 or not grid.flags.c_contiguous:
        raise TypeError("grid must be a C-contiguous float32 array")
    h, w = np.shape(diff)
    return accumulate_frame(
        grid,
        np.ascontiguousarray(diff, dtype=np.float32),
        F32(threshold),
        np.asarray(cam_pos, dtype=np.float32),
        np.ascontiguousarray(rot, dtype=np.float32),
        F32(fx),
        F32(fy),
        _HALF * F32(w) if cx is None else F32(cx),
        _HALF * F32(h) if cy is None else F32(cy),
        np.asarray(center, dtype=np.float32),
        F32(voxel_size),
        F32(alpha),
        int(mode),
    )


def cast_ray(cam_pos, direction, shape, voxel_size, center):
    """Python-friendly single-ray DDA: returns (indices[n, 3], distances[n])."""
    nx, ny, nz = shape
    out_idx = np.empty((nx + ny + nz + 3, 3), dtype=np.int64)
    out_dist = np.empty(nx + ny + nz + 3, dtype=np.float32)
    n = cast_ray_steps(
        np.asarray(cam_pos, dtype=np.float32),
        np.asarray(direction, dtype=np.float32),
        nx, ny, nz,
        F32(voxel_size),
        np.asarray(center, dtype=np.float32),
        out_idx, out_dist,
    )
    return out_idx[:n].copy(), out_dist[:n].copy()
