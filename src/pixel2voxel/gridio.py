"""Voxel grid file and wire formats.

Two .bin layouts exist and both are kept:
  * grid format  (p2v-rtsp):  int32 Nx, Ny, Nz, float32 voxel_size, data
  * cube format  (p2v-batch):     int32 N, float32 voxel_size, N^3 data
Data is float32, C order, indexed [ix, iy, iz] (iz fastest).

ZMQ live stream (p2v-rtsp -> p2v-view): a two-part message, part 1 is the
16-byte grid header, part 2 is the zlib-compressed float32 data.
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

import numpy as np

_GRID_HDR = struct.Struct("<iiif")
_CUBE_HDR = struct.Struct("<if")


def write_grid_bin(path: str | Path, grid: np.ndarray, voxel_size: float) -> None:
    nx, ny, nz = grid.shape
    with open(path, "wb") as f:
        f.write(_GRID_HDR.pack(nx, ny, nz, voxel_size))
        f.write(np.ascontiguousarray(grid, dtype=np.float32).tobytes())


def write_cube_bin(path: str | Path, grid: np.ndarray, voxel_size: float) -> None:
    n = grid.shape[0]
    if grid.shape != (n, n, n):
        raise ValueError("cube format needs an N x N x N grid")
    with open(path, "wb") as f:
        f.write(_CUBE_HDR.pack(n, voxel_size))
        f.write(np.ascontiguousarray(grid, dtype=np.float32).tobytes())


def read_bin(path: str | Path) -> tuple[np.ndarray, float]:
    """Read either .bin layout; the layout is recognised from the file size."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(16)
    if len(head) >= 16:
        nx, ny, nz, vs = _GRID_HDR.unpack(head)
        if nx > 0 and ny > 0 and nz > 0 and size == 16 + 4 * nx * ny * nz:
            data = np.fromfile(path, dtype=np.float32, offset=16)
            return data.reshape(nx, ny, nz), float(vs)
    n, vs = _CUBE_HDR.unpack(head[:8])
    if n > 0 and size == 8 + 4 * n**3:
        data = np.fromfile(path, dtype=np.float32, offset=8)
        return data.reshape(n, n, n), float(vs)
    raise ValueError(f"{path}: not a recognised voxel grid file ({size} bytes)")


def encode_zmq(grid: np.ndarray, voxel_size: float) -> tuple[bytes, bytes]:
    nx, ny, nz = grid.shape
    meta = _GRID_HDR.pack(nx, ny, nz, voxel_size)
    # Level 1 == Z_BEST_SPEED. zlib releases the GIL while compressing.
    # Compress straight from the array's buffer: .tobytes() would add a full copy
    # (500 MB at the default 500^3 grid).
    return meta, zlib.compress(memoryview(np.ascontiguousarray(grid, dtype=np.float32)), 1)


def decode_zmq(meta: bytes, data: bytes) -> tuple[np.ndarray, float]:
    nx, ny, nz, vs = _GRID_HDR.unpack(meta)
    grid = np.frombuffer(zlib.decompress(data), dtype=np.float32).reshape(nx, ny, nz)
    return grid, float(vs)
