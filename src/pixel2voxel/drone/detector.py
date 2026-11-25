"""Cluster bright voxels into drones and track them.

The grid is a numpy array indexed [x, y, z]. Voxels are visited in a fixed
order (z outer, then y, then x), and the BFS adds neighbours in that same
global order, so clusters, their member order, and drone ids are reproducible.
The neighbour search uses a spatial hash.

Velocity uses the previous position and time, captured before cluster assignment
overwrites `last_seen`. (It is not used in the CoT output.)
"""

from __future__ import annotations

import math
import secrets
import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VoxelPoint:
    x: int
    y: int
    z: int
    value: float


@dataclass
class VoxelDrone:
    id: int
    uuid: str = ""
    voxels: list[VoxelPoint] = field(default_factory=list)
    total_intensity: float = 0.0
    max_intensity: float = 0.0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)  # voxel space
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)  # world (m)
    uncertainty: float = 10.0
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_valid: bool = False
    last_seen: float = field(default_factory=time.time)
    has_position: bool = False
    # (position, last_seen, has_position) captured before each assignment pass
    prev: tuple | None = field(default=None, repr=False)

    def __post_init__(self):
        if not self.uuid:
            millis = int(time.time() * 1000)
            self.uuid = f"drone-{self.id}-{millis}-{secrets.token_hex(4)}"

    def update_position(self, voxel_size: float, grid_min) -> None:
        if not self.voxels:
            return
        w = np.array([v.value for v in self.voxels], dtype=np.float64)
        xyz = np.array([(v.x, v.y, v.z) for v in self.voxels], dtype=np.float64)
        self.max_intensity = float(w.max())
        self.total_intensity = float(w.sum())
        sw = w.sum()
        if sw > 1e-6:
            c = (xyz * w[:, None]).sum(axis=0) / sw
            var = ((w / sw)[:, None] * (xyz - c) ** 2).sum(axis=0)
            self.uncertainty = voxel_size * math.sqrt(var.sum() / 3.0)
        else:
            c = xyz[0]
            self.uncertainty = voxel_size
        self.center = tuple(float(v) for v in c)
        self.position = tuple(float(grid_min[i] + (c[i] + 0.5) * voxel_size) for i in range(3))
        self.has_position = True

    def update_velocity(self, prev_pos, prev_time: float) -> None:
        dt = time.time() - prev_time
        if dt > 0.01:
            self.velocity = tuple((self.position[i] - prev_pos[i]) / dt for i in range(3))
            self.velocity_valid = True
        else:
            self.velocity_valid = False

    @property
    def speed(self) -> float:
        return math.sqrt(sum(v * v for v in self.velocity)) if self.velocity_valid else 0.0


@dataclass
class DroneDetection:
    id: int
    uuid: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    velocity_valid: bool
    uncertainty: float
    confidence: float  # max voxel intensity
    detection_time: float

    @classmethod
    def from_drone(cls, d: VoxelDrone) -> "DroneDetection":
        return cls(d.id, d.uuid, *d.position, *d.velocity, d.velocity_valid, d.uncertainty,
                   d.max_intensity, d.last_seen)


class VoxelDroneDetector:
    def __init__(self, size_x=100, size_y=100, size_z=50, threshold=0.5, cluster_dist=1.5,
                 min_voxels=5, max_voxels=1000, inactive_timeout_seconds=30):
        self.grid_size = (size_x, size_y, size_z)
        self.grid = np.zeros(self.grid_size, dtype=np.float32)
        self.grid_min = (0.0, 0.0, 0.0)
        self.voxel_size = 0.0
        self.detection_threshold = threshold
        self.cluster_dist_sq = cluster_dist * cluster_dist
        self.min_voxels = min_voxels
        self.max_voxels = max_voxels
        self.max_inactive = inactive_timeout_seconds
        self.drones: list[VoxelDrone] = []
        self._next_id = 1
        self._lock = threading.Lock()

    # -- configuration --------------------------------------------------------
    def set_grid_parameters(self, sx, sy, sz, voxel_size, min_coords) -> None:
        self.grid_size = (sx, sy, sz)
        self.voxel_size = voxel_size
        self.grid_min = tuple(min_coords)
        self.grid = np.zeros(self.grid_size, dtype=np.float32)

    def set_voxel_size(self, size_m: float) -> None:
        self.voxel_size = size_m

    def set_cluster_parameters(self, min_size: float, max_size: float, tolerance: float) -> None:
        self.min_voxels = int(min_size)
        self.max_voxels = int(max_size)
        self.cluster_dist_sq = tolerance * tolerance

    def set_voxel_value(self, x: int, y: int, z: int, value: float) -> None:
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and 0 <= z < self.grid_size[2]:
            self.grid[x, y, z] = value

    def clear_voxel_grid(self) -> None:
        self.grid.fill(0.0)

    # -- processing -----------------------------------------------------------
    def process_voxel_grid(self, grid: np.ndarray | None = None) -> None:
        if grid is not None:
            if grid.shape != self.grid_size:
                raise ValueError(f"voxel grid shape {grid.shape} != {self.grid_size}")
            self.grid = np.asarray(grid, dtype=np.float32)
        # (z, y, x) lexicographic order (iz outer, ix inner).
        zyx = np.argwhere(self.grid.transpose(2, 1, 0) >= self.detection_threshold)
        points = [VoxelPoint(int(x), int(y), int(z), float(self.grid[x, y, z])) for z, y, x in zyx]
        with self._lock:
            clusters = self._cluster(points)
            self._assign(clusters)
            self._update_properties()
            self._remove_inactive()

    def _cluster(self, points: list[VoxelPoint]) -> list[list[VoxelPoint]]:
        index = {(p.x, p.y, p.z): i for i, p in enumerate(points)}
        r = int(math.floor(math.sqrt(self.cluster_dist_sq)))
        offsets = [(dx, dy, dz) for dx in range(-r, r + 1) for dy in range(-r, r + 1) for dz in range(-r, r + 1)
                   if (dx, dy, dz) != (0, 0, 0) and dx * dx + dy * dy + dz * dz <= self.cluster_dist_sq]
        visited = bytearray(len(points))
        clusters = []
        for start, p in enumerate(points):
            if visited[start]:
                continue
            visited[start] = 1
            cluster, queue, head = [p], [start], 0
            while head < len(queue):
                c = points[queue[head]]
                head += 1
                found = []
                for dx, dy, dz in offsets:
                    j = index.get((c.x + dx, c.y + dy, c.z + dz))
                    if j is not None and not visited[j]:
                        found.append(j)
                for j in sorted(found):  # fixed discovery order
                    visited[j] = 1
                    queue.append(j)
                    cluster.append(points[j])
            if self.min_voxels <= len(cluster) <= self.max_voxels:
                clusters.append(cluster)
        return clusters

    def _assign(self, clusters: list[list[VoxelPoint]]) -> None:
        now = time.time()
        assigned = [False] * len(clusters)
        centroids = [tuple(sum(getattr(p, a) for p in c) / len(c) for a in "xyz") for c in clusters]
        match_sq = 4.0 * self.cluster_dist_sq
        for drone in self.drones:
            # Remember the previous state for the velocity estimate (see module docstring).
            drone.prev = (drone.position, drone.last_seen, drone.has_position)
            dc = [(drone.position[i] - self.grid_min[i]) / self.voxel_size - 0.5 if self.voxel_size else 0.0
                  for i in range(3)]
            best, best_d = -1, math.inf
            for i, cc in enumerate(centroids):
                if assigned[i]:
                    continue
                d = sum((dc[k] - cc[k]) ** 2 for k in range(3))
                if d < best_d and d < match_sq:
                    best, best_d = i, d
            if best != -1:
                drone.voxels = clusters[best]
                drone.last_seen = now
                assigned[best] = True
            else:
                drone.voxels = []
        for i, c in enumerate(clusters):
            if not assigned[i]:
                d = VoxelDrone(self._next_id, voxels=c, last_seen=now)
                d.prev = (d.position, now, False)
                self._next_id += 1
                self.drones.append(d)

    def _update_properties(self) -> None:
        for d in self.drones:
            if d.voxels:
                prev_pos, prev_time, had_pos = d.prev
                d.update_position(self.voxel_size, self.grid_min)
                if had_pos:
                    d.update_velocity(prev_pos, prev_time)

    def _remove_inactive(self) -> None:
        now = time.time()
        self.drones = [d for d in self.drones if now - d.last_seen <= self.max_inactive]

    # -- queries --------------------------------------------------------------
    def get_detected_drones(self) -> list[DroneDetection]:
        now = time.time()
        with self._lock:
            return [DroneDetection.from_drone(d) for d in self.drones
                    if d.voxels or now - d.last_seen < self.max_inactive]

    def voxel_to_world(self, x: int, y: int, z: int) -> tuple[float, float, float]:
        return tuple(self.grid_min[i] + (v + 0.5) * self.voxel_size for i, v in enumerate((x, y, z)))

    def world_to_voxel(self, wx: float, wy: float, wz: float) -> tuple[int, int, int]:
        out = []
        for i, w in enumerate((wx, wy, wz)):
            v = int((w - self.grid_min[i]) / self.voxel_size)
            out.append(max(0, min(self.grid_size[i] - 1, v)))
        return tuple(out)
