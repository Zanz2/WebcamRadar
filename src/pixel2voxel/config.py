"""Camera/grid configuration.

Accepted layouts:
  * legacy: a bare array of camera objects (grid uses defaults)
  * legacy: a single camera object
  * current: {"grid": {...}, "cameras": [...]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CameraInfo:
    camera_index: int = 0
    rtsp_url: str = ""
    camera_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    fov_degrees: float = 60.0
    # Optional lens calibration written by `p2v-calibrate apply` (see calib/model.py
    # Lens.to_dict). When present it replaces fov_degrees and frames are undistorted.
    intrinsics: dict | None = None


@dataclass
class GridConfig:
    Nx: int = 500
    Ny: int = 500
    Nz: int = 500
    voxel_size: float = 6.0
    center: tuple[float, float, float] = (0.0, 0.0, 500.0)
    decay_rate: float = 1.0  # fraction of voxel value left after 1 s (1.0 = no decay)
    motion_threshold: float = 5.0
    distance_attenuation: float = 0.1
    show_debug: bool = False
    output_dir: str = "."
    save_interval_seconds: int = 30
    # Publishing compresses the whole grid (~500 MB at the default 500^3 size), so it is rate-limited.
    publish_interval_seconds: float = 0.5
    # Optional per-pixel noise threshold (noise.py). 0 = off: motion_threshold
    # is used. When on, it replaces motion_threshold.
    noise_sigma: float = 0.0  # a change must exceed this many sigmas of that pixel's own noise
    noise_floor: float = 8.0  # and at least this many grey levels
    min_cameras: int = 0  # optional camera agreement: 0 or 1 = off (plain sum)
    noise_learn_seconds: float = 5.0  # static scene needed at startup; nothing is cast meanwhile

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.Nx, self.Ny, self.Nz)


@dataclass
class AppConfig:
    cameras: list[CameraInfo] = field(default_factory=list)
    grid: GridConfig = field(default_factory=GridConfig)


def _vec3(value, default):
    if isinstance(value, list) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return default


def parse_camera(entry: dict) -> CameraInfo:
    return CameraInfo(
        camera_index=int(entry.get("camera_index", 0)),
        rtsp_url=str(entry.get("rtsp_url", "")),
        camera_position=_vec3(entry.get("camera_position"), (0.0, 0.0, 0.0)),
        yaw=float(entry.get("yaw", 0.0)),
        pitch=float(entry.get("pitch", 0.0)),
        roll=float(entry.get("roll", 0.0)),
        fov_degrees=float(entry.get("fov_degrees", 60.0)),
        intrinsics=entry.get("intrinsics") if isinstance(entry.get("intrinsics"), dict) else None,
    )


def parse_grid(g: dict) -> GridConfig:
    gc = GridConfig()
    size = g.get("size")
    if isinstance(size, list) and len(size) >= 3:
        gc.Nx, gc.Ny, gc.Nz = int(size[0]), int(size[1]), int(size[2])
    elif isinstance(size, (int, float)) and not isinstance(size, bool):
        gc.Nx = gc.Ny = gc.Nz = int(size)
    gc.voxel_size = float(g.get("voxel_size", gc.voxel_size))
    gc.center = _vec3(g.get("center"), gc.center)
    gc.decay_rate = float(g.get("decay_rate", gc.decay_rate))
    gc.motion_threshold = float(g.get("motion_threshold", gc.motion_threshold))
    gc.distance_attenuation = float(g.get("distance_attenuation", gc.distance_attenuation))
    gc.show_debug = bool(g.get("show_debug", gc.show_debug))
    gc.save_interval_seconds = int(g.get("save_interval_seconds", gc.save_interval_seconds))
    gc.publish_interval_seconds = float(g.get("publish_interval_seconds", gc.publish_interval_seconds))
    gc.min_cameras = int(g.get("min_cameras", gc.min_cameras))
    gc.noise_sigma = float(g.get("noise_sigma", gc.noise_sigma))
    gc.noise_floor = float(g.get("noise_floor", gc.noise_floor))
    gc.noise_learn_seconds = float(g.get("noise_learn_seconds", gc.noise_learn_seconds))
    return gc


def load_config(path: str | Path) -> AppConfig:
    data = json.loads(Path(path).read_text())
    cfg = AppConfig()
    if isinstance(data, list):
        cfg.cameras = [parse_camera(e) for e in data]
    elif isinstance(data, dict):
        if isinstance(data.get("cameras"), list):
            cfg.cameras = [parse_camera(e) for e in data["cameras"]]
        else:
            cfg.cameras = [parse_camera(data)]
        if isinstance(data.get("grid"), dict):
            cfg.grid = parse_grid(data["grid"])
    else:
        raise ValueError(f"{path}: JSON format not recognized")
    return cfg
