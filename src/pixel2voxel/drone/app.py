"""Drone detection app.

Note: nothing feeds this app real voxel data. In
normal mode it re-processes an internal grid that stays empty; only test mode
(a simulated drone flying a circle) produces detections. Connecting it to the
p2v-rtsp grid would need a data link plus reconciling the grids' world frames
(the RTSP grid is centred on `center`, this one starts at the origin).

Config file: simple `key = value` lines, '#' comments:
  grid_size_x/y/z, voxel_size, min_cluster_size, max_cluster_size,
  cluster_tolerance, origin_lat, origin_lon, atak_server_ip, atak_server_port,
  update_interval, test_mode

Usage:  p2v-drone [drone_detector.conf]
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

from .cot import CoTMessageGenerator, CoTNetworkHandler, Detection
from .detector import VoxelDroneDetector


@dataclass
class Config:
    grid_size_x: int = 100
    grid_size_y: int = 100
    grid_size_z: int = 50
    voxel_size: float = 0.5
    min_cluster_size: float = 5.0
    max_cluster_size: float = 100.0
    cluster_tolerance: float = 1.0
    origin_lat: float = 38.8895
    origin_lon: float = -77.0352
    atak_server_ip: str = "127.0.0.1"
    atak_server_port: int = 8087
    update_interval: int = 1
    test_mode: bool = False

    def load(self, path: str | Path) -> bool:
        try:
            lines = Path(path).read_text().splitlines()
        except OSError:
            print(f"Could not open config file: {path}", file=sys.stderr)
            return False
        types = {f.name: f.type for f in fields(self)}
        for line in lines:
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = (s.strip() for s in line.split("=", 1))
            if key not in types:
                continue
            kind = types[key]
            if kind == "bool":
                setattr(self, key, value in ("true", "1"))
            elif kind == "int":
                setattr(self, key, int(value))
            elif kind == "float":
                setattr(self, key, float(value))
            else:
                setattr(self, key, value)
        return True

    def print(self) -> None:
        print("=== Configuration ===")
        print(f"Voxel Grid: {self.grid_size_x}x{self.grid_size_y}x{self.grid_size_z} (voxel size: {self.voxel_size}m)")
        print(f"Cluster Parameters: min={self.min_cluster_size}, max={self.max_cluster_size}, "
              f"tolerance={self.cluster_tolerance}")
        print(f"Position Origin: {self.origin_lat}, {self.origin_lon}")
        print(f"ATAK Server: {self.atak_server_ip}:{self.atak_server_port} (update interval: {self.update_interval}s)")
        print(f"Test Mode: {'On' if self.test_mode else 'Off'}")


class DroneSimulator:
    """A ~19-voxel blob flying a circle through the grid."""

    def __init__(self):
        self.t = 0.0

    def step(self, det: VoxelDroneDetector) -> None:
        self.t += 0.1
        sx, sy, sz = det.grid_size
        det.clear_voxel_grid()
        cx, cy, cz = sx / 2.0, sy / 2.0, sz / 2.0
        radius = min(sx, sy) / 4.0
        px = cx + radius * math.cos(self.t)
        py = cy + radius * math.sin(self.t)
        pz = cz + 5.0 * math.sin(self.t / 2.0)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-1, 2):
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist > 2.5:
                        continue
                    det.set_voxel_value(int(px + dx), int(py + dy), int(pz + dz), max(0.1, 1.0 - dist / 3.0))
        det.process_voxel_grid()


def build(config: Config) -> VoxelDroneDetector:
    det = VoxelDroneDetector(config.grid_size_x, config.grid_size_y, config.grid_size_z)
    det.set_voxel_size(config.voxel_size)
    det.set_cluster_parameters(config.min_cluster_size, config.max_cluster_size, config.cluster_tolerance)
    return det


def to_cot_detections(drones) -> list[Detection]:
    return [Detection(d.x, d.y, d.z, d.confidence, 1, d.uuid, d.detection_time) for d in drones]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p2v-drone", description=__doc__.split("\n\n")[0])
    ap.add_argument("config", nargs="?", default="drone_detector.conf", help="key = value config file")
    args = ap.parse_args(argv)
    config = Config()
    if not config.load(args.config):
        print("Using default configuration")
    config.print()

    det = build(config)
    gen = CoTMessageGenerator("drone-detector-01")
    gen.set_sensor_location(config.origin_lat, config.origin_lon, 0.0)
    sender = CoTNetworkHandler(config.atak_server_ip, config.atak_server_port)
    sender.start()

    running = True

    def on_sigint(signum, _frame):
        nonlocal running
        print(f"\nInterrupt signal ({signum}) received. Shutting down...")
        running = False

    signal.signal(signal.SIGINT, on_sigint)
    print("Drone detection system started.\nPress Ctrl+C to exit.")

    sim = DroneSimulator()
    last_send = time.monotonic()
    while running:
        if config.test_mode:
            sim.step(det)
        else:
            det.process_voxel_grid()
        drones = det.get_detected_drones()
        print(f"\rDetected drones: {len(drones)}   ", end="", flush=True)
        now = time.monotonic()
        if now - last_send >= config.update_interval and drones:
            msgs = gen.generate_cot_messages(to_cot_detections(drones))
            sender.queue_messages(msgs)
            print(f" [Sent {len(msgs)} CoT]", end="", flush=True)
            last_send = now
        time.sleep(0.1)

    print("\nShutting down...")
    sender.stop()
    print("Drone detection system stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
