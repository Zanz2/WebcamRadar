"""Cursor-on-Target XML generation and TCP delivery to an ATAK/TAK server.

The XML layout (attribute order, quoting, number formatting) is fixed.
"""

from __future__ import annotations

import math
import queue
import random
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

_CLASS_NAMES = {0: "Unknown", 1: "Drone", 2: "Bird", 3: "Other"}
_COT_TYPES = {0: "a-u-G", 1: "a-u-A-M-F-Q-r", 2: "a-n-A-C-F", 3: "a-u-G"}


@dataclass
class Detection:
    x: float  # metres east (+) / west (-) of the sensor
    y: float  # metres north (+) / south (-)
    z: float  # metres up
    confidence: float = 0.0
    classification: int = 1  # 0 unknown, 1 drone, 2 bird, 3 other
    uid: str = ""
    timestamp: float = field(default_factory=time.time)


def format_time(t: float) -> str:
    """ISO 8601 UTC with milliseconds, e.g. 2026-01-01T10:00:00.123Z."""
    dt = datetime.fromtimestamp(t, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{int(t * 1000) % 1000:03d}Z"


class CoTMessageGenerator:
    def __init__(self, sender_uid: str = "", team_name: str = "PIXELVOXEL", team_role: str = "SENSOR"):
        self._rng = random.Random()
        self.sender_uid = sender_uid or f"PixelVoxelSensor-{self._rng.randint(100, 999)}"
        self.team_name = team_name
        self.team_role = team_role
        self.sensor_lat = self.sensor_lon = self.sensor_alt = 0.0

    def set_sensor_location(self, lat: float, lon: float, alt: float) -> None:
        self.sensor_lat, self.sensor_lon, self.sensor_alt = lat, lon, alt

    def local_to_global(self, x_east: float, y_north: float, z_up: float) -> tuple[float, float, float]:
        """Flat-earth approximation around the sensor origin."""
        lat_rad = math.radians(self.sensor_lat)
        m_per_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
        m_per_lon = 111412.84 * math.cos(lat_rad)
        if abs(m_per_lat) < 1e-6 or abs(m_per_lon) < 1e-6:
            return self.sensor_lat, self.sensor_lon, self.sensor_alt + z_up
        return (self.sensor_lat + y_north / m_per_lat,
                self.sensor_lon + x_east / m_per_lon,
                self.sensor_alt + z_up)

    def _generate_uid(self) -> str:
        return f"{self.sender_uid}-{self._rng.randint(100000, 999999)}"

    def generate_cot_message(self, det: Detection, now: float | None = None) -> str:
        lat, lon, alt = self.local_to_global(det.x, det.y, det.z)
        now = time.time() if now is None else now
        t, stale = format_time(now), format_time(now + 60)
        uid = det.uid or self._generate_uid()
        unc = max(1.0, (1.0 / det.confidence) if det.confidence > 0 else 100.0)
        return (
            "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
            f"<event version='2.0' uid='{uid}' type='{_COT_TYPES.get(det.classification, 'a-u-G')}' "
            f"time='{t}' start='{t}' stale='{stale}' how='m-p'>"
            f"<point lat='{lat:.8f}' lon='{lon:.8f}' hae='{alt:.2f}' ce='{unc:.2f}' le='{unc:.2f}'/>"
            "<detail>"
            f"<contact callsign='{_CLASS_NAMES.get(det.classification, 'Unknown')}-{det.uid[:4]}'/>"
            "<precisionlocation altsrc='DTED0' geopointsrc='User'/>"
            "<status battery='100'/>"
            "<track course='0' speed='0'/>"
            "<sensor Fov='60.0'/>"
            "</detail></event>"
        )

    def generate_cot_messages(self, detections: list[Detection]) -> list[str]:
        return [self.generate_cot_message(d) for d in detections]

    def generate_status_message(self, operational: bool = True, now: float | None = None) -> str:
        now = time.time() if now is None else now
        t, stale = format_time(now), format_time(now + 120)
        u = self.sender_uid
        return (
            "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
            f"<event version='2.0' uid='{u}' type='b-m-p-s-p-i' time='{t}' start='{t}' stale='{stale}' how='h-e'>"
            f"<point lat='{self.sensor_lat:.8f}' lon='{self.sensor_lon:.8f}' hae='{self.sensor_alt:.2f}' ce='10' le='10'/>"
            "<detail>"
            f"<contact callsign='{u}'/>"
            f"<remarks>Sensor status: {'Operational' if operational else 'Offline'}.</remarks>"
            f"<link uid='{u}' production_time='{t}' type='a-f-G' parent_callsign='{self.team_name}'/>"
            f"<status readiness='{'true' if operational else 'false'}'/>"
            f"<uid Droid='{u}'/>"
            f"<__group name='{self.team_name}' role='{self.team_role}'/>"
            "</detail></event>"
        )


class CoTNetworkHandler:
    """Background TCP sender with a queue, reconnects and re-queues on failure."""

    def __init__(self, ip: str = "127.0.0.1", port: int = 8087, connect_timeout: float = 5.0):
        self.ip, self.port = ip, port
        self._timeout = connect_timeout
        self._sock: socket.socket | None = None
        self._q: queue.Queue[str] = queue.Queue()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def start(self) -> bool:
        if self._running.is_set():
            return True
        if not self._connect():
            print("Initial connection failed; will keep retrying in the background.", file=sys.stderr)
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="cot-sender", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5)
        self._close()

    def queue_message(self, msg: str) -> None:
        self._q.put(msg)

    def queue_messages(self, msgs: list[str]) -> None:
        for m in msgs:
            self._q.put(m)

    def queue_size(self) -> int:
        return self._q.qsize()

    def _connect(self) -> bool:
        if self._sock is not None:
            return True
        try:
            s = socket.create_connection((self.ip, self.port), timeout=self._timeout)
            s.settimeout(None)
        except OSError as e:
            print(f"Connection failed to {self.ip}:{self.port}. Error: {e}", file=sys.stderr)
            return False
        self._sock = s
        print(f"Connected to {self.ip}:{self.port}")
        return True

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _send(self, msg: str) -> bool:
        if not self._connect():
            return False
        try:
            self._sock.sendall(msg.encode("utf-8"))  # Python ignores SIGPIPE, so a dead peer raises here
            return True
        except OSError as e:
            print(f"Failed to send message. Error: {e}", file=sys.stderr)
            self._close()
            return False

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while self._running.is_set() and time.monotonic() < end:
            time.sleep(0.1)

    def _run(self) -> None:
        while self._running.is_set():
            try:
                msg = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            if self._send(msg):
                continue
            print("Send failed. Attempting to reconnect...", file=sys.stderr)
            self._sleep(2)
            if self._connect() and self._send(msg):
                print("Message sent successfully after reconnect.")
                continue
            print("Reconnect failed. Re-queueing message.", file=sys.stderr)
            self._q.put(msg)
            self._sleep(5)
        print("Send thread stopped.")
