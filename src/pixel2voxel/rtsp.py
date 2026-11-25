"""Real-time multi-camera RTSP processor.

  1) one capture thread per camera pushes frames into a bounded queue
  2) one processing thread: grayscale -> frame difference -> ray casting into
     a shared voxel grid, time-based decay, periodic .bin saves
  3) a publisher thread streams the grid over ZeroMQ (rate-limited)
  4) the main thread shows the debug windows (OpenCV GUI must stay on the main
     thread on macOS) or reads 'q' / 's' commands from stdin

Usage:  p2v-rtsp <camera_config.json> [output_dir] [--no-publish]
"""

from __future__ import annotations

import os

# Must be set before any VideoCapture is created: TCP transport, 5 s socket
# timeout, no extra demuxer delay.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000|max_delay;0")

import argparse
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import zmq

from . import geometry, gridio, raycast
from .calib.model import Lens
from .config import AppConfig, CameraInfo, GridConfig, load_config
from .noise import NoiseModel

MAX_QUEUE = 30  # total frames kept; oldest dropped when full (bounded latency)
MIN_FRAME_GAP_S = 0.030  # frames closer than this to the previous one are skipped
DECAY_MIN_DT_S = 0.033  # decay is applied at most ~30 Hz


@dataclass
class FrameData:
    camera_index: int
    frame: np.ndarray
    timestamp: float  # time.monotonic()


class FrameQueue:
    def __init__(self, maxlen: int = MAX_QUEUE):
        self._q: deque[FrameData] = deque(maxlen=maxlen)  # maxlen drops the oldest
        self._cond = threading.Condition()

    def push(self, item: FrameData) -> None:
        with self._cond:
            self._q.append(item)
            self._cond.notify()

    def wait_and_pop(self, timeout: float) -> FrameData | None:
        with self._cond:
            if not self._q and not self._cond.wait_for(lambda: bool(self._q), timeout):
                return None
            return self._q.popleft()


_DEVICE_RE = re.compile(r"^(?:device:)?(\d+)(?:@(\d+)x(\d+))?$")


def parse_device(url: str) -> tuple[int, int | None, int | None] | None:
    """Local camera spec -> (index, width, height), or None for a URL / file.

    Accepted: "device:0", "device:1@1280x720", or a bare "0" / "1@1280x720".
    Pinning the resolution matters: a calibration is only valid at one resolution.
    """
    m = _DEVICE_RE.match(str(url).strip())
    if not m or os.path.isfile(url):
        return None
    w, h = m.group(2), m.group(3)
    return int(m.group(1)), int(w) if w else None, int(h) if h else None


_NAMED_RE = re.compile(r"^camera:(.+?)(?:@(\d+)x(\d+))?$")


def parse_named_camera(url: str) -> tuple[str, int | None, int | None] | None:
    """"camera:<name>[@WxH]" -> (name, width, height); None otherwise."""
    m = _NAMED_RE.match(str(url).strip())
    if not m:
        return None
    w, h = m.group(2), m.group(3)
    return m.group(1).strip(), int(w) if w else None, int(h) if h else None


def _open_capture(url: str) -> cv2.VideoCapture:
    named = parse_named_camera(url)
    if named is not None:
        # Native AVFoundation capture by camera name (see avfcam.py).
        from .avfcam import AVFCapture
        return AVFCapture(*named)
    dev = parse_device(url)
    if dev is not None:
        # Local USB / built-in camera: AVFoundation on macOS (FFmpeg can't open device indices).
        idx, w, h = dev
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
        if w and h:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return cap
    # Timeouts only take effect when passed to the constructor, not via set().
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency down where supported
    return cap


def _file_frame_period(cap: cv2.VideoCapture, url: str) -> float:
    """Seconds per frame when `url` is a local video file, else 0.

    A file decodes far faster than real time, so almost
    every frame would fall inside MIN_FRAME_GAP_S of the previous one and be
    skipped. Pacing a file at its own frame rate
    makes recorded footage replay like a live camera. RTSP streams are untouched.
    """
    if not os.path.isfile(url):
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    return 1.0 / fps if fps and fps > 0 else 1.0 / 25.0


def camera_thread(cam: CameraInfo, queue: FrameQueue, running: threading.Event) -> None:
    cap = _open_capture(cam.rtsp_url)
    if not cap.isOpened():
        print(f"ERROR: Failed to open stream for camera {cam.camera_index}", file=sys.stderr)
        return
    print(f"Camera {cam.camera_index} connected.")
    period = _file_frame_period(cap, cam.rtsp_url)
    next_due = time.monotonic()
    while running.is_set():
        if period:
            delay = next_due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            next_due = max(next_due + period, time.monotonic() - period)
        ok, frame = cap.read()
        if ok:
            if frame is None or frame.size == 0:
                print(f"WARNING: Empty frame received from camera {cam.camera_index}", file=sys.stderr)
                time.sleep(0.01)
                continue
            queue.push(FrameData(cam.camera_index, frame, time.monotonic()))
        else:
            print(f"ERROR: Failed to read frame from camera {cam.camera_index}", file=sys.stderr)
            time.sleep(0.1)
            cap.release()
            time.sleep(1.0)
            if not running.is_set():
                break
            cap = _open_capture(cam.rtsp_url)
            if cap.isOpened():
                print(f"Camera {cam.camera_index} reconnected.")
            else:
                print(f"ERROR: Failed to reconnect camera {cam.camera_index}", file=sys.stderr)
    cap.release()
    print(f"Camera {cam.camera_index} thread stopped.")


class Publisher:
    """Streams grid snapshots over ZeroMQ PUB from its own thread.

    The processing thread hands over a copy at most every `interval` seconds;
    compression (which releases the GIL) happens here so it never stalls
    ray casting. A snapshot offered while the previous one is still being
    compressed is simply skipped.
    """

    def __init__(self, address: str, voxel_size: float, interval: float):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.bind(address)  # raises zmq.ZMQError if the port is taken
        self._voxel_size = voxel_size
        self._interval = interval
        self._last = 0.0
        self._slot: np.ndarray | None = None
        self._cond = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="zmq-publisher", daemon=True)
        self._thread.start()

    def due(self) -> bool:
        with self._cond:
            return self._slot is None and time.monotonic() - self._last >= self._interval

    def offer(self, snapshot: np.ndarray) -> None:
        with self._cond:
            self._slot = snapshot
            self._last = time.monotonic()
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._slot is not None or self._stop)
                if self._stop:
                    return
                snapshot = self._slot
            meta, data = gridio.encode_zmq(snapshot, self._voxel_size)
            try:
                self._sock.send(meta, zmq.SNDMORE)
                self._sock.send(data, zmq.DONTWAIT)
            except zmq.Again:
                pass
            with self._cond:
                self._slot = None

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout=5)
        self._sock.close(linger=0)


class Processor:
    def __init__(self, app: AppConfig, publisher: Publisher | None):
        self.gc: GridConfig = app.grid
        self.cams = {c.camera_index: c for c in app.cameras}
        self.grid = np.zeros(self.gc.shape, dtype=np.float32)
        # Camera agreement (opt-in): one extra grid per camera. self.grid stays the plain sum.
        self.cam_grids: dict[int, np.ndarray] = {}
        if self.gc.min_cameras >= 2:
            if self.gc.min_cameras > len(self.cams):
                raise ValueError(f"min_cameras is {self.gc.min_cameras} but only {len(self.cams)} camera(s) are configured")
            self.cam_grids = {i: np.zeros(self.gc.shape, dtype=np.float32) for i in self.cams}
        self.grid_lock = threading.Lock()
        self.publisher = publisher
        self.debug_frames: dict[int, np.ndarray] = {}
        self.debug_lock = threading.Lock()
        self._rot = {i: geometry.rotation_rtsp(c.yaw, c.pitch, c.roll) for i, c in self.cams.items()}
        self._center = np.asarray(self.gc.center, dtype=np.float32)
        self._lens_base = {i: Lens.from_dict(c.intrinsics) for i, c in self.cams.items() if c.intrinsics}
        self._lens: dict[tuple[int, int, int], Lens] = {}
        self._noise: dict[tuple[int, int, int], NoiseModel] = {}
        # With the noise filter on, sub-noise pixels are already 0 when they reach the kernel.
        self.cast_threshold = 0.0 if self.gc.noise_sigma > 0 else self.gc.motion_threshold

    def lens_for(self, ci: int, width: int, height: int) -> Lens | None:
        """The camera's calibrated lens for this stream size (None if uncalibrated)."""
        base = self._lens_base.get(ci)
        if base is None:
            return None
        key = (ci, width, height)
        if key not in self._lens:
            self._lens[key] = base.for_size(width, height)  # raises on aspect-ratio mismatch
        return self._lens[key]

    def prepare(self, ci: int, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(float32 gray, display frame), both undistorted if the camera is calibrated.

        Frames are undistorted *before* differencing: |a - b| does not commute with
        the interpolation in the remap, so undistorting the difference would be wrong.
        """
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lens = self.lens_for(ci, gray.shape[1], gray.shape[0])
        if lens is not None:
            gray = lens.undistort(gray)
            if self.gc.show_debug:
                frame = lens.undistort(frame)
        return gray.astype(np.float32), frame

    def agreed_grid(self) -> np.ndarray:
        """Per voxel, the brightness that at least `min_cameras` cameras each reach.

        That is the min_cameras-th largest camera value, so one camera's streaks (however
        bright) contribute nothing where the others see no motion. Call with grid_lock held.
        """
        stack = np.stack(list(self.cam_grids.values()))
        k = len(stack) - self.gc.min_cameras
        return np.partition(stack, k, axis=0)[k]

    def filter_noise(self, ci: int, diff: np.ndarray, timestamp: float) -> np.ndarray | None:
        """`diff` unchanged when the noise filter is off; None while it is still learning."""
        gc = self.gc
        if gc.noise_sigma <= 0:
            return diff
        key = (ci, *diff.shape)
        if key not in self._noise:
            self._noise[key] = NoiseModel(gc.noise_sigma, gc.noise_floor, gc.noise_learn_seconds)
            print(f"Camera {ci}: learning noise for {gc.noise_learn_seconds:g} s, keep the scene still...")
        model = self._noise[key]
        if not model.ready:
            if model.learn(diff, timestamp):
                print(f"Camera {ci}: noise learned (per-pixel threshold median {np.median(model.threshold):.0f}, "
                      f"99% {np.percentile(model.threshold, 99):.0f} grey levels).")
            return None
        return model.apply(diff)

    # -- saving ---------------------------------------------------------------
    def save(self, tag: str | None = None) -> Path:
        stamp = int(time.time())
        name = f"voxel_grid_{tag}_{stamp}.bin" if tag else f"voxel_grid_{stamp}.bin"
        path = Path(self.gc.output_dir) / name
        with self.grid_lock:
            gridio.write_grid_bin(path, self.grid, self.gc.voxel_size)
            if self.cam_grids:  # saved beside the sum; files named voxel_grid_* stay the plain sum
                gridio.write_grid_bin(path.with_name(name.replace("voxel_grid", "agreed_grid", 1)),
                                      self.agreed_grid(), self.gc.voxel_size)
        print(f"Saved voxel grid to {path}")
        return path

    # -- debug overlay --------------------------------------------------------
    def _debug_overlay(self, cam_index: int, frame: np.ndarray, changed: np.ndarray) -> None:
        h, w = changed.shape
        dbg_w = min(w, 640)
        scale = dbg_w / w
        dbg_h = int(h * scale)
        overlay = cv2.resize(frame, (dbg_w, dbg_h))
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        rows = (np.arange(dbg_h) / scale).astype(np.int64)
        cols = (np.arange(dbg_w) / scale).astype(np.int64)
        small = changed[np.minimum(rows, h - 1)][:, np.minimum(cols, w - 1)]
        small &= (rows < h)[:, None] & (cols < w)[None, :]
        red = overlay[..., 2].astype(np.int16) + 180
        overlay[small, 0] = 0
        overlay[small, 1] = 0
        overlay[small, 2] = np.minimum(red[small], 255).astype(np.uint8)
        label = f"Cam {cam_index} motion: {int(changed.sum())} px"
        cv2.putText(overlay, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        with self.debug_lock:
            self.debug_frames[cam_index] = overlay

    # -- main loop ------------------------------------------------------------
    def run(self, queue: FrameQueue, running: threading.Event) -> None:
        gc = self.gc
        prev: dict[int, np.ndarray] = {}
        last_ts: dict[int, float] = {}
        last_save = time.time()
        last_decay = time.monotonic()

        while running.is_set():
            fd = queue.wait_and_pop(0.1)
            if fd is not None:
                ci = fd.camera_index
                try:
                    curr, shown = self.prepare(ci, fd.frame)
                except ValueError as e:  # calibration resolution mismatch
                    print(f"ERROR camera {ci}: {e}", file=sys.stderr)
                    running.clear()
                    break

                if ci not in prev:
                    prev[ci], last_ts[ci] = curr, fd.timestamp
                    continue
                # Too soon after this camera's previous frame: drop it without
                # updating `prev`.
                if fd.timestamp - last_ts[ci] < MIN_FRAME_GAP_S:
                    continue

                # Time-based decay, independent of camera count / FPS.
                if 0.0 < gc.decay_rate < 1.0:
                    now = time.monotonic()
                    dt = now - last_decay
                    if dt >= DECAY_MIN_DT_S:
                        factor = np.float32(gc.decay_rate**dt)
                        with self.grid_lock:
                            self.grid *= factor
                            for g in self.cam_grids.values():
                                g *= factor
                        last_decay = now

                if prev[ci].shape != curr.shape:
                    print("Images differ in size. Can't do motion detection!", file=sys.stderr)
                else:
                    diff = np.abs(prev[ci] - curr)
                    diff = self.filter_noise(ci, diff, fd.timestamp)
                    if diff is not None:
                        if gc.show_debug:
                            self._debug_overlay(ci, shown, diff > self.cast_threshold)
                        self._cast(ci, diff)

                prev[ci], last_ts[ci] = curr, fd.timestamp

            if time.time() - last_save >= gc.save_interval_seconds:
                self.save()
                last_save = time.time()

    def _cast(self, ci: int, diff: np.ndarray) -> None:
        gc, cam = self.gc, self.cams[ci]
        h, w = diff.shape
        lens = self.lens_for(ci, w, h)
        if lens is not None:
            fx, fy, cx, cy = lens.ray_intrinsics  # of the undistorted image (new_K)
        else:
            (fx, fy), cx, cy = geometry.focal_lengths(w, h, cam.fov_degrees), None, None
        with self.grid_lock:
            writes = raycast.cast_frame(
                self.grid, diff, self.cast_threshold, cam.camera_position, self._rot[ci],
                fx, fy, self._center, gc.voxel_size, gc.distance_attenuation, raycast.RAY_RTSP,
                cx=cx, cy=cy,
            )
            if ci in self.cam_grids:
                raycast.cast_frame(
                    self.cam_grids[ci], diff, self.cast_threshold, cam.camera_position, self._rot[ci],
                    fx, fy, self._center, gc.voxel_size, gc.distance_attenuation, raycast.RAY_RTSP,
                    cx=cx, cy=cy,
                )
            snapshot = None
            if self.publisher and self.publisher.due():  # the viewer shows the agreed grid when it is on
                snapshot = self.agreed_grid() if self.cam_grids else self.grid.copy()
        if writes:
            print(f"[{time.strftime('%H:%M:%S')}] Camera {ci} added voxels.")
        if snapshot is not None:
            self.publisher.offer(snapshot)


def _command_loop_stdin(proc: Processor, running: threading.Event) -> None:
    while running.is_set():
        ch = sys.stdin.read(1)
        if ch == "":  # EOF (e.g. stdin closed): just wait
            while running.is_set():
                time.sleep(0.5)
            return
        if ch in "qQ":
            running.clear()
        elif ch in "sS":
            proc.save("manual")


def _command_loop_gui(proc: Processor, running: threading.Event) -> None:
    while running.is_set():
        with proc.debug_lock:
            frames = list(proc.debug_frames.items())
        for cam_id, frame in frames:
            cv2.imshow(f"Camera {cam_id} Motion", frame)
        if not frames:
            # Some HighGUI backends return from waitKey immediately while no
            # window exists, which would spin until the first frame arrives.
            time.sleep(0.03)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q")):
            running.clear()
        elif key in (ord("s"), ord("S")):
            proc.save("manual")
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p2v-rtsp", description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="JSON config with grid parameters and cameras")
    ap.add_argument("output_dir", nargs="?", help="override output directory (default: config or '.')")
    ap.add_argument("--no-publish", action="store_true", help="don't stream the grid over ZeroMQ")
    args = ap.parse_args(argv)

    app = load_config(args.config)
    if not app.cameras:
        print("No cameras configured.", file=sys.stderr)
        return 1
    gc = app.grid
    if args.output_dir:
        gc.output_dir = args.output_dir
    Path(gc.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(app.cameras)} camera(s)")
    print(f"Grid: {gc.Nx}x{gc.Ny}x{gc.Nz}  voxel_size={gc.voxel_size}  center={gc.center}  "
          f"decay={gc.decay_rate}  debug={'ON' if gc.show_debug else 'OFF'}")
    print(f"Grid memory: {gc.Nx * gc.Ny * gc.Nz * 4 // (1024 * 1024)} MB")

    publisher = None
    if not args.no_publish:
        address = f"tcp://127.0.0.1:{os.environ.get('ZMQ_PORT', '5556')}"
        try:
            publisher = Publisher(address, gc.voxel_size, gc.publish_interval_seconds)
        except zmq.ZMQError as e:
            print(f"ERROR: cannot bind ZeroMQ socket: {e}  ({address})", file=sys.stderr)
            return 2
        print(f"ZeroMQ publisher bound to {address} (every {gc.publish_interval_seconds}s at most)")

    try:
        proc = Processor(app, publisher)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if proc.cam_grids:
        print(f"Camera agreement ON: publishing only what at least {gc.min_cameras} cameras see")
    running = threading.Event()
    running.set()
    queue = FrameQueue()

    # Compile the numba kernel before frames start arriving.
    raycast.cast_frame(np.zeros((2, 2, 2), np.float32), np.zeros((2, 2), np.float32),
                       1.0, (0, 0, 0), np.eye(3), 1.0, 1.0, (0, 0, 0), 1.0, 0.0)

    # Daemon threads: a camera stuck inside read() must not block shutdown.
    cams = [threading.Thread(target=camera_thread, args=(c, queue, running), daemon=True,
                             name=f"camera-{c.camera_index}") for c in app.cameras]
    for t in cams:
        t.start()
    worker = threading.Thread(target=proc.run, args=(queue, running), name="processing", daemon=True)
    worker.start()

    print("Real-time processing started.")
    print("Press 'q' then Enter to quit, 's' then Enter to save the current voxel grid."
          if not gc.show_debug else "Debug windows enabled - press 'q' in any window to quit, 's' to save.")
    try:
        (_command_loop_gui if gc.show_debug else _command_loop_stdin)(proc, running)
    except KeyboardInterrupt:
        pass
    finally:
        running.clear()
        worker.join(timeout=10)
        for t in cams:
            t.join(timeout=2)
        if publisher:
            publisher.close()
    print("All threads stopped. Exiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
