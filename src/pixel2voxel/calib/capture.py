"""Grab calibration images straight from a camera stream (or a video file).

capture  -- lens calibration: shows live board detection, a 3x3 coverage map and
            how much of the frame the board fills. Auto-saves a frame when the
            board has been held still for a few frames AND the view is new
            (different place, size or tilt). SPACE saves manually.
snapshot -- placement / check: averages ~15 frames into one low-noise still.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..rtsp import _open_capture  # sets the FFmpeg RTSP options; never prints the URL
from .board import BoardDetector

STABLE_PX = 1.0  # mean corner motion (pixels, at detection scale) that still counts as "still"
TILT_WEIGHT = 0.4  # ~12 deg of board tilt counts as much as moving 8% across the image
MAX_TILT = 50.0  # steeper views have foreshortened, partly defocused corners
WARMUP_S = 1.0  # live streams: skip frames before the first keyframe / while auto-exposure settles


class LatestFrameReader:
    """Reads a live stream on its own thread and keeps only the newest frame.

    Detection + drawing can take longer than the camera's frame interval; reading
    in the same loop would let FFmpeg's buffer fill and the preview fall seconds
    behind the board.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap, self._frame, self._lock = cap, None, threading.Lock()
        self._stop = threading.Event()
        self._fresh = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
            self._fresh.set()

    def read(self, timeout: float = 2.0):
        if not self._fresh.wait(timeout):
            return False, None
        with self._lock:
            self._fresh.clear()
            return True, self._frame

    def close(self):
        self._stop.set()
        self._t.join(timeout=2)


def _detect_scaled(det: BoardDetector, frame: np.ndarray, max_width: int = 1280):
    """Detect on a downscaled copy for speed; return corners in full-resolution pixels."""
    s = min(1.0, max_width / frame.shape[1])
    small = frame if s == 1.0 else cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    c, ids = det.detect(small)
    return c / s, ids, s


class StillnessTracker:
    """Board is 'still' when the same corners moved < STABLE_PX for `needed` consecutive frames."""

    def __init__(self, needed: int = 3):
        self.needed, self.count, self.prev = needed, 0, None

    def update(self, corners: np.ndarray, ids: np.ndarray, scale: float) -> bool:
        cur = dict(zip(ids.tolist(), corners))
        if self.prev is not None and cur:
            common = [i for i in cur if i in self.prev]
            if len(common) >= max(4, 0.8 * len(cur)):
                motion = np.mean([np.linalg.norm(cur[i] - self.prev[i]) for i in common]) * scale
                self.count = self.count + 1 if motion < STABLE_PX else 0
            else:
                self.count = 0
        else:
            self.count = 0
        self.prev = cur
        return self.count >= self.needed


def _view_signature(corners: np.ndarray, ids: np.ndarray, det: BoardDetector, w: int, h: int) -> np.ndarray:
    """Where the board is (centre, size) and which way it faces (tilt).

    Tilted views are what separate focal length from distortion, so a tilt in
    place must count as a new view. The board normal comes from solvePnP with a
    rough pinhole camera (f = image width); it is insensitive to that guess.
    """
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    sig = [(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, np.sqrt((x1 - x0) * (y1 - y0) / (w * h)), 0.0, 0.0]
    if len(ids) >= 4:
        K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(det.object_points(ids), corners, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            normal = cv2.Rodrigues(rvec)[0][:, 2]
            sig[3], sig[4] = TILT_WEIGHT * normal[0], TILT_WEIGHT * normal[1]
    return np.array(sig)


def capture(url: str, out_dir: str | Path, det: BoardDetector, target: int = 25, min_corners: int = 12,
            novelty: float = 0.08, min_interval: float = 1.0, gui: bool = True, auto: bool = True,
            max_seconds: float | None = None) -> int:
    """Save calibration frames into out_dir. Returns how many were saved."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = _open_capture(url)
    if not cap.isOpened():
        raise RuntimeError("could not open the camera stream / file")
    still = StillnessTracker()
    saved_sigs: list[np.ndarray] = []
    coverage = np.zeros((3, 3), int)
    n_tilted = 0  # saved views with the board tilted 15+ deg (needed to pin down the focal length)
    n_saved = len(list(out.glob("*.png")))
    is_file = Path(url).is_file()

    def now() -> float:  # a video file is read faster than real time: use its own clock
        return cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if is_file else time.monotonic()

    # A file must be read frame by frame (its clock drives the timing); a live
    # stream is read on a background thread so the preview never lags.
    reader = None if is_file else LatestFrameReader(cap)
    read = cap.read if is_file else reader.read

    last_save, t_start = -1e9, time.monotonic()
    print(f"Saving into {out}  (target {target}). "
          + ("SPACE = save, a = toggle auto, q = quit." if gui else "Auto-capturing; Ctrl+C to stop."))
    try:
        while n_saved < target:
            if max_seconds is not None and time.monotonic() - t_start > max_seconds:
                break
            ok, frame = read()
            if not ok:
                if is_file:
                    break  # end of a video file
                continue
            h, w = frame.shape[:2]
            corners, ids, scale = _detect_scaled(det, frame)
            is_still = len(ids) >= min_corners and still.update(corners, ids, scale)
            if len(ids) < min_corners:
                still.update(np.empty((0, 2)), np.empty(0, int), scale)
            width = 0.0  # board corners' span as a fraction of the image width
            tilt = 0.0
            reason = ""
            if len(ids) >= min_corners:
                sig = _view_signature(corners, ids, det, w, h)
                width = float(np.ptp(corners[:, 0]) / w)
                tilt = _tilt_deg(sig)
                new = not saved_sigs or min(np.linalg.norm(sig - s) for s in saved_sigs) > novelty
                due = now() - last_save >= min_interval
                too_steep = tilt > MAX_TILT
                reason = ("too steep: tilt less than 50 deg" if too_steep else "hold still" if not is_still else
                          "move to a new spot / angle / distance" if not new else "")
                if auto and is_still and new and due and not too_steep:
                    key_save = True
                else:
                    key_save = False
            else:
                key_save = False
            key = -1
            if gui:
                view = frame.copy()
                if len(ids):
                    for c in corners:
                        cv2.circle(view, (int(c[0]), int(c[1])), max(3, w // 300), (0, 255, 0), -1)
                _draw_coverage(view, coverage)
                need_tilt = n_saved >= 6 and n_tilted < 0.3 * n_saved
                lines = [f"saved {n_saved}/{target}   corners {len(ids)}   board width {width * 100:.0f}% "
                         f"{'(MOVE CLOSER: aim for 25-50%)' if 0 < width < 0.15 else ''}",
                         f"tilt {tilt:.0f} deg   tilted views {n_tilted}"
                         f"{'   TILT THE BOARD MORE (20-45 deg)' if need_tilt else ''}",
                         f"auto {'ON' if auto else 'OFF'}   {reason}"]
                for i, t in enumerate(lines):
                    cv2.putText(view, t, (15, 35 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                scale_disp = min(1.0, 1280 / w)
                cv2.imshow("calibration capture", cv2.resize(view, None, fx=scale_disp, fy=scale_disp))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("a"):
                    auto = not auto
            if key_save or (key == ord(" ") and len(ids) >= min_corners):
                path = out / f"calib_{n_saved:03d}.png"
                cv2.imwrite(str(path), frame)  # the frame that was actually detected
                saved_sigs.append(_view_signature(corners, ids, det, w, h))
                n_tilted += _tilt_deg(saved_sigs[-1]) >= 15
                cx = np.clip((corners[:, 0] / w * 3).astype(int), 0, 2)
                cy = np.clip((corners[:, 1] / h * 3).astype(int), 0, 2)
                np.add.at(coverage, (cy, cx), 1)
                n_saved += 1
                last_save = now()
                print(f"  saved {path.name}  ({len(ids)} corners, board width {width * 100:.0f}%, "
                      f"tilt {_tilt_deg(saved_sigs[-1]):.0f} deg)")
    except KeyboardInterrupt:
        pass
    finally:
        if reader is not None:
            reader.close()
        cap.release()
        if gui:
            cv2.destroyAllWindows()
    empty = [n for (y, x), n in zip(np.ndindex(3, 3), ["top-left", "top", "top-right", "left", "centre", "right",
                                                      "bottom-left", "bottom", "bottom-right"]) if coverage[y, x] < 4]
    if n_saved and n_tilted < 0.3 * n_saved:
        print(f"Only {n_tilted} of the views saved this session were tilted 15+ deg. Add tilted views "
              "(board turned 20-45 deg left/right/up/down) or the focal length stays uncertain.")
    if empty:
        print(f"Regions of the image with few board corners: {', '.join(empty)}. "
              "Capture a few more with the board there (corners matter most for wide lenses).")
    return n_saved


def _tilt_deg(sig: np.ndarray) -> float:
    """Board tilt away from facing the camera, from a view signature."""
    return float(np.degrees(np.arcsin(min(1.0, np.hypot(sig[3], sig[4]) / TILT_WEIGHT))))


def _draw_coverage(view: np.ndarray, coverage: np.ndarray) -> None:
    h, w = view.shape[:2]
    for y in range(3):
        for x in range(3):
            n = coverage[y, x]
            colour = (0, 0, 255) if n < 4 else (0, 200, 255) if n < 20 else (0, 200, 0)
            x0, y0 = x * w // 3, y * h // 3
            cv2.rectangle(view, (x0 + 4, y0 + 4), (x0 + w // 3 - 4, y0 + h // 3 - 4), colour, 2)


def snapshot(url: str, out_path: str | Path, frames: int = 15, det: BoardDetector | None = None,
             timeout: float = 20.0) -> int:
    """Average `frames` frames into one still (less noise -> more precise corners).

    With `det`, also report how many board corners the still shows. Returns that
    count (or 0 without a detector). The camera and the board must not move.
    """
    cap = _open_capture(url)
    if not cap.isOpened():
        raise RuntimeError("could not open the camera stream / file")
    acc, n = None, 0
    if not Path(url).is_file():
        warm = time.monotonic() + WARMUP_S
        while time.monotonic() < warm:
            cap.read()
    t0 = time.monotonic()
    try:
        while n < frames and time.monotonic() - t0 < timeout:
            ok, frame = cap.read()
            if not ok:
                if Path(url).is_file():
                    break
                time.sleep(0.1)
                continue
            acc = frame.astype(np.float64) if acc is None else acc + frame
            n += 1
    finally:
        cap.release()
    if acc is None:
        raise RuntimeError("no frames received")
    still = np.clip(acc / n + 0.5, 0, 255).astype(np.uint8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), still)
    found = 0
    if det is not None:
        found = len(det.detect_upscaled(still)[1])
        if found < 6:
            print(f"WARNING: only {found} board corners visible in {out_path}", file=sys.stderr)
    return found
