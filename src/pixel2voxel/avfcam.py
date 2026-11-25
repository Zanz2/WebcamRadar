"""Native macOS camera capture through AVFoundation (what Photo Booth uses), via PyObjC.

Why: OpenCV's own AVFoundation backend delivers zero frames from some webcams
(seen with a USB webcam that works fine in Photo Booth), and its device
numbers shift whenever a camera or an iPhone (Continuity Camera) comes and goes.
This backend opens cameras by NAME, so a config keeps pointing at the same
physical camera.

Use in a config:  "rtsp_url": "camera:HD WebCam@1280x960"
(name = any case-insensitive part of the camera's name as `p2v-calibrate
list-cameras` prints it, or "id=<unique id>" when two cameras share a name).

AVFCapture mimics the bits of cv2.VideoCapture the pipeline uses:
isOpened(), read(), get(), set(), release().
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

try:  # macOS only
    import AVFoundation as AVF
    import CoreMedia as CM
    import libdispatch
    import objc
    import Quartz  # CoreVideo pixel-buffer functions live here
    # pyobjc resolves CoreMedia functions lazily, and that lookup is not thread-safe: several camera threads
    # starting together raced on it (KeyError, one camera silently never started). Resolve it once, here.
    # The same goes for everything the frame callback uses: each camera's callback runs on its own queue thread.
    CM.CMVideoFormatDescriptionGetDimensions
    CM.CMSampleBufferGetImageBuffer
    CM.CMGetAttachment
    CM.kCMSampleBufferAttachmentKey_DroppedFrameReason
    Quartz.CVPixelBufferLockBaseAddress
    Quartz.CVPixelBufferUnlockBaseAddress
    Quartz.CVPixelBufferGetBaseAddress
    Quartz.CVPixelBufferGetBytesPerRow
    Quartz.CVPixelBufferGetWidth
    Quartz.CVPixelBufferGetHeight
    AVAILABLE = True
except ImportError:  # pragma: no cover - non-macOS
    AVAILABLE = False


def _require():
    if not AVAILABLE:
        raise RuntimeError("camera:<name> needs macOS and the pyobjc AVFoundation packages (installed by uv sync on macOS)")


def _modes(dev) -> list[tuple[int, int]]:
    out = set()
    for f in dev.formats():
        d = CM.CMVideoFormatDescriptionGetDimensions(f.formatDescription())
        out.add((int(d.width), int(d.height)))
    return sorted(out, key=lambda wh: (-wh[0] * wh[1], -wh[0]))


_LIST_LOCK = threading.Lock()


def list_devices() -> list[dict]:
    """Every video camera macOS offers (incl. external webcams and Continuity iPhones)."""
    _require()
    with _LIST_LOCK:  # camera threads start together; keep AVFoundation discovery one at a time
        return _list_devices()


def _list_devices() -> list[dict]:
    types = [AVF.AVCaptureDeviceTypeBuiltInWideAngleCamera]
    for name in ("AVCaptureDeviceTypeExternal", "AVCaptureDeviceTypeExternalUnknown", "AVCaptureDeviceTypeContinuityCamera"):
        if hasattr(AVF, name):
            types.append(getattr(AVF, name))
    try:
        session = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
            types, AVF.AVMediaTypeVideo, AVF.AVCaptureDevicePositionUnspecified)
        devs = list(session.devices())
    except Exception:  # older macOS
        devs = list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo))
    return [{"name": str(d.localizedName()), "id": str(d.uniqueID()), "modes": _modes(d), "device": d} for d in devs]


def find_device(match: str):
    """Pick one camera by name fragment (case-insensitive) or 'id=<uniqueID>'."""
    devs = list_devices()
    if match.startswith("id="):
        hits = [d for d in devs if d["id"] == match[3:]]
    else:
        exact = [d for d in devs if d["name"].lower() == match.lower()]
        hits = exact or [d for d in devs if match.lower() in d["name"].lower()]
    names = ", ".join(f"'{d['name']}'" for d in devs) or "none"
    if not hits:
        raise RuntimeError(f"no camera matching '{match}' (available: {names})")
    if len(hits) > 1:
        ids = "; ".join(f"{d['name']} -> camera:id={d['id']}" for d in hits)
        raise RuntimeError(f"'{match}' matches several cameras; use an id instead: {ids}")
    return hits[0]


if AVAILABLE:
    class _FrameSink(AVF.NSObject):
        """AVCaptureVideoDataOutput delegate: keeps the newest frame as a BGR numpy array."""

        def initWithOwner_(self, owner):
            self = objc.super(_FrameSink, self).init()
            if self is None:
                return None
            self.owner = owner
            return self

        def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sbuf, conn):
            # This runs on a GCD queue thread, which has no autorelease pool of its own.
            # Without one, every frame's buffer stays retained, AVFoundation's small
            # buffer pool runs dry after ~150 frames, and delivery silently stops
            # ("works for ~5 s, then freezes"). Drain a pool per frame.
            with objc.autorelease_pool():
                pb = CM.CMSampleBufferGetImageBuffer(sbuf)
                if pb is None:
                    return
                Quartz.CVPixelBufferLockBaseAddress(pb, 1)
                try:
                    w, h = Quartz.CVPixelBufferGetWidth(pb), Quartz.CVPixelBufferGetHeight(pb)
                    bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
                    buf = Quartz.CVPixelBufferGetBaseAddress(pb).as_buffer(bpr * h)
                    bgra = np.frombuffer(buf, np.uint8).reshape(h, bpr)[:, : w * 4].reshape(h, w, 4)
                    frame = np.ascontiguousarray(bgra[:, :, :3])  # copy out before unlocking
                    del bgra, buf
                finally:
                    Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
                del pb
            self.owner._push(frame)

        def captureOutput_didDropSampleBuffer_fromConnection_(self, output, sbuf, conn):
            with objc.autorelease_pool():
                reason = CM.CMGetAttachment(sbuf, CM.kCMSampleBufferAttachmentKey_DroppedFrameReason, None)
                self.owner._dropped(str(reason[0] if isinstance(reason, tuple) else reason))


class AVFCapture:
    def __init__(self, match: str, width: int | None = None, height: int | None = None, timeout: float = 3.0):
        _require()
        self._frame, self._seq, self._read_seq = None, 0, 0
        self.drops: dict[str, int] = {}  # AVFoundation's reasons for frames it dropped
        self._cond = threading.Condition()
        self._timeout = timeout
        self._session = None
        info = find_device(match)
        self.name = info["name"]
        dev = info["device"]
        session = AVF.AVCaptureSession.alloc().init()
        inp, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(dev, None)
        if inp is None:
            raise RuntimeError(f"cannot open '{self.name}': {err}")
        out = AVF.AVCaptureVideoDataOutput.alloc().init()
        out.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA})
        out.setAlwaysDiscardsLateVideoFrames_(True)
        self._sink = _FrameSink.alloc().initWithOwner_(self)
        self._queue = libdispatch.dispatch_queue_create(b"pixel2voxel.avfcam", None)
        out.setSampleBufferDelegate_queue_(self._sink, self._queue)
        # One configuration transaction. The session's default preset would otherwise
        # reset the camera to its own favourite format when it starts, silently
        # ignoring the requested size (seen on all three test cameras).
        session.beginConfiguration()
        session.addInput_(inp)
        session.addOutput_(out)
        fmt = None
        if width and height:
            # Setting activeFormat inside the transaction switches the session to its
            # "input priority" preset (it can't be set directly on macOS).
            fmt = self._select_format(dev, width, height)
        session.commitConfiguration()
        self._session, self._out, self._dev = session, out, dev
        session.startRunning()
        if fmt is not None:
            self._ensure_size(fmt, width, height)
        self._lock_focus()  # last: a format switch may reset the focus mode

    def _lock_focus(self) -> None:
        """Autofocus off, on every open: refocusing changes the lens and voids its calibration.

        This freezes the lens where it is; AVFoundation on macOS has no way to set a
        focus distance. Fixed-focus cameras don't offer the mode and are left alone.
        """
        dev, locked = self._dev, AVF.AVCaptureFocusModeLocked
        if not dev.isFocusModeSupported_(locked):
            return
        ok, _ = dev.lockForConfiguration_(None)
        if ok:
            dev.setFocusMode_(locked)
            dev.unlockForConfiguration()
        if not ok or dev.focusMode() != locked:
            self.release()
            raise RuntimeError(f"'{self.name}': could not lock the focus (its calibration would not hold)")
        print(f"'{self.name}': autofocus off (focus locked)")

    def _select_format(self, dev, width: int, height: int):
        self._dev_for_lock = dev
        best = None
        for f in dev.formats():
            d = CM.CMVideoFormatDescriptionGetDimensions(f.formatDescription())
            if (d.width, d.height) == (width, height):
                fps = max((r.maxFrameRate() for r in f.videoSupportedFrameRateRanges()), default=0)
                if best is None or fps > best[0]:
                    best = (fps, f)
        if best is None:
            modes = ", ".join(f"{w}x{h}" for w, h in _modes(dev))
            raise RuntimeError(f"'{self.name}' has no {width}x{height} mode (has: {modes})")
        self._apply_format(best[1])
        return best[1]

    def _apply_format(self, fmt) -> None:
        ok, _ = self._dev_for_lock.lockForConfiguration_(None)
        if ok:
            self._dev_for_lock.setActiveFormat_(fmt)
            self._dev_for_lock.unlockForConfiguration()

    def _ensure_size(self, fmt, width: int, height: int) -> None:
        """Check the frames really have the requested size; re-apply and wait, else fail loudly.

        A calibration is only valid at one resolution, so a silently different size
        must never get through. Starting the session resets some cameras to their own
        favourite size, and the switch back takes a variable time (old-size frames
        still arrive meanwhile), so poll instead of judging one frame after a fixed wait.
        """
        for attempt in range(3):
            settle = time.monotonic() + (2.0 if attempt else 0.0)  # attempt 0: just look
            while True:
                ok, frame = self.read()
                if ok and (frame.shape[1], frame.shape[0]) == (width, height):
                    return
                if not ok or time.monotonic() >= settle:
                    break
            if attempt < 2:
                self._apply_format(fmt)
        got = f"{frame.shape[1]}x{frame.shape[0]}" if ok else "no frames"
        self.release()
        raise RuntimeError(f"'{self.name}' was asked for {width}x{height} but delivers {got}")

    def _dropped(self, reason: str) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1

    def _push(self, frame: np.ndarray) -> None:
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    # -- cv2.VideoCapture-compatible surface -------------------------------------
    def isOpened(self) -> bool:
        return self._session is not None

    def read(self):
        """Wait for a frame newer than the last one returned; (False, None) on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._read_seq, self._timeout):
                return False, None
            self._read_seq = self._seq
            return True, self._frame

    def get(self, prop: int) -> float:
        with self._cond:
            f = self._frame
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(f.shape[1]) if f is not None else 0.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(f.shape[0]) if f is not None else 0.0
        return 0.0

    def set(self, prop: int, value) -> bool:
        return False  # resolution is chosen when opening ("camera:NAME@WxH")

    def release(self) -> None:
        if self._session is not None:
            self._session.stopRunning()
            self._session = None
            time.sleep(0.05)
