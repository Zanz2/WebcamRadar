"""Run the real p2v-rtsp CLI against a video file standing in for a camera.

OpenCV's FFmpeg backend opens a file path exactly like an rtsp:// URL, so this
exercises capture threads, the frame queue, motion detection, ray casting,
periodic .bin saves, the ZeroMQ stream and the stdin command loop.
"""

import json
import subprocess
import sys
import time

import cv2
import numpy as np
import zmq

from pixel2voxel import gridio
from pixel2voxel.config import AppConfig, CameraInfo, GridConfig
from pixel2voxel.rtsp import Processor


def test_debug_overlay_marks_motion_red():
    """The show_debug path (on in the example config) without opening a window."""
    proc = Processor(AppConfig([CameraInfo(0)], GridConfig(Nx=4, Ny=4, Nz=4, show_debug=True)), publisher=None)
    frame = np.full((480, 1280, 3), 50, np.uint8)  # wider than 640 -> downscaled by 2
    changed = np.zeros((480, 1280), bool)
    changed[100:200, 300:500] = True
    proc._debug_overlay(0, frame, changed)
    ov = proc.debug_frames[0]
    assert ov.shape == (240, 640, 3)
    assert tuple(ov[75, 200]) == (0, 0, 230)  # motion: B, G zeroed, R = 50 + 180
    assert tuple(ov[230, 600]) == (50, 50, 50)  # untouched


def make_video(path, frames=60, w=160, h=120):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
    assert out.isOpened()
    for i in range(frames):
        img = np.zeros((h, w, 3), np.uint8)
        x = 20 + (i * 2) % (w - 60)
        cv2.rectangle(img, (x, 45), (x + 20, 75), (255, 255, 255), -1)
        out.write(img)
    out.release()


def test_rtsp_cli_end_to_end(tmp_path):
    video = tmp_path / "cam.mp4"
    make_video(video)
    port = 15000 + (int(time.time() * 1000) % 2000)
    cfg = {
        "grid": {"size": [40, 40, 40], "voxel_size": 0.25, "center": [0.0, 0.0, -5.0],
                 "motion_threshold": 20.0, "distance_attenuation": 0.0,
                 "save_interval_seconds": 1, "publish_interval_seconds": 0.2},
        "cameras": [{"camera_index": 0, "rtsp_url": str(video), "camera_position": [0, 0, 2],
                     "yaw": 0, "pitch": 0, "roll": 0, "fov_degrees": 60}],
    }
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    out_dir = tmp_path / "out"

    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(f"tcp://127.0.0.1:{port}")

    proc = subprocess.Popen(
        [sys.executable, "-m", "pixel2voxel.rtsp", str(tmp_path / "cfg.json"), str(out_dir)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**__import__("os").environ, "ZMQ_PORT": str(port)},
    )
    try:
        msg = None
        deadline = time.time() + 30
        while time.time() < deadline and msg is None:
            if sub.poll(200):
                msg = sub.recv_multipart()
        assert msg is not None, "no ZeroMQ grid received"
        live, vs = gridio.decode_zmq(*msg)
        assert live.shape == (40, 40, 40) and vs == 0.25

        time.sleep(1.5)  # let at least one periodic save happen
        proc.stdin.write("s\n")
        proc.stdin.flush()
        time.sleep(0.5)
        proc.stdin.write("q\n")
        proc.stdin.flush()
        output, _ = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
        sub.close(linger=0)

    assert proc.returncode == 0, output
    assert "All threads stopped" in output
    bins = sorted(out_dir.glob("voxel_grid_*.bin"))
    assert any("manual" in b.name for b in bins), output
    assert any("manual" not in b.name for b in bins), output  # periodic save
    grid, _ = gridio.read_bin(bins[-1])
    assert np.count_nonzero(grid) > 0
    # The square moves horizontally in the middle of the image -> along x near y = 0,
    # so the brightest voxel column should sit near the grid's y centre.
    _, iy, _ = np.unravel_index(np.argmax(grid), grid.shape)
    assert abs(iy - 20) <= 4
