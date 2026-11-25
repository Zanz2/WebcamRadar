import json
import struct

import numpy as np
import pytest

from pixel2voxel import config, gridio


def test_legacy_array_config(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"camera_index": 3, "rtsp_url": "rtsp://x", "camera_position": [1, 2, 3], "yaw": 10}]))
    cfg = config.load_config(p)
    assert cfg.cameras[0].camera_index == 3 and cfg.cameras[0].camera_position == (1.0, 2.0, 3.0)
    assert cfg.cameras[0].fov_degrees == 60.0
    assert cfg.grid.shape == (500, 500, 500)


def test_grid_object_config(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"grid": {"size": [10, 20, 30], "voxel_size": 0.1, "decay_rate": 0.9},
                             "cameras": [{"camera_index": 0}, {"camera_index": 1}]}))
    cfg = config.load_config(p)
    assert cfg.grid.shape == (10, 20, 30) and cfg.grid.voxel_size == 0.1 and cfg.grid.decay_rate == 0.9
    assert [c.camera_index for c in cfg.cameras] == [0, 1]
    p.write_text(json.dumps({"grid": {"size": 7}, "camera_index": 5}))
    cfg = config.load_config(p)
    assert cfg.grid.shape == (7, 7, 7) and cfg.cameras[0].camera_index == 5


def test_bin_formats_roundtrip(tmp_path):
    g = np.random.default_rng(0).random((4, 5, 6)).astype(np.float32)
    gridio.write_grid_bin(tmp_path / "g.bin", g, 0.5)
    raw = (tmp_path / "g.bin").read_bytes()
    assert struct.unpack("<iiif", raw[:16]) == (4, 5, 6, 0.5)  # header layout
    g2, vs = gridio.read_bin(tmp_path / "g.bin")
    assert np.array_equal(g, g2) and vs == 0.5

    c = np.random.default_rng(1).random((5, 5, 5)).astype(np.float32)
    gridio.write_cube_bin(tmp_path / "c.bin", c, 6.0)
    assert struct.unpack("<if", (tmp_path / "c.bin").read_bytes()[:8]) == (5, 6.0)
    c2, vs = gridio.read_bin(tmp_path / "c.bin")
    assert np.array_equal(c, c2) and vs == 6.0

    (tmp_path / "bad.bin").write_bytes(b"\x00" * 30)
    with pytest.raises(ValueError):
        gridio.read_bin(tmp_path / "bad.bin")


def test_zmq_wire_format_roundtrip():
    g = np.random.default_rng(2).random((3, 4, 5)).astype(np.float32)
    meta, data = gridio.encode_zmq(g, 2.0)
    assert len(meta) == 16
    g2, vs = gridio.decode_zmq(meta, data)
    assert np.array_equal(g, g2) and vs == 2.0


def test_local_camera_specs(tmp_path):
    from pixel2voxel.rtsp import parse_device
    assert parse_device("device:0") == (0, None, None)
    assert parse_device("device:2@1280x720") == (2, 1280, 720)
    assert parse_device("1") == (1, None, None)
    assert parse_device("rtsp://u:p@10.0.0.1:554/s") is None
    f = tmp_path / "3"
    f.write_bytes(b"")
    assert parse_device(str(f)) is None  # an existing file named like an index is still a file


def test_named_camera_specs():
    from pixel2voxel.rtsp import parse_named_camera
    assert parse_named_camera("camera:HD WebCam@1280x960") == ("HD WebCam", 1280, 960)
    assert parse_named_camera("camera:USB Camera VID:1234 PID:5678@960x720") == ("USB Camera VID:1234 PID:5678", 960, 720)
    assert parse_named_camera("camera:id=0x1100000046d0825") == ("id=0x1100000046d0825", None, None)
    assert parse_named_camera("device:0") is None and parse_named_camera("rtsp://h/s") is None
