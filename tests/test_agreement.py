import numpy as np
import pytest

from pixel2voxel import gridio
from pixel2voxel.config import AppConfig, CameraInfo, GridConfig, parse_grid
from pixel2voxel.rtsp import Processor

# Three cameras looking straight down (all angles 0) at the same volume from different spots.
CAMS = [CameraInfo(camera_index=i, camera_position=p) for i, p in
        ((1, (0.0, 0.0, 3.0)), (2, (0.6, 0.0, 3.0)), (4, (0.0, 0.6, 3.0)))]


def _proc(**grid):
    gc = GridConfig(Nx=30, Ny=30, Nz=30, voxel_size=0.1, center=(0.0, 0.0, 0.0), motion_threshold=10.0,
                    distance_attenuation=0.0, **grid)
    return Processor(AppConfig(CAMS, gc), None)


def _diff_for(cam, point, value=50.0, w=320, h=240):
    """A difference image with one 3x3 spot where `cam` sees `point`."""
    f = 0.5 * w / np.tan(np.radians(30.0))  # 60 degrees horizontal
    d = np.asarray(point) - np.asarray(cam.camera_position)
    u, v = int(round(f * d[0] / -d[2] + w / 2)), int(round(-f * d[1] / -d[2] + h / 2))
    out = np.zeros((h, w), np.float32)
    out[v - 1:v + 2, u - 1:u + 2] = value
    return out


def test_off_by_default_and_sum_is_identical_either_way():
    assert parse_grid({}).min_cameras == 0
    off, on = _proc(), _proc(min_cameras=3)
    assert not off.cam_grids
    for c in CAMS:
        d = _diff_for(c, (0.05, 0.05, 0.05))
        off._cast(c.camera_index, d)
        on._cast(c.camera_index, d)
    assert off.grid.any() and off.grid.tobytes() == on.grid.tobytes()


def test_all_three_cameras_meet_at_the_target():
    p, target = _proc(min_cameras=3), np.array([0.25, -0.35, 0.15])
    for c in CAMS:
        p._cast(c.camera_index, _diff_for(c, target))
    g = p.agreed_grid()
    pos = (np.array(np.unravel_index(g.argmax(), g.shape)) + 0.5) * 0.1 - 1.5
    assert np.linalg.norm(pos - target) < 0.25


def test_a_bright_single_camera_streak_is_rejected():
    p = _proc(min_cameras=2)
    for _ in range(50):
        p._cast(1, _diff_for(CAMS[0], (0.0, 0.0, 0.0), 250.0))
    assert p.grid.max() > 1000 and not p.agreed_grid().any()


def test_two_of_three():
    two, three = _proc(min_cameras=2), _proc(min_cameras=3)
    for p in (two, three):
        for c in CAMS[:2]:
            p._cast(c.camera_index, _diff_for(c, (0.05, 0.05, 0.05)))
    assert two.agreed_grid().any() and not three.agreed_grid().any()


def test_too_many_cameras_required_is_refused():
    with pytest.raises(ValueError, match="min_cameras"):
        _proc(min_cameras=4)


def test_save_writes_the_agreed_grid_beside_the_sum(tmp_path):
    p = _proc(min_cameras=3, output_dir=str(tmp_path))
    for c in CAMS:
        p._cast(c.camera_index, _diff_for(c, (0.05, 0.05, 0.05)))
    path = p.save()
    agreed = path.with_name(path.name.replace("voxel_grid", "agreed_grid"))
    assert np.array_equal(gridio.read_bin(path)[0], p.grid)
    assert np.array_equal(gridio.read_bin(agreed)[0], p.agreed_grid())
