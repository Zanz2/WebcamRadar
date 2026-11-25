import numpy as np

from pixel2voxel import geometry, raycast
from pixel2voxel.config import AppConfig, CameraInfo, GridConfig, parse_grid
from pixel2voxel.noise import NoiseModel
from pixel2voxel.rtsp import Processor


def _proc(**grid):
    gc = GridConfig(Nx=20, Ny=20, Nz=20, voxel_size=0.1, center=(0.0, 0.0, 0.0), motion_threshold=10.0, **grid)
    return Processor(AppConfig([CameraInfo(camera_index=0, camera_position=(0.0, 0.0, 3.0))], gc), None)


def _noisy_diff(rng, sigma):
    """|difference| of two frames whose per-pixel noise (of the difference) is `sigma`."""
    return np.rint(np.abs(rng.normal(0, 1, sigma.shape) * sigma)).astype(np.float32)


def test_off_by_default_and_identical_to_the_plain_kernel():
    assert parse_grid({}).noise_sigma == 0.0
    rng = np.random.default_rng(0)
    diff = rng.integers(0, 40, (48, 64)).astype(np.float32)
    p = _proc()
    out = p.filter_noise(0, diff, 0.0)
    assert out is diff and p.cast_threshold == 10.0
    p._cast(0, out)
    ref = np.zeros((20, 20, 20), np.float32)
    fx, fy = geometry.focal_lengths(64, 48, 60.0)
    raycast.cast_frame(ref, diff, 10.0, (0.0, 0.0, 3.0), geometry.rotation_rtsp(0, 0, 0), fx, fy,
                       np.zeros(3, np.float32), 0.1, p.gc.distance_attenuation, raycast.RAY_RTSP)
    assert ref.any() and ref.tobytes() == p.grid.tobytes()


def test_ignores_noisy_edges_but_keeps_faint_motion_on_flat_areas():
    rng = np.random.default_rng(1)
    sigma = np.full((60, 80), 1.5)
    sigma[:, 40:] = 10.0  # a noisy edge region
    m = NoiseModel(k=6.0, floor=8.0, learn_seconds=3.0)
    for i in range(100):
        m.learn(_noisy_diff(rng, sigma), i / 30)
    assert m.ready
    d = _noisy_diff(rng, sigma)
    d[10:12, 10:12] = 15.0  # fainter than the noise over there, but real here
    out = m.apply(d)
    assert (out[10:12, 10:12] == 15.0).all()
    out[10:12, 10:12] = 0
    assert np.count_nonzero(out) <= 3
    assert np.count_nonzero(d[:, 40:] > 15) > 100  # a plain threshold of 15 would have fired here


def test_floor_for_pixels_without_noise_and_frozen_after_learning():
    m = NoiseModel(k=6.0, floor=8.0, learn_seconds=1.0)
    for i in range(40):
        m.learn(np.zeros((4, 4), np.float32), i / 30)
    assert (m.threshold == 8.0).all()
    assert not m.apply(np.full((4, 4), 8.0, np.float32)).any()
    for _ in range(50):  # motion in one place for ever is still motion
        assert (m.apply(np.full((4, 4), 20.0, np.float32)) == 20.0).all()


def test_nothing_is_cast_while_learning():
    p = _proc(noise_sigma=6.0, noise_learn_seconds=1.0)
    assert p.cast_threshold == 0.0
    d = np.full((48, 64), 3.0, np.float32)
    outs = [p.filter_noise(0, d, i / 30) for i in range(40)]
    assert outs[0] is None and outs[-1] is not None and not outs[-1].any()
