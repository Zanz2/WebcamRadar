import math

import numpy as np
import pytest

from pixel2voxel import geometry, raycast

F32 = np.float32


def reference_dda(cam, d, shape, vs, center):
    """Line-by-line transcription of cast_ray_into_grid (float32), plus the positive-face fix."""
    nx, ny, nz = shape
    vs = F32(vs)
    half = [F32(0.5) * (F32(n) * vs) for n in shape]
    gmin = [F32(center[i]) - half[i] for i in range(3)]
    gmax = [F32(center[i]) + half[i] for i in range(3)]
    t_min, t_max = F32(0), F32(np.inf)
    for i in range(3):
        if abs(d[i]) < F32(1e-12):
            if cam[i] < gmin[i] or cam[i] > gmax[i]:
                return []
        else:
            t1, t2 = F32((gmin[i] - cam[i]) / d[i]), F32((gmax[i] - cam[i]) / d[i])
            t_min, t_max = max(t_min, min(t1, t2)), min(t_max, max(t1, t2))
            if t_min > t_max:
                return []
    t_min = max(t_min, F32(0))
    ijk = [int(F32((cam[i] + t_min * d[i] - gmin[i]) / vs)) for i in range(3)]
    ijk = [min(ijk[i], shape[i] - 1) for i in range(3)]  # the port's positive-face fix
    if any(v < 0 for v in ijk):
        return []
    step = [1 if d[i] >= 0 else -1 for i in range(3)]

    def sdiv(a, b):
        return F32(np.inf) if abs(b) < F32(1e-12) else F32(a / b)

    tmax = [sdiv(gmin[i] + F32(ijk[i] + (1 if step[i] > 0 else 0)) * vs - cam[i], d[i]) for i in range(3)]
    tdelta = [sdiv(vs, abs(d[i])) for i in range(3)]
    t, out = t_min, []
    while t <= t_max:
        out.append((tuple(ijk), float(t)))
        if tmax[0] < tmax[1] and tmax[0] < tmax[2]:
            k = 0
        elif tmax[1] < tmax[2]:
            k = 1
        else:
            k = 2
        ijk[k] += step[k]
        t = tmax[k]
        tmax[k] = F32(tmax[k] + tdelta[k])
        if not all(0 <= ijk[i] < shape[i] for i in range(3)):
            break
    return out


@pytest.mark.parametrize("shape,vs,center", [((16, 16, 16), 6.0, (0, 0, 500)), ((9, 23, 5), 0.37, (1.5, -2.0, 0.3))])
def test_kernel_matches_reference_transcription(shape, vs, center):
    rng = np.random.default_rng(0)
    ext = np.array(shape) * vs
    for _ in range(2000):
        cam = (np.asarray(center) + rng.uniform(-1.2, 1.2, 3) * ext).astype(F32)
        target = (np.asarray(center) + rng.uniform(-0.5, 0.5, 3) * ext).astype(F32)
        d = target - cam
        d = (d / np.linalg.norm(d)).astype(F32)
        idx, dist = raycast.cast_ray(cam, d, shape, vs, center)
        ref = reference_dda(cam, d, shape, vs, center)
        assert [tuple(map(int, r)) for r in idx] == [r[0] for r in ref]
        assert np.array_equal(dist, np.array([r[1] for r in ref], dtype=F32))


def test_axis_aligned_ray_crosses_every_voxel_once():
    idx, dist = raycast.cast_ray((0.5, 0.5, -10.0), (0, 0, 1), (4, 4, 10), 1.0, (2, 2, 5))
    assert [tuple(r) for r in idx] == [(0, 0, k) for k in range(10)]
    assert np.allclose(dist, np.arange(10) + 10.0)


def test_ray_missing_the_grid_visits_nothing():
    idx, _ = raycast.cast_ray((100, 100, 100), (1, 0, 0), (4, 4, 4), 1.0, (0, 0, 0))
    assert len(idx) == 0


def test_cast_frame_rejects_float64_grid():
    with pytest.raises(TypeError):
        raycast.cast_frame(np.zeros((2, 2, 2)), np.zeros((2, 2)), 1, (0, 0, 0), np.eye(3), 1, 1, (0, 0, 0), 1)


def test_rotation_conventions_differ_by_pitch_roll_swap():
    rng = np.random.default_rng(1)
    for y, p, r in rng.uniform(-180, 180, (50, 3)):
        assert np.array_equal(geometry.rotation_batch(y, p, r), geometry.rotation_rtsp(y, r, p))
        R = geometry.rotation_rtsp(y, p, r).astype(np.float64)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_focal_lengths_square_pixels():
    fx, fy = geometry.focal_lengths(1280, 720, 87.0)
    assert fx == pytest.approx(640 / math.tan(math.radians(43.5)), rel=1e-6)
    assert fx == pytest.approx(fy, rel=1e-5)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rays_entering_through_positive_faces_are_not_dropped(axis):
    """Entry through a +x/+y/+z face maps to index N, one past the end, and must be clamped."""
    shape, vs, center = (48, 48, 48), 0.25, (0.0, 0.0, 0.0)
    rng = np.random.default_rng(axis)
    for sign in (+1, -1):
        for _ in range(500):
            cam = rng.uniform(-3, 3, 3).astype(F32)
            cam[axis] = sign * rng.uniform(8, 40)
            d = rng.uniform(-5.9, 5.9, 3).astype(F32) - cam
            d = (d / np.linalg.norm(d)).astype(F32)
            idx, _ = raycast.cast_ray(cam, d, shape, vs, center)
            assert len(idx) > 0
            if sign > 0:
                assert idx[0][axis] == shape[axis] - 1  # enters the last voxel layer
