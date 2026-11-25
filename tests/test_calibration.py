import json
import math
import re

import cv2
import numpy as np
import pytest

from pixel2voxel import geometry
from pixel2voxel.calib import apply as apply_mod
from pixel2voxel.calib import extrinsics
from pixel2voxel.calib.board import BoardDetector, make_board
from pixel2voxel.calib.capture import capture, snapshot
from pixel2voxel.calib.demo import look_at
from pixel2voxel.calib.model import (Lens, euler_from_matrix, matrix_from_euler, pose_cv_to_pipeline,
                                     pose_pipeline_to_cv)
from pixel2voxel.calib.synth import render_board_view
from pixel2voxel.config import AppConfig, CameraInfo, GridConfig
from pixel2voxel.rtsp import Processor

STD = Lens("standard", 640, 360, [[340, 0, 322], [0, 341, 178], [0, 0, 1]], [-0.30, 0.10, 0.0006, -0.0004, -0.015])
FISH = Lens("fisheye", 640, 360, [[280, 0, 318], [0, 281, 182], [0, 0, 1]], [0.04, -0.015, 0.003, -0.0008])


@pytest.fixture(scope="module")
def board(tmp_path_factory):
    d = tmp_path_factory.mktemp("board")
    spec = make_board(d, squares_x=7, squares_y=5, paper="A4")
    return d, spec, BoardDetector(spec)


# ----------------------------------------------------------------------------- frames & rotations

def test_euler_roundtrip_matches_pipeline_rotation():
    rng = np.random.default_rng(0)
    angles = list(rng.uniform([-180, -89, -180], [180, 89, 180], (200, 3))) + [(30, 90, 0), (-40, -90, 0), (0, 0, 0)]
    for y, p, r in angles:
        R = matrix_from_euler(y, p, r)
        assert np.allclose(matrix_from_euler(*euler_from_matrix(R)), R, atol=1e-9)
        assert np.allclose(geometry.rotation_rtsp(y, p, r), R, atol=1e-6)  # same convention as the pipeline


def test_pose_conversion_roundtrip():
    C, R = np.array([1.0, -2.0, 1.5]), look_at([1.0, -2.0, 1.5], [0, 0, 0])
    rv, tv = pose_pipeline_to_cv(C, R)
    C2, R2 = pose_cv_to_pipeline(rv, tv)
    assert np.allclose(C, C2) and np.allclose(R, R2)


def test_lens_resolution_rules():
    assert STD.for_size(640, 360) is STD
    half = STD.for_size(320, 180)
    assert np.allclose(half.K[:2], STD.K[:2] / 2)
    with pytest.raises(ValueError, match="aspect ratio"):
        STD.for_size(640, 480)


# ----------------------------------------------------------------------------- the key integration test

@pytest.mark.parametrize("lens", [STD, FISH], ids=["standard", "fisheye"])
def test_calibrated_camera_rays_hit_the_point_through_the_real_pipeline(lens):
    """Project a known 3D point with OpenCV (distorted), light that spot, then run the
    pipeline's own undistort -> difference -> numba kernel. The voxel peak must land on
    the point. Catches any axis/sign mistake in the OpenCV <-> pipeline conversion and
    any mix-up between K and the undistorted new_K."""
    C = np.array([0.4, -1.6, 1.2])
    for P in (np.array([0.1, 0.2, 0.3]), np.array([-0.35, 0.05, 0.1]), np.array([0.3, 0.45, 0.6])):
        R = look_at(C, [0.0, 0.0, 0.3])
        rv, tv = pose_pipeline_to_cv(C, R)
        R_cv, _ = cv2.Rodrigues(rv)
        u, v = lens.project(R_cv @ P + tv)[0]
        assert 20 < u < lens.width - 20 and 20 < v < lens.height - 20
        yaw, pitch, roll = euler_from_matrix(R)
        cam = CameraInfo(0, "", tuple(C), yaw, pitch, roll, 90.0, intrinsics=lens.to_dict())
        gc = GridConfig(Nx=60, Ny=60, Nz=60, voxel_size=0.02, center=tuple(P), motion_threshold=5.0,
                        distance_attenuation=0.0)
        proc = Processor(AppConfig([cam], gc), publisher=None)
        dark = np.zeros((lens.height, lens.width, 3), np.uint8)
        lit = dark.copy()
        cv2.circle(lit, (int(round(u)), int(round(v))), 2, (255, 255, 255), -1)
        a, _ = proc.prepare(0, dark)
        b, _ = proc.prepare(0, lit)
        proc._cast(0, np.abs(b - a))
        assert proc.grid.max() > 0
        # The peak column of the grid must pass through the point: the voxel at P is hit.
        idx = tuple(int(i) for i in np.floor((P - (P - 0.6)) / 0.02))  # P is the grid centre -> (30, 30, 30)
        neighbourhood = proc.grid[idx[0] - 1:idx[0] + 2, idx[1] - 1:idx[1] + 2, idx[2] - 1:idx[2] + 2]
        assert neighbourhood.max() > 0.5 * proc.grid.max()


def test_uncalibrated_camera_is_unchanged_by_the_lens_code():
    proc = Processor(AppConfig([CameraInfo(0)], GridConfig(Nx=4, Ny=4, Nz=4)), publisher=None)
    frame = np.random.default_rng(0).integers(0, 255, (30, 40, 3), dtype=np.uint8)
    gray, shown = proc.prepare(0, frame)
    assert np.array_equal(gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    assert shown is frame


# ----------------------------------------------------------------------------- board

def test_board_files_and_world_frame(board):
    d, spec, det = board
    corners, ids = det.detect(cv2.imread(str(d / "board.png")))
    assert len(ids) == spec.n_corners
    pdf = (d / "board.pdf").read_bytes()
    w, h = (float(x) / 72 * 25.4 for x in re.search(rb"/MediaBox \[0 0 ([\d.]+) ([\d.]+)\]", pdf).groups())
    assert {round(w), round(h)} == {210, 297}
    # Corner 0 is one square in from the pattern's TOP-left; in the floor frame (origin at
    # the BOTTOM-left, y up the page, z up) that is (sq, height - sq, 0).
    sq = spec.square_mm / 1000
    assert np.allclose(det.world_points([0])[0], [sq, spec.squares_y * sq - sq, 0.0])


# ----------------------------------------------------------------------------- placement

def test_pose_from_board_on_the_floor(board):
    _, spec, det = board
    bw, bh = spec.squares_x * spec.square_mm / 1000, spec.squares_y * spec.square_mm / 1000
    for lens in (STD, FISH):
        C = np.array([0.14, -0.12, 0.22])  # close enough that the 640x360 test image resolves the markers
        R = look_at(C, [bw / 2, bh / 2, 0])
        rv, tv = pose_pipeline_to_cv(C, R)
        R_cv, _ = cv2.Rodrigues(rv)
        A, m = np.diag([1.0, -1.0, -1.0]), np.array([0, bh, 0])
        r_obj, _ = cv2.Rodrigues(R_cv @ A)
        img = render_board_view(spec, lens, r_obj.ravel(), R_cv @ m + tv)
        res = extrinsics.pose_from_board(lens, det, img)
        assert np.linalg.norm(res.position - C) < 0.003
        assert res.position[2] > 0  # above the floor
        rot_err = math.degrees(math.acos(np.clip((np.trace(res.R @ R.T) - 1) / 2, -1, 1)))
        assert rot_err < 0.2


def test_pose_from_landmarks():
    rng = np.random.default_rng(3)
    C = np.array([3.0, -4.0, 2.5])
    R = look_at(C, [0, 0, 1])
    rv, tv = pose_pipeline_to_cv(C, R)
    R_cv, _ = cv2.Rodrigues(rv)
    world = {f"p{i}": rng.uniform([-1.5, -1.5, 0], [1.5, 1.5, 2.5]) for i in range(8)}
    pix = {n: STD.project(R_cv @ X + tv)[0] + rng.normal(0, 0.3, 2) for n, X in world.items()}
    res = extrinsics.pose_from_landmarks(STD, world, pix)
    assert np.linalg.norm(res.position - C) < 0.05
    assert res.rms_px < 1.0


# ----------------------------------------------------------------------------- capture tools

def _board_video(path, spec, lens, holds=4, hold_frames=8):
    """A board that holds still at `holds` spots, sliding between them."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (lens.width, lens.height))
    c = np.array([spec.squares_x, spec.squares_y, 0]) * spec.square_mm / 2000
    spots = [(-0.06, -0.03), (0.06, -0.02), (-0.05, 0.04), (0.07, 0.04)][:holds]
    prev = None
    for sx, sy in spots:
        pos = np.array([sx, sy, 0.45])
        steps = [pos] * hold_frames if prev is None else \
            [prev + (pos - prev) * k / 5 for k in range(1, 6)] + [pos] * hold_frames
        for p in steps:
            img = render_board_view(spec, lens, np.zeros(3), p - c, supersample=1)
            out.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        prev = pos
    out.release()


def test_capture_saves_only_still_and_new_views(board, tmp_path):
    _, spec, det = board
    video = tmp_path / "board.mp4"
    _board_video(video, spec, STD)
    n = capture(str(video), tmp_path / "caps", det, target=10, gui=False)  # default 1 s, video clock
    assert n == 4  # one per still position; nothing saved while sliding
    assert len(list((tmp_path / "caps").glob("*.png"))) == 4


def test_capture_counts_a_tilt_in_place_as_a_new_view(board, tmp_path):
    """Tilted views pin down focal length vs distortion; tilting without moving must save."""
    _, spec, det = board
    path = tmp_path / "tilts.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (STD.width, STD.height))
    c = np.array([spec.squares_x, spec.squares_y, 0]) * spec.square_mm / 2000
    centre = np.array([0.0, 0.0, 0.45])
    for k, (ax, ay) in enumerate([(0, 0), (25, 0), (-25, 0), (0, 25)]):
        R, _ = cv2.Rodrigues(np.array([math.radians(ax), math.radians(ay), 0.0]))
        rvec = cv2.Rodrigues(R)[0].ravel()
        img = cv2.cvtColor(render_board_view(spec, STD, rvec, centre - R @ c, supersample=1), cv2.COLOR_GRAY2BGR)
        if k:
            for _ in range(4):  # brief "moving" frames between tilts
                out.write(np.roll(img, 7, axis=1))
        for _ in range(10):
            out.write(img)
    out.release()
    assert capture(str(path), tmp_path / "caps", det, target=10, gui=False) == 4


def test_snapshot_averages_frames(board, tmp_path):
    _, spec, det = board
    video = tmp_path / "board.mp4"
    _board_video(video, spec, STD, holds=1)
    found = snapshot(str(video), tmp_path / "still.png", frames=5, det=det)
    assert found >= 0.8 * spec.n_corners  # mp4v compression can cost a corner or two at the edges
    assert cv2.imread(str(tmp_path / "still.png")).shape == (STD.height, STD.width, 3)


# ----------------------------------------------------------------------------- apply

def test_apply_writes_intrinsics_and_pose_and_keeps_the_rest(tmp_path):
    cfg = {"grid": {"size": 10}, "cameras": [{"camera_index": 0, "rtsp_url": "rtsp://u:p@h/s", "yaw": 1.0},
                                              {"camera_index": 1, "rtsp_url": "rtsp://u:p@h/t"}]}
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    res = tmp_path / "results"
    res.mkdir()
    (res / "cam0_intrinsics.json").write_text(json.dumps({**STD.to_dict(), "rms_px": 0.1}))
    (res / "cam0_pose.json").write_text(json.dumps({"camera_position": [1, 2, 3], "yaw": 10, "pitch": -20, "roll": 5,
                                                    "method": "board", "rms_px": 0.2}))
    log = apply_mod.apply(tmp_path / "c.json", res, tmp_path / "out.json")
    out = json.loads((tmp_path / "out.json").read_text())
    c0, c1 = out["cameras"]
    assert c0["rtsp_url"] == "rtsp://u:p@h/s" and c0["camera_position"] == [1, 2, 3] and c0["yaw"] == 10
    assert Lens.from_dict(c0["intrinsics"]).model == "standard"
    assert "intrinsics" not in c1 and "unchanged" in log[1]
    assert "rtsp://" not in "\n".join(log)  # never echo credentials


# ----------------------------------------------------------------------------- whole workflow

def test_demo_end_to_end(tmp_path):
    from pixel2voxel.calib import demo
    rep = demo.run(tmp_path, verbose=False)
    for cam in rep["cameras"].values():
        assert cam["position_error_m"] < 0.015 and cam["rotation_error_deg"] < 0.5
    assert rep["check"].miss_p95_m < 0.005  # rays meet within 5 mm
    assert rep["check"].spacing_err_median_pct < 1.0
    assert rep["check_uncalibrated"].miss_median_m > 20 * rep["check"].miss_median_m


def test_lens_solve_warns_about_straight_on_views(board):
    """All views facing the camera fit with a tiny error but can't pin down the focal length;
    the solve must say so (this happened with a real webcam). Varied tilts must not warn."""
    from pixel2voxel.calib.intrinsics import View, calibrate
    from pixel2voxel.calib.synth import random_calibration_poses
    _, spec, det = board
    rng = np.random.default_rng(5)

    def views_for(max_tilt):
        out = []
        for i, (r, t) in enumerate(random_calibration_poses(spec, STD, 16, rng, max_tilt=max_tilt, edge=0.1)):
            c, ids = det.detect(render_board_view(spec, STD, r, t, noise=1.0, rng=rng))
            if len(ids) >= 8:
                out.append(View(f"v{i}", det.object_points(ids), c))
        return out

    flat = calibrate(views_for(3.0), (STD.width, STD.height))
    assert any("tilted" in q for q in flat.quality)
    varied = calibrate(views_for(40.0), (STD.width, STD.height))
    assert not any("tilted" in q for q in varied.quality)
