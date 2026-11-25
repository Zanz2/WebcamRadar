"""End-to-end calibration demo on synthetic cameras (no hardware needed).

Builds three virtual cameras with known lenses and placements around a board on
the floor, renders every image the real workflow would produce, then runs the
same steps you would (lens -> placement -> apply -> check) and compares the
result with the ground truth.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from ..config import load_config
from . import apply as apply_mod
from . import check as check_mod
from . import extrinsics, intrinsics
from .board import BoardDetector, make_board
from .model import Lens, matrix_from_euler, pose_pipeline_to_cv, euler_from_matrix
from .synth import random_calibration_poses, render_board_view

W, H = 1280, 720


def look_at(C, target) -> np.ndarray:
    """World-from-camera R (pipeline axes: x right, y up, looking along -z)."""
    fwd = np.asarray(target, float) - np.asarray(C, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.column_stack([right, up, -fwd])


def _obj_pose(C, R, M, m):
    """Camera pose relative to the board's own (OpenCV) frame, given world = M @ obj + m."""
    rv, tv = pose_pipeline_to_cv(C, R)
    R_cv, _ = cv2.Rodrigues(rv)
    r_obj, _ = cv2.Rodrigues(R_cv @ M)
    return r_obj.ravel(), R_cv @ m + tv


def scene(out: Path, seed: int = 0):
    """Write board, captures, placement stills, check stills and a rough config."""
    rng = np.random.default_rng(seed)
    spec = make_board(out / "board", squares_x=5, squares_y=4, paper="A2")
    det = BoardDetector(spec)
    bw, bh = spec.squares_x * spec.square_mm / 1000, spec.squares_y * spec.square_mm / 1000
    centre = np.array([bw / 2, bh / 2, 0.0])
    truth = {
        0: (Lens("standard", W, H, [[675, 0, 645], [0, 677, 355], [0, 0, 1]], [-0.30, 0.10, 0.0006, -0.0004, -0.015]),
            np.array([0.4, -1.1, 1.0])),
        1: (Lens("fisheye", W, H, [[560, 0, 636], [0, 561, 364], [0, 0, 1]], [0.04, -0.015, 0.003, -0.0008]),
            np.array([1.5, 0.3, 0.9])),
        2: (Lens("standard", W, H, [[680, 0, 630], [0, 681, 368], [0, 0, 1]], [-0.27, 0.08, -0.0005, 0.0003, -0.01]),
            np.array([-0.8, 1.2, 1.1])),
    }
    A = np.diag([1.0, -1.0, -1.0])  # board frame -> floor world (see board.py)
    floor = (A, np.array([0.0, bh, 0.0]))
    a = math.radians(25)  # check placement: raised 0.3 m on a box, turned 25 deg, shifted
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    raised = (Rz @ A, Rz @ np.array([0.0, bh, 0.0]) + np.array([0.08, 0.05, 0.30]))
    cams = []
    for ci, (lens, C) in truth.items():
        R = look_at(C, centre)
        for i, (r, t) in enumerate(random_calibration_poses(spec, lens, 20, rng, edge=0.05)):
            p = out / "captures" / f"cam{ci}" / f"calib_{i:03d}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), render_board_view(spec, lens, r, t, noise=1.5, rng=rng))
        for folder, placement in (("poses", floor), ("check", raised)):
            r, t = _obj_pose(C, R, *placement)
            (out / folder).mkdir(exist_ok=True)
            cv2.imwrite(str(out / folder / f"cam{ci}.png"), render_board_view(spec, lens, r, t, noise=1.0, rng=rng))
        yaw, pitch, roll = euler_from_matrix(R)
        cams.append({"camera_index": ci, "rtsp_url": f"rtsp://USER:PASSWORD@192.168.1.10{ci}/stream",
                     # deliberately rough hand measurements, to be replaced by calibration
                     "camera_position": [round(float(v) + 0.1, 1) for v in C], "yaw": round(yaw / 10) * 10,
                     "pitch": round(pitch / 10) * 10, "roll": 0.0, "fov_degrees": 90.0})
        truth[ci] = (lens, C, R)
    grid = {"size": [120, 120, 60], "voxel_size": 0.02, "center": [0.3, 0.2, 0.6]}
    (out / "cameras.json").write_text(json.dumps({"grid": grid, "cameras": cams}, indent=2) + "\n")
    return spec, det, truth


def run(out_dir: str | Path, seed: int = 0, verbose: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    say("1) Rendering a synthetic scene (board, 3 cameras: 2 standard ~87 deg, 1 fisheye ~130 deg)...")
    spec, det, truth = scene(out, seed)
    (out / "results").mkdir(exist_ok=True)
    report = {"cameras": {}}
    for ci, (lens_t, C_t, R_t) in truth.items():
        res, scores = intrinsics.calibrate_folder(out / "captures" / f"cam{ci}", det, model="auto")
        intrinsics.save(res, out / "results" / f"cam{ci}_intrinsics.json")
        pose = extrinsics.pose_from_board(res.lens, det, cv2.imread(str(out / "poses" / f"cam{ci}.png")))
        extrinsics.save(pose, out / "results" / f"cam{ci}_pose.json")
        rot_err = math.degrees(math.acos(np.clip((np.trace(pose.R @ R_t.T) - 1) / 2, -1, 1)))
        pos_err = float(np.linalg.norm(pose.position - C_t))
        report["cameras"][ci] = {"model": res.lens.model + ("+rational" if res.rational else ""),
                                 "true_model": lens_t.model, "lens_rms_px": res.rms,
                                 "position_error_m": pos_err, "rotation_error_deg": rot_err,
                                 "pose_rms_px": pose.rms_px, "warnings": pose.warnings}
        say(f"2) camera {ci}: lens {report['cameras'][ci]['model']} (true {lens_t.model}), rms {res.rms:.3f} px | "
            f"placement error {pos_err * 1000:.1f} mm, {rot_err:.3f} deg")
    log = apply_mod.apply(out / "cameras.json", out / "results", out / "cameras.calibrated.json")
    say("3) apply:\n   " + "\n   ".join(log))
    cfg = load_config(out / "cameras.calibrated.json")
    images = {ci: cv2.imread(str(out / "check" / f"cam{ci}.png")) for ci in truth}
    rep = check_mod.check(cfg.cameras, images, det)
    say("4) check on an independent board placement (raised 30 cm, turned 25 deg):\n   "
        + rep.summary(cfg.grid.voxel_size).replace("\n", "\n   "))
    report["check"] = rep
    uncal = check_mod.check(load_config(out / "cameras.json").cameras, images, det)
    say(f"   (for comparison, the rough hand-measured config: ray miss median {uncal.miss_median_m * 1000:.0f} mm, "
        f"95% {uncal.miss_p95_m * 1000:.0f} mm)")
    report["check_uncalibrated"] = uncal
    return report
