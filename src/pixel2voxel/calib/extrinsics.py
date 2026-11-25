"""Camera placement (position + orientation in one shared world frame).

Two ways to get the world points a camera sees:
  * board     -- the ChArUco board lying face-up on the floor; its corners are the
                 world points (frame defined in board.py: origin at the pattern's
                 bottom-left corner, z up)
  * landmarks -- your own measured points (tape / laser rangefinder), for spaces
                 too big for a board. World coordinates from a CSV, image positions
                 from a CSV or clicked with `click_landmarks`.
All cameras must use the same board placement / the same landmark coordinates.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .board import BoardDetector
from .model import Lens, euler_from_matrix, pose_cv_to_pipeline, pose_pipeline_to_cv


@dataclass
class PoseResult:
    position: np.ndarray  # camera centre, world units (metres)
    R: np.ndarray  # world-from-camera, pipeline axes
    rms_px: float
    n_points: int
    method: str
    warnings: list[str]

    @property
    def yaw_pitch_roll(self) -> tuple[float, float, float]:
        return euler_from_matrix(self.R)

    def to_dict(self) -> dict:
        y, p, r = self.yaw_pitch_roll
        return {"camera_position": [round(float(v), 6) for v in self.position],
                "yaw": round(y, 6), "pitch": round(p, 6), "roll": round(r, 6),
                "R": self.R.round(9).tolist(), "rms_px": round(self.rms_px, 4),
                "points": self.n_points, "method": self.method, "warnings": self.warnings}


def solve_pose(lens: Lens, world: np.ndarray, pixels: np.ndarray, planar: bool, method: str) -> PoseResult:
    """solvePnP on undistorted (normalized) points, so both lens models share one path."""
    world = np.asarray(world, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if len(world) < 4:
        raise ValueError(f"need at least 4 points, got {len(world)}")
    norm = lens.normalized(pixels)
    K1 = np.eye(3)
    flag = cv2.SOLVEPNP_IPPE if planar else cv2.SOLVEPNP_SQPNP
    ok, rvec, tvec = cv2.solvePnP(world, norm, K1, None, flags=flag)
    if not ok:
        raise RuntimeError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(world, norm, K1, None, rvec, tvec)
    C, R = pose_cv_to_pipeline(rvec, tvec)
    # Reprojection error in real (distorted) pixels.
    rv, tv = pose_pipeline_to_cv(C, R)
    R_cv, _ = cv2.Rodrigues(rv)
    proj = lens.project((R_cv @ world.T).T + tv)
    err = float(np.sqrt(np.mean(np.sum((proj - pixels) ** 2, axis=1))))
    warnings = []
    if len(world) < (12 if planar else 6):
        warnings.append(f"only {len(world)} points; aim for {'12+ board corners' if planar else '6+ landmarks'}")
    if err > 2.0:
        warnings.append(f"reprojection error {err:.2f} px is high: check the points / lens calibration")
    return PoseResult(C, R, err, len(world), method, warnings)


def pose_from_board(lens: Lens, det: BoardDetector, image: np.ndarray, origin=(0.0, 0.0, 0.0),
                    min_corners: int = 6) -> PoseResult:
    if (image.shape[1], image.shape[0]) != (lens.width, lens.height):
        lens = lens.for_size(image.shape[1], image.shape[0])
    corners, ids = det.detect_upscaled(image)
    if len(ids) < min_corners:
        raise RuntimeError(f"board not found (only {len(ids)} corners). Is it fully in view and in focus?")
    res = solve_pose(lens, det.world_points(ids, origin), corners, planar=True, method="board")
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    frac = (x1 - x0) * (y1 - y0) / (lens.width * lens.height)
    if frac < 0.01:
        res.warnings.append(f"board covers only {frac * 100:.1f}% of the image: use a bigger board or landmarks "
                            "for a better pose")
    if res.position[2] < 0:
        res.warnings.append("camera came out BELOW the board: was it upside down / photographed from under it?")
    return res


# ----------------------------------------------------------------------------- landmarks


def read_landmarks(path: str | Path) -> dict[str, np.ndarray]:
    """CSV with header name,x,y,z (metres, your world frame)."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["name"].strip()] = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
    return out


def read_image_points(path: str | Path) -> dict[str, np.ndarray]:
    """CSV with header name,u,v (pixels in that camera's image)."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["name"].strip()] = np.array([float(row["u"]), float(row["v"])])
    return out


def pose_from_landmarks(lens: Lens, landmarks: dict, image_points: dict) -> PoseResult:
    names = [n for n in image_points if n in landmarks]
    missing = [n for n in image_points if n not in landmarks]
    if missing:
        raise ValueError(f"image points without world coordinates: {missing}")
    world = np.array([landmarks[n] for n in names])
    pix = np.array([image_points[n] for n in names])
    planar = bool(np.ptp(world[:, 2]) < 1e-9 and len(names) >= 4)
    res = solve_pose(lens, world, pix, planar=planar, method="landmarks")
    if planar:
        res.warnings.append("all landmarks are at the same height; points at different heights constrain the pose better")
    return res


def click_landmarks(image_path: str | Path, landmarks: dict, out_csv: str | Path) -> int:
    """Interactive: click each landmark in the image. Writes name,u,v CSV; returns the count.

    Keys: left-click = place, n = not visible / skip, u = undo, q / Esc = finish early.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    names = list(landmarks)
    placed: list[tuple[str, float, float]] = []
    state = {"i": 0, "click": None}
    win = "click landmarks"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while state["i"] < len(names):
        view = img.copy()
        for name, u, v in placed:
            cv2.drawMarker(view, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(view, name, (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        prompt = f"click '{names[state['i']]}'  ({state['i'] + 1}/{len(names)})   n=skip  u=undo  q=done"
        cv2.putText(view, prompt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if state["click"] is not None:
            placed.append((names[state["i"]], *state["click"]))
            state["click"] = None
            state["i"] += 1
        elif key == ord("n"):
            state["i"] += 1
        elif key == ord("u") and placed:
            last = placed.pop()
            state["i"] = names.index(last[0])
        elif key in (ord("q"), 27):
            break
    cv2.destroyWindow(win)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "u", "v"])
        w.writerows([(n, f"{u:.1f}", f"{v:.1f}") for n, u, v in placed])
    return len(placed)


def save(result: PoseResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result.to_dict(), indent=2) + "\n")
