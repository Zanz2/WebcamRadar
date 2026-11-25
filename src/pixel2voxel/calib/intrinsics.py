"""Lens calibration from photos of the ChArUco board.

Two models:
  standard -- OpenCV's pinhole + radial/tangential distortion; with
              `rational=True` (CALIB_RATIONAL_MODEL) it copes with wide lenses
  fisheye  -- OpenCV's equidistant fisheye model, usually better above ~120 deg
`model="auto"` fits both and keeps the one with the lower reprojection error.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .board import BoardDetector
from .model import Lens

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(eq=False)  # identity comparison: fields are numpy arrays
class View:
    name: str
    obj: np.ndarray  # (N, 3) board coords, metres
    img: np.ndarray  # (N, 2) pixels


@dataclass
class IntrinsicsResult:
    lens: Lens
    rms: float  # reprojection error, pixels
    per_view: dict[str, float]
    used: list[str]
    dropped: dict[str, str] = field(default_factory=dict)
    rational: bool = False
    uncovered: list[str] = field(default_factory=list)  # image regions with no board corners
    std: dict[str, float] = field(default_factory=dict)  # 1-sigma uncertainty of fx, fy, cx, cy (standard model)
    quality: list[str] = field(default_factory=list)  # human-readable problems with the capture set

    @property
    def n_params(self) -> int:
        return 4 if self.lens.model == "fisheye" else (8 if self.rational else 5)

    def to_dict(self) -> dict:
        return {**self.lens.to_dict(), "rms_px": round(self.rms, 4), "rational": self.rational,
                "views_used": len(self.used), "uncovered_regions": self.uncovered,
                "std_px": {k: round(v, 2) for k, v in self.std.items()}, "quality_warnings": self.quality,
                "per_view_rms_px": {k: round(v, 3) for k, v in self.per_view.items()}, "dropped": self.dropped}


REGIONS = {(0, 0): "top-left", (1, 0): "top", (2, 0): "top-right", (0, 1): "left", (1, 1): "centre",
           (2, 1): "right", (0, 2): "bottom-left", (1, 2): "bottom", (2, 2): "bottom-right"}


def uncovered_regions(views, size, min_corners: int = 4) -> list[str]:
    """Which of the 9 image regions (3x3) got fewer than `min_corners` corners across all views.

    Distortion is only measured where the board was seen; elsewhere it is
    extrapolated, which matters most in the corners of a wide lens.
    """
    counts = np.zeros((3, 3), int)
    for v in views:
        cx = np.clip((v.img[:, 0] / size[0] * 3).astype(int), 0, 2)
        cy = np.clip((v.img[:, 1] / size[1] * 3).astype(int), 0, 2)
        np.add.at(counts, (cy, cx), 1)
    return [REGIONS[(x, y)] for y in range(3) for x in range(3) if counts[y, x] < min_corners]


def collect_views(images: list[Path], det: BoardDetector, min_corners: int = 10):
    """Detect the board in every image. Returns (views, image_size, dropped{name: reason})."""
    views, dropped, size = [], {}, None
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            dropped[p.name] = "unreadable"
            continue
        if size is None:
            size = (img.shape[1], img.shape[0])
        elif (img.shape[1], img.shape[0]) != size:
            dropped[p.name] = f"size {img.shape[1]}x{img.shape[0]} != {size[0]}x{size[1]}"
            continue
        corners, ids = det.detect(img)
        if len(ids) < min_corners:
            dropped[p.name] = f"only {len(ids)} corners"
            continue
        views.append(View(p.name, det.object_points(ids), corners))
    return views, size, dropped


def _calibrate_standard(views, size, rational):
    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
    obj = [v.obj.astype(np.float32) for v in views]
    img = [v.img.astype(np.float32) for v in views]
    rms, K, dist, rvecs, tvecs, std_in, _, _ = cv2.calibrateCameraExtended(obj, img, size, None, None, flags=flags)
    errs = []
    for v, r, t in zip(views, rvecs, tvecs):
        proj, _ = cv2.projectPoints(v.obj, r, t, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - v.img) ** 2, axis=1)))))
    s = std_in.ravel()
    std = {"fx": float(s[0]), "fy": float(s[1]), "cx": float(s[2]), "cy": float(s[3])}
    return rms, K, dist.ravel(), errs, rvecs, std


def _calibrate_fisheye_once(views, size):
    obj = [v.obj.reshape(-1, 1, 3).astype(np.float64) for v in views]
    img = [v.img.reshape(-1, 1, 2).astype(np.float64) for v in views]
    K, D = np.zeros((3, 3)), np.zeros((4, 1))
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7)
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(obj, img, size, K, D, flags=flags, criteria=crit)
    errs = []
    for v, r, t in zip(views, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(v.obj.reshape(-1, 1, 3), r, t, K, D)
        errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - v.img) ** 2, axis=1)))))
    return rms, K, D.ravel(), errs, rvecs, {}


def _calibrate_fisheye(views, size, dropped):
    """cv2.fisheye.calibrate throws on an ill-conditioned view; drop it and retry."""
    views = list(views)
    while len(views) >= 4:
        try:
            return _calibrate_fisheye_once(views, size), views
        except cv2.error:
            # Find the view whose removal lets the solve succeed (cheap: few dozen views).
            for i in range(len(views)):
                trial = views[:i] + views[i + 1:]
                try:
                    _calibrate_fisheye_once(trial, size)
                except cv2.error:
                    continue
                dropped[views[i].name] = "ill-conditioned for the fisheye solver"
                views = trial
                break
            else:
                raise RuntimeError("fisheye calibration failed on every subset; capture more varied views")
    raise RuntimeError("fewer than 4 usable views for the fisheye model")


def calibrate(views: list[View], size, model: str = "standard", rational: bool = False,
              balance: float = 0.5, outlier_factor: float = 3.0) -> IntrinsicsResult:
    """Fit the lens; views whose error is > outlier_factor x median are dropped once and refit."""
    if len(views) < 4:
        raise RuntimeError(f"need at least 4 views with the board, got {len(views)}")
    dropped: dict[str, str] = {}

    def fit(vs):
        if model == "fisheye":
            (rms, K, D, errs, rv, std), vs = _calibrate_fisheye(vs, size, dropped)
        else:
            rms, K, D, errs, rv, std = _calibrate_standard(vs, size, rational)
        return rms, K, D, errs, vs, rv, std

    rms, K, D, errs, views, rvecs, std = fit(views)
    med = float(np.median(errs))
    limit = max(outlier_factor * med, 0.5)
    keep = [v for v, e in zip(views, errs) if e <= limit]
    if 4 <= len(keep) < len(views):
        for v, e in zip(views, errs):
            if e > limit:
                dropped[v.name] = f"outlier: {e:.2f} px vs median {med:.2f} px"
        rms, K, D, errs, views, rvecs, std = fit(keep)
    lens = Lens(model, size[0], size[1], K, D, balance)
    res = IntrinsicsResult(lens, float(rms), {v.name: e for v, e in zip(views, errs)},
                           [v.name for v in views], dropped, rational, uncovered_regions(views, size), std)
    res.quality = assess(views, rvecs, size, K, std)
    return res


def assess(views, rvecs, size, K, std) -> list[str]:
    """Problems with the capture set that a low reprojection error would hide.

    The classic trap: views that all face the camera straight on can't separate
    focal length from distance, yet still fit with a tiny error.
    """
    w, h = size
    out = []
    tilts = [math.degrees(math.acos(min(1.0, abs(cv2.Rodrigues(r)[0][2, 2])))) for r in rvecs]
    tilted = sum(t >= 15 for t in tilts)
    if tilted < max(3, 0.3 * len(tilts)):
        out.append(f"only {tilted} of {len(tilts)} views have the board tilted 15+ deg (max {max(tilts):.0f} deg): "
                   "tilt the board (or camera) 20-45 deg left/right/up/down -- straight-on views can't "
                   "separate focal length from distance")
    widths = [np.ptp(v.img[:, 0]) / w for v in views]
    if np.median(widths) < 0.15:
        out.append(f"the board is small in most images (median {np.median(widths) * 100:.0f}% of the image width): "
                   "move closer so it spans 25-50% of the width")
    if len(views) < 15:
        out.append(f"only {len(views)} usable views; aim for 20-30")
    if std:
        if std["fx"] > 0.01 * K[0, 0] or max(std["cx"], std["cy"]) > 0.01 * max(w, h):
            out.append(f"uncertain result: fx +-{std['fx']:.0f}, cx +-{std['cx']:.0f}, cy +-{std['cy']:.0f} px "
                       "(1 sigma); more varied views will tighten it")
    off_x, off_y = abs(K[0, 2] - w / 2) / w, abs(K[1, 2] - h / 2) / h
    if max(off_x, off_y) > 0.08:
        out.append(f"optical centre ({K[0, 2]:.0f}, {K[1, 2]:.0f}) is far from the image centre ({w / 2:.0f}, {h / 2:.0f}); "
                   "usually a sign of too little tilt or coverage, rarely a real lens property")
    return out


def calibrate_folder(folder: str | Path, det: BoardDetector, model: str = "auto", balance: float = 0.5,
                     min_corners: int = 10) -> tuple[IntrinsicsResult, dict]:
    """Calibrate from every image in `folder`. Returns (best result, {candidate: rms})."""
    images = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise RuntimeError(f"no images in {folder}")
    views, size, dropped = collect_views(images, det, min_corners)
    if model == "auto":
        candidates = [("standard", False), ("standard", True), ("fisheye", False)]
    elif model == "rational":
        candidates = [("standard", True)]
    else:
        candidates = [(model, False)]
    results, scores = [], {}
    for m, rational in candidates:
        name = "standard+rational" if rational else m
        try:
            r = calibrate(views, size, m, rational, balance)
        except (RuntimeError, cv2.error) as e:
            scores[name] = f"failed: {e}"
            continue
        r.dropped = {**dropped, **r.dropped}
        results.append(r)
        scores[name] = round(r.rms, 4)
    if not results:
        raise RuntimeError(f"calibration failed: {scores}")
    return pick(results), scores


def pick(results: list[IntrinsicsResult], tolerance: float = 0.05) -> IntrinsicsResult:
    """Fewest distortion parameters among models within `tolerance` of the best error.

    A more flexible model always fits the seen points a little better, but it
    extrapolates worse where the board wasn't seen (typically the image corners).
    """
    best = min(r.rms for r in results)
    ok = [r for r in results if r.rms <= best * (1 + tolerance)]
    return min(ok, key=lambda r: (r.n_params, r.rms))


def save(result: IntrinsicsResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result.to_dict(), indent=2) + "\n")


def load_lens(path: str | Path) -> Lens:
    return Lens.from_dict(json.loads(Path(path).read_text()))
