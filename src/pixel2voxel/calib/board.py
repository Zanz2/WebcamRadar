"""ChArUco calibration board: generation (PNG + exact-scale PDF + JSON spec) and detection.

A ChArUco board is a checkerboard with ArUco markers in the white squares, so
every corner has an ID and partial views still work -- important with several
cameras that each see only part of the board.

World frame defined by a board lying face-up (used for camera placement):
  origin = bottom-left corner of the printed pattern (marked on the print)
  x      = along the bottom edge, to the right
  y      = up the page, toward the top edge
  z      = up, out of the board (OpenCV's own board z points INTO the print)
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

PAPER_MM = {"A4": (210, 297), "A3": (297, 420), "A2": (420, 594), "A1": (594, 841), "A0": (841, 1189),
            "letter": (215.9, 279.4), "tabloid": (279.4, 431.8)}
MARGIN_MM = 12.0
FOOTER_MM = 28.0  # ruler + instructions


@dataclass
class BoardSpec:
    squares_x: int
    squares_y: int
    square_mm: float  # as printed (exact, after rounding to whole pixels at `dpi`)
    marker_mm: float
    dictionary: str = "DICT_5X5_100"
    paper: str = "A4"
    dpi: int = 300

    @classmethod
    def load(cls, path: str | Path) -> "BoardSpec":
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    def cv_board(self, square_mm: float | None = None) -> cv2.aruco.CharucoBoard:
        """OpenCV board in metres. Pass the square size you *measured* on the print if it differs."""
        sq = (square_mm or self.square_mm) / 1000.0
        mk = sq * self.marker_mm / self.square_mm
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), sq, mk, d)

    @property
    def n_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)


class BoardDetector:
    def __init__(self, spec: BoardSpec, square_mm: float | None = None):
        self.spec = spec
        self.board = spec.cv_board(square_mm)
        self._det = cv2.aruco.CharucoDetector(self.board)
        self.height_m = spec.squares_y * self.board.getSquareLength()

    def detect(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(corners[N, 2] float64 pixels, ids[N] int). Empty arrays if not found."""
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cc, ci, _, _ = self._det.detectBoard(gray)
        if cc is None or ci is None or len(cc) == 0:
            return np.empty((0, 2)), np.empty(0, dtype=int)
        return cc.reshape(-1, 2).astype(np.float64), ci.ravel().astype(int)

    def detect_upscaled(self, img: np.ndarray, scales=(2, 3)) -> tuple[np.ndarray, np.ndarray]:
        """Like detect(), but if the board isn't fully found at native size (small, soft markers from a
        low-resolution camera), retry on 2x and 3x enlarged copies and keep the result with the most
        corners. Corners come back in the original image's pixel coordinates. Slower: for single stills,
        not a live video loop."""
        best = self.detect(img)
        total = (self.spec.squares_x - 1) * (self.spec.squares_y - 1)
        for s in scales:
            if len(best[1]) >= total:
                break
            big = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            c, ids = self.detect(big)
            if len(ids) > len(best[1]):
                best = ((c + 0.5) / s - 0.5, ids)  # undo cv2.resize's pixel-centre mapping
        return best

    def object_points(self, ids) -> np.ndarray:
        """Board-frame (OpenCV) 3D corner positions in metres, shape (N, 3)."""
        return np.asarray(self.board.getChessboardCorners(), dtype=np.float64)[np.asarray(ids, dtype=int)]

    def world_points(self, ids, origin=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Corner positions in the floor world frame (see module docstring) + origin offset."""
        o = self.object_points(ids)
        w = np.stack([o[:, 0], self.height_m - o[:, 1], -o[:, 2]], axis=1)
        return w + np.asarray(origin, dtype=np.float64)


# ----------------------------------------------------------------------------- generation


def _fit(paper: str, sx: int, sy: int, square_mm: float | None):
    """Pick orientation (and square size if not given) that fits the paper."""
    w, h = PAPER_MM[paper]
    best = None
    for pw, ph in ((w, h), (h, w)):
        avail_w, avail_h = pw - 2 * MARGIN_MM, ph - 2 * MARGIN_MM - FOOTER_MM
        fit = min(avail_w / sx, avail_h / sy)
        if best is None or fit > best[2]:
            best = (pw, ph, fit)
    pw, ph, fit = best
    sq = fit if square_mm is None else square_mm
    if sq > fit + 1e-9:
        raise ValueError(f"{sx}x{sy} squares of {square_mm} mm don't fit on {paper} (max {fit:.1f} mm)")
    return pw, ph, float(np.floor(sq * 10) / 10)  # whole tenths of a mm


def _px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))


def write_pdf(path: str | Path, gray: np.ndarray, page_w_mm: float, page_h_mm: float) -> None:
    """Minimal one-page PDF showing an 8-bit grayscale raster at an exact physical size.

    Dependency-free: a single Flate-compressed image XObject scaled to the full page.
    """
    h, w = gray.shape
    pt_w, pt_h = page_w_mm / 25.4 * 72, page_h_mm / 25.4 * 72
    img = zlib.compress(np.ascontiguousarray(gray, dtype=np.uint8).tobytes(), 9)
    content = f"q {pt_w:.4f} 0 0 {pt_h:.4f} 0 0 cm /Im0 Do Q".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pt_w:.4f} {pt_h:.4f}] "
         f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").encode(),
        (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray "
         f"/BitsPerComponent 8 /Filter /FlateDecode /Length {len(img)} >>\nstream\n").encode() + img + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    Path(path).write_bytes(bytes(out))


def make_board(out_dir: str | Path, squares_x=7, squares_y=5, square_mm: float | None = None,
               marker_ratio=0.75, dictionary="DICT_5X5_100", paper="A4", dpi=300) -> BoardSpec:
    """Write board.png, board.pdf (print at 100%) and board.json into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pw, ph, sq = _fit(paper, squares_x, squares_y, square_mm)
    sq_px = _px(sq, dpi)
    sq_exact = sq_px * 25.4 / dpi  # what will actually be printed
    spec = BoardSpec(squares_x, squares_y, round(sq_exact, 4), round(sq_exact * marker_ratio, 4), dictionary, paper, dpi)

    page = np.full((_px(ph, dpi), _px(pw, dpi)), 255, np.uint8)
    board_img = spec.cv_board().generateImage((squares_x * sq_px, squares_y * sq_px), marginSize=0)
    bh, bw = board_img.shape
    x0 = (page.shape[1] - bw) // 2
    y0 = _px(MARGIN_MM, dpi) + (page.shape[0] - _px(2 * MARGIN_MM + FOOTER_MM, dpi) - bh) // 2
    page[y0:y0 + bh, x0:x0 + bw] = board_img

    s = dpi / 300.0  # text/line scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Origin marker + axes at the pattern's bottom-left corner.
    ox, oy = x0, y0 + bh
    arrow = _px(18, dpi)
    gap = _px(2, dpi)
    cv2.arrowedLine(page, (ox, oy + gap), (ox + arrow, oy + gap), 0, max(1, int(3 * s)), tipLength=0.25)
    cv2.arrowedLine(page, (ox - gap, oy), (ox - gap, oy - arrow), 0, max(1, int(3 * s)), tipLength=0.25)
    cv2.putText(page, "x", (ox + arrow + gap, oy + int(12 * s)), font, 0.9 * s, 0, max(1, int(2 * s)))
    cv2.putText(page, "y", (ox - _px(5, dpi), oy - arrow - gap), font, 0.9 * s, 0, max(1, int(2 * s)))
    cv2.putText(page, "<- origin (z up)", (ox + arrow + _px(7, dpi), oy + int(12 * s)), font, 0.7 * s, 0,
                max(1, int(2 * s)))

    # 100 mm ruler for checking the print scale.
    ry = page.shape[0] - _px(MARGIN_MM + 14, dpi)
    rx0 = _px(MARGIN_MM, dpi)
    rx1 = rx0 + _px(100, dpi)
    cv2.line(page, (rx0, ry), (rx1, ry), 0, max(1, int(3 * s)))
    for k in range(11):
        x = rx0 + _px(10 * k, dpi)
        cv2.line(page, (x, ry), (x, ry - _px(4 if k % 5 else 7, dpi)), 0, max(1, int(2 * s)))
    cv2.putText(page, "100 mm", (rx1 + _px(3, dpi), ry), font, 0.8 * s, 0, max(1, int(2 * s)))
    lines = [f"Print at 100% / actual size (no 'fit to page'). Measure the ruler and one square.",
             f"{squares_x}x{squares_y} squares, square {spec.square_mm:.2f} mm, marker {spec.marker_mm:.2f} mm, "
             f"{dictionary}. Mount flat and rigid."]
    for i, text in enumerate(lines):
        cv2.putText(page, text, (rx0, ry + _px(7 + 6 * i, dpi)), font, 0.62 * s, 0, max(1, int(1.5 * s)))

    cv2.imwrite(str(out / "board.png"), page)
    write_pdf(out / "board.pdf", page, pw, ph)
    spec.save(out / "board.json")
    return spec
