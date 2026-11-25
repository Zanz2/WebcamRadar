"""Synthetic cameras as video files for each stage of the calibration scripts (see run.sh)."""
import json, math, sys, cv2, numpy as np
from pixel2voxel.calib.board import BoardSpec
from pixel2voxel.calib.demo import look_at, _obj_pose
from pixel2voxel.calib.model import Lens, euler_from_matrix
from pixel2voxel.calib.synth import render_board_view, random_calibration_poses
T = sys.argv[1]; spec = BoardSpec.load(f"{T}/work/board/board.json"); rng = np.random.default_rng(1)
W, H = 1280, 720
bw, bh = spec.squares_x * spec.square_mm / 1000, spec.squares_y * spec.square_mm / 1000
centre = np.array([bw / 2, bh / 2, 0.0])
cams = {0: (Lens("standard", W, H, [[675, 0, 645], [0, 677, 355], [0, 0, 1]], [-0.30, 0.10, 0.0006, -0.0004, -0.015]), [0.25, -0.75, 0.75]),
        1: (Lens("fisheye", W, H, [[560, 0, 636], [0, 561, 364], [0, 0, 1]], [0.04, -0.015, 0.003, -0.0008]), [1.05, 0.2, 0.6]),
        2: (Lens("standard", W, H, [[680, 0, 630], [0, 681, 368], [0, 0, 1]], [-0.27, 0.08, -0.0005, 0.0003, -0.01]), [-0.5, 0.85, 0.7])}
A = np.diag([1.0, -1.0, -1.0]); a = math.radians(25)
Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
floor, raised = (A, np.array([0, bh, 0.0])), (Rz @ A, Rz @ np.array([0, bh, 0.0]) + np.array([0.05, 0.03, 0.20]))
def writer(path):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), 10, (W, H)); assert w.isOpened(), path; return w
truth, configs = {}, {k: [] for k in ("lens", "floor", "check", "rough")}
for ci, (lens, C) in cams.items():
    C = np.array(C); R = look_at(C, centre); truth[ci] = {"C": C.tolist(), "R": R.tolist(), "model": lens.model}
    w = writer(f"{T}/cam{ci}_lens.mkv")  # person holding the board still at 22 spots, moving between them
    for r, t in random_calibration_poses(spec, lens, 22, rng, edge=0.05):
        img = cv2.cvtColor(render_board_view(spec, lens, r, t, noise=1.5, rng=rng), cv2.COLOR_GRAY2BGR)
        w.write(np.clip(img.astype(int) + rng.integers(-40, 40), 0, 255).astype(np.uint8))  # "moving": changes
        for _ in range(12): w.write(img)  # held still 1.2 s
    w.release()
    for stage, placement in (("floor", floor), ("check", raised)):
        r, t = _obj_pose(C, R, *placement); img = render_board_view(spec, lens, r, t, rng=rng)
        w = writer(f"{T}/cam{ci}_{stage}.mkv")
        for _ in range(20): w.write(cv2.cvtColor(np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        w.release()
    y, p, r_ = euler_from_matrix(R)
    for k in configs:
        url = f"{T}/cam{ci}_{k}.mkv" if k != "rough" else f"rtsp://USER:PASSWORD@10.0.0.{ci}/s"
        configs[k].append({"camera_index": ci, "rtsp_url": url, "camera_position": [round(v + 0.1, 1) for v in C],
                           "yaw": round(y / 10) * 10, "pitch": round(p / 10) * 10, "roll": 0, "fov_degrees": 90})
grid = {"size": [100, 100, 60], "voxel_size": 0.02, "center": [0.2, 0.15, 0.4]}
for k, c in configs.items():
    json.dump({"grid": grid, "cameras": c}, open(f"{T}/{k}.json", "w"), indent=2)
json.dump(truth, open(f"{T}/truth.json", "w"), indent=2)
print("scene written")
