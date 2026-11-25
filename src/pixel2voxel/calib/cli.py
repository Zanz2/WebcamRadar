"""p2v-calibrate: camera calibration for pixel2voxel. See calibration/README.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


def _url(args) -> str:
    """Camera URL from --url, or from --config + --camera (never printed: it has credentials)."""
    if getattr(args, "url", None):
        return args.url
    from ..config import load_config
    cams = {c.camera_index: c for c in load_config(args.config).cameras}
    if args.camera not in cams:
        sys.exit(f"camera {args.camera} not in {args.config} (has {sorted(cams)})")
    return cams[args.camera].rtsp_url


def _detector(args):
    from .board import BoardDetector, BoardSpec
    return BoardDetector(BoardSpec.load(args.board), getattr(args, "square_mm", None))


def _source(p):
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera", type=int, help="camera_index in --config (the URL is read from the config)")
    g.add_argument("--url", help="stream URL or video file (instead of --config/--camera)")
    p.add_argument("--config", help="camera config JSON (with --camera)")


def cmd_board(a):
    from .board import make_board
    spec = make_board(a.out, a.squares[0], a.squares[1], a.square_mm, a.marker_ratio, a.dictionary, a.paper, a.dpi)
    print(f"Wrote {a.out}/board.pdf, board.png, board.json")
    print(f"  {spec.squares_x}x{spec.squares_y} squares of {spec.square_mm:.2f} mm on {spec.paper} ({spec.dictionary})")
    print("  Print the PDF at 100% / actual size, then measure the 100 mm ruler and one square.")
    print("  If a square isn't exactly the size above, pass --square-mm <measured> to pose-board / check.")


def cmd_capture(a):
    from .capture import capture
    if a.camera is not None and not a.config:
        sys.exit("--camera needs --config")
    n = capture(_url(a), a.out, _detector(a), a.target, a.min_corners, gui=not a.no_gui, auto=not a.manual)
    print(f"{n} images in {a.out}")


def cmd_snapshot(a):
    from .capture import snapshot
    if a.camera is not None and not a.config:
        sys.exit("--camera needs --config")
    det = _detector(a) if a.board else None
    found = snapshot(_url(a), a.out, a.frames, det)
    print(f"Saved {a.out}" + (f" ({found} board corners visible)" if det else ""))


def cmd_intrinsics(a):
    from . import intrinsics
    res, scores = intrinsics.calibrate_folder(a.folder, _detector(a), a.model, a.balance, a.min_corners)
    intrinsics.save(res, a.out)
    k = res.lens.K
    print(f"Candidates (reprojection error, px): {scores}")
    print(f"Chosen: {res.lens.model}{'+rational' if res.rational else ''}  rms {res.rms:.3f} px  "
          f"from {len(res.used)} images  ({len(res.dropped)} dropped)")
    print(f"  fx {k[0, 0]:.1f}  fy {k[1, 1]:.1f}  cx {k[0, 2]:.1f}  cy {k[1, 2]:.1f}  ({res.lens.width}x{res.lens.height})")
    for name, why in res.dropped.items():
        print(f"  dropped {name}: {why}")
    if res.std:
        sd = res.std
        print(f"  uncertainty (1 sigma): fx +-{sd['fx']:.1f}  fy +-{sd['fy']:.1f}  cx +-{sd['cx']:.1f}  cy +-{sd['cy']:.1f} px")
    for q in res.quality:
        print(f"  WARNING: {q}")
    if res.quality:
        print("  -> Capture more views (e.g. `calibration/02_capture_lens.sh N --target 45` adds to the existing ones) "
              "and solve again.")
    if res.uncovered:
        print(f"  WARNING: no board corners in these image regions: {', '.join(res.uncovered)}. "
              "Distortion there is extrapolated; capture more images with the board there.")
    if res.rms > 1.0:
        print("  WARNING: rms > 1 px. Blurry images, a bent board, or the wrong board.json?")
    if a.preview:
        img = cv2.imread(str(Path(a.folder) / res.used[0]))
        cv2.imwrite(a.preview, cv2.hconcat([img, res.lens.undistort(img)]))
        print(f"  Before/after preview: {a.preview} (straight lines should look straight on the right)")
    print(f"Wrote {a.out}")


def cmd_pose_board(a):
    from . import extrinsics, intrinsics
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    res = extrinsics.pose_from_board(intrinsics.load_lens(a.intrinsics), _detector(a), img, a.origin)
    extrinsics.save(res, a.out)
    _print_pose(res, a.out)


def cmd_click(a):
    from . import extrinsics
    n = extrinsics.click_landmarks(a.image, extrinsics.read_landmarks(a.landmarks), a.out)
    print(f"Wrote {n} points to {a.out}")


def cmd_pose_landmarks(a):
    from . import extrinsics, intrinsics
    res = extrinsics.pose_from_landmarks(intrinsics.load_lens(a.intrinsics), extrinsics.read_landmarks(a.landmarks),
                                         extrinsics.read_image_points(a.points))
    extrinsics.save(res, a.out)
    _print_pose(res, a.out)


def _print_pose(res, out):
    y, p, r = res.yaw_pitch_roll
    pos = ", ".join(f"{v:.3f}" for v in res.position)
    print(f"Camera at ({pos}) m   yaw {y:.2f}  pitch {p:.2f}  roll {r:.2f} deg   "
          f"({res.n_points} points, reprojection {res.rms_px:.2f} px)")
    for w in res.warnings:
        print(f"  WARNING: {w}")
    print(f"Wrote {out}")


def cmd_apply(a):
    from .apply import apply
    for line in apply(a.config, a.results, a.out):
        print(line)
    print(f"Wrote {a.out}  (run p2v-rtsp with this file)")


def cmd_check(a):
    from ..config import load_config
    from .check import check
    cfg = load_config(a.config)
    images = {}
    for spec in a.image:
        cam, _, path = spec.partition("=")
        img = cv2.imread(path)
        if img is None:
            sys.exit(f"cannot read {path}")
        images[int(cam)] = img
    rep = check(cfg.cameras, images, _detector(a))
    print(rep.summary(cfg.grid.voxel_size))


def cmd_list_cameras(a):
    import time
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles = []

    def grab(cap, seconds=1.5):
        frame, t_end = None, time.monotonic() + seconds  # let exposure settle
        while time.monotonic() < t_end:
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        return frame

    def add_tile(frame, label):
        h, w = frame.shape[:2]
        tile = cv2.resize(frame, (480, int(480 * h / w)))
        cv2.putText(tile, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        tiles.append(tile)

    from .. import avfcam
    if avfcam.AVAILABLE and not a.opencv:
        # macOS: list by name through AVFoundation (what Photo Booth uses); names don't shift.
        want = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
        for i, d in enumerate(avfcam.list_devices()):
            size = want if want in d["modes"] else d["modes"][0]
            # A name shared by two cameras needs the unique id instead.
            dup = sum(x["name"] == d["name"] for x in avfcam.list_devices()) > 1
            key = f"id={d['id']}" if dup else d["name"]
            url = f"camera:{key}@{size[0]}x{size[1]}"
            modes = ", ".join(f"{w}x{h}" for w, h in d["modes"][:8])
            try:
                frame = grab(avfcam.AVFCapture(key, *size))
            except RuntimeError as e:
                print(f"{d['name']}: cannot open ({e})")
                continue
            if frame is None:
                print(f"{d['name']}: opened but delivered no frames (in use by another app? asleep iPhone?)")
                continue
            path = out / f"camera{i}.png"
            cv2.imwrite(str(path), frame)
            note = "" if want is None or want == size else f"  (no {want[0]}x{want[1]} mode)"
            print(f"{d['name']}\n    rtsp_url: \"{url}\"{note}\n    modes: {modes}\n    preview: {path}")
            add_tile(frame, f"{d['name'][:26]}  {frame.shape[1]}x{frame.shape[0]}")
    else:
        for idx in range(a.max):
            spec = f"device:{idx}" + (f"@{a.size}" if a.size else "")
            from ..rtsp import _open_capture
            cap = _open_capture(spec)
            if not cap.isOpened():
                break  # devices are numbered 0..N-1; probing past the end only prints warnings
            frame = grab(cap)
            if frame is None:
                print(f"device:{idx}  opened but delivered no frames (try the camera:<name> form on macOS)")
                continue
            h, w = frame.shape[:2]
            cv2.imwrite(str(out / f"device{idx}.png"), frame)
            print(f"device:{idx}  {w}x{h}   -> {out / f'device{idx}.png'}")
            add_tile(frame, f"device:{idx}  {w}x{h}")
    if not tiles:
        print("No cameras delivered frames. On macOS, allow camera access for your terminal app in System Settings > "
              "Privacy & Security > Camera, then restart the terminal.")
        return
    hmax = max(t.shape[0] for t in tiles)
    sheet = cv2.hconcat([cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles])
    cv2.imwrite(str(out / "all_cameras.png"), sheet)
    print(f"Contact sheet: {out / 'all_cameras.png'}")
    print("Put the rtsp_url lines into your config. Each camera must keep its size between calibration and running.")


def cmd_test_camera(a):
    import time
    from ..rtsp import _open_capture
    if a.camera is not None and not a.config:
        sys.exit("--camera needs --config")
    cap = _open_capture(_url(a))
    if not cap.isOpened():
        sys.exit("could not open the camera")
    print(f"Streaming for {a.seconds:.0f} s (frames per second, one number per second; 0 = stalled)...")
    t0, per, n, sec, size = time.monotonic(), [], 0, 0, None
    while time.monotonic() - t0 < a.seconds:
        ok, f = cap.read()
        if ok:
            n += 1
            size = f.shape[1::-1]
        s = int(time.monotonic() - t0)
        if s > sec:
            per.append(n)
            print(f"  {s:3d}s: {n:3d} fps", flush=True)
            n, sec = 0, s
    cap.release()
    stalls = sum(v == 0 for v in per)
    first_stall = next((i + 1 for i, v in enumerate(per) if v == 0), None)
    print(f"size {size[0]}x{size[1]}" if size else "no frames at all")
    print("OK: no stalls" if not stalls else f"STALLED: {stalls} of {len(per)} seconds without frames (first at {first_stall}s)")
    drops = getattr(cap, "drops", None)
    if drops:
        print(f"frames macOS dropped, by reason: {drops}")


def cmd_demo(a):
    from . import demo
    demo.run(a.out)
    print(f"\nEverything the demo produced is in {a.out} (same layout as a real calibration).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="p2v-calibrate", description="Camera calibration for pixel2voxel "
                                 "(walkthrough: calibration/README.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("board", help="generate a printable ChArUco board (PDF/PNG + board.json)")
    p.add_argument("--out", default="calibration/work/board")
    p.add_argument("--squares", type=int, nargs=2, default=(7, 5), metavar=("X", "Y"))
    p.add_argument("--square-mm", type=float, help="square size (default: as large as fits the paper)")
    p.add_argument("--marker-ratio", type=float, default=0.75)
    p.add_argument("--paper", default="A3", help="A4, A3, A2, A1, A0, letter, tabloid (default A3)")
    p.add_argument("--dictionary", default="DICT_5X5_100")
    p.add_argument("--dpi", type=int, default=300)
    p.set_defaults(fn=cmd_board)

    p = sub.add_parser("capture", help="collect lens-calibration images from a camera (live view)")
    _source(p)
    p.add_argument("--board", default="calibration/work/board/board.json")
    p.add_argument("--out", required=True)
    p.add_argument("--target", type=int, default=25)
    p.add_argument("--min-corners", type=int, default=8)
    p.add_argument("--manual", action="store_true", help="only save on SPACE (no auto-capture)")
    p.add_argument("--no-gui", action="store_true", help="no window; auto-capture only")
    p.set_defaults(fn=cmd_capture)

    p = sub.add_parser("snapshot", help="save one averaged still (for placement / check)")
    _source(p)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--board", help="board.json, to report how many corners are visible")
    p.set_defaults(fn=cmd_snapshot)

    p = sub.add_parser("intrinsics", help="solve the lens from captured images")
    p.add_argument("folder")
    p.add_argument("--board", default="calibration/work/board/board.json")
    p.add_argument("--model", default="auto", choices=["auto", "standard", "rational", "fisheye"])
    p.add_argument("--balance", type=float, default=0.5, help="0 = crop to valid pixels, 1 = keep full FOV")
    p.add_argument("--min-corners", type=int, default=8)
    p.add_argument("--preview", help="write a before/after undistortion image here")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_intrinsics)

    p = sub.add_parser("pose-board", help="camera placement from the board lying on the floor")
    p.add_argument("--image", required=True)
    p.add_argument("--intrinsics", required=True)
    p.add_argument("--board", default="calibration/work/board/board.json")
    p.add_argument("--square-mm", type=float, help="square size measured on your print")
    p.add_argument("--origin", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                   help="world position of the board's origin corner (default 0 0 0)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_pose_board)

    p = sub.add_parser("click-landmarks", help="click measured landmarks in a still (writes name,u,v CSV)")
    p.add_argument("--image", required=True)
    p.add_argument("--landmarks", required=True, help="CSV name,x,y,z")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_click)

    p = sub.add_parser("pose-landmarks", help="camera placement from measured landmarks")
    p.add_argument("--intrinsics", required=True)
    p.add_argument("--landmarks", required=True, help="CSV name,x,y,z")
    p.add_argument("--points", required=True, help="CSV name,u,v (from click-landmarks)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_pose_landmarks)

    p = sub.add_parser("apply", help="write results into a (new) camera config")
    p.add_argument("--config", required=True)
    p.add_argument("--results", default="calibration/work/results")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_apply)

    p = sub.add_parser("check", help="measure how well the cameras' rays meet (board in a NEW spot)")
    p.add_argument("--config", required=True, help="the calibrated config")
    p.add_argument("--board", default="calibration/work/board/board.json")
    p.add_argument("--square-mm", type=float)
    p.add_argument("--image", action="append", required=True, metavar="CAM=PATH",
                   help="one per camera, e.g. --image 0=check/cam0.png")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("list-cameras", help="find local USB / built-in cameras (device:N) and save a preview of each")
    p.add_argument("--max", type=int, default=6, help="highest device index to try with --opencv (default 6)")
    p.add_argument("--opencv", action="store_true", help="probe OpenCV device:N indices instead of names")
    p.add_argument("--size", help="request a resolution, e.g. 1280x720")
    p.add_argument("--out", default="calibration/work/devices")
    p.set_defaults(fn=cmd_list_cameras)

    p = sub.add_parser("test-camera", help="stream one camera and print frames per second (finds freezes)")
    _source(p)
    p.add_argument("--seconds", type=float, default=30)
    p.set_defaults(fn=cmd_test_camera)

    p = sub.add_parser("demo", help="run the whole workflow on synthetic cameras (no hardware)")
    p.add_argument("--out", default="calibration/demo")
    p.set_defaults(fn=cmd_demo)

    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        sys.exit(f"ERROR: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
