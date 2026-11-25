# How to use pixel2voxel

Reconstruct moving objects in 3D from several cameras. Each camera's moving pixels are turned into rays,
and the rays build up in a shared voxel grid where they cross.

## 1. Install

You need only [uv](https://docs.astral.sh/uv/). It manages Python and the virtual environment for you.

```bash
uv sync                   # core tools
uv sync --extra space     # only if you want p2v-space (astronomy)
```

Every command below is run as `uv run <command>` from the repository root.

## 2. Configure your cameras

Copy the example and edit it:

```bash
cp config/camera_config.example.json config/my_cameras.json
```

```jsonc
{
  "grid": {
    "size": [100, 100, 60],       // voxels in x, y, z (or one number for a cube)
    "voxel_size": 0.05,           // metres per voxel
    "center": [2.5, 2.5, 1.5],    // world position of the grid centre
    "decay_rate": 0.85,           // fraction of brightness left after 1 s (1.0 = never fades)
    "motion_threshold": 8.0,      // grey-level change (0-255) that counts as motion
    "distance_attenuation": 0.5,  // ray weight = 1 / (1 + this * distance)
    "min_cameras": 0,             // optional: keep only what this many cameras see (0 = off, plain sum)
    "show_debug": true,           // show per-camera motion windows
    "save_interval_seconds": 60,  // how often a .bin snapshot is written
    "publish_interval_seconds": 0.5  // live-stream rate limit for the viewer
  },
  "cameras": [
    {
      "camera_index": 0,
      "rtsp_url": "rtsp://USER:PASSWORD@192.168.1.100:554/stream2",
      "camera_position": [0.0, 0.0, 2.0],   // metres, same world frame as the grid
      "yaw": 0.0, "pitch": -15.0, "roll": 0.0,  // degrees
      "fov_degrees": 130.0                    // horizontal field of view
    }
  ]
}
```

Tips:
- **Grid size.** It costs `x * y * z * 4` bytes of RAM. The default 500³ grid is 500 MB.
- **Camera positions and angles must be accurate.** Small angle errors make rays from different cameras
  miss each other. **Calibrate instead of measuring by hand** (section 2b).
- **The angle convention is not intuitive.** All angles at 0 means the camera looks *straight down*, and
  tilting toward the horizon is `roll` (horizontal = roll ≈ 90°). See `calibration/README.md`.
- **The grid should cover the space all cameras can see.**
- **`rtsp_url` can be a local camera:** `device:0`, or `device:1@1280x720` to pin the resolution; on macOS prefer
  `camera:NAME@1280x720`. List yours with `uv run p2v-calibrate list-cameras`; see WEBCAM_TEST.md.
- **`rtsp_url` can be a video file path.** It is replayed at its own frame rate, which is good for
  testing without cameras.
- **Keep real configs out of git.** `config/*.json` (except the example) is git-ignored because it
  contains passwords.
- **The old config formats still work:** a bare list of cameras, or a single camera object.

## 2b. Calibrate the cameras

Measure each camera's lens and exact placement with a printed board. The full walkthrough is in
**[calibration/README.md](calibration/README.md)**; in short:

```bash
calibration/demo.sh                  # optional: the whole flow on synthetic cameras
calibration/01_make_board.sh         # print work/board/board.pdf at 100%
calibration/02_capture_lens.sh       # all cameras in turn: hold the board still at many spots
calibration/03_solve_lens.sh
calibration/04_capture_placement.sh  # board flat on the floor, visible to all cameras
calibration/05_solve_placement.sh    # (or 05b_placement_from_landmarks.sh for big spaces)
calibration/06_apply.sh              # -> config/my_cameras.calibrated.json
calibration/07_check.sh              # board moved: do the rays meet? (mm and voxels)
```

A calibrated config gets an `intrinsics` block per camera. With it, frames are undistorted before motion
detection, and rays use the measured focal length and optical centre instead of `fov_degrees`. The stream
must have the resolution it was calibrated at; a different aspect ratio is refused.

## 3. Run live reconstruction

Terminal 1, the processor:

```bash
uv run p2v-rtsp config/my_cameras.json voxels
```

- **Output files:** `voxels/voxel_grid_<time>.bin` is saved every `save_interval_seconds`.
- **With `show_debug: true`:** a motion window opens for each camera. Press `q` in a window to quit
  and `s` to save now.
- **Without debug windows:** type `q` + Enter to quit, or `s` + Enter to save.
- **`--no-publish`:** turns off the live stream.
- **ZeroMQ port:** 5556 by default; change it with `ZMQ_PORT=5557 uv run p2v-rtsp ...`.

Terminal 2, the live 3D viewer:

```bash
uv run p2v-view                  # live stream from p2v-rtsp
uv run p2v-view --watch voxels   # or: reload the newest saved .bin
```

Viewer controls: drag to orbit, scroll to zoom, `+`/`-` to lower or raise the brightness threshold,
`q` to quit. Start with a higher threshold (`--threshold 1000`) if the view is cluttered.

### Small or far targets

By default every camera's rays are summed, so one camera's streak can outshine the real target. In one test
(three webcams, 1.5 cm balls) these settings found the small swing when the plain defaults didn't:

| Setting | Value | Why |
|---|---|---|
| `min_cameras` | 2 (or 3) | keeps only what several cameras see. 2 follows movement better, 3 is cleaner and steadier |
| `motion_threshold` | 40 | ignores webcam noise; lower it for good cameras or faint targets |
| `decay_rate` | 0.001 | old motion fades fast, so the result follows the target |
| `voxel_size` | 0.02 | about the ray miss *at the target's distance*; with smaller voxels the rays don't meet |

They were tuned on that one test, so treat them as a starting point. Depth is the least certain direction:
spacing the cameras further apart helps most. `noise_sigma` (a per-pixel noise threshold) is experimental.

## 4. Look at a saved grid

```bash
uv run p2v-view voxels/voxel_grid_1700000000.bin                 # interactive
uv run p2v-view voxels/voxel_grid_1700000000.bin --screenshot out.png
uv run p2v-cloud voxels/voxel_grid_1700000000.bin                # top-percentile point cloud,
                                                                 # saves screenshots/voxel_0001.png ...
```

`p2v-cloud` options: `--percentile 99.9`, `--hard-threshold N`, `--center X Y Z`,
`--rotate RX RY RZ` and `--no-window`.

## 5. Offline reconstruction from images (`p2v-batch`)

For rendered or recorded image sequences with a metadata file.

```bash
uv run p2v-batch metadata.json images/ voxel_grid.bin --size 500 --voxel-size 6 --center 0 0 500
```

`metadata.json` is a list with one entry per image:

```json
[{"camera_index": 0, "frame_index": 0, "image_file": "cam0_000.png",
  "camera_position": [0, -900, 500], "yaw": 0, "pitch": 90, "roll": 0, "fov_degrees": 60}]
```

Other options: `--threshold 2.0`, `--noise 1.0` (uniform pixel noise; `0` turns it
off) and `--seed N`.

Note that the batch tool's angle convention differs from the live tool's: pitch and roll are swapped.
Don't reuse one tool's angles in the other.

## 6. Drone detection to ATAK (`p2v-drone`)

```bash
uv run p2v-drone drone_detector.conf
```

The config uses `key = value` lines:

```ini
test_mode = true            # simulate a drone (the only mode that produces detections today)
cluster_tolerance = 2.0     # 1.0 (the default) can't follow the simulated drone
atak_server_ip = 127.0.0.1
atak_server_port = 8087
origin_lat = 38.8895
origin_lon = -77.0352
update_interval = 1         # seconds between CoT sends
```

Other keys: `grid_size_x`/`grid_size_y`/`grid_size_z`, `voxel_size`, `min_cluster_size` and
`max_cluster_size`. It sends Cursor-on-Target XML over TCP; stop it with Ctrl+C. It is not yet
connected to `p2v-rtsp`.

## 7. Astronomy (`p2v-space`)

```bash
uv sync --extra space
uv run p2v-space --fits-dir fits --grid-size 400 400 400
```

The FITS headers need `DATE-OBS`, `TIME-OBS`, `RA_TARG` and `DEC_TARG`. See `uv run p2v-space --help`
for the grid, sky-patch and step options.

## 8. Tests

```bash
uv run pytest                 # 52 tests, about 15 s
tools/calibration_e2e/run.sh  # run the calibration scripts end to end on synthetic cameras
```

## Troubleshooting

| Symptom | Check |
|---|---|
| Viewer shows nothing | Lower the threshold with `-`, or start with `--threshold 1`. Check that `p2v-rtsp` printed "added voxels" |
| "cannot bind ZeroMQ socket" | Another `p2v-rtsp` is running, or the port is in use; set `ZMQ_PORT` |
| A camera never connects | The URL or credentials are wrong. Try the URL in VLC. Opening it can take up to 5 s |
| Peaks in the wrong place / streaks instead of blobs | Camera positions or angles are off (calibrate, then run `calibration/07_check.sh`), or only one camera sees the object |
| "different aspect ratio" error at start | The stream's resolution doesn't match the calibration; calibrate on the stream you run (main vs sub) |
| Everything lights up | Raise `motion_threshold` (camera noise or compression artifacts) |
