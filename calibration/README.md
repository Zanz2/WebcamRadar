# Camera calibration walkthrough

The voxel method only works if rays from different cameras actually cross. That needs two things per
camera:

1. **Lens:** the focal length, optical centre and distortion. Wide lenses (87°, 130°) bend straight lines
   noticeably, and an uncorrected bend becomes a bent ray.
2. **Placement:** the camera's exact position and angles in one shared world frame. A 0.5° error moves a
   ray by about 9 cm at 10 m.

Everything here runs from the repository root. Generated files go to `calibration/work/`, which is
git-ignored because it contains photos of your space.

## 0. Try it without cameras (optional, about 10 s)

```bash
calibration/demo.sh
```

This renders three virtual cameras and runs every step below on them. It prints the recovered values
against the ground truth, so you can see what good output looks like. The output is in
`calibration/demo/`, laid out like a real run.

## 1. Settings and the board

Edit `calibration/settings.sh`:
- `CONFIG`: your camera config
- `CAMERAS`: which camera indices to calibrate
- `PAPER`: the paper size

```bash
calibration/01_make_board.sh
```

- **Print** `work/board/board.pdf` at **100% / actual size**: turn off "fit to page" and any scaling.
- **Mount it flat and rigid**, on foam board, acrylic or a clipboard. A curled board ruins the result.
- **Measure** the 100 mm ruler and one square. If the square isn't the size printed on the page, set
  `SQUARE_MM` in `settings.sh` to what you measured. This only affects placement distances, not the lens.

**Board size:**
- A3 is fine for the lens step, since you hold the board close.
- For the floor placement step the board must look reasonably big to *every* camera. If a camera is more
  than about 2 m away (wide lens), print A2/A1 or use landmarks (step 5b). The tools warn when the board
  is too small in a camera's view.

## 2. Lens: capture (each camera)

```bash
calibration/02_capture_lens.sh       # every camera in CAMERAS, in turn: press Enter to start each, s to skip
calibration/02_capture_lens.sh 2     # or just one camera
```

Each camera stops by itself after its target number of images (or press `q`), and the script moves to the
next one.

A window shows the camera with detected corners, a 3×3 coverage grid and how much of the frame the board
fills.

- **Hold the board still** for about a second at each spot. Frames save automatically when the board is
  still and the view is new. SPACE saves manually; `a` toggles auto-save; `q` quits.
- **Cover the whole image, especially the corners.** Wide-lens distortion is strongest there. Coverage
  cells turn green as they fill.
- **Vary the distance and the tilt: 20–45°, not steeper.** Steeper views are skipped automatically.
- **Keep the board sharp: this matters more than size.** Many webcams are fixed-focus and blur when the
  board is too close. If the corners look soft in the window, step back. Use good light and hold still.
- **Aim for the board spanning about 20–40% of the image width,** as long as it stays sharp.
- **Use the stream you'll actually run with.** The main and sub streams usually have different
  resolutions, and the calibration only applies to one of them.

Options: `--no-gui` (auto-capture with no window), `--manual` (SPACE only), `--target N`.

## 3. Lens: solve

```bash
calibration/03_solve_lens.sh
```

For each camera this fits three models: standard, standard+rational and fisheye. It keeps the simplest
one within 5% of the best fit. It prints the reprojection error and writes
`work/results/camN_intrinsics.json`, plus `camN_undistort_preview.png`, a before/after image where
straight edges should look straight on the right.

- **Good:** an error under about 0.5 px, and no "uncovered regions" warning.
- **Error over 1 px:** blurry images, a bent board, or the wrong `board.json`.
- **"Uncovered" warning:** capture more images with the board in those parts of the image, then re-run.

**Undistortion costs resolution (matters for tiny objects).** Frames are undistorted into an image of the
same size, so a wide lens has to shrink its centre to fit its edges in. `--balance` chooses the
trade-off. Measured on synthetic 1280×720 lenses like yours:

| `--balance` | standard (~108° measured) | fisheye (~127° measured) |
|---|---|---|
| 0 (crop edges) | 69% centre resolution, 108° kept | 88% centre resolution, 105° kept |
| 0.5 (default) | 63%, 113° | 72%, 116° |
| 1 (keep everything) | 56%, 119° | 57%, 127° |

If your objects are only 1–3 pixels across, prefer `--balance 0` and accept losing the outer edge of the
view. A planned pipeline change (casting rays straight from the raw pixels) would
remove this trade-off.

## 4. Placement: board on the floor

Lay the board **flat, face-up on the floor**, where every camera can see it. The printed **origin corner
becomes the world origin**:
- x runs along the board's bottom edge
- y runs up the page
- z points up from the floor
- units are metres

Then:

```bash
calibration/04_capture_placement.sh
```

This saves one averaged still per camera. **Don't move the board until it has finished.**

## 5. Placement: solve

```bash
calibration/05_solve_placement.sh
# or, to put the board corner elsewhere in your world:  ORIGIN="1.2 0.5 0" calibration/05_solve_placement.sh
```

This writes `work/results/camN_pose.json` and prints each camera's position and angles.

- **Sanity-check the positions with a tape measure.** They should be right to within a centimetre or so.
- **Fix any warnings:** too few corners, or the board too small in the image.

### 5b. Alternative: measured landmarks (large or outdoor spaces)

1. Copy `calibration/landmarks.example.csv` to `calibration/landmarks.csv`.
2. Fill in **6 or more** points you can see and measure: floor corners, door frames, posts. Use metres,
   in one frame you choose. Points at **different heights** give a better result.
3. Run it for each camera:

   ```bash
   calibration/05b_placement_from_landmarks.sh 0
   ```

   It grabs a still and asks you to click each landmark (`n` = not visible, `u` = undo), then solves the
   pose. At least 4 landmarks must be visible per camera, and 6 or more is better.

## 6. Apply

```bash
calibration/06_apply.sh
```

This writes `config/<name>.calibrated.json`, a copy of your config with the measured lens (`intrinsics`)
and placement (`camera_position`, `yaw`, `pitch`, `roll`) filled in. Your original config is untouched.
Run the reconstruction with the new file:

```bash
uv run p2v-rtsp config/my_cameras.calibrated.json voxels
```

You'll also need to set the `grid` (`center`, `size`, `voxel_size`) to cover the space the cameras watch,
in the same world frame (the board's origin corner).

## 7. Check (don't skip)

**Move the board somewhere new**: raise it on a box, turn it, so that at least two cameras see it. Then:

```bash
calibration/07_check.sh
```

The board is triangulated from all cameras using the pipeline's own ray model, and the check reports:
- **Ray miss:** how closely the cameras' rays to the same corner pass each other, in mm and in voxels of
  your grid. This is the number that decides whether reconstruction works.
- **Spacing error:** triangulated corner spacing against the real square size, which measures scale and
  distortion.

The verdict reads **GOOD** (rays within a voxel), **MARGINAL** (1–2 voxels) or **POOR**. Checking against
the *same* board placement used in step 4 would always look good, which is why the board must move.

## About the angles (read before hand-editing a config)

The pipeline uses this convention: R = Rz(yaw) · Ry(pitch) · Rx(roll), with the camera
looking along its −z axis.
- **All angles at 0 means the camera looks straight down**, with the top of its image toward +y.
- **Tilting toward the horizon is `roll`**: a horizontal camera has roll ≈ 90°.
- `yaw` turns the camera around the vertical axis.

So "pitch = −15" does *not* mean "tilted 15° down from horizontal". Calibration writes correct angles for
you; avoid typing them by hand.

## Files

| File | What |
|---|---|
| `settings.sh` | config path, cameras, paper size, measured square size |
| `01`–`07` scripts | the steps above (each wraps a `p2v-calibrate` subcommand) |
| `05b_placement_from_landmarks.sh` | placement from measured points instead of the floor board |
| `demo.sh` | the whole workflow on synthetic cameras |
| `landmarks.example.csv` | template for 5b |
| `work/` (generated) | `board/`, `captures/camN/`, `poses/`, `check/`, `landmarks/`, `results/` |

Every step is also available directly: `uv run p2v-calibrate --help`.
