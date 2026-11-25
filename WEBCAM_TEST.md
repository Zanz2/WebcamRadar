# Desk test with three webcams

A small, cheap setup to try the whole pipeline: three cameras around a desk watching the same patch of
air, then something small moving through it. Run everything from the repository root.

## What you need
- Three webcams (a built-in one counts). **Fixed-focus models are best.** Autofocus changes the focal length, which breaks the
  lens calibration; on macOS, `camera:NAME` cameras are locked on every open; elsewhere lock the focus yourself.
- Something to hold each USB camera rigidly: clamps, tape or a small tripod. **A camera that moves after
  calibration needs re-calibrating.** With a laptop's camera, don't touch the lid afterwards.
- A printed board (step 3), on A3 or A4. Webcams only need to be 0.5–1 m away, so A3 is plenty.
- A test object: a ball or toy hanging on a string works well (it swings slowly and repeatably).

## 1. Install and allow camera access
```bash
uv sync
```
macOS only lets apps it has approved use the camera. The first run will ask for permission; if it doesn't,
go to **System Settings → Privacy & Security → Camera**, enable your terminal app (Terminal, iTerm or
VS Code), and restart that terminal.

## 2. Find your cameras
Plug them in, directly or through a powered hub, then:
```bash
uv run p2v-calibrate list-cameras --size 1280x720
cp config/webcams.example.json config/my_cameras.json
```
On macOS it prints an `rtsp_url` value for each camera (`camera:NAME@WxH`); elsewhere it prints `device:N`. It also
saves a labelled preview, `calibration/work/devices/all_cameras.png`. Put one value into each camera entry of
`config/my_cameras.json` (write `device:N` as `device:N@WxH`). Leave the positions and angles; calibration fills them in.

- **A camera that "delivered no frames"** is usually a phone (Continuity Camera), a virtual camera, or in use elsewhere.
- **Pin the size each camera actually delivers.** It must stay the same between calibration and running.
- **`device:N` numbers can change** when cameras are re-plugged; names don't.

## 3. Calibrate (about 20 minutes)
In `calibration/settings.sh`, check `CONFIG`, `CAMERAS` and `PAPER`. If the printed squares aren't the size printed on
the page, set `SQUARE_MM` to what you measured.

Then follow [calibration/README.md](calibration/README.md):

```bash
calibration/01_make_board.sh       # print board.pdf at 100%, tape it to something flat, measure it
calibration/02_capture_lens.sh     # each camera in turn; press Enter to start each (s = skip)
                                   # ~25 auto-saved views each; or one camera: 02_capture_lens.sh 1
calibration/03_solve_lens.sh       # expect an error under ~0.5 px
```
Now **fix the cameras in their final positions**, before doing placement:
- **Spread them around the desk, 60–90° apart,** all looking at the same patch about 0.5–1 m away.
  Cameras side by side looking the same way can't triangulate.
- Angle them so the board is fully visible to all three.

```bash
calibration/04_capture_placement.sh   # board flat on the desk, visible to all three; don't move it
calibration/05_solve_placement.sh     # check each position with a ruler, roughly
calibration/06_apply.sh               # -> config/my_cameras.calibrated.json
calibration/07_check.sh               # move the board (prop it on a book, turned); expect GOOD
```

**The board's printed origin corner is now the world origin**, with z up from the desk and units in
metres. The example grid covers x −0.15…0.45, y −0.2…0.4 and z 0…0.4 m around that corner: a 60×60×40 cm
box of 1 cm voxels, 144k voxels, so it's fast. If your board sits elsewhere, adjust `grid.center` in
`config/my_cameras.calibrated.json`.

## 4. Run it
Terminal 1:
```bash
uv run p2v-rtsp config/my_cameras.calibrated.json voxels
```
Three motion windows open, one per camera; moving pixels show red. In a window, press `q` to quit and `s`
to save.

Terminal 2:
```bash
uv run p2v-view --threshold 500
```
Swing the object through the space above the board. A blob should follow it in the viewer. Use `+`/`-`
to adjust the threshold until single-camera streaks disappear and the blob remains.

## Tuning
| Problem | Try |
|---|---|
| A window is all red while nothing moves | Auto-exposure or light flicker: raise `motion_threshold` (25 → 40), steadier light, avoid LED flicker |
| Long streaks, no blob | Only one camera sees the motion, or placement is off: re-run `07_check.sh` |
| Blob lags or smears | Lower `decay_rate` (0.3 → 0.1) so old motion fades faster |
| Blob in the wrong place | Grid not around the board: check `grid.center` against the origin corner |
| Slow / laggy | Lower the resolution (recalibrate at it) or use a smaller grid |
| `p2v-rtsp` says "different aspect ratio" | A camera delivered another resolution than it was calibrated at: pin its `@WxH` to what it delivers and recalibrate that camera |

**Expect imperfections.** The three cameras aren't synchronized, so a fast object is in a different
place in each camera's frame. Keep the motion slow for this test.
