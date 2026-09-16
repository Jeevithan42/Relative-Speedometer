# Relative Speedometer

Monocular estimation of the **relative speed** of the vehicle ahead — how fast the gap
is opening or closing — from a single forward-facing camera.

Dependencies: **numpy and opencv-python, nothing else.** No torch, no ultralytics. Vehicle
detection runs an ONNX model through `cv2.dnn`.

The full derivation of everything below, with every symbol defined, is in
**[MATH.md](MATH.md)**.

## The idea in one paragraph

Relative speed is *range rate*: the derivative of a distance you never measure
directly. Two independent cues give it to you:

- **Channel A — known-size ranging.** A licence plate has a legally standardised width,
  so the pinhole relation `Z = f·W / w` turns its apparent pixel width into absolute
  metres. Needs calibration.
- **Channel B — scale ratio (looming).** Register the target against a reference frame
  and read the scale factor `s` directly. Then `TTC = Δτ / (s − 1)`, **exactly**, with
  no camera calibration and no knowledge of the target's real size.

Both become *linear* measurements once you filter in **log inverse depth**
`λ = −ln Z`, because `ln w = ln(f·W) + λ` and `λ̇ = −Ż/Z = 1/TTC`. So fusing them costs
one extra scalar Kalman update, not a second pipeline.

The plate then **calibrates the vehicle's rear bounding box**: while both are visible,
`κ_big = κ_plate + ln w_big − ln w_plate` recovers the box's unknown real width, so
absolute ranging keeps working long after the plate becomes unreadable — and the
expensive plate detector gets descheduled.

The registration that measures `s` also says **where the target moved**, so it doubles
as the tracker (MATH.md 5.4). The detector only has to run every few frames.

## Quick start

```bash
pip install -r requirements.txt

python main.py budget                      # error budget for your camera — read this first
python main.py selftest                    # validate every derivation against ground truth
python main.py synth --scenario closing    # run against synthetic ground truth
python main.py synth --scenario brake --detect-stride 5 --save out.mp4
```

## Automatic detection: video files and live cameras

```bash
python main.py fetch-model                                # YOLOX-s ONNX, 36 MB, checksum-verified
python main.py video --source clip.mp4 --focal 1663 --show
python main.py video --source 0 --show                    # live camera, wall-clock timestamped
```

`fetch-model` downloads the OpenCV Model Zoo's YOLOX-s (Apache-2.0) into `models/`. Any
YOLOX, YOLOv5, YOLOv8 or YOLOv11 ONNX export also works via `--model path.onnx`; the
output layout is identified from its shape. A trained ONNX plate detector can replace
the classical plate locator with `--plate-model`.

The detector runs every `--detect-stride` frames (default 5). Between detections each
box is moved by the registration, so a stale box never feeds the filter. Only a box the
detector actually produced is used as a width measurement. The overlay labels each box
`det` or `trk` so you can see which it is.

For a camera, `video` picks up `calibration.json` and `noise.json` automatically, but
only if they were made at the resolution the camera actually delivers. For a file you
must pass `--focal` (or `--calib`) yourself, because a webcam's calibration says nothing
about a dashcam clip.

## Testing with your own camera, no detector

You do **not** need a vehicle to test this. Point the camera at any object whose width
you can measure and move it toward and away from the lens — that exercises the entire
stack (subpixel widths, both channels, registration tracking, the EKF) against real
optics and real timing.

```bash
python main.py cameras --probe          # what does your camera actually deliver?
python main.py calibrate --method quick --object card --distance 0.5
python main.py measure-noise            # hold still; writes noise.json
python main.py live --object card
```

Then drag a box around the object and move it. The box follows the object and scales
with it. `S` reselects, `R` resets the filter, `Q` quits.

`cameras` measures the true frame rate rather than trusting the reported one, and flags
a lens that is covered or a room that is too dark — both produce frames with no
gradients, so every quality gate silently rejects everything with no visible reason.

**Calibration.** `--method quick` needs one object at one tape-measured distance and
gives you `fx`. A 1% error in that distance is a 1% error in every range reported.
`--method checkerboard` is the proper route and also recovers distortion coefficients.
Calibrate at the resolution you will measure at: `fx` is in pixels of one camera mode.
`live` and `video` default to the calibration's resolution. They scale `fx` for a mode
with the same aspect ratio, and refuse one with a different aspect ratio (the sensor
crop differs). Without a usable calibration they fall back to `--hfov`, which is a
guess: **TTC** stays correct (Channel B needs no calibration), but the metres will be
wrong.

**Noise.** `measure-noise` writes the measured `sigma_w` and `sigma_s` to `noise.json`
when the capture was valid (stationary, stable feature). `live` and `video` use it in
place of the config placeholders. On the LifeCam the placeholder `sigma_s` was 2.2x
optimistic.

Known objects: `card` (85.6 mm, ISO/IEC 7810 — the most reliable thing in your wallet),
`a4-portrait`, `a4-landscape`, `letter-portrait`, `cd`, `plate-eu`, `plate-us`. Anything
else: `--object-width 0.123`.

**Prefer frame rate over resolution.** Frame rate is measurement baseline; halving it
costs more than doubling resolution buys.

## What the output looks like

```
     t    trueZ     estZ    trueV     estV  trueTTC   estTTC  ch  kappa
  0.20    31.00       --    -5.00       --     6.20     6.08  NB  n0      <- TTC, no calibration
  1.60    24.00    23.13    -5.00    -4.90     4.80     4.72  PB  n0      <- plate resolvable: metres appear
  3.60    14.00    13.90    -5.00    -4.98     2.80     2.79  NB  lock    <- kappa locked, plate descheduled

  |Z err|   median  :   0.21 m   (steady state 0.11 m)
  |Zdot err| median :   0.06 m/s (steady state 0.04 m/s = 0.14 km/h)
```

(`synth --scenario closing --detect-stride 5`.) The `ch` column is the point: `N`/`P`/`B`
is which anchor Channel A used (none / plate / big feature), and the second letter is
whether Channel B landed. TTC is available from frame two, before any plate is readable.
Metres appear only once the plate is at least 18 px wide (here about 24 m). Below that
width the classical locator latches onto the wrong feature, and a wrong early reading
can lock the filter out (MATH.md 7.5).

Measured on the synthetic harness (exact ground truth, 960×540, 60° HFOV). Values are
medians over all calibrated frames, taken as the median across three seeds:

| scenario | stride 1: \|Z err\| / \|Ż err\| | stride 5: \|Z err\| / \|Ż err\| | calibrated frames |
|---|---|---|---|
| steady 5 m/s approach from 32 m | 0.49 m / 0.10 m/s | 0.21 m / 0.07 m/s | 94 / 140 |
| steady car-following, 22 m | 0.42 m / 0.01 m/s | 0.27 m / 0.00 m/s | 140 / 140 |
| lead pulling away from 14 m | 0.03 m / 0.10 m/s | 0.20 m / 0.09 m/s | 110 / 110 |
| hard brake, 6 m/s² | 0.14 m / 0.80 m/s | 0.12 m / 0.70 m/s | 54 / 129 |
| oscillating 22–28 m | 1.06 m / 0.76 m/s | 0.68 m / 0.72 m/s | 75 / 140 |

Stride 5 means the detector ran on 20% of frames. The brake numbers cover only the
closing, fast part of the run, since that is when the plate becomes resolvable. The
oscillating case sits right at the 18 px limit. On a real photo of a vehicle, zoomed to
an exact looming profile and run through `video`, TTC tracked truth within 2%
(4.41 s vs 4.50 s, 2.96 s vs 3.00 s).

Synthetic numbers are a floor, not a promise — see *Before trusting this on the road*.

## Layout

| File | What lives there |
|---|---|
| [MATH.md](MATH.md) | Every derivation, every symbol, the error budget, the failure modes |
| [rspeed/geometry.py](rspeed/geometry.py) | Pinhole relations and the log-inverse-depth substitution |
| [rspeed/filter.py](rspeed/filter.py) | The EKF — exact process model, both measurement channels, lock-out recovery |
| [rspeed/scale.py](rspeed/scale.py) | Channel B: ECC / Fourier-Mellin registration, keyframing, and tracking from the warp |
| [rspeed/plate.py](rspeed/plate.py) | Plate localisation (classical or ONNX) and subpixel edge fitting |
| [rspeed/calib.py](rspeed/calib.py) | The κ transfer, with a median that survives biased outliers |
| [rspeed/estimator.py](rspeed/estimator.py) | Per-vehicle fusion, box propagation, plate scheduling |
| [rspeed/dnn.py](rspeed/dnn.py) | ONNX detection through cv2.dnn: YOLOX / v5 / v8 decoding, model download |
| [rspeed/detector.py](rspeed/detector.py) | Detection protocol + IoU tracking |
| [rspeed/pipeline.py](rspeed/pipeline.py) | Orchestration, detector striding, track budget, lead selection |
| [rspeed/synth.py](rspeed/synth.py) | Synthetic renderer with exact ground truth |
| [rspeed/camera.py](rspeed/camera.py) | Live capture, intrinsic calibration, exposure diagnostics |
| [rspeed/noise.py](rspeed/noise.py) | Measures sigma_w and sigma_s on your camera; noise.json |
| [rspeed/live.py](rspeed/live.py) | Manual-target live session, wall-clock timed |
| [rspeed/cli_live.py](rspeed/cli_live.py) | The hardware-facing CLI commands |
| [tests/test_math.py](tests/test_math.py) | 27 tests, each naming the MATH.md equation it checks |
| [tests/test_live.py](tests/test_live.py) | 18 tests covering the camera path, tracking and calibration |
| [tests/test_dnn.py](tests/test_dnn.py) | 12 tests of detector decoding against hand-built tensors |
| [tools/verify_scale_convention.py](tools/verify_scale_convention.py) | Proves the ECC scale and translation sign conventions |

The measurement mathematics has no dependency on any model weights. That is deliberate:
the whole thing is testable against ground truth today, and the detector sits behind a
one-method protocol you can swap.

## Why it's built this way

**Subpixel accuracy is the whole ballgame.** `python main.py budget` prints the reason:
at 20 m, a 0.3 px width measurement gives 0.25 m/s of speed noise while a 2 px one gives
1.66 m/s. That ~7× gap is why [plate.py](rspeed/plate.py) fits gradient edges instead of
trusting a detector's box, and why Channel B measures the scale *ratio* by registration
rather than dividing two independent width estimates.

**Log inverse depth, not depth.** Filtering `Z` directly gives a nonlinear measurement
and wildly heteroscedastic noise. In `λ = −ln Z` both channels are linear, the noise is
homoscedastic, and the constant-relative-velocity propagation has an exact closed form
(`λ⁺ = λ − ln(1 − λ̇·Δt)`) that needs neither `Z` nor `v` individually.

**Fusion makes it cheaper, not more expensive.** Once κ locks, the large feature carries
Channel A and the plate detector drops to an occasional refresh. Registration tracks the
box, so the neural detector runs on one frame in five. Everything except the detector
costs about 16 ms per frame.

## Before trusting this on the road

1. **Calibrate the camera** at the resolution you will run at. Undistort too — an
   uncorrected plate at the frame edge reads wrong.
2. **Measure your noise** with `measure-noise`. The config values are placeholders.
3. **Validate against something independent.** A second vehicle with a GPS logger, or
   two cars on a straight road under cruise control.
4. **Check NIS.** It should sit near 1.0. Well above means the filter is overconfident
   and will lag a brake event; well below means it is ignoring its model.
5. **Read the failure-mode table** (MATH.md section 11) — yaw foreshortening, rolling
   shutter, unresolvable plates and track ID switches each bias the answer in a specific
   direction.

Relative speed already includes your own motion. For the lead vehicle's *ground* speed,
add your own from OBD-II or GPS: `v_lead = v_ego + Ż`.

## Status

Validated against synthetic ground truth, and end to end through the ONNX detector on
real vehicle imagery with a synthetic looming profile. **Not yet validated on real
driving footage or against a real moving car.** The classical plate locator needs about
18 px of plate, which is roughly 28 m on a 1280 px, 66° camera. Past that, range comes
from the κ transfer or waits; TTC is always available.
