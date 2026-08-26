# Relative Speedometer

Monocular estimation of the **relative speed** of the vehicle ahead — how fast the gap
is opening or closing — from a single forward-facing camera.

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

## Quick start

```bash
pip install -r requirements.txt

python main.py budget                      # error budget for your camera — read this first
python main.py selftest                    # validate every derivation against ground truth
python main.py synth --scenario closing    # run against synthetic ground truth
python main.py synth --scenario brake --save out.mp4
```

## Testing with your own camera

You do **not** need a vehicle detector to test this. Point the camera at any object
whose width you can measure and move it toward and away from the lens — that exercises
the entire stack (subpixel widths, both channels, the κ transfer, the EKF) against real
optics and real timing.

```bash
python main.py cameras --probe          # what does your camera actually deliver?
python main.py calibrate --method quick --object card --distance 0.5
python main.py live --object card
```

Then drag a box around the object and move it. `S` reselects, `R` resets the filter,
`Q` quits.

`cameras` measures the true frame rate rather than trusting the reported one, and flags
a lens that is covered or a room that is too dark — both produce frames with no
gradients, so every quality gate silently rejects everything with no visible reason.

**Calibration.** `--method quick` needs one object at one tape-measured distance and
gives you `fx`. A 1% error in that distance is a 1% error in every range reported.
`--method checkerboard` is the proper route and also recovers distortion coefficients.
Without either, `live` falls back to `--hfov`, which is a guess — it will still give
correct **TTC** (Channel B needs no calibration at all) but its metres will be wrong.

Known objects: `card` (85.6 mm, ISO/IEC 7810 — the most reliable thing in your wallet),
`a4-portrait`, `a4-landscape`, `letter-portrait`, `cd`, `plate-eu`, `plate-us`. Anything
else: `--object-width 0.123`.

**Prefer frame rate over resolution.** On the test camera, 720p delivered 8.8 fps
against 640×480 at 30 fps. Frame rate is measurement baseline; halving it costs more
than doubling resolution buys.

## Real footage

Vehicle detection needs a model:

```bash
pip install ultralytics
python main.py video --source clip.mp4 --focal 1663 --show
python main.py video --source 0 --show     # live camera, wall-clock timestamped
```

## What the output looks like

```
     t    trueZ     estZ    trueV     estV  trueTTC   estTTC  ch  kappa
  0.20    31.00       --    -5.00       --     6.20     6.09  NB  n5      <- TTC, no calibration
  0.60    29.00    28.22    -5.00    -4.85     5.80     5.82  PB  n17     <- plate anchors absolute scale
  2.40    20.00    19.79    -5.00    -5.23     4.00     3.78  BB  lock    <- kappa locked, plate off

  |Z err|   median  :   0.20 m
  |Zdot err| median :   0.09 m/s (steady state 0.04 m/s = 0.14 km/h)
  mean NIS          :   0.70   (consistent)
```

The `ch` column is the point: `N`/`P`/`B` is which anchor Channel A used (none / plate /
big feature), and the second letter is whether Channel B landed. TTC is available from
frame two, before any plate has ever been seen.

Measured on the synthetic harness (exact ground truth, 960×540, 60° HFOV):

| scenario | median \|Z err\| | median \|Ż err\| | mean NIS | plate ran on |
|---|---|---|---|---|
| steady 5 m/s approach | 0.20 m | 0.04 m/s (0.14 km/h) | 0.70 | 47% of frames |
| hard brake, 6 m/s² | 0.32 m | 0.48 m/s (1.74 km/h) | 1.32 | 69% |
| steady car-following | 0.66 m | 0.00 m/s | 0.27 | 11% |

Synthetic numbers are a floor, not a promise — see *Before trusting this on the road*.

## Layout

| File | What lives there |
|---|---|
| [MATH.md](MATH.md) | Every derivation, every symbol, the error budget, the failure modes |
| [rspeed/geometry.py](rspeed/geometry.py) | Pinhole relations and the log-inverse-depth substitution |
| [rspeed/filter.py](rspeed/filter.py) | The EKF — exact process model, both measurement channels |
| [rspeed/scale.py](rspeed/scale.py) | Channel B: ECC / Fourier-Mellin registration, adaptive keyframing |
| [rspeed/plate.py](rspeed/plate.py) | Plate localisation and subpixel edge fitting |
| [rspeed/calib.py](rspeed/calib.py) | The κ transfer, with a median that survives biased outliers |
| [rspeed/estimator.py](rspeed/estimator.py) | Per-vehicle fusion and plate scheduling |
| [rspeed/detector.py](rspeed/detector.py) | Pluggable vehicle detection + IoU tracking |
| [rspeed/pipeline.py](rspeed/pipeline.py) | Orchestration, detector striding, lead selection |
| [rspeed/synth.py](rspeed/synth.py) | Synthetic renderer with exact ground truth |
| [rspeed/camera.py](rspeed/camera.py) | Live capture, intrinsic calibration, exposure diagnostics |
| [rspeed/live.py](rspeed/live.py) | Manual-target live session, wall-clock timed |
| [rspeed/cli_live.py](rspeed/cli_live.py) | The hardware-facing CLI commands |
| [tests/test_math.py](tests/test_math.py) | 20 tests, each naming the MATH.md equation it checks |
| [tests/test_live.py](tests/test_live.py) | 15 tests covering the camera path and calibration |

The measurement mathematics has no dependency on torch or any model weights. That is
deliberate: the whole thing is testable against ground truth today, and the detector is
behind a two-method protocol you can swap.

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
Channel A and the plate detector drops to an occasional refresh — 11% of frames in
steady following. Without the transfer you would need the plate every frame forever,
and it would still lose the target past ~35 m.

## Before trusting this on the road

1. **Calibrate the camera.** `cv2.calibrateCamera` on a checkerboard, then pass
   `--focal <fx>`. The `--hfov` fallback is a guess, and absolute range inherits its
   error directly. Undistort too — an uncorrected plate at the frame edge reads wrong.
2. **Measure your noise.** `sigma_w_plate_px`, `sigma_w_bbox_px` and `sigma_s_base` in
   [config.py](rspeed/config.py) are defaults, not truth. MATH.md section 10 gives the
   procedure: park behind a stationary car, record 200 frames, take the std. dev.
3. **Validate against something independent.** A second vehicle with a GPS logger, or
   two cars on a straight road under cruise control.
4. **Check NIS.** It should sit near 1.0. Well above means the filter is overconfident
   and will lag a brake event; well below means it is ignoring its model.
5. **Read the failure-mode table** (MATH.md section 11) — yaw foreshortening, rolling
   shutter, and track ID switches each bias the answer in a specific direction.

Relative speed already includes your own motion. For the lead vehicle's *ground* speed,
add your own from OBD-II or GPS: `v_lead = v_ego + Ż`.

## Status

Validated against synthetic ground truth only. The classical plate locator is adequate
to ~25 m in good light; past that, train a plate detector and use `YoloPlateLocator`.
Not validated on real footage yet.
