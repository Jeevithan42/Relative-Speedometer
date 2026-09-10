# Relative Speedometer — Project Handoff

Paste this into a fresh chat to bring it up to speed. **MATH.md in the repo root has all
the derivations — read it first.** This file is deliberately everything *else*: hardware,
architecture, measured numbers, and the traps that cost real debugging time.

---

## 1. What this is

Monocular computer vision that measures the **relative speed (range rate)** of the
vehicle ahead, from a single ordinary camera. No radar, no stereo, no lidar.

Output per frame: time-to-contact in seconds, and — once calibrated — range in metres
and closing speed in m/s (negative = closing).

Two measurement channels, fused in one EKF over state `[lam, lamdot]`:

- **Channel A** — known-size ranging. A licence plate is legally a fixed width
  (EU 0.520 m), so `Z = f*W/w`. Gives absolute metres. Needs calibration.
- **Channel B** — scale ratio / looming. Register two views of the same patch and read
  the scale factor `s` out of the affine warp. Gives `lamdot = 1/TTC` **exactly**, with
  no calibration at all.

Neither alone gives a speed: `Zdot = -lamdot * exp(-lam)` needs the rate from B and the
metres from A. That is the core design idea.

Language: Python 3. Dependencies: **numpy + opencv-python only.** No PyTorch, no
ultralytics. Keeping it that way is a deliberate constraint.

---

## 2. Environment

- Windows 10 Pro. Repo at
  `C:\Program Files (x86)\Jeevo Projects\Relative speedometer\Relative-Speedometer`
- Shell is PowerShell; a bash tool is also available.
- Camera: **Microsoft LifeCam**, external USB, **device index 1** (index 0 is the
  built-in laptop camera). Runs at 1280x720.
- At 1280 px wide and 60 deg HFOV, `f = 1108.5 px`.
- **The C: drive is nearly full (~284 MB free).** A `pip install ultralytics` already
  failed once with "No space left on device". Assume no room for large packages.

---

## 3. File map

```
MATH.md                  All derivations. 12 sections, every symbol defined.
README.md                Usage.
main.py                  CLI entry. Subcommands: budget, synth, video, selftest,
                         plus cameras/calibrate/live/measure-noise from cli_live.

rspeed/
  geometry.py            Pinhole math. focal_length_px, kappa, depth_from_width,
                         Z<->lam conversions, speed_from_state, error-budget helper.
                         PLATE_WIDTHS_M: eu .520, us .305, jp .330, au .372
  filter.py              LogDepthEKF. State [lam, lamdot]. Exact closed-form
                         propagation, Joseph-form covariance, chi-square gating at
                         NIS > 9. update_absolute (H=[1,0]), update_scale (H=[0,1]).
  scale.py               Channel B. ScaleEstimator, ECC affine registration with a
                         Fourier-Mellin log-polar fallback. Adaptive keyframing.
  plate.py               Subpixel edge fitting (3-point parabolic peak).
                         refine_plate_width + ClassicalPlateLocator / YoloPlateLocator
                         / KnownObjectLocator.
  calib.py               KappaEstimator — the plate -> whole-car transfer.
  estimator.py           VehicleEstimator. WHERE THE TWO CHANNELS FUSE. Read this
                         first to understand the per-frame flow.
  detector.py            Detection/Track types, IouTracker, YoloVehicleDetector
                         (unused — needs ultralytics), ScriptedDetector for tests.
  pipeline.py            RelativeSpeedPipeline. Detector striding, cost accounting.
  config.py              All tunables in one dataclass.
  synth.py               Synthetic sequence renderer with EXACT ground truth. This is
                         how everything gets validated. Profiles: constant_closing,
                         constant_gap, opening, brake_event, oscillating.
  camera.py              Real-camera plumbing: enumeration, mode probing, exposure
                         settling and diagnosis, checkerboard + known-object calibration.
  live.py                LiveSession, manual target selection, overlay drawing.
  cli_live.py            cameras / calibrate / live / measure-noise commands.
  noise.py               Empirically measures sigma_w and sigma_s on YOUR camera.
  viz.py                 Overlay showing which channel produced each number.

tests/test_math.py       21 tests, all passing
tests/test_live.py       15 tests, all passing
tools/verify_scale_convention.py   Proves the ECC sign convention empirically
```

---

## 4. Numbers measured on the actual LifeCam

These were measured, not guessed. Do not re-derive them.

```
sigma_w    2.830 px   (0.8% of a 350 px feature)
sigma_s    0.00899    <- the config default of 0.004 is OPTIMISTIC by 2.2x
scale bias +0.00015   on a static scene (good; it was +0.15 before the ROI fix)
```

Live run, 25 s / 142 frames at 1280x720, static scene:

```
5.7 fps overall
mean NIS       0.91     <- filter is consistent; A and B agree
Channel B      137/142  (96%)
Channel A      115/142  (81%)  = big 84 (59%) + plate 31 (22%)
kappa locked   131/142  (92%)
reported speed  -0.0 km/h median   <- correct, nothing was moving
```

**Cost breakdown — this is the headline performance fact:**

```
TrackerMIL         109.2 ms/frame   92% of total cost
measurement math     9.3 ms/frame   (~107 fps on its own)
```

The actual speed-measuring code is fast. The tracker is the entire bottleneck.

Error budget at 1280 px / 60 deg / 30 fps (from `python main.py budget`):

```
range   plate w_px   sigma_v @0.3px      @2.0px
   5 m      115.3      0.02 m/s         0.16 m/s
  10 m       57.6      0.09 m/s         0.62 m/s
  20 m       28.8      0.37 m/s         2.49 m/s
  30 m       19.2      0.84 m/s         5.60 m/s
  60 m        9.6      3.36 m/s        22.39 m/s
```

Usable range on this camera is roughly **5-20 m**. Past 25 m it degrades fast.

---

## 5. Traps — every one of these was hit for real

1. **`opencv-python-headless` silently breaks all GUI.** `cv2.namedWindow` throws
   "The function is not implemented". It must be `opencv-python`. Verify with
   `cv2.getBuildInformation()` — it needs `GUI: WIN32UI`, not `GUI: NONE`.
   requirements.txt carries an explicit note about this.

2. **The ECC scale is INVERTED.** `cv2.findTransformECC(template=current, input=ref)`
   returns a warp in which a grown target means `det(A) < 1`. True scale is
   `1/sqrt(det(A))`, not `sqrt(det(A))`. The flag `_ECC_SCALE_IS_INVERSE = True` in
   scale.py records this; proof in tools/verify_scale_convention.py. Getting it
   backwards flips the sign of every speed reading.

3. **The Hann taper must stay OFF.** It looks like it helps — it cut static-scene bias
   from +0.0025 to -0.0001. But a fixed window is not scale-equivariant, so it drags
   the estimate toward s = 1: against a known 1.1111 scaling it recovered 1.0868, a 22%
   under-report that would under-report speed by about the same. There is a regression
   test (`test_taper_must_not_be_enabled_by_default`) guarding this.

4. **Timestamp frames on arrival with the wall clock. Never use nominal fps.** A dt
   error is not noise, it is *gain* — 2x wrong dt gives 2x wrong speed. There is a test
   documenting exactly this (`test_wrong_dt_scales_speed_proportionally`).

5. **Camera warm-up is mandatory.** The first frames come out of a buffer and give a
   bogus fps reading (measured 29.6, when the true rate was 7.5). Also: you must
   explicitly request `CAP_PROP_FPS = 30` — without it the LifeCam ran at 7.5 fps, with
   it 31.8 fps.

6. **Auto-exposure takes about 5 seconds to ramp.** Judging exposure before then reports
   TOO DARK on a perfectly fine scene. `settle_exposure` is time-based for this reason.

7. **Pick the noise-measurement ROI by temporal stability, not single-frame contrast.**
   Selecting on edge quality picked a window scoring 0.23 whose width then varied by
   70 px on a 197 px mean — the two strongest gradients belonged to different objects
   from frame to frame. Selecting on lowest relative spread gave sigma_w = 2.16 px.

8. **Keep the ROI away from the frame border.** An edge-adjacent window loses content as
   the scene drifts and the affine fit absorbs it as apparent scale: measured bias +0.15
   versus +0.002 for a central window.

9. **Do not use bash heredocs to write Python containing backslash escapes.** On this
   setup `\n` inside a heredoc gets mangled into a literal newline, producing
   SyntaxErrors. This broke cli_live.py four separate times. Use the Edit/Write tools
   for Python source.

---

## 6. Current status

- **36/36 tests pass** (21 math + 15 live).
- Validated end to end against exact synthetic ground truth: recovers range, speed and
  sign correctly, and reports zero speed on a static scene.
- Runs live on the real LifeCam with a GUI window and correct output.
- Git: one commit (`0f8f847`). Uncommitted: modifications to requirements.txt,
  cli_live.py, estimator.py, scale.py, tests/test_math.py; `rspeed/noise.py` is
  untracked and new.

**Known limitations, honestly stated:**

- **No `calibration.json` exists yet.** The `calibrate` command has never been run, so
  metric output currently rests on an assumed 60 deg HFOV. TTC is unaffected.
- **No automatic car detection.** You must drag a box around the target by hand. This is
  the single biggest blocker to it being a real speedometer.
- **Never tested against an actual car or a real licence plate.** Everything so far is
  synthetic sequences plus a desk scene.
- TrackerMIL will likely drift or lose lock on a target whose apparent size is genuinely
  changing — which is exactly the case it exists to handle.

---

## 7. Next steps, in priority order

1. **Run `calibrate`** — one minute, removes the HFOV guess.
2. **Delete TrackerMIL.** The ECC registration in scale.py *already computes translation*
   as a by-product of the affine warp, so the tracker is redundant. Expect roughly 18x
   speedup (5.7 -> 50+ fps) and one fewer dependency. Proposed, awaiting a go-ahead.
3. **ONNX vehicle detector via `cv2.dnn`** — about 12 MB, no PyTorch, fits the disk
   budget. This is what turns "thing I boxed by hand" into "the car in front of me".
4. **Test on a real parked car.** Box the plate at 5-10 m; sitting still should read
   0 km/h, rolling forward negative, reversing positive.

---

## 8. Working preferences

- Keep the numpy + opencv-only constraint unless there is a strong reason not to.
- Explanations should be short and lead with the answer; long build-ups bury the point.
- Durable conceptual explanations belong in MATH.md, not only in chat.
- Validate against synthetic ground truth before trusting anything on real footage.
