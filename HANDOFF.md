# Relative Speedometer — Project Handoff

Paste this into a fresh chat to bring it up to speed. **MATH.md in the repo root has all
the derivations — read it first.** This file is deliberately everything *else*: hardware,
architecture, measured numbers, and the traps that cost real debugging time.

---

## 1. What this is

Monocular computer vision that measures the **relative speed (range rate)** of the
vehicle ahead, from a single ordinary camera. No radar, no stereo, no lidar.

Output per frame: time-to-contact in seconds, and — once a plate is resolvable — range
in metres and closing speed in m/s (negative = closing).

Two measurement channels, fused in one EKF over state `[lam, lamdot]`:

- **Channel A** — known-size ranging. A licence plate is legally a fixed width
  (EU 0.520 m), so `Z = f*W/w`. Gives absolute metres. Needs calibration.
- **Channel B** — scale ratio / looming. Register two views of the same patch and read
  the scale factor `s` out of the affine warp. Gives `lamdot = 1/TTC` **exactly**, with
  no calibration at all. The same warp's translation is the tracker (MATH.md 5.4).

Neither alone gives a speed: `Zdot = -lamdot * exp(-lam)` needs the rate from B and the
metres from A. That is the core design idea.

Language: Python 3. Dependencies: **numpy + opencv-python only.** No PyTorch, no
ultralytics. Vehicle detection is an ONNX model run through `cv2.dnn`. Keeping it that
way is a deliberate, repeatedly-stated constraint.

---

## 2. Environment

- Current machine: Windows 11, repo under
  `C:\Users\thaya\OneDrive\Documents\Jeevo\Relative Spedometer\Relative-Speedometer`.
  Python 3.12.5, **opencv-python 5.0.0**, numpy 2.5.3. Plenty of disk.
  (The earlier machine was Windows 10 with ~284 MB free, which is why a
  `pip install ultralytics` once failed; that constraint no longer applies, but the
  no-ultralytics rule stands.)
- Shell is PowerShell; a bash tool is also available.
- Camera (as measured on the earlier machine): **Microsoft LifeCam**, external USB,
  **device index 1** (index 0 was the built-in laptop camera). Runs at 1280x720.
  Not yet re-checked on this machine — run `python main.py cameras` first.
- `calibration.json`: fx = 981.0 px at 1280x720 (HFOV 66.2 deg), known-object method.
- 8 CPU threads; YOLOX-s at 416 input takes 130–200 ms per call on OpenCV 5.

---

## 3. File map

```
MATH.md                  All derivations. Includes 5.4 (tracking from the warp) and 7.5
                         (gate lock-out, plate width floor).
README.md                Usage and measured accuracy table.
main.py                  CLI entry. budget, synth, video, fetch-model, selftest,
                         plus cameras/calibrate/live/measure-noise from cli_live.

rspeed/
  geometry.py            Pinhole math. PLATE_WIDTHS_M: eu .520, us .305, jp .330, au .372
  filter.py              LogDepthEKF. State [lam, lamdot]. Exact closed-form propagation,
                         Joseph form, chi-square gating at NIS > 9, reopen_lam().
  scale.py               Channel B. ECC affine registration (phase-correlation seeded)
                         with log-polar fallback. Adaptive keyframing. measure() also
                         returns where the anchor moved (ScaleMeasurement.center).
  plate.py               Subpixel edge fitting. ClassicalPlateLocator, DnnPlateLocator
                         (ONNX), KnownObjectLocator.
  calib.py               KappaEstimator — the plate -> whole-car transfer.
  estimator.py           VehicleEstimator. WHERE EVERYTHING FUSES. Read this first.
                         Registration -> box propagation -> plate -> Channel A -> B.
  dnn.py                 ONNX detection via cv2.dnn. Decodes YOLOX / YOLOv5 / YOLOv8(11)
                         layouts, identified from output shape. MODEL_ZOO + fetch_model.
  detector.py            Detection/Track types, IouTracker, ScriptedDetector for tests.
  pipeline.py            RelativeSpeedPipeline. Detector striding, box propagation between
                         detections, max_tracks budget, off-frame retirement, lead
                         selection with hysteresis.
  config.py              All tunables in one dataclass.
  synth.py               Synthetic renderer with EXACT ground truth.
  camera.py              Capture, exposure diagnostics, calibration, Calibration.focal_for.
  live.py                LiveSession (manual target, ECC-tracked), overlay.
  cli_live.py            cameras / calibrate / live / measure-noise, plus shared
                         resolve_intrinsics / resolve_noise used by video too.
  noise.py               Measures sigma_w and sigma_s; NoiseProfile <-> noise.json.
  viz.py                 Overlay; boxes labelled det (detected) or trk (propagated).

models/                  gitignored. `python main.py fetch-model` puts yolox_s.onnx here.
tests/test_math.py       27 tests
tests/test_live.py       18 tests
tests/test_dnn.py        12 tests (decoders against hand-built tensors; one optional
                         real-model smoke test)
tools/verify_scale_convention.py   Proves the ECC scale AND translation conventions
```

---

## 4. Numbers

### Measured on the LifeCam (earlier machine) — do not re-derive

```
sigma_w    2.830 px   (0.8% of a 350 px feature)
sigma_s    0.00899    <- the config default of 0.004 is OPTIMISTIC by 2.2x
scale bias +0.00015   on a static scene
```

Those were measured before `measure-noise` could save its result, so **no noise.json
exists yet**. Re-run `python main.py measure-noise` once and live/video will use it.

The old live run (TrackerMIL era): 5.7 fps, TrackerMIL 109 ms/frame = 92% of cost,
measurement maths 9.3 ms/frame.

### Measured on this machine

```
TrackerMIL (for reference, now removed)    138 ms/frame
YOLOX-s ONNX, 416 input                    127-200 ms/call
video, real-image clip, stride 5           18.5 fps end to end
  of which: detector 183 ms/call on 20% of frames; everything else 16 ms/frame
synthetic pipeline, stride 5, no render     ~110-170 fps
```

Registration tracking accuracy: < 0.1 px horizontal on a moved + scaled synthetic
target; converges from 60 px jumps on a 140 px window; box centre within 3 px under
4 px/frame random lateral jitter while the target grows 60%.

Synthetic accuracy table (5 scenarios x strides 1/5 x 3 seeds) is in README.md.

Error budget at 1280 px / 66.2 deg / 30 fps (`python main.py budget --width 1280 --hfov 66.2`):

```
range   plate w_px   sigma_v @0.3px      @2.0px
   5 m      102.1      0.03 m/s         0.18 m/s
  10 m       51.1      0.11 m/s         0.70 m/s
  20 m       25.5      0.42 m/s         2.81 m/s
  30 m       17.0      0.95 m/s         6.32 m/s   <- below the 18 px plate floor
```

Usable range on this camera is roughly **5-20 m**; absolute range starts at ~28 m.

---

## 5. Traps — every one of these was hit for real

1. **`opencv-python-headless` silently breaks all GUI.** Must be `opencv-python`.
   Verify `cv2.getBuildInformation()` shows `GUI: WIN32UI`.

2. **The ECC scale is INVERTED.** The warp maps current-patch coords -> keyframe-patch
   coords, so a grown target gives `det(A) < 1`; true scale is `1/sqrt(det(A))`.
   The same direction governs tracking: anchor in current frame = `A^-1 (u_ref - b)`.
   Both proven in tools/verify_scale_convention.py.

3. **The Hann taper must stay OFF.** Not scale-equivariant: recovered 1.0868 for a true
   1.1111. Regression test guards it.

4. **Timestamp frames on arrival with the wall clock. Never use nominal fps.** A dt error
   is gain, not noise.

5. **Camera warm-up is mandatory**, and you must request `CAP_PROP_FPS = 30` (LifeCam ran
   at 7.5 fps otherwise).

6. **Auto-exposure takes ~5 s to ramp.** `settle_exposure` is time-based for this reason.

7. **Pick the noise ROI by temporal stability, not single-frame contrast.**

8. **Keep the ROI away from the frame border** (bias +0.15 vs +0.002).

9. **Do not use bash heredocs to write Python containing backslash escapes.** `\n` gets
   mangled. Hit again this session (caught by an assertion before writing). Use the
   Edit/Write tools for anything with escapes.

10. **OpenCV 5 removed the Darknet importer.** `cv2.dnn.readNetFromDarknet` does not exist;
    YOLOv4-tiny .cfg/.weights will not load. ONNX only (works on 4.x and 5.x).

11. **ScriptedDetector indexed boxes by call count.** With `detect_stride > 1` it replayed
    frame 10's box on frame 50, so every strided synthetic run looked broken (4 m error)
    and striding was never actually validated. Now indexed by the frame image. Guarded by
    `test_scripted_detector_indexes_by_frame_not_call_count`.

12. **Gate lock-out.** Wrong early plate reads seed lam, P collapses, and the chi-square
    gate then rejects every CORRECT reading forever (5.6 m permanent error) while NIS
    looks fine (rejections never enter the NIS log). Root cause: plates under ~18 px,
    where the classical locator latches and repeats the same wrong width. Fix:
    `plate_min_width_px = 18` + lock-out recovery (`reopen_after = 12`). A recovery run of
    8 fired on the latches themselves and tripled brake error — see MATH.md 7.5.

13. **A propagated box is not a measurement.** Its width is w_keyframe * s. Feeding it to
    Channel A or the kappa transfer double-counts Channel B. Only detector-produced
    boxes (`box_measured=True`) anchor Channel A.
    (The old live mode fed TrackerMIL's box to kappa — MIL never resizes, so that kappa
    was garbage whenever the object moved. Gone now.)

14. **A calibration is only valid at its resolution.** `live` used to default to 640x480
    while calibration.json is 1280x720, silently applying fx = 981 to the wrong mode.
    Now defaults to the calibration's resolution and `Calibration.focal_for` scales
    (same aspect) or refuses (different aspect).

15. **Lead selection must have hysteresis.** A single spurious 0.52-score detection took
    the lead within 9 frames and reset the readout. Now: penalise tracks seen once or
    no longer detected, and switch only on a clear margin.

16. **ClassicalPlateLocator crashed on boxes overhanging the top of the frame** (negative
    slice start). Real detectors and propagated boxes produce these at close range.

---

## 6. Current status

- **57/57 tests pass** (27 math + 18 live + 12 dnn). `python main.py selftest`.
- TrackerMIL deleted; tracking comes from Channel B's warp (MATH.md 5.4).
- Automatic vehicle detection works: `fetch-model` + `video --source <file|index>`.
  Verified end to end on a real vehicle photo zoomed to an exact looming profile:
  lead held throughout, TTC within 2% of truth.
- Detector striding validated for the first time (stride 5 ≈ stride 1 accuracy).
- Git: two prior commits; this session's work is uncommitted at time of writing.

**Known limitations, honestly stated:**

- **Never tested against real driving footage or a real moving car.** The real-image
  test used a still photo with synthetic zoom — correct looming geometry, no real motion
  blur, parallax, or plate.
- **Never run on the live camera on this machine.** The live and video camera paths are
  exercised by tests with synthetic frames only.
- No noise.json yet (see section 4).
- The classical plate locator produces false plates on non-plate texture (seen on the
  truck photo, which has no readable plate). A trained ONNX plate model via
  `--plate-model` is the fix; none is bundled.
- Camera frames that queue in the driver buffer while the detector runs get arrival
  timestamps slightly late. The offset is roughly constant, so dt stays right on
  average, but it is not measured.
- NIS runs low (0.2–0.5) in several synthetic scenarios: the filter is conservative
  there. Not tuned further.

---

## 7. Next steps, in priority order

1. **`python main.py cameras`** on this machine — confirm the LifeCam index and fps.
2. **`python main.py measure-noise --index <n>`** to write noise.json.
3. **Test on a real parked car** with `video --source <n> --show`: stationary should read
   ~0 km/h, rolling toward it negative, reversing positive. Watch the det/trk labels.
4. **Record a real dashcam clip** and run `video --source clip.mp4 --focal <fx>`.
5. If plates are the bottleneck: find or train an ONNX plate detector for
   `--plate-model`.

---

## 8. Working preferences

- Keep the numpy + opencv-only constraint. No ultralytics.
- Explanations should be short and lead with the answer; long build-ups bury the point.
- Durable conceptual explanations belong in MATH.md, not only in chat.
- Validate against synthetic ground truth before trusting anything on real footage.
