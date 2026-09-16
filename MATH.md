# Relative Speedometer — The Mathematics

Everything this project does reduces to one problem: **estimate the range rate of the
vehicle ahead from a single camera.** This document derives that estimate from first
principles, defines every symbol used anywhere in the codebase, and states the
assumptions each step depends on.

---

## 0. Notation

Every symbol used in this document and in the source code.

### Geometry / camera

| Symbol | Code name | Units | Meaning |
|---|---|---|---|
| `Z` | `Z` | m | **Depth.** Distance from the camera's optical centre to the target, measured *along the optical axis* (straight ahead), not along the slant line of sight. |
| `Ż` | `Zdot` | m/s | **Range rate** — the relative speed we want. **Negative = closing** (gap shrinking), positive = opening (target pulling away). |
| `f` | `f_px` | px | **Focal length in pixels.** The conversion constant between "metres at depth Z" and "pixels on the sensor". *Not* millimetres. |
| `f_mm` | — | mm | Physical lens focal length. Related by `f = f_mm * (image_width_px / sensor_width_mm)`. Only needed if you calibrate from a spec sheet instead of a checkerboard. |
| `HFOV` | `hfov_deg` | deg | Horizontal field of view of the lens. |
| `W_img` | `image_width` | px | Image width in pixels. |
| `W` | `W_m` | m | **Real-world width** of the tracked feature (e.g. 0.520 m for an EU licence plate). |
| `w(t)` | `w_px` | px | **Apparent width** of that feature in the image at time `t`. The primary measurement. |
| `u, v` | — | px | Pixel coordinates (horizontal, vertical). |

### Derived state

| Symbol | Code name | Units | Meaning |
|---|---|---|---|
| `rho` | `rho` | 1/m | **Inverse depth**, `rho = 1/Z`. Proportional to apparent size, so it is the quantity the camera actually measures. |
| `lam` | `lam` | — | **Log inverse depth**, `lam = ln(rho) = -ln(Z)`. The state variable this project filters in. Dimensionless. |
| `lamdot` | `lamdot` | 1/s | **Fractional expansion rate**, `lamdot = -Zdot/Z`. **Positive = closing.** Exactly the reciprocal of time-to-contact. |
| `TTC` | `ttc` | s | **Time to contact**, `TTC = 1/lamdot = -Z/Zdot`. Time until `Z` reaches zero at the current closing rate. |
| `kappa` | `kappa` | — | **Log scale constant** of a feature, `kappa = ln(f * W)`. Bundles focal length and real-world width into the single number the filter needs. |
| `s` | `scale` | — | **Scale ratio** between two views of the same feature: `s = w(t2)/w(t1)`. `s > 1` means the target grew, i.e. it got closer. |
| `dtau` | `dtau` | s | **Baseline** — elapsed time between the keyframe and the current frame, over which `s` is measured. |
| `dt` | `dt` | s | Inter-frame interval, `1/fps`. |

### Filter

| Symbol | Code name | Meaning |
|---|---|---|
| `x` | `x` | State vector `[lam, lamdot]` transposed. |
| `P` | `P` | State covariance, 2x2. |
| `F` | `F` | Process Jacobian, 2x2. |
| `Q` | `Q` | Process noise covariance, 2x2. |
| `H` | `H` | Measurement Jacobian (1x2 here — both channels are linear). |
| `R` | `R` | Measurement noise variance (scalar per channel). |
| `K` | `K` | Kalman gain, 2x1. |
| `y` | `resid` | Innovation (measurement residual). |
| `S` | `S` | Innovation covariance. |

### Noise

| Symbol | Code name | Units | Meaning |
|---|---|---|---|
| `sigma_w` | `sigma_w_px` | px | Std. dev. of a single apparent-width measurement. **The dominant error source in the whole system.** |
| `sigma_lnw` | `sigma_lnw` | — | Std. dev. of `ln(w)`, equal to `sigma_w / w` (relative pixel noise). |
| `sigma_s` | `sigma_s` | — | Std. dev. of the scale ratio returned by image registration. |
| `sigma_v` | `sigma_v` | m/s | Resulting std. dev. of the relative-speed estimate. |
| `N` | `n_frames` | — | Number of frames in a slope-fitting window. |

---

## 1. The pinhole model and similar triangles

A pinhole camera projects a point at depth `Z`, offset `X` from the optical axis, to
pixel offset `u`:

```
u / f  =  X / Z
```

This is just similar triangles: the ray from the world point through the optical
centre to the sensor forms two triangles sharing the apex angle. The near triangle has
legs `(u, f)` in pixels; the far triangle has legs `(X, Z)` in metres. Equal angles
implies equal ratios.

Apply it to the two edges of a feature of real width `W` and subtract. `Z` is common to
both edges (assuming the feature is fronto-parallel — see §11), so:

```
        w(t)      W
       ------  =  ---                                               (1.1)
         f         Z
```

**Getting `f`.** Either from calibration (`cv2.calibrateCamera` on a checkerboard
returns `fx`, `fy` in the intrinsic matrix — use `fx` for horizontal widths), or from
the field of view:

```
        W_img / 2
  f  =  -----------                                                 (1.2)
        tan(HFOV/2)
```

Worked example: 1920 px wide, 60 deg HFOV -> `f = 960 / tan(30 deg) = 960 / 0.5774 = 1663 px`.

---

## 2. Channel A — known-size ranging

Rearranging (1.1) gives absolute depth whenever `W` is known:

```
        f * W
  Z  =  ------                                                      (2.1)
          w
```

Sanity check: EU plate `W = 0.520 m`, `f = 1663 px`, `Z = 20 m`
-> `w = 1663 * 0.520 / 20 = 43.2 px`. At 40 m it halves to 21.6 px. Apparent size is
*inversely* proportional to depth — remember this, §4 exploits it.

Licence plates are the ideal `W` because they are legally standardised:

| Region | Plate size (W x H) | `W` (m) |
|---|---|---|
| EU / UK / India (1-line) | 520 x 110 mm | 0.520 |
| USA / Canada | 12 x 6 in | 0.305 |
| Japan (standard) | 330 x 165 mm | 0.330 |
| Australia (standard) | 372 x 134 mm | 0.372 |

Differentiating (2.1) with respect to time gives the naive relative speed:

```
        d    ( f*W )      f*W   dw           Z    dw
  Zdot = -- (  ---  ) =  - --- * --   =   - --- * --                (2.2)
        dt   (  w  )      w^2   dt           w    dt
```

This is correct but a *bad estimator*, for the reason in §3.

---

## 3. Why naive differentiation fails

Differentiation amplifies high-frequency noise. Fit a straight line to `N` width
samples spaced `dt` apart, each with noise `sigma_w`; the ordinary-least-squares slope
has standard deviation:

```
                     sigma_w
  sigma_slope  =  -------------------------                         (3.1)
                  dt * sqrt( N(N^2-1) / 12 )
```

and by (2.2) the resulting speed error is:

```
                  Z
  sigma_v  =  ( ----- ) * sigma_slope                               (3.2)
                  w
```

**Worked error budget.** 1080p, 60 deg HFOV (`f = 1663`), EU plate at `Z = 20 m`
(`w = 43.2 px`), 30 fps, half-second window (`N = 15`, `dt = 1/30`):

- `sqrt(15 * 224 / 12) = sqrt(280) = 16.73`
- Subpixel measurement, `sigma_w = 0.3 px` -> `sigma_slope = 0.3 / (0.0333 * 16.73) = 0.538 px/s`
- `Z/w = 20 / 43.2 = 0.463 m/px`
- **`sigma_v = 0.463 * 0.538 = 0.25 m/s`, about 0.9 km/h.**

Now repeat with raw bounding-box jitter, `sigma_w = 2 px`: **`sigma_v = 1.66 m/s`, about 6 km/h.**

That ~7x gap is the entire engineering problem. Two consequences drive the design:

1. **Measure `w` to subpixel accuracy, or measure the *ratio* directly (§5).**
2. **Never take a raw finite difference.** Use a filter with an exact motion model so
   the estimate is smoothed optimally rather than by an arbitrary window.

There is also an inherent **latency/noise trade-off**: (3.1) improves as `N^1.5`, but a
long window lags reality — exactly wrong for collision warning. The filter in §7
resolves this by weighting measurements against a model instead of a fixed window.

---

## 4. The change of variable that makes everything linear

The problem with filtering `Z` directly is that the measurement equation (2.1) is
nonlinear (`w` proportional to `1/Z`) and the noise is wildly heteroscedastic: a fixed
`sigma_w` corresponds to centimetres of depth error up close and metres of depth error
far away.

Substitute **inverse depth** `rho = 1/Z`. Now (1.1) becomes `w = f*W*rho` — linear. But
constant relative speed gives `rhodot = -Zdot * rho^2`, a nonlinear process model.

Substitute **log inverse depth** instead:

```
  lam  =  ln(rho)  =  -ln(Z)                                        (4.1)
```

Take logs of (1.1):

```
  ln(w)  =  ln(f*W)  +  lam   =   kappa + lam                       (4.2)
```

where `kappa = ln(f*W)` is the **log scale constant** (§8). The measurement is now
linear in the state *and* the noise is homoscedastic: `sigma_lnw = sigma_w / w` is the
*relative* pixel error, roughly constant across a well-behaved detector.

Differentiate (4.1):

```
          d                1   dZ         Zdot
  lamdot = -- (-ln Z)  =  - - * --   =  - ----                      (4.3)
          dt               Z   dt          Z
```

So `lamdot` is the **fractional expansion rate**, and comparing with the definition of
time-to-contact:

```
                 Z            1
  TTC  =   -  ------   =   ------                                   (4.4)
                Zdot       lamdot
```

**`lamdot` *is* inverse TTC.** This identity is why the log domain is the right place
to work: the quantity a monocular camera can observe without any calibration (looming
rate) is precisely the state variable's derivative.

Recovering physical units at output time:

```
  Z     =  exp(-lam)                                                (4.5)
  Zdot  =  -Z * lamdot   =   -lamdot * exp(-lam)                    (4.6)
  TTC   =  1 / lamdot                                               (4.7)
```

Sign check: closing implies `Z` shrinking implies `Zdot < 0` implies `lamdot > 0`
implies `TTC > 0`. Consistent.

---

## 5. Channel B — scale ratio, looming, and calibration-free TTC

Instead of measuring `w` twice and subtracting, register the two image patches against
each other and recover the scale factor directly:

```
        w(t2)
  s  =  ------                                                      (5.1)
        w(t1)
```

Taking logs of (4.2) at both times, `kappa` cancels *identically*:

```
  ln(s)  =  lam(t2) - lam(t1)                                       (5.2)
```

**Neither `f` nor `W` appears.** An uncalibrated camera looking at an object of unknown
size still yields a valid change in `lam`. This is the looming cue, and it is why (5.2)
is the more robust of the two channels.

### 5.1 Exact TTC from the scale ratio

Assume constant relative velocity over the baseline `dtau = t2 - t1`, so
`Z(t2) = Z1 - v*dtau` with `v = -Zdot` the closing speed. Then by (2.1):

```
        w(t2)     Z1              Z1                    1
  s  =  ----- =  ----  =  ---------------  =  ------------------
        w(t1)     Z2       Z1 - v*dtau         1 - v*dtau/Z1
```

Using `lamdot_1 = v/Z1` from (4.3):

```
                 1
  s  =  ------------------      <=>      lamdot_1  =  (1 - 1/s) / dtau      (5.3)
         1 - lamdot_1*dtau
```

and evaluated at the *current* frame `t2` rather than at the keyframe:

```
  lamdot_2  =  (s - 1) / dtau                                       (5.4)

  TTC       =  dtau / (s - 1)                                       (5.5)
```

Both (5.3) and (5.4) are **exact** under constant relative velocity — no small-angle or
small-`dtau` approximation. This is the measurement the code uses.

Read (5.5) plainly: *if the target grew 4% over the last 0.2 s, contact is
0.2/0.04 = 5 s away.* No camera calibration, no knowledge of the car's size.

### 5.2 Why the ratio is measured, not computed

Do **not** compute `s` as `w2/w1` from two independent width detections — that inherits
both detections' noise. Instead register the pixels:

- **ECC** (`cv2.findTransformECC` with `MOTION_AFFINE`): iteratively maximises the
  enhanced correlation coefficient. Extract `s = sqrt(det(A))` where `A` is the 2x2
  linear part of the recovered affine warp. Illumination-invariant by construction.
  Accurate to roughly 0.1 px equivalent on a textured patch.
- **Fourier-Mellin** (`cv2.warpPolar` with `WARP_POLAR_LOG`, then `cv2.phaseCorrelate`):
  scale becomes a *translation* in log-polar space, so phase correlation recovers it in
  one shot. Faster, no iteration, slightly less accurate; a good fallback when ECC
  fails to converge.

### 5.3 Keyframe management

`s - 1` grows with `dtau`, so the SNR of (5.4) improves with a longer baseline. But a
long baseline means the patch has changed appearance (perspective, lighting, motion
blur) and registration degrades. Resolve with an **adaptive keyframe**:

- Hold a reference patch. Register the current frame against *it*, not against the
  previous frame — this avoids accumulating drift from frame-to-frame chaining.
- Re-anchor when any of: `|ln(s)| > ln(1.25)` (target changed size by 25%),
  `dtau > 0.5 s`, or registration confidence drops below threshold.

This gives a long baseline where the scene permits and a short one where it doesn't.

### 5.4 Tracking for free: the translation half of the warp

ECC does not return a scale. It returns a full affine warp, and `s` is only the
determinant of its linear part. The other two parameters say where the target moved —
so the registration that Channel B already pays for *is* a tracker.

Write patch coordinates `u` (canonical pixels, `n` of them across) for both the keyframe
patch and the current patch. `cv2.findTransformECC(template=current, input=keyframe)`
returns a warp mapping current-patch coordinates to keyframe-patch coordinates:

```
  u_ref  =  A * u_cur  +  b                                          (5.6)
```

Let `p_k` be the keyframe's anchor point (the rear-centre of the box when the keyframe
was dropped). Its patch coordinate is known, and inverting (5.6) says where that same
physical point sits in the current patch:

```
  u_cur  =  A^-1 * ( u_ref(p_k) - b )                                (5.7)
```

Each patch pixel covers `k = roi/n` frame pixels, with pixel centres at integers, so a
patch coordinate maps back to the frame as

```
  p  =  origin + (u + 0.5) * k  -  0.5                               (5.8)
```

with `origin` the crop's top-left corner. The box then follows: its size is the
keyframe box times `s`, and its offset from the anchor also grows by `s`.

Three properties make this better than a separate tracker, not just cheaper:

1. **It scales.** A general-purpose tracker such as TrackerMIL holds its box size fixed,
   so on a target whose distance is changing — the only case this project exists for —
   the plate bracket slides off the plate. Here the box size comes from the same `s`
   that feeds the filter.
2. **It does not drift within a keyframe.** Every frame registers against the keyframe,
   not the previous frame, so position error does not accumulate between re-anchors.
3. **It costs nothing extra.** The warp is already computed. Measured: TrackerMIL took
   110–140 ms per frame, over 90% of the live loop; the measurement maths alone runs in
   about 9 ms.

ECC is a local optimiser and only converges when the target has moved a fraction of the
patch. For larger jumps, phase correlation between the two patches seeds the translation
first (the warp starts at `b = -d` for a measured content shift `d`). Registration then
converges from 60 px jumps on a 140 px window. The seed is skipped for sub-pixel shifts,
so a static scene is registered exactly as it was when `sigma_s` was measured.

**Only a detected box is a measurement.** A propagated box's width is `w_keyframe * s`.
Feeding it to Channel A as if it were an independent width would count Channel B's
information twice and make the filter overconfident. So Channel A (and the kappa
transfer) uses the vehicle box only on frames where a detector produced it.

The sign convention of (5.6) is easy to get backwards, and a wrong sign moves the box
*away* from the target, which looks like ordinary tracking loss.
`tools/verify_scale_convention.py` checks it empirically alongside the scale convention.

---

## 6. Comparing the two channels

|  | Channel A (known size) | Channel B (scale ratio) |
|---|---|---|
| Equation | `ln(w) = kappa + lam` (4.2) | `lamdot = (s-1)/dtau` (5.4) |
| Observes | `lam` — absolute depth | `lamdot` — inverse TTC |
| Needs calibration | **Yes** (`f` and `W`) | **No** |
| Needs a known-size feature | **Yes** (plate) | **No** |
| Fails when | plate unreadable / too far | patch appearance changes fast |
| Noise | `sigma_lnw = sigma_w/w` | `sigma_s / (s*dtau)` |

They are **complementary, not redundant**: A anchors position, B anchors velocity.
Neither subsumes the other, and both are linear measurements on the state
`[lam, lamdot]`. That is what makes fusing them cheap — one shared filter, two
`update()` calls, no extra machinery. Channel B alone already yields TTC; you only need
A to convert TTC into metres per second.

---

## 7. The filter

### 7.1 State

```
  x  =  [ lam , lamdot ]
```

### 7.2 Process model — exact, in closed form

Under constant relative velocity, `Z(t+dt) = Z - v*dt`. Propagating (4.1):

```
  lam+     =  -ln(Z - v*dt)
           =  -ln(Z) - ln(1 - v*dt/Z)
           =  lam - ln(1 - lamdot*dt)                               (7.1)

  lamdot+  =  v/(Z - v*dt)  =  lamdot / (1 - lamdot*dt)             (7.2)
```

**This propagation is exact**, and remarkably it needs neither `Z` nor `v` individually
— only their ratio `lamdot`, which is the observable one. Note that `lamdot` is *not*
constant even at constant physical speed: as the gap closes, the same metres-per-second
produces faster fractional expansion. (7.2) captures that exactly, where a
constant-velocity filter on `lamdot` would not.

Guard: (7.1)-(7.2) require `lamdot*dt < 1`, i.e. `TTC > dt`. Always true short of
impact; the code clamps it anyway.

Jacobian `F`, writing `g = 1 - lamdot*dt`:

```
        [ 1     dt/g  ]
  F  =  [              ]                                            (7.3)
        [ 0     1/g^2 ]
```

Derivations: `d(lam+)/d(lamdot) = -d/d(lamdot)[ln(1 - lamdot*dt)] = dt/g`, and
`d(lamdot+)/d(lamdot) = 1/g + lamdot*dt/g^2 = (g + lamdot*dt)/g^2 = 1/g^2`.

Process noise `Q` models real acceleration of the lead vehicle (braking, throttle).
Using a continuous white-noise-acceleration model with spectral density `q`:

```
        [ dt^3/3   dt^2/2 ]
  Q  =  [                 ] * q                                     (7.4)
        [ dt^2/2   dt     ]
```

### 7.3 Measurement updates

Both channels are linear, so no measurement Jacobian approximation is needed.

**Channel A** — observe `ln(w)` of a feature with known `kappa`:

```
  z_A = ln(w) - kappa     H_A = [1, 0]     R_A = (sigma_w / w)^2    (7.5)
```

**Channel B** — observe `(s - 1)/dtau` from registration:

```
  z_B = (s-1)/dtau        H_B = [0, 1]     R_B = (sigma_s / dtau)^2 (7.6)
```

Standard EKF update for each, applied in sequence within a frame:

```
  y = z - H*x            S = H*P*H' + R           K = P*H' / S
  x <- x + K*y           P <- (I - K*H)*P
```

The code uses the Joseph form for `P` to preserve symmetry and positive-definiteness
under sequential scalar updates.

### 7.4 Gating

Reject a measurement when its normalised innovation squared exceeds a chi-square
threshold: `y^2/S > chi2(1, 0.997) = 9`. This is what protects the filter from a
mis-detection, a track ID switch, or a failed registration.

### 7.5 Gate lock-out, and why the plate needs a minimum width

Gating has a failure mode that looks like success. Suppose the first few Channel A
measurements are wrong in the same direction. They seed `lam`, `P[0,0]` collapses around
them within a handful of updates, and from then on every *correct* measurement sits
many sigma away from the overconfident filter and gets rejected. Nothing ever corrects
it. On the synthetic closing run at detector stride 3 this left a **permanent 25% range
error** (5.6 m median) while the NIS looked healthy, because the rejected measurements
never reach the NIS log.

The bad seeds all came from plates **7–14 px wide**. There the classical locator does
not return noisy widths; it latches onto a character block and returns the *same* wrong
width frame after frame (measured: 10.4 px five frames running, then 7.5 px seven frames
running, for a true 14 px). An error model of `sigma_w / w` cannot describe that, and
the filter cannot tell a steady wrong reading from a steady right one.

Two defences, in order of importance:

1. **Keep the measurement inside its valid regime.** Channel A rejects plates narrower
   than `plate_min_width_px` = 18 px. That is about 24 m at 960 px / 60°, and 28 m on the
   calibrated 1280 px LifeCam — past the 5–20 m usable range of §3 anyway. Across the
   whole synthetic grid (5 scenarios × strides 1/3/5 × 3 seeds) no lock-out occurred with
   the floor in place. Range is simply reported later; TTC is unaffected.
2. **Recover if it happens anyway.** If `reopen_after` = 12 consecutive Channel A
   measurements are all rejected, all share a sign, and their spread is under half their
   median, the filter is taken to be wrong: `lam` shifts by the median innovation, and
   `P[0,0]` re-opens to the size of that disagreement. `lamdot` is untouched, since
   Channel B keeps it honest independently.

The run length matters, because agreement is not proof: a latched locator also agrees
with itself. A run of 8 fired on those latches and tripled the range error during the
brake scenario. A run of 12 still fixes the lock-out (5.6 m → 0.28 m with the width floor
disabled) and is the tested safety net.

---

## 8. Plate-calibrates-larger-feature (the `kappa` transfer)

The plate gives excellent precision but subtends few pixels beyond about 35 m and is
often occluded, dirty, or angled. The vehicle's rear bounding box or taillight pair is
large and stable but has **unknown** real width. Bridge them.

While both are visible, apply (4.2) to each:

```
  ln(w_plate)  =  kappa_plate  +  lam        kappa_plate = ln(f * W_plate), known
  ln(w_big)    =  kappa_big    +  lam        kappa_big   = ln(f * W_big),   unknown
```

Subtract to eliminate `lam`:

```
  kappa_big  =  kappa_plate  +  ln(w_big)  -  ln(w_plate)           (8.1)
```

Note what (8.1) does **not** require: it never separates `f` from `W_big`. Only the
product matters, so an error in `f` and a compensating error in `W_big` are
indistinguishable and harmless — one fewer thing to get wrong. (Absolute `Z` does still
depend on `f` through `kappa_plate`.)

Accumulate `kappa_big` per track as a robust running estimate — the code keeps a median
over a bounded ring buffer, which shrugs off the occasional bad plate detection in a way
a running mean does not. Once its spread falls below a threshold the estimate is
**locked**, and:

1. Channel A switches to the large feature via `ln(w_big) = kappa_big + lam`, which
   keeps working long after the plate is unreadable;
2. The plate detector is **descheduled** for that track (with an occasional refresh
   frame), which is where the efficiency comes back — see §9.

`kappa_big` is valid only for that specific vehicle and only while its aspect toward the
camera is stable, so it is stored on the track and destroyed with it.

---

## 9. Cost model

Per-frame cost with the scheduling in §8, measured on an 8-thread laptop CPU
(OpenCV 5.0):

| Stage | When it runs | Cost |
|---|---|---|
| Vehicle detection (YOLOX-s ONNX, 416 input) | every `k`th frame (`k` = 5), tracked in between by §5.4 | 130–200 ms per call → 26–40 ms/frame amortised |
| Plate localisation | only while `kappa_big` unlocked, plus 1 refresh frame per ~2 s | a few ms |
| ECC registration + tracking | every frame, on a 96x96 canonical crop | a few ms |
| EKF | every frame, 2x2 | negligible |

Everything except the detector came to 16 ms/frame on a 640x480 real-image clip, for
18.5 fps end to end at `k` = 5. The tracker it replaced (TrackerMIL) cost 110–140 ms per
frame on its own.

Striding the detector is only safe because §5.4 carries the boxes between detections. A
box left stale for `k` frames drifts off a closing target, and at `k` = 3 the old
pipeline's range error was 4 m. With propagation, stride 5 matches stride 1 (see
README). Only up to `max_tracks` = 3 tracks, ranked by how plausibly each is the lead
vehicle, get the full estimator, so frame cost does not grow with the traffic.

The fusion therefore *reduces* steady-state cost rather than adding to it: without the
`kappa` transfer you would need the plate detector on every frame forever, and it would
still lose the target past 35 m. The scale-ratio channel is nearly free (a small crop
registration) and carries the velocity information, so the expensive stages can be
throttled hard once calibration locks.

---

## 10. Tuning `q`, `sigma_w`, `sigma_s`

- **`sigma_w`** — measure it empirically. Park behind a stationary car, record 200
  frames, take the std. dev. of the reported width. That number, not a guess, goes in
  the config.
- **`sigma_s`** — same procedure on the reported scale ratio; typically 0.002-0.01 for
  ECC on a textured plate.
- **`q`** — set from the strongest acceleration you expect the lead vehicle to apply.
  Hard braking is about 8 m/s^2. The corresponding fractional-rate acceleration is
  `a/Z`, so `q` is about `(a/Z)^2 * tau_c` with `tau_c` the correlation time of that
  manoeuvre (about 0.3 s). At `Z = 20 m`, `a = 8`: `q = 0.16^2 * 0.3 = 0.008`.

Validate by checking the **normalised innovation squared** over a run: it should average
about 1.0. Much above means `Q` or `R` too small (filter overconfident, will lag a brake
event). Much below means too large (filter is ignoring its model and just following
noise). `tests/test_math.py` reports this.

---

## 11. Assumptions and failure modes

| Assumption | Broken by | Consequence | Mitigation |
|---|---|---|---|
| Feature is fronto-parallel | Target on a curve, or in an adjacent lane | `w` foreshortens by `cos(theta)`, biasing `Z` **long** | Gate on plate aspect ratio; reject when it deviates from nominal |
| Constant relative velocity over `dtau` | Hard braking | (5.4) biased; filter innovation spikes | Short `dtau`, chi-square gating, `Q` sized for braking |
| Rectilinear lens | Uncorrected distortion | Width error grows toward frame edge | Undistort, or calibrate and correct per-position |
| Global shutter | Rolling shutter + lateral motion | Plate skews, `w` biased | Global-shutter camera, or accept the bias |
| Stable track identity | ID switch | `kappa_big` silently invalid | Destroy `kappa` on ID change; gating catches the rest |
| Stable lead choice | Spurious detection, adjacent car | Readout jumps to another vehicle's filter | Lead hysteresis; penalise tracks seen once or no longer detected |
| Plate resolvable | Plate under ~18 px | Locator latches, filter seeded wrong, gate lock-out (§7.5) | `plate_min_width_px` floor; lock-out recovery |
| Rigid target | Suspension pitch/dive | 1-2 Hz modulation of apparent size | Width-based ranging is largely immune (unlike ground-plane methods); `Q` absorbs the rest |

**Ego motion.** All of the above yields *relative* speed, which already includes your own
motion — that is what was asked for. To get the lead vehicle's ground speed, add your
own from OBD-II or GPS: `v_lead = v_ego + Zdot`.

---

## 12. Summary of the pipeline

```
  frame
    |
    +- ONNX vehicle detect (every kth frame) + IoU track --> bbox, track id
    |                                             |
    +- ECC register vs keyframe --> s, and where the target moved (5.4)
    |        |                                    |
    |        |                    between detections: move the box by the warp
    |        |                                    |
    +- plate localise in box (while unlocked) --> w_plate  (>= 18 px only, 7.5)
    |        |                                    |
    |        |                          kappa transfer (8.1), detected boxes only
    |        |                                    |
    +- w_big from a DETECTED box --> Channel A update (7.5)
    |        |                                    |
    |        +-------------------------> Channel B update (7.6)
    |                                             |
    |                                     EKF [lam, lamdot]
    |                                             |
    +- lead selection with hysteresis             |
                                                  |
                       Z = exp(-lam),  Zdot = -lamdot*exp(-lam),  TTC = 1/lamdot
```
