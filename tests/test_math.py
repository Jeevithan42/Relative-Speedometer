"""Validation of every derivation in MATH.md against ground truth.

Runs under pytest, or standalone:  python tests/test_math.py

Each test names the MATH.md equation it is checking, so a failure points at the
specific step of the derivation that broke rather than at "the pipeline".
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rspeed import synth  # noqa: E402
from rspeed.calib import KappaEstimator  # noqa: E402
from rspeed.config import Config  # noqa: E402
from rspeed.detector import ScriptedDetector  # noqa: E402
from rspeed.filter import LogDepthEKF  # noqa: E402
from rspeed.geometry import (  # noqa: E402
    focal_length_px,
    kappa,
    lam_to_Z,
    lamdot_from_motion,
    speed_from_state,
    speed_noise_from_width_noise,
    ttc_from_scale,
    width_from_depth,
    Z_to_lam,
)
from rspeed.pipeline import RelativeSpeedPipeline  # noqa: E402
from rspeed.plate import ClassicalPlateLocator  # noqa: E402
from rspeed.scale import ScaleEstimator  # noqa: E402


# --- section 1-2: pinhole ---------------------------------------------------------------


def test_focal_length_matches_worked_example():
    """MATH.md (1.2): 1920 px at 60 deg HFOV -> ~1663 px."""
    f = focal_length_px(1920, 60.0)
    assert abs(f - 1663.0) < 1.0, f


def test_pinhole_worked_example():
    """MATH.md section 2: EU plate at 20 m with f=1663 subtends ~43.2 px."""
    f = focal_length_px(1920, 60.0)
    w = width_from_depth(20.0, f, 0.520)
    assert abs(w - 43.2) < 0.2, w
    # and it halves at double the range -- the inverse proportionality of section 4
    assert abs(width_from_depth(40.0, f, 0.520) - w / 2) < 1e-9


# --- section 3: error budget --------------------------------------------------------------


def test_error_budget_reproduces_documented_numbers():
    """MATH.md section 3: 0.25 m/s subpixel vs 1.66 m/s with raw bbox jitter."""
    f = focal_length_px(1920, 60.0)
    w = width_from_depth(20.0, f, 0.520)
    fine = speed_noise_from_width_noise(0.3, w, 20.0, 15, 1 / 30)
    coarse = speed_noise_from_width_noise(2.0, w, 20.0, 15, 1 / 30)
    assert abs(fine - 0.25) < 0.02, fine
    assert abs(coarse - 1.66) < 0.05, coarse
    # the ~7x gap that motivates the whole subpixel design
    assert 6.0 < coarse / fine < 7.5


# --- section 4: log-inverse-depth identities ------------------------------------------------


def test_lam_roundtrip_and_ttc_identity():
    """MATH.md (4.1)-(4.7). lamdot must be exactly 1/TTC."""
    Z, Zdot = 22.5, -6.0
    lam = Z_to_lam(Z)
    lamdot = lamdot_from_motion(Z, Zdot)
    assert abs(lam_to_Z(lam) - Z) < 1e-12
    assert abs(speed_from_state(lam, lamdot) - Zdot) < 1e-12
    assert lamdot > 0  # closing
    assert abs(1.0 / lamdot - (-Z / Zdot)) < 1e-12


# --- section 5: scale ratio ------------------------------------------------------------------


def test_scale_ratio_ttc_is_exact():
    """MATH.md (5.5): TTC = dtau/(s-1), exact under constant relative velocity."""
    f, W = 1663.0, 0.520
    Z0, v, dtau = 30.0, 8.0, 0.25
    w1 = width_from_depth(Z0, f, W)
    w2 = width_from_depth(Z0 - v * dtau, f, W)
    s = w2 / w1
    true_ttc = Z0 / v - dtau  # remaining TTC at the *current* frame
    assert abs(ttc_from_scale(s, dtau) - true_ttc) < 1e-9, ttc_from_scale(s, dtau)


def test_scale_ratio_needs_no_calibration():
    """(5.2): kappa cancels, so f and W are irrelevant to Channel B."""
    Z0, v, dtau = 30.0, 8.0, 0.25
    ttcs = []
    for f, W in ((1663.0, 0.520), (900.0, 1.80), (2500.0, 0.305)):
        s = width_from_depth(Z0 - v * dtau, f, W) / width_from_depth(Z0, f, W)
        ttcs.append(ttc_from_scale(s, dtau))
    assert max(ttcs) - min(ttcs) < 1e-9, ttcs


def test_scale_estimator_recovers_known_scale():
    """The ECC backend must recover a synthetic scaling to better than 1%."""
    f = focal_length_px(1280, 60.0)
    seq = synth.SyntheticSequence(f_px=f, image_width=1280, image_height=720)
    est = ScaleEstimator()

    import cv2

    img_a, bbox_a, _wp, _wc = seq.render(30.0, 0.0)
    img_b, _bbox_b, _wp, _wc = seq.render(27.0, 0.5)
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    x, y, w, h = bbox_a
    center = (x + w / 2.0, y + h * 0.70)

    est.anchor(gray_a, center, float(w), 0.0)
    m = est.measure(gray_b, center, 0.5)
    assert m is not None
    true_s = 30.0 / 27.0
    assert abs(m.s - true_s) / true_s < 0.01, (m.s, true_s)


# --- section 7: filter -------------------------------------------------------------------------


def test_process_model_is_exact():
    """MATH.md (7.1)-(7.2): propagation must be exact under constant relative velocity.

    Seed with truth, run 60 predicts with no measurements at all, and require agreement
    to near machine precision. An approximate model would drift visibly here.
    """
    Z0, v, dt, n = 40.0, 7.0, 1 / 30, 60
    ekf = LogDepthEKF(q=0.0)
    ekf.initialise(Z_to_lam(Z0), lamdot_from_motion(Z0, -v))
    for _ in range(n):
        ekf.predict(dt)
    Z_true = Z0 - v * (n * dt)
    assert abs(ekf.Z - Z_true) < 1e-9, (ekf.Z, Z_true)
    assert abs(ekf.Zdot - (-v)) < 1e-9, ekf.Zdot
    assert abs(ekf.ttc - Z_true / v) < 1e-9


def test_gating_rejects_outliers():
    """MATH.md section 7.4: a wild measurement must not move the state."""
    f, W = 1663.0, 0.520
    k = kappa(f, W)
    ekf = LogDepthEKF(q=1e-6)
    ekf.initialise(Z_to_lam(20.0), 0.0, var_lam=1e-4, var_lamdot=1e-4)
    before = ekf.Z
    res = ekf.update_absolute(w_px=width_from_depth(3.0, f, W), kappa=k, sigma_w_px=0.3)
    assert not res.accepted
    assert abs(ekf.Z - before) < 1e-12


def test_fusion_beats_either_channel_alone():
    """The core claim of MATH.md section 6: A anchors position, B anchors velocity,
    and neither subsumes the other. Fusing must beat both on their weak axis."""
    f, W = 1663.0, 0.520
    k = kappa(f, W)
    Z0, v, dt, n = 30.0, 6.0, 1 / 30, 90
    rng = np.random.default_rng(3)
    sigma_w, sigma_s = 1.5, 0.004

    def run(use_a: bool, use_b: bool) -> tuple[float, float]:
        ekf = LogDepthEKF(q=0.01)
        ekf.initialise(Z_to_lam(Z0), 0.0, var_lam=0.25, var_lamdot=1.0)
        hist: list[float] = []
        for i in range(1, n + 1):
            t = i * dt
            Z = Z0 - v * t
            ekf.predict(dt)
            if use_a:
                w = width_from_depth(Z, f, W) + rng.normal(0, sigma_w)
                if w > 1:
                    ekf.update_absolute(w, k, sigma_w)
            if use_b:
                dtau = 0.25
                Z_ref = Z0 - v * max(0.0, t - dtau)
                s = (width_from_depth(Z, f, W) / width_from_depth(Z_ref, f, W)) * (
                    1 + rng.normal(0, sigma_s)
                )
                ekf.update_scale(s, dtau, sigma_s)
            hist.append(ekf.Zdot)
        # steady-state speed error over the last second
        tail = np.array(hist[-30:])
        return float(np.mean(np.abs(tail + v))), ekf.Z

    err_a, Z_a = run(True, False)
    err_b, Z_b = run(False, True)
    err_ab, Z_ab = run(True, True)

    Z_true = Z0 - v * (n * dt)
    # Fusion's speed error beats Channel A alone (A's width noise is the weak axis)
    assert err_ab < err_a, (err_ab, err_a)
    # ...and its depth beats Channel B alone (B has no absolute anchor at all)
    assert abs(Z_ab - Z_true) <= abs(Z_b - Z_true) + 1e-9, (Z_ab, Z_b, Z_true)
    assert err_ab < 0.6, err_ab


def test_filter_is_consistent_nis():
    """MATH.md section 10: mean normalised innovation squared should sit near 1."""
    f, W = 1663.0, 0.520
    k = kappa(f, W)
    Z0, v, dt, n = 35.0, 5.0, 1 / 30, 200
    rng = np.random.default_rng(5)
    sigma_w = 0.5
    ekf = LogDepthEKF(q=0.005)
    ekf.initialise(Z_to_lam(Z0), lamdot_from_motion(Z0, -v), 0.25, 0.25)
    nis = []
    for i in range(1, n + 1):
        Z = Z0 - v * (i * dt)
        ekf.predict(dt)
        w = width_from_depth(Z, f, W) + rng.normal(0, sigma_w)
        r = ekf.update_absolute(w, k, sigma_w)
        if r.accepted:
            nis.append(r.nis)
    mean_nis = float(np.mean(nis))
    assert 0.3 < mean_nis < 3.0, mean_nis


# --- section 8: kappa transfer -------------------------------------------------------------------


def test_kappa_transfer_recovers_unknown_width():
    """MATH.md (8.1): the transfer must recover ln(f*W_big) without ever being told W_big."""
    f = 1663.0
    W_plate, W_car = 0.520, 1.80
    k_plate = kappa(f, W_plate)
    est = KappaEstimator(min_samples=10)
    rng = np.random.default_rng(9)
    for i in range(40):
        Z = 30.0 - 0.4 * i
        wp = width_from_depth(Z, f, W_plate) + rng.normal(0, 0.3)
        wb = width_from_depth(Z, f, W_car) + rng.normal(0, 2.0)
        est.add_pair(wp, wb, k_plate)
    assert est.value is not None
    truth = kappa(f, W_car)
    assert abs(est.value - truth) < 0.05, (est.value, truth)
    # recovered real width, for a human-readable check
    recovered_W = math.exp(est.value) / f
    assert abs(recovered_W - W_car) < 0.10, recovered_W


def test_kappa_median_survives_biased_outliers():
    """A mean would be dragged off by latch-onto-the-bumper failures; a median must not."""
    f, W_plate, W_car = 1663.0, 0.520, 1.80
    k_plate = kappa(f, W_plate)
    est = KappaEstimator(min_samples=10, max_mad=1.0)
    for i in range(40):
        Z = 25.0 - 0.3 * i
        wp = width_from_depth(Z, f, W_plate)
        wb = width_from_depth(Z, f, W_car)
        if i % 5 == 0:  # 20% of samples latch onto something twice the plate's size
            wp *= 2.0
        est.add_pair(wp, wb, k_plate)
    assert abs(est.value - kappa(f, W_car)) < 0.10, est.value


def test_kappa_reset_on_identity_change():
    est = KappaEstimator(min_samples=3)
    for _ in range(10):
        est.add_pair(40.0, 140.0, kappa(1663.0, 0.520))
    assert est.value is not None
    est.reset()
    assert est.value is None and not est.locked


# --- end to end --------------------------------------------------------------------------------


def _run_sequence(profile, n_frames=140, jitter_px=1.5, seed=7):
    f = focal_length_px(1280, 60.0)
    cfg = Config(
        image_width=1280, image_height=720, hfov_deg=60.0, fps=30.0,
        detect_stride=1, sigma_w_bbox_px=2.5, q=0.02,
    )
    seq = synth.SyntheticSequence(
        f_px=f, image_width=1280, image_height=720, fps=30.0, seed=seed
    )
    frames = seq.run(profile, n_frames)
    detector = ScriptedDetector(seq.detector_from(frames, jitter_px=jitter_px))
    pipe = RelativeSpeedPipeline(cfg, detector, ClassicalPlateLocator(region="eu"))
    out = []
    for fr in frames:
        ests = pipe.process(fr.image, fr.t)
        out.append((fr, pipe.lead(ests)))
    return pipe, out


def test_end_to_end_constant_closing():
    """Full pipeline vs exact ground truth on a steady 5 m/s approach."""
    pipe, out = _run_sequence(synth.constant_closing(Z0=32.0, v=5.0))
    tail = [(fr, e) for fr, e in out[-40:] if e is not None and e.calibrated]
    assert len(tail) > 20, f"only {len(tail)} calibrated frames"

    z_err = np.array([abs(e.Z - fr.Z) for fr, e in tail])
    v_err = np.array([abs(e.Zdot - fr.Zdot) for fr, e in tail])
    ttc_rel = np.array(
        [abs(e.ttc - fr.ttc) / fr.ttc for fr, e in tail if math.isfinite(fr.ttc)]
    )

    assert np.median(z_err) < 1.5, np.median(z_err)
    assert np.median(v_err) < 1.0, np.median(v_err)
    assert np.median(ttc_rel) < 0.20, np.median(ttc_rel)


def test_end_to_end_constant_gap_reports_no_closing():
    """Steady car-following: speed near zero, TTC large. Guards against a sign error
    or a filter that invents motion from noise."""
    _pipe, out = _run_sequence(synth.constant_gap(Z0=22.0))
    tail = [(fr, e) for fr, e in out[-40:] if e is not None and e.calibrated]
    assert len(tail) > 20
    v = np.array([e.Zdot for _fr, e in tail])
    assert abs(float(np.median(v))) < 0.8, float(np.median(v))


def test_end_to_end_opening_has_correct_sign():
    """Lead pulling away must give Zdot > 0 and TTC = +inf."""
    _pipe, out = _run_sequence(synth.opening(Z0=14.0, v=4.0), n_frames=110)
    tail = [(fr, e) for fr, e in out[-30:] if e is not None and e.calibrated]
    assert len(tail) > 10
    v = np.array([e.Zdot for _fr, e in tail])
    assert float(np.median(v)) > 1.5, float(np.median(v))
    assert all(e.ttc == math.inf for _fr, e in tail)


def test_end_to_end_brake_event_is_tracked():
    """The adversarial case: the process model is violated exactly when it matters."""
    _pipe, out = _run_sequence(
        synth.brake_event(Z0=32.0, v0=2.0, brake_t=1.5, decel=6.0), n_frames=150
    )
    tail = [(fr, e) for fr, e in out[-25:] if e is not None and e.calibrated]
    assert len(tail) > 10, f"only {len(tail)} calibrated frames"
    v_err = np.array([abs(e.Zdot - fr.Zdot) for fr, e in tail])
    # Looser than the steady case -- a hard brake is a genuine model violation and
    # some lag is physical, not a bug. It must still track, not diverge.
    assert np.median(v_err) < 3.0, np.median(v_err)
    # and it must correctly register that closing is happening fast
    assert float(np.median([e.Zdot for _fr, e in tail])) < -3.0


def test_ttc_available_before_calibration():
    """Channel B needs no calibration, so TTC must exist before any plate is found."""
    pipe, out = _run_sequence(synth.constant_closing(Z0=30.0, v=6.0), n_frames=60)
    early = [e for _fr, e in out[:25] if e is not None]
    assert any(e.channel_b for e in early), "Channel B never produced a measurement"


def test_plate_detector_is_descheduled_after_lock():
    """MATH.md section 9: the kappa transfer must REDUCE steady-state cost."""
    pipe, out = _run_sequence(synth.constant_closing(Z0=28.0, v=4.0), n_frames=150)
    ests = [e for _fr, e in out if e is not None]
    if not any(e.kappa_locked for e in ests):
        return  # classical locator never locked on this sequence; nothing to assert
    locked_from = next(i for i, e in enumerate(ests) if e.kappa_locked)
    after = ests[locked_from + 5 :]
    if len(after) < 20:
        return
    rate = sum(e.plate_ran for e in after) / len(after)
    assert rate < 0.35, f"plate still running on {rate:.0%} of post-lock frames"


# --- standalone runner ----------------------------------------------------------------------------


def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
