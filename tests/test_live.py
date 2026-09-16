"""Validation of the live-camera path against synthetic ground truth.

The live path differs from the file path in ways that can silently corrupt every
reading -- wall-clock timing, a known-width object standing in for a plate, and an
externally driven box. These tests exercise that code with a synthetic sequence whose
true depth is known exactly, so the camera path is verified before any hardware is
involved.

Runs under pytest, or standalone:  python tests/test_live.py
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rspeed import synth  # noqa: E402
from rspeed.camera import Calibration, exposure_report, focal_from_known_object  # noqa: E402
from rspeed.config import Config  # noqa: E402
from rspeed.geometry import focal_length_px, width_from_depth  # noqa: E402
from rspeed.live import KNOWN_OBJECTS_M, LiveSession  # noqa: E402
from rspeed.plate import KnownObjectLocator, refine_plate_width  # noqa: E402

CAR_W = 1.80  # SceneSpec default car width, the "known object" for these tests


def _sequence(profile, n=110, width=960, height=540, fps=30.0):
    f = focal_length_px(width, 60.0)
    seq = synth.SyntheticSequence(
        f_px=f, image_width=width, image_height=height, fps=fps, seed=5
    )
    return f, seq, seq.run(profile, n)


def _session(f, width=960, height=540, q=0.02):
    cfg = Config(
        image_width=width, image_height=height, f_px=f, fps=30.0,
        feature_width_m=CAR_W, feature_aspect=None,
        q=q, detect_stride=1, sigma_w_plate_px=0.6, plate_refresh_s=0.3,
    )
    return cfg, LiveSession(cfg, KnownObjectLocator(min_quality=0.02))


# --- known-object calibration ---------------------------------------------------------


def test_focal_from_known_object_inverts_projection():
    """focal_from_known_object must invert width_from_depth exactly (MATH.md 1.1)."""
    f_true, W, Z = 1663.0, 0.0856, 0.5
    w_px = width_from_depth(Z, f_true, W)
    assert abs(focal_from_known_object(w_px, W, Z) - f_true) < 1e-9


def test_known_object_widths_are_sane():
    """Guard the lookup table against a units slip -- everything is metres."""
    assert abs(KNOWN_OBJECTS_M["card"] - 0.0856) < 1e-9  # ISO/IEC 7810 ID-1
    assert abs(KNOWN_OBJECTS_M["a4-landscape"] - 0.297) < 1e-9
    for name, w in KNOWN_OBJECTS_M.items():
        assert 0.01 < w < 1.0, f"{name} = {w} m is not a plausible hand-held width"


def test_calibration_roundtrips_through_json(tmp_path=None):
    out = Path(tmp_path or ".") / "_test_calib.json"
    c = Calibration(fx=831.4, fy=831.4, cx=320.0, cy=240.0, dist=[0.1, -0.2, 0, 0, 0],
                    image_width=640, image_height=480, method="known-object")
    try:
        c.save(out)
        back = Calibration.load(out)
        assert abs(back.fx - c.fx) < 1e-9
        assert abs(back.hfov_deg - c.hfov_deg) < 1e-9
        assert back.dist == c.dist
    finally:
        out.unlink(missing_ok=True)


def test_hfov_matches_focal_length():
    """Calibration.hfov_deg must be the inverse of focal_length_px."""
    f = focal_length_px(1920, 60.0)
    c = Calibration(fx=f, fy=f, cx=960, cy=540, dist=[0] * 5,
                    image_width=1920, image_height=1080)
    assert abs(c.hfov_deg - 60.0) < 1e-6


# --- exposure diagnostics ---------------------------------------------------------------


def test_exposure_report_flags_unusable_frames():
    """A covered lens must be reported as such, not silently produce no measurements."""
    dark = np.full((120, 160, 3), 3, np.uint8)
    assert not exposure_report(dark)["usable"]
    assert exposure_report(dark)["verdict"].startswith(("BLOCKED", "TOO DARK"))

    flat = np.full((120, 160, 3), 128, np.uint8)
    assert not exposure_report(flat)["usable"]

    rng = np.random.default_rng(0)
    textured = rng.integers(40, 210, (120, 160, 3), dtype=np.uint8)
    assert exposure_report(textured)["usable"], exposure_report(textured)


def test_exposure_report_accepts_a_real_frame():
    _f, _seq, frames = _sequence(synth.constant_gap(20.0), n=2)
    rep = exposure_report(frames[0].image)
    assert rep["usable"], rep


# --- known-object width measurement -------------------------------------------------------


def test_known_object_locator_measures_true_width():
    """The locator must recover the car's true apparent width to within a pixel."""
    f, _seq, frames = _sequence(synth.constant_gap(18.0), n=2)
    fr = frames[0]
    gray = cv2.cvtColor(fr.image, cv2.COLOR_BGR2GRAY)
    obs = KnownObjectLocator(min_quality=0.02).locate(gray, fr.car_bbox)
    assert obs is not None, "locator found no edges on a clean synthetic target"
    assert abs(obs.w_px - fr.w_car_px) < 2.0, (obs.w_px, fr.w_car_px)


def test_subpixel_refinement_beats_the_integer_box():
    """The whole error budget rests on this: refinement must be better than the box.

    The bbox is rounded to integers; the refined edge fit should land closer to truth
    across a set of depths.
    """
    f, seq, _ = _sequence(synth.constant_gap(20.0), n=1)
    box_err, refined_err = [], []
    for Z in (12.0, 16.0, 20.0, 26.0, 32.0):
        img, bbox, _wp, w_car_true = seq.render(Z, 0.0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        jittered = (bbox[0] - 2, bbox[1] - 1, bbox[2] + 3, bbox[3] + 2)
        out = refine_plate_width(gray, jittered, pad_frac=0.3)
        assert out is not None, f"no edges found at Z={Z}"
        box_err.append(abs(jittered[2] - w_car_true))
        refined_err.append(abs(out[0] - w_car_true))
    assert np.mean(refined_err) < np.mean(box_err), (refined_err, box_err)
    assert np.median(refined_err) < 1.5, refined_err


# --- live session end to end ------------------------------------------------------------------


def _run_live(profile, n=110, q=0.02):
    f, seq, frames = _sequence(profile, n=n)
    cfg, session = _session(f)
    session.set_target(frames[0].image, frames[0].car_bbox, use_tracker=False)
    out = []
    for fr in frames:
        # Externally driven box, as if from a tracker that never drifts. Timestamps are
        # passed explicitly here; a real camera uses arrival time instead.
        session.target.bbox = fr.car_bbox
        est = session.step(fr.image, t=fr.t)
        out.append((fr, est))
    return session, out


def test_live_session_recovers_range_and_speed():
    """Full live path against exact ground truth."""
    session, out = _run_live(synth.constant_closing(Z0=30.0, v=4.0))
    tail = [(fr, e) for fr, e in out[-40:] if e is not None and e.calibrated]
    assert len(tail) > 20, f"only {len(tail)} calibrated frames"
    z_err = np.array([abs(e.Z - fr.Z) for fr, e in tail])
    v_err = np.array([abs(e.Zdot - fr.Zdot) for fr, e in tail])
    assert np.median(z_err) < 1.0, np.median(z_err)
    assert np.median(v_err) < 0.8, np.median(v_err)


def test_live_session_reports_zero_speed_when_static():
    """A stationary object must not produce invented motion."""
    _session_obj, out = _run_live(synth.constant_gap(Z0=18.0))
    tail = [e for _fr, e in out[-40:] if e is not None and e.calibrated]
    assert len(tail) > 20
    v = np.array([e.Zdot for e in tail])
    assert abs(float(np.median(v))) < 0.5, float(np.median(v))


def test_live_session_sign_is_correct_when_receding():
    _session_obj, out = _run_live(synth.opening(Z0=12.0, v=3.0), n=90)
    tail = [e for _fr, e in out[-25:] if e is not None and e.calibrated]
    assert len(tail) > 10
    assert float(np.median([e.Zdot for e in tail])) > 1.0
    assert all(e.ttc == math.inf for e in tail)


def test_live_ttc_works_without_any_calibration():
    """Channel B is calibration-free, so TTC must appear before Channel A anchors."""
    _session_obj, out = _run_live(synth.constant_closing(Z0=26.0, v=5.0), n=60)
    early = [e for _fr, e in out[:20] if e is not None]
    assert any(e.channel_b for e in early), "Channel B never landed"


def test_wall_clock_timing_is_used_when_t_is_omitted():
    """A live camera must be timestamped on arrival.

    If step() fell back to a frame counter, elapsed time here would be ~0 and the
    filter would see dt = 0 forever.
    """
    f, _seq, frames = _sequence(synth.constant_gap(20.0), n=4)
    _cfg, session = _session(f)
    session.set_target(frames[0].image, frames[0].car_bbox, use_tracker=False)
    for fr in frames:
        session.step(fr.image)  # no explicit t
    assert session.estimator is not None
    assert session.estimator._t_last is not None
    assert session.estimator._t_last > 0.0, "wall clock never advanced"


def test_wrong_dt_scales_speed_proportionally():
    """Documents why wall-clock timing matters: dt errors are not noise, they are gain.

    Replaying the same sequence with timestamps stretched by 2x must roughly halve the
    reported speed. This is exactly the failure a nominal-fps camera would produce.
    """
    speeds = {}
    for stretch in (1.0, 2.0):
        f, _seq, frames = _sequence(synth.constant_closing(Z0=28.0, v=5.0), n=100)
        _cfg, session = _session(f)
        session.set_target(frames[0].image, frames[0].car_bbox, use_tracker=False)
        vals = []
        for fr in frames:
            session.target.bbox = fr.car_bbox
            est = session.step(fr.image, t=fr.t * stretch)
            if est is not None and est.calibrated:
                vals.append(est.Zdot)
        speeds[stretch] = float(np.median(vals[-30:]))
    ratio = speeds[1.0] / speeds[2.0]
    assert 1.6 < ratio < 2.5, (speeds, ratio)


def test_manual_target_follows_a_moving_object():
    """The live box is moved by Channel B's registration -- no OpenCV tracker at all.

    The object approaches (its box grows ~40%) while jittering sideways, and only the
    first frame's box is given. TrackerMIL could not do this: it never resized its box,
    so the plate bracket slid off any target whose distance was actually changing.
    """
    f = focal_length_px(960, 60.0)
    seq = synth.SyntheticSequence(f_px=f, image_width=960, image_height=540, seed=8,
                                  lateral_jitter_px=3.0)
    frames = seq.run(synth.constant_closing(Z0=16.0, v=4.0), 50)
    _cfg, session = _session(f)
    session.set_target(frames[0].image, frames[0].car_bbox, use_tracker=True)
    assert "ECC" in session.target.tracker_name

    errs = []
    for fr in frames:
        est = session.step(fr.image, t=fr.t)
        assert est is not None and not est.box_measured
        bx, _by, bw, _bh = session.target.bbox
        tx, _ty, tw, _th = fr.car_bbox
        errs.append((abs((bx + bw / 2) - (tx + tw / 2)), abs(bw - tw) / tw))
    assert frames[-1].car_bbox[2] > 1.35 * frames[0].car_bbox[2]
    assert max(e[0] for e in errs) < 3.0, max(e[0] for e in errs)
    assert max(e[1] for e in errs[5:]) < 0.05, max(e[1] for e in errs[5:])
    assert session.stats.lost_frames == 0


def test_tracked_live_session_still_measures_speed():
    """With the box tracked (not handed in), Channel A must still anchor off the
    refined edges and the full estimate must match ground truth."""
    f, _seq, frames = _sequence(synth.constant_closing(Z0=24.0, v=4.0), n=100)
    _cfg, session = _session(f)
    session.set_target(frames[0].image, frames[0].car_bbox, use_tracker=True)
    out = [(fr, session.step(fr.image, t=fr.t)) for fr in frames]
    tail = [(fr, e) for fr, e in out[-40:] if e is not None and e.calibrated]
    assert len(tail) > 20, len(tail)
    assert np.median([abs(e.Z - fr.Z) for fr, e in tail]) < 1.0
    assert np.median([abs(e.Zdot - fr.Zdot) for fr, e in tail]) < 0.8


def test_calibration_applies_only_at_a_compatible_resolution():
    """fx is in pixels of one camera mode. Using it at another mode is a silent gain
    error on every range, so it is scaled only when the aspect matches, else refused."""
    c = Calibration(fx=980.0, fy=980.0, cx=640.0, cy=360.0, dist=[0] * 5,
                    image_width=1280, image_height=720)
    fx, how = c.focal_for(1280, 720)
    assert fx == 980.0 and how == "exact"
    fx, how = c.focal_for(640, 360)
    assert abs(fx - 490.0) < 1e-9 and "scaled" in how
    fx, how = c.focal_for(640, 480)
    assert fx is None and "recalibrate" in how


def test_noise_profile_roundtrips_and_checks_resolution():
    from rspeed.noise import NoiseProfile

    out = Path(".") / "_test_noise.json"
    p = NoiseProfile(sigma_w_px=2.83, sigma_s=0.00899, mean_w_px=350.0,
                     image_width=1280, image_height=720)
    try:
        p.save(out)
        back = NoiseProfile.load(out)
        assert back == p
        assert back.matches(1280, 720) and not back.matches(640, 480)
    finally:
        out.unlink(missing_ok=True)


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
