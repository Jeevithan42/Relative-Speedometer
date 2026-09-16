"""CLI commands for real cameras: enumeration, calibration, and live measurement.

Kept out of main.py because these are the only commands that touch hardware, block on
GUI input, and depend on the machine rather than the maths.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2

from .camera import (
    Calibration,
    exposure_report,
    calibrate_checkerboard,
    focal_from_known_object,
    list_cameras,
    measure_object_width_interactive,
    open_camera,
    probe_modes,
)
from .config import Config
from .geometry import focal_length_px
from .live import KNOWN_OBJECTS_M, LiveSession, draw_live_overlay
from .noise import DEFAULT_NOISE_PROFILE, NoiseProfile
from .plate import KnownObjectLocator

DEFAULT_CALIB = "calibration.json"


# --- cameras -------------------------------------------------------------------------------


def cmd_cameras(args: argparse.Namespace) -> int:
    """Enumerate cameras and MEASURE what they actually deliver.

    The reported frame rate is routinely wrong or absent on USB cameras, and a higher
    resolution frequently collapses the frame rate because the driver falls back to
    uncompressed transfer. Both matter: resolution buys pixels on the target, frame
    rate buys measurement baseline.
    """
    cams = list_cameras(args.max_index)
    if not cams:
        print("no cameras found")
        return 1

    print(f"{'idx':>4}  {'mode':<12} {'measured':>10}  {'bright':>7}  image")
    print("-" * 62)
    for c in cams:
        mode = f"{c['width']}x{c['height']}"
        print(f"{c['index']:>4}  {mode:<12} {c['measured_fps']:>6.1f} fps  "
              f"{c['brightness']:>7.1f}  {c['verdict']}")
    if any(not c["verdict"].startswith("ok") for c in cams):
        print("\nA camera flagged above cannot be measured from: subpixel edge fitting")
        print("needs real gradients, so every quality gate will reject every frame.")

    if args.probe:
        for c in cams:
            print(f"\ncamera {c['index']} modes:")
            for m in probe_modes(c["index"]):
                rw, rh = m["requested"]
                aw, ah = m["actual"]
                note = "" if (rw, rh) == (aw, ah) else "  (fell back)"
                print(f"  {rw}x{rh:<6} -> {aw}x{ah}{note:<16} {m['measured_fps']:6.1f} fps")
        print("\nPrefer frame rate over resolution when forced to choose: halving the")
        print("frame rate costs more than doubling resolution buys (MATH.md 3.1).")
    return 0


# --- calibrate ------------------------------------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Recover fx. Absolute range is directly proportional to it (MATH.md 2.1)."""
    try:
        cap = open_camera(args.index, args.width, args.height)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        if args.method == "checkerboard":
            calib = calibrate_checkerboard(
                cap,
                board=(args.board_cols, args.board_rows),
                square_m=args.square_m,
                n_required=args.views,
            )
        else:
            width_m = _resolve_width(args)
            if width_m is None:
                return 2
            if not args.distance:
                print("quick calibration needs --distance (metres, measured with a tape)",
                      file=sys.stderr)
                return 2
            print(f"\nHold the object ({width_m * 100:.1f} cm wide) exactly "
                  f"{args.distance:g} m from the LENS,")
            print("square-on to the camera, near the centre of the frame.")
            w_px, frame = measure_object_width_interactive(cap)
            fx = focal_from_known_object(w_px, width_m, args.distance)
            h, w = frame.shape[:2]
            calib = Calibration(
                fx=fx, fy=fx, cx=w / 2.0, cy=h / 2.0, dist=[0.0] * 5,
                image_width=w, image_height=h, method="known-object", n_views=1,
            )
            print(f"\n  measured {w_px:.1f} px for {width_m * 100:.1f} cm "
                  f"at {args.distance:g} m")
    except KeyboardInterrupt as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    calib.save(args.out)
    print(f"\n{calib.summary()}")
    print(f"saved to {args.out}")
    if calib.method == "known-object":
        print("\nThis gives fx only, no distortion coefficients. A 1% error in your")
        print("distance measurement is a 1% error in every range this reports.")
    return 0


def default_resolution(args: argparse.Namespace) -> tuple[int, int]:
    """Explicit --width/--height, else the calibration's resolution, else 1280x720.

    Defaulting to the calibrated mode is what keeps fx valid: a calibration is only
    exact at the resolution it was measured at (Calibration.focal_for).
    """
    if args.width and args.height:
        return int(args.width), int(args.height)
    calib_path = getattr(args, "calib", None)
    if calib_path and Path(calib_path).exists() and not getattr(args, "focal", None):
        c = Calibration.load(calib_path)
        return int(args.width or c.image_width), int(args.height or c.image_height)
    return int(args.width or 1280), int(args.height or 720)


def resolve_intrinsics(
    args: argparse.Namespace, width: int, height: int
) -> tuple[float, Calibration | None]:
    """fx for the frames actually delivered: --focal, else a calibration valid at this
    resolution, else the --hfov guess (loudly). Returns (fx, calibration if exact).

    Resolved against the *delivered* frame size, not the requested one -- a camera that
    silently falls back to another mode must not keep the old mode's fx.
    """
    if getattr(args, "focal", None):
        print(f"intrinsics: f = {args.focal:.1f} px from --focal")
        return float(args.focal), None

    reason = "no calibration file and no --focal"
    calib_path = getattr(args, "calib", None)
    if calib_path and Path(calib_path).exists():
        calib = Calibration.load(calib_path)
        fx, how = calib.focal_for(width, height)
        if fx is not None:
            print(f"calibration: {calib.summary()}")
            print(f"intrinsics: f = {fx:.1f} px at {width}x{height} ({how})")
            return fx, (calib if how == "exact" else None)
        reason = f"{calib_path} not usable: {how}"

    f = focal_length_px(width, args.hfov)
    print(f"WARNING: {reason}.")
    print(f"         Falling back to --hfov {args.hfov:g} deg -> f = {f:.1f} px. That is a")
    print("         guess and every range inherits its error. TTC is unaffected. Fix with:")
    print("         python main.py calibrate --method quick --object card --distance 0.5")
    return f, None


def resolve_noise(args: argparse.Namespace, width: int, height: int) -> tuple[float, float]:
    """(sigma_w_px, sigma_s): explicit flags, else the measured profile, else defaults."""
    base = Config()
    sigma_w = base.sigma_w_plate_px
    sigma_s = base.sigma_s_base
    source = "config defaults (not measured on this camera)"

    path = getattr(args, "noise", None)
    if path and Path(path).exists():
        profile = NoiseProfile.load(path)
        if profile.matches(width, height):
            sigma_w, sigma_s = profile.sigma_w_px, profile.sigma_s
            source = f"{path}: {profile.summary()}"
        else:
            source = (f"config defaults ({path} was measured at "
                      f"{profile.image_width}x{profile.image_height}, frames are "
                      f"{width}x{height})")

    if getattr(args, "sigma_w", None) is not None:
        sigma_w = float(args.sigma_w)
        source += "; sigma_w from --sigma-w"
    if getattr(args, "sigma_s", None) is not None:
        sigma_s = float(args.sigma_s)
        source += "; sigma_s from --sigma-s"
    print(f"noise: sigma_w={sigma_w:.3f}px sigma_s={sigma_s:.5f}  <- {source}")
    return sigma_w, sigma_s


def add_camera_model_args(p: argparse.ArgumentParser) -> None:
    """Intrinsics and noise flags shared by every command that measures."""
    p.add_argument("--calib", default=DEFAULT_CALIB,
                   help="calibration file; used only if valid at the frame resolution")
    p.add_argument("--focal", type=float, help="fx in px, overrides --calib and --hfov")
    p.add_argument("--hfov", type=float, default=60.0, help="last-resort guess for fx")
    p.add_argument("--noise", default=DEFAULT_NOISE_PROFILE,
                   help="noise profile written by measure-noise")
    p.add_argument("--sigma-w", type=float, help="width noise px, overrides --noise")
    p.add_argument("--sigma-s", type=float, help="scale-ratio noise, overrides --noise")


def _resolve_width(args: argparse.Namespace) -> float | None:
    if getattr(args, "object", None):
        return KNOWN_OBJECTS_M[args.object]
    if getattr(args, "object_width", None):
        return float(args.object_width)
    print("need --object or --object-width", file=sys.stderr)
    print(f"known objects: {', '.join(sorted(KNOWN_OBJECTS_M))}", file=sys.stderr)
    return None


# --- live -----------------------------------------------------------------------------------


def cmd_live(args: argparse.Namespace) -> int:
    """Live camera, manually selected target of known width. No detector required."""
    width_m = _resolve_width(args)
    if width_m is None:
        if not args.ttc_only:
            print("\nWithout a known width you still get TTC -- Channel B needs no")
            print("calibration -- but not absolute range or speed. Add --ttc-only to")
            print("run in that mode.", file=sys.stderr)
            return 2
        width_m = 0.1  # placeholder; nothing absolute is trustworthy in this mode

    req_w, req_h = default_resolution(args)
    try:
        cap = open_camera(args.index, req_w, req_h)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    ok, frame = cap.read()
    if not ok:
        print("camera opened but delivered no frames", file=sys.stderr)
        cap.release()
        return 2
    h, w = frame.shape[:2]

    exposure = exposure_report(frame)
    if not exposure["usable"]:
        print(f"\nCAMERA IMAGE NOT USABLE: {exposure['verdict']}")
        print(f"  mean level {exposure['mean']:.1f}/255, gradient "
              f"{exposure['gradient']:.2f}")
        print("  Channel A measures intensity edges to subpixel accuracy, so it will")
        print("  find nothing here. Uncover the lens / add light, then retry.")
        if not args.force:
            cap.release()
            return 2
        print("  --force given, continuing anyway.\n")

    f_px, calib = resolve_intrinsics(args, w, h)
    sigma_w, sigma_s = resolve_noise(args, w, h)
    cfg = Config(
        image_width=w, image_height=h, hfov_deg=args.hfov,
        f_px=f_px,
        fps=args.fps,
        feature_width_m=width_m,
        feature_aspect=None,  # arbitrary object: no nominal shape to gate on
        q=args.q, detect_stride=1, plate_refresh_s=args.refresh_s,
        sigma_w_plate_px=sigma_w, sigma_w_bbox_px=sigma_w * 4.0, sigma_s_base=sigma_s,
        # The 0.15 default is tuned for a high-contrast licence plate. An arbitrary
        # hand-held object against a room background scores far lower and would be
        # rejected outright, so live mode gates much more permissively.
        plate_min_quality=args.min_quality,
    )
    maps = calib.undistort_maps() if (calib and any(calib.dist)) else None
    session = LiveSession(cfg, KnownObjectLocator(min_quality=args.min_quality),
                          undistort_maps=maps)

    print(f"\n{w}x{h}, f = {cfg.focal_px:.1f} px, target width {width_m * 100:.1f} cm")
    print("Drag a box around the object, then press ENTER or SPACE. ESC cancels.")
    print("Live view: S reselect, R reset filter, Q or ESC quit.\n")

    window = "relative speedometer -- live"
    frame = session.prepare(frame)

    if args.auto:
        # Pick a target without human input: the region whose measured width is most
        # stable over a short burst. Same criterion the noise tool uses, and for the
        # same reason -- contrast alone does not mean the fitter can track a width
        # there. Lets the whole pipeline be exercised unattended.
        from .noise import find_measurable_roi

        burst = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)]
        for _ in range(11):
            ok2, f2 = cap.read()
            if ok2:
                burst.append(cv2.cvtColor(session.prepare(f2), cv2.COLOR_BGR2GRAY))
        found = find_measurable_roi(burst)
        if found is None:
            print("auto mode found no measurable region -- point the camera at an "
                  "object with a clear vertical boundary", file=sys.stderr)
            cap.release()
            return 1
        box, quality = found
        print(f"auto-selected ROI {box} (edge quality {quality:.2f})")
        session.set_target(frame, box)
    elif not session.select_target(frame):
        print("no target selected", file=sys.stderr)
        cap.release()
        return 1
    print(f"tracker: {session.target.tracker_name}")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    writer = None
    if args.save:
        writer = cv2.VideoWriter(
            args.save, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
        )

    import time as _time

    t_start = _time.perf_counter()
    trace: list = []
    try:
        while True:
            if args.duration and (_time.perf_counter() - t_start) >= args.duration:
                break
            ok, raw = cap.read()
            if not ok:
                break
            frame = session.prepare(raw)
            est = session.step(frame)
            if est is not None:
                trace.append(est)
            vis = draw_live_overlay(frame, est, session.stats, cfg)
            if writer is not None:
                writer.write(vis)
            cv2.imshow(window, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                session.reset()
                if not session.select_target(frame):
                    break
            if key in (ord("r"), ord("R")) and session.target is not None:
                session.set_target(frame, session.target.bbox)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"wrote {args.save}")
        cv2.destroyAllWindows()

    st = session.stats
    print(f"\n{st.frames} frames at {st.measured_fps:.1f} fps measured, "
          f"{st.ms_per_frame:.1f} ms/frame processing, {st.lost_frames} frames target lost")
    if session.estimator is not None and not math.isnan(session.estimator.mean_nis):
        m = session.estimator.mean_nis
        verdict = ("consistent" if 0.5 < m < 2.0 else
                   "OVERCONFIDENT - raise --q" if m >= 2.0 else "too loose - lower --q")
        print(f"mean NIS {m:.2f} ({verdict})")

    if trace:
        _summarise_trace(trace)
    est_obj = session.estimator
    if est_obj is not None and sum(est_obj.reject.values()):
        r = est_obj.reject
        print("\n--- Channel A attempts ---")
        for k in ("ok", "not_found", "too_small", "low_quality", "bad_aspect", "gated"):
            if r.get(k):
                print(f"  {k:<12} {r[k]}")
        if r["ok"] == 0:
            print("\n  Channel A never anchored, so there is no absolute scale and")
            print("  range/speed stay unavailable. TTC is unaffected -- Channel B")
            print("  needs no calibration.")
            if r["low_quality"]:
                q = est_obj.last_quality
                print(f"  Cause: edge contrast below --min-quality "
                      f"(last measured {q:.3f} vs {args.min_quality:.3f}).")
                print("  Point at a higher-contrast target, or lower --min-quality.")
            elif r["not_found"]:
                print("  Cause: no opposing edge pair found in the target box at all.")
                print("  The target needs a clear vertical boundary against its")
                print("  background -- a card held up, not a flat wall.")
    return 0


def _summarise_trace(trace: list) -> None:
    """Report what each stage actually did over the run.

    A single speed number tells you nothing about whether the machinery underneath it
    worked. These counts are what distinguish a real measurement from a confident
    guess: which channel anchored, whether registration ever landed, whether the kappa
    transfer locked.
    """
    import numpy as np

    n = len(trace)
    ch_a = {}
    for e in trace:
        ch_a[e.channel_a] = ch_a.get(e.channel_a, 0) + 1
    n_b = sum(1 for e in trace if e.channel_b)
    n_cal = sum(1 for e in trace if e.calibrated)
    n_lock = sum(1 for e in trace if e.kappa_locked)

    print("\n--- pipeline behaviour over the run ---")
    print(f"  frames estimated  : {n}")
    print(f"  Channel A anchor  : " +
          ", ".join(f"{k}={v} ({v / n:.0%})" for k, v in sorted(ch_a.items())))
    print(f"  Channel B landed  : {n_b} ({n_b / n:.0%})")
    print(f"  calibrated frames : {n_cal} ({n_cal / n:.0%})")
    print(f"  kappa locked      : {n_lock} ({n_lock / n:.0%})")

    scales = [e.scale for e in trace if e.scale is not None]
    if scales:
        ss = np.array([m.s for m in scales])
        cc = np.array([m.confidence for m in scales])
        print(f"  scale ratio       : mean {ss.mean():.5f}  "
              f"spread {1.4826 * np.median(np.abs(ss - np.median(ss))):.5f}  "
              f"confidence {cc.mean():.3f}")

    widths = [e.plate.w_px for e in trace if e.plate is not None]
    if widths:
        wa = np.array(widths)
        print(f"  measured width    : mean {wa.mean():.2f} px  "
              f"sigma {1.4826 * np.median(np.abs(wa - np.median(wa))):.3f} px")

    cal = [e for e in trace if e.calibrated and e.Z is not None]
    if cal:
        z = np.array([e.Z for e in cal])
        v = np.array([e.Zdot for e in cal])
        print(f"  range             : {z.min():.2f} .. {z.max():.2f} m "
              f"(median {np.median(z):.2f})")
        print(f"  relative speed    : {v.min() * 3.6:+.1f} .. {v.max() * 3.6:+.1f} km/h "
              f"(median {np.median(v) * 3.6:+.1f})")
        ttc = [e.ttc for e in cal if math.isfinite(e.ttc)]
        if ttc:
            print(f"  TTC (finite)      : {min(ttc):.2f} .. {max(ttc):.2f} s "
                  f"({len(ttc)}/{len(cal)} frames closing)")


# --- parser registration ----------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    c = sub.add_parser("cameras", help="list cameras and measure their true frame rate")
    c.add_argument("--max-index", type=int, default=4)
    c.add_argument("--probe", action="store_true",
                   help="also try each resolution and measure the resulting fps")
    c.set_defaults(func=cmd_cameras)

    k = sub.add_parser("calibrate", help="recover the camera's focal length in pixels")
    k.add_argument("--method", default="quick", choices=["quick", "checkerboard"],
                   help="quick: one known object at a measured distance. "
                        "checkerboard: proper, also gives distortion")
    k.add_argument("--index", type=int, default=0)
    k.add_argument("--width", type=int, default=1280,
                   help="calibrate at the resolution you will measure at -- fx is per mode")
    k.add_argument("--height", type=int, default=720)
    k.add_argument("--out", default=DEFAULT_CALIB)
    k.add_argument("--object", choices=sorted(KNOWN_OBJECTS_M),
                   help="a known object (quick method)")
    k.add_argument("--object-width", type=float, help="its width in metres, if not listed")
    k.add_argument("--distance", type=float, help="lens-to-object distance in metres")
    k.add_argument("--board-cols", type=int, default=9, help="INNER corners across")
    k.add_argument("--board-rows", type=int, default=6, help="INNER corners down")
    k.add_argument("--square-m", type=float, default=0.025)
    k.add_argument("--views", type=int, default=12)
    k.set_defaults(func=cmd_calibrate)

    v = sub.add_parser("live", help="measure a known-size object with your camera")
    v.add_argument("--index", type=int, default=0)
    v.add_argument("--width", type=int,
                   help="default: the calibration's resolution, else 1280")
    v.add_argument("--height", type=int, help="default: the calibration's, else 720")
    v.add_argument("--fps", type=float, default=30.0, help="only used for --save")
    v.add_argument("--object", choices=sorted(KNOWN_OBJECTS_M))
    v.add_argument("--object-width", type=float, help="target width in metres")
    v.add_argument("--ttc-only", action="store_true",
                   help="run without a known width: TTC only, no absolute range")
    add_camera_model_args(v)
    v.add_argument("--q", type=float, default=0.05,
                   help="process noise; hand-held targets manoeuvre hard, so this is "
                        "looser than the on-road default")
    v.add_argument("--refresh-s", type=float, default=0.5)
    v.add_argument("--save", help="write an annotated mp4 here")
    v.add_argument("--force", action="store_true",
                   help="run even if the camera image is too dark to measure")
    v.add_argument("--auto", action="store_true",
                   help="pick the target automatically instead of asking you to drag "
                        "a box; lets the pipeline run unattended")
    v.add_argument("--duration", type=float,
                   help="stop after this many seconds")
    v.add_argument("--min-quality", type=float, default=0.02,
                   help="edge-contrast floor for accepting a width measurement")
    v.set_defaults(func=cmd_live)

    n = sub.add_parser("measure-noise",
                       help="measure this camera's real sigma_w and sigma_s")
    n.add_argument("--index", type=int, default=0)
    n.add_argument("--width", type=int, default=1280)
    n.add_argument("--height", type=int, default=720)
    n.add_argument("--frames", type=int, default=120)
    n.add_argument("--roi", help="x,y,w,h to measure; default picks the best region")
    n.add_argument("--focal", type=float, help="fx in px, for the implied-error table")
    n.add_argument("--hfov", type=float, default=60.0)
    n.add_argument("--force", action="store_true",
                   help="run even if the image looks unmeasurable")
    n.add_argument("--out", default=DEFAULT_NOISE_PROFILE,
                   help="write the measured profile here when the capture was valid; "
                        "live and video read it")
    n.set_defaults(func=cmd_measure_noise)


# --- measure-noise ------------------------------------------------------------------------------


def cmd_measure_noise(args: argparse.Namespace) -> int:
    """Measure this camera's real sigma_w and sigma_s -- MATH.md section 10.

    These two numbers set the accuracy of everything the project reports, and the
    config defaults are placeholders. This replaces them with measurements.
    """
    import time

    from .geometry import focal_length_px
    from .noise import measure_noise

    try:
        cap = open_camera(args.index, args.width, args.height)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    ok, first = cap.read()
    if not ok:
        print("camera delivered no frames", file=sys.stderr)
        cap.release()
        return 2

    rep0 = exposure_report(first)
    print(f"camera {args.index}: {first.shape[1]}x{first.shape[0]}, "
          f"mean level {rep0['mean']:.1f}, {rep0['verdict']}")
    if not rep0["usable"] and not args.force:
        print("image is not measurable; fix lighting first (or pass --force)",
              file=sys.stderr)
        cap.release()
        return 2

    print(f"\nHold the camera STILL and keep the scene static for "
          f"{args.frames} frames.")
    print("Point it at something with clear vertical edges -- a book spine, a box, a")
    print("card propped up. Motion during capture inflates the measured noise.\n")

    frames = []
    t0 = time.perf_counter()
    while len(frames) < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    elapsed = time.perf_counter() - t0
    cap.release()
    fps = len(frames) / elapsed if elapsed > 0 else 0.0

    roi = None
    if args.roi:
        try:
            roi = tuple(int(v) for v in args.roi.split(","))
            if len(roi) != 4:
                raise ValueError
        except ValueError:
            print("--roi must be x,y,w,h", file=sys.stderr)
            return 2

    report = measure_noise(frames, roi=roi, measured_fps=fps)
    if report is None:
        print("\nCould not find any region whose edges are measurable.")
        print("Point the camera at an object with a clear vertical boundary against")
        print("its background, then retry.")
        return 1

    print(f"captured {report.n_frames} frames at {fps:.1f} fps")
    print(f"scene motion: {report.motion_px:.2f} px/frame  "
          f"({'STATIONARY' if report.stationary else 'MOVING -- results inflated'})")
    print(f"ROI used: {report.roi}  (edge quality {report.mean_quality:.2f})")

    print("\n--- measured noise ---")
    if math.isnan(report.sigma_w_px):
        print(f"  width: only {report.n_width} usable frames, cannot estimate")
    else:
        print(f"  mean width      : {report.mean_w_px:8.2f} px "
              f"({report.n_width}/{report.n_frames} frames measured)")
        rel = report.sigma_w_px / max(report.mean_w_px, 1e-6)
        print(f"  sigma_w         : {report.sigma_w_px:8.3f} px   "
              f"-> config.sigma_w_plate_px")
        print(f"  relative        : {rel:8.1%}      "
              f"({'ok' if rel < 0.05 else 'TOO HIGH -- not a stable feature'})")
        print(f"  stable frames   : {report.stable_frac:8.1%}")
        if rel >= 0.05:
            print("\n  A sigma this large is not sensor noise. The edge fitter is")
            print("  locking onto different feature pairs between frames, so the width")
            print("  jumps rather than fluctuating. Point the camera at ONE object with")
            print("  a clean vertical boundary and plain background, or pass --roi.")
    if math.isnan(report.sigma_s):
        print(f"  scale: only {report.n_scale} usable registrations, cannot estimate")
    else:
        print(f"  sigma_s         : {report.sigma_s:8.5f}      -> config.sigma_s_base")
        print(f"  scale bias      : {report.scale_bias:+8.5f}      "
              f"(should be ~0 on a static scene)")

    if not math.isnan(report.sigma_w_px):
        f_px = args.focal or focal_length_px(first.shape[1], args.hfov)
        print(f"\n--- what that means, at f = {f_px:.0f} px, EU plate ---")
        print(f"{'range':>8} {'sigma_v':>12}")
        for Z in (10.0, 20.0, 30.0, 40.0):
            sv = report.implied_speed_noise(f_px, 0.520, Z)
            print(f"{Z:7.0f}m {sv:8.2f} m/s  ({sv * 3.6:5.1f} km/h)")
        print("\n(naive differentiation; the EKF does better. MATH.md section 3.)")

    if not report.stationary:
        print("\nThe scene was MOVING, so these sigmas include real motion and are")
        print("too large. Prop the camera up and rerun for a valid measurement.")
    if report.usable and args.out:
        report.to_profile(first.shape[1], first.shape[0]).save(args.out)
        print(f"\nsaved noise profile to {args.out} -- live and video will use it")
    elif args.out:
        print(f"\nnot saving {args.out}: the capture was not valid enough to trust")
    return 0 if report.usable else 1
