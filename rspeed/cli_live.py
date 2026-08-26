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
from .live import KNOWN_OBJECTS_M, LiveSession, draw_live_overlay
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

    calib = None
    if args.calib and Path(args.calib).exists():
        calib = Calibration.load(args.calib)
        print(f"calibration: {calib.summary()}")
    elif not args.focal:
        print(f"WARNING: no calibration file and no --focal, so falling back to "
              f"--hfov ({args.hfov:g} deg).")
        print("         That is a guess and every range inherits its error. Fix with:")
        print("         python main.py calibrate --method quick --object card "
              "--distance 0.5")

    try:
        cap = open_camera(args.index, args.width, args.height)
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

    cfg = Config(
        image_width=w, image_height=h, hfov_deg=args.hfov,
        f_px=(calib.fx if calib else args.focal),
        fps=args.fps,
        feature_width_m=width_m,
        feature_aspect=None,  # arbitrary object: no nominal shape to gate on
        q=args.q, detect_stride=1, plate_refresh_s=args.refresh_s,
        sigma_w_plate_px=args.sigma_w, sigma_w_bbox_px=args.sigma_w * 4.0,
    )
    maps = calib.undistort_maps() if (calib and any(calib.dist)) else None
    session = LiveSession(cfg, KnownObjectLocator(), undistort_maps=maps)

    print(f"\n{w}x{h}, f = {cfg.focal_px:.1f} px, target width {width_m * 100:.1f} cm")
    print("Drag a box around the object, then press ENTER or SPACE. ESC cancels.")
    print("Live view: S reselect, R reset filter, Q or ESC quit.\n")

    window = "relative speedometer -- live"
    frame = session.prepare(frame)
    if not session.select_target(frame):
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

    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            frame = session.prepare(raw)
            est = session.step(frame)
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
          f"{st.tracker_failures} tracker failures")
    if session.estimator is not None and not math.isnan(session.estimator.mean_nis):
        m = session.estimator.mean_nis
        verdict = ("consistent" if 0.5 < m < 2.0 else
                   "OVERCONFIDENT - raise --q" if m >= 2.0 else "too loose - lower --q")
        print(f"mean NIS {m:.2f} ({verdict})")
    return 0


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
    k.add_argument("--width", type=int, default=640)
    k.add_argument("--height", type=int, default=480)
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
    v.add_argument("--width", type=int, default=640)
    v.add_argument("--height", type=int, default=480)
    v.add_argument("--fps", type=float, default=30.0, help="only used for --save")
    v.add_argument("--object", choices=sorted(KNOWN_OBJECTS_M))
    v.add_argument("--object-width", type=float, help="target width in metres")
    v.add_argument("--ttc-only", action="store_true",
                   help="run without a known width: TTC only, no absolute range")
    v.add_argument("--calib", default=DEFAULT_CALIB)
    v.add_argument("--focal", type=float, help="fx in px, overrides --hfov")
    v.add_argument("--hfov", type=float, default=60.0)
    v.add_argument("--q", type=float, default=0.05,
                   help="process noise; hand-held targets manoeuvre hard, so this is "
                        "looser than the on-road default")
    v.add_argument("--sigma-w", type=float, default=0.5, help="width noise, px")
    v.add_argument("--refresh-s", type=float, default=0.5)
    v.add_argument("--save", help="write an annotated mp4 here")
    v.add_argument("--force", action="store_true",
                   help="run even if the camera image is too dark to measure")
    v.set_defaults(func=cmd_live)
