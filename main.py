"""Relative Speedometer -- monocular relative speed of the vehicle ahead.

Estimates range rate by fusing two measurement channels into one filter:

  Channel A  known-size ranging off the licence plate     -> absolute depth
  Channel B  scale ratio by image registration            -> inverse time-to-contact

The plate also calibrates the vehicle's rear bounding box, so absolute ranging
survives long after the plate is unreadable. All derivations are in MATH.md.

Usage
-----
  python main.py budget                        error budget for a camera config
  python main.py synth --scenario closing      run against synthetic ground truth
  python main.py selftest                      validate the math
  python main.py fetch-model                   download the ONNX vehicle detector
  python main.py video --source clip.mp4       run on real footage (or --source 0)
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from rspeed import cli_live, synth
from rspeed.camera import open_camera
from rspeed.config import Config
from rspeed.detector import ScriptedDetector
from rspeed.dnn import DEFAULT_MODEL, FORMATS, MODEL_ZOO, DnnVehicleDetector, fetch_model
from rspeed.geometry import (
    focal_length_px,
    speed_noise_from_width_noise,
    width_from_depth,
)
from rspeed.pipeline import RelativeSpeedPipeline
from rspeed.noise import DEFAULT_NOISE_PROFILE
from rspeed.plate import ClassicalPlateLocator, DnnPlateLocator
from rspeed.viz import draw_overlay

SCENARIOS = {
    "closing": lambda: synth.constant_closing(Z0=32.0, v=5.0),
    "gap": lambda: synth.constant_gap(Z0=22.0),
    "opening": lambda: synth.opening(Z0=14.0, v=4.0),
    "brake": lambda: synth.brake_event(Z0=32.0, v0=2.0, brake_t=1.5, decel=6.0),
    "oscillating": lambda: synth.oscillating(Z0=25.0, amp=3.0, period=4.0),
}


# --- budget ------------------------------------------------------------------------------


def cmd_budget(args: argparse.Namespace) -> int:
    """Print the MATH.md section 3 error budget for a given camera.

    Run this BEFORE buying a camera or writing pipeline code. If the numbers here are
    unacceptable at your working range, no amount of filtering will rescue it -- the
    fix is a longer lens, a higher-resolution sensor, or stereo.
    """
    cfg = Config(image_width=args.width, hfov_deg=args.hfov, fps=args.fps)
    f = cfg.focal_px
    W = cfg.plate_width_m
    n = max(2, int(round(args.window_s * cfg.fps)))

    print(f"camera   {cfg.image_width}px wide, {cfg.hfov_deg:g} deg HFOV -> f = {f:.1f} px")
    print(f"feature  {args.region.upper()} plate, W = {W:.3f} m")
    print(f"window   {args.window_s:g} s at {cfg.fps:g} fps = {n} frames")
    print(f"          (naive OLS differentiation; the EKF does better than this)\n")

    header = f"{'range':>7} {'w_px':>8} {'sigma_v @0.3px':>16} {'sigma_v @2.0px':>16}"
    print(header)
    print("-" * len(header))
    for Z in (5, 10, 15, 20, 30, 40, 60, 80):
        w = width_from_depth(float(Z), f, W)
        if w < 4.0:
            print(f"{Z:6d}m {w:8.1f} {'-- unresolvable --':>33}")
            continue
        fine = speed_noise_from_width_noise(0.3, w, float(Z), n, cfg.dt)
        coarse = speed_noise_from_width_noise(2.0, w, float(Z), n, cfg.dt)
        print(
            f"{Z:6d}m {w:8.1f} {fine:9.2f} m/s   {coarse:9.2f} m/s "
            f"  ({fine * 3.6:5.1f} / {coarse * 3.6:5.1f} km/h)"
        )
    print("\n0.3 px = subpixel edge fit; 2.0 px = raw detector box jitter.")
    print("That gap is why rspeed/plate.py refines edges instead of trusting the box.")
    return 0


# --- synthetic ----------------------------------------------------------------------------


def cmd_synth(args: argparse.Namespace) -> int:
    f = focal_length_px(args.width, args.hfov)
    cfg = Config(
        image_width=args.width,
        image_height=args.height,
        hfov_deg=args.hfov,
        fps=args.fps,
        detect_stride=args.detect_stride,
        q=args.q,
    )
    seq = synth.SyntheticSequence(
        f_px=f, image_width=args.width, image_height=args.height, fps=args.fps,
        noise_sigma=args.noise, seed=args.seed,
    )
    profile = SCENARIOS[args.scenario]()

    print(f"rendering {args.frames} frames of '{args.scenario}' ...", flush=True)
    t_render = time.perf_counter()
    frames = seq.run(profile, args.frames)
    print(f"  {len(frames)} frames in {time.perf_counter() - t_render:.1f}s")

    detector = ScriptedDetector(seq.detector_from(frames, jitter_px=args.bbox_jitter))
    pipe = RelativeSpeedPipeline(cfg, detector, ClassicalPlateLocator(region=args.region))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (args.width, args.height))

    rows = []
    t_proc = time.perf_counter()
    for fr in frames:
        ests = pipe.process(fr.image, fr.t)
        lead = pipe.lead(ests)
        rows.append((fr, lead))
        if writer is not None:
            writer.write(
                draw_overlay(fr.image, ests, lead, truth={"Z": fr.Z, "Zdot": fr.Zdot})
            )
    proc_s = time.perf_counter() - t_proc
    if writer is not None:
        writer.release()
        print(f"  wrote {args.save}")

    _report(rows, pipe, proc_s)
    return 0


def _report(rows, pipe: RelativeSpeedPipeline, proc_s: float) -> None:
    print(f"\n{'t':>6} {'trueZ':>8} {'estZ':>8} {'trueV':>8} {'estV':>8} "
          f"{'trueTTC':>8} {'estTTC':>8}  ch  kappa")
    print("-" * 78)
    step = max(1, len(rows) // 22)
    for fr, e in rows[::step]:
        if e is None:
            print(f"{fr.t:6.2f} {fr.Z:8.2f} {'--':>8}")
            continue
        estZ = f"{e.Z:8.2f}" if e.Z is not None else f"{'--':>8}"
        estV = f"{e.Zdot:8.2f}" if e.Zdot is not None else f"{'--':>8}"
        tt = "inf" if not math.isfinite(fr.ttc) else f"{fr.ttc:.2f}"
        et = "inf" if not math.isfinite(e.ttc) else f"{e.ttc:.2f}"
        ch = f"{e.channel_a[0].upper()}{'B' if e.channel_b else '-'}"
        kp = "lock" if e.kappa_locked else f"n{e.kappa_samples}"
        print(f"{fr.t:6.2f} {fr.Z:8.2f} {estZ} {fr.Zdot:8.2f} {estV} "
              f"{tt:>8} {et:>8}  {ch}  {kp}")

    valid = [(fr, e) for fr, e in rows if e is not None and e.calibrated]
    print("\n--- accuracy (calibrated frames only) ---")
    if len(valid) < 5:
        print(f"  only {len(valid)} calibrated frames -- the plate was never found "
              f"reliably enough to lock absolute scale.")
    else:
        z_err = np.array([abs(e.Z - fr.Z) for fr, e in valid])
        v_err = np.array([abs(e.Zdot - fr.Zdot) for fr, e in valid])
        tail = valid[len(valid) // 2 :]
        z_tail = np.array([abs(e.Z - fr.Z) for fr, e in tail])
        v_tail = np.array([abs(e.Zdot - fr.Zdot) for fr, e in tail])
        print(f"  frames calibrated : {len(valid)}/{len(rows)}")
        print(f"  |Z err|   median  : {np.median(z_err):6.2f} m   "
              f"(steady state {np.median(z_tail):.2f} m)")
        print(f"  |Zdot err| median : {np.median(v_err):6.2f} m/s "
              f"(steady state {np.median(v_tail):.2f} m/s"
              f" = {np.median(v_tail) * 3.6:.2f} km/h)")

    nis = [est.mean_nis for est in pipe.estimators.values()
           if not math.isnan(est.mean_nis)]
    if nis:
        m = float(np.mean(nis))
        verdict = "consistent" if 0.5 < m < 2.0 else (
            "OVERCONFIDENT - raise q or R" if m >= 2.0 else "too loose - lower q or R")
        print(f"  mean NIS          : {m:6.2f}   ({verdict})")

    cost = pipe.cost_summary
    print("\n--- cost (MATH.md section 9) ---")
    print(f"  detector ran on {cost['detector_rate']:5.1%} of frames")
    print(f"  plate    ran on {cost['plate_rate']:5.1%} of frames")
    print(f"  pipeline throughput: {len(rows) / max(proc_s, 1e-9):.1f} fps "
          f"(excludes rendering)")


# --- real video -------------------------------------------------------------------------------


class _TimedDetector:
    """Wraps a detector to account its cost separately -- it is the dominant stage, so
    it is the number that decides detect_stride (MATH.md section 9)."""

    def __init__(self, inner):
        self.inner = inner
        self.seconds = 0.0
        self.calls = 0

    def detect(self, frame_bgr):
        t0 = time.perf_counter()
        try:
            return self.inner.detect(frame_bgr)
        finally:
            self.seconds += time.perf_counter() - t0
            self.calls += 1


def cmd_video(args: argparse.Namespace) -> int:
    """Automatic detection on a video file or a live camera. No manual box needed."""
    is_live = args.source.isdigit()

    try:
        detector = _TimedDetector(DnnVehicleDetector(
            args.model, input_size=args.input_size, conf=args.conf, fmt=args.model_format,
        ))
    except FileNotFoundError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        print("The math is still fully exercisable without a model:", file=sys.stderr)
        print("    python main.py synth --scenario closing", file=sys.stderr)
        return 3

    if is_live:
        # A calibration or noise profile describes one camera, so they are only applied
        # to a camera by default. For a file they must be asked for explicitly.
        args.calib = args.calib or cli_live.DEFAULT_CALIB
        args.noise = args.noise or DEFAULT_NOISE_PROFILE
        req_w, req_h = cli_live.default_resolution(args)
        try:
            cap = open_camera(int(args.source), req_w, req_h)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        nominal_fps = args.fps
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"cannot open source {args.source!r}", file=sys.stderr)
            return 2
        nominal_fps = cap.get(cv2.CAP_PROP_FPS)
        if not nominal_fps or nominal_fps <= 0 or not math.isfinite(nominal_fps):
            print(f"file reports no frame rate; assuming --fps {args.fps:g}. A wrong "
                  f"frame rate scales every speed by the same factor.")
            nominal_fps = args.fps

    ok, frame = cap.read()
    if not ok:
        print("source delivered no frames", file=sys.stderr)
        cap.release()
        return 2
    height, width = frame.shape[:2]

    f_px, calib = cli_live.resolve_intrinsics(args, width, height)
    sigma_w, sigma_s = cli_live.resolve_noise(args, width, height)
    cfg = Config(
        image_width=width, image_height=height, hfov_deg=args.hfov, f_px=f_px,
        fps=nominal_fps, plate_region=args.region, detect_stride=args.detect_stride,
        q=args.q, sigma_w_plate_px=sigma_w, sigma_s_base=sigma_s,
    )
    maps = calib.undistort_maps() if (calib and any(calib.dist)) else None

    locator = ClassicalPlateLocator(region=args.region)
    if args.plate_model:
        locator = DnnPlateLocator(args.plate_model, region=args.region,
                                  fmt=args.model_format)
    pipe = RelativeSpeedPipeline(cfg, detector, locator)

    print(f"source {args.source}: {width}x{height} @ {nominal_fps:.1f} fps nominal, "
          f"f = {cfg.focal_px:.1f} px, detector every {cfg.detect_stride} frames")
    if is_live:
        print("live source: timestamping frames on arrival, not by frame index "
              "(nominal fps is unreliable on cameras and a wrong dt scales every "
              "speed reading)")

    writer = None
    if args.save:
        writer = cv2.VideoWriter(
            args.save, cv2.VideoWriter_fourcc(*"mp4v"), nominal_fps, (width, height)
        )

    i, t0 = 0, time.perf_counter()
    t_arrival0 = t0
    next_print = 0.0
    proc_s = 0.0
    try:
        while True:
            if maps is not None:
                frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
            # A file has a trustworthy frame rate; a camera does not. MATH.md (5.4)
            # divides by the baseline, so dt errors propagate straight into speed.
            t = (time.perf_counter() - t_arrival0) if is_live else (i / nominal_fps)
            if args.duration and t >= args.duration:
                break
            tp = time.perf_counter()
            ests = pipe.process(frame, t)
            lead = pipe.lead(ests)
            proc_s += time.perf_counter() - tp

            if writer is not None or args.show:
                vis = draw_overlay(frame, ests, lead)
                if writer is not None:
                    writer.write(vis)
                if args.show:
                    cv2.imshow("relative speedometer", vis)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                        break
            if lead is not None and t >= next_print:
                next_print = t + 0.5
                ttc = "inf" if not math.isfinite(lead.ttc) else f"{lead.ttc:5.2f}s"
                if lead.calibrated and lead.Zdot is not None:
                    print(f"t={t:6.2f}  #{lead.track_id}  Z={lead.Z:6.2f}m  "
                          f"Zdot={lead.Zdot * 3.6:+7.2f}km/h  TTC={ttc}")
                else:
                    print(f"t={t:6.2f}  #{lead.track_id}  [uncalibrated]  TTC={ttc}")
            i += 1
            ok, frame = cap.read()
            if not ok:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"wrote {args.save}")
        if args.show:
            cv2.destroyAllWindows()

    wall = time.perf_counter() - t0
    print(f"\n{i} frames in {wall:.1f}s ({i / max(wall, 1e-9):.1f} fps end to end)")
    if i:
        det_ms = 1000.0 * detector.seconds / max(1, detector.calls)
        other_ms = 1000.0 * (proc_s - detector.seconds) / i
        print(f"detector {det_ms:.0f} ms/call on {detector.calls} frames "
              f"({detector.calls / i:.0%}); everything else {other_ms:.1f} ms/frame")
    cost = pipe.cost_summary
    print(f"plate locator on {cost['plate_rate']:.1%} of frames")
    return 0


def cmd_fetch_model(args: argparse.Namespace) -> int:
    """Download the ONNX vehicle detector. Plain HTTPS + a checksum; no torch involved."""
    spec = MODEL_ZOO[args.name]
    print(f"{args.name}: {spec.note}")
    print(f"  from {spec.url}")
    try:
        path = fetch_model(args.name, args.dir)
    except (OSError, RuntimeError) as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    print(f"  ok: {path} (sha256 verified)")
    print(f"\nRun it:  python main.py video --source 0 --show --model {path}")
    return 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    """Run every validation suite. Non-zero if any test fails."""
    tests_dir = Path(__file__).parent / "tests"
    rc = 0
    for name in ("test_math.py", "test_live.py", "test_dnn.py"):
        path = tests_dir / name
        if not path.exists():
            continue
        print(f"\n=== {name} ===")
        rc |= subprocess.call([sys.executable, str(path)])
    return rc


# --- cli --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("budget", help="error budget for a camera configuration")
    b.add_argument("--width", type=int, default=1920)
    b.add_argument("--hfov", type=float, default=60.0)
    b.add_argument("--fps", type=float, default=30.0)
    b.add_argument("--region", default="eu", choices=["eu", "us", "jp", "au"])
    b.add_argument("--window-s", type=float, default=0.5)
    b.set_defaults(func=cmd_budget)

    s = sub.add_parser("synth", help="run against synthetic ground truth")
    s.add_argument("--scenario", default="closing", choices=sorted(SCENARIOS))
    s.add_argument("--frames", type=int, default=140)
    s.add_argument("--width", type=int, default=960)
    s.add_argument("--height", type=int, default=540)
    s.add_argument("--hfov", type=float, default=60.0)
    s.add_argument("--fps", type=float, default=30.0)
    s.add_argument("--region", default="eu", choices=["eu", "us", "jp", "au"])
    s.add_argument("--noise", type=float, default=2.5, help="sensor noise sigma")
    s.add_argument("--bbox-jitter", type=float, default=1.5, help="detector box noise px")
    s.add_argument("--detect-stride", type=int, default=1)
    s.add_argument("--q", type=float, default=0.02)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--save", help="write an annotated mp4 here")
    s.set_defaults(func=cmd_synth)

    v = sub.add_parser("video", help="automatic detection on a video file or camera")
    v.add_argument("--source", required=True, help="video path or camera index")
    v.add_argument("--model", default=DEFAULT_MODEL,
                   help="ONNX detector (python main.py fetch-model gets the default)")
    v.add_argument("--model-format", default="auto", choices=list(FORMATS))
    v.add_argument("--input-size", type=int, default=416,
                   help="network input side, multiple of 32; 416 is ~3x cheaper than 640")
    v.add_argument("--plate-model", help="optional ONNX plate detector")
    v.add_argument("--conf", type=float, default=0.35)
    v.add_argument("--calib", default=None,
                   help="calibration file (default: calibration.json for cameras only)")
    v.add_argument("--focal", type=float, help="fx in px, overrides --calib and --hfov")
    v.add_argument("--hfov", type=float, default=60.0, help="last-resort guess for fx")
    v.add_argument("--noise", default=None,
                   help="noise profile (default: noise.json for cameras only)")
    v.add_argument("--sigma-w", type=float, help="plate width noise px, overrides --noise")
    v.add_argument("--sigma-s", type=float, help="scale-ratio noise, overrides --noise")
    v.add_argument("--width", type=int, help="camera mode to request")
    v.add_argument("--height", type=int)
    v.add_argument("--fps", type=float, default=30.0,
                   help="used only when a file reports no frame rate, and for --save")
    v.add_argument("--region", default="eu", choices=["eu", "us", "jp", "au"])
    v.add_argument("--detect-stride", type=int, default=5,
                   help="run the detector every Nth frame; registration tracks between")
    v.add_argument("--q", type=float, default=0.008)
    v.add_argument("--duration", type=float, help="stop after this many seconds")
    v.add_argument("--show", action="store_true")
    v.add_argument("--save", help="write an annotated mp4 here")
    v.set_defaults(func=cmd_video)

    fm = sub.add_parser("fetch-model", help="download the ONNX vehicle detector")
    fm.add_argument("--name", default="yolox_s", choices=sorted(MODEL_ZOO))
    fm.add_argument("--dir", default="models")
    fm.set_defaults(func=cmd_fetch_model)

    t = sub.add_parser("selftest", help="validate every derivation in MATH.md")
    t.set_defaults(func=cmd_selftest)

    # Hardware-facing commands live in their own module -- they are the only ones that
    # touch a device, block on GUI input, and depend on the machine not the maths.
    cli_live.register(sub)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
