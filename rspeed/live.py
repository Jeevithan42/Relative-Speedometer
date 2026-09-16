"""Live camera session with a manually selected target.

Why this exists: the measurement mathematics does not care what the target is, only
that it has a known width and two findable edges. So you can exercise the entire
stack -- subpixel widths, the kappa transfer, both channels, the EKF -- against a
real camera and a real object on a desk, with no vehicle detector and no model
weights. It is also a better *first* test than YOLO, because it isolates the
measurement path from detector noise.

Two things differ from the file-based path and both matter for correctness:

  * **Wall-clock timestamps.** Frame index over nominal fps is wrong on a webcam
    (this one reports fps = -1). MATH.md (5.4) divides by the baseline, so a wrong dt
    scales every speed reading by the same factor.
  * **Measured, not assumed, frame interval.** The session reports the true rate so
    you can see whether the camera is actually delivering what it claims.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .detector import Track
from .estimator import SpeedEstimate, VehicleEstimator
from .plate import PlateLocator

# Objects whose width you can look up or measure once, for --object.
KNOWN_OBJECTS_M = {
    "card": 0.0856,  # ID-1 credit/bank card, ISO/IEC 7810 -- 85.60 mm
    "a4-portrait": 0.210,  # A4 short edge
    "a4-landscape": 0.297,  # A4 long edge
    "letter-portrait": 0.216,  # US Letter short edge, 8.5 in
    "cd": 0.120,  # CD/DVD disc
    "plate-eu": 0.520,
    "plate-us": 0.305,
}


@dataclass
class LiveStats:
    frames: int = 0
    measured_fps: float = 0.0
    lost_frames: int = 0  # frames on which registration could not place the target
    ms_per_frame: float = 0.0  # processing cost, excluding capture and display


class ManualTarget:
    """A user-selected box, moved frame to frame by Channel B's own registration.

    There is no separate tracker. The ECC warp that yields the scale ratio also yields
    where the keyframe's anchor now sits (MATH.md 5.4), and the estimator moves the box
    with it. That replaced TrackerMIL, which cost ~110-140 ms/frame -- over 90% of the
    whole live loop -- and never changed its box size, so the plate bracket drifted off
    any target whose distance was actually changing.
    """

    def __init__(self, bbox: tuple[int, int, int, int], tracked: bool = True):
        """`tracked=False` keeps the box under external control -- used by tests
        driving known trajectories, and by anyone feeding boxes from their own
        detector. Such a box counts as a measurement; a tracked one does not."""
        self.bbox = tuple(int(v) for v in bbox)
        self.tracked = bool(tracked)

    @property
    def tracker_name(self) -> str:
        return "ECC registration (Channel B)" if self.tracked else "none (external box)"

    def as_track(self, track_id: int = 1) -> Track:
        x, y, w, h = self.bbox
        return Track(id=track_id, bbox=(x, y, max(4, w), max(4, h)), score=1.0, cls=2)


class LiveSession:
    """Drives one manually selected target from a live camera or a video file."""

    def __init__(
        self,
        cfg: Config,
        locator: PlateLocator | None,
        undistort_maps: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.cfg = cfg
        self.locator = locator
        self.undistort_maps = undistort_maps
        self.estimator: VehicleEstimator | None = None
        self.target: ManualTarget | None = None
        self.stats = LiveStats()
        self._t0: float | None = None

    # -- frame prep -------------------------------------------------------------------

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        if self.undistort_maps is None:
            return frame
        return cv2.remap(frame, self.undistort_maps[0], self.undistort_maps[1],
                         cv2.INTER_LINEAR)

    # -- target selection --------------------------------------------------------------

    def select_target(self, frame: np.ndarray, window: str = "select target") -> bool:
        """Blocking ROI selection. Returns False if the user cancelled."""
        box = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window)
        if box is None or box[2] < 8 or box[3] < 8:
            return False
        self.set_target(frame, box)
        return True

    def set_target(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        use_tracker: bool = True,
    ) -> None:
        """Non-interactive target selection, for scripted tests and external detectors.

        `frame` is unused now that tracking needs no initialisation image; it is kept so
        callers written against the tracker-based API still work.
        """
        self.target = ManualTarget(bbox, tracked=use_tracker)
        self.estimator = VehicleEstimator(1, self.cfg, self.locator)
        self._t0 = None

    def reset(self) -> None:
        self.target = None
        self.estimator = None
        self._t0 = None

    # -- per frame ---------------------------------------------------------------------

    def step(self, frame: np.ndarray, t: float | None = None) -> SpeedEstimate | None:
        """Process one frame. `t` defaults to wall-clock seconds since the first frame.

        Passing t explicitly is for replaying a file at its true timestamps; leaving it
        None is correct for a live camera, where only arrival time is meaningful.
        """
        if self.target is None or self.estimator is None:
            return None

        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        if t is None:
            t = now - self._t0

        t_work = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        track = self.target.as_track()
        est = self.estimator.update(gray, track, t, box_measured=not self.target.tracked)
        self.target.bbox = track.bbox  # the estimator moved it when tracking
        if not est.tracking_ok:
            self.stats.lost_frames += 1

        self.stats.frames += 1
        work_ms = (time.perf_counter() - t_work) * 1000.0
        # Exponential average: responsive, and not dominated by a slow first frame.
        a = 0.1 if self.stats.frames > 1 else 1.0
        self.stats.ms_per_frame += a * (work_ms - self.stats.ms_per_frame)
        elapsed = now - self._t0
        if elapsed > 0:
            self.stats.measured_fps = self.stats.frames / elapsed
        return est


def draw_live_overlay(
    frame: np.ndarray,
    est: SpeedEstimate | None,
    stats: LiveStats,
    cfg: Config,
    hint: str = "",
) -> np.ndarray:
    """Overlay tuned for live use: big readouts, plus the provenance you need to
    tell a real measurement from a confidently-wrong one."""
    out = frame.copy()
    h, w = out.shape[:2]

    if est is None:
        cv2.putText(out, hint or "press S to select a target", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 190, 240), 2, cv2.LINE_AA)
        return out

    x, y, bw, bh = est.bbox
    if not est.tracking_ok:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (70, 70, 240), 1)
        cv2.putText(out, "TARGET LOST -- press S to reselect", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 240), 2, cv2.LINE_AA)
        return out
    closing = est.Zdot is not None and est.Zdot < 0
    colour = (70, 70, 240) if (math.isfinite(est.ttc) and est.ttc < 3.0) else (
        (60, 190, 240) if closing else (90, 220, 110))
    cv2.rectangle(out, (x, y), (x + bw, y + bh), colour, 2)

    if est.plate is not None:
        px, py, pw, ph = est.plate.bbox
        cv2.line(out, (int(est.plate.center[0] - est.plate.w_px / 2), py + ph // 2),
                 (int(est.plate.center[0] + est.plate.w_px / 2), py + ph // 2),
                 (60, 190, 240), 2)

    lines: list[tuple[str, tuple[int, int, int], float]] = []
    if est.calibrated and est.Zdot is not None:
        lines.append((f"{est.Zdot * 3.6:+.1f} km/h", colour, 1.1))
        lines.append((f"{est.Z:.2f} m  +-{est.sigma_Z:.2f}", (245, 245, 245), 0.7))
    else:
        lines.append(("-- no anchor --", (170, 170, 170), 0.9))
    ttc_txt = "TTC inf" if not math.isfinite(est.ttc) else f"TTC {est.ttc:.2f}s"
    lines.append((ttc_txt, colour, 0.8))

    yy = 42
    for text, col, sc in lines:
        cv2.putText(out, text, (16, yy), cv2.FONT_HERSHEY_SIMPLEX, sc, col,
                    2 if sc > 0.75 else 1, cv2.LINE_AA)
        yy += int(34 * sc) + 10

    diag = [
        f"A:{est.channel_a}  B:{'yes' if est.channel_b else 'no'}",
        f"kappa {'locked' if est.kappa_locked else f'open n={est.kappa_samples}'}",
        f"W={cfg.plate_width_m * 100:.1f}cm  f={cfg.focal_px:.0f}px",
        f"{stats.measured_fps:.1f} fps  ({stats.ms_per_frame:.1f} ms processing)",
    ]
    if est.scale is not None:
        diag.append(f"s={est.scale.s:.4f} dt={est.scale.dtau:.2f}s cc={est.scale.confidence:.2f}")
    if est.plate is not None:
        diag.append(f"w={est.plate.w_px:.2f}px q={est.plate.quality:.2f}")
    for i, d in enumerate(diag):
        cv2.putText(out, d, (16, h - 16 - 18 * (len(diag) - 1 - i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(out, "S reselect   R reset filter   Q quit", (w - 340, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return out
