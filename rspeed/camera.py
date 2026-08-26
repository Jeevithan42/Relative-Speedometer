"""Live camera capture and intrinsic calibration.

Two things here that the synthetic path never needed:

1. **Wall-clock timing.** A file has a reliable frame rate; a webcam does not -- many
   report `CAP_PROP_FPS = -1`, and the true rate drifts with exposure and USB
   contention. Since MATH.md (5.4) divides by the baseline `dtau`, a wrong dt scales
   every speed reading by exactly the same factor. Live sources must be timestamped
   on arrival, never by frame index.

2. **Real intrinsics.** Absolute range is proportional to `f` (MATH.md 2.1), so a
   guessed focal length is a proportional error in every metre reported. Two routes:
   `calibrate_checkerboard` (proper, also gives distortion) and `focal_from_known_object`
   (one measurement, no printing required).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Calibration:
    """Camera intrinsics. `fx` is the number that matters -- widths are horizontal."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float]
    image_width: int
    image_height: int
    rms_reproj_error: float = 0.0
    method: str = "checkerboard"
    n_views: int = 0

    @property
    def hfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan(self.image_width / (2.0 * self.fx)))

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.array(self.dist, dtype=np.float64)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "Calibration":
        return Calibration(**json.loads(Path(path).read_text(encoding="utf-8")))

    def undistort_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """Precomputed remap tables. Undistorting matters more than it looks: an
        uncorrected plate near the frame edge reads a biased width (MATH.md 11)."""
        size = (self.image_width, self.image_height)
        newK, _roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coeffs, size, 0)
        return cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, None, newK, size, cv2.CV_16SC2
        )

    def summary(self) -> str:
        return (
            f"fx={self.fx:.1f}px fy={self.fy:.1f}px  "
            f"({self.image_width}x{self.image_height}, HFOV {self.hfov_deg:.1f} deg)  "
            f"[{self.method}, {self.n_views} views, rms {self.rms_reproj_error:.3f}px]"
        )


# --- enumeration and capture ---------------------------------------------------------------


def list_cameras(
    max_index: int = 5, probe_frames: int = 20, settle_s: float = 3.0
) -> list[dict]:
    """Probe camera indices and MEASURE the true frame rate.

    The reported CAP_PROP_FPS is not trustworthy on USB cameras, so this times actual
    frame delivery instead. It also flags the common trap where a higher resolution is
    accepted but collapses the frame rate, because the driver falls back to
    uncompressed transfer over a bandwidth-limited bus.
    """
    found: list[dict] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FPS, 30.0)
        # Discard buffered frames before timing: they arrive almost instantly and
        # otherwise inflate the measured rate by roughly 4x. Then let auto-exposure
        # settle, or a slow-ramping camera is wrongly reported as too dark.
        for _ in range(10):
            cap.read()
        exposure = settle_exposure(cap, max_seconds=settle_s)
        ok, f = cap.read()
        if not ok:
            cap.release()
            continue
        info = {
            "index": i,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "reported_fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "measured_fps": _measure_fps(cap, probe_frames),
            "brightness": exposure["mean"],
            "verdict": exposure["verdict"],
        }
        cap.release()
        found.append(info)
    return found


def _measure_fps(cap: cv2.VideoCapture, n: int) -> float:
    t0 = time.perf_counter()
    got = 0
    while got < n:
        ok, _ = cap.read()
        if not ok:
            break
        got += 1
    dt = time.perf_counter() - t0
    return got / dt if dt > 0 else 0.0


def probe_modes(
    index: int, modes: tuple[tuple[int, int], ...] = ((640, 480), (1280, 720), (1920, 1080))
) -> list[dict]:
    """Try each resolution and measure what the camera actually delivers.

    Worth running before trusting a mode: resolution buys pixels on the plate, but
    frame rate buys measurement baseline, and on many webcams you cannot have both.
    """
    out: list[dict] = []
    for rw, rh in modes:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            continue
        out.append({
            "requested": (rw, rh),
            "actual": (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "measured_fps": _measure_fps(cap, 30),
        })
        cap.release()
    return out


def open_camera(
    index: int, width: int | None = None, height: int | None = None,
    fourcc: str | None = None, warmup: int = 12, target_fps: float | None = 30.0,
) -> cv2.VideoCapture:
    """Open a camera in a mode suitable for measurement, discarding warm-up frames.

    Three things here are not cosmetic:

    * **Requesting a frame rate.** Left alone, auto-exposure in a dim scene stretches
      the exposure until delivery collapses -- measured 7.5 fps on the test camera,
      versus 30 fps once a rate is requested. Frame rate is measurement baseline
      (MATH.md 3.1), and a long exposure also motion-blurs exactly the edges that
      Channel A needs at subpixel accuracy.
    * **Warm-up.** The first frames are mid-auto-exposure and mid-auto-focus, so their
      apparent widths are unreliable precisely while the filter is initialising and
      most sensitive to a bad measurement.
    * **Discarding, not just reading.** Frames buffered at open are delivered almost
      instantly and will fool any frame-rate measurement into reporting ~4x the truth.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {index}")
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if target_fps:
        cap.set(cv2.CAP_PROP_FPS, float(target_fps))
    for _ in range(max(0, warmup)):
        cap.read()
    settle_exposure(cap)
    return cap


def settle_exposure(
    cap: cv2.VideoCapture,
    max_seconds: float = 4.0,
    tol: float = 0.04,
    stable_needed: int = 3,
) -> dict:
    """Read frames until auto-exposure stops changing, or max_seconds elapses.

    A frame-count warm-up is the wrong unit. Auto-exposure on some cameras (the
    Microsoft LifeCam measured here) ramps for several *seconds* in a dim scene --
    brightness climbed 1.7 -> 6.0 over five seconds. Any width measured during that
    ramp is taken under changing gain and contrast, which is exactly when the subpixel
    edge fit is least trustworthy and the filter is most sensitive to it.

    Returns the final exposure_report.
    """
    t0 = time.perf_counter()
    prev: float | None = None
    stable = 0
    report = {"mean": 0.0, "gradient": 0.0, "verdict": "no frames", "usable": False}

    while time.perf_counter() - t0 < max_seconds:
        ok, frame = cap.read()
        if not ok:
            break
        report = exposure_report(frame)
        mean = report["mean"]
        if prev is not None:
            # Relative change, so the test behaves the same at any light level.
            if abs(mean - prev) <= tol * max(prev, 1.0):
                stable += 1
                if stable >= stable_needed:
                    break
            else:
                stable = 0
        prev = mean
    return report


# Below this mean level the frame carries no usable gradient, so subpixel edge fitting
# (MATH.md 3) returns noise and every quality gate rejects it -- with no obvious reason
# visible to the user. Worth catching explicitly.
DARK_FRAME_MEAN = 12.0


def exposure_report(frame: np.ndarray) -> dict:
    """Assess whether a frame is measurable at all.

    Checks mean level and gradient energy. A covered lens, a closed privacy shutter or
    an unlit room all produce a frame that looks merely 'dark' but is actually devoid
    of the edges this project measures.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    mean = float(gray.mean())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad = float(np.abs(gx).mean())
    if mean < DARK_FRAME_MEAN and grad < 0.5:
        # No level AND no structure: nothing is reaching the sensor at all.
        verdict = "BLOCKED -- lens covered or privacy shutter closed"
    elif mean < DARK_FRAME_MEAN:
        # Some structure is getting through, so the lens is clear; there is simply
        # not enough light. Distinguishing these matters -- the fixes are different.
        verdict = "TOO DARK -- lens is clear but the scene needs much more light"
    elif grad < 1.0:
        verdict = "NO DETAIL -- out of focus, or pointed at a blank surface"
    elif mean > 245:
        verdict = "BLOWN OUT -- pointed at a light source"
    else:
        verdict = "ok"
    return {"mean": mean, "gradient": grad, "verdict": verdict,
            "usable": verdict == "ok"}


# --- calibration: checkerboard -----------------------------------------------------------------


def calibrate_checkerboard(
    cap: cv2.VideoCapture,
    board: tuple[int, int] = (9, 6),
    square_m: float = 0.025,
    n_required: int = 12,
    window: str = "calibrate",
) -> Calibration:
    """Interactive checkerboard calibration.

    `board` is the number of INNER corners (a printed 10x7 squares board has 9x6 inner
    corners). `square_m` only scales the extrinsics, not fx/fy, so a rough value is
    fine here -- but get the inner-corner count exactly right or detection fails
    silently on every frame.

    Controls: SPACE captures the current view, ENTER solves, ESC aborts.
    """
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board[0], 0 : board[1]].T.reshape(-1, 2) * square_m

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    size: tuple[int, int] | None = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(f"\nShow a {board[0]}x{board[1]} inner-corner checkerboard to the camera.")
    print("SPACE = capture (vary angle and distance each time), ENTER = solve, ESC = abort\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(
            gray, board,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
        )
        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, board, corners, found)
        colour = (90, 220, 110) if found else (70, 70, 240)
        cv2.putText(vis, f"captured {len(obj_points)}/{n_required}"
                         f"{'   BOARD VISIBLE' if found else '   no board'}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
        cv2.putText(vis, "SPACE capture   ENTER solve   ESC abort",
                    (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow(window, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("calibration aborted")
        if key == 32 and found:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp.copy())
            img_points.append(refined)
            print(f"  captured view {len(obj_points)}")
        if key in (13, 10) and len(obj_points) >= 4:
            break
        if len(obj_points) >= n_required and key in (13, 10):
            break

    cv2.destroyWindow(window)
    if len(obj_points) < 4 or size is None:
        raise RuntimeError(f"need at least 4 views, got {len(obj_points)}")

    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, size, None, None
    )
    return Calibration(
        fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
        dist=[float(d) for d in dist.ravel()],
        image_width=size[0], image_height=size[1],
        rms_reproj_error=float(rms), method="checkerboard", n_views=len(obj_points),
    )


# --- calibration: known object -------------------------------------------------------------------


def focal_from_known_object(width_px: float, width_m: float, distance_m: float) -> float:
    """f = w_px * Z / W -- MATH.md (1.1) rearranged for f.

    One measurement of a known object at a measured distance. Less accurate than a
    checkerboard and gives no distortion coefficients, but it needs nothing printed
    and the error is easy to reason about: it is simply the fractional error in your
    distance measurement plus that in your pixel measurement.
    """
    if width_px <= 0 or width_m <= 0 or distance_m <= 0:
        raise ValueError("all arguments must be positive")
    return width_px * distance_m / width_m


def measure_object_width_interactive(
    cap: cv2.VideoCapture, window: str = "measure"
) -> tuple[float, np.ndarray]:
    """Let the user click the left and right edges of a known object.

    Returns (width_px, frame). Click accuracy of a few pixels is fine here: at a 2 m
    calibration distance the object fills a large part of the frame, so a 3 px error
    on a 400 px object is under 1%.
    """
    clicks: list[tuple[int, int]] = []
    frozen: dict[str, np.ndarray] = {}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print("\nSPACE freezes the frame, then click the LEFT and RIGHT edges of the object.")
    print("R resets the clicks, ENTER accepts, ESC aborts.\n")

    while True:
        if "img" in frozen:
            frame = frozen["img"]
        else:
            ok, frame = cap.read()
            if not ok:
                break
        vis = frame.copy()
        for c in clicks:
            cv2.line(vis, (c[0], 0), (c[0], vis.shape[0]), (90, 220, 110), 1)
            cv2.circle(vis, c, 4, (90, 220, 110), -1)
        if len(clicks) == 2:
            w = abs(clicks[1][0] - clicks[0][0])
            cv2.putText(vis, f"width = {w} px   ENTER to accept", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 220, 110), 2, cv2.LINE_AA)
        else:
            msg = "click LEFT then RIGHT edge" if "img" in frozen else "SPACE to freeze"
            cv2.putText(vis, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (60, 190, 240), 2, cv2.LINE_AA)
        cv2.imshow(window, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("measurement aborted")
        if key == 32 and "img" not in frozen:
            frozen["img"] = frame.copy()
        if key in (ord("r"), ord("R")):
            clicks.clear()
            frozen.pop("img", None)
        if key in (13, 10) and len(clicks) == 2:
            cv2.destroyWindow(window)
            return float(abs(clicks[1][0] - clicks[0][0])), frozen.get("img", frame)

    cv2.destroyWindow(window)
    raise RuntimeError("measurement window closed before a width was accepted")
