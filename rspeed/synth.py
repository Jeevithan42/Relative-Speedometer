"""Synthetic rear-of-vehicle renderer with exact ground truth.

The point of this module is that every number the pipeline produces can be checked
against a known answer before any real video is involved. Given a depth profile Z(t),
it renders a rear view whose plate and body widths obey the pinhole relation (1.1)
exactly, so `Z`, `Zdot` and `TTC` are known to machine precision at every frame.

Rendering is supersampled 4x and downsampled with INTER_AREA. That matters: the whole
project turns on subpixel width accuracy (MATH.md section 3), so a renderer that
snapped geometry to integer pixels would quantise the very signal under test and make
a broken estimator look fine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .detector import Detection
from .geometry import PLATE_WIDTHS_M, width_from_depth

SUPERSAMPLE = 4


@dataclass
class SceneSpec:
    """Real-world dimensions of the simulated lead vehicle, metres."""

    car_width_m: float = 1.80
    car_height_m: float = 1.45
    plate_region: str = "eu"
    plate_drop_frac: float = 0.62  # plate centre, as a fraction of car height down
    horizon_frac: float = 0.46  # image row of the horizon

    @property
    def plate_width_m(self) -> float:
        return PLATE_WIDTHS_M[self.plate_region]

    @property
    def plate_height_m(self) -> float:
        return self.plate_width_m * (110.0 / 520.0)


@dataclass
class Frame:
    """One rendered frame plus the ground truth that produced it."""

    image: np.ndarray
    t: float
    Z: float
    Zdot: float
    ttc: float
    w_plate_px: float
    w_car_px: float
    car_bbox: tuple[int, int, int, int]


@dataclass
class SyntheticSequence:
    """Renders a depth profile into frames with ground truth attached."""

    f_px: float
    image_width: int = 1280
    image_height: int = 720
    fps: float = 30.0
    scene: SceneSpec = field(default_factory=SceneSpec)
    noise_sigma: float = 2.5
    seed: int = 7
    lateral_jitter_px: float = 0.0  # simulate ego yaw / road camber

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._bg: np.ndarray | None = None
        self._noise_buf: np.ndarray | None = None
        # OpenCV's RNG is global; seeding it here keeps sequences reproducible, at the
        # cost of two SyntheticSequence objects interleaved in one process sharing a
        # noise stream. Fine for tests and demos, worth knowing before relying on it.
        cv2.setRNGSeed(int(self.seed))

    # -- geometry ---------------------------------------------------------------------

    def _project(self, Z: float) -> tuple[float, float, float, float]:
        """Return (w_car, h_car, w_plate, h_plate) in pixels at depth Z."""
        s = self.scene
        return (
            width_from_depth(Z, self.f_px, s.car_width_m),
            width_from_depth(Z, self.f_px, s.car_height_m),
            width_from_depth(Z, self.f_px, s.plate_width_m),
            width_from_depth(Z, self.f_px, s.plate_height_m),
        )

    # -- rendering --------------------------------------------------------------------

    def _background(self) -> np.ndarray:
        """Static sky/road plate, built once and reused.

        The background carries no measurement signal, so it does not need
        supersampling -- only the vehicle does. Caching it is what makes the
        renderer fast enough to run the whole test suite in seconds.
        """
        if self._bg is None:
            W, H = self.image_width, self.image_height
            bg = np.zeros((H, W, 3), dtype=np.uint8)
            horizon = int(self.scene.horizon_frac * H)
            bg[:horizon] = (190, 175, 160)
            bg[horizon:] = (72, 72, 76)
            cv2.line(bg, (W // 2, horizon), (W // 2, H), (150, 150, 150), 1)
            self._bg = bg
        return self._bg.copy()

    def render(
        self, Z: float, t: float
    ) -> tuple[np.ndarray, tuple[int, int, int, int], float, float]:
        """Render one frame. Geometry obeys (1.1) exactly, to subpixel precision.

        Only the tile containing the vehicle is supersampled. Full-frame 4x
        supersampling at 720p means a 44 MB intermediate per frame, which is both slow
        and prone to allocation failure at close range where the car fills the view.
        The tile carries all the geometry that matters, so the accuracy is identical.
        """
        S = SUPERSAMPLE
        W, H = self.image_width, self.image_height
        img = self._background()

        w_car, h_car, w_plate, h_plate = self._project(Z)
        horizon = self.scene.horizon_frac * H

        jitter = 0.0
        if self.lateral_jitter_px > 0:
            jitter = float(self._rng.normal(0.0, self.lateral_jitter_px))
        cx = W / 2.0 + jitter
        # Bottom of the car sits on the road, so it rides up the image as it recedes.
        cy_bottom = horizon + h_car * 0.55
        cy_top = cy_bottom - h_car

        bbox = (
            int(round(cx - w_car / 2)),
            int(round(cy_top)),
            int(round(w_car)),
            int(round(h_car)),
        )

        margin = max(6.0, 0.18 * w_car)
        tx0 = max(0, int(math.floor(cx - w_car / 2 - margin)))
        ty0 = max(0, int(math.floor(cy_top - margin)))
        tx1 = min(W, int(math.ceil(cx + w_car / 2 + margin)))
        ty1 = min(H, int(math.ceil(cy_bottom + margin)))
        tw, th = tx1 - tx0, ty1 - ty0
        if tw < 2 or th < 2:
            return self._finish(img), bbox, w_plate, w_car

        tile = cv2.resize(
            img[ty0:ty1, tx0:tx1], (tw * S, th * S), interpolation=cv2.INTER_LINEAR
        )

        def rect(x0, y0, x1, y1, color, thickness=-1):
            # Tile-local supersampled coordinates. lineType must be a connectivity
            # (LINE_8 / LINE_AA); cv2.FILLED is a *thickness* sentinel, not a lineType.
            cv2.rectangle(
                tile,
                (int(round((x0 - tx0) * S)), int(round((y0 - ty0) * S))),
                (int(round((x1 - tx0) * S)), int(round((y1 - ty0) * S))),
                color,
                thickness,
                lineType=cv2.LINE_8,
            )

        # body
        rect(cx - w_car / 2, cy_top, cx + w_car / 2, cy_bottom, (58, 52, 120))
        # rear window
        rect(cx - w_car * 0.40, cy_top + h_car * 0.08,
             cx + w_car * 0.40, cy_top + h_car * 0.36, (28, 28, 34))
        # taillights -- the night-time alternative feature (MATH.md section 6 table)
        for sign in (-1, 1):
            x0 = cx + sign * w_car * 0.50 - (w_car * 0.16 if sign > 0 else 0)
            rect(x0, cy_top + h_car * 0.44, x0 + w_car * 0.16,
                 cy_top + h_car * 0.60, (40, 40, 200))
        # bumper
        rect(cx - w_car / 2, cy_bottom - h_car * 0.16,
             cx + w_car / 2, cy_bottom - h_car * 0.08, (40, 36, 84))

        # plate: bright field, dark border, character blocks
        py = cy_top + h_car * self.scene.plate_drop_frac
        px0, px1 = cx - w_plate / 2, cx + w_plate / 2
        py0, py1 = py - h_plate / 2, py + h_plate / 2
        rect(px0, py0, px1, py1, (238, 240, 238))
        rect(px0, py0, px1, py1, (30, 30, 30), max(1, int(round(S * 0.6))))
        n_chars = 7
        for i in range(n_chars):
            frac0 = 0.07 + i * (0.86 / n_chars)
            frac1 = frac0 + (0.86 / n_chars) * 0.55
            rect(px0 + w_plate * frac0, py0 + h_plate * 0.18,
                 px0 + w_plate * frac1, py1 - h_plate * 0.18, (26, 26, 26))

        img[ty0:ty1, tx0:tx1] = cv2.resize(
            tile, (tw, th), interpolation=cv2.INTER_AREA
        )
        return self._finish(img), bbox, w_plate, w_car

    def _finish(self, img: np.ndarray) -> np.ndarray:
        """Apply sensor noise at the output resolution, where it physically belongs.

        cv2.randn rather than rng.normal: the numpy path costs ~120 ms per 720p frame
        and dominated total render time, while cv2's SIMD implementation is ~40x
        faster. It draws from OpenCV's global RNG, seeded in __post_init__, so
        sequences stay reproducible.
        """
        if self.noise_sigma <= 0:
            return img
        if self._noise_buf is None or self._noise_buf.shape != img.shape:
            self._noise_buf = np.empty(img.shape, dtype=np.int16)
        cv2.randn(self._noise_buf, 0.0, self.noise_sigma)
        return cv2.add(img, self._noise_buf, dtype=cv2.CV_8U)

    # -- sequence ---------------------------------------------------------------------

    def run(
        self, depth_profile: Callable[[float], float], n_frames: int
    ) -> list[Frame]:
        """Render n_frames of a depth profile Z(t).

        Zdot is taken by central difference on the *profile itself*, not on anything
        the pipeline sees, so it is exact ground truth.
        """
        dt = 1.0 / self.fps
        frames: list[Frame] = []
        for i in range(n_frames):
            t = i * dt
            Z = depth_profile(t)
            if Z <= 0.5:
                break
            h = dt * 0.5
            Zdot = (depth_profile(t + h) - depth_profile(max(0.0, t - h))) / (
                (t + h) - max(0.0, t - h)
            )
            img, bbox, w_plate, w_car = self.render(Z, t)
            ttc = math.inf if Zdot >= 0 else -Z / Zdot
            frames.append(
                Frame(
                    image=img, t=t, Z=Z, Zdot=Zdot, ttc=ttc,
                    w_plate_px=w_plate, w_car_px=w_car, car_bbox=bbox,
                )
            )
        return frames

    def detector_from(self, frames: list[Frame], jitter_px: float = 1.5, seed: int = 11):
        """A ScriptedDetector-compatible callable replaying the true boxes with jitter.

        The jitter is the point: it stands in for real box-regression noise, so tests
        exercise the case where Channel A's large-feature measurement is noisy and
        Channel B has to carry the velocity.

        The box is looked up by the image it is handed, not by the call count. A strided
        pipeline calls the detector every Nth frame, and indexing by call count silently
        replayed frame 10's box on frame 50 -- which is what made every detect_stride > 1
        run look broken.
        """
        rng = np.random.default_rng(seed)
        by_image = {id(fr.image): i for i, fr in enumerate(frames)}

        def fn(index: int, frame: np.ndarray) -> list[Detection]:
            index = by_image.get(id(frame), index)
            if index >= len(frames):
                return []
            x, y, w, h = frames[index].car_bbox
            if jitter_px > 0:
                x += int(round(rng.normal(0, jitter_px)))
                y += int(round(rng.normal(0, jitter_px)))
                w += int(round(rng.normal(0, jitter_px)))
                h += int(round(rng.normal(0, jitter_px)))
            return [Detection(bbox=(x, y, max(8, w), max(8, h)), score=0.9, cls=2)]

        return fn


# --- ready-made depth profiles ----------------------------------------------------------


def constant_closing(Z0: float = 40.0, v: float = 5.0) -> Callable[[float], float]:
    """Steady approach at v m/s. The case the process model (7.1)-(7.2) is exact for."""
    return lambda t: Z0 - v * t


def constant_gap(Z0: float = 25.0) -> Callable[[float], float]:
    """Perfect car-following. Zdot = 0, TTC = +inf."""
    return lambda t: Z0


def opening(Z0: float = 15.0, v: float = 4.0) -> Callable[[float], float]:
    """Lead vehicle pulling away."""
    return lambda t: Z0 + v * t


def brake_event(
    Z0: float = 35.0, v0: float = 2.0, brake_t: float = 1.5, decel: float = 6.0
) -> Callable[[float], float]:
    """Cruise, then the lead vehicle brakes hard.

    This is the adversarial case for the filter: the constant-relative-velocity
    process model is violated exactly when the answer matters most, so it is the
    right test of whether Q is sized correctly (MATH.md section 10).
    """

    def profile(t: float) -> float:
        if t < brake_t:
            return Z0 - v0 * t
        dt = t - brake_t
        return Z0 - v0 * brake_t - v0 * dt - 0.5 * decel * dt * dt

    return profile


def oscillating(Z0: float = 25.0, amp: float = 3.0, period: float = 4.0):
    """Depth wobble -- stresses the filter's ability to track sign changes."""
    return lambda t: Z0 + amp * math.sin(2 * math.pi * t / period)
