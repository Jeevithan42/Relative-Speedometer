"""Empirical noise characterisation -- MATH.md section 10.

`sigma_w_plate_px` and `sigma_s_base` are the two numbers that set the accuracy of
everything this project reports, and the defaults in config.py are placeholders. This
module measures them on YOUR camera, which is the procedure MATH.md prescribes:
point at a static scene, hold still, and look at the spread of repeated measurements
of something that is not moving.

The catch is that a static scene is the whole premise. If the camera or the target
moves during the capture, the spread includes real motion and the measured sigma is
too large -- so stationarity is checked first and reported, never assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .plate import refine_plate_width
from .scale import ScaleEstimator

BBox = tuple[int, int, int, int]


@dataclass
class NoiseReport:
    n_frames: int
    stationary: bool
    motion_px: float  # median inter-frame displacement of the scene
    roi: BBox

    n_width: int
    mean_w_px: float
    sigma_w_px: float  # -> config.sigma_w_plate_px
    mean_quality: float
    stable_frac: float  # fraction of widths within 2*sigma of the median

    n_scale: int
    sigma_s: float  # -> config.sigma_s_base
    scale_bias: float  # mean(s) - 1; should be ~0 on a static scene

    measured_fps: float

    @property
    def usable(self) -> bool:
        return (self.stationary and self.n_width >= 20 and self.n_scale >= 10
                and not math.isnan(self.sigma_w_px)
                and self.sigma_w_px < 0.05 * max(self.mean_w_px, 1.0))

    def implied_speed_noise(self, f_px: float, width_m: float, Z: float,
                            n_frames: int = 15, dt: float = 1 / 30.0) -> float:
        """What this camera's measured sigma_w means for speed error at range Z."""
        from .geometry import speed_noise_from_width_noise, width_from_depth
        w = width_from_depth(Z, f_px, width_m)
        return speed_noise_from_width_noise(self.sigma_w_px, w, Z, n_frames, dt)


def _candidate_boxes(shape: tuple[int, int]) -> list[BBox]:
    h, w = shape[:2]
    boxes: list[BBox] = []
    # Kept away from the frame border: a window near the edge loses content as the
    # scene drifts, which biases the affine scale fit (measured: +0.15 on a static
    # scene from an edge-adjacent ROI, versus +0.002 from a central one).
    for fw in (0.4, 0.3, 0.2, 0.15):
        for fh in (0.4, 0.3, 0.2):
            bw, bh = int(w * fw), int(h * fh)
            for cx_f in (0.4, 0.5, 0.6):
                for cy_f in (0.4, 0.5, 0.6):
                    x, y = int(w * cx_f - bw / 2), int(h * cy_f - bh / 2)
                    if x >= 0 and y >= 0 and x + bw <= w and y + bh <= h:
                        boxes.append((x, y, bw, bh))
    return boxes


def find_measurable_roi(
    grays: list[np.ndarray], min_quality: float = 0.0, sample: int = 12
) -> tuple[BBox, float] | None:
    """Find a window whose measured width is *stable over time*, not merely contrasty.

    Selecting on single-frame edge quality is wrong, and measurably so: on a cluttered
    desk scene it picked a window scoring 0.23 quality whose width then varied by 70 px
    on a 197 px mean. High contrast was present, but the two strongest gradients in the
    window belonged to different objects from frame to frame, so the "width" jumped
    between unrelated pairs rather than fluctuating about one value.

    What the noise measurement needs is a region where the fitter locks onto the *same*
    pair of edges every frame. That is a temporal property, so it is scored temporally:
    lowest relative spread wins.

    `min_quality` defaults to 0 deliberately. Quality is a normalised edge-contrast
    heuristic, and gating on it rejected a perfectly measurable scene: every one of 108
    candidate windows tracked a width with 0.17% spread, yet the best scored 0.042
    against a 0.05 threshold. Stability is direct evidence that the fitter is following
    one feature; contrast is only a proxy for it. Where they disagree, believe the
    stability and let the caller decide what to do about low contrast.
    """
    if not grays:
        return None
    step = max(1, len(grays) // sample)
    probe = grays[::step][:sample]

    best: tuple[BBox, float] | None = None
    best_spread = math.inf
    for box in _candidate_boxes(grays[0].shape):
        widths, quals = [], []
        for g in probe:
            out = refine_plate_width(g, box, pad_frac=0.25)
            if out is None:
                continue
            widths.append(out[0])
            quals.append(out[2])
        # Require a measurement in nearly every probe frame: intermittent success means
        # the fitter is not reliably seeing the same thing.
        if len(widths) < max(4, int(0.8 * len(probe))):
            continue
        mean_q = float(np.mean(quals))
        if mean_q < min_quality:
            continue
        arr = np.array(widths)
        mean_w = float(np.mean(arr))
        if mean_w <= 8.0:
            continue
        spread = 1.4826 * float(np.median(np.abs(arr - np.median(arr)))) / mean_w
        if spread < best_spread:
            best_spread, best = spread, (box, mean_q)
    return best


def measure_noise(
    frames: list[np.ndarray],
    roi: BBox | None = None,
    measured_fps: float = 0.0,
    motion_tolerance_px: float = 1.5,
) -> NoiseReport | None:
    """Characterise width and scale noise from a burst of static-scene frames."""
    if len(frames) < 10:
        return None

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]

    # --- stationarity: phase correlation between consecutive frames -----------------
    shifts = []
    ref_small = None
    for g in grays:
        small = cv2.resize(g, (256, 144)).astype(np.float32)
        if ref_small is not None:
            (dx, dy), _resp = cv2.phaseCorrelate(ref_small, small)
            shifts.append(math.hypot(dx, dy))
        ref_small = small
    # Scale the downsampled displacement back to full-resolution pixels.
    scale_back = grays[0].shape[1] / 256.0
    motion_px = float(np.median(shifts)) * scale_back if shifts else 0.0
    stationary = motion_px <= motion_tolerance_px

    if roi is None:
        found = find_measurable_roi(grays)
        if found is None:
            return None
        roi = found[0]

    # --- width noise -> sigma_w --------------------------------------------------------
    widths, qualities = [], []
    for g in grays:
        out = refine_plate_width(g, roi, pad_frac=0.25)
        if out is None:
            continue
        widths.append(out[0])
        qualities.append(out[2])

    if len(widths) < 5:
        return NoiseReport(
            n_frames=len(frames), stationary=stationary, motion_px=motion_px, roi=roi,
            n_width=len(widths), mean_w_px=float("nan"), sigma_w_px=float("nan"),
            mean_quality=float(np.mean(qualities)) if qualities else 0.0,
            stable_frac=0.0,
            n_scale=0, sigma_s=float("nan"), scale_bias=float("nan"),
            measured_fps=measured_fps,
        )

    w_arr = np.array(widths)
    # Robust sigma via MAD: a single mis-latched frame should not set the noise floor.
    mad = float(np.median(np.abs(w_arr - np.median(w_arr))))
    sigma_w = 1.4826 * mad
    # If the fitter is latching onto different feature pairs, the distribution is
    # multimodal rather than noisy. A single sigma would hide that; this exposes it.
    med = float(np.median(w_arr))
    stable_frac = float(np.mean(np.abs(w_arr - med) <= max(2.0 * sigma_w, 1.0)))

    # --- scale noise -> sigma_s ---------------------------------------------------------
    x, y, bw, bh = roi
    center = (x + bw / 2.0, y + bh / 2.0)
    est = ScaleEstimator()
    est.anchor(grays[0], center, float(bw), 0.0)
    scales = []
    fps = max(measured_fps, 1.0)
    for i, g in enumerate(grays[1:], start=1):
        t = i / fps
        m = est.measure(g, center, t)
        if m is not None and math.isfinite(m.s):
            scales.append(m.s)
        # Re-anchor on the pipeline's own policy. Measuring over one ever-growing
        # baseline would characterise a regime the pipeline never actually uses, and
        # would accumulate drift the real keyframe policy discards.
        if est.should_reanchor(m, t):
            est.anchor(g, center, float(bw), t)

    if len(scales) >= 5:
        s_arr = np.array(scales)
        s_mad = float(np.median(np.abs(s_arr - np.median(s_arr))))
        sigma_s = 1.4826 * s_mad
        scale_bias = float(np.mean(s_arr) - 1.0)
    else:
        sigma_s, scale_bias = float("nan"), float("nan")

    return NoiseReport(
        n_frames=len(frames), stationary=stationary, motion_px=motion_px, roi=roi,
        n_width=len(widths), mean_w_px=float(np.mean(w_arr)), sigma_w_px=sigma_w,
        mean_quality=float(np.mean(qualities)), stable_frac=stable_frac,
        n_scale=len(scales), sigma_s=sigma_s, scale_bias=scale_bias,
        measured_fps=measured_fps,
    )
