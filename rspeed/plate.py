"""Licence plate localisation and subpixel width measurement.

The plate is the calibration anchor: it is the one feature on a vehicle whose real
width is known a priori (MATH.md section 2), so it is what turns a calibration-free
TTC into an absolute speed in m/s.

MATH.md section 3 shows why subpixel accuracy dominates the error budget -- 0.3 px vs
2 px of width noise is a ~7x difference in the final speed uncertainty. So the plate
box from any detector (classical or CNN) is only a *bracket*; the width that actually
feeds the filter comes from `refine_plate_width`, which fits the intensity gradient.

`ClassicalPlateLocator` needs no model weights and no torch, which is what makes this
runnable today. Swap in `YoloPlateLocator` when you have a trained plate detector --
both satisfy the same protocol and the pipeline does not care which it holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .geometry import PLATE_ASPECTS

BBox = tuple[int, int, int, int]  # x, y, w, h


@dataclass
class PlateObservation:
    """One plate measurement, with the subpixel width the filter consumes."""

    w_px: float  # subpixel plate width -- the measurement that matters
    center: tuple[float, float]  # subpixel centre in full-frame coordinates
    bbox: BBox  # integer bracket, full-frame coordinates
    aspect: float  # measured width/height, for gating against nominal
    quality: float  # [0, 1] edge-fit confidence


class PlateLocator(Protocol):
    """Anything that can find a plate inside a vehicle box."""

    def locate(self, gray: np.ndarray, vehicle_bbox: BBox) -> PlateObservation | None: ...


# --- subpixel edge refinement --------------------------------------------------------


def _parabolic_peak(y_prev: float, y_mid: float, y_next: float) -> float:
    """Sub-sample peak offset in [-0.5, 0.5] from a 3-point parabola fit.

    offset = 0.5 * (y_prev - y_next) / (y_prev - 2*y_mid + y_next)

    This is the standard interpolator that buys ~0.1 px accuracy from an integer
    argmax, and it is the single cheapest accuracy win available (MATH.md section 3).
    """
    denom = y_prev - 2.0 * y_mid + y_next
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y_prev - y_next) / denom, -0.5, 0.5))


def refine_plate_width(
    gray: np.ndarray, bbox: BBox, pad_frac: float = 0.25
) -> tuple[float, float, float] | None:
    """Measure the plate's horizontal extent to subpixel accuracy.

    Collapses the plate's central horizontal band into a 1-D intensity profile, then
    locates the two strongest opposing gradients and refines each with a parabola fit.

    Returns (w_px, center_x_in_frame, quality) or None if the profile is unusable.
    """
    x, y, w, h = bbox
    if w < 8 or h < 4:
        return None

    pad = int(round(w * pad_frac))
    x0 = max(0, x - pad)
    x1 = min(gray.shape[1], x + w + pad)
    # Use the central 60% of the plate height: avoids the top/bottom frame edges and
    # any mounting screws or slot lighting bleeding into the profile.
    y0 = max(0, y + int(round(0.20 * h)))
    y1 = min(gray.shape[0], y + int(round(0.80 * h)))
    if x1 - x0 < 8 or y1 - y0 < 2:
        return None

    band = gray[y0:y1, x0:x1].astype(np.float32)
    profile = band.mean(axis=0)
    if profile.size < 8:
        return None

    profile = cv2.GaussianBlur(profile.reshape(1, -1), (5, 1), 0).ravel()
    g = np.gradient(profile)

    n = profile.size
    # The plate occupies the middle; padding on both sides is the surrounding bodywork.
    interior = profile[pad : n - pad] if n > 2 * pad + 2 else profile
    exterior = np.concatenate([profile[:pad], profile[n - pad :]]) if pad > 0 else profile
    bright_plate = float(interior.mean()) >= float(exterior.mean())

    mid = n // 2
    left_g = g[:mid]
    right_g = g[mid:]
    if left_g.size < 3 or right_g.size < 3:
        return None

    # A bright plate on darker bodywork rises at its left edge and falls at its right;
    # a dark plate does the opposite.
    if bright_plate:
        il = int(np.argmax(left_g))
        ir = mid + int(np.argmin(right_g))
        strength = float(left_g[il]) - float(g[ir])
    else:
        il = int(np.argmin(left_g))
        ir = mid + int(np.argmax(right_g))
        strength = float(g[ir]) - float(left_g[il])

    if il <= 0 or il >= n - 1 or ir <= 0 or ir >= n - 1:
        return None

    mag = np.abs(g)
    xl = il + _parabolic_peak(float(mag[il - 1]), float(mag[il]), float(mag[il + 1]))
    xr = ir + _parabolic_peak(float(mag[ir - 1]), float(mag[ir]), float(mag[ir + 1]))

    w_px = float(xr - xl)
    if w_px <= 4.0:
        return None

    center_x = x0 + 0.5 * (xl + xr)
    # Normalise edge strength into a rough [0, 1] quality. 40 grey levels of combined
    # edge contrast is a solidly readable plate.
    quality = float(np.clip(strength / 40.0, 0.0, 1.0))
    return w_px, center_x, quality


# --- classical locator ---------------------------------------------------------------


class ClassicalPlateLocator:
    """Model-free plate localisation: gradient density + morphology + shape gating.

    Fast and dependency-light. Works well on clean, well-lit, roughly fronto-parallel
    plates within ~25 m, which is exactly the regime where the kappa transfer
    (MATH.md section 8) needs it. Beyond that, use a trained detector.
    """

    def __init__(
        self,
        region: str = "eu",
        aspect_tol: float = 0.35,
        min_width_px: int = 16,
        search_top_frac: float = 0.35,
        min_quality: float = 0.15,
    ):
        """
        Args:
            region: key into PLATE_ASPECTS / PLATE_WIDTHS_M.
            aspect_tol: fractional tolerance on the nominal width/height ratio. This is
                the main defence against the fronto-parallel assumption breaking
                (MATH.md section 11) -- a yawed plate foreshortens and fails the gate.
            search_top_frac: only look below this fraction of the vehicle box height.
        """
        if region not in PLATE_ASPECTS:
            raise ValueError(f"unknown region {region!r}; expected one of {list(PLATE_ASPECTS)}")
        self.region = region
        self.nominal_aspect = PLATE_ASPECTS[region]
        self.aspect_tol = float(aspect_tol)
        self.min_width_px = int(min_width_px)
        self.search_top_frac = float(search_top_frac)
        self.min_quality = float(min_quality)

    def locate(self, gray: np.ndarray, vehicle_bbox: BBox) -> PlateObservation | None:
        vx, vy, vw, vh = vehicle_bbox
        if vw < self.min_width_px * 2 or vh < 8:
            return None

        sy0 = vy + int(round(vh * self.search_top_frac))
        sy1 = min(gray.shape[0], vy + vh)
        sx0, sx1 = max(0, vx), min(gray.shape[1], vx + vw)
        if sy1 - sy0 < 8 or sx1 - sx0 < 16:
            return None

        roi = gray[sy0:sy1, sx0:sx1]

        # Character strokes produce dense vertical edges; a wide closing kernel merges
        # them into one plate-shaped blob while leaving isolated body edges thin.
        sobel = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        sobel = np.abs(sobel)
        if sobel.max() < 1e-6:
            return None
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        sobel = cv2.GaussianBlur(sobel, (5, 5), 0)

        kw = max(9, int(round(roi.shape[1] * 0.06)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        closed = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, kernel)
        _, mask = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 5))
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: PlateObservation | None = None
        best_score = -math.inf

        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < self.min_width_px or bh < 4:
                continue
            if bw > 0.95 * roi.shape[1]:
                continue
            aspect = bw / float(bh)
            if abs(aspect / self.nominal_aspect - 1.0) > self.aspect_tol:
                continue

            frame_bbox: BBox = (sx0 + bx, sy0 + by, bw, bh)
            refined = refine_plate_width(gray, frame_bbox)
            if refined is None:
                continue
            w_px, center_x, quality = refined
            if quality < self.min_quality:
                continue

            # Prefer strong edges, plausible aspect, and lower-and-central placement --
            # that is where a rear plate actually sits.
            aspect_err = abs(aspect / self.nominal_aspect - 1.0)
            vertical_bonus = (by + bh / 2.0) / max(1.0, roi.shape[0])
            horiz_err = abs((bx + bw / 2.0) / max(1.0, roi.shape[1]) - 0.5)
            score = quality * 2.0 - aspect_err * 1.5 + vertical_bonus * 0.5 - horiz_err

            if score > best_score:
                best_score = score
                center_y = sy0 + by + bh / 2.0
                best = PlateObservation(
                    w_px=w_px,
                    center=(center_x, center_y),
                    bbox=frame_bbox,
                    aspect=aspect,
                    quality=quality,
                )

        return best


class YoloPlateLocator:
    """Trained-detector plate locator. Requires ultralytics + a plate model.

    The CNN supplies only the bracket; the subpixel width still comes from
    `refine_plate_width`, because a detector's box regression is nowhere near
    0.3 px accurate (MATH.md section 3).
    """

    def __init__(self, weights: str, region: str = "eu", conf: float = 0.25,
                 aspect_tol: float = 0.35, min_quality: float = 0.1):
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "YoloPlateLocator needs ultralytics: pip install ultralytics"
            ) from exc
        self._model = YOLO(weights)
        self.region = region
        self.nominal_aspect = PLATE_ASPECTS[region]
        self.conf = float(conf)
        self.aspect_tol = float(aspect_tol)
        self.min_quality = float(min_quality)

    def locate(self, gray: np.ndarray, vehicle_bbox: BBox) -> PlateObservation | None:
        vx, vy, vw, vh = vehicle_bbox
        crop = gray[vy : vy + vh, vx : vx + vw]
        if crop.size == 0:
            return None
        bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        res = self._model.predict(bgr, conf=self.conf, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return None

        boxes = res[0].boxes.xyxy.cpu().numpy()
        confs = res[0].boxes.conf.cpu().numpy()
        order = np.argsort(-confs)

        for i in order:
            x1, y1, x2, y2 = boxes[i]
            bw, bh = float(x2 - x1), float(y2 - y1)
            if bw < 8 or bh < 3:
                continue
            aspect = bw / bh
            if abs(aspect / self.nominal_aspect - 1.0) > self.aspect_tol:
                continue
            frame_bbox: BBox = (vx + int(x1), vy + int(y1), int(round(bw)), int(round(bh)))
            refined = refine_plate_width(gray, frame_bbox)
            if refined is None:
                continue
            w_px, center_x, quality = refined
            if quality < self.min_quality:
                continue
            return PlateObservation(
                w_px=w_px,
                center=(center_x, vy + (y1 + y2) / 2.0),
                bbox=frame_bbox,
                aspect=aspect,
                quality=quality,
            )
        return None


class KnownObjectLocator:
    """Treats the tracked ROI itself as the known-width feature.

    For live testing without a vehicle detector: point the camera at any object whose
    real width you can measure (a credit card, a sheet of A4, a book), and this
    supplies Channel A's absolute anchor by refining that object's edges to subpixel
    accuracy -- the identical measurement path a plate takes.

    It relies on the object having two strong opposing vertical edges against its
    background, which is exactly what `refine_plate_width` looks for.
    """

    def __init__(self, min_quality: float = 0.05, pad_frac: float = 0.3):
        self.min_quality = float(min_quality)
        self.pad_frac = float(pad_frac)

    def locate(self, gray: np.ndarray, vehicle_bbox: BBox) -> PlateObservation | None:
        x, y, w, h = vehicle_bbox
        refined = refine_plate_width(gray, vehicle_bbox, pad_frac=self.pad_frac)
        if refined is None:
            return None
        w_px, center_x, quality = refined
        if quality < self.min_quality:
            return None
        return PlateObservation(
            w_px=w_px,
            center=(center_x, y + h / 2.0),
            bbox=vehicle_bbox,
            aspect=w / max(1.0, float(h)),
            quality=quality,
        )
