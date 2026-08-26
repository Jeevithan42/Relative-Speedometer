"""Channel B -- scale ratio by direct image registration.  MATH.md section 5.

The central idea (MATH.md 5.2): do NOT compute s = w_now/w_ref from two independent
width detections, because that inherits both detections' noise. Register the pixels
instead and read the scale factor straight out of the recovered warp.

Two backends:
  * ECC  (cv2.findTransformECC, MOTION_AFFINE) -- iterative, illumination-invariant,
    most accurate on a textured patch. Default.
  * Log-polar phase correlation (Fourier-Mellin) -- scale becomes a translation in
    log-polar space, recovered in one shot. Faster, no iteration, used as fallback
    when ECC fails to converge.

Keyframe policy (MATH.md 5.3): register against a held reference frame rather than
the previous frame, so registration error does not accumulate. The ROI pixel size is
frozen for the lifetime of a keyframe, which means the measured canonical scale IS
the true scale -- no bounding-box jitter leaks into Channel B.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Empirically verified sign convention for the ECC affine warp, see
# tools/verify_scale_convention.py. cv2.findTransformECC(template, input, ...) returns
# a warp mapping template coordinates -> input coordinates (it is applied with
# WARP_INVERSE_MAP). With template = current frame and input = reference frame, a
# target that has GROWN means the reference must be sampled with a shrinking map, so
# det(A) < 1 and the true scale is 1/sqrt(det(A)).
_ECC_SCALE_IS_INVERSE = True


@dataclass
class ScaleMeasurement:
    """One Channel B observation."""

    s: float  # scale ratio w_now / w_ref
    dtau: float  # baseline in seconds
    confidence: float  # registration quality in [0, 1]
    sigma_s: float  # 1-sigma uncertainty on s
    backend: str


@dataclass
class _Keyframe:
    patch: np.ndarray  # canonical-size float32 grayscale
    t: float
    roi_size: int  # frozen for the keyframe's lifetime -- see module docstring
    center: tuple[float, float]


class ScaleEstimator:
    """Adaptive-keyframe scale-ratio estimator for one tracked target."""

    def __init__(
        self,
        canonical: int = 96,
        roi_factor: float = 2.2,
        roi_min: int = 48,
        roi_max: int = 256,
        max_age_s: float = 0.5,
        max_abs_log_scale: float = math.log(1.25),
        min_confidence: float = 0.55,
        sigma_s_base: float = 0.004,
        ecc_iterations: int = 40,
        ecc_eps: float = 1e-5,
    ):
        """
        Args:
            canonical: side length the ROI is resampled to before registration.
            roi_factor: ROI side as a multiple of the tracked feature width. 2.2 leaves
                enough headroom that the target cannot escape the window before the
                25% re-anchor threshold fires.
            max_age_s, max_abs_log_scale, min_confidence: the three re-anchor triggers
                from MATH.md 5.3.
            sigma_s_base: baseline 1-sigma on the recovered scale ratio. Measure this
                for your camera per MATH.md section 10 rather than trusting the default.
        """
        self.canonical = int(canonical)
        self.roi_factor = float(roi_factor)
        self.roi_min = int(roi_min)
        self.roi_max = int(roi_max)
        self.max_age_s = float(max_age_s)
        self.max_abs_log_scale = float(max_abs_log_scale)
        self.min_confidence = float(min_confidence)
        self.sigma_s_base = float(sigma_s_base)
        self._criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(ecc_iterations),
            float(ecc_eps),
        )
        self._kf: _Keyframe | None = None
        self.reanchor_count = 0

    # -- public API -------------------------------------------------------------------

    @property
    def has_keyframe(self) -> bool:
        return self._kf is not None

    def anchor(
        self,
        gray: np.ndarray,
        center: tuple[float, float],
        feature_w_px: float,
        t: float,
    ) -> None:
        """Drop a new reference keyframe at the current frame."""
        roi = int(np.clip(round(self.roi_factor * feature_w_px), self.roi_min, self.roi_max))
        patch = self._extract(gray, center, roi)
        self._kf = _Keyframe(patch=patch, t=t, roi_size=roi, center=center)
        self.reanchor_count += 1

    def measure(
        self, gray: np.ndarray, center: tuple[float, float], t: float
    ) -> ScaleMeasurement | None:
        """Register the current frame against the keyframe and return the scale ratio.

        Returns None when there is no keyframe, the baseline is degenerate, or
        registration failed. Callers should re-anchor on None.
        """
        kf = self._kf
        if kf is None:
            return None
        dtau = t - kf.t
        if dtau <= 1e-6:
            return None

        # ROI size is frozen with the keyframe, so the canonical-space scale is the
        # true scale -- no resample-factor correction needed, and no bbox jitter.
        cur = self._extract(gray, center, kf.roi_size)

        result = self._register_ecc(kf.patch, cur)
        if result is None:
            result = self._register_logpolar(kf.patch, cur)
            backend = "logpolar"
        else:
            backend = "ecc"
        if result is None:
            return None

        s, confidence = result
        if not math.isfinite(s) or s <= 0.0:
            return None

        # Heuristic: degrade sigma as registration confidence falls. Replace with an
        # empirically measured sigma_s(confidence) curve for production use.
        sigma_s = self.sigma_s_base * (1.0 + 4.0 * max(0.0, 1.0 - confidence))

        return ScaleMeasurement(
            s=s, dtau=dtau, confidence=confidence, sigma_s=sigma_s, backend=backend
        )

    def should_reanchor(self, m: ScaleMeasurement | None, t: float) -> bool:
        """The three re-anchor triggers of MATH.md 5.3."""
        if self._kf is None:
            return True
        if m is None:
            return True
        if m.confidence < self.min_confidence:
            return True
        if abs(math.log(m.s)) > self.max_abs_log_scale:
            return True
        if (t - self._kf.t) > self.max_age_s:
            return True
        return False

    # -- registration backends ---------------------------------------------------------

    def _register_ecc(
        self, ref: np.ndarray, cur: np.ndarray
    ) -> tuple[float, float] | None:
        """Affine ECC registration. Returns (scale, correlation) or None on failure."""
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(
                cur,  # templateImage
                ref,  # inputImage
                warp,
                cv2.MOTION_AFFINE,
                self._criteria,
                None,
                5,
            )
        except cv2.error:
            return None

        A = warp[:2, :2].astype(float)
        det = float(np.linalg.det(A))
        if det <= 1e-9:
            return None

        # sqrt(det) is the isotropic scale of the affine part; taking the determinant
        # rather than a single axis makes this robust to the mild shear/rotation the
        # affine model absorbs from perspective change.
        s_warp = math.sqrt(det)
        s = (1.0 / s_warp) if _ECC_SCALE_IS_INVERSE else s_warp
        return s, float(np.clip(cc, 0.0, 1.0))

    def _register_logpolar(
        self, ref: np.ndarray, cur: np.ndarray
    ) -> tuple[float, float] | None:
        """Fourier-Mellin fallback: scale -> translation in log-polar space.

        With cv2.warpPolar(..., WARP_POLAR_LOG) the radial axis maps r -> M*ln(r) with
        M = dsize.width / ln(maxRadius). A uniform scaling by s therefore shifts the
        polar image by M*ln(s) along x, which phase correlation recovers directly.
        """
        n = self.canonical
        center = (n / 2.0, n / 2.0)
        max_radius = n / 2.0
        flags = cv2.INTER_LINEAR | cv2.WARP_POLAR_LOG

        try:
            pol_ref = cv2.warpPolar(ref, (n, n), center, max_radius, flags)
            pol_cur = cv2.warpPolar(cur, (n, n), center, max_radius, flags)
        except cv2.error:
            return None

        win = cv2.createHanningWindow((n, n), cv2.CV_32F)
        (shift_x, _shift_y), response = cv2.phaseCorrelate(
            pol_ref.astype(np.float32), pol_cur.astype(np.float32), win
        )

        M = n / math.log(max_radius)
        s = math.exp(shift_x / M)
        return s, float(np.clip(response, 0.0, 1.0))

    # -- helpers -----------------------------------------------------------------------

    def _extract(
        self, gray: np.ndarray, center: tuple[float, float], roi: int
    ) -> np.ndarray:
        """Crop a square ROI (replicate-padded at the borders) and resample to canonical.

        Returns float32 in [0, 1]; ECC wants float input and is happier with a
        normalised range.
        """
        h, w = gray.shape[:2]
        cx, cy = center
        # At close range the vehicle box overflows the frame and its rear-centre can
        # land outside the image entirely, leaving an empty intersection that
        # copyMakeBorder rejects. Clamp the centre so the window always overlaps.
        cx = float(np.clip(cx, 0.0, w - 1.0))
        cy = float(np.clip(cy, 0.0, h - 1.0))
        roi = int(max(8, min(roi, 4 * max(w, h))))
        half = roi / 2.0
        x0, y0 = int(round(cx - half)), int(round(cy - half))
        x1, y1 = x0 + roi, y0 + roi

        pad_l, pad_t = max(0, -x0), max(0, -y0)
        pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(w, x1), min(h, y1)

        crop = gray[ys0:ys1, xs0:xs1]
        if crop.size == 0:
            # Degenerate window: fall back to the whole frame rather than throwing.
            crop, pad_l, pad_t, pad_r, pad_b = gray, 0, 0, 0, 0
        if pad_l or pad_t or pad_r or pad_b:
            crop = cv2.copyMakeBorder(
                crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
            )
        if crop.shape[0] != roi or crop.shape[1] != roi:
            crop = cv2.resize(crop, (roi, roi), interpolation=cv2.INTER_LINEAR)

        patch = cv2.resize(
            crop, (self.canonical, self.canonical), interpolation=cv2.INTER_AREA
        )
        patch = patch.astype(np.float32)
        if patch.max() > 1.5:
            patch /= 255.0
        # Zero-mean/unit-variance helps ECC converge from the identity initialisation.
        mean, std = float(patch.mean()), float(patch.std())
        if std > 1e-6:
            patch = (patch - mean) / std
        return np.ascontiguousarray(patch, dtype=np.float32)
