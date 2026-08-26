"""Pinhole geometry and the log-inverse-depth change of variable.

All of this is derived in MATH.md sections 1, 2 and 4. Symbol names here match the
"Code name" column of the notation tables in that document.

Sign convention used everywhere in this project:

    Zdot  < 0   closing   (gap shrinking)
    lamdot > 0  closing
    TTC   > 0   closing, seconds until contact
"""

from __future__ import annotations

import math

# Legally standardised licence plate widths, metres. MATH.md section 2.
PLATE_WIDTHS_M = {
    "eu": 0.520,  # EU / UK / India single-line, 520 x 110 mm
    "us": 0.305,  # USA / Canada, 12 x 6 in
    "jp": 0.330,  # Japan standard, 330 x 165 mm
    "au": 0.372,  # Australia standard, 372 x 134 mm
}

# Nominal width/height aspect of the same plates, used to reject skewed detections.
PLATE_ASPECTS = {"eu": 520 / 110, "us": 12 / 6, "jp": 330 / 165, "au": 372 / 134}


def focal_length_px(image_width: int, hfov_deg: float) -> float:
    """Focal length in pixels from image width and horizontal field of view.

    MATH.md (1.2):  f = (W_img/2) / tan(HFOV/2)

    Prefer a checkerboard calibration (cv2.calibrateCamera -> fx) when you can get
    one; this is the estimate to use when all you have is a spec sheet.
    """
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError(f"hfov_deg must be in (0, 180), got {hfov_deg}")
    return (image_width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def focal_length_px_from_mm(f_mm: float, image_width: int, sensor_width_mm: float) -> float:
    """Focal length in pixels from a physical lens focal length."""
    return f_mm * (image_width / sensor_width_mm)


def kappa(f_px: float, width_m: float) -> float:
    """Log scale constant of a feature.  MATH.md (4.2):  kappa = ln(f * W)

    This is the only form in which f and W ever enter the filter -- they appear
    exclusively as a product, so they never need to be separated (MATH.md section 8).
    """
    if f_px <= 0 or width_m <= 0:
        raise ValueError("f_px and width_m must be positive")
    return math.log(f_px * width_m)


# --- Direct pinhole relations, MATH.md (1.1) / (2.1) --------------------------------

def depth_from_width(w_px: float, f_px: float, width_m: float) -> float:
    """Z = f * W / w   -- known-size ranging."""
    if w_px <= 0:
        raise ValueError("w_px must be positive")
    return f_px * width_m / w_px


def width_from_depth(Z: float, f_px: float, width_m: float) -> float:
    """w = f * W / Z   -- the forward projection, used by the synthetic renderer."""
    if Z <= 0:
        raise ValueError("Z must be positive")
    return f_px * width_m / Z


# --- The log-inverse-depth substitution, MATH.md section 4 --------------------------

def Z_to_lam(Z: float) -> float:
    """lam = -ln(Z)   MATH.md (4.1)"""
    if Z <= 0:
        raise ValueError("Z must be positive")
    return -math.log(Z)


def lam_to_Z(lam: float) -> float:
    """Z = exp(-lam)   MATH.md (4.5)"""
    return math.exp(-lam)


def lamdot_from_motion(Z: float, Zdot: float) -> float:
    """lamdot = -Zdot / Z   MATH.md (4.3).  Positive when closing."""
    return -Zdot / Z


def speed_from_state(lam: float, lamdot: float) -> float:
    """Zdot = -lamdot * exp(-lam)   MATH.md (4.6).  Negative when closing."""
    return -lamdot * math.exp(-lam)


def ttc_from_lamdot(lamdot: float) -> float:
    """TTC = 1 / lamdot   MATH.md (4.7).

    Returns +inf when the target is not closing (lamdot <= 0), which is the
    physically correct answer rather than a negative "time".
    """
    if lamdot <= 0.0:
        return math.inf
    return 1.0 / lamdot


# --- Channel B closed forms, MATH.md section 5.1 ------------------------------------

def lamdot_from_scale(s: float, dtau: float) -> float:
    """lamdot at the *current* frame from a scale ratio.  MATH.md (5.4)

        lamdot = (s - 1) / dtau

    Exact under constant relative velocity -- not a small-dtau approximation.
    Requires no camera calibration and no knowledge of the target's real size.
    """
    if dtau <= 0:
        raise ValueError("dtau must be positive")
    return (s - 1.0) / dtau


def ttc_from_scale(s: float, dtau: float) -> float:
    """TTC = dtau / (s - 1)   MATH.md (5.5).  Calibration-free time to contact."""
    return ttc_from_lamdot(lamdot_from_scale(s, dtau))


# --- Error budget helper, MATH.md section 3 -----------------------------------------

def speed_noise_from_width_noise(
    sigma_w_px: float, w_px: float, Z: float, n_frames: int, dt: float
) -> float:
    """Predicted sigma_v for naive OLS slope differentiation.  MATH.md (3.1) + (3.2).

    Provided so the error budget in the docs can be reproduced (and checked) in code
    rather than trusted. This is *not* what the pipeline uses -- the EKF does better
    -- but it is the right back-of-envelope for "will this camera work at all".
    """
    if n_frames < 2:
        raise ValueError("n_frames must be >= 2")
    denom = dt * math.sqrt(n_frames * (n_frames**2 - 1) / 12.0)
    sigma_slope = sigma_w_px / denom
    return (Z / w_px) * sigma_slope
