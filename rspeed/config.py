"""Tunables for the whole pipeline, in one place.

Defaults are reasonable starting points, not calibrated truth. The three numbers that
actually matter -- sigma_w_plate_px, sigma_w_bbox_px and sigma_s_base -- should be
measured for your specific camera by the procedure in MATH.md section 10, not guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import PLATE_WIDTHS_M, focal_length_px, kappa


@dataclass
class Config:
    # -- camera -------------------------------------------------------------------
    image_width: int = 1920
    image_height: int = 1080
    hfov_deg: float = 60.0
    f_px: float | None = None  # set to override the HFOV-derived value (use fx from
    #                            cv2.calibrateCamera when you have a checkerboard)
    fps: float = 30.0

    # -- plate --------------------------------------------------------------------
    plate_region: str = "eu"  # key into PLATE_WIDTHS_M
    plate_aspect_tol: float = 0.35  # gate against the fronto-parallel assumption
    plate_min_quality: float = 0.15
    # Below this the edge fit is not a measurement. At 7-14 px the classical locator
    # does not return noisy widths, it latches onto a character block and returns the
    # SAME wrong width frame after frame (measured: 10.4 px x5, then 7.5 px x7, for a
    # true 14 px) -- which the filter cannot tell from a real reading.
    plate_min_width_px: float = 18.0

    # -- known-feature overrides (manual/live mode) --------------------------------
    # Lets any object of known width stand in for the plate as the absolute anchor.
    # feature_aspect = None disables the fronto-parallel aspect gate, which only makes
    # sense when you know the nominal shape -- so leave it None for arbitrary objects.
    feature_width_m: float | None = None
    feature_aspect: float | None = None

    # -- measurement noise (MEASURE THESE, see MATH.md section 10) ------------------
    sigma_w_plate_px: float = 0.35  # subpixel edge fit on a plate
    sigma_w_bbox_px: float = 2.0  # detector box-regression jitter
    sigma_s_base: float = 0.004  # ECC scale ratio, matches tools/ measurement

    # -- filter -------------------------------------------------------------------
    q: float = 0.008  # process noise spectral density, MATH.md (7.4)
    gate_nis: float = 9.0  # chi-square 99.7%, 1 DOF
    prior_Z_m: float = 30.0  # seed depth before any calibration exists
    prior_var_lam: float = 4.0  # deliberately huge: ~e^2 factor of depth uncertainty
    # How long Z/Zdot stay reportable after the last accepted absolute measurement.
    # Between anchors the filter carries lam forward on Channel B's lamdot, so a brief
    # plate dropout does not make the readout flicker; sigma_Z grows honestly meanwhile.
    anchor_hold_s: float = 1.0
    # Gate lock-out recovery, MATH.md 7.5: after this many consecutive rejected Channel A
    # measurements that agree with each other, believe them over the filter. A safety
    # net -- with plate_min_width_px in force it never fired across the synthetic grid;
    # without it, 12 turned a permanent 5.6 m lock-out into 0.28 m. Shorter runs (8)
    # also fired on the locator's own latches and made braking worse.
    reopen_after: int = 12

    # -- scale ratio / keyframing (MATH.md 5.3) -----------------------------------
    scale_canonical: int = 96
    scale_roi_factor: float = 1.4  # multiple of the vehicle box width; 1.4 leaves
    #                                headroom for the 25% re-anchor threshold
    scale_max_age_s: float = 0.5
    scale_max_log: float = math.log(1.25)
    scale_min_confidence: float = 0.55

    # -- kappa transfer (MATH.md section 8) ---------------------------------------
    kappa_window: int = 45
    kappa_min_samples: int = 12
    kappa_lock_mad: float = 0.035

    # -- scheduling / efficiency (MATH.md section 9) ------------------------------
    detect_stride: int = 3  # run the vehicle detector every Nth frame
    plate_stride_unlocked: int = 1  # every frame while still calibrating
    plate_refresh_s: float = 2.0  # occasional re-check once locked
    track_max_misses: int = 8
    track_iou_threshold: float = 0.3
    # Only this many tracks get the full estimator each frame, ranked by how plausibly
    # each is the vehicle directly ahead. Each costs an ECC registration and a plate
    # search; a busy scene would otherwise scale the frame cost with the traffic.
    max_tracks: int = 3
    # A coasting track retires after this many frames registration cannot place it.
    track_lost_frames: int = 15
    # One failed registration is routine (it re-anchors next frame); only this many in
    # a row is reported as a lost target.
    lost_grace_frames: int = 3
    # The lead vehicle changes only when a rival scores this much better, so one noisy
    # frame -- or a single spurious detection -- cannot reset the readout.
    lead_switch_margin: float = 0.1

    # -- derived ------------------------------------------------------------------

    @property
    def focal_px(self) -> float:
        if self.f_px is not None:
            return float(self.f_px)
        return focal_length_px(self.image_width, self.hfov_deg)

    @property
    def plate_width_m(self) -> float:
        if self.feature_width_m is not None:
            return float(self.feature_width_m)
        return PLATE_WIDTHS_M[self.plate_region]

    @property
    def kappa_plate(self) -> float:
        """ln(f * W_plate) -- the only form in which f and W enter the filter."""
        return kappa(self.focal_px, self.plate_width_m)

    @property
    def dt(self) -> float:
        return 1.0 / self.fps
