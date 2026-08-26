"""Per-vehicle estimator -- where the two channels actually get fused.

One `VehicleEstimator` owns everything specific to a single tracked vehicle:

    EKF             the shared state [lam, lamdot]          (MATH.md section 7)
    KappaEstimator  the plate -> large-feature transfer     (MATH.md section 8)
    ScaleEstimator  the keyframe and registration backend   (MATH.md section 5)

The per-frame order is: predict, then Channel A (absolute, if available), then
Channel B (scale ratio, always). Both are gated scalar updates on the same filter,
so "using both" costs one extra 2x2 update -- it is not a second pipeline.

Channel B needs no calibration, so TTC is available from the very first keyframe,
before any plate has ever been seen. Absolute Z and Zdot appear only once Channel A
has an anchor. `SpeedEstimate.calibrated` tells the caller which regime it is in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .calib import KappaEstimator
from .config import Config
from .detector import Track
from .filter import LogDepthEKF
from .geometry import PLATE_ASPECTS
from .plate import PlateLocator, PlateObservation
from .scale import ScaleEstimator, ScaleMeasurement


@dataclass
class SpeedEstimate:
    """One frame's output for one vehicle."""

    track_id: int
    bbox: tuple[int, int, int, int]

    ttc: float  # seconds, +inf if not closing. Valid WITHOUT calibration.
    lam: float
    lamdot: float  # = 1/TTC, the fractional expansion rate

    calibrated: bool  # is Z/Zdot trustworthy yet?
    Z: float | None  # metres
    Zdot: float | None  # m/s, negative = closing
    sigma_Z: float | None
    sigma_Zdot: float | None

    channel_a: str  # 'plate' | 'big' | 'none' -- which anchor was used this frame
    channel_b: bool  # did a scale measurement land this frame?
    kappa_locked: bool
    kappa_samples: int
    plate: PlateObservation | None
    scale: ScaleMeasurement | None
    plate_ran: bool  # was the plate detector invoked (cost accounting)

    @property
    def Zdot_kmh(self) -> float | None:
        return None if self.Zdot is None else self.Zdot * 3.6


class VehicleEstimator:
    """Fuses Channel A and Channel B for one tracked vehicle."""

    def __init__(self, track_id: int, cfg: Config, plate_locator: PlateLocator | None):
        self.track_id = track_id
        self.cfg = cfg
        self.plate_locator = plate_locator

        self.ekf = LogDepthEKF(q=cfg.q, gate_nis=cfg.gate_nis)
        self.kappa = KappaEstimator(
            window=cfg.kappa_window,
            min_samples=cfg.kappa_min_samples,
            lock_mad=cfg.kappa_lock_mad,
        )
        self.scale = ScaleEstimator(
            canonical=cfg.scale_canonical,
            roi_factor=cfg.scale_roi_factor,
            max_age_s=cfg.scale_max_age_s,
            max_abs_log_scale=cfg.scale_max_log,
            min_confidence=cfg.scale_min_confidence,
            sigma_s_base=cfg.sigma_s_base,
        )

        self._t_last: float | None = None
        self._t_last_plate: float = -math.inf
        self._frames = 0
        self._nis_log: list[float] = []

    # -- scheduling, MATH.md section 9 -------------------------------------------------

    def _should_run_plate(self, t: float) -> bool:
        """Plate detection is the expensive optional stage; run it only when it pays.

        While uncalibrated it is the only route to absolute scale, so run it hard.
        Once kappa is locked the large feature carries Channel A, and the plate is
        needed only to catch slow drift -- so drop to an occasional refresh. This is
        what makes the fused pipeline *cheaper* in steady state than a plate-only one.
        """
        if self.plate_locator is None:
            return False
        if not self.kappa.locked:
            return self._frames % max(1, self.cfg.plate_stride_unlocked) == 0
        return (t - self._t_last_plate) >= self.cfg.plate_refresh_s

    def _plate_gate_ok(self, obs: PlateObservation) -> bool:
        """Reject foreshortened plates -- the fronto-parallel assumption of MATH.md
        (1.1) is the one that silently biases Z long when it breaks."""
        nominal = self.cfg.feature_aspect
        if nominal is None and self.cfg.feature_width_m is None:
            nominal = PLATE_ASPECTS[self.cfg.plate_region]
        # nominal stays None for an arbitrary known-width object: there is no expected
        # shape to gate on, so quality is the only filter available.
        if nominal is not None:
            if abs(obs.aspect / nominal - 1.0) > self.cfg.plate_aspect_tol:
                return False
        return obs.quality >= self.cfg.plate_min_quality

    # -- main entry point ---------------------------------------------------------------

    def update(self, gray: np.ndarray, track: Track, t: float) -> SpeedEstimate:
        cfg = self.cfg
        dt = 0.0 if self._t_last is None else max(0.0, t - self._t_last)
        self._t_last = t
        self._frames += 1

        w_big = float(track.width)
        center = track.rear_center

        # ---- 1. predict --------------------------------------------------------------
        if self.ekf.initialised and dt > 0:
            self.ekf.predict(dt)

        # ---- 2. plate (optional, scheduled) ------------------------------------------
        plate: PlateObservation | None = None
        plate_ran = self._should_run_plate(t)
        if plate_ran:
            self._t_last_plate = t
            found = self.plate_locator.locate(gray, track.bbox)
            if found is not None and self._plate_gate_ok(found):
                plate = found
                # MATH.md (8.1): every simultaneous sighting feeds the transfer.
                self.kappa.add_pair(plate.w_px, w_big, cfg.kappa_plate)

        # ---- 3. seed the filter ------------------------------------------------------
        if not self.ekf.initialised:
            if plate is not None:
                # ln(w) = kappa + lam  =>  lam = ln(w) - kappa
                lam0 = math.log(plate.w_px) - cfg.kappa_plate
                self.ekf.initialise(lam0, 0.0, var_lam=0.25, var_lamdot=1.0)
            else:
                # No absolute anchor yet. Seed lam from a nominal depth with a huge
                # variance -- lamdot (hence TTC) is observable regardless, so the
                # filter is still useful while this stays wrong.
                lam0 = -math.log(cfg.prior_Z_m)
                self.ekf.initialise(
                    lam0, 0.0, var_lam=cfg.prior_var_lam, var_lamdot=1.0
                )

        # ---- 4. Channel A: absolute anchor, MATH.md (7.5) ----------------------------
        channel_a = "none"
        if plate is not None:
            # Prefer the plate whenever it is available: known W, subpixel edges, and
            # sigma_w/w is smallest here even though the plate is the smaller feature.
            sigma = cfg.sigma_w_plate_px / max(0.25, plate.quality)
            res = self.ekf.update_absolute(plate.w_px, cfg.kappa_plate, sigma)
            if res.accepted:
                channel_a = "plate"
                self._nis_log.append(res.nis)
        else:
            kappa_big = self.kappa.value
            if kappa_big is not None:
                # The transfer paying off: absolute ranging with no plate in sight.
                res = self.ekf.update_absolute(w_big, kappa_big, cfg.sigma_w_bbox_px)
                if res.accepted:
                    channel_a = "big"
                    self._nis_log.append(res.nis)

        # ---- 5. Channel B: scale ratio, MATH.md (7.6) --------------------------------
        measurement: ScaleMeasurement | None = None
        channel_b = False
        if self.scale.has_keyframe:
            measurement = self.scale.measure(gray, center, t)
            if measurement is not None:
                res = self.ekf.update_scale(
                    measurement.s, measurement.dtau, measurement.sigma_s
                )
                if res.accepted:
                    channel_b = True
                    self._nis_log.append(res.nis)

        if self.scale.should_reanchor(measurement, t):
            self.scale.anchor(gray, center, w_big, t)

        # ---- 6. report ----------------------------------------------------------------
        calibrated = self.kappa.value is not None or channel_a == "plate"
        return SpeedEstimate(
            track_id=self.track_id,
            bbox=track.bbox,
            ttc=self.ekf.ttc,
            lam=self.ekf.lam,
            lamdot=self.ekf.lamdot,
            calibrated=calibrated,
            Z=self.ekf.Z if calibrated else None,
            Zdot=self.ekf.Zdot if calibrated else None,
            sigma_Z=self.ekf.sigma_Z if calibrated else None,
            sigma_Zdot=self.ekf.sigma_Zdot if calibrated else None,
            channel_a=channel_a,
            channel_b=channel_b,
            kappa_locked=self.kappa.locked,
            kappa_samples=self.kappa.n_samples,
            plate=plate,
            scale=measurement,
            plate_ran=plate_ran,
        )

    # -- diagnostics ---------------------------------------------------------------------

    @property
    def mean_nis(self) -> float:
        """Mean normalised innovation squared over the track's life.

        Should sit near 1.0 for a consistent filter. Persistently above means Q or R
        are too small and the filter will lag a brake event; well below means they are
        too large and it is just following noise. MATH.md section 10.
        """
        return float(np.mean(self._nis_log)) if self._nis_log else float("nan")
