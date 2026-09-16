"""Per-vehicle estimator -- where the two channels actually get fused.

One `VehicleEstimator` owns everything specific to a single tracked vehicle:

    EKF             the shared state [lam, lamdot]          (MATH.md section 7)
    KappaEstimator  the plate -> large-feature transfer     (MATH.md section 8)
    ScaleEstimator  the keyframe and registration backend   (MATH.md section 5)

The per-frame order is: predict, register against the keyframe (which also says where
the target now is, MATH.md 5.4), then Channel A (absolute, if available), then Channel B
(scale ratio, always). Both are gated scalar updates on the same filter, so "using
both" costs one extra 2x2 update -- it is not a second pipeline.

A box is either *measured* (a detector or the caller produced it this frame) or
*propagated* (moved here by the registration). Only a measured box may feed Channel A
or the kappa transfer: a propagated width is w_keyframe * s, so treating it as a width
measurement would count Channel B's information twice.

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
from .detector import REAR_ANCHOR_FRAC, BBox, Track
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
    box_measured: bool = True  # False: bbox was propagated by registration this frame
    tracking_ok: bool = True  # False: registration could not place the target

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
        self._t_last_absolute: float = -math.inf
        self._frames = 0
        # Box size and the offset of its rear anchor from the keyframe anchor point,
        # both captured at the keyframe -- what the registration result is applied to.
        self._kf_box: tuple[float, float, float, float] | None = None
        self.lost_frames = 0  # consecutive frames registration could not place the target
        self._rejected_innov: list[float] = []  # consecutive gated Channel A innovations
        self.reopen_count = 0
        self._nis_log: list[float] = []
        # Why Channel A failed, counted. "calibrated 0%" with no reason is a dead end
        # for the user -- these distinguish "found nothing" from "found it and threw it
        # away", which have completely different fixes.
        self.reject: dict[str, int] = {
            "not_found": 0, "too_small": 0, "low_quality": 0, "bad_aspect": 0,
            "gated": 0, "ok": 0,
        }
        self.last_quality: float | None = None

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
                self.reject["bad_aspect"] += 1
                return False
        if obs.w_px < self.cfg.plate_min_width_px:
            self.reject["too_small"] += 1
            return False
        if obs.quality < self.cfg.plate_min_quality:
            self.reject["low_quality"] += 1
            return False
        return True

    # -- main entry point ---------------------------------------------------------------

    def update(
        self, gray: np.ndarray, track: Track, t: float, box_measured: bool = True
    ) -> SpeedEstimate:
        """Process one frame for this vehicle.

        `box_measured` says whether `track.bbox` was produced this frame by a detector
        or the caller. When False the box is stale, and this moves it (in place, on
        `track`) to where Channel B's registration finds the target.
        """
        cfg = self.cfg
        dt = 0.0 if self._t_last is None else max(0.0, t - self._t_last)
        self._t_last = t
        self._frames += 1

        # ---- 1. predict --------------------------------------------------------------
        if self.ekf.initialised and dt > 0:
            self.ekf.predict(dt)

        # ---- 2. register against the keyframe, MATH.md 5.2 / 5.4 ---------------------
        # Done before anything reads the box: the warp says where the target is now, so
        # the plate bracket and the next keyframe both land on the target, not on
        # wherever it was when the detector last ran.
        measurement: ScaleMeasurement | None = None
        if self.scale.has_keyframe:
            measurement = self.scale.measure(gray, track.rear_center, t)
        placed = (
            measurement is not None
            and measurement.center is not None
            and measurement.confidence >= cfg.scale_min_confidence
        )
        if not box_measured and self.scale.has_keyframe:
            if placed:
                track.bbox = self._propagate_box(measurement)
                self.lost_frames = 0
            else:
                self.lost_frames += 1  # hold the box where it was
        else:
            self.lost_frames = 0

        w_big = float(track.width)
        center = track.rear_center

        # ---- 3. plate (optional, scheduled) ------------------------------------------
        plate: PlateObservation | None = None
        plate_ran = self._should_run_plate(t)
        if plate_ran:
            self._t_last_plate = t
            found = self.plate_locator.locate(gray, track.bbox)
            if found is None:
                self.reject["not_found"] += 1
            else:
                self.last_quality = found.quality
                if self._plate_gate_ok(found):
                    self.reject["ok"] += 1
                    plate = found
                    # MATH.md (8.1): every simultaneous sighting feeds the transfer --
                    # but only against a width that was actually measured.
                    if box_measured:
                        self.kappa.add_pair(plate.w_px, w_big, cfg.kappa_plate)

        # ---- 4. seed the filter ------------------------------------------------------
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

        # ---- 5. Channel A: absolute anchor, MATH.md (7.5) ----------------------------
        channel_a = "none"
        if plate is not None:
            # Prefer the plate whenever it is available: known W, subpixel edges, and
            # sigma_w/w is smallest here even though the plate is the smaller feature.
            sigma = cfg.sigma_w_plate_px / max(0.25, plate.quality)
            res = self.ekf.update_absolute(plate.w_px, cfg.kappa_plate, sigma)
            self._watch_lockout(res)
            if res.accepted:
                channel_a = "plate"
                self._nis_log.append(res.nis)
            else:
                self.reject["gated"] += 1
        elif box_measured:
            kappa_big = self.kappa.value
            if kappa_big is not None:
                # The transfer paying off: absolute ranging with no plate in sight.
                res = self.ekf.update_absolute(w_big, kappa_big, cfg.sigma_w_bbox_px)
                self._watch_lockout(res)
                if res.accepted:
                    channel_a = "big"
                    self._nis_log.append(res.nis)
        if channel_a != "none":
            self._t_last_absolute = t

        # ---- 6. Channel B: scale ratio, MATH.md (7.6) --------------------------------
        channel_b = False
        if measurement is not None:
            res = self.ekf.update_scale(
                measurement.s, measurement.dtau, measurement.sigma_s
            )
            if res.accepted:
                channel_b = True
                self._nis_log.append(res.nis)

        if self.scale.should_reanchor(measurement, t):
            self.scale.anchor(gray, center, w_big, t)
            ax, ay = self.scale.anchor_center
            x, y, bw, bh = track.bbox
            self._kf_box = (float(bw), float(bh), center[0] - ax, center[1] - ay)

        # ---- 7. report ----------------------------------------------------------------
        calibrated = self.kappa.value is not None or (
            t - self._t_last_absolute <= cfg.anchor_hold_s
        )
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
            box_measured=box_measured,
            tracking_ok=self.lost_frames < max(1, cfg.lost_grace_frames),
        )

    def _watch_lockout(self, res) -> None:
        """Detect and undo gate lock-out on Channel A.  MATH.md 7.5

        One rejected measurement is an outlier. A run of rejected measurements that all
        say the same thing is evidence that the filter is wrong: typically a plate read
        at the edge of resolvability seeded lam badly and P collapsed around it. Left
        alone the gate rejects every correct measurement forever -- measured on the
        synthetic closing run as a permanent 25% range error.

        The test is agreement: the innovations must share a sign, and their spread must
        be small next to their median. Scattered rejections are left rejected.

        Agreement is not proof, which is why the run must be long. A plate locator that
        latches onto the wrong feature also returns consistent widths, for a handful of
        frames at a time; a short run (8) mistook those latches for filter error and
        made braking three times worse. The primary defence is upstream --
        plate_min_width_px keeps the locator out of the regime where it latches -- and
        this stays as the net for whatever gets past it.
        """
        if res.accepted:
            self._rejected_innov.clear()
            return
        if res.nis <= 0.0:
            return  # not a real rejection (invalid measurement), carries no evidence
        self._rejected_innov.append(res.innovation)
        n = max(3, self.cfg.reopen_after)
        if len(self._rejected_innov) < n:
            return
        y = np.asarray(self._rejected_innov[-n:], dtype=float)
        med = float(np.median(y))
        spread = 1.4826 * float(np.median(np.abs(y - med)))
        same_sign = bool(np.all(np.sign(y) == np.sign(med)))
        if same_sign and spread < 0.5 * abs(med):
            # Re-open to the size of the disagreement, so the next measurements refine
            # the estimate rather than being gated against a still-overconfident P.
            self.ekf.reopen_lam(med, var_lam=max(spread**2, med**2))
            self.reopen_count += 1
            self._rejected_innov.clear()

    def _propagate_box(self, m: ScaleMeasurement) -> BBox:
        """Move the keyframe's box to where registration places the target.  MATH.md 5.4

        The box scales by s about the anchor point, so both its size and the offset of
        its rear anchor from the registration anchor grow by s.
        """
        kw, kh, ox, oy = self._kf_box
        cx = m.center[0] + ox * m.s
        cy = m.center[1] + oy * m.s
        w, h = kw * m.s, kh * m.s
        return (
            int(round(cx - w / 2.0)),
            int(round(cy - REAR_ANCHOR_FRAC * h)),
            max(4, int(round(w))),
            max(4, int(round(h))),
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
