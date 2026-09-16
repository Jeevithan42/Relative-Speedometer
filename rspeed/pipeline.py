"""End-to-end orchestration: detect -> track -> fuse -> report.

The detector runs only every `cfg.detect_stride` frames. Channel B still runs on
*every* frame in between, and its registration moves each track's box to where the
target actually is (MATH.md 5.4) -- so the boxes are not stale between detections, the
IoU tracker associates the next detection against a current box, and the plate bracket
stays on the plate. The expensive stage can therefore be strided hard without degrading
the velocity channel -- MATH.md section 9.

A track that misses a detection is coasted on the same propagation rather than dropped,
until the tracker's miss budget retires it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .detector import IouTracker, VehicleDetector
from .estimator import SpeedEstimate, VehicleEstimator
from .plate import PlateLocator


class RelativeSpeedPipeline:
    def __init__(
        self,
        cfg: Config,
        detector: VehicleDetector,
        plate_locator: PlateLocator | None = None,
    ):
        self.cfg = cfg
        self.detector = detector
        self.plate_locator = plate_locator
        self.tracker = IouTracker(
            iou_threshold=cfg.track_iou_threshold, max_misses=cfg.track_max_misses
        )
        self.estimators: dict[int, VehicleEstimator] = {}
        self._frame_index = -1
        self._lead_id: int | None = None
        self.detector_calls = 0
        self.plate_calls = 0

    def process(self, frame_bgr: np.ndarray, t: float) -> list[SpeedEstimate]:
        self._frame_index += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        detect_frame = self._frame_index % max(1, self.cfg.detect_stride) == 0
        if detect_frame:
            detections = self.detector.detect(frame_bgr)
            self.detector_calls += 1
            tracks, retired = self.tracker.update(detections)
            for tid in retired:
                # kappa is valid for exactly one vehicle; an id change must destroy it
                # rather than silently carry a wrong scale forward (MATH.md section 11).
                self.estimators.pop(tid, None)
            live = {tr.id for tr in tracks}
            for tid in [k for k in self.estimators if k not in live]:
                self.estimators.pop(tid)
        else:
            tracks = self.tracker.tracks

        # Estimate only the tracks that could plausibly be the lead: each one costs a
        # registration and a plate search. The current lead is always kept.
        ranked = sorted(tracks, key=self._track_score, reverse=True)
        chosen = ranked[: max(1, self.cfg.max_tracks)]
        lead_track = self.tracker.get(self._lead_id) if self._lead_id is not None else None
        if lead_track is not None and lead_track not in chosen:
            chosen[-1] = lead_track

        h, w = gray.shape[:2]
        out: list[SpeedEstimate] = []
        for tr in chosen:
            est = self.estimators.get(tr.id)
            if est is None:
                est = VehicleEstimator(tr.id, self.cfg, self.plate_locator)
                self.estimators[tr.id] = est
            # A box is a measurement only on the frame a detection matched it; any
            # other frame it is moved by registration inside the estimator.
            measured = detect_frame and tr.misses == 0
            result = est.update(gray, tr, t, box_measured=measured)
            if result.plate_ran:
                self.plate_calls += 1

            cx, cy = tr.center
            gone = not (0.0 <= cx < w and 0.0 <= cy < h)
            if not measured and (gone or est.lost_frames > self.cfg.track_lost_frames):
                # Coasting has run out: the box has left the frame, or registration has
                # not placed the target for a while. Retire it now rather than spend
                # the miss budget ranging a box that is no longer on anything.
                self.tracker.drop(tr.id)
                self.estimators.pop(tr.id, None)
                continue
            out.append(result)
        return out

    def _track_score(self, tr) -> float:
        """How plausibly a track is the vehicle directly ahead. Higher is better.

        Apparent size (nearer is bigger) minus horizontal offset from the image centre,
        then penalties for evidence it is not a real, current vehicle: a track seen by
        the detector only once, and a track the detector has since stopped seeing.
        Crude but effective for a forward-facing dashcam; replace with lane-aware
        selection if you have lane detection.
        """
        x, _y, bw, bh = tr.bbox
        cx = self.cfg.image_width / 2.0
        offset = abs((x + bw / 2.0) - cx) / max(1.0, cx)
        score = (max(bw, 1) * max(bh, 1)) ** 0.5 / max(1.0, self.cfg.image_width) - offset
        if tr.hits < 2:
            score -= 0.5
        if tr.misses > 0:
            score -= 0.25
        return score

    def lead(self, estimates: list[SpeedEstimate]) -> SpeedEstimate | None:
        """Pick the vehicle most plausibly directly ahead, with hysteresis.

        The incumbent keeps the lead unless a rival beats it by `lead_switch_margin`.
        Switching lead switches filters, so a flickering choice would make the readout
        jump between unrelated vehicles' estimates.
        """
        if not estimates:
            self._lead_id = None
            return None

        def score(e: SpeedEstimate) -> float:
            tr = self.tracker.get(e.track_id)
            s = self._track_score(tr) if tr is not None else -1.0
            return s if e.tracking_ok else s - 1.0

        best = max(estimates, key=score)
        incumbent = next((e for e in estimates if e.track_id == self._lead_id), None)
        if incumbent is not None and score(best) - score(incumbent) < self.cfg.lead_switch_margin:
            best = incumbent
        self._lead_id = best.track_id
        return best

    @property
    def cost_summary(self) -> dict[str, float]:
        """Cost accounting for MATH.md section 9 -- how often the expensive stages ran."""
        n = max(1, self._frame_index + 1)
        return {
            "frames": float(n),
            "detector_calls": float(self.detector_calls),
            "plate_calls": float(self.plate_calls),
            "detector_rate": self.detector_calls / n,
            "plate_rate": self.plate_calls / n,  # per frame; can exceed 1 with several tracks
        }
