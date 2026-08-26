"""End-to-end orchestration: detect -> track -> fuse -> report.

The detector runs only every `cfg.detect_stride` frames. Channel B still runs on
*every* frame in between, and that is safe for a reason worth stating: the scale
estimator's ROI size is frozen with its keyframe and ECC absorbs residual translation,
so a stale bounding-box centre costs nothing. The expensive stage can therefore be
strided hard without degrading the velocity channel -- MATH.md section 9.
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
        self.detector_calls = 0
        self.plate_calls = 0

    def process(self, frame_bgr: np.ndarray, t: float) -> list[SpeedEstimate]:
        self._frame_index += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._frame_index % max(1, self.cfg.detect_stride) == 0:
            detections = self.detector.detect(frame_bgr)
            self.detector_calls += 1
            tracks, retired = self.tracker.update(detections)
            for tid in retired:
                # kappa is valid for exactly one vehicle; an id change must destroy it
                # rather than silently carry a wrong scale forward (MATH.md section 11).
                self.estimators.pop(tid, None)
        else:
            tracks = self.tracker.tracks

        out: list[SpeedEstimate] = []
        for tr in tracks:
            if tr.misses > 0 and self._frame_index % max(1, self.cfg.detect_stride) == 0:
                continue  # coasting on a miss: don't feed the filter a stale box
            est = self.estimators.get(tr.id)
            if est is None:
                est = VehicleEstimator(tr.id, self.cfg, self.plate_locator)
                self.estimators[tr.id] = est
            result = est.update(gray, tr, t)
            if result.plate_ran:
                self.plate_calls += 1
            out.append(result)
        return out

    def lead(self, estimates: list[SpeedEstimate]) -> SpeedEstimate | None:
        """Pick the vehicle most plausibly directly ahead.

        Scores by apparent size (nearer is bigger) and penalises horizontal offset from
        the image centre. Crude but effective for a forward-facing dashcam; replace
        with lane-aware selection if you have lane detection.
        """
        if not estimates:
            return None
        cx = self.cfg.image_width / 2.0
        best, best_score = None, -float("inf")
        for e in estimates:
            x, _y, w, h = e.bbox
            offset = abs((x + w / 2.0) - cx) / max(1.0, cx)
            score = (w * h) ** 0.5 / max(1.0, self.cfg.image_width) - offset
            if score > best_score:
                best, best_score = e, score
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
            "plate_rate": self.plate_calls / n,
        }
