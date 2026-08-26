"""Vehicle detection and identity tracking.

Deliberately pluggable. The measurement mathematics (geometry, filter, scale, plate)
carries all the value in this project and none of it depends on which detector is
present, so the detector is behind a two-method protocol and the package imports
cleanly with no torch installed.

Backends:
  * YoloVehicleDetector  -- ultralytics YOLO, COCO vehicle classes. Production path.
  * ScriptedDetector     -- returns boxes from a supplied callable. Used by the
                            synthetic harness so the whole pipeline is testable
                            without any model weights.

`IouTracker` gives stable integer track ids without any external tracker dependency.
Track identity matters more here than in ordinary detection work: an ID switch
silently invalidates a track's kappa calibration (MATH.md section 11), so the pipeline
destroys calibration state whenever an id is retired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

BBox = tuple[int, int, int, int]  # x, y, w, h

# COCO class ids that are rear-facing vehicles worth ranging.
COCO_VEHICLE_CLASSES = (2, 3, 5, 7)  # car, motorcycle, bus, truck


@dataclass
class Detection:
    bbox: BBox
    score: float = 1.0
    cls: int = 2


@dataclass
class Track:
    """A detected vehicle with a stable identity across frames."""

    id: int
    bbox: BBox
    score: float
    cls: int
    age: int = 0  # frames since first seen
    misses: int = 0  # consecutive frames without a matching detection
    hits: int = 1
    history: list[BBox] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    @property
    def width(self) -> float:
        return float(self.bbox[2])

    @property
    def rear_center(self) -> tuple[float, float]:
        """Centre of the lower half of the box -- closer to where a plate sits, which
        keeps the scale-ratio ROI on the rear face rather than on roofline and sky."""
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h * 0.70)


class VehicleDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


# --- backends -------------------------------------------------------------------------


class ScriptedDetector:
    """Detector driven by a callable, for synthetic sequences and unit tests."""

    def __init__(self, fn: Callable[[int, np.ndarray], list[Detection]]):
        self._fn = fn
        self._frame_index = -1

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        self._frame_index += 1
        return self._fn(self._frame_index, frame_bgr)


class YoloVehicleDetector:  # pragma: no cover - requires optional dependency
    """Ultralytics YOLO restricted to vehicle classes."""

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        conf: float = 0.35,
        imgsz: int = 640,
        classes: tuple[int, ...] = COCO_VEHICLE_CLASSES,
        device: str | None = None,
    ):
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "YoloVehicleDetector needs ultralytics: pip install ultralytics"
            ) from exc
        self._model = YOLO(weights)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.classes = list(classes)
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        res = self._model.predict(
            frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        out: list[Detection] = []
        if not res or res[0].boxes is None:
            return out
        boxes = res[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            out.append(
                Detection(
                    bbox=(int(x1), int(y1), int(round(x2 - x1)), int(round(y2 - y1))),
                    score=float(c),
                    cls=int(k),
                )
            )
        return out


# --- tracking ---------------------------------------------------------------------------


def iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, x1 - x0), max(0, y1 - y0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


class IouTracker:
    """Greedy IoU association with a miss budget.

    Adequate for the lead-vehicle case this project targets -- one dominant, slowly
    moving box near the image centre. Swap in ByteTrack or BoT-SORT for dense
    multi-target scenes; the pipeline only needs stable `Track.id` values.
    """

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 8):
        self.iou_threshold = float(iou_threshold)
        self.max_misses = int(max_misses)
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def update(self, detections: list[Detection]) -> tuple[list[Track], list[int]]:
        """Associate detections to tracks.

        Returns (live_tracks, retired_ids). Retired ids matter to the caller: any
        per-track calibration keyed on them must be destroyed (MATH.md section 11).
        """
        unmatched = set(range(len(detections)))
        pairs: list[tuple[float, int, int]] = []
        for tid, tr in self._tracks.items():
            for di, det in enumerate(detections):
                score = iou(tr.bbox, det.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, tid, di))
        pairs.sort(reverse=True)

        used_tracks: set[int] = set()
        for score, tid, di in pairs:
            if tid in used_tracks or di not in unmatched:
                continue
            tr = self._tracks[tid]
            det = detections[di]
            tr.history.append(tr.bbox)
            if len(tr.history) > 64:
                tr.history.pop(0)
            tr.bbox, tr.score, tr.cls = det.bbox, det.score, det.cls
            tr.misses = 0
            tr.hits += 1
            used_tracks.add(tid)
            unmatched.discard(di)

        for tid, tr in self._tracks.items():
            tr.age += 1
            if tid not in used_tracks:
                tr.misses += 1

        for di in sorted(unmatched):
            det = detections[di]
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(
                id=tid, bbox=det.bbox, score=det.score, cls=det.cls
            )

        retired = [tid for tid, tr in self._tracks.items() if tr.misses > self.max_misses]
        for tid in retired:
            del self._tracks[tid]

        return self.tracks, retired
