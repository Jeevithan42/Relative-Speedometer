"""Object detection through cv2.dnn -- ONNX models, no torch, no ultralytics.

This is what turns "the thing I boxed by hand" into "the car in front of me". It needs
nothing beyond opencv-python: `cv2.dnn.readNetFromONNX` runs the network, and the
decoding below is plain numpy.

Three output layouts are understood, because they are the ones ONNX detectors actually
ship in:

  yolox   (1, N, 5+C)  raw grid offsets, N = sum over strides 8/16/32 of (S/stride)^2.
                       Objectness and class scores already sigmoid-ed. BGR, 0..255 input.
                       This is the OpenCV Model Zoo's YOLOX-s -- the default model.
  yolov5  (1, 3N, 5+C) decoded centre/size in input pixels, three anchors per cell.
                       RGB, 0..1 input.
  yolov8  (1, 4+C, N)  decoded centre/size, no objectness column. RGB, 0..1 input.
                       Also covers YOLOv11 exports, which share the head.

`auto` tells them apart from the output shape alone. It never guesses from values.

Note OpenCV 5 removed the Darknet importer, so .cfg/.weights models no longer load;
ONNX works on both OpenCV 4 and 5.

The detector only supplies a *bracket*. Nothing here is precise enough to measure with
-- box regression jitters by pixels, and MATH.md section 3 shows what that costs. The
measurement still comes from registration (Channel B) and subpixel edges (Channel A).
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import COCO_VEHICLE_CLASSES, Detection

BBox = tuple[int, int, int, int]

FORMATS = ("auto", "yolox", "yolov5", "yolov8")
_STRIDES = (8, 16, 32)


@dataclass(frozen=True)
class ModelSpec:
    url: str
    sha256: str
    filename: str
    fmt: str
    input_size: int
    note: str


# Downloadable with `python main.py fetch-model`. Checksums pin the exact file tested.
MODEL_ZOO: dict[str, ModelSpec] = {
    "yolox_s": ModelSpec(
        url="https://huggingface.co/opencv/object_detection_yolox/resolve/main/"
            "object_detection_yolox_2022nov.onnx",
        sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
        filename="yolox_s.onnx",
        fmt="yolox",
        input_size=416,
        note="YOLOX-s, COCO 80 classes, 36 MB, Apache-2.0 (OpenCV Model Zoo)",
    ),
}
DEFAULT_MODEL = "models/yolox_s.onnx"


# --- preprocessing -------------------------------------------------------------------


def letterbox(img: np.ndarray, size: int, pad_value: int = 114) -> tuple[np.ndarray, float]:
    """Resize to fit a size x size canvas without distortion, anchored top-left.

    Returns (canvas, ratio). Anchoring top-left means the inverse map is just a division
    by `ratio` -- there is no pad offset to get wrong.
    """
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    nw, nh = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    interp = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return canvas, ratio


# --- decoding (pure numpy, unit-testable without a model) -----------------------------


def grid_size(input_size: int, strides: tuple[int, ...] = _STRIDES) -> int:
    """Number of prediction cells a single-anchor stride-8/16/32 head produces."""
    return sum((input_size // s) ** 2 for s in strides)


def detect_format(shape: tuple[int, ...], input_size: int) -> str:
    """Identify the output layout from its shape. Raises if it is none of the three."""
    if len(shape) == 3 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) != 2:
        raise ValueError(f"unsupported detector output shape {shape}")
    a, b = shape
    cells = grid_size(input_size)
    if a < b and b in (cells, 3 * cells) and a >= 5:
        return "yolov8"
    if a == cells and b >= 6:
        return "yolox"
    if a == 3 * cells and b >= 6:
        return "yolov5"
    raise ValueError(
        f"cannot identify detector output {shape} at input {input_size} "
        f"(expected {cells} or {3 * cells} cells); pass --model-format explicitly"
    )


def decode_yolox(
    out: np.ndarray, input_size: int, strides: tuple[int, ...] = _STRIDES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, 5+C) raw YOLOX head -> (boxes cx,cy,w,h in input px, scores, class ids)."""
    out = out.reshape(-1, out.shape[-1]).astype(np.float32)
    grids, mult = [], []
    for s in strides:
        g = input_size // s
        yy, xx = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        grids.append(np.stack([xx.ravel(), yy.ravel()], axis=1))
        mult.append(np.full((g * g, 1), s, dtype=np.float32))
    grid = np.concatenate(grids).astype(np.float32)
    stride = np.concatenate(mult)
    if grid.shape[0] != out.shape[0]:
        raise ValueError(f"YOLOX output has {out.shape[0]} cells, grid has {grid.shape[0]}")
    xy = (out[:, :2] + grid) * stride
    wh = np.exp(np.clip(out[:, 2:4], -10.0, 10.0)) * stride
    cls_scores = out[:, 4:5] * out[:, 5:]
    return _best_class(np.concatenate([xy, wh], axis=1), cls_scores)


def decode_yolov5(out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, 5+C) decoded YOLOv5 head -> (boxes, scores, class ids)."""
    out = out.reshape(-1, out.shape[-1]).astype(np.float32)
    return _best_class(out[:, :4], out[:, 4:5] * out[:, 5:])


def decode_yolov8(out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(4+C, N) decoded YOLOv8/v11 head -> (boxes, scores, class ids)."""
    if out.ndim == 3:
        out = out[0]
    out = out.T.astype(np.float32)  # (N, 4+C)
    return _best_class(out[:, :4], out[:, 4:])


def _best_class(
    boxes_cxcywh: np.ndarray, cls_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), ids]
    return boxes_cxcywh, scores, ids


def postprocess(
    boxes_cxcywh: np.ndarray,
    scores: np.ndarray,
    ids: np.ndarray,
    ratio: float,
    frame_shape: tuple[int, int],
    conf: float,
    nms: float,
    classes: tuple[int, ...] | None,
    min_size_px: int = 8,
) -> list[tuple[BBox, float, int]]:
    """Threshold, class-filter, map back to frame pixels, clip, and NMS."""
    keep = scores >= conf
    if classes is not None:
        keep &= np.isin(ids, np.asarray(classes))
    if not np.any(keep):
        return []
    b, s, k = boxes_cxcywh[keep] / ratio, scores[keep], ids[keep]

    fh, fw = frame_shape[:2]
    x0 = np.clip(b[:, 0] - b[:, 2] / 2.0, 0, fw)
    y0 = np.clip(b[:, 1] - b[:, 3] / 2.0, 0, fh)
    x1 = np.clip(b[:, 0] + b[:, 2] / 2.0, 0, fw)
    y1 = np.clip(b[:, 1] + b[:, 3] / 2.0, 0, fh)
    xywh = np.stack([x0, y0, x1 - x0, y1 - y0], axis=1)
    big = (xywh[:, 2] >= min_size_px) & (xywh[:, 3] >= min_size_px)
    xywh, s, k = xywh[big], s[big], k[big]
    if xywh.shape[0] == 0:
        return []

    # Class-agnostic on purpose: a pickup scored as both "car" and "truck" is still one
    # vehicle, and two boxes on it would become two tracks fighting over one target.
    idx =cv2.dnn.NMSBoxes(xywh.tolist(), s.astype(float).tolist(), float(conf), float(nms))
    out: list[tuple[BBox, float, int]] = []
    for i in np.asarray(idx, dtype=int).ravel():
        x, y, w, h = xywh[i]
        out.append(((int(round(x)), int(round(y)), int(round(w)), int(round(h))),
                    float(s[i]), int(k[i])))
    return out


# --- the detector ---------------------------------------------------------------------


class DnnDetector:
    """Any supported ONNX detector, returning boxes in frame pixels."""

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 416,
        conf: float = 0.35,
        nms: float = 0.45,
        classes: tuple[int, ...] | None = None,
        fmt: str = "auto",
    ):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"detector model not found: {path}\n"
                f"  fetch the default with:  python main.py fetch-model"
            )
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
        if input_size % 32:
            raise ValueError(f"input_size must be a multiple of 32, got {input_size}")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.input_size = int(input_size)
        self.conf = float(conf)
        self.nms = float(nms)
        self.classes = classes
        self.fmt = fmt
        self.path = path

    def _blob(self, canvas: np.ndarray, fmt: str) -> np.ndarray:
        if fmt == "yolox":
            return cv2.dnn.blobFromImage(canvas, 1.0, swapRB=False)
        return cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, swapRB=True)

    def detect_boxes(self, frame_bgr: np.ndarray) -> list[tuple[BBox, float, int]]:
        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        canvas, ratio = letterbox(frame_bgr, self.input_size)
        # The input normalisation depends on the format, so an `auto` model needs one
        # forward pass to learn its layout. YOLOX is tried first: it is the default.
        fmt = "yolox" if self.fmt == "auto" else self.fmt
        self.net.setInput(self._blob(canvas, fmt))
        out = np.asarray(self.net.forward())
        if self.fmt == "auto":
            detected = detect_format(out.shape, self.input_size)
            self.fmt = detected
            if detected != fmt:
                self.net.setInput(self._blob(canvas, detected))
                out = np.asarray(self.net.forward())
            fmt = detected

        if fmt == "yolox":
            boxes, scores, ids = decode_yolox(out, self.input_size)
        elif fmt == "yolov5":
            boxes, scores, ids = decode_yolov5(out)
        else:
            boxes, scores, ids = decode_yolov8(out)
        return postprocess(boxes, scores, ids, ratio, frame_bgr.shape, self.conf,
                           self.nms, self.classes)


class DnnVehicleDetector:
    """The `VehicleDetector` protocol over a COCO model, restricted to vehicles."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        input_size: int = 416,
        conf: float = 0.35,
        nms: float = 0.45,
        fmt: str = "auto",
        classes: tuple[int, ...] = COCO_VEHICLE_CLASSES,
    ):
        self.dnn = DnnDetector(model_path, input_size, conf, nms, classes, fmt)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        return [Detection(bbox=b, score=s, cls=k) for b, s, k in self.dnn.detect_boxes(frame_bgr)]


# --- model download --------------------------------------------------------------------


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_model(name: str = "yolox_s", dest_dir: str | Path = "models") -> Path:
    """Download a MODEL_ZOO entry, verify its checksum, and return its path.

    Downloads to a temporary name and renames only after the checksum matches, so an
    interrupted or tampered download never leaves a file that looks usable.
    """
    spec = MODEL_ZOO[name]
    dest = Path(dest_dir) / spec.filename
    if dest.exists() and sha256_of(dest) == spec.sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(1.0, blocks * block_size / total)
            sys.stdout.write(f"\r  {done:6.1%} of {total / 1e6:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(spec.url, tmp, reporthook=progress)
    sys.stdout.write("\n")
    digest = sha256_of(tmp)
    if digest != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {name}: got {digest}, expected {spec.sha256}")
    tmp.replace(dest)
    return dest
