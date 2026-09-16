"""Validation of the cv2.dnn detector path -- decoding, letterboxing, filtering.

The decoders are pure numpy, so they are tested against hand-built output tensors whose
correct answer is known exactly. No model file is needed; the one test that runs a real
network skips itself when models/yolox_s.onnx is absent.

Runs under pytest, or standalone:  python tests/test_dnn.py
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rspeed.detector import COCO_VEHICLE_CLASSES  # noqa: E402
from rspeed.dnn import (  # noqa: E402
    DEFAULT_MODEL,
    DnnVehicleDetector,
    decode_yolov5,
    decode_yolov8,
    decode_yolox,
    detect_format,
    grid_size,
    letterbox,
    postprocess,
)

S = 64  # input size: cells = 8*8 + 4*4 + 2*2 = 84
C = 80
REPO = Path(__file__).resolve().parents[1]


def _yolox_tensor(gx=3, gy=2, w_px=20.0, obj=0.9, cls=2, p=0.8) -> np.ndarray:
    out = np.zeros((grid_size(S), 5 + C), np.float32)
    i = gy * (S // 8) + gx  # a stride-8 cell; they come first
    out[i, 0:2] = 0.5  # offset within the cell
    out[i, 2:4] = math.log(w_px / 8.0)  # log(size / stride)
    out[i, 4] = obj
    out[i, 5 + cls] = p
    return out[None]


def test_grid_size_matches_yolox_layout():
    assert grid_size(640) == 8400  # the layout of the shipped YOLOX-s at 640
    assert grid_size(416) == 3549
    assert grid_size(S) == 84


def test_detect_format_from_shape_alone():
    n = grid_size(416)
    assert detect_format((1, n, 85), 416) == "yolox"
    assert detect_format((1, 3 * n, 85), 416) == "yolov5"
    assert detect_format((1, 84, n), 416) == "yolov8"
    try:
        detect_format((1, 1000, 85), 416)
    except ValueError:
        pass
    else:
        raise AssertionError("an unrecognisable layout must raise, not be guessed")


def test_decode_yolox_grid_offsets():
    """Centre = (cell + offset) * stride, size = exp(raw) * stride, score = obj * cls."""
    boxes, scores, ids = decode_yolox(_yolox_tensor(), S)
    i = int(np.argmax(scores))
    assert ids[i] == 2
    assert abs(scores[i] - 0.72) < 1e-6
    cx, cy, w, h = boxes[i]
    assert abs(cx - 28.0) < 1e-4 and abs(cy - 20.0) < 1e-4, (cx, cy)
    assert abs(w - 20.0) < 1e-3 and abs(h - 20.0) < 1e-3, (w, h)


def test_decode_yolov8_transposes_and_has_no_objectness():
    n = grid_size(S)
    out = np.zeros((1, 4 + C, n), np.float32)
    out[0, :4, 5] = [30.0, 22.0, 10.0, 8.0]
    out[0, 4 + 7, 5] = 0.66
    boxes, scores, ids = decode_yolov8(out)
    i = int(np.argmax(scores))
    assert i == 5 and ids[i] == 7 and abs(scores[i] - 0.66) < 1e-6
    assert np.allclose(boxes[i], [30.0, 22.0, 10.0, 8.0])


def test_decode_yolov5_multiplies_objectness():
    out = np.zeros((1, 3 * grid_size(S), 5 + C), np.float32)
    out[0, 11, :5] = [12.0, 14.0, 6.0, 6.0, 0.5]
    out[0, 11, 5 + 3] = 0.9
    boxes, scores, ids = decode_yolov5(out)
    i = int(np.argmax(scores))
    assert i == 11 and ids[i] == 3 and abs(scores[i] - 0.45) < 1e-6


def test_letterbox_is_top_left_so_inverse_is_a_division():
    img = np.full((100, 200, 3), 255, np.uint8)
    canvas, ratio = letterbox(img, S)
    assert canvas.shape == (S, S, 3)
    assert abs(ratio - 0.32) < 1e-9
    assert canvas[0, 0, 0] == 255  # image content starts at the origin
    assert canvas[S - 1, 0, 0] == 114  # padding below it


def test_postprocess_maps_back_to_frame_pixels():
    """A box decoded at input scale must land in frame pixels after dividing by ratio."""
    boxes, scores, ids = decode_yolox(_yolox_tensor(), S)
    dets = postprocess(boxes, scores, ids, ratio=0.5, frame_shape=(128, 128), conf=0.3,
                       nms=0.45, classes=COCO_VEHICLE_CLASSES)
    assert len(dets) == 1, dets
    (x, y, w, h), score, cls = dets[0]
    assert (x, y, w, h) == (36, 20, 40, 40), (x, y, w, h)
    assert cls == 2 and abs(score - 0.72) < 1e-6


def test_postprocess_keeps_only_vehicle_classes():
    boxes, scores, ids = decode_yolox(_yolox_tensor(cls=16), S)  # 16 = dog
    assert postprocess(boxes, scores, ids, 1.0, (S, S), 0.3, 0.45, COCO_VEHICLE_CLASSES) == []
    assert len(postprocess(boxes, scores, ids, 1.0, (S, S), 0.3, 0.45, None)) == 1


def test_nms_is_class_agnostic():
    """A pickup scored as both car and truck is one vehicle, so it must be one box."""
    boxes = np.array([[30, 30, 20, 20], [31, 30, 20, 21]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    ids = np.array([2, 7])
    dets = postprocess(boxes, scores, ids, 1.0, (S, S), 0.3, 0.45, COCO_VEHICLE_CLASSES)
    assert len(dets) == 1 and dets[0][2] == 2


def test_postprocess_clips_to_frame():
    boxes = np.array([[2, 2, 30, 30]], np.float32)  # overhangs the top-left corner
    dets = postprocess(boxes, np.array([0.9]), np.array([2]), 1.0, (S, S), 0.3, 0.45, None)
    (x, y, w, h), _s, _k = dets[0]
    assert x == 0 and y == 0 and w == 17 and h == 17, dets


def test_missing_model_names_the_fix():
    try:
        DnnVehicleDetector("models/definitely-not-here.onnx")
    except FileNotFoundError as exc:
        assert "fetch-model" in str(exc)
    else:
        raise AssertionError("a missing model must raise FileNotFoundError")


def test_real_model_runs_if_present():
    """Smoke test of the shipped network: loads, runs, finds nothing in a blank frame."""
    path = REPO / DEFAULT_MODEL
    if not path.exists():
        print("    (skipped: run `python main.py fetch-model` to enable)")
        return
    det = DnnVehicleDetector(path, input_size=320)
    assert det.detect(np.full((360, 640, 3), 120, np.uint8)) == []
    assert det.dnn.fmt == "yolox"


def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
