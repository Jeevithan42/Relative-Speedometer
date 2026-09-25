"""Confirm one camera index actually delivers frames, for use as a pre-flight check.

Exists because DSHOW lies twice over. With nothing plugged into index 1 on this machine,
cv2.VideoCapture(1, CAP_DSHOW).isOpened() returns True *and* reads succeed, delivering
1280x720 frames of near-black (mean 3.3). So neither the isOpened guard nor a successful
read proves a camera is there: `video` runs the whole pipeline on that feed, detecting
nothing, with no error and a black --show window. Brightness is the signal that
distinguishes a camera from a phantom, which is why it is reported here.

The caller should still bound this with a timeout and kill it -- a device that is absent
in a different way, or held open by another process, can block inside the read. Do not
add the timeout in here; a blocked DSHOW read does not return control to Python.

Usage:  python tools/probe_index.py <index> [width] [height]
Exit:   0 = frames arrived (prints "ok <w>x<h> <mean_brightness>"), 1 = they did not.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from rspeed.camera import open_camera  # noqa: E402


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: probe_index.py <index> [width] [height]", file=sys.stderr)
        return 2
    index = int(argv[0])
    width = int(argv[1]) if len(argv) > 1 else None
    height = int(argv[2]) if len(argv) > 2 else None

    try:
        cap = open_camera(index, width, height)
    except RuntimeError as exc:
        print(f"fail {exc}", file=sys.stderr)
        return 1

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"fail index {index} opened but delivered no frame", file=sys.stderr)
            return 1
        h, w = frame.shape[:2]
        mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        print(f"ok {w}x{h} {mean:.1f}")
        return 0
    finally:
        cap.release()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
