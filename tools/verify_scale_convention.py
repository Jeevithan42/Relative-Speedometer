"""Empirically pin down the ECC / log-polar scale sign conventions.

OpenCV's documentation on the direction of the warp returned by findTransformECC is
easy to misread, and a flipped convention would silently invert every speed reading.
This script synthesises a known scaling and reports which formula recovers it, so the
constant in rspeed/scale.py is a measured fact rather than an assumption.

Run:  python tools/verify_scale_convention.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rspeed.scale import ScaleEstimator  # noqa: E402


def textured_patch(n: int = 256, seed: int = 0) -> np.ndarray:
    """A patch with structure at several spatial frequencies, like a plate + car rear."""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), 90, dtype=np.uint8)
    cv2.rectangle(img, (n // 4, n // 3), (3 * n // 4, 2 * n // 3), 230, -1)
    for i in range(6):
        x = n // 4 + 8 + i * 18
        cv2.rectangle(img, (x, n // 3 + 12), (x + 9, 2 * n // 3 - 12), 30, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 2.0, img.shape), 0, 255)
    return img.astype(np.uint8)


def scaled(img: np.ndarray, s: float) -> np.ndarray:
    """Scale image content about its centre by s, keeping canvas size fixed."""
    n = img.shape[0]
    M = cv2.getRotationMatrix2D((n / 2.0, n / 2.0), 0.0, s)
    return cv2.warpAffine(img, M, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main() -> int:
    base = textured_patch()
    est = ScaleEstimator()
    n = base.shape[0]
    center = (n / 2.0, n / 2.0)

    print(f"{'true s':>8} {'ecc raw':>10} {'ecc inv':>10} {'logpolar':>10}")
    print("-" * 42)

    errs_direct, errs_inverse, errs_lp = [], [], []
    for true_s in (0.90, 0.95, 1.05, 1.10, 1.20):
        ref = est._extract(base, center, n)
        cur = est._extract(scaled(base, true_s), center, n)

        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _cc, warp = cv2.findTransformECC(
                cur, ref, warp, cv2.MOTION_AFFINE, est._criteria, None, 5
            )
            det = float(np.linalg.det(warp[:2, :2]))
            s_direct = math.sqrt(abs(det))
            s_inverse = 1.0 / s_direct
        except cv2.error:
            s_direct = s_inverse = float("nan")

        lp = est._register_logpolar(ref, cur)
        s_lp = lp[0] if lp else float("nan")

        print(f"{true_s:8.3f} {s_direct:10.4f} {s_inverse:10.4f} {s_lp:10.4f}")
        errs_direct.append(abs(s_direct - true_s))
        errs_inverse.append(abs(s_inverse - true_s))
        errs_lp.append(abs(s_lp - true_s))

    md, mi, ml = np.mean(errs_direct), np.mean(errs_inverse), np.mean(errs_lp)
    print("-" * 42)
    print(f"mean |err|   direct={md:.4f}  inverse={mi:.4f}  logpolar={ml:.4f}")
    winner = "INVERSE  (_ECC_SCALE_IS_INVERSE = True)" if mi < md else "DIRECT   (_ECC_SCALE_IS_INVERSE = False)"
    print(f"\nECC convention -> {winner}")
    if ml > 0.05:
        print("log-polar sign may need flipping (shift_x -> -shift_x)")

    # The translation half of the warp is what tracks the target (MATH.md 5.4). A sign
    # slip there moves the box AWAY from the target, which then looks like tracking loss
    # rather than an obvious error -- so it gets the same empirical treatment.
    print(f"\n{'shift':>14} {'scale':>6} {'located':>18} {'error px':>10}")
    print("-" * 52)
    big = np.full((600, 600), 90, dtype=np.uint8)
    big[172:428, 172:428] = base
    anchor = (300.0, 300.0)
    worst = 0.0
    for (dx, dy), s in (((5.0, -3.0), 1.0), ((22.0, 8.0), 1.08), ((-35.0, 14.0), 0.95)):
        M = cv2.getRotationMatrix2D(anchor, 0.0, s)
        M[:, 2] += (dx, dy)
        moved = cv2.warpAffine(big, M, (600, 600), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        tracker = ScaleEstimator(roi_factor=1.0, roi_max=512)
        tracker.anchor(big, anchor, 256.0, 0.0)
        m = tracker.measure(moved, anchor, 0.1)
        truth = (anchor[0] + dx, anchor[1] + dy)
        if m is None or m.center is None:
            print(f"({dx:+5.1f},{dy:+5.1f}) {s:6.2f} {'FAILED':>18}")
            worst = float("inf")
            continue
        err = math.hypot(m.center[0] - truth[0], m.center[1] - truth[1])
        worst = max(worst, err)
        print(f"({dx:+5.1f},{dy:+5.1f}) {s:6.2f} "
              f"({m.center[0]:7.2f},{m.center[1]:7.2f}) {err:10.3f}")
    print(f"\ntranslation -> {'OK' if worst < 1.0 else 'WRONG: check ScaleEstimator._locate'}"
          f" (worst {worst:.3f} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
