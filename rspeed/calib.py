"""The kappa transfer -- plate calibrates the larger feature.  MATH.md section 8.

The plate has a known real width but goes unreadable past ~35 m. The vehicle's rear
box is large and stable but has an unknown real width. While both are visible:

    kappa_big = kappa_plate + ln(w_big) - ln(w_plate)                     (8.1)

which eliminates lam. Note it never separates f from W_big -- only the product
matters, so a focal-length error and a compensating width error cancel exactly.

Once kappa_big is locked, Channel A switches to the large feature and the plate
detector is descheduled, which is where the efficiency comes back (MATH.md section 9).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class KappaEstimator:
    """Robust running estimate of one track's log scale constant for a large feature.

    Uses a median over a bounded ring buffer rather than a running mean. Plate
    localisation fails in a *biased* way -- it latches onto a bumper slot or a badge
    and reports a wrong width, not a noisy one -- and a mean would let a handful of
    those drag the calibration permanently off. A median shrugs them off.
    """

    def __init__(
        self,
        window: int = 45,
        min_samples: int = 12,
        lock_mad: float = 0.035,
        max_mad: float = 0.25,
    ):
        """
        Args:
            window: ring buffer length. At 30 fps, 45 samples is 1.5 s of plate views.
            min_samples: never lock before this many samples, however tight they look.
            lock_mad: lock when the median absolute deviation of ln-space samples falls
                below this. 0.035 in log space is about 3.5% width agreement, which at
                20 m is a depth spread under a metre.
            max_mad: above this the samples are incoherent -- refuse to lock and keep
                asking for plates.
        """
        self.window = int(window)
        self.min_samples = int(min_samples)
        self.lock_mad = float(lock_mad)
        self.max_mad = float(max_mad)
        self._samples: deque[float] = deque(maxlen=self.window)
        self._locked = False
        self._locked_value: float | None = None

    # -- accumulation ------------------------------------------------------------------

    def add_pair(
        self, w_plate_px: float, w_big_px: float, kappa_plate: float
    ) -> float | None:
        """Contribute one simultaneous (plate, large-feature) observation.

        Applies MATH.md (8.1). Returns the sample added, or None if rejected.
        """
        if w_plate_px <= 0 or w_big_px <= 0:
            return None
        sample = kappa_plate + math.log(w_big_px) - math.log(w_plate_px)
        if not math.isfinite(sample):
            return None
        self._samples.append(sample)
        self._maybe_lock()
        return sample

    def _maybe_lock(self) -> None:
        if self._locked or len(self._samples) < self.min_samples:
            return
        mad = self.mad
        if mad is not None and mad < self.lock_mad:
            self._locked = True
            self._locked_value = self.median

    # -- state -------------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def median(self) -> float | None:
        if not self._samples:
            return None
        return float(np.median(np.fromiter(self._samples, dtype=float)))

    @property
    def mad(self) -> float | None:
        """Median absolute deviation in log space -- a scale-free spread measure."""
        if len(self._samples) < 3:
            return None
        a = np.fromiter(self._samples, dtype=float)
        return float(np.median(np.abs(a - np.median(a))))

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def value(self) -> float | None:
        """Best available kappa_big, or None if not yet usable.

        Returns the frozen value once locked; before that, the running median as soon
        as there are enough coherent samples to be worth using.
        """
        if self._locked:
            return self._locked_value
        if len(self._samples) < self.min_samples:
            return None
        mad = self.mad
        if mad is None or mad > self.max_mad:
            return None
        return self.median

    @property
    def sigma_ln(self) -> float:
        """1-sigma uncertainty on kappa_big in log space.

        MAD -> sigma uses the Gaussian consistency factor 1.4826, divided by sqrt(n)
        for the standard error of the median (approximately -- the median's efficiency
        is lower than the mean's, so this is mildly optimistic).
        """
        mad = self.mad
        if mad is None:
            return float("inf")
        return 1.4826 * mad / math.sqrt(max(1, len(self._samples)))

    def reset(self) -> None:
        """Discard the calibration. Call on any track identity change -- kappa_big is
        valid only for one specific vehicle (MATH.md section 8)."""
        self._samples.clear()
        self._locked = False
        self._locked_value = None

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        v = self.value
        state = "locked" if self._locked else "open"
        vs = f"{v:.3f}" if v is not None else "None"
        return f"<KappaEstimator {state} n={self.n_samples} kappa={vs}>"
