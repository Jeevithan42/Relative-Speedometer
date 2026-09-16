"""Log-inverse-depth EKF -- the fusion point for both measurement channels.

State:      x = [lam, lamdot]        MATH.md section 7.1
Process:    exact closed form under constant relative velocity, (7.1)-(7.2)
Channel A:  ln(w) = kappa + lam      linear, H = [1, 0]      (7.5)
Channel B:  (s-1)/dtau  = lamdot     linear, H = [0, 1]      (7.6)

The whole reason for working in log-inverse-depth is visible here: both channels are
*linear* measurements on the state, so fusing them costs one extra scalar update and
nothing else. See MATH.md section 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# chi-square 99.7% point with 1 degree of freedom, MATH.md section 7.4
DEFAULT_GATE_NIS = 9.0

# lamdot * dt must stay below 1 for (7.1)-(7.2); clamp well short of the singularity.
_MAX_LAMDOT_DT = 0.5


@dataclass
class UpdateResult:
    """Outcome of one scalar measurement update."""

    accepted: bool
    nis: float  # normalised innovation squared, y^2 / S
    innovation: float


class LogDepthEKF:
    """Extended Kalman filter on [lam, lamdot].

    Only the *process* model is nonlinear; both measurement models are exactly linear,
    so this is an EKF in name only on the update side.
    """

    def __init__(self, q: float = 0.008, gate_nis: float = DEFAULT_GATE_NIS):
        """
        Args:
            q: process noise spectral density for the white-noise-acceleration model,
               MATH.md (7.4). Tune per MATH.md section 10 -- roughly (a_max/Z)^2 * tau_c.
            gate_nis: reject measurements whose normalised innovation squared exceeds
               this. 9.0 is the 99.7% chi-square point for 1 DOF.
        """
        self.q = float(q)
        self.gate_nis = float(gate_nis)
        self.x = np.zeros(2, dtype=float)
        self.P = np.eye(2, dtype=float)
        self._initialised = False

    # -- lifecycle ------------------------------------------------------------------

    @property
    def initialised(self) -> bool:
        return self._initialised

    def initialise(
        self,
        lam: float,
        lamdot: float = 0.0,
        var_lam: float = 1.0,
        var_lamdot: float = 1.0,
    ) -> None:
        """Seed the filter. var_lam = 1.0 is deliberately loose: it corresponds to
        roughly a factor-of-e uncertainty in depth, so the first good Channel A
        measurement dominates immediately."""
        self.x = np.array([lam, lamdot], dtype=float)
        self.P = np.diag([var_lam, var_lamdot]).astype(float)
        self._initialised = True

    # -- prediction, MATH.md section 7.2 ---------------------------------------------

    def predict(self, dt: float) -> None:
        """Propagate the state forward by dt using the exact constant-relative-velocity
        solution in log-inverse-depth space.

            lam+    = lam - ln(1 - lamdot*dt)
            lamdot+ = lamdot / (1 - lamdot*dt)

        Note this needs neither Z nor v individually, only their observable ratio.
        """
        if not self._initialised or dt <= 0:
            return

        lam, lamdot = self.x
        # Guard the singularity at lamdot*dt = 1 (i.e. TTC == dt).
        ld_dt = float(np.clip(lamdot * dt, -_MAX_LAMDOT_DT, _MAX_LAMDOT_DT))
        g = 1.0 - ld_dt

        self.x = np.array([lam - math.log(g), lamdot / g], dtype=float)

        # Jacobian, MATH.md (7.3)
        F = np.array([[1.0, dt / g], [0.0, 1.0 / (g * g)]], dtype=float)

        # White-noise-acceleration process noise, MATH.md (7.4)
        Q = self.q * np.array(
            [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]], dtype=float
        )

        self.P = F @ self.P @ F.T + Q
        self._symmetrise()

    # -- generic scalar update -------------------------------------------------------

    def _update_scalar(self, z: float, H: np.ndarray, R: float) -> UpdateResult:
        """One gated scalar Kalman update, Joseph form.

        Joseph form costs one extra 2x2 product but keeps P symmetric and positive
        definite through long runs of sequential scalar updates, where the short form
        can drift negative.
        """
        if not self._initialised:
            return UpdateResult(False, 0.0, 0.0)
        if not math.isfinite(z) or R <= 0 or not math.isfinite(R):
            return UpdateResult(False, 0.0, 0.0)

        y = float(z - H @ self.x)
        S = float(H @ self.P @ H.T + R)
        if S <= 0:
            return UpdateResult(False, 0.0, y)

        nis = (y * y) / S
        if nis > self.gate_nis:
            return UpdateResult(False, nis, y)  # MATH.md section 7.4

        K = (self.P @ H.T) / S  # shape (2,)
        self.x = self.x + K * y

        I_KH = np.eye(2) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * R
        self._symmetrise()

        return UpdateResult(True, nis, y)

    # -- Channel A: known-size ranging, MATH.md (7.5) --------------------------------

    def update_absolute(
        self, w_px: float, kappa: float, sigma_w_px: float
    ) -> UpdateResult:
        """Fuse an apparent width whose log scale constant kappa = ln(f*W) is known.

            z = ln(w) - kappa      H = [1, 0]      R = (sigma_w / w)^2

        Working in ln(w) is what makes the noise homoscedastic: a detector's pixel
        error is roughly proportional to feature size, so sigma_w/w is stable across
        the depth range while sigma_w alone is not.
        """
        if w_px <= 0:
            return UpdateResult(False, 0.0, 0.0)
        z = math.log(w_px) - kappa
        sigma_lnw = sigma_w_px / w_px
        return self._update_scalar(z, np.array([1.0, 0.0]), sigma_lnw**2)

    # -- Channel B: scale ratio, MATH.md (7.6) ---------------------------------------

    def update_scale(self, s: float, dtau: float, sigma_s: float) -> UpdateResult:
        """Fuse a scale ratio measured by image registration over baseline dtau.

            z = (s - 1)/dtau       H = [0, 1]      R = (sigma_s / dtau)^2

        Requires no calibration at all -- this channel alone yields TTC.
        """
        if dtau <= 0 or s <= 0:
            return UpdateResult(False, 0.0, 0.0)
        z = (s - 1.0) / dtau
        R = (sigma_s / dtau) ** 2
        return self._update_scalar(z, np.array([0.0, 1.0]), R)

    # -- recovery from gate lock-out, MATH.md section 7.5 -----------------------------

    def reopen_lam(self, shift: float, var_lam: float) -> None:
        """Move lam by `shift` and re-open its variance to at least `var_lam`.

        For the case gating cannot handle on its own: a bad early measurement seeds lam
        wrong, P collapses, and from then on every *correct* measurement is rejected as
        an outlier. The caller decides when the evidence says the filter, not the
        measurements, is wrong. lamdot is untouched -- Channel B keeps it honest
        independently of lam. The cross-covariance is dropped because the old
        correlation belonged to the state being discarded.
        """
        if not self._initialised:
            return
        self.x[0] += float(shift)
        self.P[0, 0] = max(float(self.P[0, 0]), float(var_lam))
        self.P[0, 1] = self.P[1, 0] = 0.0

    # -- physical outputs, MATH.md (4.5)-(4.7) ---------------------------------------

    @property
    def lam(self) -> float:
        return float(self.x[0])

    @property
    def lamdot(self) -> float:
        return float(self.x[1])

    @property
    def Z(self) -> float:
        """Depth in metres."""
        return math.exp(-self.lam)

    @property
    def Zdot(self) -> float:
        """Relative speed in m/s. Negative = closing."""
        return -self.lamdot * self.Z

    @property
    def ttc(self) -> float:
        """Time to contact in seconds, +inf if not closing."""
        return math.inf if self.lamdot <= 0 else 1.0 / self.lamdot

    @property
    def sigma_Z(self) -> float:
        """1-sigma depth uncertainty. dZ/dlam = -Z, so var_Z = Z^2 * P[0,0]."""
        return self.Z * math.sqrt(max(self.P[0, 0], 0.0))

    @property
    def sigma_Zdot(self) -> float:
        """1-sigma relative-speed uncertainty, by linear propagation of P.

        Zdot = -lamdot * exp(-lam), so the Jacobian is
            d(Zdot)/d(lam)    = +lamdot * Z
            d(Zdot)/d(lamdot) = -Z
        """
        Z = self.Z
        J = np.array([self.lamdot * Z, -Z], dtype=float)
        var = float(J @ self.P @ J.T)
        return math.sqrt(max(var, 0.0))

    def _symmetrise(self) -> None:
        self.P = 0.5 * (self.P + self.P.T)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        if not self._initialised:
            return "<LogDepthEKF uninitialised>"
        return (
            f"<LogDepthEKF Z={self.Z:.2f}+-{self.sigma_Z:.2f}m "
            f"Zdot={self.Zdot:+.2f}+-{self.sigma_Zdot:.2f}m/s ttc={self.ttc:.1f}s>"
        )
