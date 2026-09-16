"""Debug overlay. Deliberately shows the internals, not just the answer.

When this drifts on real footage you need to see *why*: whether Channel A had an
anchor, whether Channel B registered, whether kappa locked, and how confident the
filter claims to be. A clean speed readout with no provenance is unfalsifiable.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .estimator import SpeedEstimate

_GREEN = (90, 220, 110)
_AMBER = (60, 190, 240)
_RED = (70, 70, 240)
_GREY = (170, 170, 170)
_WHITE = (245, 245, 245)


def _ttc_colour(ttc: float) -> tuple[int, int, int]:
    if not math.isfinite(ttc):
        return _GREEN
    if ttc < 2.0:
        return _RED
    if ttc < 4.0:
        return _AMBER
    return _GREEN


def draw_overlay(
    frame: np.ndarray,
    estimates: list[SpeedEstimate],
    lead: SpeedEstimate | None = None,
    truth: dict[str, float] | None = None,
) -> np.ndarray:
    out = frame.copy()

    for e in estimates:
        x, y, w, h = e.bbox
        is_lead = lead is not None and e.track_id == lead.track_id
        colour = _ttc_colour(e.ttc) if is_lead else _GREY
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2 if is_lead else 1)

        if e.plate is not None:
            px, py, pw, ph = e.plate.bbox
            cv2.rectangle(out, (px, py), (px + pw, py + ph), _AMBER, 1)
            cv2.putText(out, f"{e.plate.w_px:.1f}px", (px, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _AMBER, 1, cv2.LINE_AA)

        # "det" = the detector produced this box this frame; "trk" = registration moved
        # it here. Only a "det" box may anchor Channel A (MATH.md 5.4).
        label = f"#{e.track_id} {'det' if e.box_measured else 'trk'}"
        if not e.tracking_ok:
            label += "  [LOST]"
        elif e.calibrated and e.Zdot is not None:
            label += f"  {e.Z:.1f}m  {e.Zdot * 3.6:+.1f}km/h"
        else:
            label += "  [uncalibrated]"
        cv2.putText(out, label, (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    panel = _panel_lines(lead, truth)
    if panel:
        _draw_panel(out, panel)
    return out


def _panel_lines(lead: SpeedEstimate | None, truth: dict[str, float] | None) -> list[tuple[str, tuple[int, int, int]]]:
    if lead is None:
        return [("no lead vehicle", _GREY)]

    lines: list[tuple[str, tuple[int, int, int]]] = []
    ttc_txt = "inf" if not math.isfinite(lead.ttc) else f"{lead.ttc:.2f}s"
    lines.append((f"TTC      {ttc_txt}", _ttc_colour(lead.ttc)))

    if lead.calibrated and lead.Zdot is not None:
        lines.append((f"rel.spd  {lead.Zdot * 3.6:+7.2f} km/h  +-{lead.sigma_Zdot * 3.6:.2f}", _WHITE))
        lines.append((f"range    {lead.Z:7.2f} m      +-{lead.sigma_Z:.2f}", _WHITE))
    else:
        lines.append(("rel.spd  -- (no absolute anchor yet)", _GREY))
        lines.append((f"range    -- (prior only, lam={lead.lam:.2f})", _GREY))

    ch = f"A:{lead.channel_a:<5} B:{'yes' if lead.channel_b else 'no '}"
    lines.append((f"channels {ch}", _WHITE))
    kstate = "locked" if lead.kappa_locked else f"open n={lead.kappa_samples}"
    lines.append((f"kappa    {kstate}", _GREEN if lead.kappa_locked else _AMBER))
    if lead.scale is not None:
        m = lead.scale
        lines.append((f"scale    s={m.s:.4f} dt={m.dtau:.2f}s cc={m.confidence:.2f} [{m.backend}]", _GREY))

    if truth:
        lines.append(("--- ground truth ---", _GREY))
        if "Z" in truth:
            lines.append((f"true rng {truth['Z']:7.2f} m", _GREY))
        if "Zdot" in truth:
            lines.append((f"true spd {truth['Zdot'] * 3.6:+7.2f} km/h", _GREY))
    return lines


def _draw_panel(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    pad, lh = 10, 20
    width = 8 + max(len(t) for t, _c in lines) * 9
    height = pad * 2 + lh * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(img, text, (18, 10 + pad + lh * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
