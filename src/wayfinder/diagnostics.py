"""Diagnostics for *why* a reconstruction does or doesn't match a reference.

The headline diagnostic is **directional agreement**: does our track's travel
direction actually track the reference's, or does it only look similar because
both are bounded blobs of the same scale? Scale-aligned DTW/Fréchet can't tell
those apart — this can. See docs/heading-ceiling.md for the finding that, on a
wrist-mounted IMU, travel direction is essentially *not* recoverable (this module
is how that was established, and how to re-check it on new data / sensors).
"""

from __future__ import annotations

import numpy as np


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def _travel_heading(xy: np.ndarray, w: int = 15):
    """Smoothed travel heading ψ (rad) and speed for a 2D path."""
    x = _smooth(xy[:, 0], w)
    y = _smooth(xy[:, 1], w)
    dx, dy = np.gradient(x), np.gradient(y)
    return np.unwrap(np.arctan2(dy, dx)), np.hypot(dx, dy)


def directional_agreement(est_xy: np.ndarray, ref_xy: np.ndarray,
                          *, smooth_w: int = 15,
                          move_frac: float = 0.3) -> float:
    """Mean cos(ψ_est − ψ_ref) over moving samples, in [-1, 1].

    +1 = travel directions coincide, 0 = unrelated (orthogonal on average),
    −1 = opposed. Both paths must already be resampled onto the same grid and in
    the same frame (e.g. after :func:`wayfinder.validate.compare`'s alignment).
    Restricted to samples where the reference is actually moving.
    """
    psi_e, _ = _travel_heading(est_xy, smooth_w)
    psi_r, spd = _travel_heading(ref_xy, smooth_w)
    m = spd > move_frac * np.median(spd)
    if not m.any():
        return float("nan")
    return float(np.mean(np.cos(psi_e[m] - psi_r[m])))


def series_reference_depth_corr(depth_series: np.ndarray, t_series: np.ndarray,
                                ref_depth: np.ndarray, ref_t: np.ndarray) -> float:
    """Correlation of our depth vs a reference depth (a pipeline positive control).

    Depth is measured by both the raw profile and Suunto's Z, so this should be
    ~1.0 if resampling/time-alignment is sound. Use it to sanity-check before
    trusting a *negative* horizontal result.
    """
    u = (t_series - t_series[0]) / (t_series[-1] - t_series[0])
    ur = (ref_t - ref_t[0]) / (ref_t[-1] - ref_t[0])
    ours = np.interp(ur, u, np.nan_to_num(depth_series))
    good = np.isfinite(ref_depth)
    return float(np.corrcoef(ours[good], ref_depth[good])[0, 1])
