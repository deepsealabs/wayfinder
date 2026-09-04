"""Compare a reconstructed track against Suunto's ``DiveRoute`` reference.

Suunto's frame is arbitrarily oriented and offset relative to ours (without a
trustworthy magnetometer we don't recover absolute North), and its absolute
accuracy is itself only ~±20-30 m. So we do **not** expect equality. We rigidly
align our track to the reference (rotation + translation, optionally scale) and
then measure how much shape/drift is left — that is the honest signal.

Metrics reported (horizontal / XY unless noted):
* ``ate_rmse`` — root-mean-square position error after alignment (m).
* ``endpoint_err`` — distance between the two end points after alignment (m).
* ``drift_rate`` — endpoint error per minute (m/min), the classic DR figure.
* ``path_len_ratio`` — our path length / reference path length (1.0 = same
  distance travelled; catches over/under-estimated speed).
* ``dtw`` — dynamic-time-warping distance, a sample-count/timing-robust shape
  similarity (m, normalized per matched pair).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .deadreckon import Track
from .io.suunto_json import ReferenceTrack


@dataclass
class Comparison:
    ate_rmse: float
    endpoint_err: float
    drift_rate: float
    path_len_ratio: float
    dtw: float
    frechet: float
    distance_error: float   # (our path length - DiveRouteDistance) / DiveRouteDistance
    scale: float
    n: int
    aligned_est: np.ndarray  # (M, 2) our XY after alignment
    ref_xy: np.ndarray       # (M, 2) reference XY, resampled to match
    t: np.ndarray            # (M,) normalized/resampled seconds

    def summary(self) -> dict:
        d = asdict(self)
        for k in ("aligned_est", "ref_xy", "t"):
            d.pop(k)
        return d


def _resample_xy(t: np.ndarray, xy: np.ndarray, m: int) -> np.ndarray:
    """Resample a 2D path onto ``m`` points evenly spaced in normalized time."""
    t = np.asarray(t, float)
    span = t[-1] - t[0]
    if span <= 0:
        return np.repeat(xy[:1], m, axis=0)
    u = (t - t[0]) / span
    grid = np.linspace(0.0, 1.0, m)
    return np.column_stack([np.interp(grid, u, xy[:, 0]),
                            np.interp(grid, u, xy[:, 1])])


def _procrustes_2d(src: np.ndarray, dst: np.ndarray, *, scale: bool):
    """Rigid (optionally similarity) fit mapping ``src`` onto ``dst``.

    Returns (aligned_src, s, R, t) minimizing ||s·R·src + t - dst||.
    """
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s0 = src - mu_s
    d0 = dst - mu_d
    h = s0.T @ d0
    u, sig, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:  # reflection guard
        vt[-1] *= -1
        sig[-1] *= -1
        r = vt.T @ u.T
    s = (sig.sum() / (s0 ** 2).sum()) if scale else 1.0
    t = mu_d - s * (r @ mu_s)
    aligned = (s * (r @ src.T)).T + t
    return aligned, s, r, t


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    """DTW distance between two 2D paths, normalized by path length (m)."""
    na, nb = len(a), len(b)
    cost = np.full((na + 1, nb + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, na + 1):
        ai = a[i - 1]
        for j in range(1, nb + 1):
            d = np.hypot(ai[0] - b[j - 1, 0], ai[1] - b[j - 1, 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1],
                                 cost[i - 1, j - 1])
    return float(cost[na, nb] / (na + nb))


def _frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Discrete Fréchet ("dog-leash") distance between two 2D paths (m).

    Sensitive to overall shape and ordering; the literature's preferred shape
    metric for movement trajectories. Iterative DP to avoid deep recursion.
    """
    na, nb = len(a), len(b)
    ca = np.full((na, nb), -1.0)
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    ca[0, 0] = d[0, 0]
    for i in range(1, na):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
    for j in range(1, nb):
        ca[0, j] = max(ca[0, j - 1], d[0, j])
    for i in range(1, na):
        for j in range(1, nb):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                           d[i, j])
    return float(ca[na - 1, nb - 1])


def compare(track: Track, ref: ReferenceTrack, *, scale: bool = False,
            m: int = 1000) -> Comparison:
    """Align ``track`` to ``ref`` and compute drift/shape metrics."""
    est = _resample_xy(track.t, track.xy, m)
    rxy = _resample_xy(ref.t, ref.xy, m)

    aligned, s, _r, _t = _procrustes_2d(est, rxy, scale=scale)

    diff = aligned - rxy
    ate = float(np.sqrt((diff ** 2).sum(axis=1).mean()))
    endpoint = float(np.hypot(*(aligned[-1] - rxy[-1])))

    duration_min = (ref.t[-1] - ref.t[0]) / 60.0
    drift_rate = endpoint / duration_min if duration_min > 0 else float("nan")

    est_len = np.sum(np.linalg.norm(np.diff(aligned, axis=0), axis=1))
    ref_len = np.sum(np.linalg.norm(np.diff(rxy, axis=0), axis=1))
    plr = float(est_len / ref_len) if ref_len > 0 else float("nan")

    # Shape metrics use O(N*M) DP; subsample to keep them fast without changing
    # the shape meaningfully.
    step = max(1, m // 250)
    a_s, r_s = aligned[::step], rxy[::step]
    dtw = _dtw(a_s, r_s)
    frechet = _frechet(a_s, r_s)

    if ref.route_distance:
        distance_error = (track.path_length - ref.route_distance) / ref.route_distance
    else:
        distance_error = float("nan")

    return Comparison(
        ate_rmse=ate, endpoint_err=endpoint, drift_rate=drift_rate,
        path_len_ratio=plr, dtw=dtw, frechet=frechet,
        distance_error=float(distance_error), scale=float(s), n=m,
        aligned_est=aligned, ref_xy=rxy,
        t=np.linspace(ref.t[0], ref.t[-1], m),
    )
