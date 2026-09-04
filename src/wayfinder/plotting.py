"""Diagnostic plots: reconstructed track vs Suunto reference."""

from __future__ import annotations

import numpy as np

from .deadreckon import Track
from .validate import Comparison


def plot_comparison(track: Track, cmp: Comparison, *, title: str = "",
                    path: str | None = None):
    """Top-down XY (ours vs reference) + depth profile. Saves if ``path`` set."""
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    ref = cmp.ref_xy
    est = cmp.aligned_est
    ax1.plot(ref[:, 0], ref[:, 1], "-", color="#1f77b4", lw=1.6,
             label="Suunto DiveRoute")
    ax1.plot(est[:, 0], est[:, 1], "-", color="#d62728", lw=1.2, alpha=0.85,
             label="Wayfinder (aligned)")
    ax1.plot(*ref[0], "o", color="#1f77b4", ms=8)
    ax1.plot(*ref[-1], "s", color="#1f77b4", ms=8)
    ax1.plot(*est[0], "o", color="#d62728", ms=6)
    ax1.plot(*est[-1], "s", color="#d62728", ms=6)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Horizontal track (rigid-aligned)")

    tmin = (track.t - track.t[0]) / 60.0
    ax2.plot(tmin, track.depth, color="#2ca02c", lw=1.0)
    ax2.invert_yaxis()
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Depth (m)")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Depth profile")

    metrics = (f"ATE {cmp.ate_rmse:.1f} m   endpoint {cmp.endpoint_err:.1f} m   "
               f"drift {cmp.drift_rate:.2f} m/min   "
               f"path-len x{cmp.path_len_ratio:.2f}   DTW {cmp.dtw:.2f}")
    fig.suptitle((title + "\n" if title else "") + metrics, fontsize=11)
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path
    return fig
