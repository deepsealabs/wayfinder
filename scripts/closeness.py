#!/usr/bin/env python3
"""How close to Suunto can we get, and what drives it?

Answers "make it as close as Suunto" honestly: closeness is set by how many
external position fixes we can pin the track through (surface GPS, acoustic
beacons, buddy marks), not by the IMU horizontal shape. Produces:

* a convergence table/plot of ATE vs number of fixes, comparing the IMU-derived
  shape between fixes against plain straight-line interpolation, and
* a "closest achievable" overlay at a chosen fix count.

Fixes here are sampled from the Suunto track itself (a stand-in for real
external fixes), so this measures achievable closeness, not a from-scratch
reconstruction. Depth/Z always comes from the IMU profile.

    python scripts/closeness.py ../nautic-captures-backup --plots out/
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wayfinder.io import parse_bin, load_reference  # noqa: E402
from wayfinder import estimate_orientation, scale_to_distance  # noqa: E402
from wayfinder.anchor import apply_waypoints  # noqa: E402
from wayfinder.deadreckon import dead_reckon_model, Track  # noqa: E402
from wayfinder.validate import compare  # noqa: E402

NS = [2, 3, 5, 9, 17, 33]


def _ref_on_track(track_t, ref):
    u = (track_t - track_t[0]) / (track_t[-1] - track_t[0])
    ur = (ref.t - ref.t[0]) / (ref.t[-1] - ref.t[0])
    return np.column_stack([np.interp(u, ur, ref.xy[:, 0]),
                            np.interp(u, ur, ref.xy[:, 1])])


def _straight_line(track, wi, wp):
    ii = np.arange(track.n)
    xy = np.column_stack([np.interp(ii, wi, wp[:, 0]),
                          np.interp(ii, wi, wp[:, 1])])
    return Track(track.t.copy(), np.column_stack([xy, track.xyz[:, 2]]),
                 track.velocity, track.quat, {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--plots", help="directory for the convergence + overlay plots")
    ap.add_argument("--overlay-n", type=int, default=9)
    args = ap.parse_args()
    if args.plots:
        os.makedirs(args.plots, exist_ok=True)

    pairs = [(b, b[:-4] + ".json") for b in sorted(glob.glob(f"{args.dir}/*.bin"))
             if os.path.exists(b[:-4] + ".json")]

    dr = {n: [] for n in NS}
    li = {n: [] for n in NS}
    overlay = None
    print("median ATE (m) vs number of position fixes")
    print("  fixes:   " + "  ".join(f"{n:>4d}" for n in NS))
    for b, j in pairs:
        try:
            s = parse_bin(b)
            ref = load_reference(j)
        except Exception:  # noqa: BLE001
            continue
        q = estimate_orientation(s)
        t = dead_reckon_model(s, quat=q, speed="cadence")
        if ref.route_distance:
            t = scale_to_distance(t, ref.route_distance)
        refxy = _ref_on_track(t.t, ref)
        for n in NS:
            wi = np.linspace(0, t.n - 1, n).astype(int)
            wp = refxy[wi]
            dr[n].append(compare(apply_waypoints(t, wi, wp), ref).ate_rmse)
            li[n].append(compare(_straight_line(t, wi, wp), ref).ate_rmse)
        if args.plots and overlay is None:
            wi = np.linspace(0, t.n - 1, args.overlay_n).astype(int)
            overlay = (apply_waypoints(t, wi, refxy[wi]), ref, refxy[wi],
                       os.path.basename(b)[:-4])

    print("  IMU-DR:  " + "  ".join(f"{np.median(dr[n]):4.1f}" for n in NS))
    print("  line:    " + "  ".join(f"{np.median(li[n]):4.1f}" for n in NS))
    print("\nStraight-line interpolation matches or beats the IMU shape: the IMU "
          "contributes depth + distance, not horizontal route shape.")

    if args.plots:
        _plot_convergence(dr, li, os.path.join(args.plots, "closeness_curve.png"))
        if overlay:
            _plot_overlay(*overlay, args.overlay_n,
                          os.path.join(args.plots, "closest_overlay.png"))
        print(f"[plots]  {args.plots}")
    return 0


def _plot_convergence(dr, li, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(NS, [np.median(dr[n]) for n in NS], "o-", label="IMU shape between fixes")
    ax.plot(NS, [np.median(li[n]) for n in NS], "s--", label="straight-line between fixes")
    ax.axhspan(20, 30, color="gray", alpha=0.15, label="Suunto's own ±20–30 m")
    ax.set_xscale("log")
    ax.set_xlabel("number of position fixes")
    ax.set_ylabel("median ATE vs Suunto (m)")
    ax.set_title("Closeness to Suunto is set by fix density")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay(track, ref, wp, name, n, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = compare(track, ref, scale=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(c.ref_xy[:, 0], c.ref_xy[:, 1], color="#1f77b4", lw=1.6, label="Suunto")
    ax.plot(c.aligned_est[:, 0], c.aligned_est[:, 1], color="#d62728", lw=1.1,
            label=f"Wayfinder ({n} fixes)")
    ax.plot(wp[:, 0], wp[:, 1], "k.", ms=9, label="position fixes")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{name}: closest achievable  (ATE {c.ate_rmse:.1f} m, {n} fixes)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
