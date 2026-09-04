#!/usr/bin/env python3
"""Two-anchor boundary constraint with REAL GPS (independent of Suunto).

Uses the genuine surface GPS fixes preserved in the raw .bin (entry cluster +
exit cluster) as the two anchors, warps the IMU track to hit them, and compares
to Suunto's DiveRoute (also pinned to the same two real fixes) plus a
straight-line baseline. Non-circular: the anchors are measured GPS, not sampled
from Suunto's output.

    python scripts/real_gps_anchor.py ../nautic-captures-backup/1787991757.bin

Finding on the corpus: even with two real anchors the IMU shape between them
does not match Suunto and is worse than a straight line -- the wrist-heading
ceiling (docs/heading-ceiling.md), now confirmed with real anchors.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wayfinder.io import parse_bin, load_reference  # noqa: E402
from wayfinder import estimate_orientation, gps as G  # noqa: E402
from wayfinder.anchor import apply_anchors  # noqa: E402
from wayfinder.deadreckon import dead_reckon_model, Track  # noqa: E402


def _resample(xy, m=1000):
    u = np.linspace(0, 1, len(xy))
    g = np.linspace(0, 1, m)
    return np.column_stack([np.interp(g, u, xy[:, 0]), np.interp(g, u, xy[:, 1])])


def _ate(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("--plot")
    args = ap.parse_args()

    s = parse_bin(args.bin)
    ref = load_reference(args.bin[:-4] + ".json")
    fx = s.meta.get("gps", [])
    if len(fx) < 2:
        print("dive has < 2 GPS fixes; need entry+exit")
        return 1
    ft, fxy, _ = G.to_enu(fx)
    ei, xi = int(np.argmin(ft)), int(np.argmax(ft))
    A, B = fxy[ei], fxy[xi]
    frac_entry = ft[ei] / s.duration
    frac_exit = ft[xi] / s.duration
    print(f"entry fix @ {frac_entry:.0%} of dive, exit @ {frac_exit:.0%}, "
          f"entry→exit {np.linalg.norm(B - A):.1f} m")
    if frac_entry > 0.15:
        print("WARNING: no true entry fix (all fixes late) — 2-anchor is degenerate")

    q = estimate_orientation(s)
    t = dead_reckon_model(s, quat=q, speed="cadence")
    ours = apply_anchors(t, A, B)

    reftrack = Track(np.linspace(t.t[0], t.t[-1], len(ref.xy)),
                     np.column_stack([ref.xy, -ref.depth]),
                     np.zeros((len(ref.xy), 3)), None, {})
    sun = apply_anchors(reftrack, A, B)
    line = np.column_stack([np.linspace(A[0], B[0], t.n),
                            np.linspace(A[1], B[1], t.n)])

    o, su, li = _resample(ours.xy), _resample(sun.xy), _resample(line)
    print(f"ATE  IMU-shape vs Suunto      = {_ate(o, su):.1f} m")
    print(f"ATE  straight-line vs Suunto  = {_ate(li, su):.1f} m")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(su[:, 0], su[:, 1], color="#1f77b4", lw=1.6, label="Suunto")
        ax.plot(o[:, 0], o[:, 1], color="#d62728", lw=1.1,
                label=f"Wayfinder (2 real anchors) ATE {_ate(o, su):.0f}m")
        ax.plot(li[:, 0], li[:, 1], "--", color="gray",
                label=f"straight line ATE {_ate(li, su):.0f}m")
        ax.plot(*A, "go", ms=12, label="real entry GPS")
        ax.plot(*B, "ks", ms=12, label="real exit GPS")
        ax.set_aspect("equal")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        fig.savefig(args.plot, dpi=130, bbox_inches="tight")
        print(f"[plot] {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
