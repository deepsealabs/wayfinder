#!/usr/bin/env python3
"""Reconstruct + validate every dive in a directory of Suunto .bin/.json pairs.

Prints a drift/shape metrics table and (optionally) writes a comparison PNG per
dive. This is the quick "how are we doing across the corpus" harness.

    python scripts/demo.py ../nautic-captures-backup --plots out/
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wayfinder.io import parse_bin, load_reference  # noqa: E402
from wayfinder import dead_reckon, compare  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory of paired DIVE.bin / DIVE.json files")
    ap.add_argument("--plots", help="directory to write comparison PNGs into")
    ap.add_argument("--mag", action="store_true", help="fuse magnetometer")
    args = ap.parse_args()

    if args.plots:
        os.makedirs(args.plots, exist_ok=True)

    pairs = [(b, b[:-4] + ".json") for b in sorted(glob.glob(f"{args.dir}/*.bin"))
             if os.path.exists(b[:-4] + ".json")]
    if not pairs:
        print(f"no .bin/.json pairs in {args.dir}", file=sys.stderr)
        return 1

    hdr = f"{'dive':14s} {'min':>4s} {'ATE':>7s} {'endpt':>7s} {'drift':>6s} {'plen×':>6s} {'DTW':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for b, j in pairs:
        name = os.path.splitext(os.path.basename(b))[0]
        try:
            series = parse_bin(b, name=name)
            ref = load_reference(j, name=name)
        except Exception as e:  # noqa: BLE001 - corpus has a few odd exports
            print(f"{name:14s}  skip: {e}")
            continue
        track = dead_reckon(series, use_mag=args.mag)
        c = compare(track, ref)
        print(f"{name:14s} {series.duration/60:4.0f} {c.ate_rmse:7.1f} "
              f"{c.endpoint_err:7.1f} {c.drift_rate:6.2f} "
              f"{c.path_len_ratio:6.2f} {c.dtw:6.2f}")
        if args.plots:
            from wayfinder.plotting import plot_comparison
            plot_comparison(track, c, title=name,
                            path=os.path.join(args.plots, f"{name}.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
