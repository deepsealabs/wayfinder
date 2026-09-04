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

import numpy as np  # noqa: E402

from wayfinder.io import parse_bin, load_reference  # noqa: E402
from wayfinder import (dead_reckon, dead_reckon_model, compare,  # noqa: E402
                       estimate_orientation, velocity)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory of paired DIVE.bin / DIVE.json files")
    ap.add_argument("--plots", help="directory to write comparison PNGs into")
    ap.add_argument("--method", choices=["model", "strapdown"], default="model")
    ap.add_argument("--scale-to-distance", action="store_true",
                    help="scale each track to Suunto's DiveRouteDistance")
    ap.add_argument("--anchor-endpoints", action="store_true",
                    help="pin each track to the reference track endpoints")
    ap.add_argument("--calibrate-to-ref", action="store_true",
                    help="rescale model speed to the reference path length")
    ap.add_argument("--mag", action="store_true", help="fuse magnetometer")
    args = ap.parse_args()

    if args.plots:
        os.makedirs(args.plots, exist_ok=True)

    pairs = [(b, b[:-4] + ".json") for b in sorted(glob.glob(f"{args.dir}/*.bin"))
             if os.path.exists(b[:-4] + ".json")]
    if not pairs:
        print(f"no .bin/.json pairs in {args.dir}", file=sys.stderr)
        return 1

    hdr = (f"{'dive':14s} {'min':>4s} {'ATE':>7s} {'endpt':>7s} {'drift':>6s} "
           f"{'DTW':>6s} {'Fréch':>6s} {'distErr':>7s}")
    print(f"method={args.method}")
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
        track = _reconstruct(series, ref, args)
        c = compare(track, ref)
        de = f"{c.distance_error*100:+6.0f}%" if np.isfinite(c.distance_error) else "     --"
        print(f"{name:14s} {series.duration/60:4.0f} {c.ate_rmse:7.1f} "
              f"{c.endpoint_err:7.1f} {c.drift_rate:6.2f} "
              f"{c.dtw:6.2f} {c.frechet:6.1f} {de:>7s}")
        if args.plots:
            from wayfinder.plotting import plot_comparison
            plot_comparison(track, c, title=name,
                            path=os.path.join(args.plots, f"{name}.png"))
    return 0


def _reconstruct(series, ref, args):
    from wayfinder import apply_anchors, scale_to_distance

    if args.method == "strapdown":
        track = dead_reckon(series, use_mag=args.mag)
    else:
        quat = estimate_orientation(series, use_mag=args.mag)
        speed = velocity.cadence_speed(series)
        if args.calibrate_to_ref:
            ref_len = float(np.sum(np.linalg.norm(np.diff(ref.xy, axis=0), axis=1)))
            speed = velocity.calibrate_scale(speed, series.t, ref_len)
        track = dead_reckon_model(series, quat=quat, speed=speed)

    if args.scale_to_distance and ref.route_distance:
        track = scale_to_distance(track, ref.route_distance)
    if args.anchor_endpoints:
        try:
            track = apply_anchors(track, ref.xy[0], ref.xy[-1])
        except ValueError:
            pass
    return track


if __name__ == "__main__":
    raise SystemExit(main())
