"""Command-line entry point: reconstruct and (optionally) validate a dive.

    python -m wayfinder DIVE.bin [--ref DIVE.json] [--plot out.png] [options]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .io import parse_bin, load_reference
from .deadreckon import dead_reckon, dead_reckon_model
from .validate import compare


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("wayfinder", description=__doc__)
    p.add_argument("bin", help="Suunto .bin dive profile (SBEM0103)")
    p.add_argument("--ref", help="Suunto .json export for DiveRoute validation")
    p.add_argument("--plot", help="write a comparison PNG here")
    p.add_argument("--out", help="write the reconstructed track to a CSV here")
    p.add_argument("--method", choices=["model", "strapdown"], default="model",
                   help="'model' = heading × velocity model (default, bounded); "
                        "'strapdown' = double-integrate acceleration (diverges)")
    p.add_argument("--speed-model", choices=["cadence", "constant"],
                   default="cadence", help="velocity model (--method model)")
    p.add_argument("--speed", type=float, default=0.25,
                   help="forward speed m/s for --speed-model constant")
    p.add_argument("--distance-per-kick", type=float, default=0.6,
                   help="glide distance per fin kick, m (--speed-model cadence)")
    p.add_argument("--scale-to-distance", action="store_true",
                   help="scale the track to Suunto's DiveRouteDistance (the "
                        "authoritative travelled distance; needs --ref)")
    p.add_argument("--anchor-endpoints", action="store_true",
                   help="pin the track's start/end to the reference track's "
                        "endpoints (demonstrates the two-GPS-fix boundary "
                        "constraint; needs --ref)")
    p.add_argument("--calibrate-to-ref", action="store_true",
                   help="rescale model speed so path length matches the "
                        "reference track (older stand-in; prefer "
                        "--scale-to-distance)")
    p.add_argument("--mag", action="store_true",
                   help="fuse the (uncalibrated) magnetometer for heading; off "
                        "by default because it currently degrades results")
    p.add_argument("--vertical", choices=["depth", "integrate"], default="depth")
    p.add_argument("--velocity-leak", type=float, default=0.5,
                   help="strapdown drift damping (--method strapdown)")
    p.add_argument("--no-zupt", action="store_true",
                   help="disable ZUPT (--method strapdown)")
    p.add_argument("--scale-align", action="store_true",
                   help="allow scale in the alignment (similarity, not rigid); "
                        "the fairest shape-only comparison")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    name = os.path.splitext(os.path.basename(args.bin))[0]
    series = parse_bin(args.bin, name=name)
    print(f"[ingest] {series!r}", file=sys.stderr)

    ref = load_reference(args.ref, name=name) if args.ref else None

    if args.method == "strapdown":
        track = dead_reckon(
            series, use_mag=args.mag, velocity_leak=args.velocity_leak,
            zupt=not args.no_zupt, vertical=args.vertical,
        )
    else:
        track = _model_track(series, ref, args)

    track = _apply_boundary(track, ref, args)
    print(f"[track]  method={args.method} path length {track.path_length:.0f} m",
          file=sys.stderr)

    if args.out:
        _write_csv(args.out, track)
        print(f"[out]    {args.out}", file=sys.stderr)

    if ref is not None:
        cmp = compare(track, ref, scale=args.scale_align)
        print(json.dumps(cmp.summary(), indent=2))
        if args.plot:
            from .plotting import plot_comparison
            plot_comparison(track, cmp, title=name, path=args.plot)
            print(f"[plot]   {args.plot}", file=sys.stderr)
    elif args.plot:
        print("[plot]   --plot needs --ref (nothing to compare against)",
              file=sys.stderr)

    return 0


def _model_track(series, ref, args):
    """Build a model-method track, honouring the speed-model / calibration args."""
    from . import velocity as vel
    from .orientation import estimate_orientation

    quat = estimate_orientation(series, use_mag=args.mag)
    if args.speed_model == "constant":
        speed = vel.constant_speed(series, args.speed)
    else:
        speed = vel.cadence_speed(series, distance_per_kick=args.distance_per_kick)

    if args.calibrate_to_ref and ref is not None:
        ref_len = float(np.sum(np.linalg.norm(np.diff(ref.xy, axis=0), axis=1)))
        speed = vel.calibrate_scale(speed, series.t, ref_len)

    return dead_reckon_model(series, quat=quat, speed=speed,
                             vertical=args.vertical)


def _apply_boundary(track, ref, args):
    """Apply the requested boundary constraint(s) to the track."""
    from .anchor import apply_anchors, scale_to_distance

    if args.scale_to_distance:
        if ref is None or not ref.route_distance:
            print("[warn]   --scale-to-distance needs --ref with a "
                  "DiveRouteDistance", file=sys.stderr)
        else:
            track = scale_to_distance(track, ref.route_distance)
    if args.anchor_endpoints:
        if ref is None:
            print("[warn]   --anchor-endpoints needs --ref", file=sys.stderr)
        else:
            try:
                track = apply_anchors(track, ref.xy[0], ref.xy[-1])
            except ValueError as e:
                print(f"[warn]   anchoring skipped: {e}", file=sys.stderr)
    return track


def _write_csv(path: str, track) -> None:
    header = "t,x,y,z,vx,vy,vz"
    rows = np.column_stack([track.t, track.xyz, track.velocity])
    np.savetxt(path, rows, delimiter=",", header=header, comments="")


if __name__ == "__main__":
    raise SystemExit(main())
