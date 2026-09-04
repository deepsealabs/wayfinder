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
from .deadreckon import dead_reckon
from .validate import compare


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("wayfinder", description=__doc__)
    p.add_argument("bin", help="Suunto .bin dive profile (SBEM0103)")
    p.add_argument("--ref", help="Suunto .json export for DiveRoute validation")
    p.add_argument("--plot", help="write a comparison PNG here")
    p.add_argument("--out", help="write the reconstructed track to a CSV here")
    p.add_argument("--mag", action="store_true",
                   help="fuse the (uncalibrated) magnetometer for heading; off "
                        "by default because it currently degrades results")
    p.add_argument("--vertical", choices=["depth", "integrate"], default="depth")
    p.add_argument("--velocity-leak", type=float, default=0.5)
    p.add_argument("--no-zupt", action="store_true")
    p.add_argument("--scale-align", action="store_true",
                   help="allow scale in the alignment (similarity, not rigid)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    name = os.path.splitext(os.path.basename(args.bin))[0]
    series = parse_bin(args.bin, name=name)
    print(f"[ingest] {series!r}", file=sys.stderr)

    track = dead_reckon(
        series,
        use_mag=args.mag,
        velocity_leak=args.velocity_leak,
        zupt=not args.no_zupt,
        vertical=args.vertical,
    )
    print(f"[track]  path length {track.path_length:.0f} m, "
          f"ZUPTs {track.meta['n_zupt']}", file=sys.stderr)

    if args.out:
        _write_csv(args.out, track)
        print(f"[out]    {args.out}", file=sys.stderr)

    if args.ref:
        ref = load_reference(args.ref, name=name)
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


def _write_csv(path: str, track) -> None:
    header = "t,x,y,z,vx,vy,vz"
    rows = np.column_stack([track.t, track.xyz, track.velocity])
    np.savetxt(path, rows, delimiter=",", header=header, comments="")


if __name__ == "__main__":
    raise SystemExit(main())
