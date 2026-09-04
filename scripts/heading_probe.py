#!/usr/bin/env python3
"""Can a learned model recover travel direction from the wrist IMU?

Reproduces the key negative result behind docs/heading-ceiling.md: predict the
Suunto track's *turn rate* from IMU-window features, with **cross-dive**
(GroupKFold) validation, and report the correlation. Near-zero means the turn
structure is not recoverable from this sensor even by learning. A positive
control (depth) confirms the pipeline is sound.

    python scripts/heading_probe.py ../nautic-captures-backup

Needs scikit-learn + scipy (`pip install -e ".[dev]"`).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wayfinder.io import parse_bin, load_reference  # noqa: E402
from wayfinder import estimate_orientation  # noqa: E402
from wayfinder.diagnostics import (_smooth, _travel_heading,  # noqa: E402
                                   series_reference_depth_corr)


def _features_at(series, tt, half=1.0):
    i0 = np.searchsorted(series.t, tt - half)
    i1 = np.searchsorted(series.t, tt + half)
    if i1 - i0 < 3:
        return None
    a, g = series.accel[i0:i1], series.gyro[i0:i1]
    f = []
    for arr in (a[:, 0], a[:, 1], a[:, 2], g[:, 0], g[:, 1], g[:, 2]):
        f += [arr.mean(), arr.std()]
    f += [np.linalg.norm(a, axis=1).mean(), np.linalg.norm(g, axis=1).mean(),
          g[:, 2].mean()]
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    args = ap.parse_args()

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import GroupKFold
        from scipy.stats import pearsonr
    except ImportError:
        print("needs scikit-learn + scipy: pip install -e '.[dev]'")
        return 1

    X, Y, G = [], [], []
    depth_ctrl = []
    pairs = [(b, b[:-4] + ".json") for b in sorted(glob.glob(f"{args.dir}/*.bin"))
             if os.path.exists(b[:-4] + ".json")]
    for gi, (b, j) in enumerate(pairs):
        try:
            s = parse_bin(b)
            ref = load_reference(j)
        except Exception:  # noqa: BLE001
            continue
        estimate_orientation(s)  # (kept for parity; features are frame-free)
        _psi, spd = _travel_heading(ref.xy)
        psi, _ = _travel_heading(ref.xy)
        turn = np.gradient(psi, ref.t)
        depth_ctrl.append(series_reference_depth_corr(s.depth, s.t, ref.depth, ref.t))
        for k in range(len(ref.t)):
            if spd[k] < 0.3 * np.median(spd) or not np.isfinite(turn[k]):
                continue
            f = _features_at(s, ref.t[k])
            if f is None:
                continue
            X.append(f); Y.append(turn[k]); G.append(gi)

    X, Y, G = np.array(X), np.array(Y), np.array(G)
    ndives = len(set(G.tolist()))
    print(f"samples {len(X)}  dives {ndives}")
    print(f"positive control  depth corr (median over dives) = "
          f"{np.median(depth_ctrl):+.3f}")
    if ndives < 4:
        print("need >=4 dives for cross-dive CV")
        return 0

    gkf = GroupKFold(n_splits=min(5, ndives))
    preds = np.zeros_like(Y)
    for tr, te in gkf.split(X, Y, G):
        m = HistGradientBoostingRegressor(max_iter=200, max_depth=4)
        m.fit(X[tr], Y[tr])
        preds[te] = m.predict(X[te])
    r, _ = pearsonr(preds, Y)
    rb, _ = pearsonr(X[:, -1], Y)  # device gyro-z mean vs Suunto turn
    print(f"LEARNED turn-rate prediction (cross-dive)  corr = {r:+.3f}")
    print(f"baseline device gyro-z vs Suunto turn      corr = {rb:+.3f}")
    print("\n~0 => travel direction is not recoverable from the wrist IMU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
