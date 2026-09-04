"""Real surface GPS fixes → local metric anchors.

The raw ``.bin`` download preserves genuine surface GPS (the ``.json`` export
scrubs it to a fake near-null point). These fixes are *independent* of Suunto's
computed track, so — unlike sampling the DiveRoute — anchoring to them is a real
reconstruction, not a circular one. Most dives carry only exit-cluster fixes;
some (e.g. the 79-fix dive) have them throughout and give true multi-waypoint
constraints.
"""

from __future__ import annotations

import numpy as np

EARTH_R = 6371000.0


def to_enu(fixes: list[dict], lat0: float | None = None,
           lon0: float | None = None):
    """Equirectangular lat/lon → local (east, north) metres.

    Returns (t, xy, (lat0, lon0)) where ``t`` are the fix times (s, relative to
    the dive start, as populated by the SBEM parser) and ``xy`` is (K, 2) metres
    from the reference point (first fix by default). Good to <1 m over the
    hundred-metre scale of a dive.
    """
    if not fixes:
        return np.empty(0), np.empty((0, 2)), (None, None)
    lat = np.array([f["lat"] for f in fixes], float)
    lon = np.array([f["lon"] for f in fixes], float)
    t = np.array([f.get("t", f["t_ms"] / 1000.0) for f in fixes], float)
    if lat0 is None:
        lat0 = float(lat[0])
    if lon0 is None:
        lon0 = float(lon[0])
    east = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    north = np.radians(lat - lat0) * EARTH_R
    return t, np.column_stack([east, north]), (lat0, lon0)


def fix_track_indices(track_t: np.ndarray, fix_t: np.ndarray) -> np.ndarray:
    """Nearest track-sample index for each fix time (for apply_waypoints)."""
    return np.clip(np.searchsorted(track_t, fix_t), 0, len(track_t) - 1)
