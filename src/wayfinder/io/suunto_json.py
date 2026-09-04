"""Load the Suunto app's computed ``DiveRoute`` from a ``.json`` export.

This is the *validation target*, not an input: the watch does not store a
track; the Suunto app dead-reckons it from the raw IMU with a proprietary
learned/fused model and writes the X/Y/Z result into its export. We compare our
own reconstruction against it (as a shape reference — Suunto's own absolute
accuracy is only ~±20-30 m).

Track convention (observed): X/Y are horizontal metres in an app-local frame,
Z is depth in metres, negative down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceTrack:
    """Suunto's computed dive route, resampled to its own sample times.

    t : (M,) seconds from the first route sample.
    xyz : (M, 3) metres; columns X, Y, Z (Z negative down).
    origin : optional (lat, lon, alt) of DiveRouteOrigin if present (often 0s).
    """

    t: np.ndarray
    xyz: np.ndarray
    origin: tuple | None = None
    name: str | None = None

    @property
    def xy(self) -> np.ndarray:
        return self.xyz[:, :2]

    @property
    def depth(self) -> np.ndarray:
        """Depth (positive down), i.e. -Z."""
        return -self.xyz[:, 2]


def _parse_iso_seconds(stamps: list[str]) -> np.ndarray:
    """Seconds-from-start for ISO8601 stamps, tolerant of the +01:00 offset.

    We only need relative time, so parse HH:MM:SS.mmm and unwrap across any
    (rare) midnight/day boundary without pulling in a tz library.
    """
    from datetime import datetime

    def to_dt(s: str) -> datetime:
        # Normalize "+01:00" -> "+0100" for %z.
        if len(s) >= 6 and s[-3] == ":" and s[-6] in "+-":
            s = s[:-3] + s[-2:]
        # Fractional seconds are present on some firmware, absent on others.
        fmt = "%Y-%m-%dT%H:%M:%S.%f%z" if "." in s else "%Y-%m-%dT%H:%M:%S%z"
        return datetime.strptime(s, fmt)

    dts = [to_dt(s) for s in stamps]
    t0 = dts[0]
    return np.array([(d - t0).total_seconds() for d in dts], float)


def load_reference(path: str, name: str | None = None) -> ReferenceTrack:
    """Extract the ``DiveRoute`` X/Y/Z series (and origin) from a dive JSON."""
    with open(path) as fh:
        doc = json.load(fh)
    samples = doc["DeviceLog"]["Samples"]

    ts: list[str] = []
    xyz: list[tuple] = []
    origin = None
    for s in samples:
        if "DiveRouteOrigin" in s:
            o = s["DiveRouteOrigin"]
            origin = (o.get("Latitude"), o.get("Longitude"), o.get("Altitude"))
        route = s.get("DiveRoute")
        if route is None:
            continue
        ts.append(s["TimeISO8601"])
        xyz.append((route["X"], route["Y"], route["Z"]))

    if not xyz:
        raise ValueError(f"{path}: no DiveRoute samples found")

    return ReferenceTrack(
        t=_parse_iso_seconds(ts),
        xyz=np.asarray(xyz, float),
        origin=origin,
        name=name,
    )
