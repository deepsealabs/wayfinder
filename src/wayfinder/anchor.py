"""Boundary constraints: pin a water-frame track to known references.

The dead-reckoned track is water-frame and defined only up to a similarity
(unknown absolute position, heading, and speed scale — see
docs/research-dead-reckoning.md §6-7). Two kinds of external knowledge collapse
those free parameters:

* **Two positional anchors** (GPS entry + exit fixes) → an exact 2-point
  *similarity* transform (scale + rotation + translation) that rubber-sheets the
  whole track to start at A and end at B. This is the crude form of what Suunto
  reportedly does with its GPS bookends. NB: the current dive corpus has scrubbed
  GPS, so this is validated with stand-in anchors (e.g. the reference track's own
  endpoints) and is ready for dives with real surface fixes.

* **A known travelled distance** (Suunto's ``DiveRouteDistance``) → a pure scale
  fix that works on every dive, including loop dives where the two endpoints
  coincide and the 2-anchor similarity is ill-conditioned.

Both return a new :class:`~wayfinder.deadreckon.Track`; neither touches Z (depth
is already absolute).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .deadreckon import Track


def similarity_from_endpoints(p0: np.ndarray, pN: np.ndarray,
                              a: np.ndarray, b: np.ndarray):
    """Similarity (s, R, t) mapping p0->a and pN->b exactly in 2D.

    Four DOF (scale, rotation, 2D translation) fixed by the two point pairs.
    Returns (s, R(2x2), t(2,)). Raises if the source baseline p0->pN is
    degenerate (a loop dive), where scale/rotation are unobservable.
    """
    u = np.asarray(pN, float) - np.asarray(p0, float)
    v = np.asarray(b, float) - np.asarray(a, float)
    lu = np.linalg.norm(u)
    if lu < 1e-6:
        raise ValueError("degenerate baseline (start == end); use scale_to_distance")
    s = np.linalg.norm(v) / lu
    # Rotation aligning u with v.
    ang = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])
    c, sn = np.cos(ang), np.sin(ang)
    r = np.array([[c, -sn], [sn, c]])
    t = np.asarray(a, float) - s * (r @ np.asarray(p0, float))
    return s, r, t


def apply_anchors(track: Track, a, b) -> Track:
    """Rubber-sheet ``track`` so its first XY -> ``a`` and last XY -> ``b``.

    ``a``/``b`` are 2D points in the target frame (e.g. metres from a local
    origin). Z is preserved. Emulates a two-GPS-fix boundary correction.
    """
    xy = track.xy
    s, r, t = similarity_from_endpoints(xy[0], xy[-1], a, b)
    new_xy = (s * (r @ xy.T)).T + t
    pos = track.xyz.copy()
    pos[:, :2] = new_xy
    vel = track.velocity.copy()
    vel[:, :2] = (s * (r @ vel[:, :2].T)).T  # rotate+scale planar velocity
    return replace(track, xyz=pos, velocity=vel,
                   meta={**track.meta, "anchored": True, "anchor_scale": float(s)})


def scale_to_distance(track: Track, target_distance: float) -> Track:
    """Uniformly scale the horizontal track so its path length == target (m).

    Uses an authoritative travelled distance (Suunto ``DiveRouteDistance``, or a
    GPS-derived swim distance) to fix the velocity model's one free scale
    parameter. Works on every dive, loop or point-to-point.
    """
    xy = track.xy
    length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    if length <= 0 or target_distance <= 0:
        return track
    s = target_distance / length
    pos = track.xyz.copy()
    origin = xy[0]
    pos[:, :2] = origin + (xy - origin) * s
    vel = track.velocity.copy()
    vel[:, :2] = vel[:, :2] * s
    return replace(track, xyz=pos, velocity=vel,
                   meta={**track.meta, "scaled_to_distance": float(target_distance),
                         "scale_factor": float(s)})
