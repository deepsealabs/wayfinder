"""Orientation / dead-reckoning / validation tests on synthetic + real data."""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from wayfinder import DiveSeries, dead_reckon, estimate_orientation, compare
from wayfinder import quaternion as Q
from wayfinder.io import parse_bin, load_reference

HERE = os.path.dirname(__file__)
BINS = sorted(glob.glob(os.path.join(HERE, "..", "..",
                                     "nautic-captures-backup", "*.bin")))


def _still_series(n=200, hz=10.0):
    """A perfectly still, level dive: accel = +1 g up, no rotation."""
    t = np.arange(n) / hz
    accel = np.tile([0.0, 0.0, 1.0], (n, 1))
    gyro = np.zeros((n, 3))
    mag = np.tile([1.0, 0.0, 0.0], (n, 1))  # arbitrary but constant heading
    depth = np.zeros(n)
    return DiveSeries(t, accel, gyro, mag, depth, name="still")


def test_quaternion_roundtrip():
    q = Q.normalize([0.5, 0.5, -0.5, 0.5])
    r = Q.to_matrix(q)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_orientation_level_is_stable():
    s = _still_series()
    q = estimate_orientation(s)
    up_world = np.array([Q.rotate(qi, [0, 0, 1]) for qi in q])
    # Sensor +Z should map to world +Z throughout (level, upright).
    assert np.allclose(up_world[:, 2], 1.0, atol=1e-2)


def test_still_dive_barely_moves():
    s = _still_series()
    track = dead_reckon(s, vertical="integrate")
    # No real motion -> horizontal drift stays tiny over 20 s.
    assert track.path_length < 1.0


@pytest.mark.skipif(not BINS, reason="no dive fixtures available")
def test_real_dive_end_to_end():
    name = os.path.splitext(os.path.basename(BINS[0]))[0]
    series = parse_bin(BINS[0], name=name)
    track = dead_reckon(series)
    assert track.xyz.shape == (series.n, 3)
    assert np.isfinite(track.xyz).all()
    # Depth-constrained Z should match the measured depth channel.
    good = np.isfinite(series.depth)
    assert np.allclose(track.depth[good], series.depth[good] - series.depth[good][0],
                       atol=1e-6)

    ref_path = BINS[0].replace(".bin", ".json")
    if os.path.exists(ref_path):
        ref = load_reference(ref_path, name=name)
        cmp = compare(track, ref)
        assert cmp.ate_rmse > 0
        assert np.isfinite(cmp.dtw)
        assert cmp.aligned_est.shape == cmp.ref_xy.shape
