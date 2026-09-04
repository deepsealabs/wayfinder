"""Orientation / dead-reckoning / validation tests on synthetic + real data."""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from wayfinder import (DiveSeries, dead_reckon, dead_reckon_model,
                       estimate_orientation, compare, velocity)
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


def test_constant_speed_model_straight_line():
    # Level, no rotation, constant speed -> a straight track of the right length.
    s = _still_series(n=100, hz=10.0)  # 10 s
    track = dead_reckon_model(s, speed="constant", speed_kw={"speed": 0.5},
                              vertical="integrate")
    assert track.path_length == pytest.approx(0.5 * 9.9, rel=0.05)  # v*t
    # Heading is constant -> path is straight (all points collinear).
    xy = track.xy
    span = np.ptp(xy, axis=0)
    assert min(span) < 1e-6  # zero extent on one axis


def test_calibrate_scale_hits_target():
    s = _still_series(n=200, hz=10.0)
    speed = velocity.constant_speed(s, 0.3)
    speed = velocity.calibrate_scale(speed, s.t, target_distance=42.0)
    dist = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(speed, s.t)
    assert dist == pytest.approx(42.0, rel=1e-6)


def test_cadence_detects_synthetic_kick():
    # Synthesize a 1 Hz "kick" in accel magnitude; detector should recover ~1 Hz.
    fs = 10.0
    t = np.arange(600) / fs
    kick = 1.0 + 0.3 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz on top of 1 g
    accel = np.column_stack([np.zeros_like(t), np.zeros_like(t), kick])
    s = DiveSeries(t, accel, np.zeros((len(t), 3)), np.full((len(t), 3), np.nan),
                   np.zeros_like(t), name="kick")
    cad = velocity.detect_cadence(s)
    assert 0.8 < np.median(cad.inst_freq[cad.active]) < 1.2


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
