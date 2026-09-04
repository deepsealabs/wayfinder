"""Orientation / dead-reckoning / validation tests on synthetic + real data."""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from wayfinder import (DiveSeries, dead_reckon, dead_reckon_model,
                       estimate_orientation, compare, velocity,
                       apply_anchors, scale_to_distance)
from wayfinder import quaternion as Q
from wayfinder.anchor import similarity_from_endpoints
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


def _demo_track(n=100):
    s = _still_series(n=n)
    # Give it a curved path via a constant-speed model with a slow turn.
    t = np.arange(n) / 10.0
    gyro = np.zeros((n, 3))
    gyro[:, 2] = 20.0  # 20 deg/s yaw -> curves
    s = DiveSeries(t, s.accel, gyro, s.mag, s.depth, name="demo")
    return dead_reckon_model(s, speed="constant", speed_kw={"speed": 0.5},
                             vertical="integrate")


def test_similarity_from_endpoints_exact():
    p0, pN = np.array([0.0, 0.0]), np.array([2.0, 0.0])
    a, b = np.array([10.0, 10.0]), np.array([10.0, 16.0])  # scale 3, rot 90deg
    s, r, t = similarity_from_endpoints(p0, pN, a, b)
    assert s == pytest.approx(3.0)
    assert np.allclose(s * (r @ p0) + t, a)
    assert np.allclose(s * (r @ pN) + t, b)


def test_apply_anchors_pins_endpoints():
    track = _demo_track()
    a, b = np.array([5.0, -3.0]), np.array([20.0, 8.0])
    out = apply_anchors(track, a, b)
    assert np.allclose(out.xy[0], a)
    assert np.allclose(out.xy[-1], b)
    assert out.xyz[:, 2].tolist() == track.xyz[:, 2].tolist()  # Z untouched


def test_apply_anchors_rejects_loop():
    track = _demo_track()
    track.xyz[-1, :2] = track.xyz[0, :2]  # force start == end
    with pytest.raises(ValueError):
        apply_anchors(track, np.zeros(2), np.array([10.0, 0.0]))


def test_scale_to_distance_hits_target():
    track = _demo_track()
    out = scale_to_distance(track, 42.0)
    assert out.path_length == pytest.approx(42.0, rel=1e-6)


def test_directional_agreement_bounds():
    from wayfinder.diagnostics import directional_agreement
    # Identical paths -> agreement ~ +1.
    t = np.linspace(0, 1, 200)
    path = np.column_stack([np.cos(2 * t), np.sin(2 * t)])
    assert directional_agreement(path, path) == pytest.approx(1.0, abs=1e-6)
    # Same positions but every step negated -> travel heading opposed -> ~ -1.
    opposed = 2 * path[0] - path
    assert directional_agreement(opposed, path) < -0.9


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
