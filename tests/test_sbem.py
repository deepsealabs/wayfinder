"""Ingest tests. The real dive fixtures live outside the repo; these tests skip
gracefully when they are absent so CI stays green, but run fully in dev."""

from __future__ import annotations

import glob
import os
import struct

import numpy as np
import pytest

from wayfinder.io import parse_bin
from wayfinder.io.suunto_sbem import _make_monotonic, _walk, MAGIC

FIXTURES = sorted(glob.glob(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "nautic-captures-backup", "*.bin")))


def test_make_monotonic():
    t = np.array([0.0, 0.1, 0.1, 0.1, 0.2])
    out = _make_monotonic(t)
    assert np.all(np.diff(out) > 0)
    assert out[-1] == pytest.approx(0.2)


def test_walk_simple_tlv():
    # magic + one chunk id=0x50 len=3 payload=aabbcc
    data = MAGIC + bytes([0x50, 0x03, 0xAA, 0xBB, 0xCC])
    chunks = list(_walk(data, len(data)))
    assert chunks == [(0x50, b"\xaa\xbb\xcc")]


def test_walk_255_escape():
    payload = b"\x01" * 300
    data = MAGIC + bytes([0x50, 0xFF]) + struct.pack("<I", 300) + payload
    chunks = list(_walk(data, len(data)))
    assert chunks == [(0x50, payload)]


@pytest.mark.skipif(not FIXTURES, reason="no dive fixtures available")
def test_parse_real_dive():
    series = parse_bin(FIXTURES[0])
    assert series.n > 1000
    assert 8.0 < series.rate_hz < 12.0            # ~10 Hz
    assert np.all(np.diff(series.t) > 0)          # strictly monotonic
    amag = np.linalg.norm(series.accel, axis=1)
    assert 0.8 < np.median(amag) < 1.2            # accel ≈ 1 g -> scale sane
    assert np.nanmax(series.depth) > 1.0          # a real dive has depth
