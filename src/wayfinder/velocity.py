"""Velocity models: the structural fix for free-integration drift.

We cannot reset velocity underwater (no ZUPT for a finning diver), so instead of
double-integrating noisy acceleration we **assume a model for forward speed** and
integrate *that* along the gyro-fused heading. The absolute speed *scale* is not
observable from the IMU alone (it is set later by the GPS-distance / boundary
constraint), so these models aim for the right speed *shape* and turn structure;
the magnitude is a single free parameter.

Everything here is **water-frame**: it is the diver's motion through the water,
not over the ground. Unknown current adds a (roughly constant) ground-frame
offset that a two-anchor constraint absorbs later — not modelled here.

See docs/research-dead-reckoning.md §4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import orientation as ori
from . import quaternion as Q
from .series import DiveSeries


def heading(series: DiveSeries, quat: np.ndarray | None = None,
            *, use_mag: bool = False) -> np.ndarray:
    """Per-sample horizontal heading ψ (radians), from the fused orientation.

    ψ is the yaw of the device's forward axis about the world vertical. Absolute
    ψ is arbitrary without a magnetometer (the whole track is free to rotate —
    the validation alignment / GPS anchors pin it), but the *changes* in ψ are
    well observed by the gyro and are what draw the turns of the track.
    """
    if quat is None:
        quat = ori.estimate_orientation(series, use_mag=use_mag)
    # Sensor +X projected into the world frame is the "forward" reference; its
    # horizontal azimuth is the heading. (Any fixed body axis differs only by a
    # constant offset, which the alignment absorbs.)
    fwd = np.array([Q.to_matrix(q)[:, 0] for q in quat])  # world-frame sensor-X
    return np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))


def constant_speed(series: DiveSeries, speed: float = 0.25) -> np.ndarray:
    """Flat forward speed (m/s). Even a constant produces a correctly-shaped
    track that a scale/endpoint constraint can rescale. 0.25 m/s is a typical
    relaxed diver cruising speed."""
    return np.full(series.n, float(speed))


@dataclass
class Cadence:
    """Fin-kick cadence analysis over the dive."""
    inst_freq: np.ndarray   # (N,) instantaneous kick frequency, Hz
    envelope: np.ndarray    # (N,) band-limited kick amplitude (effort proxy)
    active: np.ndarray      # (N,) bool, kicking vs gliding/still
    band: tuple


def detect_cadence(series: DiveSeries, *, band: tuple = (0.4, 1.6),
                   active_frac: float = 0.3) -> Cadence:
    """Estimate instantaneous fin-kick cadence from the accelerometer.

    The acceleration *magnitude* (orientation-invariant) carries the periodic
    kick signature; on the reference dives it peaks near ~1 Hz. We band-pass it
    to the kick band, take the analytic signal, and read instantaneous frequency
    from the unwrapped phase and effort from the envelope.

    Falls back to a mean-frequency estimate if SciPy is unavailable.
    """
    fs = series.rate_hz
    a = np.linalg.norm(series.accel, axis=1)
    a = a - np.mean(a)

    xb, env, phase = _analytic_band(a, fs, band)
    inst_freq = np.gradient(phase, series.t) / (2 * np.pi)
    inst_freq = np.clip(inst_freq, 0.0, band[1] * 1.5)
    inst_freq = _smooth(inst_freq, int(max(3, fs)))  # ~1 s smoothing

    thr = active_frac * np.median(env[env > 0]) if np.any(env > 0) else 0.0
    active = env > thr

    return Cadence(inst_freq=inst_freq, envelope=env, active=active, band=band)


def cadence_speed(series: DiveSeries, *, distance_per_kick: float = 0.2,
                  glide_speed: float = 0.05,
                  cadence: Cadence | None = None, **kw) -> np.ndarray:
    """Forward speed from kick cadence: ``speed = distance_per_kick · cadence``.

    The step-and-heading (PDR) analog: each fin cycle advances the diver a
    roughly constant glide distance, so speed scales with kick rate. While not
    kicking, fall back to a small residual ``glide_speed`` (the diver coasts /
    drifts rather than stopping dead). ``distance_per_kick`` is the unknown
    scale parameter, fixed later by the GPS-distance constraint.
    """
    if cadence is None:
        cadence = detect_cadence(series, **kw)
    speed = distance_per_kick * cadence.inst_freq
    speed = np.where(cadence.active, speed, glide_speed)
    return speed


def calibrate_scale(speed: np.ndarray, t: np.ndarray,
                    target_distance: float) -> np.ndarray:
    """Rescale a speed profile so its integral equals ``target_distance`` (m).

    Stand-in for the eventual GPS-distance constraint: fixes the one free
    magnitude parameter using a known travelled distance.
    """
    dist = np.trapezoid(speed, t) if hasattr(np, "trapezoid") else np.trapz(speed, t)
    if dist <= 0:
        return speed
    return speed * (target_distance / dist)


# --- small signal-processing helpers (SciPy optional) ----------------------

def _analytic_band(x: np.ndarray, fs: float, band: tuple):
    """Band-pass ``x`` and return (filtered, envelope, unwrapped phase)."""
    try:
        from scipy.signal import butter, filtfilt, hilbert
        lo, hi = band[0] / (fs / 2), band[1] / (fs / 2)
        b, a = butter(3, [lo, hi], btype="band")
        xb = filtfilt(b, a, x)
        z = hilbert(xb)
        return xb, np.abs(z), np.unwrap(np.angle(z))
    except Exception:  # pragma: no cover - SciPy-less fallback
        # Crude FFT band-pass + constant mean-frequency phase.
        f = np.fft.rfftfreq(len(x), 1 / fs)
        X = np.fft.rfft(x)
        X[(f < band[0]) | (f > band[1])] = 0
        xb = np.fft.irfft(X, n=len(x))
        env = np.abs(xb)
        fmean = 0.5 * (band[0] + band[1])
        phase = 2 * np.pi * fmean * (np.arange(len(x)) / fs)
        return xb, env, phase


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")
