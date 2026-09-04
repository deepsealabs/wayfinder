"""Canonical, device-agnostic dive time series.

Everything downstream of ingest works on a :class:`DiveSeries`: a regularly (or
near-regularly) sampled IMU + depth record in SI-ish units, plus an optional
reference track for validation. Ingesting a new device means writing a loader
that produces one of these; the estimator never sees vendor byte layouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Standard gravity, m/s^2. Used to convert accel from g and to build the
# gravity vector when removing it in the world frame.
GRAVITY = 9.80665


@dataclass
class DiveSeries:
    """A single dive's inertial + depth record.

    Attributes
    ----------
    t : (N,) float seconds, monotonic, starting near 0.
    accel : (N, 3) float, specific force in **g** (sensor frame, X/Y/Z).
    gyro : (N, 3) float, angular rate in **deg/s** (sensor frame).
    mag : (N, 3) float, magnetometer in raw counts (scale unknown; direction
        only). May be all-NaN if a device has no usable magnetometer.
    depth : (N,) float metres, positive down, interpolated onto ``t``. NaN
        before the first / after the last raw depth reading.
    name : optional identifier (e.g. the dive id).
    gyro_bias : optional (3,) deg/s bias to subtract before integrating; None
        means "estimate it" (see :func:`wayfinder.orientation.estimate_gyro_bias`).
    """

    t: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray
    mag: np.ndarray
    depth: np.ndarray
    name: str | None = None
    gyro_bias: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, float)
        self.accel = np.asarray(self.accel, float).reshape(-1, 3)
        self.gyro = np.asarray(self.gyro, float).reshape(-1, 3)
        self.mag = np.asarray(self.mag, float).reshape(-1, 3)
        self.depth = np.asarray(self.depth, float)
        n = len(self.t)
        for name_, arr in (("accel", self.accel), ("gyro", self.gyro),
                           ("mag", self.mag)):
            if len(arr) != n:
                raise ValueError(f"{name_} has {len(arr)} rows, expected {n}")
        if len(self.depth) != n:
            raise ValueError(f"depth has {len(self.depth)} rows, expected {n}")
        if self.gyro_bias is not None:
            self.gyro_bias = np.asarray(self.gyro_bias, float).reshape(3)

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.n else 0.0

    @property
    def rate_hz(self) -> float:
        """Median sample rate, Hz."""
        if self.n < 2:
            return float("nan")
        return 1.0 / float(np.median(np.diff(self.t)))

    @property
    def accel_ms2(self) -> np.ndarray:
        """Specific force in m/s^2."""
        return self.accel * GRAVITY

    @property
    def gyro_rad(self) -> np.ndarray:
        """Angular rate in rad/s."""
        return np.deg2rad(self.gyro)

    @property
    def has_mag(self) -> bool:
        return bool(np.isfinite(self.mag).any())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"DiveSeries(name={self.name!r}, n={self.n}, "
                f"{self.duration / 60:.1f} min @ {self.rate_hz:.1f} Hz, "
                f"depth<= {np.nanmax(self.depth):.1f} m)")
