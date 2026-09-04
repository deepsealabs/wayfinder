"""Attitude estimation from the 9-axis IMU (Mahony complementary filter).

The attitude layer is the *solved* part of the problem: a complementary filter
tracks orientation to ~a degree by fusing the integrated gyro (accurate short
term, drifts long term) with the gravity direction from the accelerometer
(noisy short term, absolute long term) and, optionally, the magnetometer for
heading. We use Mahony because it is cheap, stable, and exposes an explicit
integral term that also *estimates* residual gyro bias online.

Convention: the returned quaternions rotate sensor -> world (see
:mod:`wayfinder.quaternion`).
"""

from __future__ import annotations

import numpy as np

from . import quaternion as Q
from .series import DiveSeries


def estimate_gyro_bias(series: DiveSeries, still_frac: float = 0.1) -> np.ndarray:
    """Estimate gyro bias (deg/s) from the stillest stretch of the dive.

    We pick the samples whose accel magnitude is closest to 1 g *and* whose
    gyro magnitude is smallest (the diver is momentarily still), and average the
    gyro there. This is the pragmatic stand-in for the watch's stored
    ``DiveRouteGyroBias`` calibration, which the current exports don't expose.
    """
    a_mag = np.linalg.norm(series.accel, axis=1)
    g_mag = np.linalg.norm(series.gyro, axis=1)
    # Rank stillness: penalize deviation from 1 g and any rotation.
    score = np.abs(a_mag - 1.0) + 0.05 * g_mag
    k = max(10, int(still_frac * series.n))
    idx = np.argsort(score)[:k]
    return series.gyro[idx].mean(axis=0)


def estimate_orientation(
    series: DiveSeries,
    *,
    use_mag: bool = False,
    kp: float = 1.0,
    ki: float = 0.1,
    gyro_bias: np.ndarray | None = None,
) -> np.ndarray:
    """Run the Mahony filter over the whole dive.

    Parameters
    ----------
    use_mag : fuse the magnetometer for absolute heading. **Off by default:** the
        raw magnetometer is uncalibrated (no hard/soft-iron correction) and,
        empirically, feeding it in degrades heading versus gyro-only integration
        on every reference dive — steel tanks and the watch's own fields distort
        it. Turn it on once magnetometer calibration lands. With it off, heading
        is only constrained by gyro integration and drifts slowly (the alignment
        step absorbs the unknown absolute heading anyway).
    kp, ki : proportional / integral feedback gains (rad/s per unit error).
    gyro_bias : deg/s bias to subtract up front. If None, uses
        ``series.gyro_bias`` or estimates it from still samples.

    Returns
    -------
    (N, 4) array of sensor->world quaternions.
    """
    if gyro_bias is None:
        gyro_bias = (series.gyro_bias if series.gyro_bias is not None
                     else estimate_gyro_bias(series))
    gyro_bias_rad = np.deg2rad(np.asarray(gyro_bias, float))

    have_mag = use_mag and series.has_mag
    gyro = series.gyro_rad - gyro_bias_rad
    accel = series.accel  # g; only direction matters
    mag = series.mag
    t = series.t

    # Seed from the first stable accel/mag reading.
    q = Q.from_accel_mag(accel[0], mag[0] if have_mag else None)

    out = np.empty((series.n, 4))
    out[0] = q
    integral = np.zeros(3)  # Mahony bias-integral term (rad/s)

    for i in range(1, series.n):
        dt = t[i] - t[i - 1]
        if dt <= 0 or dt > 1.0:
            dt = 1.0 / max(series.rate_hz, 1.0)

        w = gyro[i].copy()
        a = accel[i]
        a_norm = np.linalg.norm(a)

        err = np.zeros(3)
        if a_norm > 1e-6:
            a_hat = a / a_norm
            # Gravity direction in the sensor frame, from current estimate:
            # world +Z (up) is R^T @ [0,0,1] -> last row of R.
            r = Q.to_matrix(q)
            v_grav = r[2, :]  # sensor-frame estimate of "up"
            err += np.cross(a_hat, v_grav)

        if have_mag and np.isfinite(mag[i]).all():
            m = mag[i]
            m_norm = np.linalg.norm(m)
            if m_norm > 1e-6:
                m_hat = m / m_norm
                r = Q.to_matrix(q)
                # Reference field in world frame (flatten to remove dip so mag
                # only corrects heading), then back to sensor frame.
                h = r @ m_hat
                b = np.array([0.0, np.hypot(h[0], h[1]), h[2]])
                v_mag = r.T @ b
                err += np.cross(m_hat, v_mag)

        if ki > 0:
            integral += err * dt
        w = w + kp * err + ki * integral

        # Integrate quaternion: q_dot = 0.5 * q ⊗ (0, w).
        q_dot = 0.5 * Q.mult(q, np.array([0.0, w[0], w[1], w[2]]))
        q = Q.normalize(q + q_dot * dt)
        out[i] = q

    return out
