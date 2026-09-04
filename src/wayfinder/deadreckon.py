"""Strapdown dead-reckoning: IMU (+ depth) -> a relative X/Y/Z track.

This is the deliberately-naive baseline the milestone asks for. Mechanization:

1. Estimate orientation q_i (sensor->world) with the complementary filter.
2. Rotate specific force into the world frame: f_world = R_i · a_i.
3. Remove gravity: at rest f_world ≈ (0, 0, +g), so linear accel = f_world - g·ẑ.
4. Double-integrate: v += a·dt, p += v·dt.

Free double-integration of accelerometer error grows ~t², so the raw track
drifts fast; this module exposes the standard mitigations so we can *measure*
that drift and see how far cheap fixes get us:

* ``velocity_leak`` — bleed a fraction of velocity each second (a crude
  damping / high-pass that keeps position bounded).
* ``zupt`` — zero-velocity update: when the diver is momentarily still
  (near-1 g accel, near-0 gyro) force velocity to zero, killing accumulated
  drift. The finning analogue of pedestrian foot-contact ZUPT.
* ``vertical='depth'`` — trust the measured depth for Z instead of integrating
  vertical acceleration (depth is a real sensor; use it).

None of this makes the absolute position trustworthy — that needs a velocity
model and/or GPS boundary constraints (see docs/research). It gives a shape to
compare against Suunto's track and a drift number to beat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import orientation as ori
from . import quaternion as Q
from .series import GRAVITY, DiveSeries


@dataclass
class Track:
    """A reconstructed dive path in a local world frame (metres)."""

    t: np.ndarray
    xyz: np.ndarray          # (N, 3), Z positive up
    velocity: np.ndarray     # (N, 3) m/s, world frame (diagnostic)
    quat: np.ndarray         # (N, 4) sensor->world (diagnostic)
    meta: dict

    @property
    def xy(self) -> np.ndarray:
        return self.xyz[:, :2]

    @property
    def depth(self) -> np.ndarray:
        """Depth positive down (= -Z)."""
        return -self.xyz[:, 2]

    @property
    def path_length(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1)))


def dead_reckon(
    series: DiveSeries,
    *,
    quat: np.ndarray | None = None,
    use_mag: bool = False,
    velocity_leak: float = 0.5,
    zupt: bool = True,
    zupt_gyro_dps: float = 5.0,
    zupt_accel_g: float = 0.06,
    vertical: str = "depth",
) -> Track:
    """Dead-reckon a track from a :class:`DiveSeries`.

    Parameters
    ----------
    quat : precomputed (N,4) orientations; if None they are estimated here.
    use_mag : passed to orientation estimation when ``quat`` is None.
    velocity_leak : fraction of velocity removed per second (0 = pure
        integration, 1 = fully damped). Applied as ``v *= (1-leak)^dt``.
    zupt : enable zero-velocity updates on detected still samples.
    zupt_gyro_dps, zupt_accel_g : stillness thresholds — gyro magnitude below
        ``zupt_gyro_dps`` deg/s and |accel|-1 g within ``zupt_accel_g``.
    vertical : ``'depth'`` sets Z from the measured depth channel;
        ``'integrate'`` double-integrates vertical acceleration like X/Y.
    """
    if quat is None:
        quat = ori.estimate_orientation(series, use_mag=use_mag)

    t = series.t
    accel_ms2 = series.accel_ms2
    gyro_mag = np.linalg.norm(series.gyro, axis=1)
    accel_g_mag = np.linalg.norm(series.accel, axis=1)
    g_vec = np.array([0.0, 0.0, GRAVITY])

    n = series.n
    vel = np.zeros((n, 3))
    pos = np.zeros((n, 3))
    lin_world = np.zeros((n, 3))

    n_zupt = 0
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0 or dt > 1.0:
            dt = 1.0 / max(series.rate_hz, 1.0)

        r = Q.to_matrix(quat[i])
        f_world = r @ accel_ms2[i]
        a_lin = f_world - g_vec
        lin_world[i] = a_lin

        v = vel[i - 1] + a_lin * dt
        v *= (1.0 - velocity_leak) ** dt  # damping / drift bleed

        still = (zupt and gyro_mag[i] < zupt_gyro_dps
                 and abs(accel_g_mag[i] - 1.0) < zupt_accel_g)
        if still:
            v[:] = 0.0
            n_zupt += 1

        vel[i] = v
        pos[i] = pos[i - 1] + 0.5 * (v + vel[i - 1]) * dt  # trapezoidal

    if vertical == "depth":
        z = -np.asarray(series.depth, float)
        # Fill NaN (outside depth coverage) by holding the nearest valid value.
        z = _fill_nan(z)
        pos[:, 2] = z - z[0]
        vel[:, 2] = np.gradient(pos[:, 2], t)

    return Track(
        t=t.copy(),
        xyz=pos,
        velocity=vel,
        quat=quat,
        meta={
            "velocity_leak": velocity_leak,
            "zupt": zupt,
            "n_zupt": n_zupt,
            "vertical": vertical,
            "use_mag": use_mag and series.has_mag,
        },
    )


def dead_reckon_model(
    series: DiveSeries,
    *,
    quat: np.ndarray | None = None,
    use_mag: bool = False,
    speed: np.ndarray | float | str = "cadence",
    vertical: str = "depth",
    speed_kw: dict | None = None,
) -> Track:
    """Model-based (water-frame) dead reckoning: heading × a speed model.

    Instead of double-integrating acceleration (which diverges), integrate an
    assumed forward speed along the gyro-fused heading::

        p(t) = ∫ speed(t) · [cos ψ(t), sin ψ(t)] dt

    This is the real v1 (research §4 / Phase 2): the track shape comes from the
    well-observed *turn structure* (heading changes) and a speed model, not from
    fragile linear acceleration. The result is **water-frame** (current not
    removed) and its absolute scale is a free parameter (see
    :func:`wayfinder.velocity.calibrate_scale`).

    Parameters
    ----------
    speed : an (N,) speed array (m/s), a constant float, or a model name
        (``'cadence'`` or ``'constant'``).
    speed_kw : extra kwargs forwarded to the chosen speed model.
    vertical : ``'depth'`` sets Z from the depth channel (recommended).
    """
    from . import velocity as vel

    if quat is None:
        quat = ori.estimate_orientation(series, use_mag=use_mag)
    psi = vel.heading(series, quat)

    speed_kw = speed_kw or {}
    if isinstance(speed, str):
        if speed == "cadence":
            spd = vel.cadence_speed(series, **speed_kw)
        elif speed == "constant":
            spd = vel.constant_speed(series, **speed_kw)
        else:
            raise ValueError(f"unknown speed model {speed!r}")
    elif np.isscalar(speed):
        spd = vel.constant_speed(series, float(speed))
    else:
        spd = np.asarray(speed, float)

    t = series.t
    vx = spd * np.cos(psi)
    vy = spd * np.sin(psi)
    vel_w = np.zeros((series.n, 3))
    vel_w[:, 0] = vx
    vel_w[:, 1] = vy

    pos = np.zeros((series.n, 3))
    dt = np.diff(t)
    pos[1:, 0] = np.cumsum(0.5 * (vx[1:] + vx[:-1]) * dt)
    pos[1:, 1] = np.cumsum(0.5 * (vy[1:] + vy[:-1]) * dt)

    if vertical == "depth":
        z = _fill_nan(-np.asarray(series.depth, float))
        pos[:, 2] = z - z[0]
        vel_w[:, 2] = np.gradient(pos[:, 2], t)

    return Track(
        t=t.copy(), xyz=pos, velocity=vel_w, quat=quat,
        meta={"method": "model", "speed": speed if isinstance(speed, str)
              else "array/const", "vertical": vertical,
              "use_mag": use_mag and series.has_mag,
              "mean_speed": float(np.mean(spd))},
    )


def _fill_nan(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).copy()
    mask = np.isfinite(a)
    if not mask.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~mask] = np.interp(idx[~mask], idx[mask], a[mask])
    return a
