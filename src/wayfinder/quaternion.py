"""Minimal quaternion helpers (Hamilton convention, [w, x, y, z]).

A quaternion here represents the rotation from **sensor frame to world frame**:
``v_world = q * v_sensor * q^-1``. World frame is East-North-Up-ish (we don't
pin absolute North without magnetometer; the map is only defined up to a yaw).
"""

from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that v_world = R @ v_sensor."""
    w, x, y, z = normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate sensor-frame vector v into the world frame."""
    return to_matrix(q) @ np.asarray(v, float)


def from_accel_mag(accel: np.ndarray, mag: np.ndarray | None = None) -> np.ndarray:
    """Initial orientation from a gravity (and optional magnetic) reading.

    Uses accel as the up reference (TRIAD-style). If ``mag`` is given and usable
    it fixes heading; otherwise heading is arbitrary (yaw = 0).
    """
    a = np.asarray(accel, float)
    up = a / (np.linalg.norm(a) or 1.0)  # world +Z in sensor frame ≈ gravity dir

    if mag is not None and np.isfinite(mag).all() and np.linalg.norm(mag) > 0:
        m = np.asarray(mag, float)
        east = np.cross(m, up)
        ne = np.linalg.norm(east)
        if ne > 1e-6:
            east /= ne
            north = np.cross(up, east)
            # Rows map world axes expressed in sensor frame -> R_ws (world<-sensor).
            r_ws = np.array([east, north, up])
            return _matrix_to_quat(r_ws.T)

    # No heading reference: build any frame with `up` as Z.
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    east = np.cross(ref, up)
    east /= np.linalg.norm(east) or 1.0
    north = np.cross(up, east)
    r_ws = np.array([east, north, up])
    return _matrix_to_quat(r_ws.T)


def _matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrix (v_world = R v_sensor) -> quaternion [w,x,y,z]."""
    tr = np.trace(r)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return normalize(np.array([w, x, y, z]))
