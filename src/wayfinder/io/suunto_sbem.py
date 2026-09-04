"""Decode a Suunto Ocean/Nautic ``.bin`` dive profile into raw IMU + depth.

The ``.bin`` is an uncompressed ``SBEM0103`` TLV stream (the same bytes
libdivecomputer's ``suunto_nautic_parser`` walks). We reimplement the walk in
Python so Wayfinder has no native dependency: the research prototype only needs
the high-rate IMU (chunk 0x22/0x23) and depth (chunk 0x16) channels.

Chunk format: ``[id:1][len:1][payload:len]`` with a ``len == 255`` escape to a
following ``uint32`` length. Time is delta-encoded: every chunk except the
timeline base (0x01) begins with a signed int16 LE millisecond delta; the
running sum is the absolute sample time (there is no per-sample absolute clock
and no fixed rate, though IMU lands at ~10 Hz in practice).

Scales are the ones validated in the libdc-swift reverse-engineering work
(deepsealabs/libdc-swift#29): accel 1/4096 g, gyro 1/131 deg/s, mag raw counts.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..series import DiveSeries

MAGIC = b"SBEM0103"

# Chunk ids (see libdivecomputer suunto_nautic_parser.c).
CHUNK_TIMELINE_BASE = 0x01
CHUNK_EXTENDED_STATUS = 0x16  # variable length; depth = float32 LE at offset 2
CHUNK_IMU = 0x23             # 195-byte-status firmware (Nautic)
CHUNK_IMU_ALT = 0x22         # 141-byte-status firmware (Nautic S / Ocean)

# Chunk ids whose payload length is fixed; used to reject "ghost" chunks that
# heatshrink artifacts / desync can conjure (a real chunk with this id never
# has any other length). Mirrors suunto_nautic_sbem_fixed_length().
_FIXED_LENGTH = {0x08: 6, 0x0B: 20, 0x0E: 6, 0x14: 7, 0x17: 14}

# Validated raw -> physical scales.
ACCEL_SCALE_G = 1.0 / 4096.0
GYRO_SCALE_DPS = 1.0 / 131.0


@dataclass
class _RawImu:
    t_ms: list
    accel: list  # raw int16 counts (x, y, z)
    gyro: list
    mag: list


def _find_summary(data: bytes) -> int:
    """Offset of the /Summary SBEM section, or len(data) if none.

    The driver appends the uncompressed /Summary after the profile, so the
    buffer can hold two ``SBEM0103`` signatures. The profile is bounded by the
    second one.
    """
    idx = data.find(MAGIC, len(MAGIC))
    return idx if idx != -1 else len(data)


def _walk(data: bytes, size: int):
    """Yield ``(chunk_id, payload_bytes)`` over the TLV stream in ``data[:size]``.

    Resynchronizes byte-by-byte on a malformed / ghost header, matching the C
    parser so we decode the same chunks it does.
    """
    off = len(MAGIC)
    while off + 2 <= size:
        cid = data[off]
        length = data[off + 1]
        header = 2
        if length == 255:
            if off + 6 > size:
                off += 1
                continue
            length = struct.unpack_from("<I", data, off + 2)[0]
            header = 6

        fixed = _FIXED_LENGTH.get(cid, -1)
        if fixed >= 0 and fixed != length:
            off += 1  # ghost chunk: resync
            continue
        if off + header + length > size:
            off += 1
            continue

        yield cid, data[off + header:off + header + length]
        off += header + length


def parse_bin(path: str, name: str | None = None) -> DiveSeries:
    """Parse a Suunto ``.bin`` into a :class:`DiveSeries`.

    IMU is emitted at its native ~10 Hz; depth (chunk 0x16, ~1 Hz) is linearly
    interpolated onto the IMU timeline and held NaN outside its coverage.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if not data.startswith(MAGIC):
        raise ValueError(f"{path}: not an SBEM0103 stream")

    profile_size = _find_summary(data)

    t_ms = 0
    imu_t: list[int] = []
    imu_raw: list[tuple] = []  # 9x int16
    depth_t: list[int] = []
    depth_m: list[float] = []

    for cid, payload in _walk(data, profile_size):
        if cid != CHUNK_TIMELINE_BASE and len(payload) >= 2:
            t_ms += struct.unpack_from("<h", payload, 0)[0]

        if cid in (CHUNK_IMU, CHUNK_IMU_ALT) and len(payload) >= 24:
            # [timeDelta:2][algoTS:uint32][ax,ay,az,gx,gy,gz,mx,my,mz:int16]
            imu_t.append(t_ms)
            imu_raw.append(struct.unpack_from("<9h", payload, 6))
        elif cid == CHUNK_EXTENDED_STATUS and len(payload) >= 6:
            depth_t.append(t_ms)
            depth_m.append(struct.unpack_from("<f", payload, 2)[0])

    if not imu_t:
        raise ValueError(f"{path}: no IMU chunks (0x22/0x23) found")

    imu_t_arr = np.asarray(imu_t, float)
    raw = np.asarray(imu_raw, float)  # (N, 9)
    t = (imu_t_arr - imu_t_arr[0]) / 1000.0  # seconds from dive start
    t = _make_monotonic(t)  # delta-encoding leaves occasional dt==0 ties

    accel = raw[:, 0:3] * ACCEL_SCALE_G
    gyro = raw[:, 3:6] * GYRO_SCALE_DPS
    mag = raw[:, 6:9]  # raw counts

    depth = _interp_depth(imu_t_arr, np.asarray(depth_t, float),
                          np.asarray(depth_m, float))

    return DiveSeries(
        t=t, accel=accel, gyro=gyro, mag=mag, depth=depth,
        name=name, gyro_bias=None,
        meta={"source": "suunto_sbem", "path": path,
              "n_depth_raw": len(depth_t)},
    )


def _make_monotonic(t: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Force strictly increasing times, nudging ties forward by ``eps`` s.

    The millisecond delta-encoding produces a handful of dt==0 ties per dive
    (two IMU records at the same accumulated ms). Left alone they make finite
    differences blow up; nudging by 0.1 ms is far below the 100 ms sample step.
    """
    out = t.astype(float).copy()
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out


def _interp_depth(imu_t_ms: np.ndarray, depth_t_ms: np.ndarray,
                  depth_m: np.ndarray) -> np.ndarray:
    """Linear-interpolate sparse depth onto the IMU clock; NaN outside range."""
    out = np.full(len(imu_t_ms), np.nan)
    if len(depth_t_ms) == 0:
        return out
    inside = (imu_t_ms >= depth_t_ms[0]) & (imu_t_ms <= depth_t_ms[-1])
    out[inside] = np.interp(imu_t_ms[inside], depth_t_ms, depth_m)
    return out
