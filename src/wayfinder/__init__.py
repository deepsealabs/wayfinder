"""Wayfinder: reconstruct an underwater dive track from raw dive-computer IMU.

Feed it a device-agnostic IMU + depth time series (:class:`DiveSeries`) and it
dead-reckons a relative X/Y/Z path. See the README and docs/research for the
method and its (large, expected) limitations.
"""

from __future__ import annotations

from .series import DiveSeries, GRAVITY
from .deadreckon import Track, dead_reckon, dead_reckon_model
from .orientation import estimate_orientation, estimate_gyro_bias
from .validate import Comparison, compare
from . import velocity
from .anchor import apply_anchors, scale_to_distance

__all__ = [
    "DiveSeries", "GRAVITY", "Track", "dead_reckon", "dead_reckon_model",
    "estimate_orientation", "estimate_gyro_bias", "Comparison", "compare",
    "velocity", "apply_anchors", "scale_to_distance",
]

__version__ = "0.1.0"
