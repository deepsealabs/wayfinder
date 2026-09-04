# The wrist-heading ceiling

**Finding: on a wrist-mounted dive computer, horizontal *travel direction* is
essentially not recoverable from the IMU — geometrically or by a simple learned
model. Depth and travelled distance are recoverable; the route's turn structure
is not.** This document records the evidence, because it determines what
Wayfinder can and cannot do on this class of data and redirects effort away from
a goal that is impossible for this sensor placement.

## Why we looked

The velocity model (Phase 2) integrates an assumed speed along the gyro-fused
*heading*. Its metrics looked reasonable (scale-aligned DTW ~9, ATE ~20 m), but
the reconstructed tracks were visibly more "curled" than Suunto's larger
excursions. Scale-aligned DTW/Fréchet/ATE can't distinguish a directionally
correct track from a bounded random walk of the same size — so we tested
direction directly.

## Evidence (all reproducible)

Reference "truth" for direction is Suunto's own `DiveRoute` travel heading
(itself only a shape reference, ±20–30 m). We compare *relative* structure (turn
rate / directional agreement), which is invariant to each dive's arbitrary frame.

1. **Geometric device heading vs Suunto travel heading** — turn-rate correlation
   ≈ **0** (+0.01, +0.01, −0.03, −0.02 across dives). The device's yaw does not
   track the direction of travel.
2. **Every candidate geometric direction** — the three sensor axes projected to
   horizontal, the low-passed horizontal linear-acceleration direction, and the
   strapdown velocity direction — all give directional agreement `cos(Δψ)` ≈ **0**
   (|value| ≤ 0.09). None encodes travel direction.
3. **A learned model** — HistGradientBoosting regressing Suunto turn rate from
   IMU-window features (per-axis accel/gyro mean+std, magnitudes, device yaw
   rate), with **cross-dive** GroupKFold validation — correlation **−0.012**. The
   device gyro-z ("obvious" turn signal) baseline: **−0.015**. Nothing.
4. **Positive control** — our depth (from the `.bin`) vs Suunto's Z (from the
   `.json`) correlates **+0.90 to +0.99**. The resampling / time-alignment
   pipeline is sound, so the horizontal nulls are real, not an artifact.

Reproduce: `python scripts/heading_probe.py ../nautic-captures-backup`, and the
`heading_agreement` field now reported by `wayfinder.validate.compare`.

## Why this happens

- **The wrist is not the body.** A recreational diver's wrist rotates
  independently of where they swim — arms streamline, gesture, adjust gear, hold
  the computer up to read it. There is no fixed offset between wrist orientation
  and travel direction to calibrate out (that would show as a *constant* heading
  offset with high turn-rate correlation; we see zero correlation instead).
- **Yaw is unobservable without a heading reference**, and the magnetometer
  doesn't rescue it: even a perfectly calibrated mag gives the *device's* yaw,
  which we've shown is decoupled from travel. So mag calibration is **not** the
  fix for the horizontal track (it would only help if the wrist tracked the body).
- **Suunto's own track may be unrecoverable from IMU by construction.** Suunto
  computes the route with a learned model *plus a GPS entry/exit boundary
  constraint*. If the shape is substantially determined by that boundary
  optimisation and a motion prior rather than by instantaneous IMU-derived
  heading, then no function of the IMU alone can reproduce it — consistent with
  the learned probe finding nothing.

## Implications for the roadmap

What Wayfinder *can* deliver honestly from a wrist IMU:

- **Depth / Z profile** — excellent (positive control +0.99).
- **Travelled distance** — recoverable with an external scale reference
  (`DiveRouteDistance` or a GPS-derived swim distance).
- **Effort / kick cadence over time** — the fin-kick signal is real (~1 Hz).
- **A bounded, correctly-scaled water-frame scribble** — but its *direction* is
  not trustworthy; do not present it as a route.

Paths that could break the ceiling, in rough order of leverage:

1. **A body-mounted sensor** (mask, tank, torso) where orientation tracks travel
   — then geometric heading likely returns. The cheapest scientific test:
   record a dive with an IMU fixed to the torso and re-run `heading_probe.py`.
2. **Deep sequence model trained on GPS-truth-labelled dives** (RoNIN/TLIO
   style, raw-IMU sequences, not hand-crafted features), where the labels come
   from real surface-GPS-bounded tracks — not Suunto's output. Only worth it if a
   body-mounted or otherwise directionally-informative signal exists; a deep net
   can't learn a mapping that carries no information (the probe suggests the wrist
   signal may genuinely lack it).
3. **Accept the ceiling** and ship the depth + distance + cadence products, with
   the route shown as an approximate water-frame scribble that snaps to real GPS
   bookends when available (Phase 3).

The honest headline: **direction is the wall, and on a wrist it may be a real
one.** Better attitude filters, mag calibration, or fancier geometry will not
move it; only a directionally-informative sensor or true motion labels can.
