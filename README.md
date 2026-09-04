# Wayfinder

Reconstruct underwater dive routes from raw dive-computer IMU, by dead reckoning.

Underwater there's no GPS, so a dive computer that draws your path does it with
inertial navigation: integrating accelerometer, gyroscope and magnetometer
readings over time to estimate where you moved. Suunto's Ocean and Nautic
watches log that raw IMU at ~10 Hz and their app computes the X/Y/Z track after
the fact. That computed track lives only in the app's export, never in the data
the watch hands out. Wayfinder is an open reimplementation of that step: give it
the raw IMU and it gives you back a track.

## Status

Experimental / research. This estimates a track; it will not be identical to
Suunto's, whose exact sensor-fusion filter and tuning are proprietary. Expect
drift and treat the output as an approximation.

## What it does

- **Input:** a time series of 3-axis accelerometer, gyroscope, and magnetometer
  (plus depth), such as the vendor IMU samples libdivecomputer / libdc-swift
  decode from a Suunto Ocean/Nautic dive.
- **Output:** a relative X/Y/Z path over the dive, plus whatever quality / drift
  metrics we can derive.

## Where the data comes from

Wayfinder does not talk to dive computers. Getting the raw IMU off the watch is
[libdc-swift](https://github.com/deepsealabs/libdc-swift)'s job (see the Suunto
work in deepsealabs/libdc-swift#29). Wayfinder is the sibling step that turns
that IMU into a track. Feed it the IMU series and it stays device-agnostic.

## Recommended stack

Python + numpy for the research phase: sensor fusion, filter design and
drift-correction experiments iterate fastest there, and there's good tooling for
comparing an estimated track against the Suunto app's exported track. Once the
approach is settled, a port to Rust or C (with bindings) is the path to running
it inside apps.

## Why

So the dive path isn't locked to one vendor's app. Any tool that can pull raw
IMU off a computer can then show a route.

## Install

```bash
pip install -e .            # numpy only; add ".[plot]" for the comparison plots
```

## Quickstart

Reconstruct a track from a Suunto `.bin` profile and validate it against the
app's own `DiveRoute` in the matching `.json` export:

```bash
python -m wayfinder DIVE.bin --ref DIVE.json --plot out.png
# or, as a library:
```

```python
from wayfinder.io import parse_bin, load_reference
from wayfinder import dead_reckon_model, compare

series = parse_bin("DIVE.bin")          # -> device-agnostic IMU + depth series
track  = dead_reckon_model(series)      # -> relative X/Y/Z track (velocity model)
cmp    = compare(track, load_reference("DIVE.json"))
print(cmp.summary())                    # ATE, endpoint, drift, DTW, Fréchet ...
```

The ingest (`parse_bin`) is the only Suunto-specific part; everything downstream
works on a canonical [`DiveSeries`](src/wayfinder/series.py) (accel in g, gyro in
deg/s, mag raw, depth in m), so a new device just needs a new loader.

## How it works

Two ingredients are shared by both methods:

1. **Orientation** — a Mahony complementary filter fuses gyro (short-term) with
   the gravity direction from accel (long-term), with online gyro-bias estimation
   ([`orientation.py`](src/wayfinder/orientation.py)).
2. **Depth for Z** — vertical comes straight from the depth sensor, never from
   integrating vertical acceleration.

There are two ways to get the horizontal track ([`deadreckon.py`](src/wayfinder/deadreckon.py)):

- **`--method model`** (default) — the **velocity model**: integrate an assumed
  forward speed along the gyro-fused heading, `p += speed·[cosψ, sinψ]·dt`
  ([`velocity.py`](src/wayfinder/velocity.py)). Speed is either constant or from
  **fin-kick cadence** (the accel magnitude peaks near ~1 Hz; each kick advances a
  glide distance — the PDR "stride × step-rate" analog). This is **water-frame**
  and its absolute scale is a single free parameter (set later by the GPS-distance
  / boundary constraint). It doesn't diverge and needs no drift-suppression hacks.
- **`--method strapdown`** — the naive baseline: rotate specific force to the world
  frame, subtract gravity, double-integrate. Kept as a diagnostic; it diverges
  (see below) and needs ZUPT + velocity-leak damping to stay bounded.

**Validation** rigidly aligns to Suunto's track and reports drift + shape
similarity — ATE, endpoint, drift-rate, path-length ratio, DTW, Fréchet
([`validate.py`](src/wayfinder/validate.py)).

### What the reference dives show

All matching the literature survey in
[`docs/research-dead-reckoning.md`](docs/research-dead-reckoning.md):

- **Free double-integration is hopeless** — the strapdown drifts to *hundreds of
  km* without damping (the classic t²/t³ error growth). The horizontal velocity
  model, not attitude, is the real problem.
- **The velocity model fixes the distance.** Across 10 dives it gives a consistent
  path-length ratio of ~1.0× (vs the strapdown's chaotic 0.3–6.6×) from one
  physical scale parameter, and a better median shape score (DTW 10.6 → 9.0) — with
  no ZUPT/leak tuning. Endpoint error ~10–48 m, drift ~0.3–1.5 m/min after
  alignment.
- **The uncalibrated magnetometer hurts.** Steel tanks and the watch's own fields
  distort it; gyro-only heading beats it on every dive, so mag fusion is off by
  default until hard/soft-iron calibration lands.
- **Travel direction is not recoverable from a wrist IMU** — the hard ceiling,
  documented in [`docs/heading-ceiling.md`](docs/heading-ceiling.md). The device's
  yaw is *decoupled* from where the diver swims (the wrist rotates freely), and
  neither any geometric signal nor a cross-validated learned model correlates with
  Suunto's travel heading (turn-rate corr ≈ 0), while a depth positive control is
  +0.99. So `compare` now reports **`heading_agreement`** (mean cos of travel-angle
  error, ~0 here) — the honest metric that scale-aligned DTW/ATE hide. Depth,
  distance, and cadence are recoverable; the route's *shape* is not, from this
  sensor. Breaking it needs a body-mounted sensor or GPS-truth motion labels, not
  a better filter or mag calibration.
- **Two endpoint anchors ≈ knowing the whole track.** Pinning the water-frame
  track to just its two end fixes (a similarity rubber-sheet, [`anchor.py`](src/wayfinder/anchor.py))
  lands within ~6% ATE of the *oracle* full-trajectory alignment that needs the
  entire Suunto track (median 25.1 m vs 23.6 m). That is what real GPS surface
  bookends buy you — and it's the constraint Suunto reportedly uses.

### What Suunto data we validate against

- **`DiveRoute` X/Y/Z** — the computed track; the target for all shape/drift metrics.
- **`DiveRouteDistance`** — Suunto's authoritative total distance; we scale the
  velocity model to it (`--scale-to-distance`) and report a `distance_error`. The
  cadence model's *natural* distance is a consistent ~3× off (right relative
  effort, one global scale), which this pins exactly.
- **GPS surface fixes are unusable in this corpus** — they're scrubbed to a
  0.2–0.5 m spread while the dive covers up to 44 m, and most dives only carry
  exit fixes. So the two-anchor constraint is *built and validated with stand-in
  anchors* (the reference endpoints), ready for dives with real GPS bookends.
- `Speed`/`Cadence`/`Distance` per-sample fields are null for diving.

Suunto's own track is only accurate to ±20–30 m absolutely, so we treat it as a
shape reference, not ground truth.

## Roadmap

See [`docs/research-dead-reckoning.md`](docs/research-dead-reckoning.md) for the
full survey. Build order:

1. ✅ Naive strapdown baseline + validation harness.
2. ✅ Water-frame **velocity model** (constant / kick-cadence speed along fused
   heading) — fixes the over-travel; now the default method.
3. ✅ **Boundary constraints** — the two-anchor similarity rubber-sheet
   (`--anchor-endpoints`) and `DiveRouteDistance` scaling (`--scale-to-distance`).
   The mechanism is ready; validating it on *real* GPS bookends needs dives whose
   exports keep unscrubbed surface fixes.
4. ⚠️ **Wrist-heading ceiling — investigated, appears fundamental**
   ([`docs/heading-ceiling.md`](docs/heading-ceiling.md)): travel direction isn't
   recoverable from the wrist IMU (geometric *or* learned), so route *shape* can't
   be reconstructed from this sensor. Ways forward: a body-mounted sensor (re-run
   `scripts/heading_probe.py` to test), a deep model trained on GPS-truth-labelled
   dives, or shipping the depth + distance + cadence products with the route shown
   only as an approximate scribble that snaps to real GPS bookends.
