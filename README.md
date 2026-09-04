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
- **Heading from a wrist sensor is the current ceiling.** The wrist rotates
  independently of travel direction and gyro yaw drifts, so the track's *turn
  structure* is only roughly right — the next thing to improve.

Suunto's own track is only accurate to ±20–30 m absolutely, so we treat it as a
shape reference, not ground truth.

## Roadmap

See [`docs/research-dead-reckoning.md`](docs/research-dead-reckoning.md) for the
full survey. Build order:

1. ✅ Naive strapdown baseline + validation harness.
2. ✅ Water-frame **velocity model** (constant / kick-cadence speed along fused
   heading) — fixes the over-travel; now the default method.
3. **Two-anchor GPS boundary constraint** (rubber-sheet the track between the
   entry/exit surface fixes, solving for heading bias + constant current) — the
   biggest win per line of code, and what Suunto reportedly does. Also sets the
   speed scale, removing the `--calibrate-to-ref` stand-in.
4. Magnetometer calibration + better travel-direction estimate; optionally a
   learned velocity regressor (RoNIN/TLIO style).
