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
from wayfinder import dead_reckon, compare

series = parse_bin("DIVE.bin")          # -> device-agnostic IMU + depth series
track  = dead_reckon(series)            # -> relative X/Y/Z track
cmp    = compare(track, load_reference("DIVE.json"))
print(cmp.summary())                    # ATE, endpoint error, drift rate, DTW ...
```

The ingest (`parse_bin`) is the only Suunto-specific part; everything downstream
works on a canonical [`DiveSeries`](src/wayfinder/series.py) (accel in g, gyro in
deg/s, mag raw, depth in m), so a new device just needs a new loader.

## How it works (v1)

A deliberately simple **strapdown baseline**, so we can measure the drift before
adding cleverness:

1. **Orientation** — a Mahony complementary filter fuses gyro (short-term) with
   the gravity direction from accel (long-term), with online gyro-bias estimation
   ([`orientation.py`](src/wayfinder/orientation.py)).
2. **Dead reckoning** — rotate specific force into the world frame, subtract
   gravity, and double-integrate to position ([`deadreckon.py`](src/wayfinder/deadreckon.py)).
3. **Depth for Z** — vertical comes straight from the depth sensor, never from
   integrating vertical acceleration.
4. **Validation** — rigidly align to Suunto's track and report drift + shape
   similarity ([`validate.py`](src/wayfinder/validate.py)).

Two findings from the reference dives, both matching the literature survey in
[`docs/research-dead-reckoning.md`](docs/research-dead-reckoning.md):

- **The uncalibrated magnetometer hurts.** Steel tanks and the watch's own fields
  distort it; gyro-only heading beats it on every reference dive, so mag fusion is
  off by default until hard/soft-iron calibration lands.
- **Free double-integration is hopeless** (hundreds of km of drift). The horizontal
  **velocity model**, not attitude, is the real problem — that's the next step.

On the reference dive (49 min, ~22 m), the v1 baseline lands within ~11 m endpoint
error and ~0.2 m/min drift after alignment — a shape match, not an absolute fix.
Suunto's own track is only accurate to ±20–30 m absolutely, so we treat it as a
shape reference, not ground truth.

## Roadmap

See [`docs/research-dead-reckoning.md`](docs/research-dead-reckoning.md) for the
full survey. Build order:

1. ✅ Naive strapdown baseline + validation harness (this milestone).
2. Water-frame **velocity model** (constant/slowly-varying forward speed or
   kick-cadence) to fix the over-travel.
3. **Two-anchor GPS boundary constraint** (rubber-sheet the track between the
   entry/exit surface fixes) — the biggest win per line of code, and what Suunto
   reportedly does.
4. Magnetometer calibration; optionally a learned velocity regressor (RoNIN/TLIO
   style).
