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
