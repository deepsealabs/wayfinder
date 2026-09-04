# Dead-Reckoning / Inertial Navigation for Reconstructing an Underwater Diver's Track

**Project:** wayfinder — reconstructing a relative horizontal X/Y dive track from a Suunto Ocean/Nautic 9-axis IMU + depth log.
**Date:** 2026-09-04

## Executive summary (read this first)

We have a wrist-worn 9-axis IMU at 10 Hz (accel 1/4096 g, gyro 1/131 deg/s, mag raw) plus depth at ~1 Hz, and optional surface GPS fixes only at entry/exit. The position problem is 2D because Z (depth) is measured directly. The core difficulty is **not** attitude — a 9-axis fusion filter (Madgwick/Mahony/ESKF) gives usable tilt and a rough heading — it is the **horizontal velocity model**. Free double-integration of horizontal acceleration is hopeless: attitude/gyro-bias error leaks gravity into the horizontal channel and integrates into position error that grows like **t³**, and accelerometer bias alone gives **t²** growth, so a naive strapdown solution diverges to hundreds of metres within a single dive. There is no clean ZUPT for a finning diver (feet never stop relative to the water), so the drift-reset trick that saves pedestrian INS does not directly apply. The realistic architecture is the one Suunto itself uses: a **learned or heuristic water-frame velocity model** (kick cadence / drag / RoNIN-style learned speed) integrated with the fused heading to produce a track shape, then **anchored between the two known GPS surface points** by a whole-path smoother (RTS / boundary-value optimization) that absorbs residual heading bias and mean current. Because the reference track (Suunto's app) is itself only ±20–30 m absolute, validation should emphasize **shape** (Fréchet / DTW) and **drift rate** over absolute trajectory error. Recommended build order: (1) a naive strapdown baseline purely to measure and visualize the drift, (2) a fused-attitude + constant-forward-speed (or kick-cadence) heading-and-velocity dead-reckoner in the water frame, (3) two-anchor boundary smoothing to the GPS endpoints, then (4) optionally a learned velocity regressor if we can gather labelled dives.

---

## 1. Attitude / orientation estimation from the 9-axis IMU

### Options and equations

**Complementary filter.** Blend a fast gyro integral with a slow accel/mag reference: `q̂ = (1−α)·q_gyro + α·q_accel_mag`, or in tilt form `θ = α(θ_prev + ω·dt) + (1−α)·θ_accel`. Cheap, stable, no covariance. Adequate for tilt (roll/pitch) where gravity is a strong, always-available reference.

**Mahony (explicit complementary filter on SO(3)).** Uses a PI feedback controller driving the gyro estimate toward the accel/mag reference: the error is the cross product `e = v_measured × v_estimated`, and `ω_corrected = ω_gyro − k_P·e − k_I·∫e dt`. The integral term estimates and cancels gyro bias. Robust to vibration, cheap ([IEEE comparison under external acceleration](https://ieeexplore.ieee.org/document/9701064)).

**Madgwick.** Gradient-descent formulation minimizing the same accel/mag alignment error; a single tunable gain β trades gyro-tracking vs. accel/mag correction. In head-to-head studies Madgwick best rejects transient linear acceleration, Mahony and EKF are more robust to vibration, and Mahony settles slightly faster ([SPIE/NDSU AHRS comparison](https://web.cs.ndsu.nodak.edu/~siludwig/Publish/papers/SPIE20181.pdf); [foot-mounted MIMU comparison](https://www.researchgate.net/publication/324048187_Comparison_of_attitude_and_heading_reference_systems_using_foot_mounted_MIMU_sensor_data_basic_Madgwick_and_Mahony)). Both Madgwick and Mahony are the de-facto pragmatic AHRS choices and are what open AHRS libraries ship ([Reefwing-AHRS](https://github.com/Reefwing-Software/Reefwing-AHRS)).

**EKF / UKF / error-state KF.** A quaternion (or MRP) state with a proper covariance; the **error-state Kalman filter (ESKF)** keeps the nominal quaternion outside the filter and estimates a small-angle error `δθ` plus gyro bias `b_g`, which keeps the linearization valid and handles the quaternion manifold cleanly. Canonical reference: Solà, *Quaternion kinematics for the error-state Kalman filter* ([arXiv:1711.02508](https://arxiv.org/abs/1711.02508)). The ESKF is the same machinery we will want for the position filter later, so adopting it for attitude gives one consistent framework and first-class **gyro-bias estimation** (bias is a state, corrected continuously by the gravity reference).

### Gyro bias

Gyro bias is the dominant attitude-error driver and the thing that later becomes t³ position error. All three practical filters handle it: Mahony via the PI integral term, Madgwick via a magnetic-distortion/bias branch, ESKF as an explicit state. At rest at the surface, average the gyro to get an initial bias; underwater the gravity vector (from accel, since mean specific force ≈ gravity when not accelerating hard) continuously re-observes pitch/roll and thus the bias about horizontal axes. Bias about the **vertical** (yaw) axis is only observable from the magnetometer.

### Magnetometer underwater

The mag is the **only** absolute yaw reference (gyro yaw drifts, gravity says nothing about heading). Underwater the earth field is undisturbed, but the diver's own rig is not: **hard-iron** (tank valve, weights, the watch/console electronics — additive static offset) and **soft-iron** (ferrous mass distorting the field into an ellipsoid) errors dominate ([hard/soft-iron overview, VectorNav](https://www.vectornav.com/resources/inertial-navigation-primer/specifications--and--error-budgets/specs-magerrorsources); [online hard/soft-iron estimation, arXiv:2201.02449](https://arxiv.org/pdf/2201.02449)). These are calibratable: fit an ellipsoid to raw mag samples (offset = hard-iron, shape matrix = soft-iron) and, critically, **tilt-compensate** using the fused roll/pitch before computing heading. Proper mag calibration is high-leverage: on a seafloor-mapping vehicle, calibrating the compass reduced dead-reckoned position error from ~10% to ~0.5% of distance travelled ([magnetometer calibration for AHRS/underwater vehicles](https://www.researchgate.net/publication/4350726_The_soft_iron_and_hard_iron_calibration_method_using_extended_kalman_filter_for_attitude_and_heading_reference_system)). Failure mode for us: a moving steel wreck/ledge, a dive light, or the diver bringing the console near the tank swings heading by tens of degrees; heading glitches map directly into track-shape error.

**v1 takeaway.** Use **Madgwick or Mahony** for attitude (tilt is easy and robust). Do a one-time ellipsoid mag calibration from a surface figure-eight and tilt-compensate heading. Estimate gyro bias at the surface and let the filter track it. Treat heading as "good to maybe 5–15°," not exact, and plan to correct residual heading bias globally in step 7. Migrate attitude into the ESKF later only if we adopt an ESKF for position anyway.

---

## 2. Strapdown INS mechanization and why free double-integration fails

### Mechanization

Strapdown INS integrates body-frame IMU signals through the attitude solution:

1. **Attitude:** `q̇ = ½ q ⊗ [0, ω_body]`.
2. **Specific force to navigation frame:** `a_nav = R(q)·f_body`.
3. **Remove gravity:** `a_true = a_nav − g` (g ≈ [0,0,9.81]).
4. **Integrate twice:** `v = ∫a_true dt`, `p = ∫v dt`.

(Savage, *Computational Elements for Strapdown Systems*, [NATO RTO-EN-SET-116](https://publications.sto.nato.int/publications/STO%20Educational%20Notes/RTO-EN-SET-116-2009/EN-SET-116(2009)-09.pdf); Woodman, *An Introduction to Inertial Navigation*, [Cambridge UCAM-CL-TR-696](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.pdf).)

### Why it diverges (the whole reason this project is hard)

- **Accelerometer bias `b_a`** integrates twice into position: `Δp ≈ ½ b_a t²` — **quadratic** growth ([Woodman TR-696](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.pdf); restated for divers by [PNI/underwater DR](https://www.pnisensor.com/underwater-navigation-solutions-for-gps-denied-missions/)).
- **Gyro bias / attitude error `δθ`** tilts the gravity-removal frame, so a fraction `g·δθ` of the 9.81 m/s² gravity vector leaks into the horizontal acceleration channel. That error is itself growing linearly (δθ ≈ b_g·t), so after double integration position error grows like **t³** ([Inside GNSS, *Inertial Error Propagation*](https://insidegnss.com/inertial-error-propagation-understanding-inertial-behavior/)).
- **Gravity-removal tilt sensitivity:** a 1° tilt error leaks `9.81·sin(1°) ≈ 0.17 m/s²` of false horizontal acceleration — larger than most real finning accelerations. This is why attitude is "good enough" but *residual* attitude error still destroys a naive integrator: the signal we want (diver thrust) is buried under gravity leakage.

With MEMS-grade sensors this limits pure-inertial usability to ~20–30 minutes even for well-calibrated units, and far less for a wrist IMU ([PNI](https://www.pnisensor.com/underwater-navigation-solutions-for-gps-denied-missions/)).

**v1 takeaway.** Implement the full strapdown chain **once, as a baseline whose only job is to quantify drift** and prove the failure numerically (plot position error vs. time; expect hundreds of metres). Do **not** ship free double-integration. The fix is structural (Sections 4–7), not better tuning.

---

## 3. Drift mitigation: ZUPT, ZARU, and whether any analog exists for divers

**ZUPT (zero-velocity update).** In foot-mounted pedestrian INS, the foot is momentarily stationary during each stance phase; detecting that and injecting a pseudo-measurement `v = 0` into the Kalman filter resets accumulated velocity error and, through the filter's cross-covariance, corrects accel bias and tilt too. This is the single most effective drift killer in PDR, keeping error to ~1% of distance ([Review of ZUPT-aided pedestrian INS](https://www.researchgate.net/publication/343337435_A_Review_on_ZUPT-Aided_Pedestrian_Inertial_Navigation); [drift-reduction methods, PMC6766805](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6766805/)).

**ZARU (zero angular-rate update)** and **zero-velocity of specific axes**: when stationary, gyro output = bias (ZARU re-estimates gyro bias); vehicle systems assert zero *lateral/vertical* velocity even while moving forward — the trick behind [AI-IMU Dead-Reckoning](https://arxiv.org/abs/1904.06064), where a network adapts the covariance of these pseudo-measurements.

**Heading correction / HDE.** Heuristic Drift Elimination exploits man-made straight corridors to snap heading to dominant directions ([Enhanced HDE, PMC7070454](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7070454/)) — not applicable to open-water diving.

**Does any ZUPT analog exist for a finning diver?** Essentially **no true ZUPT**: the diver never comes to rest relative to the water during a fin cycle, and even a hovering diver is being advected by current, so body velocity relative to ground is not zero. However, three weaker analogs are usable:

- **Depth-derived vertical ZUPT-like constraint:** the depth channel *directly measures Z*, so vertical velocity is observable/bounded — we should feed `v_z = d(depth)/dt` as a hard measurement rather than integrating vertical accel at all.
- **Glide/pause detection:** between kick cycles a diver briefly decelerates; the *periodicity* of thrust (Section 4) is exploitable even if velocity never hits zero.
- **Surface ZUPT:** at true entry/exit the diver is at v≈0 at a known GPS point — that bookends the trajectory (Section 7).

**v1 takeaway.** Use the depth channel as a vertical-velocity measurement (kills the vertical integrator entirely). Do not expect a horizontal ZUPT. Replace the missing zero-velocity anchor with (a) a velocity *model* and (b) the two GPS endpoints.

---

## 4. Velocity models — the structural fix

Since we cannot reset velocity, we must **assume a model for it** and integrate *that* along the fused heading, rather than integrating raw acceleration.

**Step-and-heading / cadence models (PDR).** `position += stride_length · [cos ψ, sin ψ]` per detected step, ψ = heading. For a diver the analog is a **kick-cadence** model: detect fin-kick cycles from the periodic accel/gyro signature (the same signal processing that swimming-IMU studies use to segment strokes and even estimate forward velocity from a sacrum/wrist IMU — [Inertial sensors in swimming, PubMed 31427865](https://pubmed.ncbi.nlm.nih.gov/31427865/); [swim-stroke velocity validation, PMC10813451](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10813451/)). Each kick → a roughly constant glide distance in the heading direction. Robust, interpretable, cheap.

**Constant-velocity / drag model.** Assume the diver moves at a slowly varying forward speed (0.2–0.5 m/s typical) with hydrodynamic drag coupling thrust to a terminal speed; treat speed as a slowly-varying state and heading from AHRS. Even a *constant* assumed speed produces a correctly-shaped track that the endpoint constraint can rescale.

**Learned velocity (data-driven inertial odometry).** The strongest general approach, and the closest to what Suunto describes:
- **IONet** — LSTM regresses 2D displacement/heading change from IMU windows ([survey, arXiv:2303.03757](https://arxiv.org/pdf/2303.03757)).
- **RIDI** — regresses velocity then corrects IMU to match, integrates ([survey](https://arxiv.org/pdf/2303.03757)).
- **RoNIN** — ResNet/LSTM/TCN regressing 2D velocity in a *heading-agnostic* frame, so it doesn't inject spurious yaw ([RoNIN, arXiv:1905.12853](https://arxiv.org/abs/1905.12853)).
- **TLIO** — 1D ResNet regresses 3D displacement **and its uncertainty** from 1 s gravity-aligned IMU windows, fused in a stochastic-cloning EKF; the network learns only a prior for "typical human motion," leaving integration to the filter, and measurements are in the gravity-aligned frame to avoid injecting heading ([TLIO, arXiv:2007.01867](https://arxiv.org/abs/2007.01867)).

The key insight shared by all: **the network learns a motion prior (a velocity), and the filter does the geometry.** This is exactly the velocity model we lack, learned from data.

**Applicability to finning.** Swimming/finning is more periodic and lower-bandwidth than walking, which *helps* a cadence or learned model. The catch: these models output velocity **relative to the water**, not the ground (Section 6), and would need retraining on dive data (arm/wrist motion of a finning diver ≠ pedestrian gait), so RoNIN/TLIO can't be used off-the-shelf.

**v1 takeaway.** Start with the simplest velocity model that yields a plausible shape: **constant (or slowly-varying) forward speed along fused heading**, optionally upgraded to a **kick-cadence** speed once we can detect fin cycles. Reserve a learned RoNIN/TLIO-style regressor for v2 when we have labelled dives; when we do, prefer TLIO's "displacement + uncertainty, integrate in a filter" structure.

---

## 5. How AUVs/ROVs actually navigate — and what transfers with no DVL

Production underwater navigation is **aided INS**, almost never pure inertial:

- **DVL (Doppler Velocity Log):** acoustic beams measure velocity **over ground** (bottom-lock) to cm/s. INS/DVL fusion is the workhorse and holds ~0.1–1% of distance ([INS/DVL fusion, PMC5335996](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5335996/); [Nortek subsea guide](https://www.nortekgroup.com/knowledge-center/wiki/new-to-subsea-navigation)).
- **USBL / LBL acoustic positioning:** ship- or beacon-referenced absolute fixes ([INS/USBL/DVL graph optimization, PMC9864396](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9864396/); [Advanced Navigation subsea](https://www.advancednavigation.com/subsea-navigation-systems/)).
- **Aided INS pattern:** INS provides high-rate dead reckoning; DVL bounds velocity drift; USBL/LBL/GPS-at-surface bounds absolute position; a Kalman filter (or graph optimizer) fuses them.

**What transfers to us with no DVL:** the *architecture* — INS for shape, an external velocity aid to stop drift, and sparse absolute fixes to bound position. **The DVL is exactly the piece we don't have**, and it is the piece that makes underwater INS work. Our substitutes for the DVL's ground-velocity measurement are the **velocity model** (Section 4, water-frame) and the depth sensor (vertical). Our substitutes for USBL are the **two GPS surface fixes** (Section 7). This framing tells us precisely where our error budget will blow up: horizontal velocity and current.

**v1 takeaway.** Copy the aided-INS *structure*, but recognize we are DVL-denied: our "velocity aid" is a model, not a measurement, so it will be biased (especially by current). Design the endpoint constraint to absorb that bias.

---

## 6. Underwater drifters & current handling — water frame vs. ground frame

Any velocity we derive from the diver's own motion (kick cadence, drag model, learned regressor, even a hypothetical wearable DVL locking on nearby water) is **velocity relative to the water**, not the ground:

`v_ground = v_water-relative (diver's swimming) + v_current (water's motion over ground)`

This is the Lagrangian "slip" problem: a drifter following a water parcel measures the water's motion, while a swimmer adds their own slip through that water ([Lagrangian current measurement, LibreTexts](https://geo.libretexts.org/Bookshelves/Oceanography/Introduction_to_Physical_Oceanography_(Stewart)/10:_Geostrophic_Currents/10.08:_Lagrangian_Measurements_of_Currents); [measuring drifter slip with onboard ADCP, PMC6562283](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6562283/)). A kick-cadence trajectory is drawn in the **water frame**: it captures the diver's path *through the water* but is translated (and sheared, if current varies) relative to the seabed. Consequences:

- A diver hovering motionless still drifts over the ground with the current — invisible to an IMU.
- Integrating water-frame velocity gives a track whose **shape** is roughly right but whose **net displacement** is offset by `∫v_current dt`.
- If current is approximately constant over the dive, it appears as a **constant velocity bias** — which is exactly what a two-endpoint constraint can estimate and remove (a linear drift correction). Spatially/temporally varying current is the residual we cannot fully recover without external aiding.

**v1 takeaway.** Treat our dead-reckoned track as *water-frame*. Model unknown current as a constant (or low-order) drift and let the GPS-endpoint constraint (Section 7) solve for it. Flag high-current dives as low-confidence.

---

## 7. Boundary-value / two-anchor formulation (what Suunto reportedly does)

We know two absolute points: the **GPS entry fix** and the **GPS exit fix** (each ±a few m at the surface). Instead of dead-reckoning forward and hoping, constrain the **whole trajectory** to start at A and end at B. This is a boundary-value problem, and it is the single highest-leverage trick we have.

**Methods:**
- **RTS (Rauch–Tung–Striebel) smoother:** run the forward Kalman filter, then a backward pass that redistributes the endpoint information across the entire path, so the known exit fix corrects earlier states — standard in survey-grade post-processing ([SBG Qinertia RTS smoother](https://www.sbg-systems.com/news/rts-smoother-qinertia/)). Because our exit GPS point is a strong late measurement, RTS pulls the whole track toward consistency.
- **Whole-path optimization / batch least squares:** minimize `Σ (dead-reckoning residuals) + ‖p(0)−A‖ + ‖p(T)−B‖`, solving for a heading-bias and a constant current-drift term simultaneously. Even the crude version — dead-reckon A→B', then apply a linear "rubber-sheet" correction that rotates/scales/translates the path so B'→B — removes most accumulated error and is trivial to implement.
- **Suunto's approach (as publicly described):** underwater route tracking uses "GPS, accelerometer, gyroscope, magnetometer and pressure sensor," and "the algorithm has been developed by using large amount of data from real dives, data analytics and machine learning"; it needs a satellite fix at start and end of the dive, and the GPS points are used to error-correct the inertial underwater trajectory ([Suunto Ocean support/settings](https://www.suunto.com/Support/Product-support/suunto_ocean/suunto_ocean/scuba-diving/dive-settings/); community discussion, [ScubaBoard](https://scubaboard.com/community/threads/suunto-ocean-inertial-underwater-navigation.647634/)). This is exactly the **learned velocity model + GPS boundary constraint** pattern. It fails in overhead environments (caves, wrecks), pools, or with no surface GPS — precisely because the boundary anchors and/or the motion prior break down.

**v1 takeaway.** This is our biggest win for the least code. Implement the crude linear endpoint correction (rotate+scale+translate the water-frame track to hit both GPS points) in v1; upgrade to an RTS smoother / batch optimizer that explicitly solves for heading bias + constant current in v2. It converts an unbounded-drift problem into a bounded one and directly emulates Suunto.

---

## 8. Validation metrics when the ground truth is itself imprecise

The Suunto app track is a **shape reference good only to ±20–30 m absolute**, so absolute-position metrics will be dominated by the reference's own error. Report a spread:

- **ATE (Absolute Trajectory Error):** RMS of pointwise distance after rigid alignment (Umeyama) of our track to the reference. Captures global agreement but is unfair here because both tracks are anchored to imprecise GPS — report it, but don't optimize to it ([ATE/RPE definitions and benchmark practice](https://arxiv.org/pdf/2605.11674)).
- **RPE (Relative Pose Error):** error in relative motion over a fixed window/segment — measures **local drift/odometry consistency** independent of global anchoring. More diagnostic of our velocity model than ATE.
- **Drift rate (% of distance travelled):** classic INS/DVL figure of merit — final endpoint error (before the boundary constraint is applied) divided by path length. This is the honest measure of the raw dead-reckoner and is how AUV nav is quoted (0.1–1% for INS/DVL).
- **Fréchet distance:** "dog-leash" distance; sensitive to overall path **shape** and ordering — ideal when we care about matching the geometry of the reference, not exact coordinates ([comparative analysis of trajectory similarity measures](https://www.tandfonline.com/doi/full/10.1080/15481603.2021.1908927)).
- **DTW (Dynamic Time Warping):** aligns the two tracks in time, robust to local speed differences — good when our velocity model's *timing* differs from the reference but the route is the same ([time-series similarity for movement trajectories, Springer](https://link.springer.com/article/10.1007/s00265-019-2761-1)). Fréchet and DTW consistently rank best for shape comparison on movement data.

**v1 takeaway.** Headline metrics: **drift rate** (of the raw dead-reckoner, pre-anchor) and **Fréchet + DTW shape similarity** (post-anchor, vs. the Suunto track). Report ATE/RPE for completeness but caveat that the reference is ±20–30 m. Always plot the two tracks overlaid — visual shape agreement is the most honest evidence given imprecise truth.

---

## Prioritized recommendation

**Phase 0 — Data hygiene (do first, cheap).**
Ellipsoid-calibrate the magnetometer (hard/soft iron) from a surface figure-eight; tilt-compensate heading. Estimate gyro bias from a still surface interval. Use `depth` differentiated as the vertical-velocity truth — never integrate vertical accel.

**Phase 1 — Naive strapdown baseline (to measure the enemy).**
Implement the full strapdown chain (Section 2) with Madgwick/Mahony attitude. Expected outcome: horizontal position diverges to 100s of metres; plot error vs. time to confirm t²/t³ growth. This is a *diagnostic*, not a product. **Highest priority because it calibrates expectations and gives a drift-rate number to beat.**

**Phase 2 — Model-based water-frame dead reckoner (the real v1).**
Fused attitude (Madgwick/Mahony) for heading + a **velocity model** instead of integrated accel: start with constant/slowly-varying forward speed along heading; upgrade to **kick-cadence** speed from fin-cycle detection. Depth gives Z. Output: a plausibly-shaped **water-frame** track. This is where the shape becomes usable.

**Phase 3 — Two-anchor boundary constraint (highest leverage per line of code).**
Warp the Phase-2 track to hit both GPS surface points: v1 = rigid rotate+scale+translate (rubber-sheeting); v2 = RTS smoother / batch least-squares that explicitly solves for **residual heading bias + constant current drift**. This directly emulates Suunto and converts unbounded drift into bounded error.

**Phase 4 (optional) — Learned velocity model.**
If we can gather labelled dives (dead-reckoned tracks with good GPS-bounded truth), train a RoNIN/TLIO-style regressor that outputs water-frame velocity **and uncertainty** from IMU windows, fused in the same filter. Only worthwhile once Phases 1–3 are solid and data exists.

**Ordering rationale:** Phases 1→2→3 each remove the *dominant* error term of the previous stage (double-integration divergence → replaced by a velocity model → residual bias/current absorbed by endpoints). Attitude is already "good enough" from Phase 0, so effort belongs on the velocity model and the boundary constraint, not on fancier attitude filters.

---

## References

1. Woodman, O. *An Introduction to Inertial Navigation.* Cambridge UCAM-CL-TR-696. https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.pdf
2. *Inertial Error Propagation — Understanding Inertial Behavior.* Inside GNSS. https://insidegnss.com/inertial-error-propagation-understanding-inertial-behavior/
3. Savage, P. *Computational Elements for Strapdown Systems.* NATO RTO-EN-SET-116. https://publications.sto.nato.int/publications/STO%20Educational%20Notes/RTO-EN-SET-116-2009/EN-SET-116(2009)-09.pdf
4. Solà, J. *Quaternion kinematics for the error-state Kalman filter.* arXiv:1711.02508. https://arxiv.org/abs/1711.02508
5. *Comparison of Attitude and Heading Reference Systems (basic/Madgwick/Mahony).* SPIE / NDSU. https://web.cs.ndsu.nodak.edu/~siludwig/Publish/papers/SPIE20181.pdf
6. *Comparison of AHRS using foot-mounted MIMU: basic, Madgwick, Mahony.* ResearchGate. https://www.researchgate.net/publication/324048187_Comparison_of_attitude_and_heading_reference_systems_using_foot_mounted_MIMU_sensor_data_basic_Madgwick_and_Mahony
7. *Comparison of Attitude Estimation Algorithms With IMU Under External Acceleration.* IEEE. https://ieeexplore.ieee.org/document/9701064
8. Reefwing-AHRS (open AHRS implementations). https://github.com/Reefwing-Software/Reefwing-AHRS
9. *Magnetometer error sources (hard/soft iron).* VectorNav. https://www.vectornav.com/resources/inertial-navigation-primer/specifications--and--error-budgets/specs-magerrorsources
10. *Online 3-Axis Magnetometer Hard-Iron and Soft-Iron Bias Estimation.* arXiv:2201.02449. https://arxiv.org/pdf/2201.02449
11. *Soft/hard iron calibration via EKF for AHRS.* ResearchGate. https://www.researchgate.net/publication/4350726_The_soft_iron_and_hard_iron_calibration_method_using_extended_kalman_filter_for_attitude_and_heading_reference_system
12. *A Review on ZUPT-Aided Pedestrian Inertial Navigation.* ResearchGate. https://www.researchgate.net/publication/343337435_A_Review_on_ZUPT-Aided_Pedestrian_Inertial_Navigation
13. *Novel Drift Reduction Methods in Foot-Mounted PDR.* PMC6766805. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6766805/
14. *Enhanced Heuristic Drift Elimination with Adaptive Zero-Velocity Detection.* PMC7070454. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7070454/
15. Brossard, M., Barrau, A., Bonnabel, S. *AI-IMU Dead-Reckoning.* arXiv:1904.06064. https://arxiv.org/abs/1904.06064
16. Herath, S. et al. *RoNIN: Robust Neural Inertial Navigation.* arXiv:1905.12853. https://arxiv.org/abs/1905.12853
17. Liu, W. et al. *TLIO: Tight Learned Inertial Odometry.* arXiv:2007.01867. https://arxiv.org/abs/2007.01867
18. *Deep Learning for Inertial Positioning: A Survey* (IONet, RIDI, RoNIN, TLIO). arXiv:2303.03757. https://arxiv.org/pdf/2303.03757
19. *INS/DVL Fusion with Partial DVL Measurements.* PMC5335996. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5335996/
20. *Robust INS/USBL/DVL Integrated Navigation via Graph Optimization.* PMC9864396. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9864396/
21. *A Complete Guide to Underwater Navigation.* Nortek. https://www.nortekgroup.com/knowledge-center/wiki/new-to-subsea-navigation
22. *Subsea Navigation Systems.* Advanced Navigation. https://www.advancednavigation.com/subsea-navigation-systems/
23. *Underwater Navigation Solutions for GPS-Denied Missions.* PNI Sensor. https://www.pnisensor.com/underwater-navigation-solutions-for-gps-denied-missions/
24. *Lagrangian Measurements of Currents.* Introduction to Physical Oceanography (Stewart), LibreTexts. https://geo.libretexts.org/Bookshelves/Oceanography/Introduction_to_Physical_Oceanography_(Stewart)/10:_Geostrophic_Currents/10.08:_Lagrangian_Measurements_of_Currents
25. *Measuring a Lagrangian drifter's slip with an onboard ADCP.* PMC6562283. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6562283/
26. *Fine-tune your trajectory with the RTS Smoother in Qinertia.* SBG Systems. https://www.sbg-systems.com/news/rts-smoother-qinertia/
27. Suunto Ocean — Scuba diving dive settings (underwater route tracking). https://www.suunto.com/Support/Product-support/suunto_ocean/suunto_ocean/scuba-diving/dive-settings/
28. *Suunto Ocean — inertial underwater 'navigation'.* ScubaBoard community discussion. https://scubaboard.com/community/threads/suunto-ocean-inertial-underwater-navigation.647634/
29. *Inertial Sensors in Swimming: Detection of Stroke Phases through 3D Wrist Trajectory.* PubMed 31427865. https://pubmed.ncbi.nlm.nih.gov/31427865/
30. *Validation of Automatically Quantified Swim Stroke Mechanics Using an IMU.* PMC10813451. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10813451/
31. *A Proprioceptive-Only Benchmark for State Estimation: ATE, RPE, Runtime.* arXiv:2605.11674. https://arxiv.org/pdf/2605.11674
32. *A comparative analysis of trajectory similarity measures (Fréchet, DTW, Hausdorff…).* Taylor & Francis. https://www.tandfonline.com/doi/full/10.1080/15481603.2021.1908927
33. *Using time-series similarity measures to compare animal movement trajectories.* Springer. https://link.springer.com/article/10.1007/s00265-019-2761-1
