# CLAUDE.md: QNX Edge-AI Satellite Debris Avoidance

## What this project is
Hackathon project (QNX edge-AI challenge). A **Raspberry Pi 5 running QNX 8.0** is the flight
computer of a simulated satellite. A person moves a real **orange 40 mm ping-pong ball** in front
of **two cameras** mounted on the Pi. A **TFLite CNN** confirms the ball in both images, **stereo
disparity** gives its distance, and when the ball is within **20 cm** of the board the Pi commands
simulated **cold-gas thrusters** to sidestep the "asteroid" and then return to the original orbit slot.
A **laptop** runs a **3D simulator** (Panda3D) that plays "the universe": rigid-body physics, orbit,
rendering, dashboard. **All sensing and decisions run on the Pi; the laptop only simulates and draws.**

PRD (goals, requirements, phases): https://claude.ai/artifact/2Ji33PdA42CRRXoRfDWQid

### Judging criteria (every change must keep serving these)
1. Cannot-fail embedded application: AI isolated from safety, auto-restart, fail-safe states.
2. Real-time / reliability: fixed 50 Hz control loop, priorities, core pinning, measured jitter.
3. Creative AI: CNN must confirm the ball in BOTH cameras before stereo depth is trusted.
4. Runs on embedded hardware: no cloud; the laptop only plays the physical world (hardware-in-the-loop).
Core message for the pitch: **"The AI suggests, the QNX control process decides."**

## Hardware and environment
- Raspberry Pi 5, QNX SDP 8.0 Quick Start Target Image (QSTI). Hostname `qnxpi` (`qnxpi.local`),
  user `qnxuser`, SSH enabled, VNC GUI on port 5900. Python 3 is preinstalled.
- Packages on the Pi come from `apk` (QNX open-source ports, https://oss.qnx.com):
  `python3-tflite-runtime` 2.21 (aarch64, QNX 8.0.3). **No OpenCV on the Pi**: Pi code is numpy + tflite only.
- Cameras: 2x Raspberry Pi Camera Module 3. On the QNX Pi 5 image they are sensor units **3 (camera 1)**
  and **4 (camera 2)**. Config: `/etc/config/sensor/sensor_rpi5.conf`. View one: `camera_mux -n 1 -u 3`.
  Restart the sensor service: `slay sensor` then
  `sensor -U 521:521 -r /data/share/sensor -b external -c /etc/config/sensor/sensor_rpi5.conf`.
  Frames are read through the QNX camera library (C API), not V4L2.
- Laptop: Python 3, `pip install panda3d numpy tensorflow opencv-python`.
- Link: wired Ethernet laptop <-> Pi preferred (Wi-Fi drops trigger the safety shutdowns).

## Files
| File | Runs on | Role |
|---|---|---|
| `config.py` | both (IDENTICAL copy) | every shared constant: IPs/ports, scaling, physics, thresholds |
| `spacemath.py` | both | quaternion helpers, `HandScenario` (synthetic hand-held ball) |
| `perception.py` | Pi | cameras -> CNN -> stereo depth -> Kalman -> `track` msgs (LOW priority) |
| `control.py` | Pi | 50 Hz guidance/attitude/FDIR -> `cmd` msgs (HIGH priority) |
| `supervisor.py` | Pi | starts perception + control as separate processes, restarts on crash/hang |
| `sim3d.py` | laptop | 3D rigid-body simulator + dashboard (the "universe") |
| `train_ball_cnn.py` | laptop | trains the ball/not-ball CNN, exports `ball_cnn_int8.tflite` |
| `stereo_calibrate.py` | laptop | checkerboard stereo calibration -> `calib.json`, `rectify_maps.npz` |
| `ball_cnn_int8.tflite` | Pi | model (32x32x3 int8 in, 2 classes out: 0 = ball, 1 = not ball) |
`sim_satellite.py` (old 2D sim) is retired: its message format is outdated. Do not revive it.

## Architecture and messages (all JSON over UDP)
Ports: `SIM_PORT 5005` (laptop listens), `CONTROL_PORT 5006` (Pi control), `SUPERVISOR_PORT 5007` (Pi, localhost).
IPs come from env vars `SIM_IP` (laptop) and `PI_IP` (Pi); both default to 127.0.0.1 (single-machine tests).

| From -> To | type | Rate | Fields |
|---|---|---|---|
| perception -> control (127.0.0.1:5006) and -> sim (SIM_IP:5005) | `track` | each frame (~30 Hz) | `seq, detected, reason, pos, vel` (board frame, m, m/s REAL time; null if not detected), `latency_s, classifier, cnn_ms, stereo_ms, n_candidates` (+ when detected: `disparity_px, depth_m, range_m, conf, match_ratio, uv, radius_px`) |
| perception -> sim | `depthmap` | every 6th frame | `w=80, h=60, data` (base64 uint8, 0 = invalid, 255 = DISP_MAX/2 px), `ball` ([u,v] in map px or null) |
| sim -> control (PI_IP:5006) | `nav` | 50 Hz | `t` (mission s), `pos, vel` (LVLH m, m/s mission), `q` [w,x,y,z] body->LVLH, `omega` (body rad/s), `h_w`, `mass, prop` (kg), `reset_id` |
| control -> sim (SIM_IP:5005) | `cmd` | 50 Hz | `seq, phase, motion, target, valves` (e.g. ["+z"]), `torque` (wheel torque on body, N m), `isolated` (e.g. ["+z:A"]), `branch` ({"+z":"B"}), `fault, threat{range,t_cpa,miss,trigger,approaching}, track, att_err_deg, loop{work_ms,jitter_max_ms,overruns}` |
| control, perception -> supervisor | `hb` | 10 Hz | `name` |
| supervisor -> sim | `procs` | 2 Hz | `procs{name:{pid,prio,restarts,alive,hb_age,last_event}}` |
Rule: **never change a message field without updating every sender and receiver in the same change.**
Perception and control never share memory or import each other; they only exchange messages (isolation story).

## Frames, scaling, time
- Board/camera frame (real, m): origin = midpoint between the cameras, X right, Y down, Z forward.
- LVLH frame (space, m): x radial (up, away from Earth), y along-track (cameras look along +y),
  z cross-track. Origin = the satellite's slot on its original orbit (= where the board is).
- `CAM_TO_LVLH`: x = -Y_cam, y = Z_cam, z = X_cam. `ball_to_space(p) = SPACE_SCALE * CAM_TO_LVLH @ p`.
- `SPACE_SCALE = 100` (1 cm of ball distance = 1 m), `TIME_SCALE = 5` (1 real s = 5 mission s).
  Velocities: `ball_vel_to_space(v) = SPACE_SCALE / TIME_SCALE * CAM_TO_LVLH @ v`.
- The asteroid shown = ball position x 100, drawn ONLY while `detected` is true and < 0.2 s old.
- Panda3D scene mapping: scene = (-z, y, x) of LVLH (right-handed), 1 m = 1 scene unit.

## Physics model (sim3d.py) - keep it physically honest
- Satellite: dry 23 kg + 2 kg N2, six body-fixed 5 N thrusters (`VALVES` "+x".."-z" = direction of the
  acceleration they give), Isp 65 s, propellant flow = n * F / (Isp * g0).
- Translation: a = R(q) * F_body / m (Newton 2), no drag (Newton 1: coasting persists),
  Clohessy-Wiltshire terms with n = 2*pi/5554 s (400 km orbit).
- Rotation: Euler's equations I*w' = tau_dist + tau_wheels - w x (I*w + h_wheels); disturbance torque
  = -COM_OFFSET x F_body (thrust not through the centre of mass); wheels limited to 0.1 N m, 1.0 N m s.
- Each thruster direction has a primary (A) and backup (B) branch; a stuck valve is injected as "+z:A".
- Physics at 200 steps per real second; watchdog: no `cmd` for 150 ms -> all valves closed.

## Control logic (control.py)
- Fixed 50 Hz loop (absolute next_t scheduling), counts overruns and jitter.
- Phases: STATION_KEEP -> EVADE -> HOLD_CLEAR -> RETURN -> STATION_KEEP. Fail-safe phase NO_NAV_SAFE.
- Trigger: asteroid range <= `TRIGGER_RANGE` (20 m = ball 20 cm) OR predicted miss < 8 m within 10 mission s.
- Dodge target: 10 m (`DODGE_DIST`) perpendicular to the approach line; prefer cross-track (z), then
  radial (x); never along-track (orbit drift). Needs both directions of an axis available (go AND brake).
- Translation guidance per LVLH axis = bang-coast-bang: BURN until V_MAX (1.0 m/s), COAST, BRAKE when the
  stopping distance v^2/2a (+ latency margin) reaches the remaining distance. a = THRUST_N / current mass.
  Desired LVLH direction is rotated into the body frame (R^T) before choosing valves.
- Return when the ball is unseen for 1.5 s or beyond 30 cm and not approaching (and not near the slot).
- Attitude: PD on small-angle error, torque = -KP*theta - KD*omega, clipped to wheel limits.
- FDIR: compare measured accel (from nav velocity over 1 mission s) with commanded; residual > 50% of one
  thruster's accel for 0.3 s -> isolate that branch, switch to the backup.
- Interlock: opposing thrusters never fire together. No nav for 0.3 s -> all valves closed.

## Perception (perception.py)
1. `read()` returns (t_capture, left_rgb, right_rgb), each (240, 320, 3) uint8.
2. Rectify with `rectify_maps.npz` (nearest-neighbour numpy indexing) if present.
3. `find_candidates`: orange by normalised chromaticity (R/(R+G+B) > 0.45, B/(R+G+B) < 0.20),
   connected components on a 4x grid. `crop_patch` -> 32x32 -> CNN. Ball must pass in BOTH images.
4. Checks: same row within 6 px (epipolar), disparity in [2, 170] px, silhouette-match disparity
   (search centroid disparity +/- 25 px) agrees with centroid disparity within max(4 px, 6%), and the
   minimum is not on the search edge. Otherwise the frame is rejected and `reason` says why.
5. Depth Z = f*B/d; position relative to the board centre; constant-velocity Kalman filter.
6. Display disparity map: block matching (SAD) with a left-right consistency check, invalid = 0;
   the ball blob is filled with its silhouette disparity (uniform ball cannot be matched per pixel).
- `--source synthetic` renders a hand-held ball, a textured wall and an orange box distractor.
- `--source qnx` uses `QnxStereoSource`, which is currently a **stub that raises NotImplementedError**.

## Supervisor (supervisor.py)
- Launches with QNX `on` when present: control `on -p 40 -C 0`, perception `on -p 15 -C 1 -C 2 -C 3`.
- Restarts on CRASH (process exited) or HANG (no heartbeat 0.5 s control / 1.5 s perception, after a
  startup grace of 5 s / 30 s). Kills hung processes with SIGKILL (works on stopped processes).
- Sends status from a separate UNBOUND socket (a socket bound to 127.0.0.1 cannot reach the laptop).

## How to run
Laptop (start FIRST):
    SIM_IP=<laptop-ip> PI_IP=<pi-ip> python3 sim3d.py
Pi:
    SIM_IP=<laptop-ip> python3 supervisor.py --source synthetic   # later: --source qnx
Single machine test (everything on the laptop): `python3 sim3d.py` and `python3 supervisor.py`.
Simulator keys: 1 chase, 2 side, 3 onboard camera, M mock ball (no Pi perception needed),
S stuck valve +z:A, F drain propellant, R reset, ESC quit.
Fault demos on the Pi: `kill -9 <perception pid>` (crash), `kill -STOP <control pid>` (hang);
the supervisor prints the PIDs.

## Testing rules for Claude Code
- After any change, run an end-to-end test before saying it works. Headless Panda3D test:
  `loadPrcFileData("", "load-display p3tinydisplay\nwindow-type offscreen\nframebuffer-multisample 0\nmultisamples 0\nsync-video 0")`
  after importing `sim3d` and before creating `Sim3D()`; save screenshots with `app.win.saveScreenshot(Filename(...))`.
  p3tinydisplay requires power-of-two textures (the disparity texture is 128x64 for this reason).
- Success criteria: 0 hits over 5+ encounters, `[sim3d] encounter over` lines show closest approach
  >> COLLISION_DIST (4 m), no Python tracebacks in the supervisor log, crash/hang/stuck-valve recover.
- Known past bugs (do not reintroduce): numpy float32 in `json.dumps` crashed perception (use `default=float`);
  dodging along-track let the satellite "outrun" the ball (only sideways candidates now); isolating a
  valve without a backup branch made the satellite drift away.

## Coding conventions
- Pi code: numpy + tflite_runtime + stdlib only. Keep it compatible with the QNX Python 3 version
  (check `python3 --version` on the Pi; avoid very new syntax).
- `config.py` must stay identical on both machines; put new constants there, never hard-code.
- Safety first: any new failure path must end in a defined safe state (valves closed, track dropped,
  dodge held). Keep perception unable to block control.
- Don't claim timing numbers measured on the laptop as QNX results; measure on the Pi.
- Prefer small, testable changes; keep the message schema table in this file up to date.

## Current status (as of 2026-11-21)
Done and tested with synthetic cameras: perception (CNN 99.5% on synthetic, depth error ~0.3 mm),
control, supervisor, 3D simulator, calibration (recovered a 6.00 cm baseline as 5.99 cm), training.
Results: 0 hits across ~20 encounters; crash, hang and stuck-valve recovery all verified.

## Next work (phases from the PRD)
0. QNX bring-up: both cameras visible as sensor units 3 and 4; write the **stereo capture bridge**:
   a small C program using the QNX camera library that opens units 3 and 4, grabs frames as close to
   simultaneously as possible, converts to 320x240 RGB, and writes [t_capture, left, right] into shared
   memory (`/dev/shmem/...`) with a sequence counter; then implement `QnxStereoSource.read()` in
   perception.py to read it with `mmap` (numpy only). Build C with the QNX SDP on the laptop or the
   self-hosted developer desktop on the Pi. See the QNX codelab on multiple camera sources on the Pi 5.
1. Stereo rig (cameras 6 cm apart, parallel) + calibration (`--save-pairs`, `stereo_calibrate.py`).
2. Retrain the CNN on real patches (`--save-patches data/ball`, `data/not_ball`).
3. Full system on the Pi with `--source qnx`; 10 live passes, 0 hits, >= 20 fps.
4. Real-time hardening: measure jitter on QNX; optional C port of control.py
   (SCHED_FIFO, clock_nanosleep, QNX MsgSend/MsgReceive); optional GPIO LEDs per thruster.
5. Demo and pitch: fault demos, backup video, rehearsals.
Fallback if two cameras cannot stream together: one camera, depth from the ball's known size
(Z = f * 0.04 / diameter_px).
