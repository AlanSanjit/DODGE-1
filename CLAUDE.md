# CLAUDE.md: DODGE-1, QNX edge-AI satellite debris avoidance

## What this project is
Hackathon project (QNX edge-AI challenge). A **Raspberry Pi 5 running QNX 8.0** is the flight computer of a
simulated satellite. A person pushes a real **black 35 mm ball** toward **one USB webcam** on the Pi. An **int8
TFLite CNN** (with a geometric shape fallback) recognises the ball, **depth comes from the ball's known size**,
and when the ball is within 15 cm of the satellite's virtual position (30 cm in front of the camera) the Pi
commands simulated **cold-gas thrusters** to sidestep with minimum propellant, then return to the orbit slot.
A **laptop** runs a 3D simulator (Panda3D): rigid-body physics, rendering, dashboard, the Pi's camera feed.
**All sensing and decisions run on the Pi; the laptop only simulates and draws.**
Two-camera stereo mode still exists (`CAMERA_MODE = "stereo"`) but the current setup is mono.

### Judging criteria (every change must keep serving these; README.md answers them in detail)
1. Cannot-fail embedded application: AI isolated from safety, auto-restart, fail-safe states.
2. Real-time / reliability: fixed 50 Hz control loop, priorities, measured jitter.
3. Creative AI: CNN on the edge + hybrid detection + physics-based plausibility + robust min-fuel planning.
4. Runs on embedded hardware: no cloud; the laptop only plays the physical world (hardware-in-the-loop).
Core message: **"The AI suggests, the QNX control process decides."**

## Hardware and environment
- Raspberry Pi 5, QNX SDP 8.0 Quick Start Target Image (TAGID QNXOS_805). Hostname `qnxpi`, user `qnxuser`
  (in groups sensor/521, sudo). Root via `su -` or `sudo`. Python 3.14 on the Pi.
- Pi packages via `apk`: `python3-numpy`, `python3-tflite-runtime`. **Compiler: QNX clang** is preinstalled
  (`clang`, `/usr/lib/llvm22/bin`); there is no `gcc` package. Pi code: numpy + tflite_runtime + stdlib + C.
- Laptop (Windows): `pip install panda3d numpy tensorflow opencv-python`. Laptop SSH alias `qnxpi` in
  `~/.ssh/config` forces `MACs hmac-sha2-256` (Windows OpenSSH + the Pi's sshd otherwise fail with
  "Corrupted MAC on input"). Network: iPhone hotspot, laptop 172.20.10.2, Pi 172.20.10.3 (DHCP, re-check).
- Laptop firewall must allow inbound UDP 5005 (sim), 5008 (live_view), 5009 (sim camera feed).

## Files
| File | Runs on | Role |
|---|---|---|
| `config.py` | both (IDENTICAL copy) | every shared constant |
| `spacemath.py` | both | quaternions, `ball_to_space`, `ball_in_view` (mono/stereo), `HandScenario` (synthetic hand) |
| `perception.py` | Pi, prio 15 | sources (synthetic / QNX mono / QNX stereo) -> candidates -> CNN/shape -> depth -> Kalman -> `track`; preview senders; `--calibrate-f` |
| `control.py` | Pi, prio 40 | 50 Hz guidance / attitude / FDIR / fail-safe -> `cmd` |
| `supervisor.py` | Pi | starts control, capture (C), perception with `on -p`; crash/hang restart with back-off |
| `qnx/stereo_capture.c` | Pi, prio 20 | QNX camera library -> `/dev/shmem/dodge_stereo` (seqlock); `-n 1` mono |
| `qnx/sensor_dodge_mono.conf` / `qnx/sensor_dodge_usb.conf` | Pi | sensor-service configs (1 camera wildcard / 2 cameras bus+dev) |
| `qnx/start_cameras.sh` | Pi (root) | restart the sensor service with a config; checks /dev/sensor |
| `qnx_camera.py` | Pi | `ShmStereo` reader, `to_rgb` (YUYV/UYVY/NV12/RGB...), `python3 qnx_camera.py --snapshot` check; ctypes path (crashes on QNX 8.0, kept for other systems) |
| `sim3d.py` | laptop | physics, views 1-4, dashboard, camera-feed panel, auto reset / rewind |
| `live_view.py` | laptop | full-size annotated Pi camera view; `PreviewReceiver` + `render` reused by sim3d |
| `train_ball_cnn.py` | laptop | trains `ball_cnn_int8.tflite` from synthetic (+ `data/ball`, `data/not_ball`) patches |
| `stereo_calibrate.py` | laptop | checkerboard stereo calibration (stereo mode) |
| `ball_cnn_int8.tflite` | Pi | int8 CNN, 32x32x3 in, 2 classes (0 = ball). Current model: BLACK ball |
| `calib.json` | Pi (generated, git-ignored) | focal length from `--calibrate-f` (read by `Camera()` on start) |
| `RUN.md` / `README.md` | - | run guide / project write-up for the judges |
`sim_satellite.py` (old 2D sim) is retired. Do not revive it.

## Architecture and messages (JSON over UDP unless noted)
Ports: `SIM_PORT 5005` (laptop), `CONTROL_PORT 5006` (Pi control), `SUPERVISOR_PORT 5007` (Pi localhost),
`PREVIEW_PORT 5008` (laptop live_view), `SIM_FEED_PORT 5009` (laptop sim feed). IPs from env `SIM_IP`, `PI_IP`.

| From -> To | type | Rate | Fields |
|---|---|---|---|
| perception -> control (127.0.0.1) and sim | `track` | per frame (~15-20 Hz on Pi) | `seq, detected, reason, pos, vel` (board frame m, m/s REAL; null if not detected/confirmed), `latency_s, classifier, cnn_ms, stereo_ms, n_candidates` (+ when detected `disparity_px, depth_m, range_m, conf, match_ratio, uv, radius_px`) |
| perception -> sim | `depthmap` | every 6th frame, **stereo only** | `w=80,h=60,data` (b64 uint8), `ball` |
| sim -> control | `nav` | 50 Hz | `t` (mission s), `pos, vel` (LVLH m, m/s mission), `q` [w,x,y,z] body->LVLH, `omega`, `h_w`, `mass, prop`, `reset_id` |
| control -> sim | `cmd` | 50 Hz | `seq, phase, motion, target, valves, torque, isolated, branch, fault, threat{range,t_cpa,miss,trigger,approaching}, track, att_err_deg, loop{work_ms,jitter_max_ms,overruns}` |
| control, perception -> supervisor | `hb` | 10 Hz | `name` |
| supervisor -> sim | `procs` | 2 Hz | `procs{name:{pid,prio,restarts,alive,hb_age,last_event}}` |
| perception -> live_view / sim | binary preview | 8-10 Hz | `DPVM`+JSON meta (`mono, w, h, scale, cands_l/r [u,v,r,p], detected, reason, depth_m, size_ratio, depth_src, via, streak, fps, ...`), then `DPVC` + `<u4 fid,i,n>` + 1400-byte zlib chunks of the RGB image (1/`scale`) |
Rule: **never change a message field without updating every sender and receiver in the same change.**
Perception and control never share memory or import each other (isolation story). The only shared memory is
camera frames: stereo_capture (writer) -> perception (read-only mmap).

## Frames, scaling, demo geometry
- Board/camera frame (real, m): origin = camera, X right, Y down, Z forward.
- LVLH (space, m): x radial, y along-track (camera looks along +y), z cross-track. Origin = satellite slot.
- **Satellite slot = `SAT_BOARD_Z` = 0.30 m in front of the camera.**
  `ball_to_space(p) = SPACE_SCALE * CAM_TO_LVLH @ (p - [0,0,0.30])`; `CAM_TO_LVLH`: x=-Y, y=Z, z=X.
- **`SPACE_SCALE = 300`** (1 real cm = 3 m; motion amplified 3x on request; asteroid size 1.75 m radius and
  COLLISION_DIST 4 m kept). **`TIME_SCALE = 15`** (1 real s = 15 mission s): needed so 5 N thrusters can clear
  a hand push with 15 cm of warning; 20 made station keeping chatter.
- Demo: ball ~70 cm -> pushed through 30 cm -> ~10 cm. Dodge at 15 cm before the satellite (asteroid 45 m).
- Panda3D scene = (-z, y, x) of LVLH. Real camera drawn at scene Y = -SAT_BOARD_Z*SPACE_SCALE (-90 m).

## Physics model (sim3d.py `Physics`) - keep it physically honest
Dry 23 kg + 2 kg N2, six body-fixed 5 N thrusters (A/B branches), Isp 65 s, mdot = n F / (Isp g0).
Translation a = R F/m + Clohessy-Wiltshire (n = 2 pi/5554). Rotation: I w' = tau_dist + tau_w - w x (I w + h_w),
tau_dist = -COM_OFFSET x F, wheels 0.1 N m / 1.0 N m s. Stuck valve injected as "+z:A" (stuck OPEN).
Physics 200 steps per real s in its own thread; watchdog: no `cmd` for 150 ms -> commanded valves closed.

## Control logic (control.py)
- Fixed 50 Hz loop, absolute next_t scheduling, jitter/overrun stats. Phases STATION_KEEP -> EVADE -> HOLD_CLEAR
  -> RETURN -> STATION_KEEP; fail-safe NO_NAV_SAFE (no nav 0.3 s: all valves closed).
- Asteroid from track only if fresh (<0.3 s incl. latency) and finite; extrapolated by its age.
- **Trigger**: range <= TRIGGER_RANGE (45 m) AND approaching AND `TRIGGER_FRONT_ONLY` (asteroid moving -y, i.e.
  ball moving toward the camera; pulling the ball back out never triggers). Predictive trigger off (MISS_HORIZON 0).
- **Dodge (`DODGE_MODE = "min_fuel"`)**: cross-track z first, then radial x; never along-track; needs both
  directions of the axis. Robust min-max: candidates for the crossing offset = current offset and predicted
  miss, each widened by `unc = NOISE_SIGMAS * TRACK_POS_SIGMA_M * SPACE_SCALE`; for each side
  d_side = max(side*a) + unc + sqrt(SAFE_MISS^2 - other^2); take the smaller; + DODGE_MARGIN, clipped
  [DODGE_MIN, DODGE_DIST]. Speed v = d / (0.6 t_cpa) clipped [0.25, V_MAX 2]. `"fixed"` = 10 m at V_MAX.
- Return when the asteroid is > CLEAR_RANGE (48 m = 16 cm) from the slot and not approaching, or unseen 1.5 s.
  Return speed V_RETURN 0.3 m/s (cheap).
- Guidance: bang-coast-bang per LVLH axis; deadbands scale with dv per control tick
  (`vel_db = min(max(VEL_DEADBAND, 1.2 dv_tick), 0.8 REACHED_VEL)`), else chatter burns the tank.
- Attitude PD on small-angle error, clipped to wheel limits.
- **FDIR**: measured accel (nav velocity over FDIR_WINDOW 2 mission s) vs commanded; residual > 50% of one
  thruster for 0.3 s -> isolate the branch (leak suspect first, then the commanded valve if the same axis
  trips again), switch to B; never isolate a valve's last branch. **Skips windows where mission time per real
  second is outside 0.8-1.25 x TIME_SCALE** (sim render stalls otherwise caused false isolations).
- Interlock: opposing thrusters never fire together. PROP_DEPLETED disables thrusters.

## Perception (perception.py)
1. Sources: `SyntheticStereoSource` (renders the hand/ball/scene; `mono=True` -> one camera at X=0; clock starts
   at the first frame), `QnxMonoSource` (shm camera 0), `QnxStereoSource` (shm 2 cameras, or ctypes backend).
   `read()` -> (t_capture, left, right|None). No frame for CAM_TIMEOUT_S 2 s -> raise -> supervisor restarts.
2. Candidates: `ball_mask` per `BALL_COLOR` ("black": max channel <= 100 and max-min <= 45; "white"; "orange"),
   connected components on a 4x grid, area cap (mono: up to the ball at MONO_Z_MIN). If no blob passes the CNN:
   `circle_candidates` (gradient Hough, half res, 3 radius bands x 2 peaks, outline circle fit `_refine_circle`).
   `_same_object` dedups circles vs blobs only when near AND similar size.
3. Classifier: CNN P(ball) >= 0.85; else **shape fallback** (`shape_ok`: fill 0.5-0.9, aspect <= 1.6; circle score
   >= 0.5) AND **`isolated`** (ball-coloured inside, not in a ring outside) -> p 0.7, `via = "shape"`.
4. **Mono depth** (`process_mono`): best candidate (CNN before shape); radius = max(area r, bbox/2) capped,
   refined by circle fit; reject if cut by the image edge; r -= MONO_RADIUS_BIAS_PX (0.6); Z = f*(D/2)/r;
   range [0.04, 1.2] m. Synthetic: depth within 1-4% from 8-80 cm, 98% detection on a plain background.
5. **Stereo** (`process`): pairs tried (CNN pairs first) with row tolerance EPIPOLAR_TOL_PX 20, disparity range,
   left/right size ratio; fine disparity along the measured row offset (silhouette for blobs, grey SAD for
   circles), falls back to centre disparity ("ok_centroid"); **physical size check** vs the stereo depth.
6. Kalman (constant velocity); **CONFIRM_FRAMES 2** consecutive consistent detections before `detected` is sent.
7. Previews: `--preview HOST` -> live_view (5008); `SIM_FEED` -> sim (5009, 8 fps, half res).
- Focal length: `--calibrate-f DIST` (ball still, centred, DIST m lens->ball centre) writes calib.json.
  Our webcam: f ~ 470-530 px at 320x240 (default 246 assumed a 66 deg lens: distances came out ~2x short).

## QNX 8.0 camera findings (verified on the Pi 5) - do not relearn these
- **Python ctypes callbacks segfault** (even `qsort` with a Python callback, even as root): never call libcamapi
  with callbacks from Python. The C bridge `qnx/stereo_capture.c` does it; build ON the Pi with
  `clang -O2 -o stereo_capture qnx/stereo_capture.c -lcamapi`. The camera API is declared in the .c (no SDK header).
- `CAMERA_UNIT_n` == n (0 = none). `camera_get_supported_cameras` returns sensor unit numbers.
- A ycbycr USB stream reports `camera_frametype_t` **14** -> stereo_capture trusts the configured format (`-f`).
- `camera_buffer_t` layout differs from the 7.1 docs: {height, width, stride} at **+72** (docs +48); the pixel
  pointer is at +16, timestamp (CLOCK_MONOTONIC us) at +40. stereo_capture finds the size fields by value, checks
  the pointer with a `pipe()` write (EFAULT-safe), and dumps the header on the first frame.
- Sensor service: the boot instance (sensor_rpi5.conf) fails without ribbon cameras and serves nothing ("Another
  instance of SensorService is already running" if not slayed). Start ours WITHOUT `-b external`
  (`sensor -U 521:521 -r /data/share/sensor -c <conf>`; `-b external` needs the Pi platform library line).
  The `sensor` process may not show in `pidin ar | grep`; success = `/dev/sensor/camera1` exists.
- USB device numbers are 2 on the Pi 5 (log `Adding device(046d:0825) bus B devno D`), not the hub's 1.
- Webcams: Logitech 046d:0825, 640x480 YUYV, ~15 fps each (indoor light).
- `on -C` on QNX 8.0 selects a CPU **cluster** (Pi 5: only 0); `-C 1 -C 2 -C 3` -> "Invalid cluster runmask" and
  the process dies. Supervisor uses `on -p` only (`PIN_CPUS = False`).

## Supervisor (supervisor.py)
Launch order control (40) -> capture (20, `-n 1 -u 1` in mono) -> perception (15). CRASH = exited; HANG = no
heartbeat 0.5 s control / 1.5 s perception (after grace 5 s / 30 s) -> SIGKILL + restart. Capture has no
heartbeat (crash detection only; perception detects a stalled camera). Processes failing within 3 s of start
restart with back-off 1, 2, 4, 5 s. Status sent from a separate UNBOUND socket.

## Simulator (sim3d.py)
- Views: 1 chase, 2 top (dodge zone), 3 onboard, **4 camera's eye (default)**. Keys: C feed, M mock ball,
  S stuck valve, F drain prop, R reset, ESC.
- Asteroid drawn only from fresh tracks, **display-smoothed** (AST_SMOOTH_S 0.15, hold 0.5 s, snap > 75 m).
- Encounters logged `[sim3d] encounter over: closest approach ...`; hit = < COLLISION_DIST.
- **Auto reset**: after a dodge, when the ball reaches AUTO_RESET_BALL_Z (12 cm) the pass is recorded and the
  asteroid hidden (REWIND); when the satellite is home in STATION_KEEP for 1 s -> reset (propellant, faults,
  reset_id++). REWIND ends when the ball is back beyond 50 cm or unseen 1.5 s.
- HUD shows `BALL x.xx m from camera`, Pi loop stats, procs, encounters/hits/auto resets, Pi camera feed.

## How to run
See README.md section 5 and RUN.md. Pi: `sudo sh qnx/start_cameras.sh qnx/sensor_dodge_mono.conf` after boot;
laptop `SIM_IP=... PI_IP=... python sim3d.py` first; Pi `SIM_IP=<laptop> python3 supervisor.py --source qnx`.
Laptop-only test: `python sim3d.py` + `python supervisor.py --source synthetic`.

## Testing rules for Claude Code
- After any change run an end-to-end test before saying it works. Headless Panda3D: after importing `sim3d`,
  `loadPrcFileData("", "load-display p3tinydisplay\nwindow-type offscreen\nframebuffer-multisample 0\nmultisamples 0\nsync-video 0")`,
  then `Sim3D()`, start `supervisor.py --source synthetic` as a subprocess, step `taskMgr.step()` for ~80 s,
  save screenshots with `app.win.saveScreenshot(Filename(...))`. p3tinydisplay needs power-of-two textures.
  Use other ports (config.SIM_PORT etc.) if the user's own sim is running.
- Closed-loop controller test (no network): `sim3d.Physics` + `control.Controller` with `poll`/`asteroid`
  monkeypatched; ball path 70 -> 10 cm through the slot at hand speeds 0.10-0.30 m/s, noisy track
  (2 mm, 1.5 cm/s), 6 lateral offsets. Current: 0 hits 0.10-0.20 m/s, marginal 0.25, hits 0.30 (physics limit).
- Success criteria: 0 hits over 5+ dodge passes, no tracebacks in the supervisor log, FDIR only isolates the
  injected stuck valve, crash/hang recover, auto reset after each pass.
- Known past bugs (do not reintroduce): numpy float32 in json.dumps (use default=float); dodging along-track;
  isolating a valve without backup; dodge sized from a noisy predicted miss (use the min-max robust plan);
  bang-bang chatter at high TIME_SCALE (tick-scaled deadbands, below REACHED_VEL); FDIR false isolations from
  sim clock hiccups (rate guard); live_view starving its window on a continuous stream (drain <= 40 ms);
  shape fallback accepting dark poster corners (needs `isolated`); counting the ball pulled back through the
  satellite as a hit (REWIND).
- Tool gotchas on this machine: Bash heredocs turn `\n` inside Python string literals into real newlines - write
  edit scripts with the Write tool; re-Read a file before Edit if anything else modified it.

## Coding conventions
- Pi code: numpy + tflite_runtime + stdlib (+ the C bridge). `config.py` identical on both machines; put new
  constants there. Safety first: every new failure path ends in a defined safe state.
- Don't claim laptop timing numbers as QNX results; measure on the Pi (README has a table to fill in).
- Keep the message table above up to date.

## Current status (2026-11-22)
- Working on real hardware: QNX sensor service + C bridge + mono USB camera (15 fps), perception on the Pi
  (19.9 fps, CNN 1.2 ms, ~20 ms latency), focal calibration, sim with live camera feed. Ball: black 35 mm.
- Tested on the laptop (synthetic camera): full system 12/12 passes clear (7.1-9.7 m), auto reset each pass,
  stuck valve isolated, no false FDIR.
- Not yet measured on the Pi: control-loop jitter/overruns (with and without CPU hogs) -> fill README table.

## Next work
1. Measure QNX timing on the Pi (dashboard + `pidin`), incl. the 3x CPU-hog load test; fill README.
2. Retrain the CNN on real black-ball patches (`--save-patches data/ball`, `data/not_ball`) to reduce reliance
   on the shape fallback.
3. Optional: C port of control.py (SCHED_FIFO, clock_nanosleep, QNX MsgSend/MsgReceive); QNX adaptive
   partitioning to guarantee CPU budgets; GPIO LEDs per thruster; backup demo video.
