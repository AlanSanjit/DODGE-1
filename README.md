# DODGE-1: Debris Observation and Dodge Guidance Engine

**A Raspberry Pi 5 running the QNX real-time operating system acts as the flight computer of a
satellite.** A USB camera on the Pi watches a real black ping-pong ball. An int8 CNN running on the
Pi recognises the ball, measures how far away it is, and when that "asteroid" gets within 15 cm of the
satellite's position the Pi fires simulated cold-gas thrusters to sidestep it with the least
propellant possible, then flies the satellite back to its original orbit slot. A laptop plays "the
universe": rigid-body physics, a 3D view and a dashboard with the Pi's live camera feed.

> **The AI suggests, the QNX control process decides.**
> Every sensing and every actuator decision runs on the QNX Pi. The laptop only simulates physics and
> draws. It makes no decisions.

```
 real world                     Raspberry Pi 5 - QNX 8.0 (flight computer)                        laptop
 ----------                     ------------------------------------------                        ------
               USB   +------------------------+  /dev/shmem  +------------------+  UDP   +------------------+
 black ball -------> | QNX sensor service     |------------->| perception (P15) | track  | control (P40)    |
 (hand-held)         | + camera library       |  (read-only) | CNN + depth +    |------->| 50 Hz loop:      |
                     | stereo_capture.c (P20) |              | Kalman filter    |        | threat, dodge,   |
                     +------------------------+              +--------+---------+        | attitude, FDIR,  |
                                                                      |                  | fail-safe        |
       supervisor.py: starts all three with QNX priorities,           |                  +--------+---------+
       restarts any of them on crash / hang (heartbeats)              |                      cmd  |  ^  nav
                                                                      | camera feed               v  |
                                                                      +-----------------> +------------------+
                                                                                          | sim3d (laptop):  |
                                                                                          | physics, 3D view,|
                                                                                          | dashboard        |
                                                                                          +------------------+
```

---

## 1. Why QNX: what an RTOS gives a flight computer

A desktop OS is built for throughput and fairness. A flight computer needs **guarantees**: the control
loop must run on time even when the AI is busy, a crashing component must never take the system down,
and a fault must always end in a known safe state. That is what QNX is designed for.

| QNX property | What it means | Why it matters for DODGE-1 |
|---|---|---|
| **Microkernel architecture** | The kernel only does scheduling, IPC and memory protection. Drivers and services (camera stack, USB, networking, filesystems) run as ordinary user-space processes. | The USB camera stack and the sensor service can fail or restart without touching the kernel or the control loop. Our own software follows the same pattern: perception, camera capture and control are separate processes. |
| **Deterministic, priority-preemptive scheduling** | 256 priority levels; the highest-priority ready thread always runs and preempts lower ones immediately. Scheduling latency is bounded and does not depend on load. | The 50 Hz guidance/attitude loop runs at priority 40, the CNN at 15. However heavy the vision work gets, it **cannot delay control**. That is the core of "AI isolated from safety". |
| **Memory protection between processes** | Every process has its own address space. A bug or crash in one cannot corrupt another. | A crash in the CNN, numpy or the camera bridge can never corrupt the controller's state. Control shares no memory with anything. |
| **POSIX API** | Standard `fork/exec`, signals, `shm_open`/`mmap`, `clock_gettime(CLOCK_MONOTONIC)`, sockets. | Python, numpy, TFLite and a small C program run unchanged. Development is quick, and the result is a real RTOS deployment. |
| **High-availability tooling** | Process supervision, heartbeats, `slay`/`SIGKILL`, `slog2` system logging, `pidin` introspection. | Our supervisor restarts crashed or hung processes in under a second, and `slog2info` let us debug the camera stack on the Pi. |
| **Safety-critical heritage** | QNX is used in automotive, industrial, medical and robotics systems, and QNX offers safety-certified variants of its OS. | The same design rules apply to a spacecraft: deterministic timing, isolation and defined failure behaviour. |
| **Edge AI on embedded hardware** | Runs on ARM boards such as the Raspberry Pi 5. Open-source ports (Python 3, numpy, TFLite runtime, clang) install with `apk`. | The whole pipeline (camera, CNN, decision, actuation) runs on a palm-sized board with **no cloud**. |

---

## 2. How this project answers the QNX challenge criteria

### Criterion 1: a "cannot-fail" embedded application
*AI isolated from safety, automatic restart, fail-safe states.*

**AI isolated from safety-critical control**
- Perception (camera, CNN, depth) and control are **separate QNX processes**. They never import each other or share memory, and they communicate only through small UDP messages.
- **Control owns every actuator decision.** It treats perception's output as an untrusted suggestion: it ignores tracks older than 0.3 s and rejects malformed or non-finite values.
- Perception must confirm a detection in **two consecutive frames** before control is told about it, so a one-frame false positive can never move the satellite.
- Control only reacts to an object that is **approaching from the front** and inside the trigger range. A stationary false detection further away never fires a thruster.

**Automatic restart (`supervisor.py`)**
- The supervisor starts control, camera capture and perception, and watches them with **heartbeats** (10 Hz).
- **Crash** (the process exited) → restarted immediately.
- **Hang** (no heartbeat for 0.5 s for control, 1.5 s for perception) → killed with `SIGKILL`, which works even on a stopped process, then restarted.
- A process that keeps failing at start-up is restarted with **back-off** (1, 2, 4, 5 s) instead of a tight loop.
- Demo: `kill -9 <perception pid>` or `kill -STOP <control pid>` on the Pi; the dashboard shows the restart count.

**Fail-safe states**

| Failure | Defined safe reaction |
|---|---|
| No navigation data for 0.3 s (link loss, simulator stalled) | Control enters `NO_NAV_SAFE`: **all valves closed**, no torque. Recovers to `RETURN` when data comes back. |
| No commands for 150 ms (control crashed or hung) | Simulator watchdog **closes all valves** (a real thruster driver would do the same). |
| Camera stops delivering frames | Perception exits → supervisor restarts it. Control sees no fresh track = no threat, so the satellite holds its slot. |
| Ball lost from view | Kalman filter resets; after 1.5 s without the ball, the satellite returns home. |
| **Stuck-open thruster** (key `S` injects `+z:A`) | **FDIR**: commanded and measured acceleration are compared over 2 mission seconds. A residual above 50% of one thruster for 0.3 s → that branch is isolated and the **backup branch B** takes over. The last branch of a valve is never isolated, so the satellite keeps both "go" and "brake" on every axis. |
| Simulator clock hiccup (Wi-Fi burst, render stall) | FDIR **skips** windows where mission time did not advance steadily, rather than blaming a healthy thruster. |
| Propellant exhausted | `PROP_DEPLETED`: thrusters disabled, defined state. |
| Opposing thrusters | Interlock: `+x` and `-x` (and so on) can never fire together. |

### Criterion 2: real-time behaviour and reliability
*Fixed-rate loop, priorities, measured jitter.*

- **Fixed 50 Hz control loop** with **absolute scheduling** (`next_t += period`, so no drift accumulates) on `CLOCK_MONOTONIC`. Each cycle measures **work time, wake-up jitter and overruns**, and sends them to the dashboard (`Pi loop work / jitter max / overruns`).
- **QNX priorities** set by the supervisor with `on -p`:

  | Process | Priority | Role |
  |---|---|---|
  | `control.py` | **40** | guidance, attitude, FDIR (safety critical) |
  | `stereo_capture` (C) | 20 | camera frames to shared memory |
  | `perception.py` | 15 | CNN and depth (best effort) |

  Verify on the Pi with `pidin | grep -E "python|stereo"`.
- **Load test (the "AI can't starve control" proof):** while the demo runs, start `python3 -c "while 1: pass" &` three times on the Pi. Perception slows down, but control's jitter and overruns stay where they were, because QNX's preemptive scheduler always runs the priority-40 loop first.
- **Core pinning:** QNX 8.0's `on -C` selects a CPU *cluster*, and the Pi 5 has a single cluster, so per-core pinning via `on -C` doesn't apply. Priority scheduling provides the timing guarantee (`config.PIN_CPUS` can re-enable `-C`).
- **Deterministic timing parameters:** the dodge logic accounts for command latency (braking margin), and its deadbands scale with the speed change a single control tick produces, which prevents bang-bang chatter.

**Measured on the Pi (fill in from the dashboard during your run):**

| Metric | Value on QNX / Pi 5 |
|---|---|
| Control loop work time | ___ ms |
| Control loop max jitter | ___ ms |
| Overruns (deadline misses) | ___ |
| The same three, with 3 CPU hogs running | ___ / ___ / ___ |
| Perception rate (live view header) | **19.9 fps** (measured) |
| CNN inference per candidate (int8 TFLite) | **1.2 ms** (measured) |
| Camera-to-decision latency | **~20 ms** perception latency (measured) |

### Criterion 3: creative use of AI
- **An int8 CNN on the edge:** a small convolutional network (32×32×3 input, 2 classes) trained on the laptop, **quantised to int8** and run on the Pi with `tflite_runtime`, at about 1.2 ms per candidate.
- **A hybrid detection pipeline:**
  1. A colour test for dark, colourless pixels, plus a numpy **gradient-Hough circle detector**, propose candidates.
  2. The CNN classifies each candidate.
  3. A geometric shape check is a fallback when the CNN is unsure; it needs a round, isolated blob.
- **Depth from known size, with physics as a sanity check:** with one camera, distance comes from the ball's known **35 mm diameter**: Z = f·17.5 mm / r. The radius comes from a circle fit on the ball's outline. Balls cut off by the image edge are rejected. The focal length is calibrated in one command.
  - **Stereo mode** (two cameras, still supported) adds a **physical-size check**: the CNN sees rescaled crops and can't tell a 1 cm button up close from a far ball, but stereo depth can. Small round decoys were wrongly accepted in 3 of 150 frames, down from 141 of 150.
- **Tracking and prediction:** a constant-velocity **Kalman filter** turns detections into position and velocity. Control predicts the closest approach, and plans a **robust minimum-fuel dodge**: it clears both the asteroid's current offset and its predicted miss, widened by measurement noise, on whichever side needs the smaller move. Fuel ∝ Δv, so it moves only as far and as fast as needed, and returns slowly.
- **Synthetic training data:** a ray-traced scene renderer (hand, ball, look-alike posters and boxes, random lighting) generates labelled training patches through the exact Pi pipeline. Real patches can be added with `--save-patches`.

### Criterion 4: runs on embedded hardware (no cloud)
- Everything that senses or decides runs on the **Raspberry Pi 5 under QNX 8.0**: camera driver stack, C capture bridge, CNN, depth, Kalman filter, guidance, attitude control, FDIR and supervisor.
- The laptop is **hardware-in-the-loop**: it receives thruster and wheel commands, integrates the physics (Newton, Euler's equations, Clohessy-Wiltshire orbit dynamics, propellant mass flow from Isp), and sends back navigation data, like a spacecraft simulator on a test bench.
- Pi software uses only numpy, tflite_runtime and the Python standard library, plus a ~540-line C program, all installed with QNX `apk`.

---

## 3. How QNX is used, in detail

### QNX camera stack (sensor framework + camera library)
- **Sensor service:** QNX's `sensor` service owns the USB (UVC) webcam. We wrote our own config, `qnx/sensor_dodge_mono.conf` (`type = usb_camera`, YUYV 640×480 at 30 fps). `qnx/start_cameras.sh` stops the boot-time service, which expects Pi ribbon cameras and serves nothing without them, and starts ours.
- **Camera library (`libcamapi`):** `qnx/stereo_capture.c` calls `camera_open` → `camera_set_vf_mode(VIDEO)` → `camera_start_viewfinder` with a **C callback** for every frame.
- **POSIX shared memory:** the callback publishes each frame to `/dev/shmem/dodge_stereo` behind a **sequence lock**, so perception never reads a half-written frame. Perception maps it read-only with `mmap`.

**What we found about QNX 8.0 on the Pi 5** (documented in `CLAUDE.md`):
- Python `ctypes` callbacks crash on this image, even as root, so the camera callback had to be in C. It's compiled on the Pi with QNX's clang.
- `CAMERA_UNIT_n` has the value *n*: unit 0 means none.
- A YUYV stream reports `camera_frametype_t` 14.
- The frame-size fields in `camera_buffer_t` sit at a different offset than in the 7.1 docs. `stereo_capture` finds the fields by value on the first frame, checks the pixel pointer safely with a `pipe()` write (which fails with `EFAULT` instead of crashing on a bad address), and dumps the header for diagnosis.
- The boot-time sensor service needs `-b external` only for the Pi ribbon-camera platform library. A USB-only config must start without it.

### QNX process model and scheduling
- `supervisor.py` launches each process as `on -p <priority> <program>`. QNX's priority-preemptive scheduler then guarantees the control loop over everything else.
- Heartbeats run over loopback UDP. A hang is detected by heartbeat age and handled with `SIGKILL`, which works on stopped processes. A crash is detected by process exit.
- `pidin` shows priorities, and `slog2info -b sensor_service` shows the camera stack's log.

### QNX packages
`apk` (QNX open-source ports): `python3`, `python3-numpy`, `python3-tflite-runtime`, `clang` (preinstalled), and the sensor-framework packages that come with the Quick Start Target Image.

---

## 4. The demo

**Real layout, measured along the camera's line of sight:**
```
camera ──── 10 cm ──────── 30 cm ───────── 45 cm ─────────────── 70 cm
            ball stops     SATELLITE       dodge starts          ball starts
            (auto reset)   (orbit slot)    (15 cm before it)
```
1. Push the black ball from about 70 cm toward the camera over about 4 s (about 0.15 m/s).
2. In the game, the asteroid approaches the satellite. Real motion is amplified **3×**: 1 cm = 3 m.
3. At 15 cm before the satellite, the Pi plans and fires a **minimum-fuel sidestep**.
4. The asteroid passes through the satellite's old position.
5. The satellite flies **back to its orbit slot**.
6. With the ball at the camera and the satellite home, the simulator **auto-resets** (propellant refilled). Pull the ball back out to start again; that never counts as a collision.

**Physics limit:** with 5 N thrusters and 15 cm of warning, the satellite clears a ball pushed at up to about **0.2–0.25 m/s**.

---

## 5. How to run everything

The full procedure, with troubleshooting, is in **[`RUN.md`](RUN.md)**. In short:

### Hardware and network
- Raspberry Pi 5 with the **QNX SDP 8.0 Quick Start Target Image** (hostname `qnxpi`, user `qnxuser`).
- One USB webcam in a **blue** USB port, a black ball (35 mm), and a plain **light** background.
- Pi and laptop on the same network. Laptop **172.20.10.2**, Pi **172.20.10.3** in our setup; check with `ipconfig` and `ifconfig`.

### One-time setup
**Laptop**, admin PowerShell, to open the ports for the simulator, live view and camera feed:
```powershell
pip install panda3d numpy tensorflow opencv-python
New-NetFirewallRule -DisplayName "DODGE-1 sim"      -Direction Inbound -Protocol UDP -LocalPort 5005 -Action Allow -Profile Any
New-NetFirewallRule -DisplayName "DODGE-1 preview"  -Direction Inbound -Protocol UDP -LocalPort 5008 -Action Allow -Profile Any
New-NetFirewallRule -DisplayName "DODGE-1 sim feed" -Direction Inbound -Protocol UDP -LocalPort 5009 -Action Allow -Profile Any
```
**Pi**, as root:
```sh
apk add python3-numpy python3-tflite-runtime
```
**Get the code onto the Pi.** Either `git clone` this repo on the Pi, or copy from the laptop:
```powershell
ssh qnxpi "mkdir -p DODGE-1/qnx"
scp config.py spacemath.py perception.py control.py supervisor.py qnx_camera.py ball_cnn_int8.tflite qnxpi:DODGE-1/
scp qnx/sensor_dodge_mono.conf qnx/start_cameras.sh qnx/stereo_capture.c qnxpi:DODGE-1/qnx/
```
**Build the camera bridge** (Pi):
```sh
cd ~/DODGE-1 && clang -O2 -o stereo_capture qnx/stereo_capture.c -lcamapi
```
**Calibrate the focal length** (Pi, once per camera). Hold the ball still and centred at exactly 30 cm from the lens to the ball's centre:
```sh
sudo sh qnx/start_cameras.sh qnx/sensor_dodge_mono.conf
./stereo_capture -n 1 &
python3 perception.py --source qnx --calibrate-f 0.30
slay stereo_capture
```

### Every session
1. **Pi, after each boot, start the camera service:**
   ```sh
   cd ~/DODGE-1
   sudo sh qnx/start_cameras.sh qnx/sensor_dodge_mono.conf
   ```
   Expect `OK: 1 camera(s) registered`.
2. **Laptop, start the universe first:**
   ```powershell
   $env:SIM_IP="172.20.10.2"; $env:PI_IP="172.20.10.3"; python sim3d.py
   ```
3. **Pi, start the flight computer:**
   ```sh
   SIM_IP=172.20.10.2 python3 supervisor.py --source qnx
   ```
   The red "NO FLIGHT COMPUTER" banner disappears, and the camera feed appears bottom-right.
4. **Fly:** push the ball from 70 cm to 10 cm and watch the dodge, return and auto-reset.

**Simulator keys:**
- **1** chase view, **2** top view, **3** onboard view, **4** camera's-eye view
- **C** show or hide the camera feed
- **S** stuck thruster
- **F** drain propellant
- **R** reset
- **ESC** quit

**Optional extra window:** `python live_view.py` shows the Pi's annotated camera at full size. Add `--preview 172.20.10.2` to the supervisor command for it.

### Without the Pi (laptop only)
```powershell
python sim3d.py                          # terminal 1
python supervisor.py --source synthetic  # terminal 2: synthetic camera, same pipeline
```
Or run `python control.py` and press **M** in the simulator for a scripted mock ball.

### Fault demos
| Do | See |
|---|---|
| `kill -9 <perception pid>` on the Pi | supervisor: `perception CRASH -> restart`; satellite stays safe |
| `kill -STOP <control pid>` on the Pi | `control HANG` after 0.5 s → killed and restarted; valves closed meanwhile |
| Key **S** in the simulator | `FDIR: ISOLATED +z:A -> backup B`; next dodges still clear |
| Laptop Wi-Fi off for 2 s | Pi: `NO_NAV_SAFE` (valves closed), then `nav back -> RETURN` |
| 3× `python3 -c "while 1: pass" &` on the Pi | perception slows down; control jitter unchanged |

---

## 6. Results (tested)

| Test | Result |
|---|---|
| Camera on QNX: USB webcam via sensor service + `libcamapi` + C bridge | 640×480 YUYV at 15 fps, 0 bad frames, 14–34 ms capture to Python |
| Perception on the Pi (from the live view) | 19.9 fps, CNN 1.2 ms, latency about 20 ms, ball confirmed at CNN 1.00 |
| int8 CNN, held-out patches | 99.97% (black ball model) |
| Mono depth (synthetic, plain background) | within 1–4% from 8 to 80 cm; 98% detection; 0 of 200 false alarms |
| Closed loop: real controller + physics + noisy track, ball through the satellite | **0 hits** for hand speeds 0.10–0.20 m/s (5.8–8 m clearance), about 380 g propellant per pass |
| Full system (simulator + supervisor + perception + control), 2 runs × 6 passes | **12 of 12 clear** (7.1–9.7 m), 12 auto-resets, stuck thruster isolated, 0 false FDIR |
| Fault recovery | perception crash → restart; control hang → killed in 0.5 s and restarted; stuck valve → backup branch |

Numbers marked as synthetic or closed loop were measured on the laptop; QNX timing must be measured on the Pi (table in Criterion 2).

---

## 7. Repository

| File | Runs on | Role |
|---|---|---|
| `config.py` | both (keep identical) | every shared constant: network, scaling, physics, control, vision |
| `spacemath.py` | both | quaternions, frames (`ball_to_space`), synthetic hand path |
| `perception.py` | Pi (priority 15) | camera → candidates → CNN / shape → depth → Kalman → `track`; preview streams; focal calibration |
| `control.py` | Pi (priority 40) | 50 Hz loop: threat, min-fuel dodge, return, attitude PD, FDIR, fail-safe |
| `supervisor.py` | Pi | launches processes with QNX priorities; crash/hang restart with back-off |
| `qnx/stereo_capture.c` | Pi (priority 20) | QNX camera library → `/dev/shmem` bridge (1 or 2 cameras) |
| `qnx/sensor_dodge_mono.conf`, `qnx/sensor_dodge_usb.conf` | Pi | QNX sensor-service configs (1 / 2 USB cameras) |
| `qnx/start_cameras.sh` | Pi | (re)start the sensor service with our config |
| `qnx_camera.py` | Pi | shared-memory reader + pixel-format conversion; camera self-check |
| `sim3d.py` | laptop | physics, 3D views, dashboard, camera-feed panel, auto-reset |
| `live_view.py` | laptop | full-size live view of the Pi camera with CNN overlays |
| `train_ball_cnn.py` | laptop | trains and exports `ball_cnn_int8.tflite` |
| `stereo_calibrate.py` | laptop | checkerboard stereo calibration (stereo mode) |
| `RUN.md` | | step-by-step run guide and troubleshooting |
| `CLAUDE.md` | | engineering notes: architecture, messages, QNX findings, test rules |

Two-camera stereo mode is still available: set `CAMERA_MODE = "stereo"` in `config.py` and use `qnx/sensor_dodge_usb.conf`.
