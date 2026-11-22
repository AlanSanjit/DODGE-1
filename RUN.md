# DODGE-1: full run-down

The Raspberry Pi 5 (QNX) is the flight computer: ONE USB camera (`CAMERA_MODE = "mono"`) -> CNN /
shape check finds the **black 35 mm ball** -> depth from its known size -> threat assessment -> thruster commands (min-fuel dodge, then
return to the orbit slot). The laptop is the universe: physics, 3D view, dashboard.

Network: both on the iPhone hotspot. **Laptop 172.20.10.2, Pi 172.20.10.3.**
Check every session: laptop `ipconfig` (Wi-Fi IPv4), Pi `ifconfig`. If they changed, use the new
numbers everywhere below and update `HostName` in `C:\Users\alans\.ssh\config`.

Conventions: **LAPTOP** = PowerShell on the laptop, **PI** = `ssh qnxpi` (or the Pi's own terminal).

---------------------------------------------------------------------------------------------------
## A. One-time setup (done already; repeat a step only if something changed)

A1. LAPTOP, admin PowerShell, firewall (once):
```powershell
New-NetFirewallRule -DisplayName "DODGE-1 sim" -Direction Inbound -Protocol UDP -LocalPort 5005 -Action Allow -Profile Any
New-NetFirewallRule -DisplayName "DODGE-1 preview" -Direction Inbound -Protocol UDP -LocalPort 5008 -Action Allow -Profile Any
New-NetFirewallRule -DisplayName "DODGE-1 sim feed" -Direction Inbound -Protocol UDP -LocalPort 5009 -Action Allow -Profile Any
```
A2. PI, libraries (once; root password needed): `su -` then `apk add python3-numpy python3-tflite-runtime`, `exit`.

A3. LAPTOP, copy the code to the Pi (after every code change; add `-O` after `scp` if it errors):
```powershell
cd C:\Users\alans\Downloads\DODGE-1
ssh qnxpi "mkdir -p DODGE-1/qnx"
scp config.py spacemath.py perception.py control.py supervisor.py qnx_camera.py ball_cnn_int8.tflite RUN.md qnxpi:DODGE-1/
scp qnx/sensor_dodge_usb.conf qnx/sensor_dodge_mono.conf qnx/start_cameras.sh qnx/stereo_capture.c qnx/README.md qnxpi:DODGE-1/qnx/
```
A4. PI, build the camera bridge (after every change to stereo_capture.c):
```sh
cd ~/DODGE-1
clang -O2 -o stereo_capture qnx/stereo_capture.c -lcamapi
```

---------------------------------------------------------------------------------------------------
## B. Every session

### B1. Hardware
- ONE webcam (sensor unit 1) in a **blue** USB port, fixed, centred in front of where the ball moves.
  (Two-camera stereo is still possible: `CAMERA_MODE = "stereo"`, `qnx/sensor_dodge_usb.conf`.)
- A **plain light background** behind where the ball will move (white wall, paper, light cloth).
  A black ball in front of dark things (monitor, dark clothes, shadows) is hard to see.
- Pi and laptop connected to the hotspot; laptop on its charger.

### B2. PI: start the cameras (after every boot)
```sh
cd ~/DODGE-1
sudo sh qnx/start_cameras.sh qnx/sensor_dodge_mono.conf    # expect: "OK: 1 camera(s) registered"
./stereo_capture -l                                          # expect: "camera units available: 1"
```

### B3. (Recommended) Check what the Pi sees before flying
LAPTOP:
```powershell
cd C:\Users\alans\Downloads\DODGE-1
python live_view.py
```
PI:
```sh
cd ~/DODGE-1
./stereo_capture -n 1 &
python3 perception.py --source qnx --preview 172.20.10.2
```
Hold the ball 10-60 cm in front of the camera. Pass: green box on the ball, "BALL CONFIRMED [cnn] or
[shape]: depth ..." about right, ~15 fps. Press **m**: the ball lights up pink, the background does not.

**Focal-length calibration (once per camera / resolution):** Ctrl+C perception, then hold the ball
still, centred, at a tape-measured distance (lens to ball centre), e.g. 0.30 m, and run:
```sh
python3 perception.py --source qnx --calibrate-f 0.30
```
It writes `calib.json` (read on every start). Re-check: the live view depth now matches the tape.
Then stop: Ctrl+C, and `slay stereo_capture`. (Keep live_view.py open if you like.)

### B4. LAPTOP: start the universe (first)
```powershell
cd C:\Users\alans\Downloads\DODGE-1
$env:SIM_IP="172.20.10.2"; $env:PI_IP="172.20.10.3"; python sim3d.py
```
A window opens with a red banner "NO FLIGHT COMPUTER" - expected until B5.

### B5. PI: start the flight computer
```sh
cd ~/DODGE-1
slay stereo_capture 2>/dev/null
SIM_IP=172.20.10.2 python3 supervisor.py --source qnx --preview 172.20.10.2
```
The supervisor starts and watches 3 processes: control (priority 40), stereo_capture (priority 20,
with `-n 1` in mono mode) and perception (priority 15).
In the simulator the red banner disappears; top-right shows `PERCEPTION (Pi)` and the processes `UP`.

### B6. Fly
The satellite sits **30 cm in front of the camera** (its orbit slot); the dodge starts when the ball is
within **15 cm of the satellite** (ball ~45 cm from the camera).
1. Hold the black ball ~70 cm in front of the camera, on the camera's axis.
2. Push it steadily toward the camera over ~4 s (about 0.15 m/s; faster than ~0.2 m/s the 5 N
   thrusters cannot clear in time), straight through the 30 cm point, down to ~10 cm.
   Closer than ~8 cm it no longer fits in the image.
3. Hold it there: once the satellite is back in its slot the sim AUTO RESETS (propellant refilled).
4. Bring the ball back out to >= 50 cm (the sim says "RESET DONE - bring the ball back out"); the
   asteroid is hidden meanwhile, so pulling it back through the satellite is not a collision.
5. Repeat from 2.
3. Pull it back, or take it out of view.

What you see in the simulator (press **2** for the top view):
- the asteroid appears (ball distance x 100 = metres); the line to the satellite turns red = threat;
- `PHASE EVADE`: the Pi fires the thrusters just enough to make the predicted closest approach >= 8 m,
  only as fast as needed (min-fuel); dashboard shows valves, motion, target;
- `HOLD_CLEAR` while the ball is close, then `RETURN` (slow, 0.3 m/s) once it is gone or receding,
  back to the green slot cube -> `STATION_KEEP`;
- bottom line: encounters / hits / closest approach. Watch `propellant`; press **R** when low.

Keys: 1 chase view, 2 top view, 3 onboard, S stuck thruster, F drain fuel, R reset, ESC quit.
Do NOT press M (mock ball) - the ball comes from the Pi.

### B7. Fault demos (judging: "cannot fail")
Second PI window (`ssh qnxpi`), PIDs are printed by the supervisor:
```sh
kill -9 <perception pid>     # crash: restarted; satellite stays safe meanwhile
kill -STOP <control pid>     # hang: detected in 0.5 s, killed, restarted; sim watchdog closes valves
```
Simulator key **S**: stuck thruster +z:A -> Pi isolates it and switches to the backup branch.
Turn the laptop's Wi-Fi off for 2 s: Pi goes NO_NAV_SAFE (valves closed), then recovers.

### B8. Stop
PI: Ctrl+C (supervisor stops everything). LAPTOP: ESC in the simulator.
A Pi reboot restores its default camera configuration (run B2 again next time).

---------------------------------------------------------------------------------------------------
## C. If something is wrong

| You see | Do |
|---|---|
| Red "NO FLIGHT COMPUTER" stays | supervisor not running, or wrong PI_IP on the laptop |
| Satellite moves but no PERCEPTION panel | laptop firewall (A1) or wrong SIM_IP on the Pi |
| "CONTROL LINK LOST" flickers | Wi-Fi latency spikes: keep the phone close, Personal Hotspot screen open |
| `start_cameras.sh` says PROBLEM | `sudo slog2info -b sensor_service \| tail -n 20`; replug cameras; reboot |
| `stereo_capture` shows `bad` increasing | send its first lines (header dump) |
| Live view `no_candidates_*` | press m: ball not pink -> raise `BLACK_MAX_V` (config.py) to 80, more light |
| Live view `cnn_rejected_*` | CNN trained on synthetic images: retrain on real ones (qnx/README.md step 7) |
| Live view `epipolar_mismatch` | align camera heights or calibrate (qnx/README.md step 6) |
| Live view `disparity_out_of_range` | ball closer than ~10 cm, or cameras swapped: add `--swap-lr` |
| Perception below ~10 fps on the Pi | with a clean background set `CIRCLE_MAX = 0` in config.py, copy it, restart |

Improving accuracy (optional, see qnx/README.md): step 6 stereo calibration (checkerboard),
step 7 retrain the CNN on real camera images.
