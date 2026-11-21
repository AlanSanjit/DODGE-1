# DODGE-1
DODGE-1 (Debris Observation and Dodge Guidance Engine): a Raspberry Pi 5 running the QNX RTOS acts as a satellite flight computer. A TFLite CNN and stereo cameras track a real ping-pong ball, and when it gets within 20 cm the Pi fires simulated thrusters to dodge the "asteroid" in a 3D physics simulation, with fault-tolerant process isolation.

**The AI suggests, the QNX control process decides.** All sensing and decisions run on the Pi; the laptop only simulates physics and draws.

## Quick start
Laptop: `pip install panda3d numpy tensorflow opencv-python`

Single machine (everything on the laptop, synthetic cameras):
```
python sim3d.py              # start first
python supervisor.py         # perception + control, auto-restart
```
Laptop + Pi (wired Ethernet):
```
laptop$ SIM_IP=<laptop-ip> PI_IP=<pi-ip> python3 sim3d.py
pi$     SIM_IP=<laptop-ip> python3 supervisor.py --source synthetic      # later: --source qnx
```
No Pi perception yet? Start `python control.py` and press **M** in the simulator for the mock ball.

Keys: 1 chase, 2 side, 3 onboard, M mock ball, S stuck valve +z:A, F drain propellant, R reset, ESC quit.

| File | Runs on | Role |
|---|---|---|
| `config.py` | both | shared constants (keep identical) |
| `spacemath.py` | both | quaternions, frames, synthetic hand-held ball |
| `perception.py` | Pi | cameras -> CNN (both images) -> stereo depth -> Kalman -> `track` |
| `control.py` | Pi | 50 Hz guidance / attitude / FDIR -> `cmd` |
| `supervisor.py` | Pi | process isolation, priorities, crash/hang restart |
| `sim3d.py` | laptop | rigid-body physics + 3D view + dashboard |
| `train_ball_cnn.py` | laptop | trains and exports `ball_cnn_int8.tflite` |
| `stereo_calibrate.py` | laptop | checkerboard calibration -> `calib.json`, `rectify_maps.npz` |

See `CLAUDE.md` for architecture, message formats, physics, and status.
