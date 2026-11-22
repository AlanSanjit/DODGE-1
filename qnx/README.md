# Two USB cameras on the QNX Pi 5: live stereo + CNN check

Goal: both USB webcams streaming through the QNX sensor service, the CNN confirming the black
ball in both images on the Pi, and a live view of it on the laptop.

Pipeline: USB webcams -> QNX `sensor` service (UVC) -> QNX camera library `libcamapi` ->
`stereo_capture` (small C program, frames into `/dev/shmem/dodge_stereo`) -> `perception.py`
(CNN + stereo depth) -> `--preview` stream -> `live_view.py` on the laptop.
(Python ctypes callbacks crash on the QNX 8.0 Pi image - `qnx_camera.py --selftest` - hence the C bridge.)

## 0. Copy the files (laptop)
```powershell
cd C:\Users\alans\Downloads\DODGE-1
ssh qnxpi "mkdir -p DODGE-1/qnx"
scp config.py spacemath.py perception.py control.py supervisor.py qnx_camera.py ball_cnn_int8.tflite qnxpi:DODGE-1/
scp qnx/sensor_dodge_usb.conf qnx/start_cameras.sh qnx/stereo_capture.c qnxpi:DODGE-1/qnx/
```
Allow the preview stream through the laptop firewall (admin PowerShell, once):
```powershell
New-NetFirewallRule -DisplayName "DODGE-1 preview" -Direction Inbound -Protocol UDP -LocalPort 5008 -Action Allow -Profile Any
```

## 1. Mount and plug in the cameras
- Fix both webcams on one rigid bar, 7 cm apart (lens centre to lens centre), facing the same way,
  same height, lenses parallel. Measure the real distance; it goes into calibration.
- Plug them into the two **blue USB 3.0 ports** (separate controllers = enough bandwidth).
- Note which camera is LEFT (seen from behind the cameras, looking where they look).

## 2. Find each camera's bus and device (Pi)
```sh
usb -vvv
```
For each webcam (look for its Vendor/Product), note **Upstream Host Controller** (= bus) and
**Device Address** (= device). Unplug one camera and run `usb` again if you can't tell them apart.

## 3. Configure and start the sensor service (Pi, as root)
```sh
cd ~/DODGE-1
vi qnx/sensor_dodge_usb.conf      # replace LEFT_BUS LEFT_DEV RIGHT_BUS RIGHT_DEV with the numbers
su -c "sh /home/qnxuser/DODGE-1/qnx/start_cameras.sh"
```
The script prints the sensor service log and `/dev/sensor`. A reboot restores the image's default
camera configuration.

## 4. Build and run the capture program, check both cameras (Pi, as qnxuser)
```sh
cd ~/DODGE-1
clang -O2 -o stereo_capture qnx/stereo_capture.c -lcamapi      # once (and after every change)
./stereo_capture -l                                          # lists camera units: expect 1 2
./stereo_capture &                                           # streams into /dev/shmem/dodge_stereo
python3 qnx_camera.py --snapshot                             # reads the shared memory
```
Expected: `left (unit 1): YCBYCR 640x480 ... ~30 fps  bad 0`, same for right, and `cam_left.ppm`,
`cam_right.ppm` saved (view on the laptop: `scp qnxpi:DODGE-1/cam_*.ppm .`).
`stereo_capture` also prints fps every 5 s. Stop it with `slay stereo_capture` (or `kill %1`).
The supervisor starts it automatically for `--source qnx` (step 8), so stop your manual one first then.

## 5. Live view with the CNN (the real test)
Laptop:
```powershell
cd C:\Users\alans\Downloads\DODGE-1
python live_view.py
```
Pi:
```sh
cd ~/DODGE-1
./stereo_capture &          # if not already running
python3 perception.py --source qnx --preview 172.20.10.2 --epipolar-tol 20
```
(`--epipolar-tol 20` tolerates imperfect camera alignment until you calibrate; drop it after step 6.)

Hold the ball 25-60 cm in front of the cameras, centred between them. Pass:
- green `CNN 0.9x` circle on the ball in BOTH images, red on other dark things,
- green `BALL CONFIRMED ... depth` line, depth roughly matching a tape measure,
- `Pi perception` >= 20 fps.

| Live view says | Do this |
|---|---|
| `no_candidates_*` | press **m**: the ball must light up (pink overlay). If not: more light, or adjust `WHITE_MIN_MEAN` / `WHITE_MAX_CHROMA` in config.py. (white ball) / `BLACK_MAX_V` / `BLACK_MAX_CHROMA` (black ball). A plain LIGHT background behind the black ball helps most |
| `cnn_rejected_*` | the synthetic-trained CNN does not like your real images: collect real patches and retrain (step 7) |
| `epipolar_mismatch` (rows differ N px) | align the cameras' height/tilt, or calibrate (step 6) |
| `disparity_out_of_range` | ball closer than ~9-11 cm (not fully visible to both cameras), or left/right swapped: add `--swap-lr` |
| depth consistently off by a factor | wrong focal length/baseline: calibrate (step 6) |

## 6. Stereo calibration (makes depth accurate)
Print a checkerboard (9x6 inner corners, measure the square size). Pi:
```sh
python3 perception.py --source qnx --save-pairs pairs --epipolar-tol 50
```
Move the board around in view of both cameras for ~1 minute (25+ positions, tilted, near/far), Ctrl+C.
Laptop:
```powershell
scp -r qnxpi:DODGE-1/pairs .
python stereo_calibrate.py --pairs pairs --cols 9 --rows 6 --square 0.025
scp calib.json rectify_maps.npz qnxpi:DODGE-1/
```
Check: printed baseline ~ 7 cm, rms < 0.5 px. Restart step 5 without `--epipolar-tol`.

## 7. Retrain the CNN on your real camera (if it rejects the real ball)
Pi (ball in view, moving around, no other dark things):
```sh
python3 perception.py --source qnx --save-patches data/ball --epipolar-tol 50
```
then (ball hidden, dark things in view: shadows, hair, phone/screen, black cables, dark clothes): `--save-patches data/not_ball`.
Laptop: `scp -r qnxpi:DODGE-1/data .`, `python train_ball_cnn.py`, `scp ball_cnn_int8.tflite qnxpi:DODGE-1/`.

## 8. Full system with real cameras
Laptop: `python sim3d.py` (with `$env:SIM_IP` / `$env:PI_IP` set). Pi:
```sh
slay stereo_capture 2>/dev/null     # the supervisor starts (and restarts) its own
SIM_IP=172.20.10.2 python3 supervisor.py --source qnx --preview 172.20.10.2
```
(extra options after `--source` are passed to perception, so `--preview` works here too).
