"""DODGE-1 shared constants.

This file must be IDENTICAL on the laptop (sim3d.py) and on the Raspberry Pi 5 / QNX
(perception.py, control.py, supervisor.py). Put every new constant here; never hard-code.
"""
import os

# --------------------------------------------------------------------------- network
SIM_IP = os.environ.get("SIM_IP", "127.0.0.1")    # laptop running sim3d.py
PI_IP = os.environ.get("PI_IP", "127.0.0.1")      # Raspberry Pi 5 running QNX
SIM_PORT = 5005          # laptop listens: track, depthmap, cmd, procs
CONTROL_PORT = 5006      # Pi control listens: nav, track
SUPERVISOR_PORT = 5007   # Pi supervisor listens on localhost: hb

# --------------------------------------------------------------------------- rates
CONTROL_HZ = 50          # fixed control loop (real time)
NAV_HZ = 50              # sim -> control nav rate
PHYSICS_HZ = 200         # physics steps per REAL second
CAMERA_FPS = 30          # stereo frame rate
HB_HZ = 10               # heartbeats to the supervisor
PROCS_HZ = 2             # supervisor status to the sim
DEPTHMAP_EVERY = 6       # send a depth map every Nth frame

# --------------------------------------------------------------------------- scaling
SPACE_SCALE = 100.0      # 1 cm of ball distance = 1 m in space
TIME_SCALE = 5.0         # 1 real second = 5 mission seconds
# board/camera frame (X right, Y down, Z forward) -> LVLH (x radial, y along-track, z cross-track)
# x = -Y_cam, y = Z_cam, z = X_cam
CAM_TO_LVLH = [[0.0, -1.0, 0.0],
               [0.0, 0.0, 1.0],
               [1.0, 0.0, 0.0]]

# --------------------------------------------------------------------------- satellite physics
DRY_MASS = 23.0          # kg
PROP_MASS = 2.0          # kg of N2
THRUST_N = 5.0           # per thruster
ISP = 65.0               # s (cold gas N2)
G0 = 9.80665
VALVES = ["+x", "-x", "+y", "-y", "+z", "-z"]   # direction of the acceleration each gives (body)
BRANCHES = ["A", "B"]                            # primary / backup per valve
INERTIA = [1.10, 1.00, 0.90]                     # kg m^2 principal moments (body)
COM_OFFSET = [0.010, -0.006, 0.008]              # m, centre of mass offset from thrust centre
WHEEL_TORQUE_MAX = 0.1   # N m per axis
WHEEL_H_MAX = 1.0        # N m s per axis
ORBIT_PERIOD = 5554.0    # s, 400 km circular orbit
SIM_CMD_TIMEOUT = 0.15   # real s without cmd -> sim closes all commanded valves
COLLISION_DIST = 4.0     # m, centre-to-centre distance counted as a hit
ASTEROID_RADIUS = 2.0    # m, drawn size (ball 40 mm x 100)
SAT_SIZE = 1.2           # m, drawn body size

# --------------------------------------------------------------------------- control
TRIGGER_RANGE = 20.0     # m (= ball 20 cm)
MISS_TRIGGER = 8.0       # m, predicted miss distance that triggers a dodge
MISS_HORIZON = 10.0      # mission s, look-ahead for the predicted miss
DODGE_DIST = 10.0        # m perpendicular to the approach line
V_MAX = 1.0              # m/s mission, max translation speed
LATENCY_MARGIN_S = 0.25  # mission s of extra braking margin
POS_DEADBAND = 0.25      # m
VEL_DEADBAND = 0.03      # m/s
REACHED_POS = 0.5        # m, target reached
REACHED_VEL = 0.1        # m/s
RETURN_UNSEEN_S = 1.5    # real s without a detection -> return
RETURN_BALL_RANGE = 0.30 # m real (30 cm): beyond this and not approaching -> return
APPROACH_SPEED_MIN = 0.3 # m/s mission, radial speed counted as "approaching"
TRACK_VALID_S = 0.3      # real s, a track older than this is not used for threat assessment
NAV_TIMEOUT = 0.3        # real s without nav -> NO_NAV_SAFE (all valves closed)
ATT_KP = 2.0             # N m / rad
ATT_KD = 2.5             # N m / (rad/s)
FDIR_WINDOW = 1.0        # mission s, window for measured acceleration
FDIR_RESIDUAL_FRAC = 0.5 # residual > this fraction of one thruster's acceleration
FDIR_PERSIST_S = 0.3     # real s the residual must persist
PROP_MIN = 0.005         # kg, below this the thrusters are declared empty

# --------------------------------------------------------------------------- display / sim
TRACK_DRAW_S = 0.2       # real s, draw the asteroid only while the track is this fresh
ENCOUNTER_END_S = 1.0    # real s without asteroid -> encounter over

# --------------------------------------------------------------------------- cameras / perception
IMG_W = 320
IMG_H = 240
FOCAL_PX = 246.0         # Camera Module 3 (66 deg HFOV) at 320 px width
CX = 160.0
CY = 120.0
BASELINE_M = 0.06        # camera separation
BALL_DIAM_M = 0.04       # orange ping-pong ball
CNN_SIZE = 32
CNN_BALL_CLASS = 0       # 0 = ball, 1 = not ball
CNN_THRESHOLD = 0.6      # P(ball) needed in each image
MODEL_PATH = "ball_cnn_int8.tflite"
CALIB_PATH = "calib.json"
RECTIFY_PATH = "rectify_maps.npz"
ORANGE_R_MIN = 0.45      # R/(R+G+B)
ORANGE_B_MAX = 0.20      # B/(R+G+B)
ORANGE_MIN_SUM = 120     # ignore very dark pixels
GRID_STEP = 4            # candidate search on a 4x grid
MIN_BLOB_CELLS = 2       # grid cells
MAX_CANDIDATES = 4       # per image
EPIPOLAR_TOL_PX = 6
DISP_MIN = 2.0
DISP_MAX = 170.0
SIL_SEARCH_PX = 25
SIL_AGREE_PX = 4.0
SIL_AGREE_FRAC = 0.06
KF_ACCEL_SIGMA = 1.5     # m/s^2 real, Kalman process noise
KF_RESET_S = 0.5         # real s without detection -> restart the filter
DEPTH_W = 80
DEPTH_H = 60
DEPTH_BLOCK = 5          # SAD window (low-res px)

# --------------------------------------------------------------------------- supervisor
CONTROL_PRIO = 40
PERCEPTION_PRIO = 15
CONTROL_HB_TIMEOUT = 0.5
PERCEPTION_HB_TIMEOUT = 1.5
CONTROL_GRACE_S = 5.0
PERCEPTION_GRACE_S = 30.0
