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
SPACE_SCALE = 300.0      # 1 cm of ball movement = 3 m in space (motion amplified 3x; asteroid size unchanged)
TIME_SCALE = 15.0        # 1 real second = 15 mission seconds: with a 15 cm trigger the 5 N thrusters clear a
                         # ball pushed at up to ~0.2 m/s (closed-loop test); 20 makes station keeping chatter
SAT_BOARD_Z = 0.30       # m: the satellite's orbit slot is this far in front of the camera (board Z axis)
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
ASTEROID_RADIUS = 1.75   # m, drawn/collision size (kept from the 1 cm = 1 m scale: ball 35 mm -> 3.5 m)
SAT_SIZE = 1.2           # m, drawn body size

# --------------------------------------------------------------------------- control
TRIGGER_RANGE = 45.0     # m: dodge when the asteroid is within 45 m (= ball within 15 cm of the satellite)
TRIGGER_FRONT_ONLY = True  # only an asteroid coming from the front (ball moving toward the camera) is a threat
CLEAR_RANGE = 48.0       # m: ball passed and beyond this (16 cm) from the satellite, not approaching -> RETURN
AUTO_RESET = True        # sim: reset after a dodge once the ball reached the camera and the satellite is home
AUTO_RESET_BALL_Z = 0.12 # m: the ball counts as 'at the camera' this close
AUTO_RESET_DELAY_S = 1.0 # real s the satellite must be home before the reset
REWIND_BALL_Z = 0.50     # m: after a reset the asteroid is hidden (no collisions) until the ball is back out
REWIND_UNSEEN_S = 1.5    #   this far from the camera, or unseen this long (taken away to start again)
MISS_TRIGGER = 8.0       # m, predicted miss distance that triggers a dodge
MISS_HORIZON = 0.0       # mission s, look-ahead for the predicted-miss trigger (0 = off: dodge only at TRIGGER_RANGE)
DODGE_DIST = 10.0        # m perpendicular to the approach line (max dodge)
DODGE_MODE = "min_fuel"  # "min_fuel": only as far/fast as needed, slow return; "fixed": DODGE_DIST at V_MAX
SAFE_MISS = 6.0          # m, min_fuel: required closest approach after the dodge (collision = COLLISION_DIST)
TRACK_POS_SIGMA_M = 0.002  # m real: ball position noise (per frame)
NOISE_SIGMAS = 2.0       # dodge sizing allows this many sigma of position noise (x SPACE_SCALE)
DODGE_MARGIN = 1.0       # m added to the required dodge distance (perception noise)
DODGE_MIN = 2.0          # m, smallest dodge
DODGE_TIME_FRAC = 0.6    # min_fuel: plan to arrive within this fraction of t_cpa
V_DODGE_MIN = 0.25       # m/s, slowest dodge speed
V_RETURN = 0.3           # m/s, return to the slot (no deadline: slow = cheap)
V_MAX = 2.0              # m/s mission, max translation speed
LATENCY_MARGIN_S = 0.5   # mission s of extra braking margin
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
FDIR_WINDOW = 2.0        # mission s, window for measured acceleration
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
CAMERA_MODE = "mono"     # "mono": ONE camera, depth from the ball's known size; "stereo": two cameras
MONO_UNIT = 1            # sensor unit of the single camera (qnx/sensor_dodge_mono.conf)
MONO_Z_MIN = 0.04        # m, plausible depth range for a size-based depth
MONO_Z_MAX = 1.2
MONO_RADIUS_SIGMA_PX = 0.8  # radius measurement noise -> depth sigma Z * sigma / r
MONO_RADIUS_BIAS_PX = 0.6   # measured outline radius minus true radius (subtracted before depth)
EDGE_MARGIN_PX = 3       # mono: a ball cut by the image edge has a wrong size -> rejected
BASELINE_M = 0.07        # camera separation (USB stereo rig, 7 cm; stereo mode only)
BALL_DIAM_M = 0.035      # the real ball: 35 mm diameter (measured)
BALL_COLOR = "black"     # "black", "white" or "orange" (candidate colour test + matching CNN model)
CNN_SIZE = 32
CNN_BALL_CLASS = 0       # 0 = ball, 1 = not ball
CNN_THRESHOLD = 0.85     # P(ball) needed in each image
SHAPE_FALLBACK = True    # if the CNN says no, accept a roughly round blob (stereo + size + 2 frames decide)
SHAPE_SCORE = 0.7        # p_ball given to a shape-accepted candidate
SHAPE_FILL_MIN = 0.50    # blob area / bounding box: a circle is 0.785, a partly hidden ball ~0.55
SHAPE_FILL_MAX = 0.90    #   a filled rectangle is ~1.0 -> rejected
SHAPE_ASPECT_MAX = 1.6   # bounding box width/height (or inverse) at most this
SHAPE_CIRCLE_SCORE = 0.5 # circle-detector candidates: min fraction of the outline found
SHAPE_MIN_RADIUS_PX = 4.0
ISOLATED_INSIDE_MIN = 0.6  # shape fallback: >= this fraction ball-coloured inside the circle
ISOLATED_RING_MAX = 0.35   #   and <= this just outside it (ball stands out from the background)
SHAPE_SIZE_RATIO_MIN = 0.6   # shape-only pairs must match the ball size more closely than CNN pairs
SHAPE_SIZE_RATIO_MAX = 1.6
MODEL_PATH = "ball_cnn_int8.tflite"
CALIB_PATH = "calib.json"
RECTIFY_PATH = "rectify_maps.npz"
ORANGE_R_MIN = 0.45      # R/(R+G+B)
ORANGE_B_MAX = 0.20      # B/(R+G+B)
ORANGE_MIN_SUM = 120     # ignore very dark pixels
WHITE_MIN_MEAN = 130     # white ball: (R+G+B)/3 at least this (shaded side of the ball drops below)
WHITE_MAX_CHROMA = 45    # white ball: max(R,G,B) - min(R,G,B) at most this (colourless)
BLACK_MAX_V = 100        # black ball: max(R,G,B) at most this (webcams show a black ball as dark grey) (its lit side/highlight goes above: the
                         # circle detector still finds the outline)
BLACK_MAX_CHROMA = 45    # black ball: max(R,G,B) - min(R,G,B) at most this (colourless)
MAX_BLOB_AREA = 9000     # px: bigger blobs (walls, shadows, screens, paper) are not the ball
GRID_STEP = 4            # candidate search on a 4x grid
MIN_BLOB_CELLS = 2       # grid cells
MAX_CANDIDATES = 6       # per image
CIRCLE_MAX = 3           # radius bands (small/medium/large) of the gradient Hough circle detector;
                         # 0 = off (fastest; fine when nothing ball-coloured is behind the ball)
CIRCLE_PER_BAND = 2      # circle candidates kept per band
CIRCLE_RMIN = 6          # px full-res ball radius range searched (ball 0.6 m -> 8 px, 0.10 m -> 49 px)
CIRCLE_RMAX = 56          # 7 cm baseline: ball radius 49 px at 10 cm
CIRCLE_EDGE_MIN = 24.0   # min gradient (grey levels over 2 px, half-res) for an edge pixel to vote
CIRCLE_SCORE_MIN = 0.35  # min fraction of the circumference that voted
GRAY_SEARCH_PX = 6       # disparity refinement window for circle candidates
EPIPOLAR_TOL_PX = 20     # max row difference left/right (camera misalignment); calibrate to tighten
DISP_MIN = 2.0
DISP_MAX = 200.0         # 7 cm baseline: 191 px at 9 cm (closest range both cameras see the ball)
SIL_SEARCH_PX = 25
SIL_AGREE_PX = 8.0
SIL_AGREE_FRAC = 0.12
PAIR_RADIUS_RATIO = 2.0  # left/right radius of the same ball differ by at most this factor
SIZE_RATIO_MIN = 0.45    # measured radius / radius the ball (BALL_DIAM_M) would have at the stereo depth
SIZE_RATIO_MAX = 2.0     #   (rejects small/large round things the CNN cannot tell apart by shape)
MAX_PAIRS_TRIED = 4      # left/right candidate pairs tried per frame
CONFIRM_FRAMES = 2       # consecutive consistent detections before control is told "detected"
CONFIRM_JUMP_M = 0.08    # m: a detection further than this from the previous one restarts the count
KF_ACCEL_SIGMA = 1.5     # m/s^2 real, Kalman process noise
KF_RESET_S = 0.5         # real s without detection -> restart the filter
DEPTH_W = 80
DEPTH_H = 60
DEPTH_BLOCK = 5          # SAD window (low-res px)

CAM_TIMEOUT_S = 2.0      # no frame from a QNX camera this long -> perception exits (supervisor restarts it)
PREVIEW_PORT = 5008      # perception --preview -> live_view.py on the laptop
PREVIEW_FPS = 10.0
PREVIEW_SCALE = 2        # 2 = send 160x120 per camera (Wi-Fi friendly), 1 = full 320x240
SIM_FEED = True          # perception also streams the annotated camera image to the simulator window
SIM_FEED_PORT = 5009     # laptop UDP port for that feed (allow it in the laptop firewall)
SIM_FEED_FPS = 8.0
SIM_FEED_SCALE = 2
AST_SMOOTH_S = 0.15      # sim display only: low-pass time constant of the drawn asteroid position
AST_HOLD_S = 0.5         # sim display only: keep drawing the asteroid through dropouts this long
AST_SNAP_M = 75.0        # sim display only: jumps larger than this are drawn immediately

# --------------------------------------------------------------------------- supervisor
CONTROL_PRIO = 40
PERCEPTION_PRIO = 15
PIN_CPUS = False         # QNX 8.0 `on -C` = CPU cluster (Pi 5: only 0), not a core; leave off
CAPTURE_PRIO = 20         # qnx/stereo_capture: above perception, below control
CONTROL_HB_TIMEOUT = 0.5
PERCEPTION_HB_TIMEOUT = 1.5
CONTROL_GRACE_S = 5.0
PERCEPTION_GRACE_S = 30.0
