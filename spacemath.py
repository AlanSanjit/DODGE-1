"""Quaternion helpers, frame conversions and the synthetic hand-held ball (HandScenario).

Used on both machines; numpy + stdlib only.
Quaternions are [w, x, y, z] and rotate body -> LVLH.
"""
import math
import numpy as np

import config

CAM_TO_LVLH = np.array(config.CAM_TO_LVLH, dtype=float)
MEAN_MOTION = 2.0 * math.pi / config.ORBIT_PERIOD


# --------------------------------------------------------------------------- quaternions
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    s = math.sin(angle / 2.0) / n
    return np.array([math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_to_rot(q):
    """Rotation matrix R (body -> LVLH): v_lvlh = R @ v_body."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_integrate(q, omega_body, dt):
    """Propagate q with a constant body rate over dt (exact exponential map)."""
    w = np.asarray(omega_body, dtype=float)
    ang = np.linalg.norm(w) * dt
    if ang < 1e-12:
        return quat_normalize(q)
    dq = quat_from_axis_angle(w, ang)
    return quat_normalize(quat_mul(q, dq))


def quat_error_vec(q):
    """Small-angle attitude error (rad, body axes) of q relative to the identity target."""
    q = quat_normalize(q)
    if q[0] < 0:
        q = -q
    return 2.0 * q[1:4]


# --------------------------------------------------------------------------- frames
SAT_BOARD = np.array([0.0, 0.0, config.SAT_BOARD_Z])


def ball_to_space(p):
    """Ball position (board frame, m) -> asteroid position (LVLH, m) relative to the satellite's slot,
    which sits SAT_BOARD_Z in front of the camera."""
    return config.SPACE_SCALE * (CAM_TO_LVLH @ (np.asarray(p, dtype=float) - SAT_BOARD))


def ball_vel_to_space(v):
    """Ball velocity (board frame, m/s real) -> asteroid velocity (LVLH, m/s mission)."""
    return (config.SPACE_SCALE / config.TIME_SCALE) * (CAM_TO_LVLH @ np.asarray(v, dtype=float))


def cw_accel(pos, vel, n=MEAN_MOTION):
    """Clohessy-Wiltshire acceleration (LVLH: x radial, y along-track, z cross-track)."""
    x, _, z = pos
    vx, vy, _ = vel
    return np.array([3.0 * n * n * x + 2.0 * n * vy, -2.0 * n * vx, -n * n * z])


def project(p, cam_x):
    """Pinhole projection of a board-frame point into the camera centred at X=cam_x."""
    X, Y, Z = p
    if Z <= 1e-6:
        return None
    return (config.FOCAL_PX * (X - cam_x) / Z + config.CX, config.FOCAL_PX * Y / Z + config.CY)


def ball_in_view(p, margin_px=2.0):
    """True if the whole ball is inside the image(s): mono = the one camera at X=0; stereo = BOTH
    cameras and the disparity in range."""
    p = np.asarray(p, dtype=float)
    if p[2] <= 0.02:
        return False
    r_px = config.FOCAL_PX * (config.BALL_DIAM_M / 2.0) / p[2]
    if config.CAMERA_MODE == "mono":
        cams = (0.0,)
        if not (config.MONO_Z_MIN <= p[2] <= config.MONO_Z_MAX):
            return False
    else:
        cams = (-config.BASELINE_M / 2.0, config.BASELINE_M / 2.0)
        disp = config.FOCAL_PX * config.BASELINE_M / p[2]
        if not (config.DISP_MIN <= disp <= config.DISP_MAX):
            return False
    for cam_x in cams:
        u, v = project(p, cam_x)
        if u - r_px < margin_px or u + r_px > config.IMG_W - margin_px:
            return False
        if v - r_px < margin_px or v + r_px > config.IMG_H - margin_px:
            return False
    return True


# --------------------------------------------------------------------------- synthetic hand-held ball
def _smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


class HandScenario:
    """A person moving the ball: enter -> hover far -> approach the board -> hold -> leave -> pause.

    Positions are in the board frame (m, real time). Deterministic for a given seed.
    Two leave styles alternate: retreat straight back (ball stays visible beyond 30 cm, then exits)
    or pull away sideways out of view (ball unseen).
    """

    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)
        self.encounters = []   # list of (t_start, t_end, segments)
        self._t_end = 0.0

    def _new_encounter(self):
        r = self.rng
        side = 1.0 if r.random() < 0.5 else -1.0
        far = np.array([r.uniform(-0.03, 0.03), r.uniform(-0.02, 0.02), r.uniform(0.66, 0.72)])
        near = np.array([r.uniform(-0.012, 0.012), r.uniform(-0.008, 0.008), r.uniform(0.09, 0.11)])  # pushed through the satellite (30 cm) to ~10 cm
        off_in = np.array([side * 0.75, far[1], far[2]])
        style = len(self.encounters) % 2          # 0: retreat back, 1: sideways exit
        segs = [(1.0, off_in, far)]               # enter from the side
        segs.append((0.8, far, far))              # hover
        segs.append((r.uniform(4.0, 5.0), far, near))  # approach: avg 0.12-0.15 m/s, peak ~0.2 m/s (ease in/out)
        segs.append((r.uniform(2.5, 3.5), near, near)) # hold at the camera while the satellite flies home
        if style == 0:
            back = np.array([near[0], near[1], 0.5])
            segs.append((1.6, near, back))
            segs.append((1.0, back, back + np.array([-side * 0.8, 0.0, 0.0])))
            last = back + np.array([-side * 0.8, 0.0, 0.0])
        else:
            out = np.array([-side * 0.30, near[1], 0.16])
            segs.append((0.8, near, out))
            last = out
        segs.append((r.uniform(3.5, 4.5), last, last))  # out of view
        t0 = self._t_end
        dur = sum(s[0] for s in segs)
        self.encounters.append((t0, t0 + dur, segs))
        self._t_end = t0 + dur

    def position(self, t):
        t = max(t, 0.0)
        while self._t_end <= t:
            self._new_encounter()
        for t0, t1, segs in self.encounters:
            if t0 <= t < t1:
                tt = t - t0
                for dur, a, b in segs:
                    if tt < dur:
                        p = a + (b - a) * _smooth(tt / dur)
                        break
                    tt -= dur
                else:
                    p = segs[-1][2]
                # hand tremor (a few mm)
                trem = 0.002 * np.array([math.sin(7.1 * t), math.sin(5.3 * t + 1.0), math.sin(3.7 * t + 2.0)])
                return p + trem
        return self.encounters[-1][2][-1][2]

    def state(self, t, dt=0.01):
        p = self.position(t)
        v = (self.position(t + dt) - p) / dt
        return p, v
