"""DODGE-1 perception (runs on the Raspberry Pi 5 / QNX, LOW priority).

cameras -> ball-coloured (config.BALL_COLOR) candidates -> TFLite CNN in BOTH images -> stereo checks -> depth -> Kalman
-> `track` messages to control (127.0.0.1) and to the sim, plus a coarse `depthmap` for display.

numpy + tflite_runtime + stdlib only (no OpenCV on the Pi).
Perception never imports control and shares no memory with it: it only sends UDP messages.

    python3 perception.py --source synthetic        # rendered hand-held ball, wall, same-colour box + posters
    python3 perception.py --source qnx              # QNX camera units 3 and 4 (bridge not written yet)
"""
import argparse
import base64
import json
import math
import os
import socket
import sys
import time

import numpy as np

import config
import spacemath

HERE = os.path.dirname(os.path.abspath(__file__))


# =========================================================================== camera model
class Camera:
    """Rectified pinhole stereo pair. Defaults from config, overridden by calib.json."""

    def __init__(self):
        self.f = config.FOCAL_PX
        self.cx = config.CX
        self.cy = config.CY
        self.baseline = config.BASELINE_M
        self.source = "config"
        path = os.path.join(HERE, config.CALIB_PATH)
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    c = json.load(fh)
                self.f = float(c["f"])
                self.cx = float(c["cx"])
                self.cy = float(c["cy"])
                self.baseline = float(c["baseline"])
                self.source = "calib.json"
            except (OSError, KeyError, ValueError) as exc:
                print("[perception] ignoring bad calib.json: %s" % exc, flush=True)


def load_rectify_maps():
    path = os.path.join(HERE, config.RECTIFY_PATH)
    if not os.path.exists(path):
        return None
    try:
        m = np.load(path)
        maps = {k: m[k].astype(np.intp) for k in ("map_lx", "map_ly", "map_rx", "map_ry")}
        if maps["map_lx"].shape != (config.IMG_H, config.IMG_W):
            raise ValueError("map shape %s" % (maps["map_lx"].shape,))
        return maps
    except Exception as exc:  # a broken file must not stop perception: run unrectified
        print("[perception] ignoring bad rectify maps: %s" % exc, flush=True)
        return None


def rectify(img, mx, my):
    """Nearest-neighbour remap with precomputed integer maps (numpy fancy indexing)."""
    return img[my, mx]


# =========================================================================== synthetic world
def _block_noise(rng, h, w, cell):
    g = rng.random((h // cell + 2, w // cell + 2)).astype(np.float32)
    return np.repeat(np.repeat(g, cell, axis=0), cell, axis=1)[:h, :w]


def _palette(rng, randomize, color):
    """Scene colours for the configured ball colour: ball, a same-coloured BOX and posters as hard
    negatives, and a wall tint that never passes that colour's candidate test."""
    U = rng.uniform
    if color == "black":
        r0 = U(185, 215) if randomize else 195.0            # light (white-ish) wall: max channel stays > BLACK_MAX_V
        wall = np.array([r0, r0 + (U(3, 12) if randomize else 8), r0 + (U(10, 30) if randomize else 22)], np.float32)
        neg = np.array([35, 35, 40], np.float32)             # dark posters
        b0 = U(15, 50) if randomize else 30.0
        box = np.array([b0, b0 + (U(-4, 4) if randomize else 0), b0 + (U(-3, 6) if randomize else 3)], np.float32)
        k0 = U(15, 40) if randomize else 28.0
        ball = np.array([k0, k0 + (U(-3, 3) if randomize else 0), k0 + (U(-2, 5) if randomize else 2)], np.float32)
    elif color == "orange":
        wall = np.array(U([90, 100, 115], [140, 145, 170]) if randomize else [110, 120, 140], np.float32)
        neg = np.array([230, 230, 225], np.float32)
        box = np.array([U(200, 255), U(80, 140), U(10, 45)] if randomize else [235, 112, 30], np.float32)
        ball = np.array([U(235, 255), U(95, 135), U(5, 35)] if randomize else [250, 115, 20], np.float32)
    else:  # white
        if randomize:     # bluish tint: chroma stays > WHITE_MAX_CHROMA where it is bright enough to be white
            r0 = U(40, 90)
            wall = np.array([r0, r0 + U(5, 25), r0 + U(40, 70)], np.float32)
        else:
            wall = np.array([65, 85, 120], np.float32)
        neg = np.array([230, 230, 225], np.float32)
        w0 = U(200, 250) if randomize else 235.0
        box = np.array([w0, w0 + (U(-8, 8) if randomize else 0), w0 + (U(-15, 5) if randomize else -7)], np.float32)
        ball = np.array([U(232, 252), U(232, 252), U(225, 248)] if randomize else [245, 245, 240], np.float32)
    return wall, neg, box, ball


class SyntheticScene:
    """Renders what the two cameras would see: tinted textured wall, posters (some in the ball's colour),
    a BOX in the ball's colour as distractor, a skin-coloured hand and the ping-pong ball (ray-cast
    spheres, Lambert shading). The wall tint never passes the ball-colour test (config.BALL_COLOR)."""

    TEX_RES = 400.0          # texels per metre on the wall
    TEX_X = (-2.0, 2.0)
    TEX_Y = (-1.5, 1.5)

    def __init__(self, seed=0, randomize=False, cam=None, mono=False, distractors=True):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.cam = cam or Camera()
        self.wall_z = rng.uniform(0.9, 1.8) if randomize else 1.2
        th = int((self.TEX_Y[1] - self.TEX_Y[0]) * self.TEX_RES)
        tw = int((self.TEX_X[1] - self.TEX_X[0]) * self.TEX_RES)
        n = 0.5 * _block_noise(rng, th, tw, 64) + 0.3 * _block_noise(rng, th, tw, 16) + 0.2 * _block_noise(rng, th, tw, 4)
        base, neg_col, box_col, ball_col = _palette(rng, randomize, config.BALL_COLOR)
        tex = base[None, None, :] * (0.55 + 0.9 * n[..., None])
        # posters: coloured, and some in the ball's colour (hard negatives)
        for _ in range(6):
            y0 = rng.integers(0, th - 200); x0 = rng.integers(0, tw - 200)
            hh = rng.integers(60, 200); ww = rng.integers(60, 200)
            col = np.array([rng.uniform(30, 120), rng.uniform(80, 220), rng.uniform(120, 240)], np.float32)
            if rng.random() < 0.3 and distractors:
                col = neg_col
            tex[y0:y0 + hh, x0:x0 + ww] = col * (0.8 + 0.4 * n[y0:y0 + hh, x0:x0 + ww, None])
        self.tex = np.clip(tex, 0, 255).astype(np.uint8)
        # box distractor in the ball's colour (front face, fronto-parallel)
        if randomize:
            bz = rng.uniform(0.35, 0.9)
            bx = rng.uniform(-0.2, 0.15); by = rng.uniform(-0.1, 0.12)
            bs = rng.uniform(0.04, 0.1)
            self.box = (bx, bx + bs * rng.uniform(0.7, 1.6), by, by + bs, bz)
        else:
            self.box = (0.12, 0.20, 0.03, 0.11, 0.70)
        self.box_col = box_col
        if not distractors:              # plain background: no same-coloured box
            self.box = (5.0, 5.1, 5.0, 5.1, 0.7)
        self.ball_col = ball_col
        self.skin_col = np.array([205, 160, 130], np.float32)
        ld = np.array([-0.35, -0.6, -0.7]) if not randomize else np.array(
            [rng.uniform(-0.7, 0.7), rng.uniform(-0.9, -0.2), rng.uniform(-0.9, -0.4)])
        self.light = ld / np.linalg.norm(ld)
        self.noise = 2.5
        # mono: one camera at the board centre; stereo: left/right at -/+ baseline/2
        self.cams_x = (0.0,) if mono else (-self.cam.baseline / 2.0, self.cam.baseline / 2.0)
        self.bg = [self._background(cx) for cx in self.cams_x]

    def _background(self, cam_x):
        c = self.cam
        u = np.arange(config.IMG_W, dtype=np.float32)
        v = np.arange(config.IMG_H, dtype=np.float32)
        X = (u - c.cx) / c.f * self.wall_z + cam_x
        Y = (v - c.cy) / c.f * self.wall_z
        tj = np.clip(((X - self.TEX_X[0]) * self.TEX_RES).astype(np.intp), 0, self.tex.shape[1] - 1)
        ti = np.clip(((Y - self.TEX_Y[0]) * self.TEX_RES).astype(np.intp), 0, self.tex.shape[0] - 1)
        img = self.tex[ti[:, None], tj[None, :]].astype(np.float32)
        depth = np.full((config.IMG_H, config.IMG_W), self.wall_z, np.float32)
        x0, x1, y0, y1, bz = self.box
        u0 = int(round(c.f * (x0 - cam_x) / bz + c.cx)); u1 = int(round(c.f * (x1 - cam_x) / bz + c.cx))
        v0 = int(round(c.f * y0 / bz + c.cy)); v1 = int(round(c.f * y1 / bz + c.cy))
        u0, u1 = max(u0, 0), min(u1, config.IMG_W)
        v0, v1 = max(v0, 0), min(v1, config.IMG_H)
        if u1 > u0 and v1 > v0:
            grad = np.linspace(1.05, 0.8, v1 - v0, dtype=np.float32)[:, None, None]
            img[v0:v1, u0:u1] = self.box_col[None, None, :] * grad
            depth[v0:v1, u0:u1] = bz
        return img, depth

    def _sphere(self, img, depth, P, R, cam_x, col, spec=True):
        c = self.cam
        X, Y, Z = P[0] - cam_x, P[1], P[2]
        if Z <= R + 0.005:
            return
        uc = c.f * X / Z + c.cx
        vc = c.f * Y / Z + c.cy
        rp = c.f * R / max(Z - R, 1e-3) * 1.15 + 2
        u0 = max(int(uc - rp), 0); u1 = min(int(uc + rp) + 1, config.IMG_W)
        v0 = max(int(vc - rp), 0); v1 = min(int(vc + rp) + 1, config.IMG_H)
        if u1 <= u0 or v1 <= v0:
            return
        uu = (np.arange(u0, u1, dtype=np.float32) - c.cx) / c.f
        vv = (np.arange(v0, v1, dtype=np.float32) - c.cy) / c.f
        dx = uu[None, :]; dy = vv[:, None]
        a = dx * dx + dy * dy + 1.0
        b = dx * X + dy * Y + Z
        cc = X * X + Y * Y + Z * Z - R * R
        disc = b * b - a * cc
        hit = disc > 0
        t = (b - np.sqrt(np.maximum(disc, 0))) / a
        hit &= t < depth[v0:v1, u0:u1]
        if not hit.any():
            return
        nx = (t * dx - X) / R; ny = (t * dy - Y) / R; nz = (t - Z) / R
        # self.light points from the scene TOWARD the light (up-left, toward the cameras)
        lam = np.maximum(0.0, nx * self.light[0] + ny * self.light[1] + nz * self.light[2])
        shade = 0.38 + 0.72 * lam
        out = col[None, None, :] * shade[..., None]
        if spec:
            # Blinn highlight, viewer direction ~ (0, 0, -1)
            hx, hy, hz = self.light[0], self.light[1], self.light[2] - 1.0
            hn = math.sqrt(hx * hx + hy * hy + hz * hz)
            sp = np.maximum(0.0, (nx * hx + ny * hy + nz * hz) / hn) ** 30
            out = out + 60.0 * sp[..., None]
        reg = img[v0:v1, u0:u1]
        reg[hit] = out[hit]
        depth[v0:v1, u0:u1][hit] = t[hit]

    def render(self, ball_p, hand=True, rng=None, extras=None):
        """extras: optional [(position, radius_m, rgb)] spheres, e.g. small round decoys for testing."""
        rng = rng if rng is not None else self.rng
        out = []
        for i, cam_x in enumerate(self.cams_x):
            img = self.bg[i][0].copy()
            depth = self.bg[i][1].copy()
            for P_e, R_e, col_e in (extras or []):
                self._sphere(img, depth, np.asarray(P_e, float), R_e, cam_x, np.asarray(col_e, np.float32))
            if ball_p is not None:
                P = np.asarray(ball_p, dtype=float)
                if hand:
                    self._sphere(img, depth, P + np.array([0.004, 0.052, 0.03]), 0.040, cam_x, self.skin_col, spec=False)
                    self._sphere(img, depth, P + np.array([0.01, 0.12, 0.07]), 0.045, cam_x, self.skin_col * 0.95, spec=False)
                self._sphere(img, depth, P, config.BALL_DIAM_M / 2.0, cam_x, self.ball_col)
            gain = 1.0 if i == 0 else 0.97
            img = img * gain + rng.normal(0.0, self.noise, img.shape).astype(np.float32)
            out.append(np.clip(img, 0, 255).astype(np.uint8))
        return out[0], (out[1] if len(out) > 1 else None)


class SyntheticStereoSource:
    """Stand-in for the two Camera Module 3 units: renders a person moving the ball."""

    def __init__(self, fps=config.CAMERA_FPS, seed=1, cam=None, mono=False):
        self.period = 1.0 / fps
        self.scenario = spacemath.HandScenario(seed=seed)
        self.scene = SyntheticScene(seed=seed, cam=cam, mono=mono)
        self.t0 = time.monotonic()
        self.next_t = self.t0
        self.first = True
        self.truth = None

    def read(self):
        now = time.monotonic()
        if self.first:                      # start the hand's clock at the first frame (after model load)
            self.first = False
            self.t0 = self.next_t = now
        if now < self.next_t:
            time.sleep(self.next_t - now)
        self.next_t = max(self.next_t + self.period, time.monotonic() - self.period)
        t_cap = time.monotonic()
        p = self.scenario.position(t_cap - self.t0)
        self.truth = p
        left, right = self.scene.render(p)
        return t_cap, left, right


class QnxMonoSource:
    """ONE USB camera (CAMERA_MODE "mono"): frames from qnx/stereo_capture -n 1 in /dev/shmem.
    read() returns (t_capture, image, None). No frame for CAM_TIMEOUT_S -> raises (supervisor restarts)."""

    def __init__(self, timeout=config.CAM_TIMEOUT_S):
        import qnx_camera
        self.qc = qnx_camera
        self.shm = qnx_camera.ShmStereo(wait_s=10.0)
        self.units = [self.shm.units[0]]
        self.timeout = timeout
        self.last = 0
        self.skew_ms = 0.0
        self.info = ""

    def read(self):
        f = self.shm.wait_newer(0, self.last, self.timeout)
        if f is None:
            raise RuntimeError("camera timeout: no frame from sensor unit %d for %.1f s (stereo_capture heartbeat %.1f s ago)"
                               % (self.units[0], self.timeout, self.shm.heartbeat_age()))
        f = self.shm.read(0) or f
        self.last = f.seq
        t_cap = f.t_us / 1e6
        now = time.monotonic()
        if not (now - 5.0 < t_cap <= now + 0.1):
            t_cap = now
        self.info = "%s %dx%d mono" % (self.qc.FT_NAME.get(f.ftype, f.ftype), f.w, f.h)
        return t_cap, self.qc.to_rgb(f, config.IMG_W, config.IMG_H), None

    def close(self):
        self.shm.close()


class QnxStereoSource:
    """Two USB cameras owned by the QNX sensor service, read through the QNX camera library.

    backend "shm" (default): qnx/stereo_capture (C) receives the frames in libcamapi's viewfinder
    callback and publishes them in /dev/shmem/dodge_stereo; this class only reads that memory.
    backend "ctypes": open the cameras from Python (needs working ctypes callbacks; the QNX 8.0 Pi
    image crashes on them).
    read() waits until BOTH cameras have a new frame, then converts the freshest pair to 320x240 RGB.
    If a camera stops delivering, read() raises -> perception exits -> the supervisor restarts it
    (and control sees no fresh track = no asteroid: the safe default)."""

    def __init__(self, units=None, swap=False, timeout=config.CAM_TIMEOUT_S, backend="shm"):
        import qnx_camera
        self.qc = qnx_camera
        self.backend = backend
        self.timeout = timeout
        self.last = [0, 0]
        self.skew_ms = 0.0
        self.info = ""
        self.cams = []
        if backend == "shm":
            self.shm = qnx_camera.ShmStereo(wait_s=10.0)
            self.idx = [1, 0] if swap else [0, 1]
            self.units = [self.shm.units[i] for i in self.idx]
            return
        avail = qnx_camera.list_cameras()
        if units is None:
            if len(avail) < 2:
                raise RuntimeError("need 2 cameras, sensor service reports units %s "
                                   "(start it with qnx/sensor_dodge_usb.conf)" % avail)
            units = avail[:2]
        if swap:
            units = list(units)[::-1]
        self.units = list(units)
        self.cams = []
        try:
            for u in self.units:
                self.cams.append(qnx_camera.QnxCamera(u))
        except Exception:
            self.close()
            raise

    def read(self):
        if self.backend == "shm":
            got = []
            for k, i in enumerate(self.idx):
                f = self.shm.wait_newer(i, self.last[k], self.timeout)
                if f is None:
                    raise RuntimeError("camera timeout: no frame from sensor unit %d for %.1f s "
                                       "(stereo_capture heartbeat %.1f s ago)"
                                       % (self.units[k], self.timeout, self.shm.heartbeat_age()))
                got.append(f)
            fresh = [self.shm.read(i) for i in self.idx]          # freshest pair after waiting
            fl, fr = [fresh[k] or got[k] for k in range(2)]
        else:
            for i, c in enumerate(self.cams):
                if c.wait_newer(self.last[i], self.timeout) is None:
                    raise RuntimeError("camera timeout: no frame from sensor unit %d for %.1f s"
                                       % (self.units[i], self.timeout))
            fl, fr = self.cams[0].latest, self.cams[1].latest      # freshest pair
        self.last = [fl.seq, fr.seq]
        self.skew_ms = (fl.t_us - fr.t_us) / 1000.0
        t_cap = 0.5 * (fl.t_us + fr.t_us) / 1e6                # CLOCK_MONOTONIC, same as time.monotonic()
        now = time.monotonic()
        if not (now - 5.0 < t_cap <= now + 0.1):               # unexpected clock base: fall back to now
            t_cap = now
        self.info = "%s %dx%d" % (self.qc.FT_NAME.get(fl.ftype, fl.ftype), fl.w, fl.h)
        left = self.qc.to_rgb(fl, config.IMG_W, config.IMG_H)
        right = self.qc.to_rgb(fr, config.IMG_W, config.IMG_H)
        return t_cap, left, right

    def close(self):
        if getattr(self, "shm", None) is not None:
            self.shm.close()
        for c in self.cams:
            try:
                c.close()
            except Exception:
                pass


# =========================================================================== candidates
def ball_mask(img):
    """Pixels that can belong to the ball (config.BALL_COLOR)."""
    if config.BALL_COLOR == "black":
        return black_mask(img)
    if config.BALL_COLOR == "white":
        return white_mask(img)
    return orange_mask(img)


def black_mask(img):
    """Dark and colourless: max channel <= BLACK_MAX_V and max-min channel <= BLACK_MAX_CHROMA."""
    f = img.astype(np.int32)
    mx = f.max(axis=2)
    return (mx <= config.BLACK_MAX_V) & (mx - f.min(axis=2) <= config.BLACK_MAX_CHROMA)


def white_mask(img):
    """Bright and colourless: mean >= WHITE_MIN_MEAN and max-min channel <= WHITE_MAX_CHROMA."""
    f = img.astype(np.int32)
    s = f[..., 0] + f[..., 1] + f[..., 2]
    chroma = f.max(axis=2) - f.min(axis=2)
    return (s >= 3 * config.WHITE_MIN_MEAN) & (chroma <= config.WHITE_MAX_CHROMA)


def orange_mask(img):
    f = img.astype(np.int32)
    s = f[..., 0] + f[..., 1] + f[..., 2]
    return ((s >= config.ORANGE_MIN_SUM)
            & (f[..., 0] * 100 > int(config.ORANGE_R_MIN * 100) * s)
            & (f[..., 2] * 100 < int(config.ORANGE_B_MAX * 100) * s))


def _components(mask):
    """8-connected components of a small boolean grid (pure Python BFS over set cells only)."""
    ys, xs = np.nonzero(mask)
    cells = set(zip(ys.tolist(), xs.tolist()))
    comps = []
    while cells:
        seed = cells.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    n = (y + dy, x + dx)
                    if n in cells:
                        cells.remove(n)
                        stack.append(n)
                        comp.append(n)
        comps.append(comp)
    return comps


def _gray(img):
    return img.astype(np.float32).mean(axis=2)


_ANG = np.linspace(0, 2 * np.pi, 48, endpoint=False)
_COS, _SIN = np.cos(_ANG), np.sin(_ANG)


def _refine_circle(g, u, v, r):
    """Full-res refinement: outline radius = the radius with the strongest mean radial brightness step
    over 48 directions (searched 0.6r..2r), then the centre = midpoint of the strongest edge per direction
    pair. The Hough peak often sits on the shading terminator, inside the true outline.
    g = full-resolution grey image."""
    H, W = g.shape
    rs = np.arange(max(3.0, 0.6 * r), min(2.0 * r, config.CIRCLE_RMAX * 1.2) + 1.0, 1.0)
    if len(rs) < 3:
        return u, v, r
    xs = np.clip(np.rint(u + rs[:, None] * _COS[None, :]).astype(np.intp), 0, W - 1)
    ys = np.clip(np.rint(v + rs[:, None] * _SIN[None, :]).astype(np.intp), 0, H - 1)
    prof = g[ys, xs]                                   # (radius, angle)
    step = np.abs(prof[2:] - prof[:-2])                # radial brightness change at rs[1:-1]
    peak = step.max(axis=0)
    good = peak >= config.CIRCLE_EDGE_MIN              # directions with a real edge
    if good.sum() < 8:
        return u, v, r
    # outermost strong edge per direction: the outline lies outside the shading terminator
    strong = step >= 0.5 * peak[None, :]
    last = len(rs) - 3 - np.argmax(strong[::-1], axis=0)
    re = rs[1:-1][last]
    px = u + re * _COS; py = v + re * _SIN
    sel = good.copy()
    for _ in range(2):                                 # algebraic circle fit, then drop outliers and refit
        A = np.column_stack([px[sel], py[sel], np.ones(sel.sum())])
        bb = -(px[sel] ** 2 + py[sel] ** 2)
        try:
            D, E, F = np.linalg.lstsq(A, bb, rcond=None)[0]
        except np.linalg.LinAlgError:
            return u, v, r
        cu, cv = -D / 2.0, -E / 2.0
        rad2 = cu * cu + cv * cv - F
        if rad2 <= 0:
            return u, v, r
        rf = math.sqrt(rad2)
        res = np.abs(np.hypot(px - cu, py - cv) - rf)
        sel = good & (res < max(1.5, 0.15 * rf))
        if sel.sum() < 8:
            return u, v, r
    if not (0.5 * r <= rf <= 2.2 * r) or math.hypot(cu - u, cv - v) > r:
        return u, v, r                                 # implausible fit: keep the Hough estimate
    return float(cu), float(cv), float(rf)


def circle_candidates(img, max_n=config.CIRCLE_MAX):
    """Round shapes from a gradient Hough transform (half resolution, numpy only).
    Each strong edge pixel votes for centres at distance r along +/- its gradient, so a ball is found
    even when its colour merges with the background (white ball in front of a white wall)."""
    if max_n <= 0:                                       # CIRCLE_MAX = 0 disables the detector
        return []
    gfull = _gray(img)
    g = gfull[::2, ::2]
    H, W = g.shape
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    thr = max(config.CIRCLE_EDGE_MIN, float(np.percentile(mag, 90)))
    ys, xs = np.nonzero(mag > thr)
    if len(ys) < 8:
        return []
    if len(ys) > 2500:                                   # keep the strongest edges
        keep = np.argpartition(mag[ys, xs], -2500)[-2500:]
        ys, xs = ys[keep], xs[keep]
    m = mag[ys, xs]
    ux, uy = gx[ys, xs] / m, gy[ys, xs] / m
    # radii spaced ~8% apart (1 px steps for small circles, coarser for big ones: the 3x3 vote smoothing covers it)
    radii = np.unique(np.round(np.geomspace(config.CIRCLE_RMIN / 2.0, config.CIRCLE_RMAX / 2.0, 22))).astype(np.float32)
    nr = len(radii)
    flat = []
    for sign in (1.0, -1.0):
        cx = np.rint(xs[None, :] + sign * radii[:, None] * ux[None, :]).astype(np.int64)
        cy = np.rint(ys[None, :] + sign * radii[:, None] * uy[None, :]).astype(np.int64)
        ri = np.broadcast_to(np.arange(nr)[:, None], cx.shape)
        ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        flat.append((ri[ok] * H + cy[ok]) * W + cx[ok])
    acc = np.bincount(np.concatenate(flat), minlength=nr * H * W).astype(np.float32).reshape(nr, H, W)
    # collect votes that land one cell off (3x3 box sum)
    p = np.pad(acc, ((0, 0), (1, 1), (1, 1)))           # separable 3x3 box sum
    p = p[:, :, :-2] + p[:, :, 1:-1] + p[:, :, 2:]
    acc = p[:, :-2, :] + p[:, 1:-1, :] + p[:, 2:, :]
    score = acc / (2.0 * math.pi * radii)[:, None, None]  # ~ fraction of the circumference with edges
    out = []
    # best circle per size band (small / medium / large), so texture blobs cannot crowd out a near ball
    bands = np.array_split(np.arange(nr), max_n)
    for band in bands:
        if len(band) == 0:
            continue
        sub = score[band[0]:band[-1] + 1].copy()
        for _ in range(config.CIRCLE_PER_BAND):     # top peaks per band (a corner can outscore a half-hidden ball)
            k = int(np.argmax(sub))
            bi, y, x = np.unravel_index(k, sub.shape)
            ri = band[0] + bi
            s = float(sub[bi, y, x])
            if s < config.CIRCLE_SCORE_MIN:
                break
            r = float(radii[ri])
            u, v, rf = _refine_circle(gfull, 2.0 * x + 0.5, 2.0 * y + 0.5, 2.0 * r)
            out.append({"u": u, "v": v, "r": rf, "area": int(math.pi * rf ** 2),
                        "bbox": (int(u - rf), int(v - rf), int(2 * rf), int(2 * rf)),
                        "fill": math.pi / 4.0, "aspect": 1.0, "kind": "circle", "score": s})
            rr = int(r) + 1                          # suppress this peak (this band only)
            sub[:, max(0, y - rr):y + rr + 1, max(0, x - rr):x + rr + 1] = 0.0
    return out


def _same_object(c, b):
    """A circle duplicates a colour blob only if it is near it AND about the same size; a ball merged
    with a same-coloured background (bigger, off-centre blob) keeps its circle candidate."""
    near = math.hypot(c["u"] - b["u"], c["v"] - b["v"]) <= 0.35 * max(c["r"], b["r"])
    return near and 0.7 <= c["r"] / max(b["r"], 1e-6) <= 1.4


def find_candidates(img, circles=True):
    """Ball candidates: colour blobs (ball colour test on a 4x grid, connected components, full-res
    refinement) plus round shapes from the circle detector that no colour blob already covers."""
    step = config.GRID_STEP
    max_area = config.MAX_BLOB_AREA
    if config.CAMERA_MODE == "mono":        # ball at MONO_Z_MIN: radius f*R/Z px
        rmax = config.FOCAL_PX * (config.BALL_DIAM_M / 2.0) / config.MONO_Z_MIN
        max_area = max(max_area, int(math.pi * rmax * rmax * 1.2))
    grid = ball_mask(img[::step, ::step])
    cands = []
    for comp in _components(grid):
        if len(comp) < config.MIN_BLOB_CELLS or len(comp) * step * step > 2 * max_area:
            continue                      # too small, or a wall/paper-sized bright region
        gy = [c[0] for c in comp]; gx = [c[1] for c in comp]
        y0 = max(min(gy) * step - step, 0); y1 = min(max(gy) * step + 2 * step, config.IMG_H)
        x0 = max(min(gx) * step - step, 0); x1 = min(max(gx) * step + 2 * step, config.IMG_W)
        m = ball_mask(img[y0:y1, x0:x1])
        ys, xs = np.nonzero(m)
        area = len(ys)
        if area < 12 or area > max_area:
            continue
        bw = xs.max() - xs.min() + 1; bh = ys.max() - ys.min() + 1
        cands.append({
            "u": float(xs.mean() + x0), "v": float(ys.mean() + y0),
            "r": math.sqrt(area / math.pi), "area": int(area),
            "bbox": (int(xs.min() + x0), int(ys.min() + y0), int(bw), int(bh)),
            "fill": area / float(bw * bh), "aspect": bw / float(bh), "kind": "blob",
        })
    cands.sort(key=lambda c: -c["area"])
    cands = cands[:config.MAX_CANDIDATES]
    for c in (circle_candidates(img) if circles else []):
        if not any(_same_object(c, b) for b in cands):
            cands.append(c)
    return cands[:config.MAX_CANDIDATES + config.CIRCLE_MAX * config.CIRCLE_PER_BAND]


def isolated(img, c):
    """Ball-coloured inside the circle, NOT ball-coloured in a ring just outside it (a dark ball in front
    of a light background). Rejects corners/edges of big dark things that the circle detector finds."""
    H, W = img.shape[:2]
    u, v, r = c["u"], c["v"], max(c["r"], 2.0)
    ro = 1.6 * r
    y0, y1 = max(int(v - ro), 0), min(int(v + ro) + 1, H)
    x0, x1 = max(int(u - ro), 0), min(int(u + ro) + 1, W)
    if y1 - y0 < 3 or x1 - x0 < 3:
        return False
    m = ball_mask(img[y0:y1, x0:x1])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - u, yy - v)
    inner = d <= 0.8 * r
    ring = (d >= 1.2 * r) & (d <= ro)
    if inner.sum() < 5 or ring.sum() < 5:
        return False
    return m[inner].mean() >= config.ISOLATED_INSIDE_MIN and m[ring].mean() <= config.ISOLATED_RING_MAX


def shape_ok(c):
    """Lenient geometric test used when the CNN rejects a candidate: a roughly round blob, allowed to be
    partly cut off, merged with fingers or a bit oval - but not a filled rectangle (fill ~1.0)."""
    if c["r"] < config.SHAPE_MIN_RADIUS_PX:
        return False
    if c.get("kind") == "circle":
        return c.get("score", 0.0) >= config.SHAPE_CIRCLE_SCORE
    asp = c.get("aspect", 1.0)
    return (config.SHAPE_FILL_MIN <= c.get("fill", 0.0) <= config.SHAPE_FILL_MAX
            and 1.0 / config.SHAPE_ASPECT_MAX <= asp <= config.SHAPE_ASPECT_MAX)


def crop_patch(img, cand, size=config.CNN_SIZE):
    """Square crop around the blob (2.4 x radius), nearest-neighbour resize to size x size."""
    side = max(8.0, 2.4 * cand["r"])
    k = (np.arange(size, dtype=np.float32) + 0.5) * side / size - side / 2.0
    ys = np.clip((cand["v"] + k).astype(np.intp), 0, img.shape[0] - 1)
    xs = np.clip((cand["u"] + k).astype(np.intp), 0, img.shape[1] - 1)
    return img[ys[:, None], xs[None, :]]


# =========================================================================== classifier
def _load_interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter  # the Pi (QNX apk python3-tflite-runtime)
        return Interpreter
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf  # laptop fallback for single-machine tests
        return tf.lite.Interpreter
    except ImportError:
        return None


class BallClassifier:
    """int8 TFLite CNN: 32x32x3 patch -> P(ball). Falls back to a roundness heuristic if no model."""

    def __init__(self, model_path):
        self.kind = "heuristic"
        self.interp = None
        if model_path and os.path.exists(model_path):
            Interp = _load_interpreter_class()
            if Interp is None:
                print("[perception] no TFLite runtime found, using heuristic classifier", flush=True)
            else:
                try:
                    self.interp = Interp(model_path=model_path, num_threads=1)
                except TypeError:
                    self.interp = Interp(model_path=model_path)
                self.interp.allocate_tensors()
                self.inp = self.interp.get_input_details()[0]
                self.out = self.interp.get_output_details()[0]
                self.kind = "cnn"
        else:
            print("[perception] model %s not found, using heuristic classifier" % model_path, flush=True)

    def p_ball(self, patch, cand):
        if self.interp is None:
            round_ok = 0.62 <= cand["fill"] <= 0.9 and 0.75 <= cand["aspect"] <= 1.33
            return 0.9 if round_ok else 0.1
        x = patch.astype(np.float32)[None, ...] / 255.0
        dt = self.inp["dtype"]
        if dt in (np.int8, np.uint8):
            scale, zp = self.inp["quantization"]
            info = np.iinfo(dt)
            x = np.clip(np.round(x / scale + zp), info.min, info.max).astype(dt)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])[0].astype(np.float32)
        if self.out["dtype"] in (np.int8, np.uint8):
            scale, zp = self.out["quantization"]
            y = (y - zp) * scale
        s = float(np.sum(y))
        if s <= 0:
            return 0.0
        return float(y[config.CNN_BALL_CLASS] / s)


# =========================================================================== stereo
def _rows(a, y0, y1, shift):
    """Rows y0-shift .. y1-shift of a (zero outside the image): the right camera may sit higher/lower."""
    out = np.zeros((y1 - y0,) + a.shape[1:], a.dtype)
    s0, s1 = y0 - shift, y1 - shift
    c0, c1 = max(s0, 0), min(s1, a.shape[0])
    if c1 > c0:
        out[c0 - s0:c1 - s0] = a[c0:c1]
    return out


def silhouette_disparity(mask_l, mask_r, cl, d0, dv=0.0):
    """Match the left ball silhouette against the right mask over d0 +/- SIL_SEARCH_PX.
    Returns (disparity_subpx, on_edge, match_ratio)."""
    H, W = mask_l.shape
    r = cl["r"]
    y0 = max(int(cl["v"] - 1.3 * r - 2), 0); y1 = min(int(cl["v"] + 1.3 * r + 3), H)
    x0 = max(int(cl["u"] - 1.3 * r - 2), 0); x1 = min(int(cl["u"] + 1.3 * r + 3), W)
    L = mask_l[y0:y1, x0:x1].astype(np.int16)
    pad = int(config.DISP_MAX) + config.SIL_SEARCH_PX + 2
    Rp = np.zeros((y1 - y0, W + 2 * pad), np.int16)
    Rp[:, pad:pad + W] = _rows(mask_r, y0, y1, int(round(dv)))
    dc = int(round(d0))
    ds = np.arange(dc - config.SIL_SEARCH_PX, dc + config.SIL_SEARCH_PX + 1)
    costs = np.empty(len(ds), np.float64)
    for i, d in enumerate(ds):
        Rw = Rp[:, x0 - d + pad: x1 - d + pad]
        costs[i] = np.abs(L - Rw).sum()
    i = int(np.argmin(costs))
    on_edge = i == 0 or i == len(ds) - 1
    sub = 0.0
    if not on_edge:
        c0, c1, c2 = costs[i - 1], costs[i], costs[i + 1]
        den = c0 - 2 * c1 + c2
        if den > 1e-9:
            sub = 0.5 * (c0 - c2) / den
    total = float(L.sum()) * 2.0
    match_ratio = 1.0 - costs[i] / max(total, 1.0)
    return float(ds[i] + sub), on_edge, float(match_ratio)


def gray_disparity(left, right, cl, d0, search=config.GRAY_SEARCH_PX, dv=0.0):
    """Refine disparity by grey-level SAD inside the ball's circle (radius 1.05 r) over d0 +/- search.
    Used when a candidate came from the circle detector (its colour mask may include background).
    Returns (disparity_subpx, on_edge, match_ratio)."""
    H, W = left.shape[:2]
    r = cl["r"]
    rr = int(math.ceil(1.05 * r)) + 1
    u, v = int(round(cl["u"])), int(round(cl["v"]))
    y0, y1 = max(v - rr, 0), min(v + rr + 1, H)
    x0, x1 = max(u - rr, 0), min(u + rr + 1, W)
    gl = _gray(left[y0:y1, x0:x1])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    wmask = ((xx - cl["u"]) ** 2 + (yy - cl["v"]) ** 2) <= (1.05 * r) ** 2
    gr_full = _gray(_rows(right, y0, y1, int(round(dv))))
    dc = int(round(d0))
    search = max(search, int(0.6 * r))                 # circle centres can be off by ~0.3 r per image
    ds = np.arange(dc - search, dc + search + 1)
    costs = np.full(len(ds), np.inf)
    for i, d in enumerate(ds):
        a, b = x0 - d, x1 - d
        if a < 0 or b > W:
            continue
        costs[i] = np.abs(gl - gr_full[:, a:b])[wmask].mean()
    if not np.isfinite(costs).any():
        return float(d0), True, 0.0
    i = int(np.argmin(costs))
    on_edge = i == 0 or i == len(ds) - 1 or not np.isfinite(costs[i - 1]) or not np.isfinite(costs[i + 1])
    sub = 0.0
    if not on_edge:
        c0, c1, c2 = costs[i - 1], costs[i], costs[i + 1]
        den = c0 - 2 * c1 + c2
        if den > 1e-9:
            sub = 0.5 * (c0 - c2) / den
    finite = costs[np.isfinite(costs)]
    match = 1.0 - costs[i] / max(float(np.median(finite)), 1e-6)
    return float(ds[i] + sub), on_edge, float(max(0.0, min(1.0, match)))


def _box_sum(a, k):
    """k x k box sum over the last two axes (same size, zero padded)."""
    p = k // 2
    pad = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(p + 1, p), (p + 1, p)])
    c = pad.cumsum(-1).cumsum(-2)
    return c[..., k:, k:] - c[..., :-k, k:] - c[..., k:, :-k] + c[..., :-k, :-k]


def disparity_map(left, right):
    """Coarse display disparity (80x60): SAD block matching + left-right consistency.
    Returns uint8 map: 0 invalid, 255 = DISP_MAX/2 full-res px."""
    sh, sw = config.IMG_H // config.DEPTH_H, config.IMG_W // config.DEPTH_W
    def gray(img):
        g = img.astype(np.float32).mean(axis=2)
        return g.reshape(config.DEPTH_H, sh, config.DEPTH_W, sw).mean(axis=(1, 3))
    gl, gr = gray(left), gray(right)
    H, W = gl.shape
    D = int(config.DISP_MAX / sw) + 1
    raw = np.full((D, H, W), 1e9, np.float32)
    for d in range(D):
        raw[d, :, d:] = np.abs(gl[:, d:] - gr[:, :W - d])
    big = raw >= 1e8
    raw[big] = 255.0
    agg = _box_sum(raw, config.DEPTH_BLOCK)
    agg[big] = 1e9
    dl = np.argmin(agg, axis=0)
    # right-image disparity: cost of right pixel xr at d is agg[d, :, xr + d]
    aggr = np.full_like(agg, 1e9)
    for d in range(D):
        aggr[d, :, :W - d] = agg[d, :, d:]
    dr = np.argmin(aggr, axis=0)
    xs = np.arange(W)[None, :] - dl
    ok = xs >= 0
    dr_at = np.take_along_axis(dr, np.clip(xs, 0, W - 1), axis=1)
    valid = ok & (np.abs(dl - dr_at) <= 1) & (dl > 0)
    # texture check: flat regions cannot be matched
    mean = _box_sum(gl[None], config.DEPTH_BLOCK)[0] / config.DEPTH_BLOCK ** 2
    var = _box_sum(gl[None] ** 2, config.DEPTH_BLOCK)[0] / config.DEPTH_BLOCK ** 2 - mean ** 2
    valid &= var > 4.0
    val = np.clip(dl * sw / (config.DISP_MAX / 2.0) * 255.0, 1, 255)
    return np.where(valid, val, 0).astype(np.uint8)


# =========================================================================== Kalman filter
class CVKalman:
    """3-D constant-velocity Kalman filter in the board frame (m, m/s real)."""

    def __init__(self):
        self.x = None
        self.P = None
        self.t = None

    def reset(self):
        self.x = None

    def update(self, t, z, r_diag):
        z = np.asarray(z, dtype=float)
        R = np.diag(r_diag)
        if self.x is None or t - self.t > config.KF_RESET_S:
            self.x = np.concatenate([z, np.zeros(3)])
            self.P = np.diag(np.concatenate([r_diag, [0.25, 0.25, 0.25]]))
            self.t = t
            return
        dt = max(t - self.t, 1e-3)
        F = np.eye(6); F[0:3, 3:6] = np.eye(3) * dt
        q = config.KF_ACCEL_SIGMA ** 2
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = np.eye(3) * q * dt ** 4 / 4
        Q[0:3, 3:6] = Q[3:6, 0:3] = np.eye(3) * q * dt ** 3 / 2
        Q[3:6, 3:6] = np.eye(3) * q * dt ** 2
        x = F @ self.x
        P = F @ self.P @ F.T + Q
        H = np.zeros((3, 6)); H[:, 0:3] = np.eye(3)
        y = z - H @ x
        S = H @ P @ H.T + R
        Si = np.linalg.inv(S)
        if float(y @ Si @ y) > 60.0:      # gross mismatch: restart rather than smear
            self.x = np.concatenate([z, np.zeros(3)])
            self.P = np.diag(np.concatenate([r_diag, [0.25, 0.25, 0.25]]))
            self.t = t
            return
        K = P @ H.T @ Si
        self.x = x + K @ y
        self.P = (np.eye(6) - K @ H) @ P
        self.t = t

    def state(self):
        return self.x[0:3].copy(), self.x[3:6].copy()


# =========================================================================== per-frame pipeline
class Perception:
    def __init__(self, model_path, cam=None):
        self.cam = cam or Camera()
        self.clf = BallClassifier(model_path)
        self.kf = CVKalman()
        self.epi_tol = config.EPIPOLAR_TOL_PX

    def process_mono(self, img):
        """ONE camera: candidates -> CNN (or shape fallback) -> depth from the ball's known size.
        Z = f * (BALL_DIAM_M / 2) / r. The radius comes from a circle fit on the ball's outline
        (robust to fingers covering part of it); a ball cut by the image edge is rejected (size wrong)."""
        res = {"detected": False, "reason": "ok", "n_candidates": 0, "cnn_ms": 0.0, "stereo_ms": 0.0,
               "cands": ([], [])}
        cnn_s = [0.0]
        cands, ok = self._detect(img, cnn_s)
        res["cands"] = (cands, [])
        res["n_candidates"] = len(cands)
        res["cnn_ms"] = cnn_s[0] * 1000.0
        if not cands:
            res["reason"] = "no_candidates"
            return res
        if not ok:
            res["reason"] = "cnn_rejected"
            return res
        t1 = time.perf_counter()
        c = self.cam
        g = _gray(img)
        H, W = g.shape
        reason = "cut_by_edge"
        order = sorted(ok, key=lambda q: (q.get("via") == "shape", -q["p_ball"]))
        for q in order[:config.MAX_PAIRS_TRIED]:
            u, v, r = q["u"], q["v"], q["r"]
            if q.get("kind") == "blob":
                bw, bh = q["bbox"][2], q["bbox"][3]
                r = min(max(r, 0.5 * max(bw, bh)), 1.35 * q["r"])   # area radius underestimates partly hidden balls
            u2, v2, r2 = _refine_circle(g, u, v, r)
            if abs(r2 / max(r, 1e-6) - 1.0) <= 0.35 and math.hypot(u2 - u, v2 - v) <= 0.5 * r:
                u, v, r = u2, v2, r2
            m = config.EDGE_MARGIN_PX
            if u - r < m or u + r > W - m or v - r < m or v + r > H - m:
                reason = "cut_by_edge"
                continue
            r = max(r - config.MONO_RADIUS_BIAS_PX, 1.0)   # outline edge is found ~0.6 px outside the true rim
            Z = c.f * (config.BALL_DIAM_M / 2.0) / r
            if not (config.MONO_Z_MIN <= Z <= config.MONO_Z_MAX):
                reason = "out_of_range"
                continue
            X = (u - c.cx) * Z / c.f
            Y = (v - c.cy) * Z / c.f
            sr = config.MONO_RADIUS_SIGMA_PX
            res.update({
                "detected": True, "reason": "ok", "pos_raw": [X, Y, Z], "disparity_px": 0.0,
                "depth_m": Z, "range_m": math.sqrt(X * X + Y * Y + Z * Z), "conf": q["p_ball"],
                "match_ratio": 1.0, "uv": [u, v], "radius_px": r, "dv": 0.0, "size_ratio": 1.0,
                "depth_src": "size", "via": q.get("via", "cnn"),
                "sigma": [0.7 * Z / c.f, 0.7 * Z / c.f, Z * sr / max(r, 1e-6)],
            })
            res["stereo_ms"] = (time.perf_counter() - t1) * 1000.0
            return res
        res["reason"] = reason
        res["stereo_ms"] = (time.perf_counter() - t1) * 1000.0
        return res

    def _balls(self, img, cands, cnn_s):
        t = time.perf_counter()
        out = []
        for c in cands:
            p = self.clf.p_ball(crop_patch(img, c), c)
            c["p_cnn"] = p
            c["via"] = "cnn"
            if p < config.CNN_THRESHOLD and config.SHAPE_FALLBACK and shape_ok(c) and isolated(img, c):
                p, c["via"] = config.SHAPE_SCORE, "shape"   # round, isolated blob: let stereo/size decide
            c["p_ball"] = p
            if p >= min(config.CNN_THRESHOLD, config.SHAPE_SCORE if config.SHAPE_FALLBACK else 1.0):
                out.append(c)
        cnn_s[0] += time.perf_counter() - t
        return out

    def _detect(self, img, cnn_s):
        """Colour blobs first; the (slower) circle detector only if no blob passes the CNN."""
        cands = find_candidates(img, circles=False)
        ok = self._balls(img, cands, cnn_s)
        if not any(c.get("via") == "cnn" for c in ok):     # shape-only matches do not stop the search
            extra = [c for c in circle_candidates(img)
                     if not any(_same_object(c, b) for b in cands)]
            ok = ok + self._balls(img, extra, cnn_s)
            cands = cands + extra
        return cands, ok

    def process(self, left, right):
        """Returns a dict with detection results (no filtering)."""
        res = {"detected": False, "reason": "ok", "n_candidates": 0, "cnn_ms": 0.0, "stereo_ms": 0.0}
        cnn_s = [0.0]

        def balls(img, cands):
            t = time.perf_counter()
            out = []
            for c in cands:
                p = self.clf.p_ball(crop_patch(img, c), c)
                c["p_cnn"] = p
                c["via"] = "cnn"
                if p < config.CNN_THRESHOLD and config.SHAPE_FALLBACK and shape_ok(c):
                    p, c["via"] = config.SHAPE_SCORE, "shape"   # roughly round: let stereo + size decide
                c["p_ball"] = p
                if p >= min(config.CNN_THRESHOLD, config.SHAPE_SCORE if config.SHAPE_FALLBACK else 1.0):
                    out.append(c)
            cnn_s[0] += time.perf_counter() - t
            return out

        def detect(img):
            """Colour blobs first; the (slower) circle detector only if no blob passes the CNN."""
            cands = find_candidates(img, circles=False)
            ok = balls(img, cands)
            if not any(c.get("via") == "cnn" for c in ok):     # shape-only matches do not stop the search
                extra = [c for c in circle_candidates(img)
                         if not any(_same_object(c, b) for b in cands)]
                ok = ok + balls(img, extra)
                cands = cands + extra
            return cands, ok

        cl_all, bl = self._detect(left, cnn_s)
        cr_all, br = self._detect(right, cnn_s)
        res["cands"] = (cl_all, cr_all)            # for the live preview
        res["n_candidates"] = len(cl_all) + len(cr_all)
        res["cnn_ms"] = cnn_s[0] * 1000.0
        if not cl_all or not cr_all:
            res["reason"] = "no_candidates_" + ("left" if not cl_all else "right")
            return res
        if not bl or not br:
            res["reason"] = "cnn_rejected_" + ("left" if not bl else "right")
            return res
        t1 = time.perf_counter()
        c = self.cam
        # diagnostics for the live view: row offset and raw disparity of the best-looking pair
        a0 = max(bl, key=lambda q: q["p_ball"]); b0 = max(br, key=lambda q: q["p_ball"])
        res["dv"] = a0["v"] - b0["v"]
        res["d_raw"] = a0["u"] - b0["u"]
        pairs = []
        why = {"epi": 0, "disp": 0, "lr_size": 0}
        for a in bl:
            for b in br:
                dv = a["v"] - b["v"]                     # vertical offset (camera misalignment)
                d = a["u"] - b["u"]
                if abs(dv) > self.epi_tol:
                    why["epi"] += 1; continue
                if not (config.DISP_MIN <= d <= config.DISP_MAX):
                    why["disp"] += 1; continue
                rr = a["r"] / max(b["r"], 1e-6)
                if not (1.0 / config.PAIR_RADIUS_RATIO <= rr <= config.PAIR_RADIUS_RATIO):
                    why["lr_size"] += 1; continue          # the same ball has about the same size in both images
                score = abs(dv) / max(self.epi_tol, 1.0) + abs(math.log(rr)) + (1.0 - min(a["p_ball"], b["p_ball"]))
                pairs.append((score, a, b, d, dv))
        if not pairs:
            res["reason"] = ("size_mismatch_left_right" if why["lr_size"] else
                             "disparity_out_of_range" if why["disp"] else "epipolar_mismatch")
            res["stereo_ms"] = (time.perf_counter() - t1) * 1000.0
            return res
        # CNN-approved pairs first; pairs that rely on the lenient shape fallback only after those
        pairs.sort(key=lambda p: (p[1].get("via") == "shape" or p[2].get("via") == "shape", p[0]))
        masks = None
        reason = "size_mismatch"
        for _, a, b, d0, dv in pairs[:config.MAX_PAIRS_TRIED]:
            by_shape = a.get("via") == "shape" or b.get("via") == "shape"
            lo, hi = ((config.SHAPE_SIZE_RATIO_MIN, config.SHAPE_SIZE_RATIO_MAX) if by_shape
                      else (config.SIZE_RATIO_MIN, config.SIZE_RATIO_MAX))
            if a.get("kind") == "blob" and b.get("kind") == "blob":
                if masks is None:
                    masks = (ball_mask(left), ball_mask(right))
                d_fine, edge, match = silhouette_disparity(masks[0], masks[1], a, d0, dv)
            else:       # circle-detector candidate: its colour mask may include the background
                d_fine, edge, match = gray_disparity(left, right, a, d0, dv=dv)
            agree = max(config.SIL_AGREE_PX, config.SIL_AGREE_FRAC * d0)
            if a.get("kind") != "blob" or b.get("kind") != "blob":
                agree = max(agree, 0.6 * a["r"])         # circle centres are less exact than blob centroids
            if edge or abs(d_fine - d0) > agree:
                # fine match unsure (misaligned / partly hidden ball): fall back to the centre disparity,
                # the physical size check below still guards against wrong pairs
                d_use, src, match = d0, "centroid", 0.5 * max(match, 0.0)
            else:
                d_use, src = d_fine, "fine"
            if not (config.DISP_MIN <= d_use <= config.DISP_MAX):
                reason = "disparity_out_of_range"
                continue
            Z = c.f * c.baseline / d_use
            # physical size check: the ball (BALL_DIAM_M) at depth Z has radius f*(D/2)/Z px. Small round things
            # (buttons, eyes, coins, holes) look like a far ball to the CNN, but their stereo depth gives
            # them away.
            r_exp = c.f * (config.BALL_DIAM_M / 2.0) / Z
            size_ratio = 0.5 * (a["r"] + b["r"]) / r_exp
            if not (lo <= size_ratio <= hi):
                reason = "size_mismatch"
                res["size_ratio"] = size_ratio
                continue
            u_mid = a["u"] - d_use / 2.0 - c.cx      # average of left and (shifted) right
            X = u_mid * Z / c.f
            Y = (0.5 * (a["v"] + b["v"]) - c.cy) * Z / c.f
            res.update({
                "detected": True, "reason": "ok" if src == "fine" else "ok_centroid", "pos_raw": [X, Y, Z],
                "disparity_px": d_use, "depth_m": Z, "range_m": math.sqrt(X * X + Y * Y + Z * Z),
                "conf": min(a["p_ball"], b["p_ball"]) * max(0.0, min(1.0, match)),
                "match_ratio": match, "uv": [a["u"], a["v"]], "radius_px": a["r"], "dv": dv,
                "size_ratio": size_ratio, "depth_src": src, "via": "shape" if by_shape else "cnn",
                "sigma": [0.5 * Z / c.f, 0.5 * Z / c.f,
                          (0.5 if src == "fine" else 1.5) * Z * Z / (c.f * c.baseline)],
            })
            res["stereo_ms"] = (time.perf_counter() - t1) * 1000.0
            return res
        res["reason"] = reason
        res["stereo_ms"] = (time.perf_counter() - t1) * 1000.0
        return res


# =========================================================================== main loop
def _send(sock, msg, addr):
    try:
        sock.sendto(json.dumps(msg, default=float).encode(), addr)
    except OSError:
        pass  # network trouble must never stop perception


PREVIEW_CHUNK = 1400     # bytes of image data per UDP packet (no IP fragmentation on Wi-Fi)


class PreviewSender:
    """Streams both camera images (downscaled, zlib) + detection overlay data to live_view.py.
    Best-effort UDP: never blocks perception; lost packets just drop a preview frame."""

    def __init__(self, host, port=config.PREVIEW_PORT, fps=config.PREVIEW_FPS, scale=config.PREVIEW_SCALE):
        import zlib
        self.zlib = zlib
        self.addr = (host, port)
        self.period = 1.0 / max(fps, 0.1)
        self.scale = max(1, int(scale))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.last = 0.0
        self.fid = 0

    def maybe_send(self, left, right, res, extra):
        now = time.monotonic()
        if now - self.last < self.period:
            return
        self.last = now
        self.fid = (self.fid + 1) & 0xFFFFFFFF
        s = self.scale
        mono = right is None
        img = left[::s, ::s] if mono else np.concatenate([left[::s, ::s], right[::s, ::s]], axis=1)
        blob = self.zlib.compress(np.ascontiguousarray(img).tobytes(), 1)
        n = (len(blob) + PREVIEW_CHUNK - 1) // PREVIEW_CHUNK
        cl, cr = res.get("cands", ([], []))
        def cl_list(cs):
            return [[round(c["u"], 1), round(c["v"], 1), round(c["r"], 1), round(c.get("p_ball", -1.0), 3)] for c in cs]
        meta = {"fid": self.fid, "n": n, "w": img.shape[1] if mono else img.shape[1] // 2, "h": img.shape[0],
                "scale": s, "mono": mono,
                "cands_l": cl_list(cl), "cands_r": cl_list(cr), "detected": res["detected"],
                "reason": res["reason"], "cnn_ms": res["cnn_ms"], "stereo_ms": res["stereo_ms"],
                "dv": res.get("dv"), "d_raw": res.get("d_raw")}
        for k in ("disparity_px", "depth_m", "range_m", "conf", "uv", "radius_px", "pos_raw", "size_ratio",
                  "depth_src", "streak", "via"):
            if k in res:
                meta[k] = res[k]
        meta.update(extra)
        try:
            self.sock.sendto(b"DPVM" + json.dumps(meta, default=float).encode(), self.addr)
            for i in range(n):
                self.sock.sendto(b"DPVC" + np.array([self.fid, i, n], "<u4").tobytes()
                                 + blob[i * PREVIEW_CHUNK:(i + 1) * PREVIEW_CHUNK], self.addr)
        except OSError:
            pass   # buffer full / network down: drop this preview frame


def calibrate_focal(source, per, cam, dist_m, mono):
    """Mono focal-length calibration: ball held still at a measured distance (lens to ball centre).
    f = r_px * Z / (D/2), median over ~4 s, written to calib.json (read by Camera() on start)."""
    print("[calibrate] hold the ball still, centred, %.2f m from the lens ..." % dist_m, flush=True)
    rs = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 6.0 and len(rs) < 60:
        _, left, right = source.read()
        res = per.process_mono(left) if right is None else per.process(left, right)
        if res["detected"]:
            rs.append(res["radius_px"])
    if len(rs) < 10:
        print("[calibrate] only %d detections - is the ball visible (live view)? nothing written" % len(rs), flush=True)
        return 1
    r = float(np.median(rs))
    f_new = r * dist_m / (config.BALL_DIAM_M / 2.0)
    path = os.path.join(HERE, config.CALIB_PATH)
    data = {"f": f_new, "cx": cam.cx, "cy": cam.cy, "baseline": cam.baseline,
            "note": "mono focal from ball radius %.2f px at %.3f m" % (r, dist_m)}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print("[calibrate] median radius %.2f px (spread %.2f) -> f = %.1f px (was %.1f). Wrote %s" % (
        r, float(np.std(rs)), f_new, cam.f, path), flush=True)
    return 0


def save_npy(dirname, prefix, idx, arr):
    os.makedirs(dirname, exist_ok=True)
    np.save(os.path.join(dirname, "%s_%06d.npy" % (prefix, idx)), arr)


def main():
    ap = argparse.ArgumentParser(description="DODGE-1 perception (Pi)")
    ap.add_argument("--source", choices=["synthetic", "qnx"], default="synthetic")
    ap.add_argument("--model", default=os.path.join(HERE, config.MODEL_PATH))
    ap.add_argument("--fps", type=float, default=config.CAMERA_FPS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save-patches", default=None, help="save every candidate patch (.npy) to DIR")
    ap.add_argument("--save-pairs", default=None, help="save every 10th stereo pair (.npz) to DIR")
    ap.add_argument("--cam-units", default=None,
                    help="sensor units for left,right, e.g. 1,2 (default: first two the sensor service reports)")
    ap.add_argument("--swap-lr", action="store_true", help="swap left/right cameras")
    ap.add_argument("--qnx-backend", choices=["shm", "ctypes"], default="shm",
                    help="shm: frames from qnx/stereo_capture (C); ctypes: libcamapi callbacks into Python")
    ap.add_argument("--epipolar-tol", type=float, default=config.EPIPOLAR_TOL_PX,
                    help="max row difference (px) between the ball in both images; raise it before calibration")
    ap.add_argument("--preview", default=None, metavar="LAPTOP_IP",
                    help="stream annotated camera images to live_view.py on this host")
    ap.add_argument("--preview-fps", type=float, default=config.PREVIEW_FPS)
    ap.add_argument("--preview-scale", type=int, default=config.PREVIEW_SCALE, help="1 = full 320x240, 2 = half")
    ap.add_argument("--calibrate-f", type=float, default=None, metavar="DIST_M",
                    help="mono: hold the ball at this distance (m, lens to ball centre) -> writes the focal length to calib.json")
    args = ap.parse_args()
    import faulthandler
    faulthandler.enable()      # native crash (e.g. camera library) -> print the Python stack

    cam = Camera()
    mono = config.CAMERA_MODE == "mono"
    if args.source == "qnx" and mono:
        source = QnxMonoSource()
        print("[perception] QNX camera: sensor unit %s (mono, depth from the %.0f mm ball size)" % (
            source.units[0], config.BALL_DIAM_M * 1000), flush=True)
    elif args.source == "qnx":
        units = None
        if args.cam_units:
            units = [int(u) for u in args.cam_units.split(",")]   # sensor unit n = CAMERA_UNIT_n = value n
        source = QnxStereoSource(units=units, swap=args.swap_lr, backend=args.qnx_backend)
        print("[perception] QNX cameras: sensor units %s (left, right)" % source.units, flush=True)
    else:
        source = SyntheticStereoSource(fps=args.fps, seed=args.seed, cam=cam, mono=mono)
    maps = load_rectify_maps()
    per = Perception(args.model, cam=cam)
    per.epi_tol = args.epipolar_tol
    preview = PreviewSender(args.preview, fps=args.preview_fps, scale=args.preview_scale) if args.preview else None
    sim_feed = (PreviewSender(config.SIM_IP, port=config.SIM_FEED_PORT, fps=config.SIM_FEED_FPS,
                              scale=config.SIM_FEED_SCALE) if config.SIM_FEED else None)
    if mono:
        maps = None
    print("[perception] mode=%s source=%s classifier=%s camera=%s (f=%.1f px, baseline=%.3f m) rectify=%s preview=%s pid=%d" % (
        config.CAMERA_MODE, args.source, per.clf.kind, cam.source, cam.f, cam.baseline, maps is not None,
        args.preview, os.getpid()), flush=True)
    if args.calibrate_f:
        return calibrate_focal(source, per, cam, args.calibrate_f, mono)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_addr = ("127.0.0.1", config.CONTROL_PORT)
    sim_addr = (config.SIM_IP, config.SIM_PORT)
    sup_addr = ("127.0.0.1", config.SUPERVISOR_PORT)
    hb_period = 1.0 / config.HB_HZ
    last_hb = 0.0
    seq = 0
    last_det_t = -1e9
    patch_idx = 0
    fps_t0 = time.monotonic(); fps_n = 0; fps_rate = 0.0; fps_report = 0
    streak = 0; prev_det = None; frames_since_det = 0

    while True:
        t_cap, left, right = source.read()
        if maps is not None and right is not None:
            left = rectify(left, maps["map_lx"], maps["map_ly"])
            right = rectify(right, maps["map_rx"], maps["map_ry"])
        res = per.process_mono(left) if right is None else per.process(left, right)
        seq += 1

        if args.save_patches:
            for img, side in ((left, "L"), (right, "R")):
                if img is None:
                    continue
                for c in find_candidates(img):
                    save_npy(args.save_patches, "p%s" % side, patch_idx, crop_patch(img, c))
                    patch_idx += 1
        if args.save_pairs and seq % 10 == 0:
            os.makedirs(args.save_pairs, exist_ok=True)
            if right is not None:
                np.savez_compressed(os.path.join(args.save_pairs, "pair_%06d.npz" % seq), left=left, right=right)

        msg = {"type": "track", "seq": seq, "detected": False, "reason": res["reason"],
               "pos": None, "vel": None, "classifier": per.clf.kind,
               "cnn_ms": res["cnn_ms"], "stereo_ms": res["stereo_ms"], "n_candidates": res["n_candidates"]}
        if res["detected"]:
            # temporal confirmation: a one-frame false positive must not reach control
            p_now = np.array(res["pos_raw"])
            if prev_det is not None and np.linalg.norm(p_now - prev_det) <= config.CONFIRM_JUMP_M * max(1, frames_since_det):
                streak += 1
            else:
                streak = 1
            prev_det, frames_since_det = p_now, 0
            s = res["sigma"]
            per.kf.update(t_cap, res["pos_raw"], [s[0] ** 2, s[1] ** 2, s[2] ** 2])
            pos, vel = per.kf.state()
            last_det_t = t_cap
            if streak >= config.CONFIRM_FRAMES:
                msg.update({"detected": True, "pos": pos.tolist(), "vel": vel.tolist()})
                for k in ("disparity_px", "depth_m", "range_m", "conf", "match_ratio", "uv", "radius_px"):
                    msg[k] = res[k]
            else:
                msg["reason"] = "confirming"
        else:
            frames_since_det += 1
            if frames_since_det > 2:
                streak = 0
            if t_cap - last_det_t > config.KF_RESET_S:
                per.kf.reset()
        res["streak"] = streak
        msg["latency_s"] = time.monotonic() - t_cap
        _send(sock, msg, ctrl_addr)
        _send(sock, msg, sim_addr)

        if right is not None and seq % config.DEPTHMAP_EVERY == 0:
            dm = disparity_map(left, right)
            ball = None
            if res["detected"]:
                sx = config.DEPTH_W / float(config.IMG_W)
                bu, bv, br = res["uv"][0] * sx, res["uv"][1] * sx, res["radius_px"] * sx
                yy, xx = np.mgrid[0:config.DEPTH_H, 0:config.DEPTH_W]
                inside = (xx - bu) ** 2 + (yy - bv) ** 2 <= (br + 0.5) ** 2
                dm[inside] = int(np.clip(res["disparity_px"] / (config.DISP_MAX / 2.0) * 255, 1, 255))
                ball = [bu, bv]
            _send(sock, {"type": "depthmap", "w": config.DEPTH_W, "h": config.DEPTH_H,
                         "data": base64.b64encode(dm.tobytes()).decode("ascii"), "ball": ball}, sim_addr)

        now = time.monotonic()
        if now - last_hb >= hb_period:
            _send(sock, {"type": "hb", "name": "perception"}, sup_addr)
            last_hb = now
        fps_n += 1
        if now - fps_t0 >= 1.0:
            fps_rate = fps_n / (now - fps_t0)
            fps_t0, fps_n = now, 0
            fps_report += 1
            if fps_report % 10 == 0:
                print("[perception] %.1f fps, last reason=%s" % (fps_rate, res["reason"]), flush=True)
        if preview is not None or sim_feed is not None:
            extra = {
                "fps": fps_rate, "classifier": per.clf.kind, "latency_ms": (now - t_cap) * 1000.0,
                "skew_ms": getattr(source, "skew_ms", 0.0), "source": args.source,
                "src_info": getattr(source, "info", ""), "kf_pos": msg.get("pos"),
                "epi_tol": per.epi_tol, "baseline": cam.baseline, "f": cam.f}
            for sender in (preview, sim_feed):
                if sender is not None:
                    sender.maybe_send(left, right, res, extra)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
