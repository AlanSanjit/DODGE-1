"""DODGE-1 perception (runs on the Raspberry Pi 5 / QNX, LOW priority).

cameras -> orange candidates -> TFLite CNN in BOTH images -> stereo checks -> depth -> Kalman
-> `track` messages to control (127.0.0.1) and to the sim, plus a coarse `depthmap` for display.

numpy + tflite_runtime + stdlib only (no OpenCV on the Pi).
Perception never imports control and shares no memory with it: it only sends UDP messages.

    python3 perception.py --source synthetic        # rendered hand-held ball, wall, orange box
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


class SyntheticScene:
    """Renders what the two cameras would see: textured wall, an orange BOX distractor,
    a skin-coloured hand and the orange ping-pong ball (ray-cast spheres with Lambert shading)."""

    TEX_RES = 400.0          # texels per metre on the wall
    TEX_X = (-2.0, 2.0)
    TEX_Y = (-1.5, 1.5)

    def __init__(self, seed=0, randomize=False, cam=None):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.cam = cam or Camera()
        self.wall_z = rng.uniform(0.9, 1.8) if randomize else 1.2
        th = int((self.TEX_Y[1] - self.TEX_Y[0]) * self.TEX_RES)
        tw = int((self.TEX_X[1] - self.TEX_X[0]) * self.TEX_RES)
        n = 0.5 * _block_noise(rng, th, tw, 64) + 0.3 * _block_noise(rng, th, tw, 16) + 0.2 * _block_noise(rng, th, tw, 4)
        base = np.array(rng.uniform([90, 100, 115], [140, 145, 170]) if randomize else [110, 120, 140], np.float32)
        tex = base[None, None, :] * (0.55 + 0.9 * n[..., None])
        # posters in non-orange colours
        for _ in range(6):
            y0 = rng.integers(0, th - 200); x0 = rng.integers(0, tw - 200)
            hh = rng.integers(60, 200); ww = rng.integers(60, 200)
            col = np.array([rng.uniform(30, 120), rng.uniform(80, 220), rng.uniform(120, 240)], np.float32)
            if rng.random() < 0.3:
                col = np.array([230, 230, 225], np.float32)
            tex[y0:y0 + hh, x0:x0 + ww] = col * (0.8 + 0.4 * n[y0:y0 + hh, x0:x0 + ww, None])
        self.tex = np.clip(tex, 0, 255).astype(np.uint8)
        # orange box distractor (front face, fronto-parallel)
        if randomize:
            bz = rng.uniform(0.35, 0.9)
            bx = rng.uniform(-0.2, 0.15); by = rng.uniform(-0.1, 0.12)
            bs = rng.uniform(0.04, 0.1)
            self.box = (bx, bx + bs * rng.uniform(0.7, 1.6), by, by + bs, bz)
            self.box_col = np.array([rng.uniform(200, 255), rng.uniform(80, 140), rng.uniform(10, 45)], np.float32)
        else:
            self.box = (0.12, 0.20, 0.03, 0.11, 0.70)
            self.box_col = np.array([235, 112, 30], np.float32)
        self.ball_col = (np.array([rng.uniform(235, 255), rng.uniform(95, 135), rng.uniform(5, 35)], np.float32)
                         if randomize else np.array([250, 115, 20], np.float32))
        self.skin_col = np.array([205, 160, 130], np.float32)
        ld = np.array([-0.35, -0.6, -0.7]) if not randomize else np.array(
            [rng.uniform(-0.7, 0.7), rng.uniform(-0.9, -0.2), rng.uniform(-0.9, -0.4)])
        self.light = ld / np.linalg.norm(ld)
        self.noise = 2.5
        self.cams_x = (-self.cam.baseline / 2.0, self.cam.baseline / 2.0)
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

    def render(self, ball_p, hand=True, rng=None):
        rng = rng if rng is not None else self.rng
        out = []
        for i, cam_x in enumerate(self.cams_x):
            img = self.bg[i][0].copy()
            depth = self.bg[i][1].copy()
            if ball_p is not None:
                P = np.asarray(ball_p, dtype=float)
                if hand:
                    self._sphere(img, depth, P + np.array([0.004, 0.052, 0.03]), 0.040, cam_x, self.skin_col, spec=False)
                    self._sphere(img, depth, P + np.array([0.01, 0.12, 0.07]), 0.045, cam_x, self.skin_col * 0.95, spec=False)
                self._sphere(img, depth, P, config.BALL_DIAM_M / 2.0, cam_x, self.ball_col)
            gain = 1.0 if i == 0 else 0.97
            img = img * gain + rng.normal(0.0, self.noise, img.shape).astype(np.float32)
            out.append(np.clip(img, 0, 255).astype(np.uint8))
        return out[0], out[1]


class SyntheticStereoSource:
    """Stand-in for the two Camera Module 3 units: renders a person moving the ball."""

    def __init__(self, fps=config.CAMERA_FPS, seed=1, cam=None):
        self.period = 1.0 / fps
        self.scenario = spacemath.HandScenario(seed=seed)
        self.scene = SyntheticScene(seed=seed, cam=cam)
        self.t0 = time.monotonic()
        self.next_t = self.t0
        self.truth = None

    def read(self):
        now = time.monotonic()
        if now < self.next_t:
            time.sleep(self.next_t - now)
        self.next_t = max(self.next_t + self.period, time.monotonic() - self.period)
        t_cap = time.monotonic()
        p = self.scenario.position(t_cap - self.t0)
        self.truth = p
        left, right = self.scene.render(p)
        return t_cap, left, right


class QnxStereoSource:
    """QNX camera units 3 (left) and 4 (right) via a C capture bridge writing to /dev/shmem.

    NOT IMPLEMENTED YET (PRD phase 0): write the C bridge using the QNX camera library, then read
    [seq, t_capture, left 320x240x3, right 320x240x3] from shared memory here with mmap + numpy.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "QnxStereoSource needs the stereo capture bridge (CLAUDE.md 'Next work' phase 0). "
            "Use --source synthetic for now.")

    def read(self):
        raise NotImplementedError


# =========================================================================== candidates
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


def find_candidates(img):
    """Orange blobs: chromaticity test on a 4x grid, connected components, full-res refinement."""
    step = config.GRID_STEP
    grid = orange_mask(img[::step, ::step])
    cands = []
    for comp in _components(grid):
        if len(comp) < config.MIN_BLOB_CELLS:
            continue
        gy = [c[0] for c in comp]; gx = [c[1] for c in comp]
        y0 = max(min(gy) * step - step, 0); y1 = min(max(gy) * step + 2 * step, config.IMG_H)
        x0 = max(min(gx) * step - step, 0); x1 = min(max(gx) * step + 2 * step, config.IMG_W)
        m = orange_mask(img[y0:y1, x0:x1])
        ys, xs = np.nonzero(m)
        area = len(ys)
        if area < 12:
            continue
        bw = xs.max() - xs.min() + 1; bh = ys.max() - ys.min() + 1
        cands.append({
            "u": float(xs.mean() + x0), "v": float(ys.mean() + y0),
            "r": math.sqrt(area / math.pi), "area": int(area),
            "bbox": (int(xs.min() + x0), int(ys.min() + y0), int(bw), int(bh)),
            "fill": area / float(bw * bh), "aspect": bw / float(bh),
        })
    cands.sort(key=lambda c: -c["area"])
    return cands[:config.MAX_CANDIDATES]


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
def silhouette_disparity(mask_l, mask_r, cl, d0):
    """Match the left ball silhouette against the right mask over d0 +/- SIL_SEARCH_PX.
    Returns (disparity_subpx, on_edge, match_ratio)."""
    H, W = mask_l.shape
    r = cl["r"]
    y0 = max(int(cl["v"] - 1.3 * r - 2), 0); y1 = min(int(cl["v"] + 1.3 * r + 3), H)
    x0 = max(int(cl["u"] - 1.3 * r - 2), 0); x1 = min(int(cl["u"] + 1.3 * r + 3), W)
    L = mask_l[y0:y1, x0:x1].astype(np.int16)
    pad = int(config.DISP_MAX) + config.SIL_SEARCH_PX + 2
    Rp = np.zeros((y1 - y0, W + 2 * pad), np.int16)
    Rp[:, pad:pad + W] = mask_r[y0:y1]
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

    def process(self, left, right):
        """Returns a dict with detection results (no filtering)."""
        res = {"detected": False, "reason": "ok", "n_candidates": 0, "cnn_ms": 0.0, "stereo_ms": 0.0}
        cl_all = find_candidates(left)
        cr_all = find_candidates(right)
        res["n_candidates"] = len(cl_all) + len(cr_all)
        if not cl_all or not cr_all:
            res["reason"] = "no_candidates_" + ("left" if not cl_all else "right")
            return res
        t0 = time.perf_counter()
        def balls(img, cands):
            out = []
            for c in cands:
                p = self.clf.p_ball(crop_patch(img, c), c)
                c["p_ball"] = p
                if p >= config.CNN_THRESHOLD:
                    out.append(c)
            return out
        bl = balls(left, cl_all)
        br = balls(right, cr_all) if bl else []
        res["cnn_ms"] = (time.perf_counter() - t0) * 1000.0
        if not bl or not br:
            res["reason"] = "cnn_rejected_" + ("left" if not bl else "right")
            return res
        t1 = time.perf_counter()
        best = None
        for a in bl:
            for b in br:
                dv = abs(a["v"] - b["v"])
                d = a["u"] - b["u"]
                if dv > config.EPIPOLAR_TOL_PX:
                    continue
                if not (config.DISP_MIN <= d <= config.DISP_MAX):
                    continue
                score = dv + abs(a["r"] - b["r"])
                if best is None or score < best[0]:
                    best = (score, a, b, d)
        if best is None:
            ok_epi = any(abs(a["v"] - b["v"]) <= config.EPIPOLAR_TOL_PX for a in bl for b in br)
            res["reason"] = "disparity_out_of_range" if ok_epi else "epipolar_mismatch"
            res["stereo_ms"] = (time.perf_counter() - t1) * 1000.0
            return res
        _, a, b, d0 = best
        ml, mr = orange_mask(left), orange_mask(right)
        d_sil, edge, match = silhouette_disparity(ml, mr, a, d0)
        res["stereo_ms"] = (time.perf_counter() - t1) * 1000.0
        if edge:
            res["reason"] = "silhouette_edge"
            return res
        if abs(d_sil - d0) > max(config.SIL_AGREE_PX, config.SIL_AGREE_FRAC * d0):
            res["reason"] = "silhouette_disagree"
            return res
        if not (config.DISP_MIN <= d_sil <= config.DISP_MAX):
            res["reason"] = "disparity_out_of_range"
            return res
        c = self.cam
        Z = c.f * c.baseline / d_sil
        u_mid = a["u"] - d_sil / 2.0 - c.cx      # average of left and (silhouette-shifted) right
        X = u_mid * Z / c.f
        Y = (0.5 * (a["v"] + b["v"]) - c.cy) * Z / c.f
        res.update({
            "detected": True, "reason": "ok", "pos_raw": [X, Y, Z],
            "disparity_px": d_sil, "depth_m": Z, "range_m": math.sqrt(X * X + Y * Y + Z * Z),
            "conf": min(a["p_ball"], b["p_ball"]) * max(0.0, min(1.0, match)),
            "match_ratio": match, "uv": [a["u"], a["v"]], "radius_px": a["r"],
            "sigma": [0.5 * Z / c.f, 0.5 * Z / c.f, 0.5 * Z * Z / (c.f * c.baseline)],
        })
        return res


# =========================================================================== main loop
def _send(sock, msg, addr):
    try:
        sock.sendto(json.dumps(msg, default=float).encode(), addr)
    except OSError:
        pass  # network trouble must never stop perception


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
    args = ap.parse_args()

    cam = Camera()
    if args.source == "qnx":
        source = QnxStereoSource()
    else:
        source = SyntheticStereoSource(fps=args.fps, seed=args.seed, cam=cam)
    maps = load_rectify_maps()
    per = Perception(args.model, cam=cam)
    print("[perception] source=%s classifier=%s camera=%s rectify=%s pid=%d" % (
        args.source, per.clf.kind, cam.source, maps is not None, os.getpid()), flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_addr = ("127.0.0.1", config.CONTROL_PORT)
    sim_addr = (config.SIM_IP, config.SIM_PORT)
    sup_addr = ("127.0.0.1", config.SUPERVISOR_PORT)
    hb_period = 1.0 / config.HB_HZ
    last_hb = 0.0
    seq = 0
    last_det_t = -1e9
    patch_idx = 0
    fps_t0 = time.monotonic(); fps_n = 0

    while True:
        t_cap, left, right = source.read()
        if maps is not None:
            left = rectify(left, maps["map_lx"], maps["map_ly"])
            right = rectify(right, maps["map_rx"], maps["map_ry"])
        res = per.process(left, right)
        seq += 1

        if args.save_patches:
            for img, side in ((left, "L"), (right, "R")):
                for c in find_candidates(img):
                    save_npy(args.save_patches, "p%s" % side, patch_idx, crop_patch(img, c))
                    patch_idx += 1
        if args.save_pairs and seq % 10 == 0:
            os.makedirs(args.save_pairs, exist_ok=True)
            np.savez_compressed(os.path.join(args.save_pairs, "pair_%06d.npz" % seq), left=left, right=right)

        msg = {"type": "track", "seq": seq, "detected": False, "reason": res["reason"],
               "pos": None, "vel": None, "classifier": per.clf.kind,
               "cnn_ms": res["cnn_ms"], "stereo_ms": res["stereo_ms"], "n_candidates": res["n_candidates"]}
        if res["detected"]:
            s = res["sigma"]
            per.kf.update(t_cap, res["pos_raw"], [s[0] ** 2, s[1] ** 2, s[2] ** 2])
            pos, vel = per.kf.state()
            last_det_t = t_cap
            msg.update({"detected": True, "pos": pos.tolist(), "vel": vel.tolist()})
            for k in ("disparity_px", "depth_m", "range_m", "conf", "match_ratio", "uv", "radius_px"):
                msg[k] = res[k]
        elif t_cap - last_det_t > config.KF_RESET_S:
            per.kf.reset()
        msg["latency_s"] = time.monotonic() - t_cap
        _send(sock, msg, ctrl_addr)
        _send(sock, msg, sim_addr)

        if seq % config.DEPTHMAP_EVERY == 0:
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
        if now - fps_t0 >= 10.0:
            print("[perception] %.1f fps, last reason=%s" % (fps_n / (now - fps_t0), res["reason"]), flush=True)
            fps_t0, fps_n = now, 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
