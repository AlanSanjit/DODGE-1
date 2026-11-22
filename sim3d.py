"""DODGE-1 3D simulator: "the universe" (laptop).

Rigid-body physics of the satellite (6 cold-gas thrusters with A/B branches, reaction wheels,
Clohessy-Wiltshire relative orbit), the asteroid drawn from the Pi's `track` messages, and a
dashboard. It makes NO decisions: every valve and wheel command comes from control.py on the Pi.

    SIM_IP=<laptop-ip> PI_IP=<pi-ip> python3 sim3d.py        (start FIRST)
Keys: 1 chase, 2 side, 3 onboard camera, M mock ball, S stuck valve +z:A, F drain propellant,
      R reset, ESC quit.
"""
import argparse
import base64
import collections
import json
import math
import socket
import sys
import threading
import time

import numpy as np

import config
import spacemath as sm

from panda3d.core import loadPrcFileData

loadPrcFileData("", "window-title DODGE-1 simulator\nwin-size 1280 720\nsync-video 0")

from direct.showbase.ShowBase import ShowBase                       # noqa: E402
from direct.gui.OnscreenText import OnscreenText                    # noqa: E402
from direct.gui.OnscreenImage import OnscreenImage                  # noqa: E402
from panda3d.core import (AmbientLight, DirectionalLight, Geom, GeomNode, GeomPoints,  # noqa: E402
                          GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter,
                          LineSegs, LMatrix3f, LPoint3f, LQuaternionf, NodePath, TextNode,
                          Texture, TextureStage, TransparencyAttrib, Vec3, Vec4)

AXES = "xyz"
M_SCENE = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])   # scene = (-z, y, x) of LVLH


def to_scene(p):
    s = M_SCENE @ np.asarray(p, dtype=float)
    return LPoint3f(float(s[0]), float(s[1]), float(s[2]))


def valve_dir(v):
    d = np.zeros(3)
    d[AXES.index(v[1])] = 1.0 if v[0] == "+" else -1.0
    return d


# =========================================================================== physics (no Panda)
class Physics:
    """Satellite rigid body in LVLH, mission time. Physically honest: Newton 1/2, CW, Euler."""

    def __init__(self):
        self.I = np.array(config.INERTIA, float)
        self.com = np.array(config.COM_OFFSET, float)
        self.n = sm.MEAN_MOTION
        self.reset()

    def reset(self):
        self.t = 0.0
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        self.h_w = np.zeros(3)
        self.prop = config.PROP_MASS
        self.stuck = None            # e.g. "+z:A" (stuck OPEN)
        self.active = set()          # branches actually thrusting

    @property
    def mass(self):
        return config.DRY_MASS + self.prop

    def step(self, dt, valves, branch, isolated, torque):
        active = set()
        for v in valves:
            if v in config.VALVES:
                active.add(v + ":" + branch.get(v, "A"))
        if self.stuck:
            active.add(self.stuck)
        active -= set(isolated)       # an isolated branch's latch valve is closed
        if self.prop <= 0.0:
            active = set()
        F = np.zeros(3)
        for b in active:
            F += valve_dir(b.split(":")[0]) * config.THRUST_N
        flow = len(active) * config.THRUST_N / (config.ISP * config.G0)
        if flow > 0:
            dm = flow * dt
            if dm > self.prop:
                F *= self.prop / dm
                dm = self.prop
            self.prop -= dm
        m = self.mass
        R = sm.quat_to_rot(self.q)
        a = R @ F / m + sm.cw_accel(self.pos, self.vel, self.n)
        self.vel = self.vel + a * dt                   # semi-implicit Euler
        self.pos = self.pos + self.vel * dt
        # rotation: I w' = tau_dist + tau_w - w x (I w + h_w);   wheels: h_w' = -tau_w
        tau_dist = -np.cross(self.com, F)
        tw = np.clip(np.asarray(torque, float), -config.WHEEL_TORQUE_MAX, config.WHEEL_TORQUE_MAX)
        h_new = np.clip(self.h_w - tw * dt, -config.WHEEL_H_MAX, config.WHEEL_H_MAX)
        tw = (self.h_w - h_new) / dt                   # saturated wheels give less torque
        self.h_w = h_new
        w = self.omega
        wdot = (tau_dist + tw - np.cross(w, self.I * w + self.h_w)) / self.I
        self.omega = w + wdot * dt
        self.q = sm.quat_integrate(self.q, self.omega, dt)
        self.t += dt
        self.active = active


# =========================================================================== geometry helpers
def _mesh(name, verts, norms, cols, tris):
    vdata = GeomVertexData(name, GeomVertexFormat.getV3n3c4(), Geom.UHStatic)
    vdata.setNumRows(len(verts))
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    cw = GeomVertexWriter(vdata, "color")
    for p, n, c in zip(verts, norms, cols):
        vw.addData3(*p); nw.addData3(*n); cw.addData4(*c)
    prim = GeomTriangles(Geom.UHStatic)
    for a, b, c in tris:
        prim.addVertices(int(a), int(b), int(c))
    g = Geom(vdata); g.addPrimitive(prim)
    node = GeomNode(name); node.addGeom(g)
    return NodePath(node)


def make_box(name, sx, sy, sz, color):
    h = np.array([sx, sy, sz]) / 2.0
    verts, norms, cols, tris = [], [], [], []
    for axis in range(3):
        for sgn in (1.0, -1.0):
            n = np.zeros(3); n[axis] = sgn
            u = np.zeros(3); u[(axis + 1) % 3] = 1.0
            v = np.cross(n, u)
            c = n * h
            uu = u * h; vv = v * h
            base = len(verts)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                verts.append(c + a * uu + b * vv); norms.append(n); cols.append(color)
            tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return _mesh(name, verts, norms, cols, tris)


def make_sphere(name, radius, color, segs=32, rings=16, rough=0.0, seed=0, color_fn=None):
    rng = np.random.default_rng(seed)
    bumps = [(rng.normal(size=3), rng.uniform(0.3, 0.9)) for _ in range(12)] if rough else []
    verts, norms, cols, tris = [], [], [], []
    for i in range(rings + 1):
        th = math.pi * i / rings
        for j in range(segs + 1):
            ph = 2 * math.pi * j / segs
            d = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
            r = radius
            for bd, bw in bumps:
                r += rough * radius * bw * math.exp(-6.0 * (1.0 - d @ (bd / np.linalg.norm(bd))))
                r -= rough * radius * 0.25 * bw
            verts.append(d * r); norms.append(d)
            cols.append(color_fn(d) if color_fn else color)
    for i in range(rings):
        for j in range(segs):
            a = i * (segs + 1) + j; b = a + segs + 1
            tris += [(a, b, b + 1), (a, b + 1, a + 1)]
    return _mesh(name, verts, norms, cols, tris)


def _lut():
    """Dark-blue -> cyan -> yellow -> red colormap for disparity (near = warm)."""
    x = np.linspace(0, 1, 256)
    r = np.clip(1.5 * x - 0.2, 0, 1); g = np.clip(1.6 - np.abs(3.0 * x - 1.6), 0, 1); b = np.clip(1.0 - 1.6 * x, 0.15, 1)
    lut = (np.stack([r, g, b], 1) * 255).astype(np.uint8)
    lut[0] = 0
    return lut


# =========================================================================== simulator app
class Sim3D(ShowBase):
    TEX_W, TEX_H = 128, 64           # power of two for p3tinydisplay

    def __init__(self, mock=False, seed=1):
        ShowBase.__init__(self)
        self.lock = threading.Lock()
        self.phys = Physics()
        self.reset_id = 0
        self.cmd = None
        self.cmd_rx = -1e9
        self.branch = {v: "A" for v in config.VALVES}
        self.isolated = []
        self.track = None
        self.track_rx = -1e9
        self.track_times = collections.deque(maxlen=60)
        self.depth = None
        self.depth_new = False
        self.procs = None
        self.mock = mock
        self.mock_seed = seed
        self.mock_scn = sm.HandScenario(seed=seed)
        self.mock_t0 = time.monotonic()
        self.mock_seq = 0
        self.enc = None
        self.encounters = []
        self.hits = 0
        self.last_enc_text = "-"
        self.sat_trail = collections.deque(maxlen=900)
        self.ast_trail = collections.deque(maxlen=300)
        self.view = 4                # camera's-eye view by default (matches the real layout)
        self.running = True
        self.ar = {"dodged": False, "ball_home": False, "home_since": None}
        self.auto_resets = 0
        self.rewind = False
        self.rewind_last_seen = 0.0
        self.ast_vis = None; self.ast_vis_t = None; self.ast_raw_t = 0.0
        self.feed_rx = None
        if config.SIM_FEED:
            try:
                fs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                fs.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
                fs.bind(("0.0.0.0", config.SIM_FEED_PORT))
                fs.setblocking(False)
                import live_view
                self.feed_rx = live_view.PreviewReceiver(fs)
            except OSError as exc:
                print("[sim3d] camera feed disabled (port %d: %s)" % (config.SIM_FEED_PORT, exc), flush=True)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", config.SIM_PORT))
        self.sock.setblocking(False)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ctrl_addr = (config.PI_IP, config.CONTROL_PORT)

        self._build_scene()
        self._build_hud()
        for key, fn in (("1", lambda: self.set_view(1)), ("2", lambda: self.set_view(2)),
                        ("3", lambda: self.set_view(3)), ("4", lambda: self.set_view(4)), ("c", self.toggle_feed), ("m", self.toggle_mock), ("s", self.inject_stuck),
                        ("f", self.drain_prop), ("r", self.reset_sim), ("escape", self.quit)):
            self.accept(key, fn)
        self.taskMgr.add(self._render_task, "render")
        self.thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.thread.start()
        print("[sim3d] listening on :%d, nav -> %s:%d%s" % (config.SIM_PORT, config.PI_IP, config.CONTROL_PORT,
                                                           "  [MOCK BALL]" if mock else ""), flush=True)

    # ------------------------------------------------------------------ scene
    def _build_scene(self):
        self.setBackgroundColor(0.0, 0.0, 0.02, 1)
        self.disableMouse()
        self.camLens.setFov(60)
        self.camLens.setNearFar(0.3, 40000)
        sun = DirectionalLight("sun"); sun.setColor(Vec4(1.0, 0.97, 0.9, 1))
        sun_np = self.render.attachNewNode(sun); sun_np.setHpr(-35, -40, 0)
        amb = AmbientLight("amb"); amb.setColor(Vec4(0.32, 0.32, 0.38, 1))
        self.render.setLight(sun_np); self.render.setLight(self.render.attachNewNode(amb))

        # stars
        rng = np.random.default_rng(3)
        vd = GeomVertexData("stars", GeomVertexFormat.getV3c4(), Geom.UHStatic)
        vw = GeomVertexWriter(vd, "vertex"); cw = GeomVertexWriter(vd, "color")
        pts = GeomPoints(Geom.UHStatic)
        for i in range(1500):
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            if d[2] < -0.2:
                d[2] = -d[2]
            vw.addData3(*(d * 20000)); b = rng.uniform(0.4, 1.0); cw.addData4(b, b, b, 1)
            pts.addVertex(i)
        g = Geom(vd); g.addPrimitive(pts); gn = GeomNode("stars"); gn.addGeom(g)
        stars = self.render.attachNewNode(gn); stars.setLightOff(); stars.setRenderModeThickness(2)
        stars.setBin("background", 0); stars.setDepthWrite(False)

        # Earth (below = -x radial = scene -Z), drawn as a big distant sphere
        def earth_col(d):
            n = math.sin(5 * d[0] + 2 * math.sin(3 * d[1])) * math.cos(4 * d[1] + d[2] * 3)
            return (0.15, 0.55, 0.2, 1) if n > 0.35 else (0.08, 0.22, 0.6, 1)
        earth = make_sphere("earth", 9000, None, segs=64, rings=32, color_fn=earth_col)
        earth.reparentTo(self.render); earth.setPos(0, 6000, -10500)

        # satellite (built in scene axes: body LVLH-x up = Z, y = Y, z = -X)
        self.sat = self.render.attachNewNode("sat")
        s = config.SAT_SIZE
        make_box("body", s, s, s, (0.85, 0.68, 0.25, 1)).reparentTo(self.sat)
        for sx in (-1, 1):
            p = make_box("panel", 3.2, 0.06, 1.1, (0.12, 0.18, 0.45, 1)); p.reparentTo(self.sat)
            p.setPos(sx * (s / 2 + 1.75), 0, 0)
        for cx in (-0.15, 0.15):     # the two "cameras" look along-track (+y)
            c = make_box("cam", 0.12, 0.1, 0.12, (0.05, 0.05, 0.05, 1)); c.reparentTo(self.sat)
            c.setPos(cx, s / 2 + 0.05, 0.2)
        self.plumes = {}
        for v in config.VALVES:
            d = M_SCENE @ valve_dir(v)       # acceleration direction (scene); exhaust goes opposite
            dims = np.where(np.abs(d) > 0.5, 1.4, 0.28)
            pl = make_box("plume", dims[0], dims[1], dims[2], (0.65, 0.88, 1.0, 0.75))
            pl.reparentTo(self.sat); pl.setLightOff(); pl.setTransparency(TransparencyAttrib.MAlpha)
            pl.setPos(*(-d * (s / 2 + 0.75)))
            pl.hide()
            self.plumes[v] = pl

        self.asteroid = make_sphere("asteroid", config.ASTEROID_RADIUS, (0.48, 0.42, 0.36, 1),
                                    segs=28, rings=14, rough=0.18, seed=5)
        self.asteroid.reparentTo(self.render); self.asteroid.hide()

        ls = LineSegs(); ls.setThickness(1.5); ls.setColor(0.2, 0.9, 0.4, 1)
        h = 1.0
        for a, b in (((-h, -h, -h), (h, -h, -h)), ((h, -h, -h), (h, h, -h)), ((h, h, -h), (-h, h, -h)),
                     ((-h, h, -h), (-h, -h, -h)), ((-h, -h, h), (h, -h, h)), ((h, -h, h), (h, h, h)),
                     ((h, h, h), (-h, h, h)), ((-h, h, h), (-h, -h, h)), ((-h, -h, -h), (-h, -h, h)),
                     ((h, -h, -h), (h, -h, h)), ((h, h, -h), (h, h, h)), ((-h, h, -h), (-h, h, h))):
            ls.moveTo(*a); ls.drawTo(*b)
        slot = self.render.attachNewNode(ls.create()); slot.setLightOff()
        ls = LineSegs(); ls.setThickness(2); ls.setColor(1.0, 0.85, 0.1, 1)
        for a, b in (((-0.8, 0, 0), (0.8, 0, 0)), ((0, -0.8, 0), (0, 0.8, 0)), ((0, 0, -0.8), (0, 0, 0.8))):
            ls.moveTo(*a); ls.drawTo(*b)
        self.target_np = self.render.attachNewNode(ls.create()); self.target_np.setLightOff()
        # grid on the orbit plane through the slot for depth cues
        ls = LineSegs(); ls.setThickness(1); ls.setColor(0.15, 0.2, 0.3, 1)
        g0 = int(-config.SAT_BOARD_Z * config.SPACE_SCALE) - 10               # behind the real camera
        g1 = int((0.75 - config.SAT_BOARD_Z) * config.SPACE_SCALE) + 10       # beyond the ball's start
        for k in range(-40, 41, 10):
            ls.moveTo(k, g0, -3); ls.drawTo(k, g1, -3)
        for k in range(g0 - g0 % 10, g1, 10):
            ls.moveTo(-40, k, -3); ls.drawTo(40, k, -3)
        grid = self.render.attachNewNode(ls.create()); grid.setLightOff()
        self.cam_y = -config.SAT_BOARD_Z * config.SPACE_SCALE          # real camera position (scene Y)
        cm = make_box("realcam", 2.0, 1.2, 1.4, (0.15, 0.15, 0.18, 1)); cm.reparentTo(self.render)
        cm.setPos(0, self.cam_y, 0)
        lens = make_box("lens", 0.8, 0.3, 0.8, (0.3, 0.6, 1.0, 1)); lens.reparentTo(cm); lens.setPos(0, 0.7, 0)
        ls = LineSegs(); ls.setThickness(1.5); ls.setColor(0.3, 0.6, 1.0, 1)
        ls.moveTo(0, self.cam_y, 0); ls.drawTo(0, (0.75 - config.SAT_BOARD_Z) * config.SPACE_SCALE, 0)  # ball path
        axis = self.render.attachNewNode(ls.create()); axis.setLightOff()
        self.trail_np = None
        self.line_np = None
        self._trail_t = 0.0

    def _build_hud(self):
        try:
            font = self.loader.loadFont("cmtt12.egg")
        except Exception:
            font = None
        kw = dict(fg=(0.85, 0.95, 1, 1), shadow=(0, 0, 0, 1), align=TextNode.ALeft, mayChange=True, scale=0.042)
        if font is not None:
            kw["font"] = font
        self.hud_l = OnscreenText(text="", pos=(0.04, -0.07), parent=self.a2dTopLeft, **kw)
        self.hud_r = OnscreenText(text="", pos=(-1.22, -0.07), parent=self.a2dTopRight, **kw)
        self.hud_b = OnscreenText(text="", pos=(0.04, 0.14), parent=self.a2dBottomLeft, **kw)
        kw3 = dict(kw); kw3.update(fg=(1, 0.25, 0.2, 1), scale=0.06, align=TextNode.ACenter)
        self.banner = OnscreenText(text="", pos=(0, -0.62), parent=self.a2dTopCenter, **kw3)
        kw2 = dict(kw); kw2["fg"] = (0.6, 0.7, 0.8, 1); kw2["scale"] = 0.036
        self.hud_help = OnscreenText(text="1 chase  2 top  3 onboard  4 camera  M mock ball  S stuck valve +z:A  "
                                          "F drain prop  R reset  ESC quit",
                                     pos=(0.04, 0.05), parent=self.a2dBottomLeft, **kw2)
        self.lut = _lut()
        self.dtex = Texture("disparity")
        self.dtex.setup2dTexture(self.TEX_W, self.TEX_H, Texture.T_unsigned_byte, Texture.F_rgb)
        self.dtex.setMagfilter(Texture.FT_nearest); self.dtex.setMinfilter(Texture.FT_nearest)
        self.dtex.setRamImage(bytes(self.TEX_W * self.TEX_H * 3))
        self.dimg = OnscreenImage(image=self.dtex, pos=(-0.36, 0, 0.33), scale=(0.32, 1, 0.24),
                                  parent=self.a2dBottomRight)
        self.dlabel = OnscreenText(text="Pi stereo disparity", pos=(-0.66, 0.6), parent=self.a2dBottomRight,
                                   **dict(kw2, mayChange=False))
        # live camera feed from the Pi (perception's annotated preview: CNN boxes, depth, reason)
        self.FEED_TEX = 512
        self.ftex = Texture("feed")
        self.ftex.setup2dTexture(self.FEED_TEX, self.FEED_TEX, Texture.T_unsigned_byte, Texture.F_rgb)
        self.ftex.setRamImage(bytes(self.FEED_TEX * self.FEED_TEX * 3))
        self.fimg = OnscreenImage(image=self.ftex, pos=(-0.47, 0, 0.55), scale=(0.44, 1, 0.462),
                                  parent=self.a2dBottomRight)
        self.fimg.hide()
        self.show_feed = True
        self.feed_t = None

    # ------------------------------------------------------------------ user actions
    def set_view(self, n):
        self.view = n

    def toggle_feed(self):
        self.show_feed = not self.show_feed

    def _update_feed(self, now):
        """Poll the Pi's camera feed and show it (bottom right) instead of the stereo disparity inset."""
        if self.feed_rx is None:
            return
        got = self.feed_rx.poll()
        if got is not None:
            import live_view
            img, meta = got
            frame = live_view.render(img, meta, 400, False, 0.0)          # BGR, with CNN overlays
            h, w = frame.shape[:2]
            if h > self.FEED_TEX or w > self.FEED_TEX:
                s = min(self.FEED_TEX / float(h), self.FEED_TEX / float(w))
                import cv2
                frame = cv2.resize(frame, (int(w * s), int(h * s)))
                h, w = frame.shape[:2]
            buf = np.zeros((self.FEED_TEX, self.FEED_TEX, 3), np.uint8)
            buf[:h, :w] = frame[::-1]                                     # Panda: bottom row first
            self.ftex.setRamImage(buf.tobytes())
            self.fimg.setTexScale(TextureStage.getDefault(), w / float(self.FEED_TEX), h / float(self.FEED_TEX))
            self.fimg.setScale(0.44, 1, 0.44 * h / float(w))
            self.feed_t = now
        live = self.show_feed and self.feed_t is not None and now - self.feed_t < 3.0
        if live:
            self.fimg.show(); self.dimg.hide(); self.dlabel.hide()
        else:
            self.fimg.hide(); self.dimg.show(); self.dlabel.show()

    def _smooth_asteroid(self, ap, now):
        """Display only: low-pass the drawn asteroid (camera depth noise x100 = visible jitter) and keep it
        through short detection dropouts. Control and collision counting use the raw track."""
        dt = max(now - self.ast_vis_t, 0.0) if self.ast_vis_t is not None else 0.0
        self.ast_vis_t = now
        if ap is not None:
            self.ast_raw_t = now
            if self.ast_vis is None or float(np.linalg.norm(ap - self.ast_vis)) > config.AST_SNAP_M:
                self.ast_vis = np.array(ap, float)
            else:
                a = 1.0 - math.exp(-dt / max(config.AST_SMOOTH_S, 1e-3))
                self.ast_vis = self.ast_vis + a * (ap - self.ast_vis)
        elif self.ast_vis is not None and (self.rewind or now - self.ast_raw_t > config.AST_HOLD_S):
            self.ast_vis = None
        return self.ast_vis

    def toggle_mock(self):
        with self.lock:
            self.mock = not self.mock
            self.mock_scn = sm.HandScenario(seed=self.mock_seed)
            self.mock_t0 = time.monotonic()
        print("[sim3d] mock ball %s" % ("ON" if self.mock else "OFF"), flush=True)

    def inject_stuck(self):
        with self.lock:
            self.phys.stuck = None if self.phys.stuck else "+z:A"
        print("[sim3d] stuck valve %s" % (self.phys.stuck or "cleared"), flush=True)

    def drain_prop(self):
        with self.lock:
            self.phys.prop = min(self.phys.prop, 0.05)
        print("[sim3d] propellant drained to %.3f kg" % self.phys.prop, flush=True)

    def reset_sim(self):
        with self.lock:
            self._reset_locked()
        print("[sim3d] reset (reset_id=%d)" % self.reset_id, flush=True)

    def _reset_locked(self):
        self.phys.reset()
        self.reset_id += 1
        self.isolated = []
        self.branch = {v: "A" for v in config.VALVES}
        self.sat_trail.clear(); self.ast_trail.clear()
        self.ar = {"dodged": False, "ball_home": False, "home_since": None}

    def _rewind_update(self, now):
        """After an auto reset the person brings the ball back out THROUGH the satellite's position:
        hide the asteroid and count no collision until the ball is back at the start (or out of view)."""
        if not self.rewind:
            return
        t = self.track
        seen = t is not None and t.get("detected") and t.get("pos") is not None and now - self.track_rx < 0.5
        if seen:
            self.rewind_last_seen = now
            if t["pos"][2] >= config.REWIND_BALL_Z:
                self.rewind = False
        elif now - self.rewind_last_seen > config.REWIND_UNSEEN_S:
            self.rewind = False
        if not self.rewind:
            print("[sim3d] rewind done: ready for the next pass", flush=True)

    def _auto_reset(self, now):
        """After a dodge: once the ball came close to the camera and the satellite is back home in
        STATION_KEEP for AUTO_RESET_DELAY_S, reset the run (propellant, faults) for the next pass."""
        if not config.AUTO_RESET:
            return
        ar = self.ar
        cmd = self.cmd if (self.cmd and now - self.cmd_rx < config.SIM_CMD_TIMEOUT) else None
        if cmd and cmd.get("phase") in ("EVADE", "HOLD_CLEAR"):
            ar["dodged"] = True
        t = self.track
        if (ar["dodged"] and not ar["ball_home"] and t is not None and t.get("detected") and t.get("pos") is not None
                and now - self.track_rx < 0.5 and t["pos"][2] <= config.AUTO_RESET_BALL_Z):
            ar["ball_home"] = True
            # the pass is over once the ball reached the camera: record it and hide the asteroid, so
            # pulling the ball back out through the satellite (before or after the reset) is no collision
            self._finish_encounter()
            self.rewind, self.rewind_last_seen = True, now
        home = (cmd is not None and cmd.get("phase") == "STATION_KEEP"
                and float(np.linalg.norm(self.phys.pos)) < config.REACHED_POS)
        if ar["dodged"] and ar["ball_home"] and home:
            if ar["home_since"] is None:
                ar["home_since"] = now
            elif now - ar["home_since"] >= config.AUTO_RESET_DELAY_S:
                self._finish_encounter()          # (already reported when the ball reached the camera)
                rewinding = self.rewind
                self._reset_locked()
                self.auto_resets += 1
                if rewinding:                     # keep waiting for the ball to be brought back out
                    self.rewind = True
                print("[sim3d] AUTO RESET #%d: satellite back in orbit, ball at the camera (reset_id=%d)" % (
                    self.auto_resets, self.reset_id), flush=True)
        else:
            ar["home_since"] = None

    def quit(self):
        self.running = False
        self.userExit()

    # ------------------------------------------------------------------ physics thread
    def _physics_loop(self):
        period = 1.0 / config.PHYSICS_HZ
        dt_m = config.TIME_SCALE / config.PHYSICS_HZ
        nav_every = config.PHYSICS_HZ // config.NAV_HZ
        k = 0
        next_t = time.perf_counter()
        while self.running:
            now_p = time.perf_counter()
            if now_p < next_t:
                time.sleep(next_t - now_p)
            now = time.monotonic()
            with self.lock:
                self._poll(now)
                cmd_ok = now - self.cmd_rx < config.SIM_CMD_TIMEOUT
                valves = self.cmd.get("valves", []) if (cmd_ok and self.cmd) else []
                torque = self.cmd.get("torque", [0, 0, 0]) if (cmd_ok and self.cmd) else [0, 0, 0]
                self.phys.step(dt_m, valves, self.branch, self.isolated, torque)
                if k % nav_every == 0:
                    self._send_nav()
                if self.mock and k % 7 == 0:
                    self._send_mock(now)
                self._rewind_update(now)
                self._encounter(now)
                self._auto_reset(now)
            k += 1
            next_t += period
            if time.perf_counter() - next_t > 0.1:      # never try to catch up more than 100 ms
                next_t = time.perf_counter()

    def _poll(self, now):
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            t = msg.get("type")
            if t == "cmd":
                self.cmd = msg; self.cmd_rx = now
                if isinstance(msg.get("branch"), dict):
                    self.branch = dict(msg["branch"])
                if isinstance(msg.get("isolated"), list):
                    self.isolated = list(msg["isolated"])
            elif t == "track" and not self.mock:
                self._on_track(msg, now)
            elif t == "depthmap":
                try:
                    a = np.frombuffer(base64.b64decode(msg["data"]), np.uint8).reshape(msg["h"], msg["w"])
                    self.depth = (a, msg.get("ball")); self.depth_new = True
                except (KeyError, ValueError, TypeError):
                    pass
            elif t == "procs":
                self.procs = msg.get("procs")

    def _on_track(self, msg, now):
        self.track = msg; self.track_rx = now
        self.track_times.append(now)

    def _send_nav(self):
        p = self.phys
        msg = {"type": "nav", "t": p.t, "pos": p.pos.tolist(), "vel": p.vel.tolist(), "q": p.q.tolist(),
               "omega": p.omega.tolist(), "h_w": p.h_w.tolist(), "mass": p.mass, "prop": p.prop,
               "reset_id": self.reset_id}
        try:
            self.tx.sendto(json.dumps(msg).encode(), self.ctrl_addr)
        except OSError:
            pass

    def _send_mock(self, now):
        """Mock ball: synthetic hand-held ball sent as `track` straight to control (no Pi perception)."""
        pos, vel = self.mock_scn.state(now - self.mock_t0)
        det = sm.ball_in_view(pos)
        self.mock_seq += 1
        msg = {"type": "track", "seq": self.mock_seq, "detected": bool(det),
               "reason": "ok" if det else "out_of_view", "pos": pos.tolist() if det else None,
               "vel": vel.tolist() if det else None, "latency_s": 0.0, "classifier": "mock",
               "cnn_ms": 0.0, "stereo_ms": 0.0, "n_candidates": 1 if det else 0}
        if det:
            msg.update({"depth_m": float(pos[2]), "range_m": float(np.linalg.norm(pos)), "conf": 1.0})
        self._on_track(msg, now)
        try:
            self.tx.sendto(json.dumps(msg).encode(), self.ctrl_addr)
        except OSError:
            pass

    def asteroid_pos(self, now):
        if self.rewind:                     # ball being brought back out after a reset: not a threat
            return None
        t = self.track
        if t is None or not t.get("detected") or t.get("pos") is None:
            return None
        if now - self.track_rx > config.TRACK_DRAW_S:
            return None
        return sm.ball_to_space(t["pos"])

    def _encounter(self, now):
        ap = self.asteroid_pos(now)
        if ap is not None:
            d_sat = float(np.linalg.norm(ap - self.phys.pos))
            d_slot = float(np.linalg.norm(ap))
            if self.enc is None:
                self.enc = {"t0": now, "min_sat": d_sat, "min_slot": d_slot, "last": now, "max_disp": 0.0}
            e = self.enc
            e["min_sat"] = min(e["min_sat"], d_sat); e["min_slot"] = min(e["min_slot"], d_slot)
            e["last"] = now
            e["max_disp"] = max(e["max_disp"], float(np.linalg.norm(self.phys.pos)))
        elif self.enc is not None and now - self.enc["last"] > config.ENCOUNTER_END_S:
            self._finish_encounter()

    def _finish_encounter(self):
        e = self.enc
        if e is not None:
            self.enc = None
            hit = e["min_sat"] < config.COLLISION_DIST
            self.hits += int(hit)
            self.encounters.append(e)
            self.last_enc_text = "closest %.1f m (slot %.1f m) %s" % (e["min_sat"], e["min_slot"],
                                                                      "HIT" if hit else "clear")
            print("[sim3d] encounter over: closest approach %.2f m to satellite (asteroid to slot %.2f m, "
                  "max dodge %.1f m) -> %s | encounters %d hits %d" % (
                      e["min_sat"], e["min_slot"], e["max_disp"], "HIT" if hit else "clear",
                      len(self.encounters), self.hits), flush=True)

    # ------------------------------------------------------------------ rendering (main thread)
    def _render_task(self, task):
        now = time.monotonic()
        with self.lock:
            p = self.phys
            pos, vel, q, omega, h_w = p.pos.copy(), p.vel.copy(), p.q.copy(), p.omega.copy(), p.h_w.copy()
            active = set(p.active)
            mass, prop, t, stuck = p.mass, p.prop, p.t, p.stuck
            cmd = dict(self.cmd) if self.cmd else None
            cmd_age = now - self.cmd_rx
            track = dict(self.track) if self.track else None
            track_age = now - self.track_rx
            ap = self.asteroid_pos(now)
            depth = self.depth if self.depth_new else None
            self.depth_new = False
            procs = dict(self.procs) if self.procs else None
            rate = 0.0
            if len(self.track_times) > 2 and now - self.track_times[-1] < 1.0:
                rate = (len(self.track_times) - 1) / max(self.track_times[-1] - self.track_times[0], 1e-3)
            enc_n, hits, last_enc, mock = len(self.encounters), self.hits, self.last_enc_text, self.mock

        # satellite pose
        sp = to_scene(pos)
        self.sat.setPos(sp)
        Rs = M_SCENE @ sm.quat_to_rot(q) @ M_SCENE.T
        m3 = LMatrix3f(*[float(x) for x in Rs.T.flatten()])
        qq = LQuaternionf(); qq.setFromMatrix(m3)
        self.sat.setQuat(qq)
        for v, pl in self.plumes.items():
            on = any(b.startswith(v + ":") for b in active)
            pl.show() if on else pl.hide()
        if cmd:
            self.target_np.setPos(to_scene(cmd.get("target", [0, 0, 0])))
        ap = self._smooth_asteroid(ap, now)        # display: smoothed + held through dropouts
        if ap is not None:
            self.asteroid.show(); self.asteroid.setPos(to_scene(ap))
            self.asteroid.setHpr(now * 20, now * 7, 0)
        else:
            self.asteroid.hide()
        self._update_feed(now)

        if now - self._trail_t > 0.1:
            self._trail_t = now
            self.sat_trail.append(sp)
            if ap is not None:
                self.ast_trail.append(to_scene(ap))
            elif self.enc is None:
                self.ast_trail.clear()
            if self.trail_np is not None:
                self.trail_np.removeNode()
            ls = LineSegs(); ls.setThickness(2)
            for trail, col in ((self.sat_trail, (0.3, 1.0, 0.5, 1)), (self.ast_trail, (1.0, 0.45, 0.2, 1))):
                if len(trail) > 1:
                    ls.setColor(*col); ls.moveTo(trail[0])
                    for tp in list(trail)[1:]:
                        ls.drawTo(tp)
            self.trail_np = self.render.attachNewNode(ls.create()); self.trail_np.setLightOff()
        if self.line_np is not None:
            self.line_np.removeNode(); self.line_np = None
        threat = cmd.get("threat") if cmd else None
        if ap is not None:
            ls = LineSegs(); ls.setThickness(1.5)
            ls.setColor(1, 0.2, 0.2, 1) if (threat and threat.get("trigger")) else ls.setColor(0.6, 0.6, 0.6, 1)
            ls.moveTo(sp); ls.drawTo(to_scene(ap))
            self.line_np = self.render.attachNewNode(ls.create()); self.line_np.setLightOff()

        # camera
        cam = self.camera
        far = (0.75 - config.SAT_BOARD_Z) * config.SPACE_SCALE                  # ball's start (scene Y)
        if self.view == 1:
            cam.setPos(sp + Vec3(12, -40, 14)); cam.lookAt(sp + Vec3(0, 30, 0))
        elif self.view == 2:
            # top view of the dodge zone around the satellite (the ball's last ~15 cm before it and after)
            cam.setPos(Vec3(sp[0] * 0.5, 8.0, 85.0)); cam.lookAt(Vec3(sp[0] * 0.5, 9.0, 0), Vec3(0, 1, 0))
        elif self.view == 3:
            cam.setPos(sp + qq.xform(Vec3(0, config.SAT_SIZE / 2 + 0.2, 0.2))); cam.setQuat(qq)
        else:   # 4: from the real camera's position, looking out past the satellite toward the ball's start
            # the real camera is far behind (SAT_BOARD_Z * SPACE_SCALE): look from part way, along its axis
            cam.setPos(Vec3(2.5, max(self.cam_y * 0.3, -28.0), 4.0)); cam.lookAt(Vec3(0, 0.35 * far, 0))

        # disparity texture
        if depth is not None:
            a, ball = depth
            ys = (np.arange(self.TEX_H) * a.shape[0] / self.TEX_H).astype(int)
            xs = (np.arange(self.TEX_W) * a.shape[1] / self.TEX_W).astype(int)
            rgb = self.lut[a[ys[:, None], xs[None, :]]]
            if ball is not None:
                bu = int(ball[0] * self.TEX_W / a.shape[1]); bv = int(ball[1] * self.TEX_H / a.shape[0])
                rgb[max(bv - 1, 0):bv + 2, max(bu - 4, 0):bu + 5] = (255, 255, 255)
                rgb[max(bv - 4, 0):bv + 5, max(bu - 1, 0):bu + 2] = (255, 255, 255)
            self.dtex.setRamImage(np.ascontiguousarray(rgb[::-1, :, ::-1]).tobytes())   # Panda: bottom-up, BGR

        # HUD
        f3 = lambda v: "[%6.2f %6.2f %6.2f]" % tuple(v)
        L = ["DODGE-1   mission t %7.1f s   x%g time   view %s%s" % (
            t, config.TIME_SCALE, {1: "chase", 2: "top", 3: "onboard", 4: "camera's eye"}[self.view], "   [MOCK BALL]" if mock else "")]
        if cmd is None:
            self.banner.setText("NO FLIGHT COMPUTER - satellite cannot act\n"
                                "start  python control.py  (or supervisor.py)  -> commands to %s:%d"
                                % (config.PI_IP, config.CONTROL_PORT))
        elif cmd_age > config.SIM_CMD_TIMEOUT:
            self.banner.setText("CONTROL LINK LOST (%.1f s) - watchdog closed all valves" % cmd_age)
        else:
            self.banner.setText("")
        if cmd is None or cmd_age > config.SIM_CMD_TIMEOUT:
            L.append("CONTROL LINK LOST - watchdog closed all valves")
        if cmd:
            L.append("PHASE %-12s motion %-6s valves %s" % (cmd.get("phase"), cmd.get("motion"),
                                                             " ".join(cmd.get("valves") or []) or "-"))
            L.append("target %s m" % f3(cmd.get("target", [0, 0, 0])))
        L.append("pos    %s m   |r| %.2f m" % (f3(pos), float(np.linalg.norm(pos))))
        L.append("vel    %s m/s" % f3(vel))
        L.append("att err %.2f deg   wheel h %s Nms" % (cmd.get("att_err_deg", 0.0) if cmd else 0.0, f3(h_w)))
        L.append("mass %.2f kg   propellant %.3f kg" % (mass, prop))
        if threat:
            L.append("THREAT range %.1f m  t_cpa %.1f s  miss %.1f m %s" % (
                threat["range"], threat["t_cpa"], threat["miss"], " TRIGGER" if threat["trigger"] else ""))
        else:
            L.append("THREAT none")
        if cmd:
            br = " ".join("%s:%s" % (k, v) for k, v in sorted((cmd.get("branch") or {}).items()) if v != "A")
            L.append("FDIR %s   isolated %s   backup %s" % (cmd.get("fault") or "nominal",
                                                            ",".join(cmd.get("isolated") or []) or "-", br or "-"))
            lp = cmd.get("loop") or {}
            L.append("Pi loop work %.2f ms  jitter max %.2f ms  overruns %d" % (
                lp.get("work_ms", 0), lp.get("jitter_max_ms", 0), lp.get("overruns", 0)))
        if stuck:
            L.append("INJECTED FAULT: valve %s stuck open" % stuck)
        if self.rewind:
            L.append("RESET DONE - bring the ball back out (>= %d cm) for the next pass" % int(config.REWIND_BALL_Z * 100))
        self.hud_l.setText("\n".join(L))

        R = ["PERCEPTION (Pi)  %.1f Hz" % rate]
        if track and track_age < 1.0 and track.get("detected") and track.get("pos") is not None:
            R.append("BALL %.2f m from camera (sat %.2f, dodge <%.2f)" % (
                track["pos"][2], config.SAT_BOARD_Z, config.SAT_BOARD_Z + config.TRIGGER_RANGE / config.SPACE_SCALE))
        else:
            R.append("BALL not detected")
        if track and track_age < 1.0:
            R.append("classifier %s  detected %s" % (track.get("classifier"), track.get("detected")))
            R.append("reason %s" % track.get("reason"))
            if track.get("detected"):
                R.append("depth %.3f m  conf %.2f" % (track.get("depth_m", 0), track.get("conf", 0)))
            R.append("cnn %.1f ms  stereo %.1f ms  cand %d" % (track.get("cnn_ms", 0), track.get("stereo_ms", 0),
                                                              track.get("n_candidates", 0)))
        else:
            R.append("no track data")
        R.append("")
        R.append("PROCESSES (supervisor)")
        if procs:
            for n, s in sorted(procs.items()):
                R.append("%-10s pid %-6s p%-3s %s  hb %.2fs  restarts %d" % (
                    n, s.get("pid"), s.get("prio"), "UP  " if s.get("alive") else "DOWN", s.get("hb_age", 0),
                    s.get("restarts", 0)))
                if s.get("last_event") and s.get("last_event") != "init":
                    R.append("    last: %s" % s["last_event"])
        else:
            R.append("no supervisor status")
        self.hud_r.setText("\n".join(R))
        self.hud_b.setText("encounters %d   hits %d   auto resets %d   last: %s" % (enc_n, hits, self.auto_resets, last_enc))
        return task.cont


def main():
    ap = argparse.ArgumentParser(description="DODGE-1 3D simulator (laptop)")
    ap.add_argument("--mock", action="store_true", help="start with the mock ball (no Pi perception)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    app = Sim3D(mock=args.mock, seed=args.seed)
    try:
        app.run()
    finally:
        app.running = False


if __name__ == "__main__":
    sys.exit(main())
