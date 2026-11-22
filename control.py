"""DODGE-1 control (runs on the Raspberry Pi 5 / QNX, HIGH priority, core 0).

Fixed 50 Hz loop: nav + track in -> threat assessment -> phase machine -> bang-coast-bang
translation guidance -> valve selection (interlocked) -> attitude PD on the reaction wheels
-> FDIR (stuck valve isolation, backup branch) -> `cmd` to the sim.

"The AI suggests, the QNX control process decides": perception only sends `track` messages;
this process validates them (freshness) and owns every actuator decision and every safe state.
numpy + stdlib only.
"""
import collections
import json
import math
import os
import socket
import sys
import time

import numpy as np

import config
import spacemath as sm

AXES = "xyz"
PERIOD = 1.0 / config.CONTROL_HZ


def valve_dir(v):
    d = np.zeros(3)
    d[AXES.index(v[1])] = 1.0 if v[0] == "+" else -1.0
    return d


VALVE_DIRS = {v: valve_dir(v) for v in config.VALVES}


def opposite(v):
    return ("-" if v[0] == "+" else "+") + v[1]


class Controller:
    def __init__(self):
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind(("0.0.0.0", config.CONTROL_PORT))
        self.rx.setblocking(False)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sim_addr = (config.SIM_IP, config.SIM_PORT)
        self.sup_addr = ("127.0.0.1", config.SUPERVISOR_PORT)
        self.nav = None
        self.nav_rx = -1e9
        self.reset_id = None
        self.track = None
        self.track_rx = -1e9
        self.last_seen = -1e9
        self.seq = 0
        # loop statistics
        self.overruns = 0
        self.jitters = collections.deque(maxlen=config.CONTROL_HZ * 2)
        self.work_ms = 0.0
        self.reset_state()

    # ------------------------------------------------------------------ state
    def reset_state(self):
        self.phase = "STATION_KEEP"
        self.target = np.zeros(3)
        self.dodge_axis = None
        self.v_limit = config.V_MAX
        self.branch = {v: "A" for v in config.VALVES}
        self.isolated = set()
        self.fault = None
        self.fdir_since = None
        self.fdir_quiet_until = 0.0
        self.fdir_last_blame = {}           # axis -> valve blamed last time
        self.nav_hist = collections.deque(maxlen=200)   # (t_mission, vel, rx_time)
        self.cmd_hist = collections.deque(maxlen=200)   # (rx_time, accel_cmd_lvlh, valves)

    # ------------------------------------------------------------------ io
    def poll(self, now):
        while True:
            try:
                data, _ = self.rx.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            try:
                msg = json.loads(data)
                kind = msg.get("type")
            except (ValueError, AttributeError):
                continue
            if kind == "nav":
                try:
                    nav = {
                        "t": float(msg["t"]), "pos": np.array(msg["pos"], float), "vel": np.array(msg["vel"], float),
                        "q": sm.quat_normalize(msg["q"]), "omega": np.array(msg["omega"], float),
                        "mass": float(msg["mass"]), "prop": float(msg["prop"]), "reset_id": msg.get("reset_id"),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
                if not all(np.all(np.isfinite(nav[k])) for k in ("pos", "vel", "q", "omega")):
                    continue
                if self.reset_id is not None and nav["reset_id"] != self.reset_id:
                    print("[control] sim reset %s -> %s" % (self.reset_id, nav["reset_id"]), flush=True)
                    self.reset_state()
                self.reset_id = nav["reset_id"]
                self.nav = nav
                self.nav_rx = now
                self.nav_hist.append((nav["t"], nav["vel"].copy(), now))
            elif kind == "track":
                self.track = msg
                self.track_rx = now
                if msg.get("detected") and msg.get("pos") is not None:
                    self.last_seen = now

    def send(self, msg, addr):
        try:
            self.tx.sendto(json.dumps(msg, default=float).encode(), addr)
        except OSError:
            pass

    # ------------------------------------------------------------------ helpers
    def valve_ok(self, v):
        return any((v + ":" + b) not in self.isolated for b in config.BRANCHES)

    def axis_ok(self, k):
        return self.valve_ok("+" + AXES[k]) and self.valve_ok("-" + AXES[k])

    def asteroid(self, now):
        """Validated asteroid state (LVLH) or None. Stale or malformed tracks are ignored."""
        t = self.track
        if t is None or not t.get("detected") or t.get("pos") is None or t.get("vel") is None:
            return None
        try:
            lat = float(t.get("latency_s") or 0.0)
            age = (now - self.track_rx) + max(0.0, min(lat, 0.5))
            p = np.array(t["pos"], float); v = np.array(t["vel"], float)
        except (TypeError, ValueError):
            return None
        if age > config.TRACK_VALID_S or p.shape != (3,) or v.shape != (3,):
            return None
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v))):
            return None
        ap = sm.ball_to_space(p); av = sm.ball_vel_to_space(v)
        return {"pos": ap + av * age * config.TIME_SCALE, "vel": av, "ball_range": float(np.linalg.norm(p))}

    def assess(self, ast, pos, vel):
        r = ast["pos"] - pos
        v = ast["vel"] - vel
        rng = float(np.linalg.norm(r))
        vv = float(v @ v)
        closing = float(r @ v) < 0.0
        t_cpa = max(0.0, -float(r @ v) / vv) if vv > 1e-9 else 0.0
        m = r + v * t_cpa
        miss = float(np.linalg.norm(m))
        approaching = closing and math.sqrt(vv) > 0.1
        trigger = rng <= config.TRIGGER_RANGE or (
            approaching and t_cpa <= config.MISS_HORIZON and miss < config.MISS_TRIGGER)
        if config.TRIGGER_FRONT_ONLY:
            # a threat comes from the front (along-track +y) toward the satellite: the ball moving toward
            # the camera. Taking the ball back out after a pass must not trigger a dodge.
            trigger = trigger and approaching and v[1] < -0.1 and r[1] > -config.COLLISION_DIST
        return {"range": rng, "t_cpa": t_cpa, "miss": miss, "trigger": bool(trigger),
                "approaching": bool(approaching), "miss_vec": m, "rel_vel": v}

    def plan_dodge(self, thr, ast):
        """Dodge perpendicular to the approach line: cross-track (z) first, then radial (x).
        Never along-track. Needs both directions of the axis (go AND brake).
        Returns (target LVLH, axis, speed limit) or (None, None, None).

        DODGE_MODE "fixed":    always DODGE_DIST at V_MAX.
        DODGE_MODE "min_fuel": fuel ~ total delta-v = 2 x speed (burn + brake), so move only as far as
        needed for a SAFE_MISS closest approach (+ margin) and only as fast as needed to get there
        before the asteroid does (t_cpa)."""
        u = ast["vel"]
        if np.linalg.norm(u) < 0.2:
            u = -(ast["pos"] - self.nav["pos"])
        u = u / max(np.linalg.norm(u), 1e-9)
        for k in (2, 0):
            if not self.axis_ok(k):
                continue
            if abs(u[k]) > 0.8:
                continue
            # Robust (min-max) dodge. Where will the asteroid cross the satellite's plane on this axis?
            # Two estimates: its CURRENT offset and the PREDICTED miss (extrapolated with a noisy velocity,
            # and a hand path may drift sideways); each may also be off by NOISE_SIGMAS of position
            # noise (SPACE_SCALE times the ball's). For each side, compute the move that clears the worst
            # of these, then take the side that needs the smaller move.
            r_now = ast["pos"] - self.nav["pos"]
            m_pred = thr["miss_vec"]
            unc = config.NOISE_SIGMAS * config.TRACK_POS_SIGMA_M * config.SPACE_SCALE
            cands = (r_now[k], m_pred[k])
            # clearance still needed on this axis given the (smallest) offset on the other axes
            m_other = np.where(np.abs(r_now) < np.abs(m_pred), r_now, m_pred)
            m_other = np.sign(m_other) * np.maximum(np.abs(m_other) - unc, 0.0)
            other2 = max(0.0, float(m_other @ m_other) - m_other[k] * m_other[k])
            need = math.sqrt(max(0.0, config.SAFE_MISS ** 2 - other2))
            # moving s*d: clearance to an asteroid at lateral a is s*d - a  ->  d >= s*a + unc + need
            d_side = {sg: max(sg * a for a in cands) + unc + need for sg in (1.0, -1.0)}
            s = min(d_side, key=lambda sg: (d_side[sg], -sg * -r_now[k]))
            tgt = np.zeros(3)
            if config.DODGE_MODE != "min_fuel":
                tgt[k] = s * config.DODGE_DIST
                return tgt, k, config.V_MAX
            d = min(max(d_side[s] + config.DODGE_MARGIN, config.DODGE_MIN), config.DODGE_DIST)
            tgt[k] = max(-config.DODGE_DIST, min(config.DODGE_DIST, self.nav["pos"][k] + s * d))
            T = thr["t_cpa"] if thr["approaching"] else 0.0
            if T > 0.0:
                v = d / max(config.DODGE_TIME_FRAC * T, 0.5)   # arrive with time to spare
            else:
                v = config.V_RETURN                           # no deadline: slow and cheap
            v = min(max(v, config.V_DODGE_MIN), config.V_MAX)
            return tgt, k, v
        return None, None, None

    def guidance(self, pos, vel, mass, v_max=None):
        """Bang-coast-bang per LVLH axis. Returns desired accel direction (-1/0/+1 per axis), motion."""
        a = config.THRUST_N / mass
        v_lim = config.V_MAX if v_max is None else v_max
        # one control tick of thrust changes the speed by dv_tick (mission time per tick grows with
        # TIME_SCALE); deadbands smaller than that make bang-bang chatter and burn the tank
        dv_tick = a * config.TIME_SCALE / config.CONTROL_HZ
        # (must stay below REACHED_VEL, or RETURN never completes)
        vel_db = min(max(config.VEL_DEADBAND, 1.2 * dv_tick), 0.8 * config.REACHED_VEL)
        pos_db = max(config.POS_DEADBAND, 2.0 * dv_tick * config.TIME_SCALE / config.CONTROL_HZ)
        d = np.zeros(3)
        modes = []
        for k in range(3):
            e = self.target[k] - pos[k]
            v = vel[k]
            if abs(e) < pos_db:
                if abs(v) > vel_db:
                    d[k] = -math.copysign(1.0, v); modes.append("BRAKE")
                else:
                    modes.append("HOLD")
                continue
            sgn = math.copysign(1.0, e)
            v_toward = v * sgn
            stop = v * v / (2 * a) + abs(v) * config.LATENCY_MARGIN_S
            if v_toward > 0 and stop >= abs(e):
                d[k] = -sgn; modes.append("BRAKE")
            elif v_toward < v_lim:
                d[k] = sgn; modes.append("BURN")
            elif v_toward > v_lim + max(0.1 * v_lim, 1.5 * dv_tick):
                d[k] = -sgn; modes.append("BRAKE")
            else:
                modes.append("COAST")
        for m in ("BRAKE", "BURN", "COAST"):
            if m in modes:
                return d, m
        return d, "HOLD"

    def select_valves(self, d_lvlh, R):
        d_body = R.T @ d_lvlh
        valves = []
        for k in range(3):
            if d_body[k] > 0.5:
                v = "+" + AXES[k]
            elif d_body[k] < -0.5:
                v = "-" + AXES[k]
            else:
                continue
            if self.valve_ok(v):
                valves.append(v)
        # interlock: opposing thrusters never fire together
        return [v for v in valves if opposite(v) not in valves]

    def fdir(self, now, R, mass, valves):
        """Compare measured accel (nav velocity over FDIR_WINDOW mission s) with commanded."""
        a_one = config.THRUST_N / mass
        a_cmd_now = R @ sum((VALVE_DIRS[v] for v in valves), np.zeros(3)) * a_one
        self.cmd_hist.append((now, a_cmd_now, tuple(valves)))
        if now < self.fdir_quiet_until or len(self.nav_hist) < 5:
            self.fdir_since = None
            return
        t1, v1, rx1 = self.nav_hist[-1]
        old = None
        for t0, v0, rx0 in reversed(self.nav_hist):
            if t1 - t0 >= config.FDIR_WINDOW:
                old = (t0, v0, rx0)
                break
        if old is None:
            return
        t0, v0, rx0 = old
        # Commands are timed in real time, the measured velocity in mission time. If the sim's mission
        # clock did not advance steadily over the window (sim hiccup, Wi-Fi burst), the two windows do
        # not line up and the residual is meaningless: skip rather than blame a healthy thruster.
        rate = (t1 - t0) / max(rx1 - rx0, 1e-6)
        if not (0.8 * config.TIME_SCALE <= rate <= 1.25 * config.TIME_SCALE):
            self.fdir_since = None
            return
        a_meas = (v1 - v0) / (t1 - t0) - sm.cw_accel(self.nav["pos"], self.nav["vel"])
        win = [c for c in self.cmd_hist if rx0 <= c[0] < rx1]
        if len(win) < 3:
            return
        a_cmd = sum((c[1] for c in win), np.zeros(3)) / len(win)
        res = a_meas - a_cmd
        if np.linalg.norm(res) <= config.FDIR_RESIDUAL_FRAC * a_one:
            self.fdir_since = None
            return
        if self.fdir_since is None:
            self.fdir_since = now
            return
        if now - self.fdir_since < config.FDIR_PERSIST_S:
            return
        # --- diagnose: residual in body axes
        rb = R.T @ res
        k = int(np.argmax(np.abs(rb)))
        sgn = "+" if rb[k] > 0 else "-"
        leak = sgn + AXES[k]                  # uncommanded thrust along the residual
        dead = opposite(leak)                 # commanded thrust that did not happen
        dead_frac = sum(1 for c in win if dead in c[2]) / float(len(win))
        # The two explanations look the same while the opposite valve is firing. Blame the leak
        # first; if the same axis trips again after that, blame the commanded valve instead.
        if dead_frac < 0.3 or self.fdir_last_blame.get(k) != leak:
            suspect, why = leak, "uncommanded thrust"
        else:
            suspect, why = dead, "no thrust when commanded"
        self.fdir_last_blame[k] = suspect
        b = self.branch[suspect]
        if b == "A" and (suspect + ":B") not in self.isolated:
            self.isolated.add(suspect + ":A")
            self.branch[suspect] = "B"
            self.fault = "ISOLATED %s:A (%s) -> backup B" % (suspect, why)
        else:
            # never isolate the last branch of a valve (keeps go AND brake on the axis)
            self.fault = "FDIR %s:%s suspect (%s), no backup left" % (suspect, b, why)
        print("[control] FDIR: %s residual=%s" % (self.fault, np.round(res, 3).tolist()), flush=True)
        self.fdir_since = None
        self.fdir_quiet_until = now + config.FDIR_WINDOW / config.TIME_SCALE + 0.4
        self.nav_hist.clear()

    # ------------------------------------------------------------------ one control cycle
    def step(self, now):
        self.poll(now)
        self.seq += 1
        nav_ok = self.nav is not None and now - self.nav_rx < config.NAV_TIMEOUT
        cmd = {"type": "cmd", "seq": self.seq, "phase": self.phase, "motion": "HOLD",
               "target": self.target.tolist(), "valves": [], "torque": [0.0, 0.0, 0.0],
               "isolated": sorted(self.isolated), "branch": dict(self.branch), "fault": self.fault,
               "threat": None, "track": None, "att_err_deg": 0.0,
               "loop": {"work_ms": self.work_ms, "jitter_max_ms": max(self.jitters) if self.jitters else 0.0,
                        "overruns": self.overruns}}
        if not nav_ok:
            # FAIL-SAFE: no navigation -> all valves closed, no torque
            if self.phase != "NO_NAV_SAFE" and self.nav is not None:
                print("[control] no nav for %.2f s -> NO_NAV_SAFE" % (now - self.nav_rx), flush=True)
            if self.nav is not None:
                self.phase = "NO_NAV_SAFE"
            cmd["phase"] = self.phase
            return cmd
        nav = self.nav
        pos, vel, mass = nav["pos"], nav["vel"], nav["mass"]
        R = sm.quat_to_rot(nav["q"])
        if self.phase == "NO_NAV_SAFE":
            # nav is back: head for the slot; a live threat re-triggers EVADE below
            self.target = np.zeros(3)
            self.dodge_axis = None
            self.phase = "RETURN" if np.linalg.norm(pos) > config.REACHED_POS else "STATION_KEEP"
            print("[control] nav back -> %s" % self.phase, flush=True)

        ast = self.asteroid(now)
        thr = self.assess(ast, pos, vel) if ast is not None else None
        if self.track is not None:
            cmd["track"] = {"detected": ast is not None, "age": now - self.track_rx,
                            "pos": ast["pos"].tolist() if ast is not None else None,
                            "reason": self.track.get("reason")}

        # ------------------------------------------------ phase machine
        trigger = thr is not None and thr["trigger"]
        if (trigger and config.DODGE_MODE == "min_fuel" and not thr["approaching"]
                and thr["range"] >= config.SAFE_MISS + config.DODGE_MARGIN):
            trigger = False          # close but not approaching and already at a safe distance: no fuel spent
        if self.phase in ("STATION_KEEP", "RETURN"):
            if trigger:
                tgt, k, v = self.plan_dodge(thr, ast)
                if tgt is not None:
                    self.target, self.dodge_axis, self.phase, self.v_limit = tgt, k, "EVADE", v
                    print("[control] EVADE range=%.1f t_cpa=%.1f miss=%.1f -> target %s at %.2f m/s" % (
                        thr["range"], thr["t_cpa"], thr["miss"], np.round(tgt, 2).tolist(), v), flush=True)
                else:
                    self.fault = "NO_DODGE_AXIS"
            elif self.phase == "RETURN" and self._reached(pos, vel):
                self.phase = "STATION_KEEP"
                self.dodge_axis = None
                print("[control] back in slot -> STATION_KEEP", flush=True)
        elif self.phase in ("EVADE", "HOLD_CLEAR"):
            unseen = now - self.last_seen > config.RETURN_UNSEEN_S
            far_away = False
            if ast is not None:
                p_slot = ast["pos"]
                radial = float(p_slot @ ast["vel"]) / max(np.linalg.norm(p_slot), 1e-6)
                # passed the satellite (or stopped/receding) and clear of it -> fly home
                far_away = (np.linalg.norm(p_slot) > config.CLEAR_RANGE
                            and radial > -config.APPROACH_SPEED_MIN)
            if unseen or far_away:
                self.target = np.zeros(3)
                self.phase = "RETURN"
                self.v_limit = config.V_RETURN if config.DODGE_MODE == "min_fuel" else config.V_MAX
                print("[control] threat gone (%s) -> RETURN" % ("unseen" if unseen else "receding"), flush=True)
            elif self.phase == "EVADE" and self._reached(pos, vel):
                self.phase = "HOLD_CLEAR"
            elif (self.phase == "HOLD_CLEAR" and thr is not None and thr["approaching"]
                  and thr["t_cpa"] <= config.MISS_HORIZON and thr["miss"] < config.MISS_TRIGGER):
                tgt, k, v = self.plan_dodge(thr, ast)
                if tgt is not None and np.linalg.norm(tgt - self.target) > 1.0:
                    self.target, self.dodge_axis, self.phase, self.v_limit = tgt, k, "EVADE", v
                    print("[control] re-plan EVADE -> %s at %.2f m/s" % (np.round(tgt, 2).tolist(), v), flush=True)

        # ------------------------------------------------ translation
        valves = []
        motion = "HOLD"
        if nav["prop"] <= config.PROP_MIN:
            self.fault = "PROP_DEPLETED"
        else:
            v_lim = self.v_limit
            if self.phase in ("STATION_KEEP", "RETURN") and config.DODGE_MODE == "min_fuel":
                v_lim = config.V_RETURN
            d, motion = self.guidance(pos, vel, mass, v_lim)
            valves = self.select_valves(d, R)
            self.fdir(now, R, mass, valves)
            valves = [v for v in valves if self.valve_ok(v)]

        # ------------------------------------------------ attitude (hold body = LVLH)
        th = sm.quat_error_vec(nav["q"])
        tq = -config.ATT_KP * th - config.ATT_KD * nav["omega"]
        tq = np.clip(tq, -config.WHEEL_TORQUE_MAX, config.WHEEL_TORQUE_MAX)

        cmd.update({
            "phase": self.phase, "motion": motion, "target": self.target.tolist(), "valves": valves,
            "torque": tq.tolist(), "isolated": sorted(self.isolated), "branch": dict(self.branch),
            "fault": self.fault, "att_err_deg": math.degrees(float(np.linalg.norm(th))),
        })
        if thr is not None:
            cmd["threat"] = {"range": thr["range"], "t_cpa": thr["t_cpa"], "miss": thr["miss"],
                             "trigger": thr["trigger"], "approaching": thr["approaching"]}
        return cmd

    def _reached(self, pos, vel):
        return (np.linalg.norm(pos - self.target) < config.REACHED_POS
                and np.linalg.norm(vel) < config.REACHED_VEL)

    # ------------------------------------------------------------------ fixed-rate loop
    def run(self):
        print("[control] 50 Hz loop pid=%d sim=%s:%d" % (os.getpid(), config.SIM_IP, config.SIM_PORT), flush=True)
        hb_every = max(1, config.CONTROL_HZ // config.HB_HZ)
        next_t = time.perf_counter() + PERIOD
        while True:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            wake = time.perf_counter()
            self.jitters.append((wake - next_t) * 1000.0)
            cmd = self.step(time.monotonic())
            self.send(cmd, self.sim_addr)
            if self.seq % hb_every == 0:
                self.send({"type": "hb", "name": "control"}, self.sup_addr)
            done = time.perf_counter()
            self.work_ms = (done - wake) * 1000.0
            next_t += PERIOD                    # absolute schedule: no drift
            if done > next_t:                   # missed the next deadline
                self.overruns += 1
                next_t = done + PERIOD


def main():
    Controller().run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
