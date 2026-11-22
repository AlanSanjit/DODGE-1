"""DODGE-1 supervisor (runs on the Raspberry Pi 5 / QNX).

Starts perception and control as SEPARATE processes (no shared memory, UDP only), with QNX
priorities and core pinning via `on` when available:
    control     on -p 40 -C 0          (high priority, core 0)
    perception  on -p 15 -C 1 -C 2 -C 3 (low priority, cores 1-3)
Restarts a process on CRASH (exited) or HANG (no heartbeat for 0.5 s control / 1.5 s perception,
after a startup grace of 5 s / 30 s). Hung processes are killed with SIGKILL (works on stopped
processes). Status goes to the sim at 2 Hz from a separate UNBOUND socket.

    python3 supervisor.py --source synthetic
"""
import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

import config

HERE = os.path.dirname(os.path.abspath(__file__))


class Proc:
    def __init__(self, name, script, args, prio, cpus, hb_timeout, grace, exe=None):
        self.name, self.script, self.args, self.exe = name, script, args, exe
        self.prio, self.cpus = prio, cpus
        self.hb_timeout, self.grace = hb_timeout, grace
        self.p = None
        self.started = 0.0
        self.last_hb = None
        self.restarts = 0
        self.last_event = "init"
        self.quick_fails = 0        # consecutive crashes within 3 s of starting
        self.restart_at = None      # back-off: when to start again

    def cmdline(self):
        if self.exe:                # native program (stereo_capture)
            cmd = [self.exe] + self.args
        else:
            cmd = [sys.executable, "-u", os.path.join(HERE, self.script)] + self.args
        if shutil.which("on"):      # QNX: priority (+ optional pinning)
            pre = ["on", "-p", str(self.prio)]
            # QNX 8.0: `on -C` selects a CPU *cluster* (the Pi 5 has one: 0), not a core, and accepts a
            # single value - "-C 1 -C 2 -C 3" fails with "Invalid cluster runmask". So pinning is off by
            # default; priority is what protects the control loop.
            if config.PIN_CPUS and self.cpus:
                pre += ["-C", str(self.cpus[0])]
            cmd = pre + cmd
        return cmd

    def start(self):
        self.p = subprocess.Popen(self.cmdline(), cwd=HERE)
        self.started = time.monotonic()
        self.last_hb = None
        print("[supervisor] started %s pid=%d prio=%d" % (self.name, self.p.pid, self.prio), flush=True)

    def kill(self):
        if self.p is None:
            return
        try:
            if hasattr(signal, "SIGKILL"):
                os.kill(self.p.pid, signal.SIGKILL)
            else:
                self.p.kill()
        except OSError:
            pass
        try:
            self.p.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass

    def hb_age(self, now):
        ref = self.last_hb if self.last_hb is not None else self.started
        return now - ref

    def check(self, now):
        """Returns 'CRASH', 'HANG' or None."""
        if self.p.poll() is not None:
            return "CRASH"
        if self.hb_timeout is None:   # no heartbeat protocol (native capture): crash detection only
            return None
        in_grace = self.last_hb is None and now - self.started < self.grace
        if not in_grace and self.hb_age(now) > self.hb_timeout:
            return "HANG"
        return None


def main():
    ap = argparse.ArgumentParser(description="DODGE-1 supervisor (Pi)")
    ap.add_argument("--source", choices=["synthetic", "qnx"], default="synthetic")
    args, extra = ap.parse_known_args()

    procs = {
        "control": Proc("control", "control.py", [], config.CONTROL_PRIO, [0],
                        config.CONTROL_HB_TIMEOUT, config.CONTROL_GRACE_S),
        "perception": Proc("perception", "perception.py", ["--source", args.source] + extra,
                           config.PERCEPTION_PRIO, [1, 2, 3],
                           config.PERCEPTION_HB_TIMEOUT, config.PERCEPTION_GRACE_S),
    }
    use_capture = args.source == "qnx" and not ("--qnx-backend" in extra and "ctypes" in extra)
    if use_capture:
        exe = None
        for cand in (os.path.join(HERE, "stereo_capture"), os.path.join(HERE, "qnx", "stereo_capture")):
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                exe = cand
                break
        if exe is None:
            print("[supervisor] WARNING: --source qnx but no compiled stereo_capture found "
                  "(clang -O2 -o stereo_capture qnx/stereo_capture.c -lcamapi)", flush=True)
        else:
            cap_args = ["-n", "1", "-u", str(config.MONO_UNIT)] if config.CAMERA_MODE == "mono" else []
            procs["capture"] = Proc("capture", None, cap_args, config.CAPTURE_PRIO, [1, 2, 3], None, 0.0, exe=exe)
            procs = {k: procs[k] for k in ("control", "capture", "perception")}   # start order
    hb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hb_sock.bind(("127.0.0.1", config.SUPERVISOR_PORT))
    hb_sock.setblocking(False)
    status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # unbound: can reach the laptop
    sim_addr = (config.SIM_IP, config.SIM_PORT)

    print("[supervisor] pid=%d, QNX 'on' %s" % (os.getpid(), "found" if shutil.which("on") else "not found (plain launch)"),
          flush=True)
    for p in procs.values():   # control first: it owns the safe state
        p.start()

    last_status = 0.0
    try:
        while True:
            now = time.monotonic()
            while True:
                try:
                    data, _ = hb_sock.recvfrom(4096)
                except (BlockingIOError, InterruptedError):
                    break
                except OSError:
                    break
                try:
                    msg = json.loads(data)
                except ValueError:
                    continue
                if msg.get("type") == "hb" and msg.get("name") in procs:
                    procs[msg["name"]].last_hb = now
            for p in procs.values():
                if p.restart_at is not None:           # waiting out a back-off
                    if now >= p.restart_at:
                        p.restart_at = None
                        p.start()
                    continue
                ev = p.check(now)
                if ev is None:
                    if now - p.started > 10.0:
                        p.quick_fails = 0
                    continue
                code = p.p.poll()
                detail = "exit code %s" % code if ev == "CRASH" else "no heartbeat %.2f s" % p.hb_age(now)
                if ev == "HANG":
                    p.kill()
                p.restarts += 1
                p.last_event = "%s at %s" % (ev, time.strftime("%H:%M:%S"))
                p.quick_fails = p.quick_fails + 1 if now - p.started < 3.0 else 0
                delay = min(5.0, 0.5 * (2 ** (p.quick_fails - 1))) if p.quick_fails > 1 else 0.0
                print("[supervisor] %s %s (pid %d, %s) -> restart%s" % (
                    p.name, ev, p.p.pid, detail, " in %.1f s (keeps failing at start)" % delay if delay else ""),
                    flush=True)
                if delay:
                    p.restart_at = now + delay
                else:
                    p.start()
            if now - last_status >= 1.0 / config.PROCS_HZ:
                last_status = now
                st = {"type": "procs", "procs": {}}
                for n, p in procs.items():
                    st["procs"][n] = {"pid": p.p.pid, "prio": p.prio, "restarts": p.restarts,
                                      "alive": p.p.poll() is None,
                                      "hb_age": round(p.hb_age(now), 3), "last_event": p.last_event}
                try:
                    status_sock.sendto(json.dumps(st).encode(), sim_addr)
                except OSError:
                    pass
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs.values():
            p.kill()
        print("[supervisor] stopped", flush=True)


if __name__ == "__main__":
    main()
