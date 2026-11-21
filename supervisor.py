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
    def __init__(self, name, script, args, prio, cpus, hb_timeout, grace):
        self.name, self.script, self.args = name, script, args
        self.prio, self.cpus = prio, cpus
        self.hb_timeout, self.grace = hb_timeout, grace
        self.p = None
        self.started = 0.0
        self.last_hb = None
        self.restarts = 0
        self.last_event = "init"

    def cmdline(self):
        cmd = [sys.executable, "-u", os.path.join(HERE, self.script)] + self.args
        if shutil.which("on"):      # QNX: priority + core pinning
            pre = ["on", "-p", str(self.prio)]
            for c in self.cpus:
                pre += ["-C", str(c)]
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
                ev = p.check(now)
                if ev is None:
                    continue
                code = p.p.poll()
                detail = "exit code %s" % code if ev == "CRASH" else "no heartbeat %.2f s" % p.hb_age(now)
                print("[supervisor] %s %s (pid %d, %s) -> restart" % (p.name, ev, p.p.pid, detail), flush=True)
                if ev == "HANG":
                    p.kill()
                p.restarts += 1
                p.last_event = "%s at %s" % (ev, time.strftime("%H:%M:%S"))
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
