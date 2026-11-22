"""Live stereo view (laptop): both Pi camera images with what the Pi's CNN and stereo see.

On the Pi:      python3 perception.py --source qnx --preview <laptop-ip>
On the laptop:  python live_view.py
(Allow UDP port 5008 through the laptop firewall.)

Overlays: every ball-coloured candidate (config.BALL_COLOR) with its CNN P(ball) (green = accepted, red = rejected),
the confirmed ball with depth, the epipolar row, and why a frame was rejected.
Keys: m = show the ball-colour mask, s = save a snapshot, q / ESC = quit.
"""
import argparse
import json
import socket
import time
import zlib

import cv2
import numpy as np

import config
import perception

GREEN = (60, 220, 60)
RED = (60, 60, 230)
GRAY = (160, 160, 160)
CYAN = (220, 200, 40)
WHITE = (240, 240, 240)
ORANGE = (40, 160, 255)


def put(img, text, org, color=WHITE, scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(img_rgb, meta, view_w, show_mask, preview_fps):
    w, h = meta["w"], meta["h"]
    mono = bool(meta.get("mono"))
    left, right = (img_rgb, None) if mono else (img_rgb[:, :w], img_rgb[:, w:])
    k = view_w / float(config.IMG_W)                    # full-res (320x240) coords -> display px
    view_h = int(round(config.IMG_H * k))
    panels = []
    for img in ((left,) if mono else (left, right)):
        if show_mask:
            m = perception.ball_mask(img)
            img = np.where(m[..., None], np.array([255, 60, 200], np.uint8), (img // 3).astype(np.uint8))
        panels.append(cv2.resize(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR), (view_w, view_h),
                                 interpolation=cv2.INTER_LINEAR))
    thr = config.CNN_THRESHOLD
    for panel, cands, name in (((panels[0], meta.get("cands_l", []), "CAMERA"),) if mono else
                               ((panels[0], meta.get("cands_l", []), "LEFT"),
                                (panels[1], meta.get("cands_r", []), "RIGHT"))):
        put(panel, name, (8, 22), WHITE, 0.6, 2)
        for u, v, r, p in cands:
            col = GRAY if p < 0 else (GREEN if p >= thr else RED)
            c = (int(u * k), int(v * k))
            cv2.circle(panel, c, max(3, int(r * k)), col, 2, cv2.LINE_AA)
            put(panel, "CNN %.2f" % p if p >= 0 else "?", (c[0] - 30, c[1] - int(r * k) - 6), col, 0.45)
    if meta.get("detected"):
        u, v = meta["uv"]; r = meta["radius_px"]; d = meta["disparity_px"]
        for panel, uu in (((panels[0], u),) if mono else ((panels[0], u), (panels[1], u - d))):
            x0, y0 = int((uu - r * 1.3) * k), int((v - r * 1.3) * k)
            x1, y1 = int((uu + r * 1.3) * k), int((v + r * 1.3) * k)
            cv2.rectangle(panel, (x0, y0), (x1, y1), GREEN, 3)
            put(panel, "BALL %.3f m" % meta["depth_m"], (x0, min(y1 + 20, view_h - 6)), GREEN, 0.6, 2)
        for panel in panels:
            cv2.line(panel, (0, int(v * k)), (view_w, int(v * k)), CYAN, 1, cv2.LINE_AA)
    canvas = panels[0] if mono else np.concatenate([panels[0], np.full((view_h, 4, 3), 40, np.uint8), panels[1]], axis=1)

    bar = np.full((96, canvas.shape[1], 3), 25, np.uint8)
    put(bar, "Pi perception %.1f fps | %s | CNN %.1f ms  stereo %.1f ms | latency %.0f ms | camera skew %+.1f ms" % (
        meta.get("fps", 0), meta.get("classifier", "?"), meta.get("cnn_ms", 0), meta.get("stereo_ms", 0),
        meta.get("latency_ms", 0), meta.get("skew_ms", 0)), (8, 20), WHITE, 0.5)
    put(bar, "source %s %s | baseline %.2f m  f %.0f px | preview %.1f fps" % (
        meta.get("source", "?"), meta.get("src_info", ""), meta.get("baseline", 0), meta.get("f", 0), preview_fps),
        (8, 42), GRAY, 0.5)
    if meta.get("detected"):
        X, Y, Z = meta.get("pos_raw", [0, 0, 0])
        st = meta.get("streak", 0)
        tag = "BALL CONFIRMED" if st >= config.CONFIRM_FRAMES else "BALL (confirming %d/%d)" % (st, config.CONFIRM_FRAMES)
        put(bar, "%s: depth %.3f m%s  X %+.3f  Y %+.3f  disp %.1f px  size x%.2f  conf %.2f" % (
            tag + (" [shape]" if meta.get("via") == "shape" else " [cnn]"), Z,
            " (centre)" if meta.get("depth_src") == "centroid" else "", X, Y, meta.get("disparity_px", 0),
            meta.get("size_ratio", 0), meta.get("conf", 0)), (8, 70), GREEN, 0.55, 2)
    else:
        why = meta.get("reason", "?")
        hint = {"epipolar_mismatch": "ball rows differ by %s px (tol %s): align cameras / calibrate / --epipolar-tol",
                "disparity_out_of_range": "raw disparity %s px: too close (<~20 cm) or cameras swapped (--swap-lr)",
                "size_mismatch": "round thing of the wrong size for its distance (not the %d mm ball), size x%%s" % round(config.BALL_DIAM_M * 1000),
                "size_mismatch_left_right": "left/right candidates differ in size",
                "cnn_rejected_left": "CNN says not a ball in LEFT image", "cnn_rejected_right": "CNN says not a ball in RIGHT image",
                "no_candidates_left": "no ball-coloured blob in LEFT image (press m; lighting? merged with a same-colour background?)",
                "no_candidates_right": "no ball-coloured blob in RIGHT image (press m)"}
        h_txt = hint.get(why, "")
        if why == "epipolar_mismatch":
            h_txt = h_txt % ("%.1f" % meta["dv"] if meta.get("dv") is not None else "?", meta.get("epi_tol"))
        elif why == "size_mismatch":
            h_txt = h_txt % ("%.2f" % meta["size_ratio"] if meta.get("size_ratio") is not None else "?")
        elif why == "disparity_out_of_range":
            h_txt = h_txt % ("%.1f" % meta["d_raw"] if meta.get("d_raw") is not None else "?")
        put(bar, "NO BALL: %s   %s" % (why, h_txt), (8, 70), ORANGE, 0.55, 2)
    foot = np.full((24, canvas.shape[1], 3), 25, np.uint8)
    put(foot, "m: ball-colour mask %s   s: snapshot   q: quit" % ("ON" if show_mask else "off"), (8, 17), GRAY, 0.45)
    return np.concatenate([bar, canvas, foot], axis=0)


class PreviewReceiver(object):
    """Reassembles the perception preview stream (DPVM meta + DPVC zlib chunks) from a non-blocking
    UDP socket. poll() drains what is waiting and returns the newest complete (image, meta) or None."""

    def __init__(self, sock):
        self.sock = sock
        self.pending = {}
        self.latest = None
        self.t_latest = None

    def poll(self, max_packets=600):
        new = None
        for _ in range(max_packets):
            try:
                data, _ = self.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError, socket.timeout):
                break
            except OSError:
                break
            now = time.monotonic()
            try:
                if data[:4] == b"DPVM":
                    meta = json.loads(data[4:].decode())
                    self.pending.setdefault(meta["fid"], {"chunks": {}, "t": now})["meta"] = meta
                elif data[:4] == b"DPVC" and len(data) >= 16:
                    fid, i, n = (int(x) for x in np.frombuffer(data[4:16], "<u4"))
                    e = self.pending.setdefault(fid, {"chunks": {}, "t": now})
                    e["chunks"][i] = data[16:]
                    e["n"] = n
                else:
                    continue
            except (ValueError, KeyError, UnicodeDecodeError):
                continue
            for fid in [f for f, e in self.pending.items() if "meta" in e and len(e["chunks"]) == e.get("n", -1)]:
                e = self.pending.pop(fid)
                m = e["meta"]
                try:
                    raw = zlib.decompress(b"".join(e["chunks"][j] for j in range(e["n"])))
                    img = np.frombuffer(raw, np.uint8).reshape(m["h"], (1 if m.get("mono") else 2) * m["w"], 3)
                except (zlib.error, ValueError, KeyError):
                    continue
                new = (img, m)
                self.latest, self.t_latest = new, now
        now = time.monotonic()
        for fid in [f for f, e in self.pending.items() if now - e["t"] > 1.0]:
            self.pending.pop(fid)
        return new


def waiting_screen(width, port, since):
    img = np.full((300, width, 3), 25, np.uint8)
    put(img, "Waiting for the Pi preview stream on UDP port %d ... (%.0f s)" % (port, since), (20, 60), WHITE, 0.7, 2)
    put(img, "On the Pi:  python3 perception.py --source qnx --preview <this laptop's IP>", (20, 110), GRAY, 0.6)
    try:
        ips = sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
        put(img, "This laptop's IPs: %s" % ", ".join(ips), (20, 150), GRAY, 0.6)
    except OSError:
        pass
    put(img, "Firewall: allow inbound UDP %d (admin PowerShell)" % port, (20, 190), GRAY, 0.6)
    return img


def main():
    ap = argparse.ArgumentParser(description="DODGE-1 live stereo view")
    ap.add_argument("--port", type=int, default=config.PREVIEW_PORT)
    ap.add_argument("--width", type=int, default=560, help="display width of each camera view")
    ap.add_argument("--save-after", type=float, default=None, help="(testing) save one frame after N s and exit")
    ap.add_argument("--out", default="live_view.png")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(0.02)
    pending = {}            # fid -> {"meta", "chunks", "t"}
    shown = None
    show_mask = False
    t_start = time.monotonic()
    last_frame_t = None
    shown_times = []
    win = "DODGE-1 live stereo (Pi CNN + depth)"
    headless = args.save_after is not None
    stats = {"pkts": 0, "bytes": 0, "src": "-", "ok": 0, "dropped": 0, "bad": 0}
    if not headless:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    while True:
        # drain packets for at most 40 ms, then ALWAYS redraw (a continuous stream must not starve the window)
        t_drain = time.monotonic()
        while time.monotonic() - t_drain < 0.04:
            try:
                data, src = sock.recvfrom(65536)
            except socket.timeout:
                break
            except OSError:
                break
            now = time.monotonic()
            stats["pkts"] += 1; stats["bytes"] += len(data); stats["src"] = src[0]
            try:
                if data[:4] == b"DPVM":
                    meta = json.loads(data[4:].decode())
                    pending.setdefault(meta["fid"], {"chunks": {}, "t": now})["meta"] = meta
                elif data[:4] == b"DPVC" and len(data) >= 16:
                    fid, i, n = (int(x) for x in np.frombuffer(data[4:16], "<u4"))
                    e = pending.setdefault(fid, {"chunks": {}, "t": now})
                    e["chunks"][i] = data[16:]
                    e["n"] = n
            except (ValueError, KeyError, UnicodeDecodeError):
                stats["bad"] += 1
                continue
            for fid in [f for f, e in pending.items() if "meta" in e and len(e["chunks"]) == e.get("n", -1)]:
                e = pending.pop(fid)
                m = e["meta"]
                try:
                    raw = zlib.decompress(b"".join(e["chunks"][j] for j in range(e["n"])))
                    img = np.frombuffer(raw, np.uint8).reshape(m["h"], (1 if m.get("mono") else 2) * m["w"], 3)
                except (zlib.error, ValueError, KeyError):
                    stats["bad"] += 1
                    continue
                shown = (img, m)
                last_frame_t = now
                shown_times.append(now)
                stats["ok"] += 1
        now = time.monotonic()
        for fid in [f for f, e in pending.items() if now - e["t"] > 1.0]:
            pending.pop(fid)              # incomplete (lost packets): drop
            stats["dropped"] += 1
        shown_times = [t for t in shown_times if now - t < 2.0]
        pfps = len(shown_times) / 2.0
        if shown is None or now - last_frame_t > 3.0:
            frame = waiting_screen(2 * args.width + 4, args.port, now - t_start)
            put(frame, "received: %d packets (%.1f MB) from %s | frames shown %d, dropped incomplete %d, bad %d" % (
                stats["pkts"], stats["bytes"] / 1e6, stats["src"], stats["ok"], stats["dropped"], stats["bad"]),
                (20, 240), WHITE, 0.55, 1)
            if stats["pkts"] == 0:
                put(frame, "nothing arrives: perception not started with --preview <this IP>, wrong IP, or firewall",
                    (20, 270), RED, 0.55, 1)
            elif stats["ok"] == 0:
                put(frame, "packets arrive but frames are incomplete (Wi-Fi loss): try perception --preview-scale 4",
                    (20, 270), RED, 0.55, 1)
            if shown is not None:
                put(frame, "stream stopped %.1f s ago" % (now - last_frame_t), (20, 295), RED, 0.6, 2)
        else:
            frame = render(shown[0], shown[1], args.width, show_mask, pfps)
        if headless:
            if now - t_start >= args.save_after:
                cv2.imwrite(args.out, frame)
                print("saved %s (preview %.1f fps, last reason %s)" % (
                    args.out, pfps, shown[1].get("reason") if shown else None))
                return
            continue
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            show_mask = not show_mask
        if key == ord("s"):
            name = time.strftime("snapshot_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, frame)
            print("saved", name)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
