"""QNX Camera Library (libcamapi) access from Python via ctypes - no C build needed (Pi only).

The QNX sensor service owns the USB cameras (see qnx/sensor_dodge_usb.conf); this module opens
camera units through libcamapi, receives frames in the viewfinder callback, and converts them to
RGB with numpy. API per the QNX camera library reference (camera/camera_api.h).

Diagnostic (run on the Pi after starting the sensor service):
    python3 qnx_camera.py                 # list cameras, stream 5 s, print fps / format / size
    python3 qnx_camera.py --snapshot      # also save cam<N>.ppm snapshots (view them on the laptop)
"""
import ctypes
import ctypes.util
import os
import sys
import threading
import time

import numpy as np

# ---------------------------------------------------------------- camera_api.h values
CAMERA_EOK = 0                     # = EOK on QNX
CAMERA_MODE_RW = 0x0F              # PREAD | PWRITE | DREAD | DWRITE
CAMERA_VFMODE_DEFAULT = 0
CAMERA_VFMODE_VIDEO = 1
# camera_frametype_t (enum order from the reference, no explicit values)
FT = {"UNSPECIFIED": 0, "NV12": 1, "RGB8888": 2, "RGB888": 3, "GRAY8": 4, "BAYER": 5, "CBYCRY": 6,
      "COMPRESSEDVIDEO": 7, "RGB565": 8, "YCBCR420P": 9, "YCBYCR": 10, "YCRYCB": 11, "CRYCBY": 12,
      "NV16": 15, "BGR8888": 17, "NV21": 22}
FT_NAME = {v: k for k, v in FT.items()}
# position of (Y0, Cb, Y1, Cr) inside one 4-byte macro-pixel for the packed 4:2:2 formats
PACKED_422 = {FT["YCBYCR"]: (0, 1, 2, 3), FT["CBYCRY"]: (1, 0, 3, 2),
              FT["YCRYCB"]: (0, 3, 2, 1), FT["CRYCBY"]: (1, 2, 3, 0)}


class FrameDesc(ctypes.Structure):
    """Common prefix of the camera_framedesc_t union members we use:
    ycbycr/cbycry/rgb8888/rgb888/gray8 = {height, width, stride}; nv12 adds uv_offset, uv_stride."""
    _fields_ = [("height", ctypes.c_uint32), ("width", ctypes.c_uint32), ("stride", ctypes.c_uint32),
                ("uv_offset", ctypes.c_int64), ("uv_stride", ctypes.c_int64)]


class CameraBuffer(ctypes.Structure):
    _fields_ = [("frametype", ctypes.c_int), ("framesize", ctypes.c_uint64),
                ("framebuf", ctypes.c_void_p), ("framemetasize", ctypes.c_uint64),
                ("framemeta", ctypes.c_void_p), ("frametimestamp", ctypes.c_int64),
                ("framedesc", FrameDesc)]


DEBUG = False            # --debug: log every step and dump the first buffer headers
NO_COPY = False          # --no-copy: never read pixel memory (isolates layout problems)
DUMP_FIRST = 3           # buffer headers kept per camera for diagnosis
DUMP_BYTES = 80
BYTES_PER_PIXEL = {FT["YCBYCR"]: 2, FT["CBYCRY"]: 2, FT["YCRYCB"]: 2, FT["CRYCBY"]: 2, FT["NV12"]: 1,
                   FT["NV21"]: 1, FT["RGB8888"]: 4, FT["BGR8888"]: 4, FT["RGB888"]: 3, FT["GRAY8"]: 1}


def _log(msg):
    sys.stderr.write("[qnx_camera] %s\n" % msg)
    sys.stderr.flush()


def _hexdump(raw):
    return " ".join("%02x" % c for c in bytearray(raw))


def validate(ft, d, ptr):
    """Reject a frame whose description is implausible BEFORE touching its pixel memory."""
    if not ptr:
        return "framebuf is NULL"
    bpp = BYTES_PER_PIXEL.get(ft)
    if bpp is None:
        return "frametype %d (%s) not supported" % (ft, FT_NAME.get(ft, "unknown value"))
    if not (16 <= d.width <= 8192 and 16 <= d.height <= 8192):
        return "implausible size"
    if not (d.width * bpp <= d.stride <= d.width * bpp * 4 + 4096):
        return "implausible stride"
    if ft in (FT["NV12"], FT["NV21"]):
        if not (d.stride * d.height <= d.uv_offset <= 4 * d.stride * d.height) or not (d.width <= d.uv_stride <= 4 * d.width + 4096):
            return "implausible NV12 uv_offset/uv_stride (%d, %d)" % (d.uv_offset, d.uv_stride)
    return None


VF_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.POINTER(CameraBuffer), ctypes.c_void_p)
STATUS_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.c_int, ctypes.c_uint16, ctypes.c_void_p)

_lib = None


def lib():
    global _lib
    if _lib is not None:
        return _lib
    if DEBUG:
        _log("loading libcamapi (find_library -> %s)" % ctypes.util.find_library("camapi"))
    names = [ctypes.util.find_library("camapi"), "libcamapi.so", "libcamapi.so.1"]
    err = None
    for n in names:
        if not n:
            continue
        try:
            _lib = ctypes.CDLL(n)
            break
        except OSError as e:
            err = e
    if _lib is None:
        raise OSError("cannot load libcamapi (QNX camera library): %s" % err)
    L = _lib
    if DEBUG:
        _log("loaded %s" % getattr(L, "_name", L))
    L.camera_get_supported_cameras.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
    L.camera_open.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)]
    L.camera_close.argtypes = [ctypes.c_int32]
    L.camera_get_supported_vf_modes.argtypes = [ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                                                ctypes.POINTER(ctypes.c_int)]
    L.camera_set_vf_mode.argtypes = [ctypes.c_int32, ctypes.c_int]
    L.camera_start_viewfinder.argtypes = [ctypes.c_int32, VF_CALLBACK, STATUS_CALLBACK, ctypes.c_void_p]
    L.camera_stop_viewfinder.argtypes = [ctypes.c_int32]
    for f in ("camera_get_supported_cameras", "camera_open", "camera_close", "camera_get_supported_vf_modes",
              "camera_set_vf_mode", "camera_start_viewfinder", "camera_stop_viewfinder"):
        getattr(L, f).restype = ctypes.c_int
    return L


def list_cameras():
    """Camera unit values: CAMERA_UNIT_n = n = sensor unit n (0 = none; verified on the QNX 8.0 Pi 5)."""
    L = lib()
    n = ctypes.c_uint(0)
    if DEBUG:
        _log("camera_get_supported_cameras (presize)")
    err = L.camera_get_supported_cameras(0, ctypes.byref(n), None)
    if err != CAMERA_EOK:
        raise RuntimeError("camera_get_supported_cameras failed: error %d" % err)
    arr = (ctypes.c_int * max(n.value, 1))()
    err = L.camera_get_supported_cameras(n.value, ctypes.byref(n), arr)
    if err != CAMERA_EOK:
        raise RuntimeError("camera_get_supported_cameras failed: error %d" % err)
    return sorted(arr[i] for i in range(n.value))


def frame_nbytes(ft, d):
    if ft == FT["NV12"] or ft == FT["NV21"]:
        return int(d.uv_offset + (d.height // 2) * d.uv_stride)
    return int(d.height * d.stride)


class Frame(object):
    __slots__ = ("data", "ftype", "w", "h", "stride", "uv_offset", "uv_stride", "t_us", "seq")


class QnxCamera(object):
    """One camera unit: latest raw frame kept from the viewfinder callback (camera library thread)."""

    def __init__(self, unit):
        self.unit = unit
        self.handle = ctypes.c_int32(-1)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.latest = None
        self.seq = 0
        self.status_events = []
        self.callbacks = 0
        self.bad = None
        self.dumps = []
        self._vf_cb = VF_CALLBACK(self._on_frame)          # keep references: ctypes callbacks must not be GC'd
        self._st_cb = STATUS_CALLBACK(self._on_status)
        L = lib()
        if DEBUG:
            _log("camera_open(sensor unit %d)" % unit)
        err = L.camera_open(unit, CAMERA_MODE_RW, ctypes.byref(self.handle))
        if err != CAMERA_EOK:
            raise RuntimeError("camera_open(sensor unit %d) failed: error %d" % (unit, err))
        if DEBUG:
            _log("  handle %d; querying viewfinder modes" % self.handle.value)
        mode = self._pick_mode()
        if DEBUG:
            _log("  camera_set_vf_mode(%d)" % mode)
        err = L.camera_set_vf_mode(self.handle, mode)
        if err != CAMERA_EOK:
            self.close()
            raise RuntimeError("camera_set_vf_mode(%d) failed on unit %d: error %d" % (mode, unit, err))
        if DEBUG:
            _log("  camera_start_viewfinder (frames arrive in a camera-library thread)")
        err = L.camera_start_viewfinder(self.handle, self._vf_cb, self._st_cb, None)
        if DEBUG:
            _log("  viewfinder started, error code %d" % err)
        if err != CAMERA_EOK:
            self.close()
            raise RuntimeError("camera_start_viewfinder failed on unit %d: error %d" % (unit, err))
        self.started = True

    def _pick_mode(self):
        L = lib()
        n = ctypes.c_uint32(0)
        if L.camera_get_supported_vf_modes(self.handle, 0, ctypes.byref(n), None) != CAMERA_EOK or n.value == 0:
            return CAMERA_VFMODE_VIDEO
        arr = (ctypes.c_int * n.value)()
        L.camera_get_supported_vf_modes(self.handle, n.value, ctypes.byref(n), arr)
        modes = [arr[i] for i in range(n.value)]
        if CAMERA_VFMODE_VIDEO in modes:
            return CAMERA_VFMODE_VIDEO
        non_default = [m for m in modes if m != CAMERA_VFMODE_DEFAULT]
        return non_default[0] if non_default else CAMERA_VFMODE_VIDEO

    def _on_frame(self, handle, buf_p, arg):
        try:
            self.callbacks += 1
            if not buf_p:
                self.bad = "NULL buffer pointer"
                return
            if self.callbacks <= DUMP_FIRST:
                # raw camera_buffer_t header, so the real layout can be checked against the docs
                raw = ctypes.string_at(ctypes.cast(buf_p, ctypes.c_void_p), DUMP_BYTES)
                self.dumps.append(raw)
                if DEBUG:
                    _log("unit %d callback #%d header: %s" % (self.unit, self.callbacks, _hexdump(raw)))
            b = buf_p.contents
            d = b.framedesc
            f = Frame()
            f.ftype = b.frametype
            f.w, f.h, f.stride = d.width, d.height, d.stride
            f.uv_offset, f.uv_stride = d.uv_offset, d.uv_stride
            f.t_us = b.frametimestamp
            problem = validate(b.frametype, d, b.framebuf)
            if problem:
                # do NOT touch the pixel memory: a wrong layout guess would read a bad address
                self.bad = "frametype %d w %d h %d stride %d framebuf 0x%x: %s" % (
                    b.frametype, d.width, d.height, d.stride, b.framebuf or 0, problem)
                return
            if NO_COPY:
                f.data = None
            else:
                f.data = ctypes.string_at(b.framebuf, frame_nbytes(b.frametype, d))
            with self.cond:
                self.seq += 1
                f.seq = self.seq
                self.latest = f
                self.cond.notify_all()
        except Exception as exc:      # never let an exception escape into the camera library thread
            sys.stderr.write("[qnx_camera] callback error: %s\n" % exc)

    def _on_status(self, handle, status, extra, arg):
        self.status_events.append((time.monotonic(), status, extra))

    def wait_newer(self, seq, timeout):
        with self.cond:
            end = time.monotonic() + timeout
            while self.latest is None or self.latest.seq <= seq:
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self.cond.wait(left)
            return self.latest

    def close(self):
        L = lib()
        if getattr(self, "started", False):
            L.camera_stop_viewfinder(self.handle)
            self.started = False
        if self.handle.value >= 0:
            L.camera_close(self.handle)
            self.handle = ctypes.c_int32(-1)


# ---------------------------------------------------------------- shared memory from qnx/stereo_capture.c
SHM_PATH = "/dev/shmem/dodge_stereo"
SHM_MAGIC = 0x31435344           # "DSC1"
SHM_HDR = 4096
SHM_CAM0 = 64                    # camera blocks: 64 bytes each


class ShmStereo(object):
    """Reads the latest frame of each camera published by stereo_capture (C) - no ctypes callbacks.
    Sequence lock: a block's seq is odd while the C callback writes it; read, copy, re-check."""

    def __init__(self, path=SHM_PATH, wait_s=10.0):
        import mmap
        import struct
        self.struct = struct
        t0 = time.monotonic()
        while True:
            try:
                fd = os.open(path, os.O_RDONLY)
                size = os.fstat(fd).st_size
                if size > SHM_HDR:
                    self.mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
                    os.close(fd)
                    if struct.unpack_from("<I", self.mm, 0)[0] == SHM_MAGIC:
                        break
                    self.mm.close()
                else:
                    os.close(fd)
            except OSError:
                pass
            if time.monotonic() - t0 > wait_s:
                raise RuntimeError("no stereo_capture shared memory at %s after %.0f s "
                                   "(is ./stereo_capture running?)" % (path, wait_s))
            time.sleep(0.2)
        self.path = path
        _, _, self.ncams, self.slot_size, self.pid = struct.unpack_from("<IIIIQ", self.mm, 0)
        self.units = [struct.unpack_from("<I", self.mm, SHM_CAM0 + i * 64 + 52)[0] for i in range(self.ncams)]

    def heartbeat_age(self):
        hb = self.struct.unpack_from("<q", self.mm, 24)[0]
        return time.monotonic() - hb / 1e6

    def bad(self, i):
        return self.struct.unpack_from("<Q", self.mm, 32 + 8 * i)[0]

    def seq(self, i):
        return self.struct.unpack_from("<Q", self.mm, SHM_CAM0 + i * 64)[0]

    def read(self, i):
        """Consistent copy of camera i's latest frame, or None if nothing published yet."""
        off = SHM_CAM0 + i * 64
        for _ in range(50):
            s1 = self.struct.unpack_from("<Q", self.mm, off)[0]
            if s1 == 0:
                return None
            if s1 & 1:
                time.sleep(0.0005)
                continue
            (t_us, ftype, w, h, stride, uvo, uvs, nbytes, unit, frames) = self.struct.unpack_from(
                "<qIIIIqqIIQ", self.mm, off + 8)
            start = SHM_HDR + i * self.slot_size
            data = self.mm[start:start + nbytes]
            if self.struct.unpack_from("<Q", self.mm, off)[0] != s1:
                continue                     # overwritten while copying: retry
            f = Frame()
            f.data, f.ftype, f.w, f.h, f.stride = data, ftype, w, h, stride
            f.uv_offset, f.uv_stride, f.t_us, f.seq = uvo, uvs, t_us, s1 // 2
            return f
        return None

    def wait_newer(self, i, seq, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.seq(i) // 2 > seq and self.heartbeat_age() < 1.0:
                f = self.read(i)
                if f is not None and f.seq > seq:
                    return f
            time.sleep(0.002)
        return None

    def close(self):
        try:
            self.mm.close()
        except Exception:
            pass


def shm_check(seconds, snapshot, path=SHM_PATH):
    """Diagnostic for the C capture path: fps, format, snapshot of each camera."""
    sh = ShmStereo(path, wait_s=5.0)
    print("stereo_capture shared memory OK: pid %d, units %s, heartbeat %.2f s ago" % (
        sh.pid, sh.units, sh.heartbeat_age()))
    s0 = [sh.seq(i) // 2 for i in range(sh.ncams)]
    time.sleep(seconds)
    for i in range(sh.ncams):
        n = sh.seq(i) // 2 - s0[i]
        f = sh.read(i)
        side = "left" if i == 0 else "right"
        if f is None:
            print("%s (unit %d): NO frames (bad frames: %d)" % (side, sh.units[i], sh.bad(i)))
            continue
        lag = (time.monotonic() * 1e6 - f.t_us) / 1000.0
        print("%s (unit %d): %s %dx%d stride %d  %.1f fps  bad %d  timestamp lag %.0f ms" % (
            side, sh.units[i], FT_NAME.get(f.ftype, f.ftype), f.w, f.h, f.stride, n / seconds, sh.bad(i), lag))
        try:
            rgb = to_rgb(f, 320, 240)
            print("         RGB mean colour %s" % np.round(rgb.reshape(-1, 3).mean(0), 1).tolist())
            if snapshot:
                p = os.path.abspath("cam_%s.ppm" % side)
                save_ppm(p, rgb)
                print("         saved %s" % p)
        except ValueError as exc:
            print("         %s" % exc)
    sh.close()
    return 0


# ---------------------------------------------------------------- pixel conversion (numpy)
def _yuv_to_rgb(Y, Cb, Cr):
    """BT.601 limited range (what UVC webcams send) -> uint8 RGB."""
    y = (Y.astype(np.float32) - 16.0) * 1.164
    cb = Cb.astype(np.float32) - 128.0
    cr = Cr.astype(np.float32) - 128.0
    rgb = np.stack([y + 1.596 * cr, y - 0.392 * cb - 0.813 * cr, y + 2.017 * cb], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def to_rgb(f, out_w, out_h):
    """Raw frame -> (out_h, out_w, 3) uint8 RGB, nearest/box resampled."""
    if f.data is None:
        raise ValueError("empty frame")
    w, h, s = f.w, f.h, f.stride
    ry = (np.arange(out_h) * h // out_h)
    rx = (np.arange(out_w) * w // out_w)
    raw = np.frombuffer(f.data, np.uint8)
    if f.ftype in PACKED_422:
        iy0, icb, iy1, icr = PACKED_422[f.ftype]
        a = raw[:h * s].reshape(h, s)[:, :2 * w].reshape(h, w // 2, 4)[ry][:, rx // 2]
        Y = (a[..., iy0].astype(np.uint16) + a[..., iy1]) // 2
        return _yuv_to_rgb(Y, a[..., icb], a[..., icr])
    if f.ftype in (FT["NV12"], FT["NV21"]):
        Yp = raw[:h * s].reshape(h, s)[:, :w]
        uv = raw[f.uv_offset:f.uv_offset + (h // 2) * f.uv_stride].reshape(h // 2, f.uv_stride)[:, :w]
        uv = uv.reshape(h // 2, w // 2, 2)[ry // 2][:, rx // 2]
        cb, cr = (uv[..., 0], uv[..., 1]) if f.ftype == FT["NV12"] else (uv[..., 1], uv[..., 0])
        return _yuv_to_rgb(Yp[ry][:, rx], cb, cr)
    if f.ftype in (FT["RGB8888"], FT["BGR8888"]):
        a = raw[:h * s].reshape(h, s)[:, :4 * w].reshape(h, w, 4)[ry][:, rx]
        return np.ascontiguousarray(a[..., :3] if f.ftype == FT["RGB8888"] else a[..., 2::-1])
    if f.ftype == FT["RGB888"]:
        return np.ascontiguousarray(raw[:h * s].reshape(h, s)[:, :3 * w].reshape(h, w, 3)[ry][:, rx])
    if f.ftype == FT["GRAY8"]:
        g = raw[:h * s].reshape(h, s)[:, :w][ry][:, rx]
        return np.repeat(g[..., None], 3, axis=2)
    raise ValueError("unsupported frame type %s (%d) - set default_video_format = ycbycr in the sensor config"
                     % (FT_NAME.get(f.ftype, "?"), f.ftype))


def save_ppm(path, rgb):
    with open(path, "wb") as fh:
        fh.write(b"P6\n%d %d\n255\n" % (rgb.shape[1], rgb.shape[0]))
        fh.write(np.ascontiguousarray(rgb).tobytes())


# ---------------------------------------------------------------- diagnostic
def callback_selftest():
    """Can this Python run ctypes callbacks at all? (libffi closures) Uses libc qsort, no camera."""
    libc = ctypes.CDLL(ctypes.util.find_library("c") or ("msvcrt" if os.name == "nt" else "libc.so"))
    CMP = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    calls = [0]
    def cmp(a, b):
        calls[0] += 1
        return a[0] - b[0]
    arr = (ctypes.c_int * 5)(5, 1, 4, 2, 3)
    cb = CMP(cmp)
    _log("selftest: calling qsort with a Python callback (if this crashes, ctypes callbacks are broken here)")
    libc.qsort(arr, 5, ctypes.sizeof(ctypes.c_int), cb)
    ok = list(arr) == [1, 2, 3, 4, 5]
    _log("selftest: %s (callback called %d times, result %s)" % ("OK" if ok else "WRONG", calls[0], list(arr)))
    # same thing from a non-Python thread is what the camera library does; qsort runs in this thread only
    return ok


def main():
    global DEBUG, NO_COPY
    import argparse
    import faulthandler
    faulthandler.enable()          # a segfault now prints the Python line it happened at
    ap = argparse.ArgumentParser(description="QNX camera check")
    ap.add_argument("--snapshot", action="store_true", help="save cam<N>.ppm")
    ap.add_argument("--debug", action="store_true", help="log every step, dump buffer headers")
    ap.add_argument("--no-copy", action="store_true", help="do not read pixel memory")
    ap.add_argument("--unit", type=int, default=None, help="test only this sensor unit (1, 2, ...)")
    ap.add_argument("--selftest", action="store_true", help="only test ctypes callbacks (no camera)")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--direct", action="store_true",
                    help="open the cameras from Python via ctypes callbacks (crashes on images without libffi closures)")
    args = ap.parse_args()
    DEBUG, NO_COPY = args.debug, args.no_copy
    if args.selftest:
        return 0 if callback_selftest() else 1
    if not args.direct:
        return shm_check(args.seconds, args.snapshot)
    units = list_cameras()
    print("camera units available: %s" % " ".join(str(u) for u in units))
    sys.stdout.flush()
    if not units:
        print("no cameras: is the sensor service running with the USB config? (see qnx/README.md)")
        return 1
    if args.unit is not None:
        units = [args.unit]
    cams = []
    for u in units:
        try:
            cams.append(QnxCamera(u))
        except RuntimeError as exc:
            print("unit %d: %s" % (u, exc))
    t0 = time.monotonic()
    first = {c.unit: None for c in cams}
    while time.monotonic() - t0 < args.seconds:
        time.sleep(0.25)
        for c in cams:
            if first[c.unit] is None and c.latest is not None:
                first[c.unit] = c.latest.seq
    for c in cams:
        print("unit %d: %d callbacks" % (c.unit, c.callbacks))
        for k, raw in enumerate(c.dumps[:1]):
            print("         buffer header #%d: %s" % (k + 1, _hexdump(raw)))
        if c.bad:
            print("         frames skipped (not read): %s" % c.bad)
        f = c.latest
        if f is None:
            print("         NO usable frames in %.0f s (status events: %s)" % (args.seconds, c.status_events[:5]))
            continue
        fps = (f.seq - (first[c.unit] or 0)) / max(args.seconds - 0.25, 0.1)
        lag_ms = (time.monotonic() * 1e6 - f.t_us) / 1000.0
        print("         %s (%d) %dx%d stride %d  ~%.1f fps  timestamp lag %.1f ms" % (
            FT_NAME.get(f.ftype, "?"), f.ftype, f.w, f.h, f.stride, fps, lag_ms))
        if f.data is None:
            continue
        try:
            rgb = to_rgb(f, 320, 240)
            print("         RGB 320x240 mean colour %s" % np.round(rgb.reshape(-1, 3).mean(0), 1).tolist())
            if args.snapshot:
                p = os.path.abspath("cam%d.ppm" % c.unit)
                save_ppm(p, rgb)
                print("         saved %s" % p)
        except ValueError as exc:
            print("         %s" % exc)
    sys.stdout.flush()
    for c in cams:
        c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
