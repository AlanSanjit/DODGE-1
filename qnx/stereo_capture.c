/*
 * DODGE-1 stereo capture bridge (runs on the Raspberry Pi 5 / QNX 8.0).
 *
 * Opens two cameras through the QNX camera library (libcamapi; the cameras are owned by the
 * sensor service, see qnx/sensor_dodge_usb.conf), receives frames in the viewfinder callback
 * (in C - Python ctypes callbacks crash on this QNX image), and publishes the latest raw frame of
 * each camera in shared memory with a sequence lock. perception.py reads it with mmap + numpy.
 *
 * Build on the Pi:   clang -O2 -o stereo_capture qnx/stereo_capture.c -lcamapi
 * Run:               ./stereo_capture            (sensor units 1 = left, 2 = right)
 *                    ./stereo_capture -l         (list camera units)
 *                    ./stereo_capture -u 2,1     (choose/swap units)
 *
 * Shared memory layout (little endian), file /dev/shmem/dodge_stereo:
 *   header (4096 bytes): magic 'DSC1', version 1, ncams, slot_size, pid, heartbeat_us,
 *                        bad[2], cam[2] blocks of 64 bytes at offset 64 (see cam_block_t)
 *   slot i (raw frame bytes of camera i) at 4096 + i * slot_size
 * Camera block: seq is odd while the frame is being written (reader retries), even when stable.
 *
 * The camera API is declared here from the QNX camera library reference (camera/camera_api.h),
 * so the SDK header is not needed.
 */
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------ camera library declarations */
typedef int32_t camera_handle_t;
typedef int camera_error_t;        /* CAMERA_EOK = EOK = 0 */
typedef int camera_unit_t;         /* CAMERA_UNIT_n = n (value 0 = no unit; verified on QNX 8.0 Pi 5) */
typedef int camera_vfmode_t;
typedef int camera_frametype_t;
typedef int camera_devstatus_t;

typedef struct {                   /* common prefix of the camera_framedesc_t union members used */
    uint32_t height;
    uint32_t width;
    uint32_t stride;
    int64_t  uv_offset;            /* nv12 only */
    int64_t  uv_stride;            /* nv12 only */
} dodge_framedesc_t;

typedef struct {                   /* camera_buffer_t */
    camera_frametype_t frametype;
    uint64_t framesize;            /* size of this struct, NOT of the image */
    uint8_t *framebuf;
    uint64_t framemetasize;
    void    *framemeta;
    int64_t  frametimestamp;       /* CLOCK_MONOTONIC, microseconds */
    dodge_framedesc_t framedesc;   /* (a union in the real header; only read, never allocated here) */
} camera_buffer_t;

#define CAMERA_EOK             0
#define CAMERA_MODE_RW         0x0F
#define CAMERA_VFMODE_DEFAULT  0
#define CAMERA_VFMODE_VIDEO    1

#define FT_NV12     1
#define FT_RGB8888  2
#define FT_RGB888   3
#define FT_GRAY8    4
#define FT_CBYCRY   6
#define FT_YCBYCR   10
#define FT_YCRYCB   11
#define FT_CRYCBY   12
#define FT_BGR8888  17
#define FT_NV21     22

extern camera_error_t camera_get_supported_cameras(unsigned int numasked, unsigned int *numsupported,
                                                   camera_unit_t *cameras);
extern camera_error_t camera_open(camera_unit_t unit, uint32_t mode, camera_handle_t *handle);
extern camera_error_t camera_close(camera_handle_t handle);
extern camera_error_t camera_get_supported_vf_modes(camera_handle_t handle, uint32_t numasked,
                                                    uint32_t *numsupported, camera_vfmode_t *modes);
extern camera_error_t camera_set_vf_mode(camera_handle_t handle, camera_vfmode_t mode);
extern camera_error_t camera_start_viewfinder(camera_handle_t handle,
        void (*viewfinder_callback)(camera_handle_t, camera_buffer_t *, void *),
        void (*status_callback)(camera_handle_t, camera_devstatus_t, uint16_t, void *),
        void *arg);
extern camera_error_t camera_stop_viewfinder(camera_handle_t handle);

/* ------------------------------------------------------------ shared memory layout */
#define MAGIC     0x31435344u      /* "DSC1" */
#define VERSION   1u
#define NCAMS     2
#define HDR_SIZE  4096

typedef struct {
    uint64_t seq;        /*  0 */
    int64_t  t_us;       /*  8 */
    uint32_t frametype;  /* 16 */
    uint32_t width;      /* 20 */
    uint32_t height;     /* 24 */
    uint32_t stride;     /* 28 */
    int64_t  uv_offset;  /* 32 */
    int64_t  uv_stride;  /* 40 */
    uint32_t nbytes;     /* 48 */
    uint32_t unit;       /* 52  sensor unit (1-based) */
    uint64_t frames;     /* 56 */
} cam_block_t;           /* 64 bytes */

typedef struct {
    uint32_t magic;          /*  0 */
    uint32_t version;        /*  4 */
    uint32_t ncams;          /*  8 */
    uint32_t slot_size;      /* 12 */
    uint64_t pid;            /* 16 */
    int64_t  heartbeat_us;   /* 24 */
    uint64_t bad[NCAMS];     /* 32 */
    uint8_t  pad[16];        /* 48 */
    cam_block_t cam[NCAMS];  /* 64 */
} shm_header_t;

_Static_assert(sizeof(cam_block_t) == 64, "cam_block_t must be 64 bytes");
_Static_assert(sizeof(shm_header_t) == 64 + NCAMS * 64, "header layout");

typedef struct {
    int index;
    camera_unit_t unit;
    camera_handle_t handle;
    int open, started;
    shm_header_t *hdr;
    uint8_t *slot;
    uint32_t slot_size;
    char last_bad[160];
    int logged_first;
    int desc_off;        /* byte offset of {height, width, stride} inside camera_buffer_t (-1 = not found yet) */
    int buf_off;         /* byte offset of the framebuf pointer (-1 = not found yet) */
} cam_t;

static volatile sig_atomic_t g_stop = 0;
/* Pixel format of the frames. The camera_frametype_t NUMBERS on QNX 8.0 differ from the 7.1 docs
 * (a ycbycr USB stream reports type 14), so by default we trust the format we configured in
 * sensor_dodge_usb.conf (default_video_format) and only sanity-check it against stride/width.
 * -f auto trusts the reported number (7.1 numbering). Published in shared memory with 7.1 numbering. */
static int g_format = FT_YCBYCR;
static int g_trust_type = 0;

static void on_signal(int sig) { (void)sig; g_stop = 1; }

static int64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

static const char *ft_name(int ft)
{
    switch (ft) {
    case FT_NV12: return "NV12"; case FT_RGB8888: return "RGB8888"; case FT_RGB888: return "RGB888";
    case FT_GRAY8: return "GRAY8"; case FT_CBYCRY: return "CBYCRY"; case FT_YCBYCR: return "YCBYCR";
    case FT_YCRYCB: return "YCRYCB"; case FT_CRYCBY: return "CRYCBY"; case FT_BGR8888: return "BGR8888";
    case FT_NV21: return "NV21"; default: return "?";
    }
}

static int bytes_per_pixel(int ft)
{
    switch (ft) {
    case FT_CBYCRY: case FT_YCBYCR: case FT_YCRYCB: case FT_CRYCBY: return 2;
    case FT_NV12: case FT_NV21: case FT_GRAY8: return 1;
    case FT_RGB8888: case FT_BGR8888: return 4;
    case FT_RGB888: return 3;
    default: return 0;
    }
}

/* ---- layout discovery: QNX 8.0's camera_buffer_t does not match the 7.1 documentation
 * (the frame size fields sit 24 bytes later). We locate fields by their values instead. */
#define SCAN_BYTES 256

static uint32_t rd32(const uint8_t *p, int off) { uint32_t v; memcpy(&v, p + off, 4); return v; }
static uint64_t rd64(const uint8_t *p, int off) { uint64_t v; memcpy(&v, p + off, 8); return v; }

/* {height, width, stride} plausible for the configured format? */
static int desc_ok(uint32_t h, uint32_t w, uint32_t stride, int bpp)
{
    return w >= 16 && w <= 8192 && h >= 16 && h <= 8192 && (w % 2) == 0 &&
           stride >= w * (uint32_t)bpp && stride <= w * (uint32_t)bpp * 4 + 4096;
}

static int find_desc(const uint8_t *raw, int bpp)
{
    int off;
    for (off = 24; off + 12 <= SCAN_BYTES; off += 4)
        if (desc_ok(rd32(raw, off), rd32(raw, off + 4), rd32(raw, off + 8), bpp)) return off;
    return -1;
}

/* Is [p, p+n) readable? write() reports EFAULT instead of crashing on a bad address. */
static int readable(const void *p, size_t n)
{
    int fds[2], ok = 1;
    char tmp[2];
    if (p == NULL || n == 0) return 0;
    if (pipe(fds) != 0) return 1;            /* cannot check: assume readable */
    if (write(fds[1], p, 1) != 1) ok = 0;
    else if (read(fds[0], tmp, 1) != 1) ok = 0;
    if (ok && write(fds[1], (const char *)p + n - 1, 1) != 1) ok = 0;
    close(fds[0]);
    close(fds[1]);
    return ok;
}

static void dump_header(int unit, const uint8_t *raw)
{
    int off;
    fprintf(stderr, "[stereo_capture] unit %d camera_buffer_t header (64-bit words):\n", unit);
    for (off = 0; off < 128; off += 8)
        fprintf(stderr, "  +%3d: 0x%016llx  (u32 %u, %u)\n", off, (unsigned long long)rd64(raw, off),
                rd32(raw, off), rd32(raw, off + 4));
}

/* Frame size in bytes for the configured format (0 = implausible, reason in why). */
static uint64_t frame_bytes_at(const uint8_t *raw, int doff, int ft, char *why, size_t why_len)
{
    uint32_t h = rd32(raw, doff), w = rd32(raw, doff + 4), stride = rd32(raw, doff + 8);
    int bpp = bytes_per_pixel(ft);
    if (bpp == 0) { snprintf(why, why_len, "format %d not supported", ft); return 0; }
    if (!desc_ok(h, w, stride, bpp)) {
        snprintf(why, why_len, "size fields at +%d not plausible: %ux%u stride %u", doff, w, h, stride);
        return 0;
    }
    if (ft == FT_NV12 || ft == FT_NV21) {
        int64_t uvo = (int64_t)rd64(raw, doff + 16), uvs = (int64_t)rd64(raw, doff + 24);
        uint64_t ysz = (uint64_t)stride * h;
        if (uvo < (int64_t)ysz || uvo > (int64_t)(4 * ysz) || uvs < (int64_t)w || uvs > (int64_t)(4 * w + 4096)) {
            snprintf(why, why_len, "implausible NV12 uv_offset %lld uv_stride %lld", (long long)uvo, (long long)uvs);
            return 0;
        }
        return (uint64_t)uvo + (uint64_t)(h / 2) * (uint64_t)uvs;
    }
    return (uint64_t)stride * h;
}

/* Called by the camera library in its own thread for every frame. */
static void vf_callback(camera_handle_t handle, camera_buffer_t *buf, void *arg)
{
    cam_t *c = (cam_t *)arg;
    cam_block_t *blk;
    char why[160];
    uint64_t n, s;
    int64_t ts;
    const uint8_t *raw = (const uint8_t *)buf;
    const uint8_t *pix;
    int ft = g_format, bpp, doff;
    (void)handle;
    if (c == NULL || buf == NULL) return;
    blk = &c->hdr->cam[c->index];
    if (g_trust_type) ft = buf->frametype;
    bpp = bytes_per_pixel(ft);

    if (!c->logged_first) {
        c->logged_first = 1;
        dump_header((int)c->unit, raw);
    }
    if (c->desc_off < 0) {                       /* locate {height, width, stride} once */
        c->desc_off = find_desc(raw, bpp > 0 ? bpp : 2);
        if (c->desc_off < 0) {
            snprintf(why, sizeof why, "no {height,width,stride} fitting %s found in the buffer header", ft_name(ft));
            __atomic_add_fetch(&c->hdr->bad[c->index], 1, __ATOMIC_RELAXED);
            strncpy(c->last_bad, why, sizeof c->last_bad - 1);
            return;
        }
        fprintf(stderr, "[stereo_capture] unit %d: size fields at +%d: %ux%u stride %u (docs say +48)\n",
                (int)c->unit, c->desc_off, rd32(raw, c->desc_off + 4), rd32(raw, c->desc_off), rd32(raw, c->desc_off + 8));
    }
    doff = c->desc_off;
    n = frame_bytes_at(raw, doff, ft, why, sizeof why);
    if (n == 0 || n > c->slot_size) {
        if (n > c->slot_size) snprintf(why, sizeof why, "frame %llu bytes > slot %u", (unsigned long long)n, c->slot_size);
        __atomic_add_fetch(&c->hdr->bad[c->index], 1, __ATOMIC_RELAXED);
        strncpy(c->last_bad, why, sizeof c->last_bad - 1);
        return;
    }
    if (c->buf_off < 0) {                        /* locate the pixel pointer once: docs say +16 */
        int cand[16], nc = 0, k, off;
        cand[nc++] = 16;
        for (off = 8; off < doff && nc < 16; off += 8) if (off != 16) cand[nc++] = off;
        for (k = 0; k < nc; k++) {
            const void *p = (const void *)(uintptr_t)rd64(raw, cand[k]);
            if (((uintptr_t)p & 7) == 0 && readable(p, (size_t)n)) { c->buf_off = cand[k]; break; }
        }
        if (c->buf_off < 0) {
            snprintf(why, sizeof why, "no readable pixel pointer for %llu bytes in the buffer header",
                     (unsigned long long)n);
            __atomic_add_fetch(&c->hdr->bad[c->index], 1, __ATOMIC_RELAXED);
            strncpy(c->last_bad, why, sizeof c->last_bad - 1);
            return;
        }
        fprintf(stderr, "[stereo_capture] unit %d: pixel pointer at +%d (docs say +16), frame %llu bytes\n",
                (int)c->unit, c->buf_off, (unsigned long long)n);
    }
    pix = (const uint8_t *)(uintptr_t)rd64(raw, c->buf_off);
    if (pix == NULL) {
        __atomic_add_fetch(&c->hdr->bad[c->index], 1, __ATOMIC_RELAXED);
        strncpy(c->last_bad, "NULL pixel pointer", sizeof c->last_bad - 1);
        return;
    }
    ts = (int64_t)rd64(raw, 40);                 /* docs: frametimestamp (CLOCK_MONOTONIC us) at +40 */
    if (ts < now_us() - 10000000 || ts > now_us() + 1000000) ts = now_us();   /* not plausible: use arrival time */
    s = __atomic_load_n(&blk->seq, __ATOMIC_RELAXED);
    __atomic_store_n(&blk->seq, s + 1, __ATOMIC_RELAXED);          /* odd: writing */
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    memcpy(c->slot, pix, (size_t)n);
    blk->t_us = ts;
    blk->frametype = (uint32_t)ft;
    blk->height = rd32(raw, doff);
    blk->width = rd32(raw, doff + 4);
    blk->stride = rd32(raw, doff + 8);
    blk->uv_offset = (ft == FT_NV12 || ft == FT_NV21) ? (int64_t)rd64(raw, doff + 16) : 0;
    blk->uv_stride = (ft == FT_NV12 || ft == FT_NV21) ? (int64_t)rd64(raw, doff + 24) : 0;
    blk->nbytes = (uint32_t)n;
    blk->frames += 1;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __atomic_store_n(&blk->seq, s + 2, __ATOMIC_RELEASE);          /* even: stable */
}

static void status_callback(camera_handle_t handle, camera_devstatus_t status, uint16_t extra, void *arg)
{
    cam_t *c = (cam_t *)arg;
    (void)handle;
    fprintf(stderr, "[stereo_capture] unit %d status %d (%u)\n", c ? (int)c->unit : -1, (int)status,
            (unsigned)extra);
}

static int list_units(camera_unit_t *out, unsigned int max)
{
    unsigned int n_pre = 0, n = 0;
    camera_error_t e1, e2;
    /* presizing query, then a direct query with a full array (some versions only fill one form) */
    e1 = camera_get_supported_cameras(0, &n_pre, NULL);
    e2 = camera_get_supported_cameras(max, &n, out);
    fprintf(stderr, "[stereo_capture] camera_get_supported_cameras: presize err %d n %u | direct err %d n %u\n",
            e1, n_pre, e2, n);
    if (e2 != CAMERA_EOK && e1 != CAMERA_EOK) return -1;
    if (e2 != CAMERA_EOK) return 0;
    if (n > max) n = max;
    return (int)n;
}

static int start_camera(cam_t *c)
{
    camera_error_t err;
    camera_vfmode_t modes[16];
    uint32_t nm = 0, i;
    camera_vfmode_t mode = CAMERA_VFMODE_VIDEO;

    err = camera_open(c->unit, CAMERA_MODE_RW, &c->handle);
    if (err != CAMERA_EOK) {
        fprintf(stderr, "[stereo_capture] camera_open(sensor unit %d) failed: %d (%s)\n",
                (int)c->unit, err, strerror(err));
        return -1;
    }
    c->open = 1;
    if (camera_get_supported_vf_modes(c->handle, 16, &nm, modes) == CAMERA_EOK && nm > 0) {
        int has_video = 0;
        for (i = 0; i < nm && i < 16; i++) if (modes[i] == CAMERA_VFMODE_VIDEO) has_video = 1;
        if (!has_video) {
            for (i = 0; i < nm && i < 16; i++) if (modes[i] != CAMERA_VFMODE_DEFAULT) { mode = modes[i]; break; }
        }
    }
    err = camera_set_vf_mode(c->handle, mode);
    if (err != CAMERA_EOK) {
        fprintf(stderr, "[stereo_capture] camera_set_vf_mode(%d) on unit %d failed: %d (%s)\n",
                mode, (int)c->unit, err, strerror(err));
        return -1;
    }
    err = camera_start_viewfinder(c->handle, vf_callback, status_callback, c);
    if (err != CAMERA_EOK) {
        fprintf(stderr, "[stereo_capture] camera_start_viewfinder on unit %d failed: %d (%s)\n",
                (int)c->unit, err, strerror(err));
        return -1;
    }
    c->started = 1;
    fprintf(stderr, "[stereo_capture] sensor unit %d streaming (viewfinder mode %d)\n", (int)c->unit, mode);
    return 0;
}

static void stop_camera(cam_t *c)
{
    if (c->started) camera_stop_viewfinder(c->handle);
    if (c->open) camera_close(c->handle);
    c->started = c->open = 0;
}

static void usage(void)
{
    fprintf(stderr, "usage: stereo_capture [-l] [-p] [-n 1|2] [-u LEFT,RIGHT] [-o /dev/shmem/dodge_stereo] [-s slot_MB]\n"
                    "       -f ycbycr|cbycry|ycrycb|crycby|nv12|rgb8888|bgr8888|rgb888|gray8|auto  (default ycbycr)\n");
}

/* -p: try camera_open on sensor units 1..8 directly, independent of the supported-cameras list */
static void probe_units(void)
{
    int u;
    for (u = 1; u <= 8; u++) {
        camera_handle_t h = -1;
        camera_error_t err = camera_open((camera_unit_t)u, CAMERA_MODE_RW, &h);
        if (err == CAMERA_EOK) {
            fprintf(stderr, "[stereo_capture] probe: sensor unit %d opens OK (read/write)\n", u);
            camera_close(h);
        } else {
            camera_error_t err2 = camera_open((camera_unit_t)u, 0x05 /* CAMERA_MODE_RO */, &h);
            if (err2 == CAMERA_EOK) {
                fprintf(stderr, "[stereo_capture] probe: sensor unit %d opens READ-ONLY only (rw err %d %s)\n",
                        u, err, strerror(err));
                camera_close(h);
            } else {
                fprintf(stderr, "[stereo_capture] probe: sensor unit %d: rw err %d (%s), ro err %d (%s)\n",
                        u, err, strerror(err), err2, strerror(err2));
            }
        }
    }
}

int main(int argc, char **argv)
{
    const char *path = "/dev/shmem/dodge_stereo";
    int units1[NCAMS] = {1, 2};      /* sensor units (1-based) */
    unsigned int slot_mb = 8;
    int list_only = 0, probe = 0, opt, i, fd;
    int ncams = NCAMS;               /* -n 1: one camera (mono) */
    size_t total;
    shm_header_t *hdr;
    uint8_t *base;
    cam_t cams[NCAMS];
    camera_unit_t avail[64];
    int navail;
    int64_t t_report;
    uint64_t last_frames[NCAMS] = {0, 0};

    while ((opt = getopt(argc, argv, "lpn:u:o:s:f:h")) != -1) {
        switch (opt) {
        case 'l': list_only = 1; break;
        case 'n': ncams = atoi(optarg); if (ncams < 1) ncams = 1; if (ncams > NCAMS) ncams = NCAMS; break;
        case 'p': probe = 1; break;
        case 'f': {
            static const struct { const char *n; int ft; } fm[] = {
                {"ycbycr", FT_YCBYCR}, {"yuyv", FT_YCBYCR}, {"cbycry", FT_CBYCRY}, {"uyvy", FT_CBYCRY},
                {"ycrycb", FT_YCRYCB}, {"crycby", FT_CRYCBY}, {"nv12", FT_NV12}, {"nv21", FT_NV21},
                {"rgb8888", FT_RGB8888}, {"bgr8888", FT_BGR8888}, {"rgb888", FT_RGB888}, {"gray8", FT_GRAY8}};
            size_t k;
            if (strcmp(optarg, "auto") == 0) { g_trust_type = 1; break; }
            for (k = 0; k < sizeof fm / sizeof fm[0]; k++) if (strcmp(optarg, fm[k].n) == 0) g_format = fm[k].ft;
            break;
        }
        case 'u': if (sscanf(optarg, "%d,%d", &units1[0], &units1[1]) < 1) { usage(); return 2; } break;
        case 'o': path = optarg; break;
        case 's': slot_mb = (unsigned int)atoi(optarg); break;
        default: usage(); return 2;
        }
    }
    if (probe) { probe_units(); return 0; }
    navail = list_units(avail, 64);
    if (navail < 0) {
        fprintf(stderr, "[stereo_capture] camera_get_supported_cameras failed (is the sensor service running?)\n");
        return 1;
    }
    fprintf(stderr, "[stereo_capture] camera units available:");
    for (i = 0; i < navail; i++) fprintf(stderr, " %d", (int)avail[i]);
    fprintf(stderr, "\n");
    if (list_only) return 0;
    if (slot_mb < 1 || slot_mb > 64) slot_mb = 8;

    total = HDR_SIZE + (size_t)NCAMS * slot_mb * 1024u * 1024u;
    fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { fprintf(stderr, "[stereo_capture] open %s: %s\n", path, strerror(errno)); return 1; }
    if (ftruncate(fd, (off_t)total) != 0) { fprintf(stderr, "[stereo_capture] ftruncate: %s\n", strerror(errno)); return 1; }
    base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { fprintf(stderr, "[stereo_capture] mmap: %s\n", strerror(errno)); return 1; }
    memset(base, 0, HDR_SIZE);
    hdr = (shm_header_t *)base;
    hdr->version = VERSION;
    hdr->ncams = (uint32_t)ncams;
    hdr->slot_size = slot_mb * 1024u * 1024u;
    hdr->pid = (uint64_t)getpid();
    hdr->heartbeat_us = now_us();
    for (i = 0; i < ncams; i++) hdr->cam[i].unit = (uint32_t)units1[i];
    __atomic_store_n(&hdr->magic, MAGIC, __ATOMIC_RELEASE);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    memset(cams, 0, sizeof cams);
    for (i = 0; i < ncams; i++) {
        cams[i].index = i;
        cams[i].desc_off = -1;
        cams[i].buf_off = -1;
        cams[i].unit = (camera_unit_t)units1[i];
        cams[i].hdr = hdr;
        cams[i].slot_size = hdr->slot_size;
        cams[i].slot = base + HDR_SIZE + (size_t)i * hdr->slot_size;
        if (start_camera(&cams[i]) != 0) {
            for (i = 0; i < ncams; i++) stop_camera(&cams[i]);
            munmap(base, total); close(fd); unlink(path);
            return 1;
        }
    }
    if (ncams == 1)
        fprintf(stderr, "[stereo_capture] pid %d publishing to %s (mono: unit %d)\n", (int)getpid(), path, units1[0]);
    else
        fprintf(stderr, "[stereo_capture] pid %d publishing to %s (left = unit %d, right = unit %d)\n",
                (int)getpid(), path, units1[0], units1[1]);

    t_report = now_us();
    while (!g_stop) {
        usleep(100000);
        __atomic_store_n(&hdr->heartbeat_us, now_us(), __ATOMIC_RELEASE);
        if (now_us() - t_report >= 5000000) {
            double dt = (now_us() - t_report) / 1e6;
            t_report = now_us();
            for (i = 0; i < ncams; i++) {
                cam_block_t *b = &hdr->cam[i];
                uint64_t f = __atomic_load_n(&b->frames, __ATOMIC_RELAXED);
                fprintf(stderr, "[stereo_capture] unit %d: %.1f fps %s %ux%u stride %u, bad %llu%s%s\n",
                        units1[i], (f - last_frames[i]) / dt, ft_name((int)b->frametype), b->width, b->height,
                        b->stride, (unsigned long long)hdr->bad[i], cams[i].last_bad[0] ? " last: " : "",
                        cams[i].last_bad);
                last_frames[i] = f;
            }
        }
    }
    fprintf(stderr, "[stereo_capture] stopping\n");
    for (i = 0; i < ncams; i++) stop_camera(&cams[i]);
    munmap(base, total);
    close(fd);
    unlink(path);
    return 0;
}
