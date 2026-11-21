"""Stereo calibration (laptop, OpenCV) -> calib.json + rectify_maps.npz for the Pi.

1. On the Pi:  python3 perception.py --source qnx --save-pairs pairs/   (show a checkerboard, move it)
2. Copy pairs/ to the laptop, then:
       python stereo_calibrate.py --pairs pairs/ --cols 9 --rows 6 --square 0.025
3. Copy calib.json and rectify_maps.npz next to perception.py on the Pi.

Self-test without cameras (renders checkerboard pairs with a known 6 cm baseline):
       python stereo_calibrate.py --synthetic 25 --out-dir /tmp/calibtest
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

import config

HERE = os.path.dirname(os.path.abspath(__file__))


def board_points(cols, rows, square):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return objp


def synthetic_pairs(n, cols, rows, square, baseline=config.BASELINE_M, seed=0):
    """Checkerboard rendered by a known pinhole stereo pair (for testing this script)."""
    rng = np.random.default_rng(seed)
    K = np.array([[config.FOCAL_PX, 0, config.CX], [0, config.FOCAL_PX, config.CY], [0, 0, 1]], float)
    px = 40                                 # texture px per square
    tw, th = (cols + 3) * px, (rows + 3) * px
    tex = np.full((th, tw), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0 = px + r * px; x0 = px + c * px
                tex[y0:y0 + px, x0:x0 + px] = 0
    # texture px -> board metres; inner corner (0,0) is at texture (2px, 2px)
    S = np.array([[square / px, 0, -2 * square], [0, square / px, -2 * square], [0, 0, 1]], float)
    pairs = []
    while len(pairs) < n:
        rv = rng.uniform(-0.45, 0.45, 3); rv[2] = rng.uniform(-0.3, 0.3)
        R, _ = cv2.Rodrigues(rv)
        cw, ch = (cols - 1) * square / 2, (rows - 1) * square / 2
        Z = rng.uniform(0.35, 0.65)
        t = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.04, 0.04), Z]) - R @ np.array([cw, ch, 0])
        imgs = []
        for cam_x in (-baseline / 2, baseline / 2):
            tc = t - np.array([cam_x, 0, 0])
            H = K @ np.column_stack([R[:, 0], R[:, 1], tc]) @ S
            img = cv2.warpPerspective(tex, H, (config.IMG_W, config.IMG_H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=128)
            img = np.clip(img.astype(np.float32) + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        pairs.append((imgs[0], imgs[1]))
    return pairs


def load_pairs(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "pair_*.npz"))):
        z = np.load(f)
        out.append((z["left"], z["right"]))
    return out


def calibrate(pairs, cols, rows, square):
    objp = board_points(cols, rows, square)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    O, PL, PR = [], [], []
    size = None
    for left, right in pairs:
        gl = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY); gr = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        size = gl.shape[::-1]
        okl, cl = cv2.findChessboardCorners(gl, (cols, rows))
        okr, cr = cv2.findChessboardCorners(gr, (cols, rows))
        if not (okl and okr):
            continue
        cl = cv2.cornerSubPix(gl, cl, (5, 5), (-1, -1), crit)
        cr = cv2.cornerSubPix(gr, cr, (5, 5), (-1, -1), crit)
        O.append(objp); PL.append(cl); PR.append(cr)
    print("checkerboard found in %d / %d pairs" % (len(O), len(pairs)))
    if len(O) < 8:
        raise SystemExit("need at least 8 good pairs")
    _, K1, D1, _, _ = cv2.calibrateCamera(O, PL, size, None, None)
    _, K2, D2, _, _ = cv2.calibrateCamera(O, PR, size, None, None)
    rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        O, PL, PR, K1, D1, K2, D2, size, criteria=crit, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)
    m = {}
    for side, K, D, Rr, P in (("l", K1, D1, R1, P1), ("r", K2, D2, R2, P2)):
        mx, my = cv2.initUndistortRectifyMap(K, D, Rr, P, size, cv2.CV_32FC1)
        m["map_%sx" % side] = np.clip(np.round(mx), 0, size[0] - 1).astype(np.int16)
        m["map_%sy" % side] = np.clip(np.round(my), 0, size[1] - 1).astype(np.int16)
    calib = {"f": float(P1[0, 0]), "cx": float(P1[0, 2]), "cy": float(P1[1, 2]),
             "baseline": float(np.linalg.norm(T)), "rms_px": float(rms), "width": size[0], "height": size[1],
             "pairs_used": len(O)}
    return calib, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", help="directory with pair_*.npz from perception.py --save-pairs")
    ap.add_argument("--synthetic", type=int, default=0, help="self-test with N rendered pairs")
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--square", type=float, default=0.025, help="square size in metres")
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()
    if args.synthetic:
        pairs = synthetic_pairs(args.synthetic, args.cols, args.rows, args.square)
    elif args.pairs:
        pairs = load_pairs(args.pairs)
    else:
        raise SystemExit("give --pairs DIR or --synthetic N")
    calib, maps = calibrate(pairs, args.cols, args.rows, args.square)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, config.CALIB_PATH), "w") as fh:
        json.dump(calib, fh, indent=2)
    np.savez_compressed(os.path.join(args.out_dir, config.RECTIFY_PATH), **maps)
    print("baseline %.2f cm  f %.1f px  c (%.1f, %.1f)  rms %.3f px" % (
        calib["baseline"] * 100, calib["f"], calib["cx"], calib["cy"], calib["rms_px"]))
    print("wrote %s and %s in %s" % (config.CALIB_PATH, config.RECTIFY_PATH, args.out_dir))


if __name__ == "__main__":
    main()
