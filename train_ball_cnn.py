"""Train the ball / not-ball CNN and export ball_cnn_int8.tflite (laptop).

Training patches come from the SAME pipeline perception uses on the Pi
(find_candidates -> crop_patch, 32x32x3), run on randomized synthetic scenes, plus procedural
non-ball shapes in the ball's colour (boxes, sheets, bars...), plus any real patches saved with
    python3 perception.py --source qnx --save-patches data/ball       (ball only in view)
    python3 perception.py --source qnx --save-patches data/not_ball   (distractors in the ball's colour, no ball)

    python train_ball_cnn.py [--scenes 60] [--epochs 15]
Output: ball_cnn_int8.tflite (int8 in, int8 out, class 0 = ball, 1 = not ball).
"""
import argparse
import glob
import os

import numpy as np

import config
import perception
import spacemath

HERE = os.path.dirname(os.path.abspath(__file__))


def synthetic_patches(n_scenes, frames, seed=100):
    X, y = [], []
    rng = np.random.default_rng(seed)
    for s in range(n_scenes):
        scene = perception.SyntheticScene(seed=seed + s, randomize=True)
        for _ in range(frames):
            has_ball = rng.random() > 0.15
            Z = rng.uniform(0.09, 0.65)
            half_w = Z * config.CX / config.FOCAL_PX
            half_h = Z * config.CY / config.FOCAL_PX
            p = np.array([rng.uniform(-half_w, half_w), rng.uniform(-half_h, half_h), Z])
            hand = rng.random() > 0.2
            L, R = scene.render(p if has_ball else None, hand=hand, rng=rng)
            for img, cam_x in ((L, -config.BASELINE_M / 2), (R, config.BASELINE_M / 2)):
                uv = spacemath.project(p, cam_x)
                r_true = config.FOCAL_PX * config.BALL_DIAM_M / 2 / Z
                for c in perception.find_candidates(img):
                    is_ball = (has_ball and math_dist(c, uv) < 0.6 * r_true and c["r"] > 0.5 * r_true)
                    X.append(perception.crop_patch(img, c))
                    y.append(0 if is_ball else 1)
        print("scene %d/%d: %d patches (%d ball)" % (s + 1, n_scenes, len(y), y.count(0)), flush=True)
    return X, y


def math_dist(c, uv):
    return ((c["u"] - uv[0]) ** 2 + (c["v"] - uv[1]) ** 2) ** 0.5


def shape_negatives(n, seed=7):
    """Ball-coloured things that are NOT a ball: boxes, bars, triangles, elongated blobs, paper with text."""
    rng = np.random.default_rng(seed)
    out = []
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float32)
    for i in range(n):
        bg = rng.uniform(60, 200, 3)[None, None, :] * (0.8 + 0.4 * rng.random((32, 32, 1)))
        img = bg.copy()
        if config.BALL_COLOR == "black":
            k0 = rng.uniform(5, 70)
            col = np.array([k0, k0 + rng.uniform(-5, 5), k0 + rng.uniform(-4, 8)])
            bg = rng.uniform(90, 220, 3)[None, None, :] * (0.8 + 0.4 * rng.random((32, 32, 1)))
            img = bg.copy()
        elif config.BALL_COLOR == "white":
            w0 = rng.uniform(170, 255)
            col = np.array([w0, w0 + rng.uniform(-10, 10), w0 + rng.uniform(-20, 8)])
        else:
            col = np.array([rng.uniform(200, 255), rng.uniform(80, 140), rng.uniform(5, 45)])
        kind = i % 6
        cx, cy = rng.uniform(12, 20, 2)
        if kind == 0:   # box / square
            a, b = rng.uniform(8, 15, 2)
            m = (np.abs(xx - cx) < a) & (np.abs(yy - cy) < b)
        elif kind == 1:  # bar
            m = (np.abs(xx - cx) < rng.uniform(2, 5)) | (np.abs(yy - cy) < rng.uniform(2, 5))
        elif kind == 2:  # triangle
            m = (yy > cy - 10) & (np.abs(xx - cx) < (yy - (cy - 10)) * 0.8) & (yy < cy + 10)
        elif kind == 3:  # elongated ellipse
            a = rng.uniform(12, 16); b = rng.uniform(3, 6); th = rng.uniform(0, np.pi)
            xr = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
            yr = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
            m = (xr / a) ** 2 + (yr / b) ** 2 < 1
        elif kind == 4:  # flat (unshaded) full patch
            m = np.ones((32, 32), bool)
        else:            # sheet of paper with text lines
            m = (np.abs(xx - cx) < rng.uniform(9, 15)) & (np.abs(yy - cy) < rng.uniform(9, 15))
        img[m] = col * rng.uniform(0.85, 1.05)
        if kind == 5:
            for r0 in range(int(cy) - 8, int(cy) + 9, 4):
                img[r0:r0 + 1, int(cx) - 8:int(cx) + rng.integers(0, 8)] = (
                    rng.uniform(150, 230) if config.BALL_COLOR == "black" else rng.uniform(20, 80))
        img += rng.normal(0, 3, img.shape)
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def real_patches():
    X, y = [], []
    for label, sub in ((0, "ball"), (1, "not_ball")):
        for f in glob.glob(os.path.join(HERE, "data", sub, "*.npy")):
            a = np.load(f)
            if a.shape == (32, 32, 3):
                X.append(a.astype(np.uint8)); y.append(label)
        pngs = glob.glob(os.path.join(HERE, "data", sub, "*.png"))
        if pngs:
            import cv2
            for f in pngs:
                a = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                X.append(cv2.resize(a, (32, 32), interpolation=cv2.INTER_NEAREST)); y.append(label)
    if X:
        print("real patches: %d ball, %d not ball" % (y.count(0), y.count(1)))
    return X, y


def build_model(tf):
    k = tf.keras
    return k.Sequential([
        k.layers.Input((32, 32, 3)),
        k.layers.Conv2D(8, 3, padding="same", activation="relu"),
        k.layers.MaxPooling2D(),
        k.layers.Conv2D(16, 3, padding="same", activation="relu"),
        k.layers.MaxPooling2D(),
        k.layers.Conv2D(32, 3, padding="same", activation="relu"),
        k.layers.MaxPooling2D(),
        k.layers.Flatten(),
        k.layers.Dense(32, activation="relu"),
        k.layers.Dropout(0.2),
        k.layers.Dense(2, activation="softmax"),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(HERE, config.MODEL_PATH))
    args = ap.parse_args()

    import tensorflow as tf
    tf.random.set_seed(0)

    X, y = synthetic_patches(args.scenes, args.frames)
    neg = shape_negatives(max(1000, len(X) // 4))
    X += neg; y += [1] * len(neg)
    rx, ry = real_patches()
    X += rx * 3; y += ry * 3       # weight real data
    X = np.stack(X); y = np.array(y)
    # augmentation: flips, brightness, colour cast (webcam white balance), blur (webcam focus)
    rng = np.random.default_rng(0)
    Xf = X[:, :, ::-1]
    Xb = np.clip(X.astype(np.float32) * rng.uniform(0.7, 1.25, (len(X), 1, 1, 1)), 0, 255).astype(np.uint8)
    Xc = np.clip(X.astype(np.float32) * rng.uniform(0.85, 1.15, (len(X), 1, 1, 3)), 0, 255).astype(np.uint8)
    Xs = X.astype(np.float32)
    Xs = (Xs + np.roll(Xs, 1, 1) + np.roll(Xs, -1, 1) + np.roll(Xs, 1, 2) + np.roll(Xs, -1, 2)) / 5.0
    Xs = np.clip(Xs + rng.normal(0, 3, Xs.shape), 0, 255).astype(np.uint8)
    X = np.concatenate([X, Xf, Xb, Xc, Xs]); y = np.concatenate([y, y, y, y, y])
    idx = rng.permutation(len(X)); X, y = X[idx], y[idx]
    n_test = len(X) // 7
    Xte, yte, Xtr, ytr = X[:n_test], y[:n_test], X[n_test:], y[n_test:]
    print("train %d (ball %d) test %d" % (len(Xtr), int((ytr == 0).sum()), len(Xte)))

    w0 = len(ytr) / (2.0 * max(1, (ytr == 0).sum())); w1 = len(ytr) / (2.0 * max(1, (ytr == 1).sum()))
    model = build_model(tf)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr.astype(np.float32) / 255.0, ytr, epochs=args.epochs, batch_size=64,
              validation_data=(Xte.astype(np.float32) / 255.0, yte), class_weight={0: w0, 1: w1}, verbose=2)

    def rep():
        for i in range(0, min(len(Xtr), 500)):
            yield [Xtr[i:i + 1].astype(np.float32) / 255.0]

    import tempfile
    saved = os.path.join(tempfile.mkdtemp(prefix="ballcnn_"), "saved")
    if hasattr(model, "export"):
        model.export(saved)          # Keras 3: frozen SavedModel with a serving signature
    else:
        tf.saved_model.save(model, saved)
    conv = tf.lite.TFLiteConverter.from_saved_model(saved)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    with open(args.out, "wb") as fh:
        fh.write(conv.convert())
    print("wrote %s (%d bytes)" % (args.out, os.path.getsize(args.out)))

    # evaluate the exported int8 model through the exact Pi code path
    clf = perception.BallClassifier(args.out)
    dummy = {"fill": 0.78, "aspect": 1.0}
    pred = np.array([0 if clf.p_ball(p, dummy) >= config.CNN_THRESHOLD else 1 for p in Xte])
    acc = float((pred == yte).mean())
    fp = int(((pred == 0) & (yte == 1)).sum()); fn = int(((pred == 1) & (yte == 0)).sum())
    print("int8 tflite test accuracy %.2f%% (false ball %d, missed ball %d of %d)" % (acc * 100, fp, fn, len(yte)))


if __name__ == "__main__":
    main()
