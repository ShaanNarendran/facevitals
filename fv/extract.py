"""Video -> per-frame timestamps and per-patch mean RGB traces for the target face."""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from . import degrade, face


def load_stamps(meta_path: str | None) -> np.ndarray | None:
    """All per-frame capture times (s from first frame) from an MCD-rPPG meta file."""
    if not (meta_path and os.path.exists(meta_path)):
        return None
    return smooth_stamps(read_timestamps(meta_path, 0, 30.0))


def smooth_stamps(t: np.ndarray, win: int = 31) -> np.ndarray:
    """Remove logging-timer quantisation from frame times.

    MCD-rPPG stamps come in ~16 ms steps (gaps of 0.032 / 0.033 / 0.048 s
    around a steady 30 fps), so each is up to ~8 ms off. The deviation from a
    constant frame period is averaged over ~1 s, which keeps real drift and
    dropped frames but removes the timer jitter.
    """
    if len(t) < win:
        return t
    k = np.arange(len(t))
    period = (t[-1] - t[0]) / (len(t) - 1)
    r = t - k * period
    rs = np.convolve(np.pad(r, win // 2, mode="edge"), np.ones(win) / win, mode="valid")
    out = k * period + rs
    return out - out[0]


def read_timestamps(meta_path: str | None, n: int, fps: float) -> np.ndarray:
    """Per-frame capture times (s). MCD-rPPG meta files hold wall-clock stamps.

    n = 0 returns every stamp in the file.
    """
    if meta_path and os.path.exists(meta_path):
        from datetime import datetime

        ts = []
        with open(meta_path) as f:
            for line in f:
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    s = parts[1].strip()
                    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in s else "%Y-%m-%d %H:%M:%S"
                    ts.append(datetime.strptime(s, fmt).timestamp())
        ts = np.array(ts)
        if n == 0:
            return ts - ts[0]
        if len(ts) >= n:
            return ts[:n] - ts[0]
    return np.arange(n) / fps


def roi_crop(video: str, scale: float = 2.2) -> tuple[int, int, int, int]:
    """Square, even-sized crop around the face found in the first second."""
    cap = cv2.VideoCapture(video)
    fe = face.FaceFrontEnd()
    boxes = []
    for k in range(30):
        ok, fr = cap.read()
        if not ok:
            break
        b = fe.detect(fr)
        if b:
            boxes.append(max(b, key=lambda r: (r[2] - r[0]) * (r[3] - r[1])))
    h, w = fr.shape[:2]
    fe.close()
    b = np.median(boxes, axis=0)
    c = (b[:2] + b[2:]) / 2
    side = int(min(scale * max(b[2] - b[0], b[3] - b[1]), w, h)) // 2 * 2
    x = int(np.clip(c[0] - side / 2, 0, w - side)) // 2 * 2
    y = int(np.clip(c[1] - side / 2, 0, h - side)) // 2 * 2
    return side, side, x, y


def enc_params(condition: str) -> dict:
    """Encoder settings for a condition name.

    bandwidth = 10 fps + 500 kbps H.264 (C5 as in the brief). enc_* names
    combine tokens: fpsN, kbpsN, roi (stream a face-centred crop), bin
    (average the frames between sends), zl (x264 zerolatency: no B-frames).
    """
    if condition == "bandwidth":
        return dict(fps=10, kbps=500)
    enc = {}
    if condition.startswith("enc_"):
        for p in condition.split("_")[1:]:
            if p in ("roi", "bin", "zl"):
                enc[p] = 1
            elif p.startswith("fps"):
                enc["fps"] = int(p[3:])
            elif p.startswith("kbps"):
                enc["kbps"] = int(p[4:])
    return enc


def encoded_path(video: str, enc: dict) -> str:
    base = os.path.splitext(os.path.basename(video))[0]
    tag = "_".join(f"{k}{v}" for k, v in sorted(enc.items()))
    return os.path.join("data", "degraded", f"{base}_{tag}_v2.mp4")


def extract(video: str, condition: str = "baseline", meta: str | None = None,
            bystander_video: str | None = None, max_seconds: float | None = None,
            seed: int = 0, landmark_every: int | None = None, keep_crops: int = 0,
            front_hz: float = 10.0, stamps: np.ndarray | None = None) -> dict:
    """Run face tracking + patch sampling on a (possibly degraded) video.

    condition: baseline | lowlight50 | lowlight20 | lowres | bandwidth | mask | multiface
    keep_crops: if > 0, also return the target face crop resized to this size
                (square, RGB) for every frame, plus landmarks, for submission files.
    front_hz:   face detection and landmark refresh rate in real time. The frame
                cadence follows the stream's rate (every 3rd frame at 30 fps,
                every frame at 10 fps), so a low-fps stream does not also get
                a 3x slower face mesh.
    stamps:     per-frame capture times (s) for videos without an MCD-rPPG meta
                file (e.g. UBFC-rPPG ground-truth timesteps); smoothed like
                meta stamps.
    """
    rng = np.random.default_rng(seed)
    src = video
    enc = enc_params(condition)
    stamps = smooth_stamps(np.asarray(stamps, float) - stamps[0]) if stamps is not None else load_stamps(meta)
    cap = cv2.VideoCapture(video)
    hdr_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if stamps is None:
        # No capture log: trust the container frame rate.
        n_hdr = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT)) or 100000
        stamps = np.arange(n_hdr) / hdr_fps
    if enc:
        os.makedirs(os.path.join("data", "degraded"), exist_ok=True)
        crop = roi_crop(video) if enc.get("roi") else None
        src = degrade.encode_frames(video, stamps, encoded_path(video, enc),
                                    fps=enc.get("fps"), kbps=enc.get("kbps"), crop=crop,
                                    seconds=(max_seconds + 1) if max_seconds else None,
                                    bin_frames=bool(enc.get("bin")), zerolatency=bool(enc.get("zl")))
        stamps = np.load(src + ".t.npy")

    cap = cv2.VideoCapture(src)
    fps = hdr_fps
    rate = 1.0 / np.median(np.diff(stamps[:300])) if len(stamps) > 1 else fps
    every = max(1, int(round(rate / front_hz)))
    landmark_every = landmark_every or every
    cap_b = cv2.VideoCapture(bystander_video) if condition == "multiface" else None

    front = face.FaceFrontEnd(landmark_every=landmark_every, detect_every=every)
    lm_clean = face.make_landmarker(num_faces=1) if condition == "mask" else None
    sampler = face.PatchSampler()
    mask_pts = None

    rgbs, counts, nfaces, found, on_target, frame_idx = [], [], [], [], [], []
    crops, boxes, landmarks = [], [], []
    t_proc = 0.0
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i >= len(stamps) or (max_seconds and stamps[i] > max_seconds):
            break
        t_ms = int(stamps[i] * 1000) + i  # strictly increasing for MediaPipe

        if condition == "lowlight50":
            fr = degrade.low_light(fr, rng, 50 / 300)
        elif condition == "lowlight20":
            fr = degrade.low_light(fr, rng, 20 / 300)
        elif condition == "lowres":
            fr = degrade.low_res(fr)
        elif condition == "mask":
            if i % every == 0:  # mask geometry from clean landmarks, refreshed at front_hz
                clean = face.detect_faces(lm_clean, fr, t_ms)
                mask_pts = clean[0] if clean else None
            if mask_pts is not None:
                fr = face.apply_mask_overlay(fr, mask_pts, rng)
        elif condition == "multiface":
            okb, frb = cap_b.read()
            if not okb:
                cap_b.set(cv2.CAP_PROP_POS_FRAMES, 0)
                okb, frb = cap_b.read()
            frb = cv2.resize(frb, (fr.shape[1], fr.shape[0]))
            fr = np.hstack([fr, frb])  # target on the left, bystander on the right
            if i == 0:
                front.tracker.select_point = (fr.shape[1] * 0.25, fr.shape[0] * 0.5)

        t0 = time.perf_counter()
        pts = front.process(fr, t_ms)
        if pts is not None:
            rgb, cnt = sampler.sample(fr, pts)
        else:
            rgb, cnt = np.full((len(face.ALL_REGIONS), 3), np.nan, np.float32), np.zeros(len(face.ALL_REGIONS), np.int32)
        t_proc += time.perf_counter() - t0

        rgbs.append(rgb)
        counts.append(cnt)
        nfaces.append(front.n_faces)
        found.append(pts is not None)
        if condition == "multiface":
            on_target.append(pts is not None and face.bbox(pts)[2] <= fr.shape[1] / 2 + 5)
        if keep_crops:
            k = keep_crops
            if pts is not None:
                b = face.bbox(pts)
                c = (b[:2] + b[2:]) / 2
                half = 0.6 * max(b[2] - b[0], b[3] - b[1])
                x0, y0 = (np.maximum(c - half, 0)).astype(int)
                x1, y1 = int(min(c[0] + half, fr.shape[1])), int(min(c[1] + half, fr.shape[0]))
                crops.append(cv2.cvtColor(cv2.resize(fr[y0:y1, x0:x1], (k, k), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
                boxes.append(b)
                landmarks.append(pts)
            else:
                crops.append(np.zeros((k, k, 3), np.uint8))
                boxes.append(np.full(4, np.nan))
                landmarks.append(np.full((468, 2), np.nan, np.float32))
        frame_idx.append(i)
        i += 1
    cap.release()
    front.close()

    n = len(rgbs)
    t = stamps[:n]
    ftype = degrade.frame_types(src)[:n]
    ftype = np.concatenate([ftype, np.full(n - len(ftype), "P")])
    out = dict(t=t, rgb=np.array(rgbs), counts=np.array(counts), nfaces=np.array(nfaces), found=np.array(found),
               fps=float(1 / np.median(np.diff(t))) if n > 1 else fps, regions=np.array(face.ALL_REGIONS), ms_per_frame=1000 * t_proc / max(n, 1),
               condition=condition, ftype=ftype)
    if on_target:
        out["on_target"] = np.array(on_target)
    if keep_crops:
        out.update(crops=np.array(crops), boxes=np.array(boxes), landmarks=np.array(landmarks))
    return out
