"""Constraint simulations (C1 low light, C4 low resolution, C5 bandwidth/fps).

C3 (multi-face) and C7 (mask) are applied inside extraction because they need
a second video or the face landmarks.
"""
from __future__ import annotations

import os
import subprocess

import cv2
import numpy as np


_TO_LINEAR = ((np.arange(256, dtype=np.float32) / 255.0) ** 2.2).astype(np.float32)
_TO_SRGB8 = np.round(((np.arange(4096, dtype=np.float32) / 4095.0) ** (1 / 2.2)) * 255).astype(np.uint8)


def low_light(frame_bgr: np.ndarray, rng: np.random.Generator, light_ratio: float,
              full_well_e: float = 3000.0, read_noise_e: float = 3.0) -> np.ndarray:
    """Physically motivated low-light simulation.

    Undo the display gamma, scale the scene light by `light_ratio`
    (e.g. 50/300 lux), draw Poisson photon shot noise and Gaussian read noise at
    the sensor, then apply the camera's auto-gain back to the original
    brightness and re-encode to 8-bit sRGB. This reproduces the amplified noise
    a real camera produces in dim light, rather than just a darker image.
    """
    lin = _TO_LINEAR[frame_bgr]
    electrons = lin * (light_ratio * full_well_e)
    # Shot noise (variance = mean, Gaussian approximation of Poisson) + read noise.
    sigma = np.sqrt(electrons + read_noise_e ** 2)
    noisy = electrons + sigma * rng.standard_normal(lin.shape, dtype=np.float32)
    gain = 1.0 / (light_ratio * full_well_e)
    code = np.clip(noisy * gain * 4095, 0, 4095).astype(np.int32)
    return _TO_SRGB8[code]


def low_res(frame_bgr: np.ndarray, size=(320, 240)) -> np.ndarray:
    return cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)


def reencode(src: str, dst: str, fps: float | None = None, kbps: int | None = None,
             size: tuple[int, int] | None = None, crop: tuple[int, int, int, int] | None = None,
             seconds: float | None = None) -> str:
    """Re-encode with ffmpeg (H.264) to emulate low frame rate / bitrate links.

    crop = (w, h, x, y) encodes only that region (ROI-aware streaming).
    """
    if os.path.exists(dst):
        return dst
    vf = [f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}"] if crop else []
    if fps:
        vf.append(f"fps={fps}")
    if size:
        vf.append(f"scale={size[0]}:{size[1]}:flags=bicubic")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src]
    if seconds:
        cmd += ["-t", str(seconds)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p", "-threads", "2"]
    if kbps:
        cmd += ["-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{2 * kbps}k"]
    else:
        cmd += ["-crf", "18"]
    tmp = dst + ".part.mp4"
    cmd += ["-an", tmp]
    subprocess.run(cmd, check=True)
    os.replace(tmp, dst)
    return dst


def encode_frames(video: str, times: np.ndarray, dst: str, fps: float | None = None, kbps: int | None = None,
                  crop: tuple[int, int, int, int] | None = None, seconds: float | None = None,
                  bin_frames: bool = False, zerolatency: bool = False) -> str:
    """Frame-rate / bitrate degradation that keeps true capture times.

    A lower frame rate keeps every Nth captured frame (N = capture rate / fps,
    rounded up so the limit is never exceeded), as a device sender would.
    Picking the frame nearest to each 1/fps tick instead gives an uneven
    2/3/4-frame pattern whenever the stamps are jittery, and that pattern turns
    100 Hz light flicker into a strong in-band artefact. With bin_frames, each
    sent frame is the average of the N frames since the previous one (a 1/fps
    exposure; at 30 -> 10 fps this also cancels the 10 Hz flicker alias), and
    its time is their mean time. The chosen capture times are saved next to
    the file as <dst>.t.npy and are what the extractor uses as the time axis.

    zerolatency uses x264's live-streaming tuning (no B-frames, no look-ahead).
    Default x264 sends P B B B P ...; B-frames are quantised more coarsely, so
    the face colour steps with that 4-frame cycle, which at 10 fps is 2.5 Hz
    (150 bpm), inside the heart-rate band.
    """
    if os.path.exists(dst) and os.path.exists(dst + ".t.npy"):
        return dst
    end = times[-1] if seconds is None else min(times[-1], seconds)
    if fps:
        rate = 1.0 / np.median(np.diff(times[:300]))
        step = max(1, int(np.ceil(rate / fps - 0.05)))
        pick = np.arange(step - 1 if bin_frames else 0, int(np.searchsorted(times, end, side="right")), step)
    else:
        pick = np.flatnonzero(times <= end)
    cap = cv2.VideoCapture(video)
    ok, fr = cap.read()
    h, w = fr.shape[:2]
    if crop:
        w, h = crop[0], crop[1]
    tmp = dst + ".part.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(fps or 30), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
           "-threads", "2"]
    if zerolatency:
        cmd += ["-tune", "zerolatency"]
    cmd += ["-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{2 * kbps}k"] if kbps else ["-crf", "18"]
    proc = subprocess.Popen(cmd + ["-an", tmp], stdin=subprocess.PIPE)
    want = set(pick.tolist())
    i, written = 0, []
    acc, acc_t = None, []
    while ok and i <= pick[-1]:
        f = fr[crop[3]:crop[3] + crop[1], crop[2]:crop[2] + crop[0]] if crop else fr
        if bin_frames:
            acc = f.astype(np.float32) if acc is None else acc + f
            acc_t.append(times[i])
        if i in want:
            if bin_frames:
                out = np.round(acc / len(acc_t)).astype(np.uint8)
                written.append(float(np.mean(acc_t)))
                acc, acc_t = None, []
            else:
                out = f
                written.append(times[i])
            proc.stdin.write(np.ascontiguousarray(out).tobytes())
        ok, fr = cap.read()
        i += 1
    proc.stdin.close()
    proc.wait()
    np.save(dst + ".t.npy", np.array(written))
    os.replace(tmp, dst)
    return dst


def frame_types(path: str) -> np.ndarray:
    """Picture type of every decoded frame ('I', 'P' or 'B'), from ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pict_type",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout
    return np.array([c for c in out.replace(",", " ").split() if c in ("I", "P", "B")])
