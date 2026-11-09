"""Sanity checks with a known heart rate.

1. Signal level: synthetic skin RGB traces with a pulse that ramps 70 -> 110 bpm,
   a strong 2nd harmonic, noise, and motion-like intensity drift.
2. End to end: a real face frame held still for 30 s with a 75 bpm pulse
   injected into the skin pixels, run through landmarks, patches and fusion.
"""
import cv2
import numpy as np

from fv import extract, face, hr, metrics, pipeline, signals

rng = np.random.default_rng(1)

# ---------- 1. signal level ----------
fs, T = 30.0, 120.0
t = np.arange(0, T, 1 / fs)
hr_true = np.interp(t, [0, 40, 80, 120], [70, 70, 110, 100])
phase = 2 * np.pi * np.cumsum(hr_true / 60) / fs
pulse = np.sin(phase) + 0.6 * np.sin(2 * phase + 0.5)  # strong harmonic
pvec = np.array([0.33, 0.77, 0.53])  # blood-volume pulse signature in RGB
base = np.array([150.0, 105.0, 85.0])
# Head-motion-like brightness changes (all channels) and specular glare
# (adds white light) in a 0.8-2 Hz band that overlaps the HR band.
motion = 0.02 * np.sin(2 * np.pi * 1.3 * t + 0.4 * np.sin(2 * np.pi * 0.1 * t))
glare = 3.0 * np.clip(np.sin(2 * np.pi * 0.9 * t), 0, None)
drift = 1 + 0.03 * np.sin(2 * np.pi * 0.15 * t) + motion
rgb = (base * drift[:, None] * (1 + 0.003 * pulse[:, None] * pvec) + glare[:, None]
       + rng.normal(0, 0.15, (len(t), 3)))

ends = hr.windows(T, 10, 1)
ref = np.interp(ends - 5, t, hr_true)
print("Signal-level test (ramp 70->110 bpm, strong 2nd harmonic, drift + noise)")
for name, fn in signals.METHODS.items():
    S, ok = hr.window_spectra(t, fn(rgb, fs), fs, ends, 10)
    pk = metrics.hr_metrics(hr.peak_hr(S), ref)
    vt = metrics.hr_metrics(hr.viterbi(S, ok, 1.0), ref)
    print(f"  {name:6s} peak MAE {pk['mae']:5.2f}  |  +Viterbi MAE {vt['mae']:5.2f}")

# ---------- 2. end to end with pulse injection ----------
cap = cv2.VideoCapture("data/mcd/video/1020_FullHDwebcam_before.avi")
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
_, frame = cap.read()
lm = face.make_landmarker(1)
pts = face.detect_faces(lm, frame, 0)[0]
skin = np.zeros(frame.shape[:2], np.uint8)
cv2.fillPoly(skin, [np.round(pts[face.FACE_OVAL]).astype(np.int32)], 1)
skin = skin.astype(bool)

HR_INJ = 75.0
n = int(30 * fs)
vw = cv2.VideoWriter("out/synthetic_75bpm.avi", cv2.VideoWriter_fourcc(*"FFV1"), fs, frame.shape[1::-1])
for i in range(n):
    p = np.sin(2 * np.pi * HR_INJ / 60 * i / fs)
    f = frame.astype(np.float32)
    f[skin] *= (1 + 0.004 * p * pvec[::-1])  # BGR order
    f += rng.normal(0, 1.0, f.shape)
    vw.write(np.clip(f, 0, 255).astype(np.uint8))
vw.release()

tr = extract.extract("out/synthetic_75bpm.avi")
res = pipeline.estimate(tr, "proposed")
print(f"\nEnd-to-end pulse injection: true {HR_INJ:.0f} bpm -> estimated "
      f"{np.median(res['hr']):.1f} bpm (median over {len(res['hr'])} windows), "
      f"SNR {np.nanmedian(res['snr_db']):.1f} dB, {tr['ms_per_frame']:.1f} ms/frame face+patch cost")
