"""Learned pulse branch: a small temporal CNN on region colour traces.

Input (per 10 s window at 30 Hz): band-passed, mean-normalised RGB of the
whole face, 6 skin patches and the background, plus whole-face POS, CHROM and
GREEN. Output: a pulse waveform, whose spectrum joins the classical spectra in
the same SNR fusion and tracker. Trained with a spectral loss against the
contact-PPG heart rate, which ignores the face-to-finger pulse delay.

Kept tiny (~25k parameters) so it runs on a Raspberry Pi in real time.
"""
from __future__ import annotations

import numpy as np

from . import face, hr, signals

FS = 30.0
WIN = 300  # 10 s at 30 Hz
REGIONS = list(face.PATCHES) + ["face", "bg"]
N_IN = 3 * len(REGIONS) + 3


def features(rgb_u: np.ndarray, regions: list[str], fs: float = FS) -> np.ndarray:
    """(N, R, 3) uniformly resampled traces -> (N_IN, N) network inputs.

    Each region's RGB is divided by its own mean, band-passed and scaled by
    the region's overall pulse-band std, which keeps the ratios between its
    colour channels (what POS relies on) but equalises regions. Missing
    regions are zeros. The three classical pulse signals are z-normalised.
    """
    n = len(rgb_u)
    out = np.zeros((N_IN, n), np.float32)
    for j, r in enumerate(REGIONS):
        x = rgb_u[:, regions.index(r)]
        if np.isfinite(x).mean() < 0.5:
            continue
        x = signals._fill_nan(x)
        y = signals.bandpass(x / x.mean(axis=0) - 1.0, fs)
        out[3 * j:3 * j + 3] = (y / (np.sqrt((y ** 2).mean()) + 1e-9)).T
    fx = rgb_u[:, regions.index("face")]
    if np.isfinite(fx).mean() >= 0.5:
        for k, m in enumerate(("pos", "chrom", "green")):
            p = signals.METHODS[m](fx, fs)
            out[3 * len(REGIONS) + k] = p / (p.std() + 1e-9)
    return np.nan_to_num(out)


def _torch():
    import torch
    return torch


def build_model(width: int = 32):
    torch = _torch()
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.conv = nn.Conv1d(width, width, 5, dilation=d, padding=2 * d)
            self.bn = nn.BatchNorm1d(width)

        def forward(self, x):
            return x + nn.functional.elu(self.bn(self.conv(x)))

    class PulseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Conv1d(N_IN, width, 1), nn.BatchNorm1d(width), nn.ELU())
            self.blocks = nn.Sequential(*[Block(d) for d in (1, 2, 4, 8, 16)])
            self.drop = nn.Dropout(0.1)
            self.out = nn.Conv1d(width, 1, 1)

        def forward(self, x):  # (B, N_IN, T) -> (B, T)
            return self.out(self.drop(self.blocks(self.inp(x))))[:, 0]

    return PulseNet()


def spectral_basis(n: int = WIN, fs: float = FS):
    """Hann-tapered DFT rows at the HR grid, as (cos, sin) float32 tensors."""
    torch = _torch()
    t = np.arange(n) / fs
    w = np.hanning(n)
    ang = 2 * np.pi * np.outer(hr.BPM / 60.0, t)
    return (torch.tensor(np.cos(ang) * w, dtype=torch.float32),
            torch.tensor(np.sin(ang) * w, dtype=torch.float32))


def power(y, basis):
    """(B, T) waveforms -> (B, len(BPM)) normalised power spectra."""
    c, s = basis
    y = y - y.mean(dim=1, keepdim=True)
    p = (y @ c.T) ** 2 + (y @ s.T) ** 2
    return p / (p.sum(dim=1, keepdim=True) + 1e-9)


def target(hr_bpm, sigma: float = 2.5):
    torch = _torch()
    bpm = torch.tensor(hr.BPM, dtype=torch.float32, device=hr_bpm.device)
    g = torch.exp(-0.5 * ((bpm[None] - hr_bpm[:, None]) / sigma) ** 2)
    return g / g.sum(dim=1, keepdim=True)


def loss_fn(y, hr_bpm, basis):
    torch = _torch()
    return -(target(hr_bpm) * torch.log(power(y, basis) + 1e-6)).sum(dim=1).mean()


class Branch:
    """Loaded model(s); spectra() gives per-window spectra like hr.window_spectra."""

    def __init__(self, paths: list[str]):
        torch = _torch()
        torch.set_num_threads(2)
        self.models = []
        for p in paths:
            m = build_model()
            m.load_state_dict(torch.load(p, map_location="cpu"))
            m.eval()
            self.models.append(m)

    def spectra(self, grid: np.ndarray, rgb_u: np.ndarray, regions: list[str], ends: np.ndarray):
        torch = _torch()
        x = features(rgb_u, regions)
        nan = ~np.all(np.isfinite(rgb_u[:, regions.index("face")]), axis=1)
        idx_end = np.searchsorted(grid, ends, side="right")
        S = np.zeros((len(ends), len(hr.BPM)))
        ok = np.zeros(len(ends), bool)
        batch, rows = [], []
        for i, e in enumerate(idx_end):
            if e - WIN < 0 or nan[e - WIN:e].any():
                continue
            batch.append(x[:, e - WIN:e])
            rows.append(i)
        if not rows:
            return S, ok
        with torch.no_grad():
            xb = torch.tensor(np.stack(batch))
            ys = [m(xb).numpy() for m in self.models]
        for k, i in enumerate(rows):
            S[i] = np.mean([hr.spectrum(y[k], FS) for y in ys], axis=0)
            ok[i] = True
        return S, ok
