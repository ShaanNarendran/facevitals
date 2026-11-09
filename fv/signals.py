"""Colour-trace conditioning and classical pulse extraction (GREEN, CHROM, POS)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

HR_BAND_HZ = (0.66, 3.5)  # ~40-210 bpm


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float, max_gap_s: float = 1.0):
    """Resample (N, C) samples at timestamps t onto a uniform grid at fs.

    Rows that are NaN (no face) are interpolated across if the gap is short;
    longer gaps stay NaN so downstream windows can be rejected.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    out = np.full((len(grid), x.shape[1]), np.nan)
    for c in range(x.shape[1]):
        ok = np.isfinite(x[:, c])
        if ok.sum() < 2:
            continue
        out[:, c] = np.interp(grid, t[ok], x[ok, c])
        # Re-blank long gaps.
        tok = t[ok]
        idx = np.searchsorted(tok, grid)
        idx = np.clip(idx, 1, len(tok) - 1)
        gap = tok[idx] - tok[idx - 1]
        out[gap > max_gap_s, c] = np.nan
    return grid, out


def bandpass(x: np.ndarray, fs: float, band=HR_BAND_HZ, order: int = 3) -> np.ndarray:
    sos = butter(order, [band[0] / (fs / 2), min(band[1] / (fs / 2), 0.99)], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _fill_nan(rgb: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    for c in range(out.shape[1]):
        ok = np.isfinite(out[:, c])
        if ok.any() and not ok.all():
            out[~ok, c] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), out[ok, c])
    return out


def green(rgb: np.ndarray, fs: float) -> np.ndarray:
    g = _fill_nan(rgb)[:, 1]
    return bandpass(g / np.mean(g) - 1.0, fs)


def pos(rgb: np.ndarray, fs: float, win_s: float = 1.6) -> np.ndarray:
    """Plane-Orthogonal-to-Skin (Wang et al., 2017), overlap-add form (vectorised)."""
    rgb = _fill_nan(rgb)
    n = len(rgb)
    l = max(int(round(win_s * fs)), 2)
    if n < l:
        return np.zeros(n)
    win = np.lib.stride_tricks.sliding_window_view(rgb, l, axis=0)  # (n-l+1, 3, l)
    cn = win / (win.mean(axis=2, keepdims=True) + 1e-9)
    s1 = cn[:, 1] - cn[:, 2]
    s2 = -2 * cn[:, 0] + cn[:, 1] + cn[:, 2]
    a = s1.std(axis=1, keepdims=True) / (s2.std(axis=1, keepdims=True) + 1e-9)
    p = s1 + a * s2
    p -= p.mean(axis=1, keepdims=True)
    h = np.zeros(n)
    m = n - l + 1
    for k in range(l):  # overlap-add, one pass per in-window offset
        h[k:k + m] += p[:, k]
    return bandpass(h, fs)


def chrom(rgb: np.ndarray, fs: float, win_s: float = 1.6) -> np.ndarray:
    """Chrominance method (de Haan & Jeanne, 2013), Hann overlap-add."""
    rgb = _fill_nan(rgb)
    n = len(rgb)
    l = max(int(round(win_s * fs)), 4)
    l += l % 2
    hop = l // 2
    win = np.hanning(l)
    out = np.zeros(n)
    # Normalise by a local mean, then band-limit the two chrominance signals
    # before the per-window alpha tuning, as in the original method.
    k = max(int(fs), 1)
    local = np.apply_along_axis(lambda v: np.convolve(v, np.ones(k) / k, mode="same"), 0, rgb)
    cn = rgb / (local + 1e-9)
    x = bandpass(3 * cn[:, 0] - 2 * cn[:, 1], fs)
    y = bandpass(1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2], fs)
    for s in range(0, n - l + 1, hop):
        xf, yf = x[s:s + l], y[s:s + l]
        a = xf.std() / (yf.std() + 1e-9)
        out[s:s + l] += win * (xf - a * yf)
    return out


METHODS = {"green": green, "chrom": chrom, "pos": pos}
