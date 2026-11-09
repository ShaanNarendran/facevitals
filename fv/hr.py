"""Heart-rate estimation: windowed spectra, SNR quality index, multi-patch
fusion, and harmonic-aware Viterbi tracking."""
from __future__ import annotations

import numpy as np

BPM = np.arange(40.0, 200.5, 1.0)  # state / spectrum grid


_E_CACHE: dict = {}


def _dft_matrix(n: int, fs: float, bpm: np.ndarray = BPM) -> np.ndarray:
    key = (n, fs, len(bpm))
    if key not in _E_CACHE:
        t = np.arange(n) / fs
        _E_CACHE[key] = np.exp(-2j * np.pi * np.outer(bpm / 60.0, t)) * np.hanning(n)
    return _E_CACHE[key]


def spectrum(x: np.ndarray, fs: float, bpm: np.ndarray = BPM) -> np.ndarray:
    """Power at each bpm bin via a direct DFT with a Hann taper."""
    x = np.asarray(x, float)
    return np.abs(_dft_matrix(len(x), fs, bpm) @ (x - x.mean())) ** 2


def snr_db(spec: np.ndarray, hr: float, bpm: np.ndarray = BPM, half_width: float = 6.0) -> float:
    """de Haan-style SNR: power near HR and its 2nd harmonic vs the rest of the band."""
    near = (np.abs(bpm - hr) <= half_width) | (np.abs(bpm - 2 * hr) <= 2 * half_width)
    sig, noise = spec[near].sum(), spec[~near].sum()
    return float(10 * np.log10((sig + 1e-12) / (noise + 1e-12)))


def windows(t_end: float, win_s: float, hop_s: float, t_start: float = 0.0) -> np.ndarray:
    """Window end-times from t_start+win_s to t_end."""
    return np.arange(t_start + win_s, t_end + 1e-9, hop_s)


def window_spectra(t: np.ndarray, sig: np.ndarray, fs: float, ends: np.ndarray, win_s: float):
    """Return (W, B) spectra and a (W,) validity mask (False if the window has NaNs)."""
    specs = np.zeros((len(ends), len(BPM)))
    ok = np.zeros(len(ends), bool)
    n = int(round(win_s * fs))
    idx_end = np.searchsorted(t, ends, side="right")
    for i, e in enumerate(idx_end):
        if e - n < 0:
            continue
        seg = sig[e - n:e]
        if not np.all(np.isfinite(seg)):
            continue
        specs[i] = spectrum(seg, fs)
        ok[i] = True
    return specs, ok


def _snr_db_rows(S: np.ndarray, half_width: float = 6.0) -> np.ndarray:
    """de Haan SNR of each row of S, measured at that row's own peak."""
    hr0 = BPM[np.argmax(S, axis=1)][:, None]
    near = (np.abs(BPM[None] - hr0) <= half_width) | (np.abs(BPM[None] - 2 * hr0) <= 2 * half_width)
    sig = (S * near).sum(axis=1)
    noise = (S * ~near).sum(axis=1)
    return 10 * np.log10((sig + 1e-12) / (noise + 1e-12))


def fuse(spec_list: list[np.ndarray], gamma: float = 2.0, min_snr_db: float = -12.0):
    """SNR-weighted fusion of per-patch (W, B) spectra.

    Each spectrum is normalised to unit power, its SNR is measured at its own
    peak, and the weight is (linear SNR)^gamma, so clean patches dominate and
    occluded / noisy patches fade out. Returns fused spectra and (W, P) weights.
    """
    W = spec_list[0].shape[0]
    weights = np.zeros((W, len(spec_list)))
    fused = np.zeros_like(spec_list[0])
    for p, S in enumerate(spec_list):
        tot = S.sum(axis=1, keepdims=True)
        live = tot[:, 0] > 1e-11
        q = _snr_db_rows(S)
        wgt = np.where(live & (q >= min_snr_db), (10 ** (q / 10)) ** gamma, 0.0)
        weights[:, p] = wgt
        fused += wgt[:, None] * S / (tot + 1e-12)
    norm = weights.sum(axis=1, keepdims=True)
    fused = np.where(norm > 0, fused / (norm + 1e-12), 0.0)
    return fused, weights / (norm + 1e-12)


def harmonic_score(spec: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Boost bins whose 2nd harmonic is also present.

    score(f) = S(f) * (1 + beta * min(S(f), S(2f)) / max S). Using the min means
    a sub-harmonic f/2 with little power of its own gains almost nothing from
    the true peak at f, so the boost cannot pull the estimate to half rate.
    """
    out = spec.copy()
    mx = spec.max(axis=-1, keepdims=True) + 1e-12
    i = np.flatnonzero(2 * BPM <= BPM[-1])
    j = np.searchsorted(BPM, 2 * BPM[i])
    out[..., i] = spec[..., i] * (1 + beta * np.minimum(spec[..., i], spec[..., j]) / mx)
    return out


def _transition(sigma_bpm: float, floor: float = 1e-4) -> np.ndarray:
    d = BPM[:, None] - BPM[None, :]
    T = np.exp(-0.5 * (d / sigma_bpm) ** 2) + floor
    return np.log(T / T.sum(axis=1, keepdims=True))


def viterbi(specs: np.ndarray, valid: np.ndarray, hop_s: float, sigma_bpm_per_s: float = 3.0,
            beta: float = 0.0) -> np.ndarray:
    """Offline most-likely HR path through the windowed spectra.

    Emissions are log of the harmonic-boosted, normalised spectrum; invalid
    windows emit uniformly so the path coasts through them.
    """
    H = harmonic_score(specs, beta)
    H = H / (H.sum(axis=1, keepdims=True) + 1e-12)
    logE = np.where(valid[:, None], np.log(H + 1e-6), 0.0)
    logT = _transition(sigma_bpm_per_s * hop_s)
    W, B = logE.shape
    score = logE[0].copy()
    back = np.zeros((W, B), np.int32)
    for w in range(1, W):
        cand = score[:, None] + logT  # from i to j
        back[w] = np.argmax(cand, axis=0)
        score = cand[back[w], np.arange(B)] + logE[w]
    path = np.zeros(W, np.int32)
    path[-1] = int(np.argmax(score))
    for w in range(W - 1, 0, -1):
        path[w - 1] = back[w, path[w]]
    return BPM[path]


class OnlineTracker:
    """Causal version of the Viterbi tracker for live use (forward max-product)."""

    def __init__(self, hop_s: float, sigma_bpm_per_s: float = 3.0, beta: float = 0.0):
        self.logT = _transition(sigma_bpm_per_s * hop_s)
        self.beta = beta
        self.score: np.ndarray | None = None

    def step(self, spec: np.ndarray | None) -> float:
        if spec is None or spec.sum() <= 0:
            logE = np.zeros(len(BPM))
        else:
            H = harmonic_score(spec[None], self.beta)[0]
            logE = np.log(H / (H.sum() + 1e-12) + 1e-6)
        if self.score is None:
            self.score = logE
        else:
            self.score = np.max(self.score[:, None] + self.logT, axis=0) + logE
            self.score -= self.score.max()
        return float(BPM[np.argmax(self.score)])


def peak_hr(specs: np.ndarray) -> np.ndarray:
    return BPM[np.argmax(specs, axis=1)]
