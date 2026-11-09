"""Traces -> windowed heart rate, for the proposed method and its ablations."""
from __future__ import annotations

import numpy as np

from . import face, hr, signals

FS = 30.0
WIN_S = 10.0
HOP_S = 1.0
PATCH_NAMES = list(face.PATCHES)


def occlusion_mask(t: np.ndarray, rgb: np.ndarray, regions: list[str], ends: np.ndarray,
                   skin_ratio: float = 0.5, far: float = 0.08) -> np.ndarray:
    """(W, P) True where a patch no longer looks like skin.

    Skin reflects much more red than blue under ordinary lighting; masks
    (light blue, white, black) do not. A patch is flagged when its red-minus-
    blue chromaticity falls below `skin_ratio` of the forehead's, or its
    chromaticity is very far (> `far`) from the forehead's. Moderate colour
    differences from side lighting are left to the SNR weighting.
    """
    pidx = [regions.index(p) for p in PATCH_NAMES]
    fidx = regions.index("forehead")
    occ = np.zeros((len(ends), len(pidx)), bool)
    for w, te in enumerate(ends):
        m = (t > te - WIN_S) & (t <= te)
        if not m.any():
            continue
        mean = np.nanmean(rgb[m], axis=0)  # (R, 3)
        ch = mean / (mean.sum(axis=1, keepdims=True) + 1e-9)
        ref = ch[fidx]
        if not np.all(np.isfinite(ref)):
            ref = np.nanmedian(ch[[regions.index(p) for p in face.UPPER_PATCHES]], axis=0)
        rb = ch[pidx, 0] - ch[pidx, 2]
        d = np.linalg.norm(ch[pidx, :2] - ref[:2], axis=1)
        flag = ~np.isfinite(rb) | (rb < skin_ratio * (ref[0] - ref[2])) | (d > far)
        flag[PATCH_NAMES.index("forehead")] = ~np.all(np.isfinite(ch[fidx]))
        occ[w] = flag
    return occ


def cancel_illumination(rgb_u: np.ndarray, regions: list[str]) -> np.ndarray:
    """Remove the part of each skin trace that moves with the background.

    Light flicker (and its alias at low frame rates) and global brightness
    changes modulate skin and background alike; the pulse is only in skin.
    Each normalised, band-passed skin channel is regressed on the three
    band-passed background channels and the fitted part is subtracted.
    """
    if "bg" not in regions:
        return rgb_u
    b = rgb_u[:, regions.index("bg")]
    if np.isfinite(b).mean() < 0.5:
        return rgb_u
    nanrow = ~np.all(np.isfinite(rgb_u.reshape(len(rgb_u), -1)), axis=1)
    bf = signals._fill_nan(b)
    B = signals.bandpass(bf / bf.mean(axis=0) - 1.0, FS)  # (N, 3)
    out = rgb_u.copy()
    for k, name in enumerate(regions):
        if name == "bg" or np.isfinite(rgb_u[:, k]).mean() < 0.5:
            continue
        x = signals._fill_nan(rgb_u[:, k])
        mu = x.mean(axis=0)
        xn = x / mu - 1.0
        xb = signals.bandpass(xn, FS)
        coef, *_ = np.linalg.lstsq(B, xb, rcond=None)  # (3, 3)
        out[:, k] = (1.0 + xn - B @ coef) * mu
    out[nanrow] = np.nan
    return out


def remove_codec_pattern(rgb: np.ndarray, ftype: np.ndarray, max_phase: int = 8) -> np.ndarray:
    """Remove colour offsets locked to the encoder's B-frame cycle.

    H.264 streams often repeat P B B B; each position in that cycle gets its
    own quantisation, which shifts the mean skin colour by a fixed amount per
    position. At 10 fps the cycle is 2.5 Hz (150 bpm), inside the HR band.
    Each frame's phase is the number of frames since the last I/P frame; per
    region and channel, the detrended, normalised trace is regressed on the
    phase indicators and that fitted offset is removed. A stream without
    B-frames has a single phase and is returned unchanged.
    """
    ftype = np.asarray(ftype).astype(str)
    if len(ftype) != len(rgb) or not (ftype == "B").any():
        return rgb
    phase = np.zeros(len(ftype), int)
    for i in range(1, len(ftype)):
        phase[i] = min(phase[i - 1] + 1, max_phase) if ftype[i] == "B" else 0
    anchors = np.flatnonzero(phase == 0)
    cycle = max(int(np.median(np.diff(anchors))), 2) if len(anchors) > 2 else 4
    D = np.stack([phase == k for k in range(1, phase.max() + 1)], axis=1).astype(float)
    D -= D.mean(axis=0)
    n_ma = cycle * max(1, int(round(10 / cycle)))  # ~1 s, a whole number of cycles
    out = rgb.copy()
    flat = out.reshape(len(rgb), -1)
    for c in range(flat.shape[1]):
        x = flat[:, c]
        ok = np.isfinite(x)
        if ok.sum() < 3 * n_ma:
            continue
        xf = signals._fill_nan(x[:, None])[:, 0]
        mu = xf.mean()
        xn = xf / mu - 1.0
        hp = xn - np.convolve(xn, np.ones(n_ma) / n_ma, mode="same")
        coef, *_ = np.linalg.lstsq(D[ok], hp[ok], rcond=None)
        flat[ok, c] = (1.0 + xn[ok] - D[ok] @ coef) * mu
    return out


def prepare(tr: dict, illum: bool = True, codec: bool = True):
    """Traces -> (uniform time grid, (N, R, 3) cleaned RGB, region names)."""
    regions = list(tr["regions"])
    rgb = tr["rgb"]
    if codec and "ftype" in tr:
        rgb = remove_codec_pattern(rgb, tr["ftype"])
    grid, rgb_u = signals.resample_uniform(tr["t"], rgb.reshape(len(tr["t"]), -1), FS)
    rgb_u = rgb_u.reshape(len(grid), len(regions), 3)
    if illum:
        rgb_u = cancel_illumination(rgb_u, regions)
    return grid, rgb_u, regions


DISAGREE_BPM = 10.0
VOTE_TOL_BPM = 6.0


def vote_disagreements(fused, S_pos, S_dl, peak_green, peak_chrom, both_ok):
    """Where POS and the learned branch disagree strongly, let GREEN and CHROM vote.

    A vote goes to a branch when that method's peak is within VOTE_TOL_BPM of
    the branch's peak. The branch with more votes is used alone in that
    window; a tie keeps the SNR fusion.
    """
    p_pos = hr.BPM[np.argmax(S_pos, axis=1)]
    p_dl = hr.BPM[np.argmax(S_dl, axis=1)]
    out = fused.copy()
    for i in np.flatnonzero(both_ok & (np.abs(p_pos - p_dl) > DISAGREE_BPM)):
        votes = [peak_green[i], peak_chrom[i]]
        v_pos = sum(abs(v - p_pos[i]) <= VOTE_TOL_BPM for v in votes)
        v_dl = sum(abs(v - p_dl[i]) <= VOTE_TOL_BPM for v in votes)
        if v_pos != v_dl:
            S = S_pos[i] if v_pos > v_dl else S_dl[i]
            out[i] = S / (S.sum() + 1e-12)
    return out


def estimate(tr: dict, variant: str = "proposed", t_end: float | None = None,
             gamma: float = 1.0, include_face: bool = False, beta: float = 0.0,
             illum: bool = True, codec: bool = True, learned=None, fusion: str = "snr") -> dict:
    """Run one method variant on extracted traces.

    Variants (ablation ladder):
      green_face, chrom_face, pos_face   single whole-face ROI, spectral peak
      pos_face_viterbi                   + harmonic Viterbi tracking
      pos_patches                        6 skin patches, SNR-weighted fusion, peak
      proposed                           POS+CHROM x 6 patches, occlusion gate,
                                         SNR fusion, harmonic Viterbi
      final                              whole-face POS + harmonic Viterbi; windows
                                         where a mask is detected switch to fused
                                         unoccluded patches
      dl                                 learned branch alone + Viterbi (needs learned)
      final_dl                           final, with POS and the learned spectrum
                                         SNR-fused in unmasked windows
    illum: cancel background-correlated illumination first (needs a "bg" region).
    codec: remove B-frame-cycle colour offsets first (needs per-frame "ftype").
    fusion: how final_dl combines whole-face POS with the learned spectrum:
           "snr"  SNR-weighted fusion of the two;
           "vote" the same, except where their peaks differ by more than
                  DISAGREE_BPM: then GREEN and CHROM on the whole face vote,
                  and the branch with more votes is used alone;
           "all4" SNR-weighted fusion of POS, CHROM, GREEN and learned.
           Out-of-fold on subjects 1-60 "vote" and "all4" were no better than
           "snr" (and "vote" was worse for C5), so "snr" is the default.
    beta:  harmonic boost in the tracker. Off by default: the pulse band ends at
           210 bpm, so rates above ~105 bpm can never earn the boost while their
           half-rate sub-harmonic can, which biased the tracker to half rate.
    """
    grid, rgb_u, regions = prepare(tr, illum, codec)
    t_end = grid[-1] if t_end is None else t_end
    ends = hr.windows(t_end, WIN_S, HOP_S, t_start=grid[0])

    def method_spectra(method: str, region: str):
        k = regions.index(region)
        x = rgb_u[:, k]
        if np.isfinite(x).sum() < FS * WIN_S:
            return np.zeros((len(ends), len(hr.BPM))), np.zeros(len(ends), bool)
        nan = ~np.all(np.isfinite(x), axis=1)
        p = signals.METHODS[method](x, FS)
        p[nan] = np.nan
        return hr.window_spectra(grid, p, FS, ends, WIN_S)

    out = dict(t=ends)
    if variant in ("green_face", "chrom_face", "pos_face", "pos_face_viterbi"):
        method = variant.split("_")[0]
        S, ok = method_spectra(method, "face")
        out["hr"] = hr.viterbi(S, ok, HOP_S, beta=beta) if variant.endswith("viterbi") else hr.peak_hr(S)
        out["valid"] = ok
        out["snr_db"] = np.array([hr.snr_db(S[i], out["hr"][i]) if ok[i] else np.nan for i in range(len(ends))])
        return out

    if variant == "dl":
        S, ok = learned.spectra(grid, rgb_u, regions, ends)
        S = S / (S.sum(axis=1, keepdims=True) + 1e-12)
        out["hr"] = hr.viterbi(S, ok, HOP_S, beta=beta)
        out["valid"] = ok
        out["spec"] = S
        out["snr_db"] = np.array([hr.snr_db(S[i], out["hr"][i]) if ok[i] else np.nan for i in range(len(ends))])
        return out

    if variant in ("final", "final_dl"):
        S_face, ok_face = method_spectra("pos", "face")
        S_green, ok_green = method_spectra("green", "face")
        S_chrom, ok_chrom = method_spectra("chrom", "face")
        # Other classical methods' peaks: quality evidence, and votes for "vote".
        out["peak_green"] = hr.BPM[np.argmax(S_green, axis=1)]
        out["peak_chrom"] = hr.BPM[np.argmax(S_chrom, axis=1)]
        if variant == "final_dl":
            S_dl, ok_dl = learned.spectra(grid, rgb_u, regions, ends)
            out["S_pos"], out["S_dl"] = S_face, S_dl
            if fusion == "all4":
                both, _ = hr.fuse([S * k[:, None] for S, k in ((S_face, ok_face), (S_chrom, ok_chrom),
                                                                (S_green, ok_green), (S_dl, ok_dl))],
                                  gamma=gamma, min_snr_db=-99)
            else:
                both, _ = hr.fuse([S_face * ok_face[:, None], S_dl * ok_dl[:, None]], gamma=gamma, min_snr_db=-99)
            if fusion == "vote":
                both = vote_disagreements(both, S_face, S_dl, out["peak_green"], out["peak_chrom"], ok_face & ok_dl)
            S_face, ok_face = both, ok_face | ok_dl
        occ = occlusion_mask(grid, rgb_u, regions, ends)
        lo = [PATCH_NAMES.index("lowcheek_l"), PATCH_NAMES.index("lowcheek_r")]
        masked = occ[:, lo].all(axis=1)
        fused = np.zeros_like(S_face)
        if masked.any():  # patch spectra are only used in masked windows
            specs = []
            for m in ("pos", "chrom"):
                for j, p in enumerate(PATCH_NAMES):
                    S, ok = method_spectra(m, p)
                    specs.append(S * ok[:, None] * (~occ[:, j])[:, None])
            fused, _ = hr.fuse(specs, gamma=gamma)
        tot = S_face.sum(axis=1, keepdims=True) + 1e-12
        S = np.where(masked[:, None], fused, S_face / tot)
        valid = np.where(masked, fused.sum(axis=1) > 0, ok_face)
        out["hr"] = hr.viterbi(S, valid, HOP_S, beta=beta)
        out["valid"] = valid
        out["mask_detected"] = masked
        out["spec"] = S
        out["snr_db"] = np.array([hr.snr_db(S[i], out["hr"][i]) if valid[i] else np.nan for i in range(len(ends))])
        return out

    methods = ["pos"] if variant == "pos_patches" else ["pos", "chrom"]
    occ = occlusion_mask(grid, rgb_u, regions, ends) if variant == "proposed" else None
    specs, names = [], []
    for m in methods:
        for j, p in enumerate(PATCH_NAMES):
            S, ok = method_spectra(m, p)
            S = S * ok[:, None]
            if occ is not None:
                S = S * (~occ[:, j])[:, None]
            specs.append(S)
            names.append(f"{m}:{p}")
        if include_face:
            S, ok = method_spectra(m, "face")
            specs.append(S * ok[:, None])
            names.append(f"{m}:face")
    fused, wts = hr.fuse(specs, gamma=gamma)
    valid = fused.sum(axis=1) > 0
    out["hr"] = hr.viterbi(fused, valid, HOP_S, beta=beta) if variant == "proposed" else hr.peak_hr(fused)
    out["valid"] = valid
    out["snr_db"] = np.array([hr.snr_db(fused[i], out["hr"][i]) if valid[i] else np.nan for i in range(len(ends))])
    out["weights"] = wts
    out["weight_names"] = names
    if occ is not None:
        out["occluded_frac"] = dict(zip(PATCH_NAMES, occ.mean(axis=0)))
    return out


def live_window(t: np.ndarray, rgb: np.ndarray, regions: list[str], learned=None) -> dict | None:
    """Everything the live app computes and shows for the latest WIN_S seconds.

    Same steps as the "final_dl" variant: illumination cancellation, whole-face
    POS SNR-fused with the learned branch (when given); if both lower cheeks
    look masked, the fused POS/CHROM spectrum of the unoccluded patches instead.
    Returns None until WIN_S seconds are buffered. Keys:
      t, raw, bg     last WIN_S s: face and background RGB, % change from their mean
      pulse          whole-face POS after cleaning (None if the face was lost)
      S_pos, S_dl    POS and learned-branch spectra (S_dl None without a branch)
      spec           spectrum the tracker uses (None if nothing usable)
      peak_green, peak_chrom   whole-face GREEN / CHROM peaks (quality evidence)
      weights, occ, masked     per-patch fusion weight, occlusion flags, mask gate
    """
    grid, raw = signals.resample_uniform(t, rgb.reshape(len(t), -1), FS)
    raw = raw.reshape(len(grid), len(regions), 3)
    if grid[-1] - grid[0] < WIN_S:
        return None
    rgb_u = cancel_illumination(raw, regions)
    end = np.array([grid[-1]])
    n = int(WIN_S * FS)
    occ = occlusion_mask(grid, rgb_u, regions, end)[0]
    occ_d = dict(zip(PATCH_NAMES, occ))
    rel = lambda x: 100 * (x / np.nanmean(x, axis=0) - 1)
    out = dict(t=grid[-n:], raw=rel(raw[-n:, regions.index("face")]), bg=rel(raw[-n:, regions.index("bg")]),
               pulse=None, S_pos=None, S_dl=None, spec=None, peak_green=np.nan, peak_chrom=np.nan,
               weights={}, occ=occ_d, masked=bool(occ_d["lowcheek_l"] and occ_d["lowcheek_r"]))
    x = rgb_u[:, regions.index("face")]
    if np.all(np.isfinite(x[-n:])):
        sig = signals.pos(x, FS)
        S, _ = hr.window_spectra(grid, sig, FS, end, WIN_S)
        out.update(pulse=sig[-n:], S_pos=S[0])
        for m in ("green", "chrom"):
            Sm, _ = hr.window_spectra(grid, signals.METHODS[m](x, FS), FS, end, WIN_S)
            out[f"peak_{m}"] = float(hr.BPM[np.argmax(Sm[0])])
        if learned is not None:
            S_dl, ok_dl = learned.spectra(grid, rgb_u, regions, end)
            if ok_dl[0]:
                out["S_dl"] = S_dl[0]
                S, _ = hr.fuse([S, S_dl], gamma=1.0, min_snr_db=-99)
        if not out["masked"]:
            out.update(spec=S[0], weights={p: float(not occ_d[p]) for p in PATCH_NAMES})
            return out
    if not out["masked"]:
        return out
    specs, names, pulses = [], [], {}
    for m in ("pos", "chrom"):
        for j, p in enumerate(PATCH_NAMES):
            xp = rgb_u[:, regions.index(p)]
            if occ[j] or not np.all(np.isfinite(xp[-n:])):
                specs.append(np.zeros((1, len(hr.BPM))))
            else:
                sig = signals.METHODS[m](xp, FS)
                S, _ = hr.window_spectra(grid, sig, FS, end, WIN_S)
                specs.append(S)
                pulses[f"{m}:{p}"] = sig[-n:]
            names.append(f"{m}:{p}")
    fused, w = hr.fuse(specs)
    if fused.sum() > 0:
        out["weights"] = {p: float(sum(w[0, k] for k, nm in enumerate(names) if nm.endswith(p))) for p in PATCH_NAMES}
        out["spec"] = fused[0]
        out["pulse"] = pulses.get(names[int(np.argmax(w[0]))])
    return out


def latest_fused_spectrum(t: np.ndarray, rgb: np.ndarray, regions: list[str], learned=None):
    """(spectrum or None, {patch: weight}, {patch: occluded}, pulse) for the latest WIN_S seconds."""
    d = live_window(t, rgb, regions, learned)
    if d is None:
        return None, {}, {}, None
    return d["spec"], d["weights"], d["occ"], d["pulse"]


def reference_hr(t_frames: np.ndarray, ppg: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Reference HR from the contact PPG with the same windows (spectral peak)."""
    n = min(len(t_frames), len(ppg))
    grid, p = signals.resample_uniform(t_frames[:n], ppg[:n].astype(float), FS)
    p = signals.bandpass(p[:, 0] - np.nanmean(p[:, 0]), FS)
    S, ok = hr.window_spectra(grid, p, FS, ends, WIN_S)
    ref = hr.peak_hr(S).astype(float)
    ref[~ok] = np.nan
    return ref
