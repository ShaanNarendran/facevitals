"""Signal-quality abstention: is this window's heart rate trustworthy?

Per-window features come from the fused spectrum, the tracker and the
agreement between methods; a small gradient-boosted classifier, fitted on the
development subjects (out-of-fold AUC 0.90), turns them into
P(|error| <= 5 bpm). A window is reported when P >= REPORT_P; otherwise it is
flagged low quality (the HR is still given, so a caller can hold the last
good value). REPORT_P was set on the development subjects so that at least
95% of reported windows are within 5 bpm. Skin colour is deliberately not a
feature, so abstention does not depend on skin tone directly.
"""
from __future__ import annotations

import pickle

import numpy as np

from . import hr

REPORT_P = 0.8

FEATURES = ["snr_db", "peak_share", "track_gap", "branch_gap", "share_trend", "green_gap", "chrom_gap"]


def window_features(res: dict) -> np.ndarray:
    """(W, len(FEATURES)) from an estimate() result of the final variants."""
    S = res["spec"]
    S = S / (S.sum(axis=1, keepdims=True) + 1e-12)
    est = res["hr"]
    near = np.abs(hr.BPM[None] - est[:, None]) <= 4
    share = (S * near).sum(axis=1)
    peak = hr.BPM[np.argmax(S, axis=1)]
    if "S_dl" in res:
        gap = np.abs(hr.BPM[np.argmax(res["S_dl"], axis=1)] - hr.BPM[np.argmax(res["S_pos"], axis=1)])
    else:
        gap = np.zeros(len(est))
    # Median share over the surrounding +-5 windows: one clean window inside a
    # bad stretch is less trustworthy than one inside a clean stretch.
    pad = np.pad(share, 5, mode="edge")
    trend = np.array([np.median(pad[i:i + 11]) for i in range(len(share))])
    snr = np.nan_to_num(res["snr_db"], nan=-20.0)
    # Disagreement with other classical methods (GREEN, CHROM) on the whole
    # face: when the colour mix POS assumes does not hold, they point elsewhere.
    g = np.minimum(np.abs(res["peak_green"] - est), 60) if "peak_green" in res else np.zeros(len(est))
    c = np.minimum(np.abs(res["peak_chrom"] - est), 60) if "peak_chrom" in res else np.zeros(len(est))
    return np.stack([snr, share, np.minimum(np.abs(est - peak), 60), np.minimum(gap, 60), trend, g, c], axis=1)


def live_features(d: dict, est: float, share_history: list) -> np.ndarray:
    """(1, len(FEATURES)) for the newest live window (pipeline.live_window output).

    Appends this window's power share to share_history. share_trend is the
    median over the last 11 windows; offline it is centred on the window,
    which would need future windows.
    """
    S = np.asarray(d["spec"], float)[None]
    res = dict(spec=S, hr=np.array([est]), snr_db=np.array([hr.snr_db(S[0], est)]),
               peak_green=np.array([d["peak_green"]]), peak_chrom=np.array([d["peak_chrom"]]))
    if d.get("S_dl") is not None and d.get("S_pos") is not None:
        res.update(S_dl=d["S_dl"][None], S_pos=d["S_pos"][None])
    F = window_features(res)
    share_history.append(F[0, 1])
    F[0, 4] = np.median(share_history[-11:])
    return F


class Abstainer:
    def __init__(self, model=None):
        self.model = model

    def fit(self, X: np.ndarray, good: np.ndarray):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                                    min_samples_leaf=50, random_state=0).fit(X, good)
        return self

    def p_good(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        pickle.dump(dict(features=FEATURES, model=self.model), open(path, "wb"))

    @classmethod
    def load(cls, path: str) -> "Abstainer":
        d = pickle.load(open(path, "rb"))
        assert d["features"] == FEATURES, "abstainer was fitted on different features"
        return cls(d["model"])
