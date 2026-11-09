"""C5 diagnosis on the 25 development subjects (never the fresh held-out set).

Separates frame rate from bitrate:
  base30       30 fps, original video
  decim10      30 fps traces, every 3rd sample kept (10 fps, no codec)
  enc_fps10    10 fps frames, high-quality H.264
  enc_kbps500  30 fps, 500 kbps H.264
  bandwidth    10 fps + 500 kbps (as in the brief)
  enc_zl_fps10_kbps500  same, sender uses x264 zerolatency (no B-frames)
Writes out/c5_diag.csv (one row per video x condition x variant).
"""
import glob
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

CONDS = ["base30", "decim10", "enc_fps10", "enc_kbps500", "bandwidth", "enc_zl_fps10_kbps500"]
VARIANTS = [("pos_face", False, False), ("pos_face_viterbi", False, False), ("final", False, False),
            ("final", True, False), ("final", True, True)]


def decimate(tr, fps=10):
    """Keep every Nth sample, as the encoder's frame decimation does."""
    t = tr["t"]
    pick = np.arange(0, len(t), max(1, int(np.ceil(1 / np.median(np.diff(t)) / fps - 0.05))))
    d = dict(tr)
    d["t"], d["rgb"] = t[pick], tr["rgb"][pick]
    return d


def load(base, cond):
    name = "baseline" if cond in ("base30", "decim10") else cond
    tr = dict(np.load(f"cache/{base}__{name}.npz", allow_pickle=True))
    tr["regions"] = list(tr["regions"])
    return decimate(tr) if cond == "decim10" else tr


def run(base, ppg_path, variants=VARIANTS, conds=CONDS):
    from fv import pipeline
    ppg = np.array([float(l.split()[0]) for l in open(ppg_path) if l.strip()])
    tb = np.load(f"cache/{base}__baseline.npz")["t"]
    out = []
    for c in conds:
        if c not in ("base30", "decim10") and not os.path.exists(f"cache/{base}__{c}.npz"):
            continue
        tr = load(base, c)
        for v, il, cod in variants:
            res = pipeline.estimate(tr, v, t_end=min(tb[-1], tr["t"][-1]), illum=il, codec=cod)
            ref = pipeline.reference_hr(tb, ppg, res["t"])
            out.append(dict(video=base, cond=c, variant=v + ("+illum" if il else "") + ("+codec" if cod else ""), ref=np.nanmean(ref),
                            mae=np.nanmean(np.abs(res["hr"] - ref)), bias=np.nanmean(res["hr"] - ref)))
    return out


if __name__ == "__main__":
    db = pd.read_csv("data/mcd/db.csv")
    db = db[db.camera == "FullHDwebcam"]
    ids = sorted(db.patient_id.unique())[:25]
    todo = []
    for f in sorted(glob.glob("cache/*__baseline.npz")):
        base = os.path.basename(f).split("__")[0]
        r = db[db.video.str.contains(base, regex=False)].iloc[0]
        if r.patient_id in ids:
            todo.append((base, "data/mcd/" + r.ppg_sync))
    with ProcessPoolExecutor(6) as ex:
        rows = [x for part in ex.map(run, *zip(*todo)) for x in part]
    df = pd.DataFrame(rows)
    df.to_csv("out/c5_diag.csv", index=False)
    print(f"{df.video.nunique()} videos")
    print("median per-video MAE"); print(df.pivot_table(index="variant", columns="cond", values="mae", aggfunc="median")[CONDS].round(2))
    print("mean per-video MAE"); print(df.pivot_table(index="variant", columns="cond", values="mae", aggfunc="mean")[CONDS].round(2))
