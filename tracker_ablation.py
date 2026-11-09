"""Harmonic boost on/off in the HR tracker, on the 25 development subjects.

The boost rewards a bin f when 2f also has power, but the pulse band stops at
210 bpm, so true rates above ~105 bpm can never earn it while their half-rate
sub-harmonic can. Writes out/tracker_ablation.csv.
"""
import glob
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from c5_diag import load

CONDS = ["base30", "decim10", "enc_fps10", "bandwidth", "enc_zl_fps10_kbps500", "enc_bin_fps10_kbps500",
         "enc_bin_zl_fps10_kbps500", "lowres", "lowlight50", "lowlight20", "mask", "multiface"]
BETAS = (1.0, 0.0)


def run(base, ppg_path):
    from fv import pipeline
    ppg = np.array([float(l.split()[0]) for l in open(ppg_path) if l.strip()])
    tb = np.load(f"cache/{base}__baseline.npz")["t"]
    out = []
    for c in CONDS:
        if c not in ("base30", "decim10") and not os.path.exists(f"cache/{base}__{c}.npz"):
            continue
        tr = load(base, c)
        for beta in BETAS:
            res = pipeline.estimate(tr, "final", t_end=min(tb[-1], tr["t"][-1]), beta=beta, codec=False)
            ref = pipeline.reference_hr(tb, ppg, res["t"])
            out.append(dict(video=base, cond=c, beta=beta, ref=np.nanmean(ref), mae=np.nanmean(np.abs(res["hr"] - ref))))
    return out


if __name__ == "__main__":
    db = pd.read_csv("data/mcd/db.csv")
    db = db[db.camera == "FullHDwebcam"]
    ids = sorted(db.patient_id.unique())[:25]
    todo = []
    for f in sorted(glob.glob("cache/*__baseline.npz")):
        b = os.path.basename(f).split("__")[0]
        r = db[db.video.str.contains(b, regex=False)].iloc[0]
        if r.patient_id in ids:
            todo.append((b, "data/mcd/" + r.ppg_sync))
    with ProcessPoolExecutor(4) as ex:
        df = pd.DataFrame([x for p in ex.map(run, *zip(*todo)) for x in p])
    df.to_csv("out/tracker_ablation.csv", index=False)
    pd.set_option("display.width", 250)
    cols = [c for c in CONDS if c in df.cond.unique()]
    for agg in ("median", "mean"):
        print(agg)
        print(df.pivot_table(index="beta", columns="cond", values="mae", aggfunc=agg)[cols].round(2).to_string())
    print("videos per cond:", df[df.beta == 1].groupby("cond").size().to_dict())
