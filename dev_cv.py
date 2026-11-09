"""Development evaluation (out-of-fold): classical vs learned branch, fusion rules, abstention.

    python dev_cv.py                                  # subjects 1-60, models/pulsenet_fold{k}.pt
    python dev_cv.py --tag base --out dev60_base      # compare a model set trained with --tag base
    python dev_cv.py --save-abstainers --fusion vote  # also fit the deployed abstention models

Every number here is out-of-fold: each subject's learned spectra come from the
cross-validation model that did not see that subject (dl_train.py cv), and the
abstention model is fitted on the other folds' windows. Untouched check
subjects are never read.
"""
import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from c5_extract import rows_for

CONDS = {"baseline": "Normal", "lowres": "C4 320x240", "lowlight50": "C1 50 lux", "lowlight20": "C1 20 lux",
         "mask": "C7 mask", "multiface": "C3 bystander", "enc_zl_fps10_kbps500": "C5 our sender",
         "bandwidth": "C5 default x264"}
FOLDS = 5
FUSIONS = ("snr", "vote", "all4")
BANDS = [0, 55, 60, 70, 80, 90, 100, 110, 200]
HARD = ["1513_FullHDwebcam_after", "1581_FullHDwebcam_before", "1097_FullHDwebcam_after", "1181_FullHDwebcam_after",
        "1287_FullHDwebcam_before", "1287_FullHDwebcam_after", "1932_FullHDwebcam_after", "1932_FullHDwebcam_before"]


def run(base, pid, fold, ppg_path, model_sfx):
    from fv import learned, pipeline, quality
    ppg = np.array([float(l.split()[0]) for l in open(ppg_path) if l.strip()])
    tb = np.load(f"cache/{base}__baseline.npz")["t"]
    vs = {"classical": dict(variant="final")}
    mpath = f"models/pulsenet{model_sfx}_fold{fold}.pt"
    branch = None
    if os.path.exists(mpath):
        branch = learned.Branch([mpath])
        vs["learned"] = dict(variant="dl")
        vs.update({f"fused_{f}": dict(variant="final_dl", fusion=f) for f in FUSIONS})
    out = []
    for c in CONDS:
        path = f"cache/{base}__{c}.npz"
        if not os.path.exists(path):
            continue
        tr = dict(np.load(path, allow_pickle=True))
        tr["regions"] = list(tr["regions"])
        for name, kw in vs.items():
            res = pipeline.estimate(tr, t_end=min(tb[-1], tr["t"][-1]), learned=branch, **kw)
            ref = pipeline.reference_hr(tb, ppg, res["t"])
            F = quality.window_features(res)
            mk = res.get("mask_detected", np.zeros(len(ref), bool))
            for i in range(len(ref)):
                out.append(dict(subject=pid, fold=fold, video=base, cond=c, method=name, t=res["t"][i], est=res["hr"][i],
                                ref=ref[i], masked=bool(mk[i]), **dict(zip(quality.FEATURES, F[i]))))
    return out


def add_abstention(pw):
    from fv import quality
    pw["p_good"] = np.nan
    for m in pw.method.unique():
        for k in range(FOLDS):
            tr = pw[(pw.method == m) & (pw.fold != k) & pw.ref.notna()]
            te = (pw.method == m) & (pw.fold == k)
            ab = quality.Abstainer().fit(tr[quality.FEATURES].values, (tr.est - tr.ref).abs().values <= 5)
            pw.loc[te, "p_good"] = ab.p_good(pw.loc[te, quality.FEATURES].values)
    return pw


def summarise(pw):
    from fv import quality
    pw = pw[pw.ref.notna()].assign(ae=lambda d: (d.est - d.ref).abs())
    pw["keep"] = pw.p_good >= quality.REPORT_P
    rows = []
    for (m, c), g in pw.groupby(["method", "cond"]):
        k = g[g.keep]
        rows.append(dict(method=m, cond=CONDS[c], videos=g.video.nunique(), mae=g.ae.mean(),
                         median_video=g.groupby("video").ae.mean().median(), within5=100 * (g.ae <= 5).mean(),
                         coverage=100 * g.keep.mean(), mae_kept=k.ae.mean(), within5_kept=100 * (k.ae <= 5).mean()))
    tab = pd.DataFrame(rows)
    tab["cond"] = pd.Categorical(tab.cond, list(CONDS.values()))
    base = pw[pw.cond == "baseline"].assign(band=lambda d: pd.cut(d.ref, BANDS))
    bands = base.pivot_table(index="band", columns="method", values="ae", aggfunc="mean", observed=True)
    bands["windows"] = base[base.method == "classical"].groupby("band", observed=True).size()
    hard = pw[pw.video.isin(HARD) & (pw.cond == "baseline")].groupby(["video", "method"]).ae.mean().unstack()
    hard["ref"] = pw[pw.video.isin(HARD) & (pw.cond == "baseline")].groupby("video").ref.mean()
    return tab.sort_values(["cond", "method"]), bands, hard


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=60)
    ap.add_argument("--tag", default="", help="model set suffix (dl_train.py --tag)")
    ap.add_argument("--out", default="dev60")
    ap.add_argument("--save-abstainers", action="store_true")
    ap.add_argument("--fusion", default="vote", help="fusion whose abstainer is saved as final_dl")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    rows = rows_for(args.lo, args.hi)
    subjects = sorted({r["patient_id"] for r in rows})
    todo = [(os.path.basename(r["video"])[:-4], r["patient_id"], subjects.index(r["patient_id"]) % FOLDS,
             "data/mcd/" + r["ppg_sync"], sfx) for r in rows]
    with ProcessPoolExecutor(5) as ex:
        pw = pd.DataFrame([x for part in ex.map(run, *zip(*todo)) for x in part])
    pw = add_abstention(pw)
    pw.to_csv(f"out/{args.out}_per_window.csv", index=False)
    from fv import quality
    if args.save_abstainers:
        # Deployed abstention models: fitted on all out-of-fold development windows.
        for name, variant in (("classical", "final"), (f"fused_{args.fusion}", "final_dl")):
            d = pw[(pw.method == name) & pw.ref.notna()]
            quality.Abstainer().fit(d[quality.FEATURES].values, (d.est - d.ref).abs().values <= 5).save(
                f"models/abstainer_{variant}.pkl")
            print(f"saved models/abstainer_{variant}.pkl from {name}")
    tab, bands, hard = summarise(pw)
    lines = [f"# Development results, subjects {args.lo + 1}-{args.hi} (out-of-fold), models '{args.tag or 'default'}'", "",
             f"Abstention: a window is reported when the quality model gives P(error <= 5 bpm) >= {quality.REPORT_P}.", "",
             tab.round(2).to_markdown(index=False), "",
             "## Normal condition: MAE by reference heart-rate band", "", bands.round(2).to_markdown(), "",
             "## Hard subjects, normal condition (MAE)", "", hard.round(2).to_markdown(), ""]
    open(f"out/{args.out}_summary.md", "w").write("\n".join(lines))
    print("\n".join(lines))
