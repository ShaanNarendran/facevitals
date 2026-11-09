"""Score configurations on cached traces for a chosen subject split.

    python dev_eval.py dev    # first 5 subjects (used for tuning)
    python dev_eval.py test   # remaining subjects (held out)
"""
import glob, os, sys, warnings
import numpy as np, pandas as pd
from fv import pipeline, metrics
warnings.filterwarnings("ignore")

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
CONDS = ["baseline", "lowres", "bandwidth", "lowlight50", "lowlight20", "mask", "multiface"]
CONFIGS = {
    "green_face": dict(variant="green_face"),
    "chrom_face": dict(variant="chrom_face"),
    "pos_face": dict(variant="pos_face"),
    "pos_face_viterbi": dict(variant="pos_face_viterbi"),
    "proposed g2": dict(variant="proposed", gamma=2.0),
    "proposed g1": dict(variant="proposed", gamma=1.0),
    "proposed g1+face": dict(variant="proposed", gamma=1.0, include_face=True),
    "proposed g2+face": dict(variant="proposed", gamma=2.0, include_face=True),
    "pos_patches": dict(variant="pos_patches"),
}
if len(sys.argv) > 2:
    CONFIGS = {k: v for k, v in CONFIGS.items() if k in sys.argv[2].split(",")}
db = pd.read_csv("data/mcd/db.csv"); db = db[db.camera == "FullHDwebcam"]
subjects = sorted(db.patient_id.unique())
pick = {"dev": subjects[:5], "test": subjects[5:12], "fresh": sorted(db.patient_id.unique())[12:25],
        "all": sorted(db.patient_id.unique())[:25]}[split]
bases = [os.path.basename(v)[:-4] for v in db[db.patient_id.isin(pick)].video]
bases = [b for b in bases if all(os.path.exists(f"cache/{b}__{c}.npz") for c in CONDS)]
print(f"{split}: {len(bases)} videos (all 7 conditions) from {len({b.split('_')[0] for b in bases})} subjects")
rows = []
for b in bases:
    r = db[db.video == f"video/{b}.avi"].iloc[0]
    ppg = np.array([float(l.split()[0]) for l in open("data/mcd/" + r.ppg_sync) if l.strip()])
    tb = np.load(f"cache/{b}__baseline.npz")["t"]
    for c in CONDS:
        tr = dict(np.load(f"cache/{b}__{c}.npz", allow_pickle=True)); tr["regions"] = list(tr["regions"])
        for name, kw in CONFIGS.items():
            kw = dict(kw); v = kw.pop("variant")
            res = pipeline.estimate(tr, v, t_end=min(tb[-1], tr["t"][-1]), **kw)
            ref = pipeline.reference_hr(tb, ppg, res["t"])
            est = np.where(res["valid"] | v.startswith("proposed"), res["hr"], np.nan)
            for e, rf in zip(est, ref):
                rows.append(dict(video=b, subj=r.patient_id, state=r.step, cond=c, config=name, est=e, ref=rf))
pw = pd.DataFrame(rows)
pw.to_csv(f"out/{split}_per_window.csv", index=False)
mae = pw.assign(ae=(pw.est - pw.ref).abs()).groupby(["config", "cond"]).ae.mean().unstack()[CONDS]
mae["mean"] = mae.mean(axis=1)
print(mae.reindex(list(CONFIGS)).round(2).to_string())

# Leave-one-subject-out mean predictor: the floor any method must beat.
first = list(CONFIGS)[0]
bl = pw[pw.config == first]
mp = bl.copy(); mp["config"] = "mean_predictor"
for s_ in mp.subj.unique():
    mp.loc[mp.subj == s_, "est"] = bl[(bl.subj != s_) & (bl.cond == "baseline")].ref.mean()
allpw = pd.concat([pw, mp])
from fv import metrics as M
full = []
for (cfg, c), g in allpw.groupby(["config", "cond"]):
    m = M.hr_metrics(g.est.values, g.ref.values); m.update(config=cfg, cond=c); full.append(m)
full = pd.DataFrame(full)
full.to_csv(f"out/{split}_metrics.csv", index=False)
print("\nmean predictor MAE:", full[full.config == "mean_predictor"].set_index("cond").mae.reindex(CONDS).round(2).to_dict())
