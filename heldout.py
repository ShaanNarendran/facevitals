"""Clean held-out evaluation on subjects never used during development.

    python heldout.py                      # subjects 13-25 -> out/heldout_summary.md
    python heldout.py --lo 25 --hi 40 --tag fresh   # subjects 26-40 -> out/fresh_summary.md
    python heldout.py --lo 40 --hi 60 --tag check   # subjects 41-60 -> out/check_summary.md
    python heldout.py --lo 60 --hi 80 --tag check2  # subjects 61-80 -> out/check2_summary.md

Subjects 13-25 were the first held-out set; the C5 diagnosis then looked at
them, so subjects 26-40 were the clean check after that work. The learned
branch and abstention were then developed on subjects 1-40, so subjects 41-60
(fetch_mcd.py 40 60) were the clean check for those. The low-heart-rate work
then developed on subjects 1-60, so subjects 61-80 are the clean check after
it. Settings are frozen:
nothing here changes based on these results.
"""
import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

MODEL = "models/pulsenet_all.pt"
METHODS = {
    "Mean predictor (always guess average HR)": None,
    "GREEN, whole face": dict(variant="green_face", illum=False),
    "POS, whole face (standard baseline)": dict(variant="pos_face", illum=False),
    "Classical final (frozen after C5 work)": dict(variant="final"),
}
if os.path.exists(MODEL):
    METHODS["Learned branch alone"] = dict(variant="dl")
    METHODS["Classical + learned (fused)"] = dict(variant="final_dl")
FINAL = "Classical + learned (fused)" if os.path.exists(MODEL) else "Classical final (frozen after C5 work)"
CLASSICAL = "Classical final (frozen after C5 work)"


def job(base, cond, meta, by, seconds):
    import cv2
    cv2.setNumThreads(1)
    from fv import extract
    path = f"cache/{base}__{cond}.npz"
    if not os.path.exists(path):
        tr = extract.extract(f"data/mcd/video/{base}.avi", cond, meta=meta, bystander_video=by, max_seconds=seconds)
        np.savez_compressed(path, **tr)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c5", default="enc_zl_fps10_kbps500", help="extra C5 condition (sender-side fix)")
    ap.add_argument("--lo", type=int, default=12)
    ap.add_argument("--hi", type=int, default=25)
    ap.add_argument("--tag", default="heldout")
    ap.add_argument("--extract-only", action="store_true", help="cache traces, compute no results")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    conds = {"baseline": "Normal (VGA, 30 fps)", "lowres": "C4: 320x240", "bandwidth": "C5: 10 fps + 500 kbps (default x264)",
             "lowlight50": "C1: ~50 lux (simulated)", "lowlight20": "C1 stress: ~20 lux (simulated)",
             "mask": "C7: surgical mask (simulated)", "multiface": "C3: bystander in frame"}
    if args.c5 != "bandwidth":
        conds[args.c5] = {"enc_bin_fps10_kbps500": "C5 with sender binning (3-frame average)",
                          "enc_zl_fps10_kbps500": "C5, our sender (every 3rd frame, x264 zerolatency)"}.get(args.c5, args.c5)

    db = pd.read_csv("data/mcd/db.csv")
    db = db[db.camera == "FullHDwebcam"]
    have = db[db.video.apply(lambda v: os.path.exists("data/mcd/" + v))]
    ids = sorted(db.patient_id.unique())[args.lo:args.hi]
    rows = have[have.patient_id.isin(ids)].sort_values(["patient_id", "step"]).to_dict("records")
    print(f"held-out: {len(rows)} videos from {len(ids)} subjects: {ids}", flush=True)

    todo = []
    for i, r in enumerate(rows):
        base = os.path.basename(r["video"])[:-4]
        by = "data/mcd/" + rows[(i + 2) % len(rows)]["video"]
        for c in conds:
            todo.append((base, c, "data/mcd/" + r["meta"], by, args.seconds))
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(job, *t) for t in todo]
        for k, f in enumerate(as_completed(futs), 1):
            f.result()
            if k % 20 == 0 or k == len(futs):
                print(f"  extracted {k}/{len(futs)}", flush=True)
    if args.extract_only:
        return

    from fv import learned, metrics, pipeline, quality
    branch = learned.Branch([MODEL]) if os.path.exists(MODEL) else None
    abstainers = {v: quality.Abstainer.load(f"models/abstainer_{v}.pkl")
                  for v in ("final", "final_dl") if os.path.exists(f"models/abstainer_{v}.pkl")}
    recs = []
    for r in rows:
        base = os.path.basename(r["video"])[:-4]
        ppg = np.array([float(l.split()[0]) for l in open("data/mcd/" + r["ppg_sync"]) if l.strip()])
        tb = np.load(f"cache/{base}__baseline.npz")["t"]
        for c in conds:
            tr = dict(np.load(f"cache/{base}__{c}.npz", allow_pickle=True))
            tr["regions"] = list(tr["regions"])
            for name, kw in METHODS.items():
                if kw is None:
                    continue
                kw = dict(kw)
                variant = kw.pop("variant")
                res = pipeline.estimate(tr, variant, t_end=min(tb[-1], tr["t"][-1]), learned=branch, **kw)
                ref = pipeline.reference_hr(tb, ppg, res["t"])
                final_like = variant in ("final", "final_dl", "dl")
                est = np.where(res["valid"] | final_like, res["hr"], np.nan)
                pg = (abstainers[variant].p_good(quality.window_features(res)) if variant in abstainers
                      else np.full(len(est), np.nan))
                for t, e, rf, p in zip(res["t"], est, ref, pg):
                    recs.append(dict(subject=r["patient_id"], state=r["step"], video=base, cond=c,
                                     method=name, t=t, est=e, ref=rf, p_good=p))
    pw = pd.DataFrame(recs)
    first = [m for m, k in METHODS.items() if k][0]
    mp = pw[pw.method == first].copy()
    mp["method"] = "Mean predictor (always guess average HR)"
    for s in mp.subject.unique():  # leave-one-subject-out average of baseline reference HR
        mp.loc[mp.subject == s, "est"] = pw[(pw.method == first) & (pw.cond == "baseline") & (pw.subject != s)].ref.mean()
    pw = pd.concat([pw, mp], ignore_index=True)
    pw.to_csv(f"out/{args.tag}_per_window.csv", index=False)

    order = list(conds)
    pw["ae"] = (pw.est - pw.ref).abs()
    pooled = pw.groupby(["method", "cond"]).ae.mean().unstack()[order].reindex(list(METHODS))
    median_video = pw.groupby(["method", "cond", "video"]).ae.mean().groupby(["method", "cond"]).median().unstack()[order].reindex(list(METHODS))
    within5 = (pw.assign(ok=pw.ae <= 5).groupby(["method", "cond"]).ok.mean() * 100).unstack()[order].reindex(list(METHODS))
    full = []
    for c in order:
        g = pw[(pw.method == FINAL) & (pw.cond == c)]
        m = metrics.hr_metrics(g.est.values, g.ref.values)
        full.append(dict(condition=conds[c], **m))
    full = pd.DataFrame(full)
    abst = []
    for name in pw.method.unique():
        for c in order:
            g = pw[(pw.method == name) & (pw.cond == c) & pw.p_good.notna() & pw.ref.notna()]
            if len(g):
                k = g[g.p_good >= quality.REPORT_P]
                abst.append({"method": name, "condition": conds[c], "coverage %": 100 * len(k) / len(g),
                             "MAE kept": k.ae.mean(), "within 5 kept %": 100 * (k.ae <= 5).mean(),
                             "MAE all": g.ae.mean()})
    abst = pd.DataFrame(abst)
    per_video = pw[(pw.cond == "baseline") & pw.method.isin([
        "POS, whole face (standard baseline)", CLASSICAL, FINAL])]
    per_video = per_video.groupby(["video", "method"]).agg(mae=("ae", "mean"), ref=("ref", "mean")).unstack().round(1)

    cols = {c: conds[c] for c in order}
    lines = ["# Held-out results (subjects never used in development)", "",
             f"{len(rows)} videos, {len(ids)} subjects ({', '.join(map(str, ids))}), rest + post-exercise, "
             f"first {args.seconds:.0f} s each, 10 s windows / 1 s hop, reference = contact PPG.", "",
             "## Pooled MAE (bpm), all windows", "", pooled.rename(columns=cols).round(2).to_markdown(), "",
             "## Typical video: median of per-video MAE (bpm)", "", median_video.rename(columns=cols).round(2).to_markdown(), "",
             "## % of windows within 5 bpm", "", within5.rename(columns=cols).round(0).to_markdown(), "",
             "## Final method, full metrics", "", full.round(2).to_markdown(index=False), "",
             f"## With signal-quality abstention (report windows with P(error <= 5 bpm) >= {quality.REPORT_P})", "",
             (abst.round(2).to_markdown(index=False) if len(abst) else "(no abstention model)"), "",
             "## Per-video MAE, normal condition", "", per_video.to_markdown(), ""]
    open(f"out/{args.tag}_summary.md", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
