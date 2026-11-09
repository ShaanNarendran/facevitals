"""Benchmark the pipeline and its ablations on MCD-rPPG under SP Cup constraints.

Usage:  python benchmark_mcd.py [--seconds 90] [--workers 8]

Stage 1 extracts per-patch colour traces for every (video, condition) pair in
parallel and caches them in cache/. Stage 2 runs every method variant on the
cached traces, scores it against the contact-PPG reference with the same
10 s windows, and writes out/results.csv, out/summary.md and plots.
"""
import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

os.environ.setdefault("GLOG_minloglevel", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore", category=RuntimeWarning)

CONDITIONS = {
    "baseline": "VGA 30 fps, as recorded",
    "lowres": "C4: 320x240 (QVGA)",
    "bandwidth": "C5: 10 fps, 500 kbps H.264",
    "lowlight50": "C1: ~50 lux simulated (shot + read noise)",
    "lowlight20": "C1 stress: ~20 lux simulated",
    "mask": "C7: surgical-mask overlay on lower face",
    "multiface": "C3: bystander face beside the target",
}
VARIANTS = ["mean_predictor", "green_face", "chrom_face", "pos_face", "pos_face_viterbi", "pos_patches", "proposed"]


def load_ppg(path):
    with open(path) as f:
        return np.array([float(line.split()[0]) for line in f if line.strip()])


def job(row, cond, seconds, bystander):
    import cv2
    cv2.setNumThreads(1)
    from fv import extract

    key = f"{os.path.splitext(os.path.basename(row['video']))[0]}__{cond}"
    path = os.path.join("cache", key + ".npz")
    if not os.path.exists(path):
        tr = extract.extract(os.path.join("data/mcd", row["video"]), cond,
                             meta=os.path.join("data/mcd", row["meta"]),
                             bystander_video=bystander, max_seconds=seconds)
        np.savez_compressed(path, **tr)
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--subjects", type=int, default=12)
    args = ap.parse_args()
    conds = args.conditions.split(",")
    os.makedirs("cache", exist_ok=True)
    os.makedirs("out", exist_ok=True)
    os.makedirs("data/degraded", exist_ok=True)

    db = pd.read_csv("data/mcd/db.csv")
    db = db[db.camera == "FullHDwebcam"]
    db = db[db.video.apply(lambda v: os.path.exists(os.path.join("data/mcd", v)))
            & db.ppg_sync.apply(lambda v: os.path.exists(os.path.join("data/mcd", v)))]
    keep = sorted(db.patient_id.unique())[:args.subjects]
    db = db[db.patient_id.isin(keep)]
    rows = db.to_dict("records")
    print(f"{len(rows)} videos from {db.patient_id.nunique()} subjects; conditions: {conds}", flush=True)

    # ---- stage 1: extraction ----
    todo = []
    for i, r in enumerate(rows):
        by = os.path.join("data/mcd", rows[(i + 2) % len(rows)]["video"])  # a different subject
        for c in conds:
            todo.append((r, c, args.seconds, by))
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(job, *a) for a in todo]
        for k, f in enumerate(as_completed(futs), 1):
            f.result()
            if k % 10 == 0 or k == len(futs):
                print(f"  extracted {k}/{len(futs)}", flush=True)

    # ---- stage 2: estimation and scoring ----
    from fv import face, metrics, pipeline

    recs, per_window = [], []
    for r in rows:
        base = os.path.splitext(os.path.basename(r["video"]))[0]
        ppg = load_ppg(os.path.join("data/mcd", r["ppg_sync"]))
        bl = np.load(os.path.join("cache", f"{base}__baseline.npz"), allow_pickle=True)
        t_frames = bl["t"]
        rgb_bl = bl["rgb"]
        regions = list(bl["regions"])
        skin = np.nanmean(rgb_bl[:, [regions.index(p) for p in face.UPPER_PATCHES]], axis=(0, 1))
        ita = face.ita_degrees(skin)
        for c in conds:
            tr = dict(np.load(os.path.join("cache", f"{base}__{c}.npz"), allow_pickle=True))
            tr["regions"] = list(tr["regions"])
            t_end = min(t_frames[-1], tr["t"][-1])
            for v in VARIANTS[1:]:
                res = pipeline.estimate(tr, v, t_end=t_end)
                ref = pipeline.reference_hr(t_frames, ppg, res["t"])
                # Windows without a usable face are NaN for single-window methods;
                # the tracked method coasts through them, so it is scored on every window.
                est = np.where(res["valid"] | (v == "proposed"), res["hr"], np.nan)
                for te, e, rf, q in zip(res["t"], est, ref, res["snr_db"]):
                    per_window.append(dict(subject=r["patient_id"], state=r["step"], condition=c, variant=v,
                                           t=te, est=e, ref=rf, snr_db=q))
            recs.append(dict(subject=r["patient_id"], state=r["step"], condition=c, ita=ita,
                             ms_per_frame=float(tr["ms_per_frame"]),
                             on_target=float(np.mean(tr["on_target"])) if "on_target" in tr else np.nan,
                             face_found=float(np.mean(tr["found"]))))
    pw = pd.DataFrame(per_window)
    info = pd.DataFrame(recs)

    # Mean predictor (leave-one-subject-out) - the floor every method must beat.
    mp = pw[(pw.variant == "proposed")].copy()
    mp["variant"] = "mean_predictor"
    subj_mean = pw[(pw.variant == "proposed") & (pw.condition == "baseline")].groupby("subject").ref.mean()
    tot = pw[(pw.variant == "proposed") & (pw.condition == "baseline")].ref
    for s in mp.subject.unique():
        others = tot[pw.loc[tot.index, "subject"] != s].mean()
        mp.loc[mp.subject == s, "est"] = others
    pw = pd.concat([pw, mp], ignore_index=True)
    pw.to_csv("out/per_window.csv", index=False)
    info.to_csv("out/video_info.csv", index=False)

    rows_out = []
    for (c, v), g in pw.groupby(["condition", "variant"]):
        m = metrics.hr_metrics(g.est.values, g.ref.values)
        # Per-video mean HR error (whole-clip protocol used by many papers).
        vid = g.groupby(["subject", "state"]).agg(est=("est", "mean"), ref=("ref", "mean"))
        m["mae_clip"] = float((vid.est - vid.ref).abs().mean())
        rows_out.append(dict(condition=c, variant=v, **m))
    res = pd.DataFrame(rows_out)
    res.to_csv("out/results.csv", index=False)

    # ---- summary ----
    lines = ["# FaceVitals prototype - MCD-rPPG benchmark", "",
             f"{len(rows)} front-camera videos, {db.patient_id.nunique()} subjects (rest + post-exercise), "
             f"first {args.seconds:.0f} s of each. 10 s windows, 1 s hop. Reference: contact PPG, same windowing.", ""]
    piv = res.pivot(index="variant", columns="condition", values="mae").reindex(VARIANTS)[[c for c in CONDITIONS if c in conds]]
    lines += ["## MAE (bpm) per 10 s window", "", piv.round(2).to_markdown(), ""]
    prop = res[res.variant == "proposed"].set_index("condition").reindex([c for c in CONDITIONS if c in conds])
    prop["description"] = [CONDITIONS[c] for c in prop.index]
    lines += ["## Proposed method - full metrics", "",
              prop[["description", "n", "mae", "rmse", "r", "ba_bias", "ba_loa_low", "ba_loa_high", "within5", "mae_clip"]]
              .round(2).to_markdown(), ""]
    ops = info.groupby("condition")[["ms_per_frame", "face_found", "on_target"]].mean().round(3)
    lines += ["## Face tracking cost and coverage (Apple M3 Pro, 1 process)", "", ops.to_markdown(), ""]
    bl = pw[(pw.variant == "proposed") & (pw.condition == "baseline")].merge(
        info[info.condition == "baseline"][["subject", "state", "ita"]], on=["subject", "state"])
    bl["ita_band"] = bl.ita.apply(face.ita_category)
    by_ita = bl.groupby("ita_band").apply(lambda g: pd.Series(metrics.hr_metrics(g.est.values, g.ref.values)))[["n", "mae", "rmse"]]
    lines += ["## Baseline error by ITA skin-tone band (C2 reporting format)", "",
              "ITA here is measured from uncalibrated video under warm indoor light, so bands are shifted darker "
              "than the subjects' true skin tone; colour calibration is needed before using this for C2 claims.", "",
              by_ita.round(2).to_markdown(), ""]
    by_state = pw[pw.variant.isin(["pos_face", "proposed"])].groupby(["condition", "state", "variant"]).apply(
        lambda g: np.nanmean(np.abs(g.est - g.ref))).unstack().round(2)
    lines += ["## Rest vs post-exercise (MAE bpm)", "", by_state.to_markdown(), ""]
    open("out/summary.md", "w").write("\n".join(lines))
    print("\n".join(lines))

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = pw[(pw.variant == "proposed") & (pw.condition == "baseline")].dropna(subset=["est", "ref"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(g.ref, g.est, s=4, alpha=0.3)
    lo, hi = g.ref.min() - 5, g.ref.max() + 5
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[0].set(xlabel="Reference HR (bpm)", ylabel="Estimated HR (bpm)", title="Proposed, baseline condition")
    mean, diff = (g.est + g.ref) / 2, g.est - g.ref
    b, sd = diff.mean(), diff.std()
    ax[1].scatter(mean, diff, s=4, alpha=0.3)
    for y, ls in [(b, "-"), (b + 1.96 * sd, "--"), (b - 1.96 * sd, "--")]:
        ax[1].axhline(y, color="k", ls=ls, lw=1)
    ax[1].set(xlabel="Mean of methods (bpm)", ylabel="Estimate - reference (bpm)",
              title=f"Bland-Altman: bias {b:.2f}, LoA [{b - 1.96 * sd:.1f}, {b + 1.96 * sd:.1f}]")
    fig.tight_layout()
    fig.savefig("out/bland_altman_baseline.png", dpi=130)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    show = [c for c in CONDITIONS if c in conds]
    x = np.arange(len(show))
    vs = ["mean_predictor", "green_face", "pos_face", "pos_patches", "proposed"]
    for k, v in enumerate(vs):
        vals = [res[(res.condition == c) & (res.variant == v)].mae.values[0] for c in show]
        ax.bar(x + (k - 2) * 0.16, vals, 0.16, label=v)
    ax.set_xticks(x, show)
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("HR error by constraint and method (lower is better)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("out/mae_by_condition.png", dpi=130)


if __name__ == "__main__":
    main()
