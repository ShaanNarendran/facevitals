"""Cross-dataset check on UBFC-rPPG DATASET_2 with the frozen pipeline.

    python ubfc.py fetch            # download each subject, cache traces for every condition, delete the raw video
    python ubfc.py eval             # -> out/ubfc_per_window.csv, out/ubfc_summary.md

UBFC-rPPG (Bobbia et al., Pattern Recognit. Lett. 2019) was recorded in another
lab with another webcam (Logitech C920, 640x480 uncompressed, ~30 fps), under
varying sunlight and room light, while subjects played a time-pressured maths
game. The reference is a CMS50E transmissive pulse oximeter, one PPG sample per
video frame. 42 subjects are shared for research use; the faces of subjects 21
and 27 must not appear in any publication, so no UBFC frame is shown anywhere.

Nothing is re-trained or re-tuned: models, abstention threshold and settings
are the ones frozen on MCD-rPPG subjects 1-60. Raw videos are ~1.8 GB each
(75 GB in all), downloaded from the authors' OneDrive share (their Google Drive
copy hits its download quota) and deleted once their traces are cached.
"""
import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

SHARE = ("https://univbourgogne-my.sharepoint.com/:f:/g/personal/yannick_benezeth_u-bourgogne_fr/"
         "IgDQvHIcDjiMSrzknzArGE25AVSjt4fjw11QgLpsJX2QPec?e=ySQtZt")
SITE = "https://univbourgogne-my.sharepoint.com/personal/yannick_benezeth_u-bourgogne_fr"
ROOT = "/personal/yannick_benezeth_u-bourgogne_fr/Documents/SharedDatasets/UBFC_DATASET/DATASET_2"
CONDS = {"baseline": "Normal (VGA, 30 fps)", "lowres": "C4: 320x240", "lowlight50": "C1: ~50 lux (simulated)",
         "lowlight20": "C1 stress: ~20 lux (simulated)", "multiface": "C3: bystander in frame",
         "enc_zl_fps10_kbps500": "C5, our sender (every 3rd frame, x264 zerolatency)",
         "bandwidth": "C5: 10 fps + 500 kbps (default x264)", "mask": "C7: surgical mask (simulated)"}
BYSTANDER = "data/mcd/video/2003_FullHDwebcam_before.avi"  # another person's face (MCD-rPPG)


def listing():
    """Subjects and file sizes of the authors' OneDrive share (DATASET_2), saved in ubfc_files.tsv."""
    rows = [l.rstrip("\n").split("\t") for l in open("ubfc_files.tsv") if l.strip()]
    sizes = {(s, f): int(n) for s, f, n in rows}
    return sorted({s for s, _, _ in rows}, key=lambda s: int(s[7:])), sizes


def name(sub: str) -> str:
    return f"ubfc_s{int(sub[7:]):02d}"


def ground_truth(base: str):
    """(PPG per video frame, frame times in s) from a DATASET_2 ground_truth.txt."""
    lines = [l for l in open(f"data/ubfc/gt/{base}.txt").read().splitlines() if l.strip()]
    ppg, _sensor_hr, t = (np.array(l.split(), float) for l in lines[:3])
    return ppg, t


def download(sub: str, fname: str, dst: str, size: int):
    import requests
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return
    for attempt in range(5):
        try:
            s = requests.Session()
            s.get(SHARE, timeout=60)  # anonymous guest cookie for the shared folder
            r = s.get(f"{SITE}/_layouts/15/download.aspx", params={"SourceUrl": f"{ROOT}/{sub}/{fname}"},
                      stream=True, timeout=120)
            r.raise_for_status()
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
            if os.path.getsize(tmp) != size:
                raise IOError(f"got {os.path.getsize(tmp)} bytes, expected {size}")
            os.replace(tmp, dst)
            return
        except Exception as e:  # network hiccup or expired guest cookie: retry from scratch
            print(f"  {sub}/{fname}: attempt {attempt + 1} failed ({e})", flush=True)
            time.sleep(30)
    raise RuntimeError(f"could not download {sub}/{fname}")


def fetch_one(sub: str, sizes: dict):
    import cv2
    cv2.setNumThreads(1)
    from fv import extract
    base = name(sub)
    todo = [c for c in CONDS if not os.path.exists(f"cache/{base}__{c}.npz")]
    if not todo:
        return base, "already cached"
    download(sub, "ground_truth.txt", f"data/ubfc/gt/{base}.txt", sizes[(sub, "ground_truth.txt")])
    video = f"data/ubfc/video/{base}.avi"
    download(sub, "vid.avi", video, sizes[(sub, "vid.avi")])
    _, t = ground_truth(base)
    for c in todo:
        tr = extract.extract(video, c, stamps=t, bystander_video=BYSTANDER if c == "multiface" else None)
        np.savez_compressed(f"cache/{base}__{c}.npz", **tr)
    os.remove(video)
    return base, f"{len(todo)} conditions cached"


def eval_one(base: str):
    from heldout import METHODS
    from fv import learned, pipeline, quality
    branch = learned.Branch(["models/pulsenet_all.pt"])
    abstainers = {v: quality.Abstainer.load(f"models/abstainer_{v}.pkl") for v in ("final", "final_dl")}
    ppg, _ = ground_truth(base)
    tb = np.load(f"cache/{base}__baseline.npz")["t"]
    recs = []
    for c in CONDS:
        tr = dict(np.load(f"cache/{base}__{c}.npz", allow_pickle=True))
        tr["regions"] = list(tr["regions"])
        for meth, kw in METHODS.items():
            if kw is None:
                continue
            kw = dict(kw)
            variant = kw.pop("variant")
            res = pipeline.estimate(tr, variant, t_end=min(tb[-1], tr["t"][-1]), learned=branch, **kw)
            ref = pipeline.reference_hr(tb, ppg, res["t"])
            est = np.where(res["valid"] | (variant in ("final", "final_dl", "dl")), res["hr"], np.nan)
            pg = (abstainers[variant].p_good(quality.window_features(res)) if variant in abstainers
                  else np.full(len(est), np.nan))
            for t, e, rf, p in zip(res["t"], est, ref, pg):
                recs.append(dict(subject=base, cond=c, method=meth, t=t, est=e, ref=rf, p_good=p))
    return recs


def summarise(pw: pd.DataFrame) -> str:
    from heldout import FINAL, METHODS
    from fv import metrics, quality
    pw = pw[pw.ref.notna()].assign(ae=lambda d: (d.est - d.ref).abs())
    order = list(CONDS)
    names = [m for m, k in METHODS.items() if k]
    pooled = pw.groupby(["method", "cond"]).ae.mean().unstack()[order].reindex(names).rename(columns=CONDS)
    full, abst = [], []
    for c in order:
        g = pw[(pw.method == FINAL) & (pw.cond == c)]
        full.append(dict(condition=CONDS[c], **metrics.hr_metrics(g.est.values, g.ref.values)))
        k = g[g.p_good >= quality.REPORT_P]
        abst.append(dict(condition=CONDS[c], coverage=100 * len(k) / len(g), mae_kept=k.ae.mean(),
                         within5_kept=100 * (k.ae <= 5).mean(), mae_all=g.ae.mean()))
    per = pw[(pw.cond == "baseline")].groupby(["subject", "method"]).ae.mean().unstack()[names]
    per["ref_mean"] = pw[(pw.cond == "baseline") & (pw.method == FINAL)].groupby("subject").ref.mean()
    return "\n".join([
        "# UBFC-rPPG DATASET_2 (cross-dataset, frozen pipeline)", "",
        f"{pw.subject.nunique()} subjects, whole videos, 10 s windows / 1 s hop, reference = contact PPG "
        "(CMS50E) spectral peak in the same windows. Models and settings frozen on MCD-rPPG subjects 1-60.", "",
        "## Pooled MAE (bpm), all windows", "", pooled.round(2).to_markdown(), "",
        "## Final method, full metrics", "", pd.DataFrame(full).round(2).to_markdown(index=False), "",
        f"## With signal-quality abstention (P >= {quality.REPORT_P})", "",
        pd.DataFrame(abst).round(2).to_markdown(index=False), "",
        "## Per-subject MAE, normal condition", "", per.round(2).to_markdown(), ""])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fetch", "eval"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    for d in ("data/ubfc/gt", "data/ubfc/video", "cache", "out"):
        os.makedirs(d, exist_ok=True)
    subs, sizes = listing()
    if args.mode == "fetch":
        with ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(fetch_one, s, sizes): s for s in subs}
            for k, f in enumerate(as_completed(futs), 1):
                base, msg = f.result()
                print(f"{k}/{len(subs)} {base}: {msg}", flush=True)
    else:
        done = [name(s) for s in subs if all(os.path.exists(f"cache/{name(s)}__{c}.npz") for c in CONDS)]
        print(f"evaluating {len(done)} subjects", flush=True)
        with ProcessPoolExecutor(6) as ex:
            pw = pd.DataFrame([r for part in ex.map(eval_one, done) for r in part])
        pw.to_csv("out/ubfc_per_window.csv", index=False)
        text = summarise(pw)
        open("out/ubfc_summary.md", "w").write(text)
        print(text)
