"""Statistics, LaTeX tables and figures for the Round 1 report.

    python report/stats.py      # after report/ladder.py

Reads saved per-window results only (no re-tuning):
  out/check2_per_window.csv        held-out subjects 61-80 (models trained on 1-60)
  out/dev60_per_window.csv         development subjects 1-60, out-of-fold
  out/report_ladder_per_window.csv development ablation ladder (report/ladder.py)
Writes report/tables/*.tex, report/figs/*.pdf and prints the numbers quoted in the text.
"""
import glob
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fv import face, metrics, quality

COND = {"baseline": "Normal (VGA, 30 fps)", "lowres": "C4: QVGA 320$\\times$240",
        "lowlight50": "C1: 50 lux (sim.)", "lowlight20": "C1: 20 lux (sim.)",
        "multiface": "C3: bystander in frame", "enc_zl_fps10_kbps500": "C5: 10 fps, 500 kbps, our sender",
        "bandwidth": "C5: 10 fps, 500 kbps, default x264", "mask": "C7: surgical mask (sim.)"}
SHORT = {"baseline": "Normal", "lowres": "C4", "lowlight50": "C1 50", "lowlight20": "C1 20",
         "multiface": "C3", "enc_zl_fps10_kbps500": "C5 ours", "bandwidth": "C5 x264", "mask": "C7"}
POS, CLS, FUS = "POS, whole face (standard baseline)", "Classical final (frozen after C5 work)", "Classical + learned (fused)"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6,
                     "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
OUT = {}


def tex(path, body):
    open(os.path.join("report", "tables", path), "w").write(body)


def f(x, d=2):
    return "--" if not np.isfinite(x) else f"{x:.{d}f}"


# ---------------------------------------------------------------- held-out and cross-dataset tables
def condition_table(df, key, path):
    """Per-condition table (as Table I) from a per-window results file; returns the frame."""
    df = df[df.ref.notna()]
    rows = []
    for c in COND:
        g = df[df.cond == c]
        p, k, u = (g[g.method == m] for m in (POS, CLS, FUS))
        m = metrics.hr_metrics(u.est.values, u.ref.values)
        keep = u[u.p_good >= quality.REPORT_P]
        ae = (keep.est - keep.ref).abs()
        rows.append(dict(cond=c, pos=(p.est - p.ref).abs().mean(), cls=(k.est - k.ref).abs().mean(), **m,
                         coverage=100 * len(keep) / len(u), mae_kept=ae.mean(), w5_kept=100 * (ae <= 5).mean()))
    H = pd.DataFrame(rows).set_index("cond")
    OUT[key] = H.round(3).to_dict("index")
    lines = []
    for c, r in H.iterrows():
        best = min(r.pos, r.cls, r.mae)
        cell = lambda v: f"\\textbf{{{f(v)}}}" if abs(v - best) < 1e-9 else f(v)
        lines.append(f"{COND[c]} & {cell(r.pos)} & {cell(r.cls)} & {cell(r.mae)} & {f(r.rmse)} & {f(r.r)} & "
                     f"{f(r.ba_bias)} [{f(r.ba_loa_low, 1)}, {f(r.ba_loa_high, 1)}] & {f(100 * r.within5, 0)} & "
                     f"{f(r.coverage, 0)} & {f(r.mae_kept)} & {f(r.w5_kept, 0)} \\\\")
    tex(path, "\n".join(lines) + "\n")
    return H


ho = pd.read_csv("out/check2_per_window.csv")
ho = ho[ho.ref.notna()]
condition_table(ho, "heldout", "heldout.tex")
OUT["heldout_n"] = dict(videos=int(ho.video.nunique()), subjects=int(ho.subject.nunique()),
                        windows_per_cond=int(len(ho[(ho.cond == "baseline") & (ho.method == FUS)])))
if os.path.exists("out/ubfc_per_window.csv"):
    ub = pd.read_csv("out/ubfc_per_window.csv")
    condition_table(ub, "ubfc", "ubfc.tex")
    b = ub[(ub.cond == "baseline") & (ub.method == FUS) & ub.ref.notna()]
    per = b.assign(ae=(b.est - b.ref).abs()).groupby("subject").ae.mean()
    OUT["ubfc_n"] = dict(subjects=int(ub.subject.nunique()), windows_per_cond=int(len(b)),
                         hr_min=float(b.ref.min()), hr_max=float(b.ref.max()), hr_mean=float(b.ref.mean()),
                         median_subject_mae=float(per.median()), subjects_over_5=per[per > 5].round(2).to_dict())

# ---------------------------------------------------------------- Bland-Altman + scatter
fig, ax = plt.subplots(2, 2, figsize=(3.45, 3.25))
for col, (c, title) in enumerate((("baseline", "Normal"), ("enc_zl_fps10_kbps500", "C5, our sender"))):
    u = ho[(ho.cond == c) & (ho.method == FUS)]
    keep = u.p_good >= quality.REPORT_P
    a = ax[0, col]
    a.plot([40, 160], [40, 160], color=GREY, lw=0.6, ls="--", zorder=1)
    a.scatter(u.ref[~keep], u.est[~keep], s=4, color=ORANGE, marker="x", lw=0.5, label="flagged", zorder=2)
    a.scatter(u.ref[keep], u.est[keep], s=3, color=BLUE, lw=0, alpha=0.6, label="reported", zorder=3)
    a.set(xlim=(45, 130), ylim=(45, 160), title=title)
    a.set_xlabel("Reference HR (bpm)")
    if col == 0:
        a.set_ylabel("Estimated HR (bpm)")
        a.legend(loc="upper left", fontsize=6.5, handletextpad=0.2, markerscale=2)
    m = metrics.hr_metrics(u.est.values, u.ref.values)
    a.text(0.97, 0.04, f"r = {m['r']:.2f}\nMAE = {m['mae']:.2f}", transform=a.transAxes, ha="right", va="bottom", fontsize=6.5)
    b = ax[1, col]
    mean, diff = (u.est + u.ref) / 2, u.est - u.ref
    b.scatter(mean, diff, s=3, color=BLUE, lw=0, alpha=0.6, zorder=3)
    for y, ls in ((m["ba_bias"], "-"), (m["ba_loa_low"], "--"), (m["ba_loa_high"], "--")):
        b.axhline(y, color=GREY, lw=0.6, ls=ls, zorder=1)
        b.text(131, y, f"{y:+.1f}", fontsize=6, va="center", color=GREY)
    b.set(xlim=(45, 130), ylim=(-40, 40))
    b.set_xlabel("Mean of est. and ref. (bpm)")
    if col == 0:
        b.set_ylabel("Est. $-$ ref. (bpm)")
fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.6)
fig.savefig("report/figs/bland_altman.pdf")
plt.close(fig)

# ---------------------------------------------------------------- ablation ladder (development, out-of-fold)
if os.path.exists("out/report_ladder_per_window.csv"):
    L = pd.read_csv("out/report_ladder_per_window.csv")
    L = L[L.ref.notna()].assign(ae=lambda d: (d.est - d.ref).abs())
    T = L.pivot_table(index="method", columns="cond", values="ae", aggfunc="mean")[list(COND)]
    OUT["ladder"] = T.round(2).to_dict("index")
    OUT["ladder_n"] = dict(videos=int(L.video.nunique()), subjects=int(L.subject.nunique()))
    lines = []
    main = [m for m in T.index if not m.startswith("x")]
    for m in T.index:
        if m.startswith("x") and not lines[-1].endswith("\\midrule"):
            lines.append("\\midrule")
        col_best = {c: T.loc[main, c].min() for c in T.columns}
        cells = [f"\\textbf{{{f(v)}}}" if (not m.startswith("x") and abs(v - col_best[c]) < 1e-9) else f(v)
                 for c, v in T.loc[m].items()]
        name = m[2:] if not m.startswith("x") else m[2:].capitalize()
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    tex("ladder.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- C5 spectra figure
if os.path.exists("out/report_spectra.npz"):
    Z = np.load("out/report_spectra.npz")
    from fv import hr
    fig, a = plt.subplots(figsize=(3.45, 2.0))
    curves = [("baseline|5 + mask gate (classical final)", "Normal, 30 fps", GREY, "-"),
              ("bandwidth|x classical final, no codec removal", "C5 default x264, no codec fix", ORANGE, "-"),
              ("bandwidth|5 + mask gate (classical final)", "C5 default x264, codec fix", BLUE, "--"),
              ("enc_zl_fps10_kbps500|5 + mask gate (classical final)", "C5 our sender (no B-frames)", AQUA, ":")]
    for key, lab, col, ls in curves:
        a.plot(hr.BPM, 100 * Z[key], color=col, lw=1.2 if ls != "-" else 1.0, ls=ls, label=lab)
    top = 1.25 * max(100 * Z[k].max() for k, *_ in curves)
    a.axvline(150, color=GREY, lw=0.5, ls=(0, (1, 2)))
    a.text(148, top * 0.97, "2.5 Hz = 150 bpm", fontsize=6.5, color=GREY, ha="right", va="top")
    a.set(xlim=(40, 200), ylim=(0, top), xlabel="Frequency (bpm)", ylabel="Mean power (%)")
    a.legend(fontsize=6.2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig("report/figs/c5_spectra.pdf")
    plt.close(fig)
    i150 = np.abs(hr.BPM - 150) <= 3
    OUT["c5_power_near_150"] = {lab: float(100 * Z[key][i150].sum()) for key, lab, _, _ in curves}

# ---------------------------------------------------------------- strata (C2 ITA, C6 age, sex)
db = pd.read_csv("data/mcd/db.csv")
db = db[db.camera == "FullHDwebcam"].drop_duplicates("patient_id").set_index("patient_id")
dv = pd.read_csv("out/dev60_per_window.csv")
dv = dv[(dv.method == "fused_snr") & (dv.cond == "baseline") & dv.ref.notna()][["subject", "video", "est", "ref", "p_good"]]
hb = ho[(ho.method == FUS) & (ho.cond == "baseline")][["subject", "video", "est", "ref", "p_good"]]
P = pd.concat([dv.assign(split="dev (out-of-fold)"), hb.assign(split="held-out")])
ita = {}
for v in P.video.unique():
    z = np.load(f"cache/{v}__baseline.npz", allow_pickle=True)
    rgb = z["rgb"][:, list(z["regions"]).index("face")]
    ita[v] = face.ita_degrees(np.nanmedian(rgb, axis=0))
P["ita_video"] = P.video.map(ita)
sub_ita = P.groupby("subject").ita_video.mean()
P["ita"] = P.subject.map(sub_ita)
tert = pd.qcut(sub_ita, 3, labels=["lightest third", "middle third", "darkest third"])
P["ita_cat"] = P.subject.map(tert)
OUT["ita_tertile_ranges"] = {str(k): [float(sub_ita[tert == k].min()), float(sub_ita[tert == k].max())] for k in tert.cat.categories}
P["age"] = P.subject.map(db.age)
P["sex"] = P.subject.map(db.sex)
P["age_band"] = pd.cut(P.age, [0, 29, 59, 200], labels=["18--29", "30--59", "$\\geq$60"])
P["ae"] = (P.est - P.ref).abs()
P["keep"] = P.p_good >= quality.REPORT_P
pop = P.ae.mean()
OUT["strata_pop_mae"] = pop
OUT["ita_range"] = [float(sub_ita.min()), float(sub_ita.max())]
lines, strata = [], []
order_ita = ["lightest third", "middle third", "darkest third"]
for label, col, levels in (("Measured ITA, relative (uncalibrated)", "ita_cat", order_ita),
                           ("Age (years)", "age_band", ["18--29", "30--59", "$\\geq$60"]),
                           ("Sex", "sex", ["F", "M"])):
    lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{label}}}}} \\\\")
    for lv in levels:
        g = P[P[col] == lv]
        if g.empty:
            lines.append(f"\\quad {lv} & 0 & -- & -- & -- & -- & -- \\\\")
            continue
        k = g[g.keep]
        strata.append(dict(group=col, level=str(lv), subjects=int(g.subject.nunique()), mae=g.ae.mean(),
                           dev=g.ae.mean() - pop, cov=100 * g.keep.mean(), mae_kept=k.ae.mean()))
        lines.append(f"\\quad {lv} & {g.subject.nunique()} & {f(g.ae.mean())} & {g.ae.mean() - pop:+.2f} & "
                     f"{f(100 * (g.ae <= 5).mean(), 0)} & {f(100 * g.keep.mean(), 0)} & {f(k.ae.mean())} \\\\")
lines.append("\\midrule")
lines.append(f"All & {P.subject.nunique()} & {f(pop)} & -- & {f(100 * (P.ae <= 5).mean(), 0)} & "
             f"{f(100 * P.keep.mean(), 0)} & {f(P[P.keep].ae.mean())} \\\\")
tex("strata.tex", "\n".join(lines) + "\n")
OUT["strata"] = strata
old = P[P.age >= 60].drop_duplicates("subject")[["subject", "age", "split"]]
OUT["older_subjects"] = old.to_dict("records")
OUT["per_subject_mid_age"] = P[(P.age > 29) & (P.age < 60)].groupby(["subject", "age"]).ae.mean().round(2).reset_index().to_dict("records")
OUT["ita_by_subject_cat_counts"] = {str(k): int(v) for k, v in pd.Series(sub_ita.apply(face.ita_category)).value_counts().items()}

# ---------------------------------------------------------------- dataset summary
def describe(ids, P_ref):
    s = db.loc[sorted(ids)]
    return dict(subjects=len(s), female=int((s.sex == "F").sum()), age_median=float(s.age.median()),
                age_min=float(s.age.min()), age_max=float(s.age.max()),
                hr_min=float(P_ref.ref.min()), hr_max=float(P_ref.ref.max()), hr_mean=float(P_ref.ref.mean()))
OUT["data_dev"] = describe(set(dv.subject), dv)
OUT["data_held"] = describe(set(hb.subject), hb)

# ---------------------------------------------------------------- C3: did the tracker stay on the target?
same, found, two = [], [], []
for v in ho.video.unique():
    b = np.load(f"cache/{v}__baseline.npz", allow_pickle=True)
    m = np.load(f"cache/{v}__multiface.npz", allow_pickle=True)
    fi = list(b["regions"]).index("face")
    n = min(len(b["t"]), len(m["t"]))
    rb, rm = b["rgb"][:n, fi], m["rgb"][:n, fi]
    ok = np.all(np.isfinite(rb), axis=1) & np.all(np.isfinite(rm), axis=1)
    rel = np.abs(rm[ok] - rb[ok]).max(axis=1) / rb[ok].mean(axis=1)
    same.append((rel < 0.05).mean())
    found.append(m["found"][:n].mean())
    two.append((m["nfaces"][:n] >= 2).mean())
OUT["c3"] = dict(videos=len(same), frac_frames_target_matches_clean=float(np.mean(same)),
                 worst_video_match=float(np.min(same)), target_found=float(np.mean(found)),
                 frames_two_faces=float(np.mean(two)))

# ---------------------------------------------------------------- C7: mask coverage and detection
import cv2
lm = face.make_landmarker(num_faces=1)
cov_face, cov_low = [], []
for k, v in enumerate(sorted(ho.video.unique())[::4]):
    cap = cv2.VideoCapture(f"data/mcd/video/{v}.avi")
    ok, fr = cap.read()
    cap.release()
    pts = face.detect_faces(lm, fr, 1000 * (k + 1))
    if not pts:
        continue
    p = pts[0]
    oval = np.zeros(fr.shape[:2], np.uint8)
    mask = np.zeros(fr.shape[:2], np.uint8)
    cv2.fillPoly(oval, [np.round(p[face.FACE_OVAL]).astype(np.int32)], 1)
    cv2.fillPoly(mask, [np.round(p[face.MASK_POLY]).astype(np.int32)], 1)
    y_mid = (p[face.FACE_OVAL][:, 1].min() + p[face.FACE_OVAL][:, 1].max()) / 2
    low = oval.copy()
    low[: int(y_mid)] = 0
    cov_face.append((oval & mask).sum() / oval.sum())
    cov_low.append((low & mask).sum() / low.sum())
lm.close()
dm = pd.read_csv("out/dev60_per_window.csv", usecols=["cond", "method", "masked"])
dm = dm[dm.method == "classical"]
OUT["c7"] = dict(frames=len(cov_face), face_covered=float(np.mean(cov_face)), lower_half_covered=float(np.mean(cov_low)),
                 detected_in_mask=float(dm[dm.cond == "mask"].masked.mean()),
                 false_alarm_other=float(dm[dm.cond != "mask"].masked.mean()))

# ---------------------------------------------------------------- runtime
ms30 = [float(np.load(p, allow_pickle=True)["ms_per_frame"]) for p in glob.glob("cache/*__baseline.npz")]
ms10 = [float(np.load(p, allow_pickle=True)["ms_per_frame"]) for p in glob.glob("cache/*__enc_zl_fps10_kbps500.npz")]
from fv import hr, learned, pipeline
br = learned.Branch(["models/pulsenet_all.pt"])
z = dict(np.load("cache/2003_FullHDwebcam_after__baseline.npz", allow_pickle=True))
regions = list(z["regions"])
sel = z["t"] <= z["t"][0] + 12
args = (z["t"][sel], z["rgb"][sel], regions)
pipeline.latest_fused_spectrum(*args, learned=br)
tic = time.perf_counter()
for _ in range(50):
    S, *_ = pipeline.latest_fused_spectrum(*args, learned=br)
t_back = (time.perf_counter() - tic) / 50 * 1000
trk = hr.OnlineTracker(1.0)
tic = time.perf_counter()
for _ in range(1000):
    trk.step(S)
t_trk = (time.perf_counter() - tic)
OUT["runtime_ms"] = dict(front_per_frame_30fps=float(np.median(ms30)), front_per_frame_10fps=float(np.median(ms10)),
                         backend_per_estimate=t_back, tracker_per_step=t_trk,
                         params_learned=int(sum(p.numel() for p in br.models[0].parameters())))
print(json.dumps(OUT, indent=1, default=float))
json.dump(OUT, open("report/numbers.json", "w"), indent=1, default=float)
