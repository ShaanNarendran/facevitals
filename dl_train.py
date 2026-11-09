"""Train the learned pulse branch on the development subjects (default 1-60).

    python dl_train.py cv                 # 5 folds by subject -> models/pulsenet_fold{k}.pt, out/dl_cv.csv
    python dl_train.py all                # all dev subjects   -> models/pulsenet_all.pt
    python dl_train.py cv --stretch wide --balance --tag wide_bal  # ablations, saved under that tag

Uses the full ~3 min of each baseline video (cache/<video>__full.npz from
dl_extract.py). Each video is pre-stretched in time by a few factors (the
label scales with it), so high heart rates, which are rare, are better
covered. Windows are sampled at random each step, with region dropout and
added noise so the model does not rely on any single patch.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

from fv import hr, learned, pipeline, signals
from c5_extract import rows_for

STRETCH = {
    "base": (0.8, 0.9, 1.0, 1.12, 1.25),
    # Tried and rejected (subjects 1-60, out-of-fold): wider stretching, with
    # or without heart-rate-balanced sampling, made errors below 55 bpm worse
    # (10.9 -> 12.5 / 13.2 bpm) and overall error slightly worse.
    "wide": (0.65, 0.75, 0.85, 1.0, 1.15, 1.3, 1.5),
}
SCALES = STRETCH["base"]
BALANCE = False  # --balance: sample windows inversely to how common their heart rate is
STEPS, BATCH, LR = 4000, 64, 2e-3
FOLDS = 5
WIN = learned.WIN
PPG_MIN_SNR_DB = -3.0  # skip windows whose contact PPG is itself unclear
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def video_arrays(base, ppg_path, meta_path):
    from fv import extract
    tr = dict(np.load(f"cache/{base}__full.npz", allow_pickle=True))
    grid, rgb_u, regions = pipeline.prepare(tr)
    X = learned.features(rgb_u, regions)
    ppg = np.array([float(l.split()[0]) for l in open(ppg_path) if l.strip()])
    t = tr["t"]
    n = min(len(t), len(ppg))
    _, p = signals.resample_uniform(t[:n], ppg[:n].astype(float), learned.FS)
    p = signals.bandpass(signals._fill_nan(p)[:, 0], learned.FS)
    m = min(X.shape[1], len(p))
    bad = ~np.all(np.isfinite(rgb_u[:m, regions.index("face")]), axis=1)
    return X[:, :m], p[:m], bad


def stretched(X, p, bad, scale):
    """Play the traces `scale` times faster: HR in the result is scale x HR."""
    idx = np.arange(0, X.shape[1] - 1, scale)
    i0 = idx.astype(int)
    w = (idx - i0).astype(np.float32)
    Xs = X[:, i0] * (1 - w) + X[:, i0 + 1] * w
    ps = p[i0] * (1 - w) + p[i0 + 1] * w
    bs = bad[i0] | bad[i0 + 1]
    starts, labels = [], []
    for s in range(0, len(ps) - WIN, 15):
        if bs[s:s + WIN].any():
            continue
        S = hr.spectrum(ps[s:s + WIN], learned.FS)
        h = hr.BPM[np.argmax(S)]
        if h <= hr.BPM[0] + 1 or h >= hr.BPM[-1] - 1 or hr.snr_db(S, h) < PPG_MIN_SNR_DB:
            continue
        starts.append(s)
        labels.append(h)
    return Xs.astype(np.float32), np.array(starts), np.array(labels, np.float32)


def load_all(rows):
    data = []
    for r in rows:
        base = os.path.basename(r["video"])[:-4]
        if not os.path.exists(f"cache/{base}__full.npz"):
            continue
        X, p, bad = video_arrays(base, "data/mcd/" + r["ppg_sync"], "data/mcd/" + r["meta"])
        for sc in SCALES:
            Xs, st, lab = stretched(X, p, bad, sc)
            if len(st):
                data.append(dict(subject=r["patient_id"], base=base, scale=sc, X=Xs, starts=st, labels=lab))
    return data


def flat_index(data, balance):
    """All (set, window) pairs and their sampling probabilities.

    With balance, each 5-bpm label band gets equal total probability (bands
    with fewer than 20 windows are capped at 20 so a handful of windows is not
    replayed endlessly).
    """
    si = np.concatenate([np.full(len(d["starts"]), k) for k, d in enumerate(data)])
    wi = np.concatenate([np.arange(len(d["starts"])) for d in data])
    lab = np.concatenate([d["labels"] for d in data])
    if not balance:
        return si, wi, np.full(len(si), 1 / len(si))
    band = (lab // 5).astype(int)
    cnt = np.bincount(band)
    p = 1.0 / np.maximum(cnt[band], 20)
    return si, wi, p / p.sum()


def batch(data, rng, index, n=BATCH, augment=True):
    si, wi, p = index
    pick = rng.choice(len(si), size=n, p=p)
    xs, ys = [], []
    for k in pick:
        d = data[si[k]]
        j = wi[k]
        x = d["X"][:, d["starts"][j]:d["starts"][j] + WIN].copy()
        if augment:
            if rng.random() < 0.5:
                x = -x
            for r in range(len(learned.REGIONS)):
                if rng.random() < 0.15:
                    x[3 * r:3 * r + 3] = 0
            x[:3 * len(learned.REGIONS)] += rng.normal(0, rng.uniform(0, 0.7), (3 * len(learned.REGIONS), WIN))
        xs.append(x)
        ys.append(d["labels"][j])
    return torch.tensor(np.stack(xs)), torch.tensor(np.array(ys))


def train(data, seed=0, log=None, balance=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    index = flat_index(data, BALANCE if balance is None else balance)
    model = learned.build_model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS)
    basis = tuple(b.to(DEV) for b in learned.spectral_basis())
    model.train()
    t0 = time.time()
    for step in range(STEPS):
        x, y = batch(data, rng, index)
        x, y = x.to(DEV), y.to(DEV)
        loss = learned.loss_fn(model(x), y, basis)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (step + 1) % 500 == 0:
            log(f"    step {step + 1}/{STEPS} loss {loss.item():.3f} ({time.time() - t0:.0f} s)")
    model.eval()
    return model.cpu()


def evaluate(model, data):
    """Peak-HR error on unstretched windows (1 s hop) of held-out videos."""
    basis = learned.spectral_basis()
    out = []
    for d in data:
        if d["scale"] != 1.0:
            continue
        keep = np.arange(0, len(d["starts"]), 2)
        x = torch.tensor(np.stack([d["X"][:, d["starts"][j]:d["starts"][j] + WIN] for j in keep]))
        with torch.no_grad():
            P = learned.power(model(x), basis).numpy()
        pos = x[:, 3 * len(learned.REGIONS)].numpy()
        for k, j in enumerate(keep):
            out.append(dict(subject=d["subject"], video=d["base"], ref=d["labels"][j],
                            dl=hr.BPM[np.argmax(P[k])], pos=hr.BPM[np.argmax(hr.spectrum(pos[k], learned.FS))]))
    return pd.DataFrame(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv", "all"])
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=60)
    ap.add_argument("--stretch", choices=list(STRETCH), default="base")
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--tag", default="", help="model/result name suffix for ablations")
    args = ap.parse_args()
    SCALES = STRETCH[args.stretch]
    BALANCE = args.balance
    sfx = f"_{args.tag}" if args.tag else ""
    torch.set_num_threads(4)
    os.makedirs("models", exist_ok=True)
    rows = rows_for(args.lo, args.hi)
    subjects = sorted({r["patient_id"] for r in rows})
    print(f"loading {len(rows)} videos, stretch {SCALES}, balance {BALANCE} ...", flush=True)
    data = load_all(rows)
    n_win = sum(len(d["starts"]) for d in data if d["scale"] == 1.0)
    print(f"{len(data)} (video, stretch) sets, {n_win} unstretched windows at 0.5 s hop", flush=True)
    if args.mode == "all":
        model = train(data, log=lambda s: print(s, flush=True))
        torch.save(model.state_dict(), f"models/pulsenet{sfx}_all.pt")
        print(f"saved models/pulsenet{sfx}_all.pt")
        sys.exit()
    res = []
    for k in range(FOLDS):
        val_s = set(subjects[k::FOLDS])
        tr_d = [d for d in data if d["subject"] not in val_s]
        va_d = [d for d in data if d["subject"] in val_s]
        print(f"fold {k}: train {len({d['subject'] for d in tr_d})} subjects, validate {len(val_s)}", flush=True)
        model = train(tr_d, seed=k)
        torch.save(model.state_dict(), f"models/pulsenet{sfx}_fold{k}.pt")
        ev = evaluate(model, va_d).assign(fold=k)
        print(f"    val MAE: learned {np.abs(ev.dl - ev.ref).mean():.2f}  POS peak {np.abs(ev.pos - ev.ref).mean():.2f}", flush=True)
        res.append(ev)
    res = pd.concat(res)
    res.to_csv(f"out/dl_cv{sfx}.csv", index=False)
    res["band"] = pd.cut(res.ref, [0, 55, 60, 70, 80, 90, 100, 110, 200])
    by = res.assign(learned=(res.dl - res.ref).abs(), pos=(res.pos - res.ref).abs()).groupby("band", observed=True)[["learned", "pos"]].mean()
    by["n"] = res.groupby("band", observed=True).size()
    print(by.round(2).to_string())
    per = res.assign(e_dl=(res.dl - res.ref).abs(), e_pos=(res.pos - res.ref).abs()).groupby("video")[["e_dl", "e_pos"]].mean()
    print("all folds, window MAE: learned %.2f  POS %.2f;  median per-video: learned %.2f  POS %.2f" % (
        np.abs(res.dl - res.ref).mean(), np.abs(res.pos - res.ref).mean(), per.e_dl.median(), per.e_pos.median()))
