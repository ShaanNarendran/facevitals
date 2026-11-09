"""Ablation ladder for the report: each pipeline stage added in turn.

    python report/ladder.py      # subjects 1-60 (development), out-of-fold models

Development subjects only: the learned branch for each subject is the
cross-validation model that did not see it (models/pulsenet_fold{k}.pt).
Writes out/report_ladder_per_window.csv and out/report_spectra.npz (mean
normalised spectra for the C5 figure).
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from c5_extract import rows_for

CONDS = ["baseline", "lowres", "lowlight50", "lowlight20", "multiface", "enc_zl_fps10_kbps500", "bandwidth", "mask"]
LADDER = [  # (name, estimate kwargs)
    ("1 POS, spectral peak", dict(variant="pos_face", illum=False, codec=False)),
    ("2 + Viterbi tracking", dict(variant="pos_face_viterbi", illum=False, codec=False)),
    ("3 + illumination cancellation", dict(variant="pos_face_viterbi", illum=True, codec=False)),
    ("4 + codec-pattern removal", dict(variant="pos_face_viterbi", illum=True, codec=True)),
    ("5 + mask gate (classical final)", dict(variant="final")),
    ("6 + learned branch (proposed)", dict(variant="final_dl", fusion="snr")),
    ("x classical final, harmonic boost on", dict(variant="final", beta=1.0)),
    ("x classical final, no codec removal", dict(variant="final", codec=False)),
]
SPEC_FOR = {"5 + mask gate (classical final)", "x classical final, no codec removal"}


def run(base, pid, fold, ppg_path):
    from fv import learned, pipeline
    ppg = np.array([float(l.split()[0]) for l in open(ppg_path) if l.strip()])
    tb = np.load(f"cache/{base}__baseline.npz")["t"]
    branch = learned.Branch([f"models/pulsenet_fold{fold}.pt"])
    rows, spectra = [], {}
    for c in CONDS:
        path = f"cache/{base}__{c}.npz"
        if not os.path.exists(path):
            continue
        tr = dict(np.load(path, allow_pickle=True))
        tr["regions"] = list(tr["regions"])
        for name, kw in LADDER:
            res = pipeline.estimate(tr, t_end=min(tb[-1], tr["t"][-1]), learned=branch, **kw)
            ref = pipeline.reference_hr(tb, ppg, res["t"])
            for i in range(len(ref)):
                rows.append(dict(subject=pid, fold=fold, video=base, cond=c, method=name, t=res["t"][i],
                                 est=res["hr"][i], ref=ref[i]))
            if name in SPEC_FOR and c in ("baseline", "bandwidth", "enc_zl_fps10_kbps500"):
                S = res["spec"][res["valid"]]
                S = S / (S.sum(axis=1, keepdims=True) + 1e-12)
                spectra[(c, name)] = S.mean(axis=0)
    return rows, spectra


if __name__ == "__main__":
    rows = rows_for(0, 60)
    subjects = sorted({r["patient_id"] for r in rows})
    todo = [(os.path.basename(r["video"])[:-4], r["patient_id"], subjects.index(r["patient_id"]) % 5,
             "data/mcd/" + r["ppg_sync"]) for r in rows]
    out, spec = [], {}
    with ProcessPoolExecutor(5) as ex:
        for k, (r, s) in enumerate(ex.map(run, *zip(*todo))):
            out += r
            for key, v in s.items():
                spec.setdefault(key, []).append(v)
            print(f"{k + 1}/{len(todo)}", flush=True)
    pd.DataFrame(out).to_csv("out/report_ladder_per_window.csv", index=False)
    np.savez("out/report_spectra.npz", **{f"{c}|{n}": np.mean(v, axis=0) for (c, n), v in spec.items()})
    print("done")
