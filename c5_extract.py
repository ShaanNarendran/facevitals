"""Extract traces for the C5 diagnosis on the 25 subjects already used in development.

    python c5_extract.py 0 25 baseline enc_fps10 enc_kbps500 bandwidth

enc_fps10 = 10 fps at high quality (no bitrate cap); enc_kbps500 = 30 fps at
500 kbps; bandwidth = both, as in the brief.
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def job(base, cond, meta):
    import cv2
    cv2.setNumThreads(1)
    from fv import extract
    path = f"cache/{base}__{cond}.npz"
    if not os.path.exists(path):
        np.savez_compressed(path, **extract.extract(f"data/mcd/video/{base}.avi", cond, meta=meta, max_seconds=40))
    return path


def rows_for(lo, hi):
    db = pd.read_csv("data/mcd/db.csv")
    db = db[db.camera == "FullHDwebcam"]
    ids = sorted(db.patient_id.unique())[lo:hi]
    have = db[db.patient_id.isin(ids) & db.video.apply(lambda v: os.path.exists("data/mcd/" + v))]
    return have.sort_values(["patient_id", "step"]).to_dict("records")


if __name__ == "__main__":
    lo, hi, conds = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    todo = [(os.path.basename(r["video"])[:-4], c, "data/mcd/" + r["meta"]) for r in rows_for(lo, hi) for c in conds]
    with ProcessPoolExecutor(5) as ex:
        futs = [ex.submit(job, *t) for t in todo]
        for k, f in enumerate(as_completed(futs), 1):
            f.result()
            if k % 10 == 0 or k == len(futs):
                print(f"extracted {k}/{len(futs)}", flush=True)


def backfill_ftype(path):
    """Add per-frame picture types to a cache made before extract() stored them."""
    from fv import degrade, extract
    tr = dict(np.load(path, allow_pickle=True))
    if "ftype" in tr:
        return
    base, cond = os.path.basename(path)[:-4].split("__")
    enc = extract.enc_params(cond)
    src = extract.encoded_path(f"data/mcd/video/{base}.avi", enc) if enc else f"data/mcd/video/{base}.avi"
    n = len(tr["t"])
    ft = degrade.frame_types(src)[:n]
    tr["ftype"] = np.concatenate([ft, np.full(n - len(ft), "P")])
    np.savez_compressed(path, **tr)
