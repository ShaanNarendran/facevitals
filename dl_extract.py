"""Full-length (~3 min) baseline traces for training the learned branch.

    python dl_extract.py 0 40      # subjects 1-40 -> cache/<video>__full.npz

Evaluation still uses the first 40 s of each video; the rest is extra
training material from the same development subjects.
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")

import numpy as np

from c5_extract import rows_for


def job(base, meta):
    import cv2
    cv2.setNumThreads(1)
    from fv import extract
    path = f"cache/{base}__full.npz"
    if not os.path.exists(path):
        np.savez_compressed(path, **extract.extract(f"data/mcd/video/{base}.avi", "baseline", meta=meta))
    return path


if __name__ == "__main__":
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    todo = [(os.path.basename(r["video"])[:-4], "data/mcd/" + r["meta"]) for r in rows_for(lo, hi)]
    with ProcessPoolExecutor(5) as ex:
        futs = [ex.submit(job, *t) for t in todo]
        for k, f in enumerate(as_completed(futs), 1):
            f.result()
            print(f"extracted {k}/{len(futs)}", flush=True)
