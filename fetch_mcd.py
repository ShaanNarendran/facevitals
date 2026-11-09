"""Download a small front-camera subset of MCD-rPPG (CC BY 4.0) for benchmarking."""
import os
import sys
import pandas as pd
from huggingface_hub import hf_hub_download

# python fetch_mcd.py 25        -> subjects 1-25 (sorted ids)
# python fetch_mcd.py 25 40     -> subjects 26-40
START, N = (0, int(sys.argv[1])) if len(sys.argv) < 3 else (int(sys.argv[1]), int(sys.argv[2]))
if not os.path.exists("data/mcd/db.csv"):  # dataset index (subjects, files, health data)
    hf_hub_download("kyegorov/mcd_rppg", "db.csv", repo_type="dataset", local_dir="data/mcd")
db = pd.read_csv("data/mcd/db.csv")
front = db[db.camera == "FullHDwebcam"]
ids = sorted(front.patient_id.unique())[START:N]
rows = front[front.patient_id.isin(ids)]
for _, r in rows.iterrows():
    for col in ("video", "ppg_sync", "meta"):
        try:
            hf_hub_download("kyegorov/mcd_rppg", r[col], repo_type="dataset", local_dir="data/mcd")
        except Exception as e:
            print("ERR", r[col], e, flush=True)
    print("ok", r.video, flush=True)
print("DONE")
