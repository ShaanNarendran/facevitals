"""Round 1 processing script: video + reference -> SP Cup submission files.

    python process_round1.py --video data/mcd/video/1020_FullHDwebcam_before.avi \
        --ppg data/mcd/ppg_sync/1020_FullHDwebcam_before.txt \
        --meta data/mcd/meta/1020_FullHDwebcam_before.txt \
        --condition Baseline --constraint None --seconds 60 --age 23

Writes submission/Round1_<Condition>/ with:
  Round1_<Condition>_<Constraint>.mat   facial_video, roi_data, reference_signal,
                                        processed_signal, metrics, params
  target_video.mp4, reference_signal.csv, processed_output.csv

`facial_video` holds the tracked target-face crop (72x72 RGB per frame) rather
than full frames, to keep files small; pass --crop 0 to skip it or a larger size.
"""
import argparse
import os
import shutil
import subprocess

os.environ.setdefault("GLOG_minloglevel", "2")

import numpy as np
import pandas as pd
from scipy.io import savemat

from fv import extract, face, learned, metrics, pipeline, quality, signals

CONDITION_TO_SIM = {"Baseline": "baseline", "LowLight": "lowlight50", "LowResolution": "lowres",
                    "LowBandwidth": "enc_zl_fps10_kbps500", "Masked": "mask", "MultiFace": "multiface"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ppg", required=True, help="reference PPG, one value per video frame (first column)")
    ap.add_argument("--meta", help="per-frame timestamps file (MCD-rPPG format)")
    ap.add_argument("--condition", default="Baseline", choices=list(CONDITION_TO_SIM))
    ap.add_argument("--constraint", default="None", help="e.g. C1, C4, C7 or None")
    ap.add_argument("--bystander", help="second video for MultiFace")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--crop", type=int, default=72)
    ap.add_argument("--age", type=float, default=np.nan)
    ap.add_argument("--lux", type=float, default=np.nan)
    ap.add_argument("--distance-cm", type=float, default=np.nan)
    ap.add_argument("--out", default="submission")
    args = ap.parse_args()

    sim = CONDITION_TO_SIM[args.condition]
    tr = extract.extract(args.video, sim, meta=args.meta, bystander_video=args.bystander,
                         max_seconds=args.seconds, keep_crops=args.crop)
    base_t = tr["t"] if not extract.enc_params(sim) else extract.extract(
        args.video, "baseline", meta=args.meta, max_seconds=args.seconds)["t"]
    branch = learned.Branch(["models/pulsenet_all.pt"]) if os.path.exists("models/pulsenet_all.pt") else None
    variant = "final_dl" if branch else "final"
    res = pipeline.estimate(tr, variant, t_end=min(base_t[-1], tr["t"][-1]), learned=branch)
    ab_path = f"models/abstainer_{variant}.pkl"
    p_good = (quality.Abstainer.load(ab_path).p_good(quality.window_features(res)) if os.path.exists(ab_path)
              else np.full(len(res["t"]), np.nan))
    reported = p_good >= quality.REPORT_P

    ppg = np.array([float(l.split()[0]) for l in open(args.ppg) if l.strip()])
    ref_hr = pipeline.reference_hr(base_t, ppg, res["t"])
    m = metrics.hr_metrics(res["hr"], ref_hr)

    # Continuous pulse waveform from the highest-weighted patch (for plots / judges).
    regions = list(tr["regions"])
    rgb = pipeline.remove_codec_pattern(tr["rgb"], tr["ftype"])
    grid, rgb_u = signals.resample_uniform(tr["t"], rgb.reshape(len(tr["t"]), -1), pipeline.FS)
    rgb_u = rgb_u.reshape(len(grid), len(regions), 3)
    rgb_u = pipeline.cancel_illumination(rgb_u, regions)
    best = "pos:face"
    pulse = signals.pos(rgb_u[:, regions.index("face")], pipeline.FS)

    skin = np.nanmean(tr["rgb"][:, [regions.index(p) for p in face.UPPER_PATCHES]], axis=(0, 1))
    ita = face.ita_degrees(skin)
    h, w = (480, 1280) if sim == "multiface" else ((240, 320) if sim == "lowres" else (480, 640))

    name = f"Round1_{args.condition}"
    outdir = os.path.join(args.out, name)
    os.makedirs(outdir, exist_ok=True)
    n = min(len(base_t), len(ppg))

    mat = {
        "facial_video": tr.get("crops", np.zeros((0,), np.uint8)),
        "roi_data": {
            "region_names": np.array(regions, dtype=object),
            "patch_landmark_indices": {k: np.array(v) + 1 for k, v in face.PATCHES.items()},  # 1-based for MATLAB
            "face_boxes_xyxy": tr.get("boxes", np.zeros((0, 4))),
            "landmarks_xy": tr.get("landmarks", np.zeros((0, 468, 2))),
            "mean_rgb": tr["rgb"],
            "pixel_counts": tr["counts"],
            "frame_time_s": tr["t"],
        },
        "reference_signal": {"ppg": ppg[:n], "time_s": base_t[:n], "hr_bpm": ref_hr, "hr_time_s": res["t"],
                             "device": "contact PPG (dataset reference)"},
        "processed_signal": {"hr_bpm": res["hr"], "time_s": res["t"], "snr_db": res["snr_db"],
                             "pulse_waveform": pulse, "pulse_time_s": grid, "pulse_source": best,
                             "mask_detected": res["mask_detected"], "p_good": p_good, "reported": reported},
        "metrics": {**{k: float(v) for k, v in m.items()}, "window_s": pipeline.WIN_S, "hop_s": pipeline.HOP_S,
                    "coverage": float(reported.mean()),
                    **{f"{k}_reported": float(v) for k, v in metrics.hr_metrics(res["hr"][reported], ref_hr[reported]).items()}},
        "params": {"resolution_wh": np.array([w, h]), "frame_rate": float(tr["fps"]),
                   "illuminance_lux": args.lux, "camera_distance_cm": args.distance_cm,
                   "ita_deg": ita, "skin_tone_ita_band": face.ita_category(ita),
                   "skin_tone_note": "ITA from uncalibrated video; not a clinical Fitzpatrick rating",
                   "subject_age": args.age, "condition": args.condition, "constraints": args.constraint,
                   "simulation": sim, "method": ("codec-pattern removal + illumination cancellation; whole-face POS SNR-fused with a learned "
                                 "pulse branch, Viterbi tracking; mask -> fused unoccluded patches; quality-gated"),
                   "ms_per_frame": float(tr["ms_per_frame"])},
    }
    savemat(os.path.join(outdir, f"{name}_{args.constraint}.mat"), mat, do_compression=True, long_field_names=True)

    pd.DataFrame({"time_s": base_t[:n], "ppg": ppg[:n]}).to_csv(os.path.join(outdir, "reference_signal.csv"), index=False)
    pd.DataFrame({"time_s": res["t"], "reference_hr_bpm": ref_hr, "estimated_hr_bpm": res["hr"],
                  "snr_db": res["snr_db"], "p_good": p_good, "reported": reported}).to_csv(os.path.join(outdir, "processed_output.csv"), index=False)
    target = os.path.join(outdir, "target_video.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video, "-t", str(args.seconds),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an", target], check=True)
    shutil.copy(__file__, os.path.join(outdir, "process_round1.py"))

    print(f"{name}_{args.constraint}: MAE {m['mae']:.2f} bpm (reported windows: {100 * reported.mean():.0f}%), RMSE {m['rmse']:.2f}, r {m['r']:.3f}, "
          f"Bland-Altman {m['ba_bias']:+.2f} [{m['ba_loa_low']:.1f}, {m['ba_loa_high']:.1f}] -> {outdir}")


if __name__ == "__main__":
    main()
