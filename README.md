# FaceVitals prototype (SP Cup 2027, Round 1: heart rate)

rPPG pipeline from the research plan: a classical core plus a small learned
branch and a signal-quality gate, runnable on a laptop and designed to port to
a Raspberry Pi 5.

**Start here**

| | |
|---|---|
| Round 1 report (IEEE format, draft) | [`report/main.pdf`](report/main.pdf) |
| Demo video of the live dashboard | [`demo/facevitals_demo.mp4`](demo/facevitals_demo.mp4) |
| Project context, decisions and learnings | [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) |
| First-day solution plan and research notes | [`docs/solution_plan.md`](docs/solution_plan.md), [`docs/research_notes/`](docs/research_notes/) |

Datasets are not included. `fetch_mcd.py` downloads MCD-rPPG (CC BY 4.0) and
`ubfc.py fetch` downloads UBFC-rPPG (research use only, not redistributable)
into `data/`.

```
sender (C5) ─► keep every 3rd camera frame (30 -> 10 fps) ─► H.264 500 kbps,
               x264 zerolatency (no B-frames)

camera/video ─► YuNet face detection + MediaPipe landmarks on the target crop,
                both at ~10 Hz in real time (every frame at 10 fps)
             ─► target tracker (IoU + constant velocity; bystanders ignored)      C3
             ─► whole face + 6 skin patches + background boxes beside the face
             ─► mean RGB per region, on smoothed real frame timestamps           C5
             ─► codec-pattern removal: undo colour offsets tied to the
                encoder's P B B B cycle (only if the stream has B-frames)        C5
             ─► illumination cancellation: remove what skin shares with
                the background (mains flicker, light changes)                    C1/C5
             ─► whole-face POS, SNR-fused with the learned branch (27k-param
                temporal CNN on all regions' colour traces + POS/CHROM/GREEN)
                 └─ if both lower cheeks look masked: SNR-fused POS+CHROM
                    over the patches that still look like skin                     C7
             ─► Viterbi HR tracking ─► HR
             ─► quality gate: P(error <= 5 bpm) from spectrum, tracker and
                method agreement; report if >= 0.8, else flag low quality
```

## Hard subjects: learned branch + abstention

Some subjects failed even at full quality (MCD 1097, 1181, 1287, 1513, 1581;
mostly flushed, post-exercise, high heart rate). Their contact-PPG reference
was checked against beat counting and the separate ECG, so the errors were
ours. In these videos plain GREEN often peaked at the right rate while POS and
CHROM did not: the colour mix POS assumes did not hold.

- **Learned branch** (`fv/learned.py`, `dl_train.py`): learns its own colour
  and region mixing from the full 3-minute videos of subjects 1-60, with
  time-stretch augmentation (0.8-1.25x) so high heart rates are better
  covered. Spectral loss against the PPG heart rate. Trains in ~2 min on the
  Mac GPU; 1.2 ms per estimate on one CPU core.
- **Abstention** (`fv/quality.py`): a small gradient-boosted classifier on
  SNR, spectral share, tracker-vs-peak gap and agreement between POS, GREEN,
  CHROM and the learned branch. Skin colour is deliberately not an input.

Development results are 5-fold out-of-fold by subject (`dev_cv.py` →
`out/dev60_summary.md`, subjects 1-60). Two clean checks so far, each on 20
subjects untouched until then: subjects 41-60 (model trained on 1-40,
`out/check_summary.md`) and subjects 61-80 (model trained on 1-60,
`out/check2_summary.md`).

Latest clean check (subjects 61-80, 40 videos), pooled MAE in bpm:

| Condition | POS | Classical final | Classical + learned | + abstention: reported / MAE / within 5 |
|---|---|---|---|---|
| Normal | 5.84 | 3.00 | **1.49** | 85% / 1.21 / 98% |
| C4 320x240 | 6.46 | 3.20 | **1.73** | 82% / 1.22 / 97% |
| C1 50 lux (sim.) | 6.66 | 4.69 | **2.14** | 83% / 1.29 / 98% |
| C1 20 lux (sim.) | 7.49 | 3.95 | **2.38** | 76% / 1.17 / 97% |
| C3 bystander | 6.08 | 3.35 | **1.44** | 85% / 1.27 / 97% |
| C5 our sender | 9.99 | 4.78 | **2.25** | 73% / 1.61 / 95% |
| C5 default x264 | 12.73 | 6.34 | **4.04** | 65% / 2.19 / 94% |
| C7 mask (sim.) | 10.48 | **8.49** | 8.56 | 37% / 1.25 / 97% |

The two check sets contain different subjects (this one has fewer hard,
high-rate videos), so compare methods within a table, not across them.

Previous clean check (subjects 41-60, model trained on 1-40):

| Condition | POS | Classical final | Classical + learned | + abstention: reported / MAE / within 5 |
|---|---|---|---|---|
| Normal | 5.87 | 3.14 (86%) | **2.65** (92%) | 83% / 1.55 / 97% |
| C4 320x240 | 6.25 | 3.40 | **2.91** | 85% / 1.31 / 97% |
| C1 50 lux (sim.) | 6.82 | 3.06 | **2.98** | 84% / 1.24 / 98% |
| C1 20 lux (sim.) | 8.92 | 5.36 | **4.46** | 74% / 1.29 / 98% |
| C3 bystander | 6.17 | 3.58 | **2.75** | 83% / 1.45 / 97% |
| C5 our sender | 10.31 | 4.88 | **4.01** | 73% / 1.95 / 93% |
| C5 default x264 | 12.29 | 9.51 | **3.96** | 60% / 1.84 / 94% |
| C7 mask (sim.) | 13.45 | 10.06 | 10.06 | 34% / 2.72 / 92% |

### Cross-dataset check: UBFC-rPPG (`ubfc.py`, `out/ubfc_summary.md`)

All 42 shared subjects of UBFC-rPPG DATASET_2 (another lab, Logitech C920,
uncompressed, varying sunlight, subjects doing a stressful maths game; CMS50E
pulse oximeter), run with everything frozen on MCD subjects 1-60. Pooled MAE, bpm:

| Condition | POS | Classical | Classical + learned | + abstention: reported / MAE |
|---|---|---|---|---|
| Normal | 0.91 | 0.83 | 0.85 | 98% / 0.70 |
| C4 320x240 | 1.03 | 0.88 | 0.88 | 97% / 0.71 |
| C1 50 lux (sim.) | 1.09 | 0.88 | 0.90 | 97% / 0.74 |
| C1 20 lux (sim.) | 1.51 | 1.07 | 0.95 | 96% / 0.78 |
| C3 bystander | 0.96 | 0.87 | 0.85 | 98% / 0.73 |
| C5 our sender | 9.15 | 4.11 | 3.24 | 79% / 1.38 |
| C5 default x264 | 29.77 | 32.67 | 10.04 | 55% / 4.20 |
| C7 mask (sim.) | 2.03 | 0.85 | 0.85 | 92% / 0.60 |

Default x264 at 10 fps / 500 kbps fails for fast heart rates: the classical
branch averages 2.6 bpm for the 4 subjects under 75 bpm but locks onto
43-85 bpm in 25 of the 28 subjects at 90 bpm or more. Not yet explained; one
candidate is that low-bitrate B-frames carry little new colour information.
No UBFC face images may be shown for subjects 21 and 27 (dataset readme); none
are shown anywhere.

### Low heart rates: still open

The 41-60 check found subject 1932 (47-52 bpm), where the learned branch read
80-95 bpm and won the fusion. Tested on subjects 1-60, out-of-fold:

| Idea | Result | Kept? |
|---|---|---|
| Wider time-stretch (0.65-1.5x) | error below 55 bpm 10.9 -> 12.5 | no |
| ... plus heart-rate-balanced sampling | 10.9 -> 13.2; better above 100 bpm, worse overall | no |
| GREEN/CHROM vote when POS and learned differ > 10 bpm | no better than SNR fusion; worse for C5 (4.2 vs 3.6) | no (`fusion="vote"` kept as an option) |
| SNR-fuse POS, CHROM, GREEN and learned | about equal to SNR fusion | no (`fusion="all4"`) |
| Withhold windows where branches differ > 10 bpm | same accuracy as raising the abstention threshold, but fewer windows kept | no |

Why: 78% of windows below 60 bpm were already right; the failures are mostly
subject 1932, whose pulse is barely present in any region (~10% of spectral
power at the true rate, vs 30-40% normally) and whose face is darker and
dimmer than most. Neither branch really sees the pulse there, so abstaining is
the right behaviour, and it withholds 65% of those windows. Below 55 bpm on
the development set the fused method is still worse than the classical one
(12.1 vs 3.7 bpm, 88 windows). The 61-80 check had too few low-rate windows
to test this. Because the weak signal came with a darker face, this is
also a C2 (skin tone) risk to test on more diverse data.

## What broke C5 (10 fps + 500 kbps) and what fixed it

Diagnosed on the 25 development subjects (`c5_diag.py`, `tracker_ablation.py`),
then checked once on 15 untouched subjects (`out/fresh_summary.md`).

| Cause | Symptom | Fix |
|---|---|---|
| Default x264 repeats P B B B; each position quantises the face colour differently | false peak at 10 fps / 4 = 2.5 Hz = **150 bpm** | sender: x264 `zerolatency` (no B-frames); receiver: regress out per-position offsets (`remove_codec_pattern`) |
| Frames picked "nearest to each 0.1 s" from timestamps that jitter in ~16 ms timer steps gave an uneven 2/3/4-frame pattern; with 100 Hz lighting that becomes another 2.5 Hz artefact (and made frame binning worse) | same 150 bpm lock, also with no B-frames | keep exactly every 3rd frame; smooth timestamps (`smooth_stamps`) |
| Face mesh refreshed every 3rd *frame*, i.e. only 3.3 Hz at 10 fps | regions drift during movement | detection + landmarks at ~10 Hz in real time |
| Tracker's harmonic boost: the band ends at 210 bpm, so rates > 105 bpm never earn it but their half-rate sub-harmonic can | high heart rates read as about half | boost off (`beta=0`) |

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Models (already in `models/`): MediaPipe `face_landmarker.task`, OpenCV Zoo
`face_detection_yunet_2023mar.onnx`.

## Run it

| What | Command |
|---|---|
| Live webcam (dashboard with every step) | `.venv/bin/python live.py` (add `--no-learned` for the classical branch only) |
| Replay a dataset video with the finger-sensor reference on screen | `.venv/bin/python live.py --video data/mcd/video/2187_FullHDwebcam_after.avi --meta data/mcd/meta/2187_FullHDwebcam_after.txt --ppg data/mcd/ppg_sync/2187_FullHDwebcam_after.txt` |
| Record the demo video | the replay command plus `--no-display --seconds 92 --record demo/facevitals_demo.mp4 --schedule "28:lowlight20,45:none,58:mask,76:none"` |
| Live with a constraint simulated | `.venv/bin/python live.py --simulate mask` (`lowlight50`, `lowlight20`, `lowres`, `lowfps`) |
| Log live readings to compare with a watch / oximeter | `.venv/bin/python live.py --log session.csv` |
| Known-answer tests | `.venv/bin/python synth_test.py` |
| Download benchmark data (MCD-rPPG, CC BY 4.0) | `.venv/bin/python fetch_mcd.py 25` |
| Development benchmark (subjects 1-12) | `.venv/bin/python dev_eval.py dev` / `test` |
| C5 diagnosis (fps vs bitrate vs codec, dev subjects) | `.venv/bin/python c5_extract.py 0 25 baseline enc_fps10 bandwidth enc_zl_fps10_kbps500` then `.venv/bin/python c5_diag.py` |
| Tracker harmonic-boost ablation (dev subjects) | `.venv/bin/python tracker_ablation.py` |
| First held-out set (subjects 13-25; later used for diagnosis) | `.venv/bin/python heldout.py` → `out/heldout_summary.md` |
| Clean check (subjects 26-40) | `.venv/bin/python fetch_mcd.py 25 40` then `.venv/bin/python heldout.py --lo 25 --hi 40 --tag fresh` → `out/fresh_summary.md` |
| Training traces, full length (subjects 1-60) | `.venv/bin/python dl_extract.py 0 60` |
| Learned branch: cross-validation, then final model | `.venv/bin/python dl_train.py cv` then `.venv/bin/python dl_train.py all` |
| Augmentation ablations | `.venv/bin/python dl_train.py cv --stretch wide --balance --tag wide_bal` then `.venv/bin/python dev_cv.py --tag wide_bal --out dev60_wide_bal` |
| Development evaluation + abstention models (subjects 1-60, out-of-fold) | `.venv/bin/python dev_cv.py --save-abstainers --fusion snr` → `out/dev60_summary.md`, `models/abstainer_*.pkl` |
| Clean checks (subjects 41-60, then 61-80) | `.venv/bin/python fetch_mcd.py 60 80` then `.venv/bin/python heldout.py --lo 60 --hi 80 --tag check2` → `out/check2_summary.md` |
| Report: ablation ladder, then tables, figures and quoted numbers | `.venv/bin/python report/ladder.py` then `.venv/bin/python report/stats.py` → `report/tables/`, `report/figs/`, `report/numbers.json` |
| Build the report PDF (IEEE conference format) | `cd report && tectonic main.tex` (or upload `report/` to Overleaf) |
| Cross-dataset check on UBFC-rPPG (downloads ~1.8 GB per video, deletes each after caching traces) | `.venv/bin/python ubfc.py fetch` then `.venv/bin/python ubfc.py eval` → `out/ubfc_summary.md` |
| Make Round 1 submission files | `.venv/bin/python process_round1.py --video ... --ppg ... --meta ... --condition Baseline --constraint None` |

The live window (`fv/dashboard.py`) shows the annotated video, the heart-rate
card with the quality model's confidence (and the finger sensor when replaying
with `--ppg`), and four graphs in pipeline order: raw forehead colour vs the
background, the cleaned pulse, the spectrum (POS, learned branch, combined) and
heart rate over time. `q` quits, `m` toggles a simulated mask, `l` cycles
low-light levels, and clicking a face makes it the target. MCD-rPPG AVI headers
can state the wrong frame rate (2161/2187 say 24 fps; they are ~30), so replays
need `--meta` for the real frame times.

## Code map

| File | Contents |
|---|---|
| `fv/face.py` | YuNet + MediaPipe front end, target tracker, skin patches, mask overlay, ITA |
| `fv/signals.py` | timestamp resampling, band-pass, GREEN / CHROM / POS |
| `fv/hr.py` | windowed spectra, SNR quality index, fusion, Viterbi + online tracker |
| `fv/pipeline.py` | method variants (ablation ladder), occlusion gate, reference HR |
| `fv/degrade.py` | low light (shot + read noise), low resolution, ffmpeg bitrate / fps |
| `fv/extract.py` | video → per-patch RGB traces under each simulated condition |
| `fv/metrics.py` | MAE, RMSE, Pearson r, Bland–Altman, % within 5 bpm |
| `fv/learned.py` | learned pulse branch: input features, model, spectral loss, inference |
| `fv/quality.py` | per-window quality features and the abstention model |
| `fv/dashboard.py` | live dashboard: annotated video, heart-rate card, four step-by-step graphs |

## Known limitations

- C5 simulations decimate by frame count and use the (smoothed) capture
  timestamps: some MCD-rPPG AVI headers state the wrong frame rate, which
  silently mis-scaled time (and heart rate) when ffmpeg did the frame dropping.
- If the organisers' C5 videos use default x264 settings, only the receiver-side
  codec fix applies; it helps on average but is weaker than sending without
  B-frames.
- The learned branch and the abstention model are trained on one dataset
  (MCD-rPPG: mostly young, lighter-skinned adults, indoor lighting). Check
  them on other data, especially darker skin (C2), before relying on them.
- The learned branch is not used in masked windows (it was trained without
  masks); those keep the classical patch fusion, and abstain more often.
- The live app uses the abstention model on the newest window; its share-trend
  feature looks at the last 11 windows instead of the 11 centred on it.
- Talking and smiling: benchmarks score the first 40 s of each video, where
  people sit still. Later in subject 2161's video (after 60 s, no simulation)
  she smiles and moves, the error rises to 13 bpm and two thirds of readings
  are withheld; one wrong reading (79 vs 123 bpm) still passed the gate.
- Low light, masks and multi-person scenes are **simulated** on MCD-rPPG; they
  must be confirmed on real recordings (MSLR has real 3/8/35 lux video).
- ITA skin-tone bands from uncalibrated video are shifted by room lighting;
  add a grey card / colour checker before using them for C2 claims.
- `process_round1` writes the target-face crop (72×72) as `facial_video`
  instead of full frames to keep files small; confirm with the organisers.
- The learned branch works on region colour traces, not pixels; a video-level
  model (FactorizePhys / EfficientPhys size) is the next step if more is
  needed.
- Code was written from scratch (no rPPG-Toolbox / pyVHR code) so licences do
  not conflict with the MathWorks commercial-use clause.
