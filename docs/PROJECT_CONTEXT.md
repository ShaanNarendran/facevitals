# FaceVitals: project context, decisions and learnings

What this project is, how it got to where it is, what we learned on the way,
and what is still open. For how to run things, see the [README](../README.md).
For the results as submitted, see the [Round 1 report](../report/main.pdf) and
the [demo video](../demo/facevitals_demo.mp4).

## 1. The challenge

IEEE Signal Processing Cup 2027, **FaceVitals**: estimate vital signs from RGB
facial video only (no thermal, NIR or contact sensing), in real time, on
consumer cameras.

| Round | Task | Deadline |
|---|---|---|
| 1 | Heart rate (HR) from video under controlled **and** uncontrolled conditions | 30 Nov 2026 (results 5 Dec 2026) |
| 2 | HR + SpO2 | 16 Jan 2027 |
| 3 | HR + SpO2 + blood pressure, clinically validated, live demo (top 3 teams, at ICASSP/ICIP) | final |

- **Round 1 is judged by a PhD panel** on innovation, technical feasibility and
  clarity of documentation, plus how close results are to medical-grade
  references. There is no pass mark, and scores are not published.
- **Each submission needs:**
  - an IEEE conference-format report;
  - code;
  - `/Round1_Baseline/Round1_Baseline_None.mat` containing `facial_video`, `roi_data`,
    `reference_signal`, `processed_signal`, `metrics` and `params`;
  - `target_video.mp4`, `reference_signal.csv` and `processed_output.csv`.
- **Metrics:** MAE, RMSE, Pearson r and Bland-Altman limits of agreement.
- **Constraints:** at least two of C1-C7 must be addressed, and more count in
  the team's favour:

| | Constraint | Specification |
|---|---|---|
| C1 | Low light | ≤ 50 lux at the face |
| C2 | Skin tone | Fitzpatrick I-VI, per-type error within a pre-set tolerance (ITA stratification) |
| C3 | Several people in view | track one designated target, reject bystanders |
| C4 | Low resolution | ≤ 640×480, stress case 320×240 |
| C5 | Low bandwidth | ≤ 10 fps, ≤ 500 kbps, offline / on-device mode |
| C6 | Ageing skin | ages 60-70+, reported separately |
| C7 | Masks | ≥ 40 % of the lower face covered; detect and compensate |

- **Data rules:** public datasets are allowed. The organisers also have a
  hospital dataset, available after signing an agreement. Teams may record
  their own data, but only with IRB approval, submitted in Round 1.

## 2. Where things are

| Path | What |
|---|---|
| `docs/solution_plan.md` | first-day research and plan: methods, datasets, constraints, edge deployment, Round 2/3 ideas |
| `docs/research_notes/` | the literature notes behind that plan (HR methods, robustness, datasets, SpO2, BP, edge) |
| `fv/` | the pipeline library (front end, signals, spectra and tracking, learned branch, quality gate, dashboard) |
| `live.py` | live / replay app with the step-by-step dashboard and demo recording |
| `process_round1.py` | writes the Round 1 submission files |
| `dev_cv.py`, `dl_train.py`, `heldout.py`, `ubfc.py` | development, training, held-out and cross-dataset evaluation |
| `report/` | the IEEE-format report; `stats.py` regenerates every table and figure from `out/` |
| `out/` | per-window results and summaries behind every number |
| `models/` | trained learned branch and quality gate, plus the face detector / landmarker |
| `submission/` | a sample Round 1 package (one MCD-rPPG video) |

## 3. Approach in one paragraph

A face detector (YuNet) runs about 10 times a second and an IoU tracker keeps
one target. MediaPipe's 468-point mesh, run on the target only, gives six skin
patches, the face oval and two background boxes. Their mean colours become
traces, which are:

1. put on the real capture timestamps;
2. cleaned of the H.264 B-frame colour pattern;
3. cleaned of lighting changes by regressing out the background.

Two branches turn each 10 s window into a pulse spectrum: whole-face POS, and
a 27k-parameter temporal CNN trained on the traces. They are fused by SNR and
tracked with a Viterbi decoder. If both lower cheeks look covered, the system
switches to the uncovered patches. A gradient-boosted quality gate estimates
the probability that each window is within 5 bpm and reports only those at
0.8 or above. Everything runs at about 18 % of one laptop core.

## 4. How the work went (and why each step happened)

1. **Plan (week 1).** We surveyed the methods, datasets and constraints
   (`docs/`). We chose a classical core that can be explained, plus a small
   learned part, rather than a large end-to-end network: judges reward
   clarity, and Round 3 needs a Raspberry Pi.
2. **Prototype and first benchmark.** We wrote everything from scratch: no
   rPPG-Toolbox or pyVHR code, to avoid licence conflicts with the MathWorks
   clause. The benchmark was MCD-rPPG (600 subjects, CC BY 4.0, 100 Hz PPG
   and ECG), with all seven constraints simulated on the same videos.
3. **C5 failure (10 fps + 500 kbps).** Many videos locked onto exactly
   150 bpm. Separating frame rate from bitrate found four causes, all fixed:
   - default x264's P-B-B-B cycle at 10 fps puts a colour oscillation at
     2.5 Hz, which is 150 bpm;
   - picking frames by nearest timestamp from jittery stamps, combined with
     100 Hz light, made another 2.5 Hz artefact;
   - the face mesh refreshed every 3rd frame, which is only 3.3 Hz at 10 fps;
   - the tracker's harmonic boost biased fast heart rates to half rate.
4. **Hard subjects.** A few subjects failed even at full quality; most were
   flushed, post-exercise and fast. We checked their references against beat
   counting and ECG, so the errors were ours. We added the learned branch and
   the quality gate, developed on subjects 1-40 and checked on 41-60.
5. **Low heart rates.** We tried wider time-stretch augmentation,
   heart-rate-balanced sampling and fusion votes. None helped (see Learnings),
   so none were kept. We checked on 61-80.
6. **Round 1 report.** Every table is regenerated from saved results by
   `report/stats.py`. We added a stage-by-stage ablation on subjects 1-60.
7. **Cross-dataset check on UBFC-rPPG.** We ran the system unchanged on all
   42 shared subjects of another lab's dataset.
8. **Live dashboard and demo.** The app now shows every stage, the quality
   gate's confidence and, for dataset replays, the finger sensor.

**Evaluation discipline.** Every reported number comes from subjects the
scoring models never saw:
- development used 5-fold cross-validation by subject;
- each held-out set was scored once, with everything frozen;
- once a held-out set had been looked at for diagnosis (subjects 13-25), it
  was retired and a fresh one was used.

## 5. Results

**MCD-rPPG held-out subjects 61-80** (model trained on subjects 1-60). Pooled MAE in bpm:

| Condition | POS | Classical | Proposed | With gate: reported / MAE |
|---|---|---|---|---|
| Normal (VGA) | 5.84 | 3.00 | **1.49** (r = 0.95) | 85 % / 1.21 |
| C4 320×240 | 6.46 | 3.20 | **1.73** | 82 % / 1.22 |
| C1 50 lux (sim.) | 6.66 | 4.69 | **2.14** | 83 % / 1.29 |
| C1 20 lux (sim.) | 7.49 | 3.95 | **2.38** | 76 % / 1.17 |
| C3 bystander | 6.08 | 3.35 | **1.44** | 84 % / 1.27 |
| C5 our sender | 9.99 | 4.78 | **2.25** | 73 % / 1.61 |
| C5 default x264 | 12.73 | 6.34 | **4.04** | 65 % / 2.19 |
| C7 mask (sim.) | 10.48 | **8.49** | 8.56 | 37 % / 1.25 |

**UBFC-rPPG, 42 subjects, nothing retrained.** Pooled MAE in bpm:

| Condition | Normal | C4 | C1 50 | C1 20 | C3 | C5 ours | C5 x264 | C7 |
|---|---|---|---|---|---|---|---|---|
| MAE | 0.85 | 0.88 | 0.90 | 0.95 | 0.85 | 3.24 | 10.04 | 0.85 |

**Live demo** (92 s replay of held-out subject 2187 with simulated 20 lux and
a mask): 1.3 bpm, with 81 % of readings reported.

## 6. Learnings

**Signal and system**
- **At 10 fps, encoder settings matter as much as bitrate.** Sending without
  B-frames (x264 `zerolatency`) removes most of the C5 damage. On UBFC,
  default x264 still fails for fast heart rates: the classical branch locks
  onto 43-85 bpm in 25 of 28 subjects at 90 bpm or more, but is fine below
  75 bpm. One unconfirmed explanation is that low-bitrate B-frames carry
  little new colour information.
- **Video headers can be wrong about frame rate.** Some MCD-rPPG AVIs claim
  24 fps but were captured at about 30, which silently mis-scales heart rate.
  Always use capture timestamps, and smooth away timer quantisation (about
  16 ms steps).
- **Pick decimated frames by count, not by nearest timestamp.** With jittery
  stamps, the nearest-timestamp rule makes an uneven 2/3/4-frame pattern that
  turns light flicker into an in-band artefact.
- **The common harmonic boost in HR trackers is biased.** The band ends at
  about 210 bpm, so rates above 105 bpm can never earn the boost while their
  half-rate sub-harmonic can. Turning it off helped every condition.
- **Mains flicker can be 10× the pulse in raw traces** (±5 % in one MCD room).
  The background regression removes it, but raw plots need smoothing to be
  readable.

**Learning and fusion**
- **A tiny learned branch on region colour traces (not pixels) goes a long
  way.** It has 27k parameters, trains in 2 minutes and runs in 1.2 ms per
  window, and it cut the error by 43-64 % out-of-fold in every unmasked
  condition. On the easier UBFC data it neither helps nor hurts.
- **Ideas tested and rejected** (out-of-fold, subjects 1-60):
  - wider time-stretch (0.65-1.5×): error below 55 bpm went from 10.9 to 12.5;
  - the same plus heart-rate-balanced sampling: 13.2;
  - letting GREEN and CHROM vote when the branches disagree: no better, and
    worse for C5;
  - fusing all four spectra: about equal;
  - withholding windows where the branches disagree: same accuracy at lower
    coverage.
- **Most low-heart-rate failures came from one subject** with a barely visible
  pulse: about 10 % of spectral power at the true rate, against 30-40 %
  normally. The right behaviour there is to abstain, and the gate withholds
  65 % of that subject's windows.
- **The mask gate detects masks perfectly but did not beat whole-face POS** on
  simulated masks (9.99 against 8.70 bpm). The simulated mask is a static
  texture with no pulse, which flatters whole-face averaging. Real masks are
  needed.

**Quality gate**
- **A gradient-boosted classifier on 7 window features** (SNR, peak share,
  tracker gap, branch gap, share trend, GREEN and CHROM gaps) gets an AUC of
  0.90 out-of-fold and 0.89 held-out. At P ≥ 0.8, 94-98 % of reported windows
  are within 5 bpm. Skin colour is deliberately not a feature.
- **Motion is the blind spot.** Benchmarks score the first 40 s of each video,
  while people sit still. In a later part of one video, where the subject
  smiles and moves, the error rose to 13 bpm. Most readings were withheld,
  but one wrong reading (79 against 123 bpm) passed the gate.

**Skin tone (C2)**
- **ITA from an uncalibrated webcam is meaningless in absolute terms.** Faces
  are exposed dimly (L* ≈ 40), so every MCD subject comes out as Fitzpatrick
  V-VI. Use relative strata only, or add a grey card / colour checker. Error
  did not rise with darker measured ITA, but real darker-skin data is still
  needed.

**Practical**
- **UBFC-rPPG's Google Drive copy hits download quotas**, but the authors'
  OneDrive mirror works. The videos are uncompressed (about 1.8 GB each,
  75 GB in all), so `ubfc.py` streams them one at a time and deletes each
  after caching its traces. Its readme forbids showing the faces of subjects
  21 and 27, so no UBFC frame is shown anywhere.

## 7. Open items

**Before the Round 1 deadline (30 Nov 2026):**
- [ ] Team names, affiliation and supervisor in the report (red TO DO marks in `report/main.tex`).
- [ ] Confirm team registration on the SPS platform (registration was due 25 Oct 2026).
- [ ] `facial_video` should hold raw RGB frames (it currently holds 72×72 face crops).
- [ ] Fill `params` with lux, distance and Fitzpatrick type where a recording has them.
- [ ] A dataset with darker skin tones (Annexure I: India / Sierra Leone) for C2.
- [ ] Measured Raspberry Pi 5 timings, or drop the claim.
- [ ] IRB approval and consent forms if the team records its own videos.

**Technical, for Round 2:**
- [ ] Test real low light (for example MSLR's 3 / 8 / 35 lux video) and real masks; train the learned branch with mask augmentation.
- [ ] Add motion features to the quality gate; make the regions robust to motion.
- [ ] Test the B-frame hypothesis for default-x264 C5.
- [ ] SpO2: per-channel AC/DC ratio-of-ratios from the same regions, with the quality gate extended to SpO2.

## 8. Data and licences

- **MCD-rPPG** (Egorov et al., ACM MM 2025; Hugging Face `kyegorov/mcd_rppg`),
  CC BY 4.0. Fetch it with `fetch_mcd.py`. Frames from it appear in the demo
  video, the report figure and `submission/`.
- **UBFC-rPPG** (Bobbia et al., Pattern Recognition Letters 2019) is shared by
  its authors for research use and must be cited. It is not redistributed
  here: only per-window heart-rate results (`out/ubfc_per_window.csv`) are
  included. Fetch it with `ubfc.py fetch`.
- **Third-party models** in `models/`: see `models/README.md`.
