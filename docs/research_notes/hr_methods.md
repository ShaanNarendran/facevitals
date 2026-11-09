# Heart-rate estimation from RGB facial video (rPPG): classical vs deep, generalisation, and edge suitability

Scope: HR-from-face-video state of the art for an IEEE SP Cup 2027 "FaceVitals" entry (Raspberry Pi, real-time, judged on accuracy, robustness, innovation, complexity). All MAE/RMSE in BPM unless stated. "Cross-dataset" = trained on one dataset, tested on another; classical methods are training-free, so their numbers are effectively always "cross-dataset".

Key acronyms: UBFC = UBFC-rPPG (42 subjects, 640x480, uncompressed, mostly still); PURE (10 subjects, controlled head motion, 640x480 lossless PNG); MMPD (mobile-phone videos, varied skin tones/lighting/motion, compressed — the hardest common benchmark); UBFC-Phys (stress tasks, MJPEG); iBVP; SCAMPS (synthetic).

---

## Q1. Classical methods (GREEN, ICA, PCA, CHROM, POS, LGI, PBV, OMIT, 2SR/SSR): which are best cross-dataset?

### Takeaway
POS and CHROM are consistently the best training-free methods: about 3.7–4.5 BPM MAE on PURE/UBFC/UBFC-Phys and about 12–14 BPM on MMPD in the rPPG-Toolbox benchmark. GREEN and ICA are worst. pyVHR's statistical analysis finds POS, CHROM, PCA and SSR (2SR) statistically tied at the top. Classical methods are much more robust to resolution, frame-rate and frame-drop degradations than several deep models, but they collapse under low-bitrate compression and on MMPD-like "in-the-wild" data.

### Cited Findings
**rPPG-Toolbox benchmark (Liu et al., NeurIPS 2023 Datasets & Benchmarks; MAE / MAPE). Protocol: 2nd-order Butterworth band-pass 0.75–2.5 Hz, then FFT or peak detection; SNR over 45–150 bpm using the 1st and 2nd harmonics of the ground-truth HR** — [rPPG-Toolbox paper, arXiv 2210.00716](https://arxiv.org/html/2210.00716)

| Method | PURE | UBFC-rPPG | UBFC-Phys | MMPD |
|---|---|---|---|---|
| GREEN | 10.09 / 10.28 | 19.81 / 18.78 | 13.55 / 16.01 | 21.68 / 24.39 |
| ICA | 4.77 / 4.47 | 14.70 / 14.34 | 10.03 / 11.85 | 18.60 / 20.88 |
| CHROM | 5.77 / 11.52 | 3.98 / 3.78 | 4.49 / 6.00 | 13.66 / 15.99 |
| LGI | 4.61 / 4.96 | 15.80 / 14.70 | 6.27 / 7.83 | 17.08 / 18.98 |
| PBV | 3.91 / 4.82 | 15.90 / 15.17 | 12.34 / 14.63 | 17.95 / 20.18 |
| POS | 3.67 / 7.25 | 4.00 / 3.86 | 4.51 / 6.12 | 12.36 / 14.43 |

- The same toolbox numbers with RMSE and Pearson r, re-reported in RhythmMamba Table 2 (MAE / RMSE / MAPE / r) — [RhythmMamba, arXiv 2404.06483](https://arxiv.org/html/2404.06483):
  - **POS:** PURE 3.67 / 11.82 / 7.25 / 0.88; UBFC 4.08 / 7.72 / 3.93 / 0.92; MMPD 12.36 / 17.71 / 14.43 / 0.18.
  - **CHROM:** PURE 5.77 / 14.93 / 11.52 / 0.81; UBFC 4.06 / 8.83 / 3.84 / 0.89; MMPD 13.66 / 18.76 / 16.00 / 0.08.
  - **OMIT:** PURE MAE 4.66, RMSE 15.82; UBFC 16.99 / 29.54 / 15.91 / 0.34; MMPD 17.02 / 23.23 / 18.89 / 0.04. The OMIT PURE r is printed as 8.76, which is impossible and likely a typo.
  - **GREEN:** MMPD r = −0.01, i.e. no correlation at all.
- The rPPG-Toolbox supports 7 unsupervised methods: GREEN (Verkruysse 2008), ICA (Poh 2011), CHROM (de Haan 2013), LGI (Pilz 2018), PBV (de Haan 2014), POS (Wang 2016) and OMIT (Álvarez 2023). It does not include PCA or 2SR — [rPPG-Toolbox GitHub](https://github.com/ubicomplab/rPPG-Toolbox).
- **pyVHR / Boccignone et al. (IEEE Access 2020):** 8 methods on 5 public datasets with nonparametric statistics. The top four were POS, CHROM, PCA and SSR, and they were not significantly different from each other — [pyVHR search summary / HAL](https://hal-lara.archives-ouvertes.fr/LIRFAI/hal-03652383v1); [IEEE Access 2020 record](https://archivesic.ccsd.cnrs.fr/LIRFAI/hal-03046044v1).
- **pyVHR (PeerJ CS 2022), UBFC2 (CCC metric):** the Friedman test rejected equal medians across CHROM, POS and GREEN. Nemenyi found no significant CHROM–POS difference, but both differed from GREEN. Mean ranks were CHROM 1.385, POS 1.769, GREEN 2.846 — [pyVHR PeerJ / PMC9044207](https://pmc.ncbi.nlm.nih.gov/articles/PMC9044207).
- **Independent 72x72 MediaPipe-crop evaluation** (10 s windows, 1 s step, 0.75–4 Hz, Welch nFFT 4096). Baseline UBFC-rPPG MAE / PCC: POS 2.74 / 0.99, CHROM 3.05 / 0.97, OMIT 3.49 / 0.94. Deep models in the same setting: EfficientPhys 7.73 / 0.69 (trained on UBFC, so favoured) and Contrast-Phys 3.72 / 0.94 — [Evaluation of Video-Based rPPG in Challenging Environments, arXiv 2405.01230](https://arxiv.org/html/2405.01230).

### Inferences
- POS is the safest classical default. It is best or tied-best on PURE and MMPD, and within 0.1 BPM of CHROM on UBFC and UBFC-Phys. CHROM is a close second and a natural partner for fusion or ensembling.
- The 3–4 BPM MAE of POS/CHROM on UBFC versus about 1 BPM for the best deep cross-dataset models (Q2) is the main accuracy gap. On MMPD, POS (12.4) is roughly on par with most supervised models trained on PURE or UBFC (about 9–17), so on "hard" data classical methods are not much worse.
- The very high RMSE relative to MAE (for example POS PURE: MAE 3.67, RMSE 11.82) shows the classical error is dominated by a few gross failures (harmonic or motion-peak locking). That is exactly what SQI gating and spectral tracking (Q3/Q4) can fix cheaply.

### Gaps
- I did not retrieve 2SR/SSR or PCA numbers in the rPPG-Toolbox protocol. Only pyVHR's statistical "tie" result was found.
- I could not extract per-dataset MAE tables from pyVHR PeerJ: the page was truncated, and the PeerJ HTML returned 403.

---

## Q2. Deep supervised and unsupervised models: cross-dataset accuracy, params, FLOPs, latency

### Takeaway
On "easy" cross-dataset pairs (PURE↔UBFC), most deep models reach about 1–2 BPM MAE, versus about 4 BPM for POS. On MMPD every model degrades to about 9–17 BPM. The best generalisation-per-parameter is **FactorizePhys** (about 52 K params, 0.22 MB): 0.48 MAE UBFC→PURE, 1.04 PURE→UBFC, and it is far better than others when trained on synthetic SCAMPS. **RhythmFormer** and **RhythmMamba** lead on MMPD cross-dataset (about 9–10.6 MAE). RhythmMamba is lightweight (1.07 M params, 80.9 M MACs). PhysFormer (7.4 M params) is the heaviest and among the worst at generalising.

### Cited Findings
**rPPG-Toolbox supervised cross-dataset (MAE / MAPE)** — [arXiv 2210.00716](https://arxiv.org/html/2210.00716)

| Model | Train | →PURE | →UBFC | →UBFC-Phys | →MMPD |
|---|---|---|---|---|---|
| TS-CAN | UBFC | 3.69 | – | 5.13 | 14.00 |
| TS-CAN | PURE | – | 1.29 | 5.72 | 13.93 |
| TS-CAN | SCAMPS | 4.66 | 3.62 | 5.55 | 19.05 |
| PhysNet | UBFC | 8.06 | – | 5.79 | 9.47 |
| PhysNet | PURE | – | 0.98 | 4.78 | 13.93 |
| PhysNet | SCAMPS | 13.30 | 5.40 | 8.53 | 20.78 |
| PhysFormer | UBFC | 12.92 | – | 6.63 | 12.10 |
| PhysFormer | PURE | – | 1.44 | 6.04 | 14.57 |
| PhysFormer | SCAMPS | 26.58 | 4.56 | 11.91 | 22.69 |
| DeepPhys | UBFC | 5.54 | – | 6.62 | 17.49 |
| DeepPhys | PURE | – | 1.21 | 8.42 | 16.92 |
| DeepPhys | SCAMPS | 3.95 | 3.10 | 4.75 | 15.22 |
| EfficientPhys-C | UBFC | 5.47 | – | 4.93 | 13.78 |
| EfficientPhys-C | PURE | – | 2.07 | 5.31 | 14.03 |
| EfficientPhys-C | SCAMPS | 10.24 | 12.64 | 6.97 | 20.41 |

- **BigSmall** (multitask, BP4D+ AU subset): rPPG MAE 3.23, MAPE 3.51, r 0.83 — [arXiv 2210.00716](https://arxiv.org/html/2210.00716).

**RhythmFormer / RhythmMamba cross-dataset (MAE / RMSE / MAPE / r)** — [RhythmMamba, arXiv 2404.06483 (v2 Feb 2025)](https://arxiv.org/html/2404.06483)
- **UBFC→PURE:** RhythmFormer 0.97 / 3.36 / 1.60 / 0.99; RhythmMamba 1.98 / 6.51 / 3.59 / 0.96; TS-CAN 3.69; PhysNet 8.06; PhysFormer 12.92; Spiking-Phys 3.83.
- **UBFC→MMPD:** RhythmFormer 9.08 / 15.07 / 11.17 / 0.53; RhythmMamba 10.63 / 17.14 / 12.14 / 0.34; PhysNet 9.47; TS-CAN 14.01.
- **PURE→UBFC:** RhythmFormer 0.89 / 1.83 / 0.97 / 0.99; RhythmMamba 0.95 / 1.83 / 1.04 / 0.99; PhysNet 0.98; DeepPhys 1.21.
- **PURE→MMPD:** RhythmFormer 8.98 / 14.85 / 11.11 / 0.51; RhythmMamba 10.44 / 16.70 / 12.25 / 0.36.
- **Intra-dataset only (for reference, not generalisation):**
  - RhythmMamba: PURE 0.23 MAE, UBFC 0.50, VIPL-HR 4.30, MMPD 3.16.
  - RhythmFormer: MMPD 3.07, PURE 0.27.
  - Contrast-Phys+: PURE 0.48, UBFC 0.21.
  - PhysFormer++: VIPL-HR 4.88.
  - Intra-MMPD for older models: PhysNet 4.80, TS-CAN 9.71, PhysFormer 11.99, EfficientPhys 13.47.
- **Efficiency on 128x128 input (Params M / MACs M / throughput / peak GPU mem MB)** — [RhythmMamba](https://arxiv.org/html/2404.06483):
  - DeepPhys 1.98 / 744.45 / 5.65 / 37.28
  - PhysNet 0.77 / 438.24 / 11.80 / 11.43
  - TS-CAN 1.98 / 744.45 / 5.21 / 38.91
  - PhysFormer 7.38 / 316.29 / 8.50 / 28.63
  - EfficientPhys 1.91 / 373.72 / 8.42 / 26.68
  - RhythmFormer 3.25 / 240.55 / 6.30 / 29.06
  - **RhythmMamba 1.07 / 80.90 / 20.09 / 6.66**
- A 2026 Chinese journal table lists RhythmMamba MMPD MAE as 6.54 rather than 3.16, so the intra-MMPD protocols differ between papers — [J. Computer Applications 2026](https://www.joca.cn/EN/Y2026/V46/I8/2699).

**FactorizePhys (Joshi et al., NeurIPS 2024): cross-dataset MAE / RMSE / r, mean ± SE** — [arXiv 2411.01542](https://arxiv.org/html/2411.01542)
- **UBFC→PURE:** FactorizePhys 0.48 / 1.39 / 0.998; EfficientPhys+FSAM 3.69; EfficientPhys (SASN) 4.71; PhysFormer 8.90; PhysNet 10.38.
- **iBVP→PURE:** FactorizePhys 0.60; EfficientPhys+FSAM 0.44; PhysFormer 6.58; PhysNet 7.78.
- **SCAMPS(synthetic)→PURE:** FactorizePhys 5.43 / 15.80 / 0.80; PhysNet 26.74; PhysFormer 16.64.
- **PURE→UBFC:** FactorizePhys 1.04 / 2.44 / 0.989; PhysFormer 1.01; PhysNet 1.23.
- **SCAMPS→UBFC:** FactorizePhys 1.17; PhysNet 11.24; PhysFormer 8.42.
- **iBVP→UBFC:** FactorizePhys 1.04; PhysFormer 9.88.
- **PURE→iBVP:** FactorizePhys 1.66; PhysNet 1.63; PhysFormer 2.50.
- **UBFC→iBVP:** FactorizePhys 1.74; PhysFormer 7.86.
- **SCAMPS→iBVP:** FactorizePhys 2.71 versus 13.4–41.7 for all others.
- **Params / CPU latency / GPU latency / size** (i7-10870H CPU; RTX 3070 laptop GPU):
  - PhysFormer 7,380,871 / 450.47 ms / 26.86 ms / 29.8 MB
  - PhysNet 768,583 / 272.89 ms / 1.36 ms / 3.1 MB
  - EfficientPhys-SASN 2,163,081 / 371.08 ms / 1.31 ms / 8.7 MB
  - EfficientPhys+FSAM 140,655 / 82.19 ms / 5.62 ms / 0.57 MB
  - **FactorizePhys+FSAM 52,168 / 95.75 ms / 8.97 ms / 0.22 MB**
  - FLOPs are not given numerically.
- **FSAM idea:** nonnegative matrix factorisation of voxel embeddings computes joint spatial, temporal and channel attention, using rank-1 and a one-step gradient approximation — [arXiv 2411.01542](https://arxiv.org/html/2411.01542).
- The toolbox now also includes iBVPNet, PhysMamba, RhythmFormer and FactorizePhys — [rPPG-Toolbox GitHub](https://github.com/ubicomplab/rPPG-Toolbox).

**Unsupervised / self-supervised**
- **Contrast-Phys+ (fully labelled variant)**, UBFC→MMSE-HR: MAE 1.76, RMSE 5.34, r 0.92 — [Contrast-Phys+, arXiv 2309.06924](https://arxiv.org/pdf/2309.06924).
- **Third-party re-runs** — [FCUS-rPPG, arXiv 2606.03050](https://arxiv.org/pdf/2606.03050):
  - UBFC→BSIPL-RPPG: SiNC MAE 3.03, r 0.83; Contrast-Phys+ (0% labels) 2.15, r 0.88.
  - UBFC→MMPD: Contrast-Phys+ 13.33, r 0.25.
  - BSIPL-motion→PURE: SiNC 7.41 versus Contrast-Phys+ 13.63.
- **Contrast-Phys is fragile to frame rate.** UBFC MAE goes 3.72 (30 fps) → 21.55 (20) → 38.45 (15) → 48.38 (10), with PCC dropping to −0.17 — [arXiv 2405.01230](https://arxiv.org/html/2405.01230).

**2026 low-latency work**
- **StreamPPG** (arXiv 2606.23186, Jun/Sep 2026) does frame-wise rather than clip-wise estimation. It argues that clip-wise methods need more than 100 frames before inference, causing seconds of delay. It uses ground-truth rPPG as "privileged information" during training and claims real-time throughput on edge devices, but the abstract gives no device or numbers — [StreamPPG](https://arxiv.org/abs/2606.23186).

### Inferences
- **For a Pi:** FactorizePhys (52 K params, about 96 ms CPU per clip on a laptop i7) and RhythmMamba (1.07 M params, 81 M MACs) are the "strong and tiny" candidates. PhysNet (0.77 M params, 438 M MACs) is a well-understood middle option. PhysFormer, DeepPhys and TS-CAN are poor choices on cost per accuracy.
- **FactorizePhys's SCAMPS→real results** (1.17 on UBFC, 2.71 on iBVP) suggest synthetic pre-training plus a tiny model is viable. That matters if the challenge provides limited labelled data.
- Even SOTA deep models only reach about 9 MAE on MMPD cross-dataset. Real-world, mobile-like, compressed, diverse-skin data remains unsolved, and that is likely closest to "robustness" judging.

### Gaps
- PhysMamba's own cross-dataset numbers, params and FLOPs were not retrieved. EfficientPhys and PhysNet FLOPs are only available as MACs from the RhythmMamba table.
- No paper found reports measured latency on a Raspberry Pi 4/5 CPU for FactorizePhys or RhythmMamba. The team must benchmark this itself.
- No original SiNC (Speth et al., CVPR 2023) table was retrieved; only third-party re-runs.

---

## Q3. Hybrid / signal-processing innovations (multi-ROI, SNR weighting, spectral tracking, window length, FFT vs peak detection)

### Takeaway
Multi-ROI / patch-based processing with robust aggregation (median plus MAD in pyVHR; PCA or SNR/spectral-concentration weighting elsewhere) gives consistent but modest gains, about 0.5–1 BPM for POS. Temporal tracking (Kalman, neighbour-aware peak tracking) is reported to give sub-1 BPM MAE on easy datasets. Evidence for Viterbi or harmonic-aware tracking specifically in rPPG was not found, which leaves room for an "innovation" claim.

### Cited Findings
- **pyVHR patch processing:** per-patch BPM is aggregated with the median, and the median absolute deviation (MAD) is used as an uncertainty estimate. The authors argue patches are more robust to shadows and uneven lighting; no quantitative holistic-vs-patch table was visible. Default windows are 6 s with 1 s overlap — [pyVHR PeerJ / PMC9044207](https://pmc.ncbi.nlm.nih.gov/articles/PMC9044207).
- **Angle-guided multi-ROI + graph-signal denoising (arXiv 2026):** POS MAE on COHFACE goes 5.18 (single ROI) → 4.63 (angle-guided ROIs) → 4.06 (plus graph denoising). ROI signals were fused with PCA — [arXiv 2604.11395](https://arxiv.org/pdf/2604.11395).
- **Adaptive Kalman + wavelet denoising + multi-channel fusion (PLOS One 2026):** reported MAE 0.72 BPM on PURE. The contribution of the Kalman stage alone was not isolated — [PMC12818640](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12818640/).
- **Reliability-weighted multi-ROI fusion:** a Polimi thesis weights per-ROI HR candidates by spectral energy concentration using iteratively reweighted least squares — [Politesi](https://www.politesi.polimi.it/handle/10589/241005). A related 2026 paper proposes reliability-aware weighted multi-scale spatio-temporal maps — [arXiv 2603.26836](https://arxiv.org/pdf/2603.26836).
- **SSA spectral peak tracking with neighbour-aware tracker:** 1.16 BPM average absolute error on 12 subjects during physical activity (small study) — [arXiv 1704.00224](https://arxiv.org/pdf/1704.00224v1).
- **Face-quality fusion:** Balmaekers & de Haan combine quality scores and short-term stability across facial regions to determine HR (via search summary) — [ResearchGate CHROM page / search](https://www.researchgate.net/publication/237070892_Robust_Pulse_Rate_From_Chrominance-Based_rPPG).
- **Typical windowing and spectral HR:**
  - Robustness study: 10 s windows, 1 s step, 0.75–4 Hz band-pass, Welch PSD with nFFT 4096 — [arXiv 2405.01230](https://arxiv.org/html/2405.01230).
  - Codec study: 10 s windows, 0.75–2.5 Hz, FFT peak — [arXiv 2606.04198](https://arxiv.org/html/2606.04198).
  - rPPG-Toolbox: 0.75–2.5 Hz, FFT or peak detection, chunked 180-frame clips; the paper does not say which HR estimator was used for the benchmark tables — [arXiv 2210.00716](https://arxiv.org/html/2210.00716).
- **Frame drops and timestamp interpolation:** at 50% random frame loss on UBFC, POS MAE goes 65.89 with no mitigation → 12.19 after recomputing fps → 3.03 with timestamp-based interpolation, nearly the 2.74 baseline. OMIT goes 62.75 → 12.83 → 3.65. Deep models did worse after mitigation (EfficientPhys 14.46, Contrast-Phys 36.47) — [arXiv 2405.01230](https://arxiv.org/html/2405.01230).
- **Facemask occlusion:** simply excluding occluded skin beat GAN inpainting for HR. POS goes 4.72 → 2.91 with occluded-skin removal — [arXiv 2405.01230](https://arxiv.org/html/2405.01230).

### Inferences
- For a Pi, timestamp-aware resampling of the RGB trace to a uniform grid is a cheap, high-value step, because Pi capture often has jittery or dropped frames. The evidence shows it rescues classical methods almost completely.
- A defensible novelty package:
  1. Multi-patch POS/CHROM.
  2. Per-patch SNR weighting.
  3. HR tracking across windows (Kalman or Viterbi over the spectrogram, with a harmonic-consistency prior).
  4. SQI-gated output with a confidence value.
- Each component is cheap, and the literature contains no unified quantitative ablation of this combination. That is my inference from not finding one.

### Gaps
- No quantitative study found comparing window lengths (e.g. 4 vs 6 vs 10 vs 15 s) for HR MAE in a modern benchmark. The usual practice is 6–10 s.
- No rPPG-specific Viterbi or harmonic-tracking study was found within the search budget.
- No head-to-head FFT-vs-peak-detection HR MAE comparison was found.

---

## Q4. Signal quality indices (SQI) for rPPG

### Takeaway
The de Haan & Jeanne SNR (power within ±a few BPM of the HR fundamental and its first harmonic, versus the rest of the band) is the most widely used and among the most predictive SQIs. Combining SNR with consecutive-segment cross-correlation and the relative difference of spectral peaks in a small classifier detected up to 81% of erroneous HR windows. Among retained windows, accuracy rose by up to 60% relative.

### Cited Findings
- **SNR definition (Rasche implementation of de Haan & Jeanne 2013):**
  - SNR = 10·log10 [ Σ Π(k)|X(k)|² / Σ (1−Π(k))|X(k)|² ] over 30–200 bpm.
  - Π(k) = 1 within ±5 bpm of the HR peak and ±5 bpm of 2×HR.
  - The HR peak can be estimated from the rPPG signal itself, so no reference is needed.
  - Source: [Ernst, Malberg & Schmidt, CinC 2020](https://www.cinc.org/archives/2020/pdf/CinC2020-165.pdf).
- **SQIs evaluated in Ernst et al. 2020** (BP4D+, 140 subjects, 25 fps, 10 s segments with 5 s overlap, 8207 segments):
  - Time domain: standard deviation (sSQI), pulse strength (psSQI), and max normalised cross-correlation of consecutive 2 s segments (aCCmSQI / sCCmSQI).
  - Frequency domain: snrSQI, and rdspSQI = (|X(k1)|−|X(k2)|)/|X(k1)|.
  - Source: [CinC 2020](https://www.cinc.org/archives/2020/pdf/CinC2020-165.pdf).
- **Correctness criterion and results:**
  - A window counts as correct if |HR error| ≤ max(5 bpm, 10% of reference HR).
  - Effect sizes (rank-biserial r): snrSQI −0.52 / −0.55 / −0.54 (G / H / Q channels); aCCmSQI about −0.50; rdspSQI −0.54 / −0.43 / −0.46.
  - Decision-tree SQI filter: sensitivity to erroneous windows was 81% (G); FP rate 21% (G), 9% (Q), 5% (H).
  - Accuracy of retained windows: G 0.50 → 0.80 (+60%), H 0.73 → 0.84, Q 0.70 → 0.83.
  - Motion-rich stimuli benefited most (+191% relative for G in the startle task).
  - Source: [CinC 2020](https://www.cinc.org/archives/2020/pdf/CinC2020-165.pdf).
- **Alternative SNR template:** binary windows of ±0.2 Hz around fHR and 2·fHR, over 0.7–4 Hz — [USPTO 11229372 (via search)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11229372).
- **Study-specific rejection threshold:** a driving-simulator rPPG/HRV study discarded segments with SNR below −17 dB — [arXiv 2605.24525](https://arxiv.org/pdf/2605.24525).
- **Learned/motion-based confidence scores (2026):** confidence is predicted from motion-related variables, with bagged-tree ensembles performing best and high AUC for flagging inaccurate HR — [J Med Syst 2026](https://link.springer.com/article/10.1007/s10916-026-02412-2).
- **pyVHR MAD:** the MAD across patches serves as an uncertainty estimate — [PMC9044207](https://pmc.ncbi.nlm.nih.gov/articles/PMC9044207).
- **SAC (Spatial Artifact Coherence)**, the ratio of off-diagonal to diagonal energy in the inter-patch green-channel covariance (0.75–2.5 Hz), predicts codec-induced failure. SAC explained 93.8% of between-variant variance in a single-author 2026 preprint, so it is a candidate compression-SQI — [arXiv 2606.04198](https://arxiv.org/html/2606.04198).

### Inferences
- **Practical recipe:** compute per-window SNR (fundamental + harmonic mask), peak-ratio (rdsp) and inter-window correlation. Reject or hold the last good HR when quality is low, and expose a confidence value.
- This directly targets the RMSE ≫ MAE "gross failure" pattern of classical methods, and is essentially free computationally.
- The challenge may penalise missing outputs, so prefer weighting and temporal hold or smoothing over hard rejection. That is a judgement call; check the scoring rules.

### Gaps
- I did not retrieve the original de Haan & Jeanne 2013 TBME paper text; the SNR definition is taken from citing papers.
- No modern rPPG-Toolbox-scale study quantifying MAE versus rejection-rate curves was found.

---

## Q5. Robustness to low frame rate, compression, low resolution

### Takeaway
- **Resolution:** classical methods are almost unaffected down to a 36x36 face crop, while deep models (Contrast-Phys) degrade.
- **Frame rate:** classical POS/OMIT are stable down to 10 fps (UBFC MAE about 2.7–3.5), while Contrast-Phys fails (48 MAE at 10 fps) and EfficientPhys roughly doubles its error.
- **Compression:** the main killer for everything. McDuff found rPPG SNR degrades even under mild compression and recommended about 10 Mb/s or more. A deep TDM model trained on raw video rises from 1.66 to 14.4 MAE cross-dataset as H.264 CRF goes 0 → 25. Compression-matched training is the established mitigation.

### Cited Findings
- **Resolution (UBFC-rPPG, face crop 128 / 72 / 36 px; MAE / PCC)** — [arXiv 2405.01230](https://arxiv.org/html/2405.01230):
  - POS 2.72 / 0.99, 2.74 / 0.99, 2.72 / 0.99
  - CHROM 3.06 → 3.02
  - OMIT about 3.45–3.49
  - Contrast-Phys 3.14 / 0.97 → 3.72 / 0.94 → 4.91 / 0.87
- **Frame rate (UBFC-rPPG, 30 / 20 / 15 / 10 fps; MAE)** — [arXiv 2405.01230](https://arxiv.org/html/2405.01230):
  - POS 2.74 / 3.08 / 2.79 / 2.69
  - OMIT 3.49 / 4.10 / 3.48 / 3.35
  - EfficientPhys 7.73 / 9.33 / 6.55 / 13.27
  - Contrast-Phys 3.72 / 21.55 / 38.45 / 48.38
  - The authors summarise frame-rate effects as inconsistent: non-learning methods are stable, while Contrast-Phys degrades sharply.
- **Additive noise (UBFC):** POS 2.74 → 13.00; CHROM 3.05 → 17.12; OMIT 3.49 → 13.62; EfficientPhys 7.73 → 32.81; Contrast-Phys 3.72 → 6.11. Blur had a mild or no effect. Lower bit depth degrades HR, but 6-bit stays close to 8-bit — [arXiv 2405.01230](https://arxiv.org/html/2405.01230).
- **Compression, McDuff et al. (FG 2017):** clear rPPG SNR degradation even under mild compression. Only videos at roughly 10 Mb/s or more retained reasonably good SNR, and H.265 outperformed H.264, especially with large motion — [cited in Comas et al., arXiv 2405.02652](https://arxiv.org/html/2405.02652).
- **Compression, TBME 2019 "Effects of Video Encoding on Camera-Based HR Estimation":** CRF is the most important parameter, even very small CRF values can be detrimental depending on the codec, and the authors do not recommend H.264 for camera-based HR — [TBME 2019](https://doi.org/10.1109/tbme.2019.2904326) (via search summary).
- **Deep model under an H.264 CRF sweep (Comas, Ruiz & Sukno 2024)**, TDM model trained on raw UCLA-rPPG:
  - Intra-UCLA MAE: 1.02 (CRF 0) → 1.26 (CRF 15) → 1.94 (CRF 20) → 2.98 (CRF 25).
  - **Cross-dataset (UCLA→UBFC) MAE:** 1.66 → 3.14 (CRF 5) → 4.57 (CRF 10) → 6.75 (CRF 15) → 12.82 (CRF 20) → 14.44 (CRF 25), with R falling from 0.95 to 0.43.
  - Their pulse-signal magnification network keeps error near the CRF-0 level up to CRF 20.
  - Source: [arXiv 2405.02652](https://arxiv.org/html/2405.02652).
- **Highly compressed datasets** — [arXiv 2405.02652](https://arxiv.org/html/2405.02652):
  - MAHNOB-HCI (H.264, 4200 kb/s): CHROM 13.49 MAE versus DeepPhys 4.57 (cross) and the proposed method 3.66 (cross).
  - COHFACE (MPEG-4, 250 kb/s): CHROM 7.8 MAE versus 0.70 for the compression-matched intra model; the cross-dataset proposed model reached 10.42 MAE with R ≈ −0.07, i.e. failure.
- **Matched-compression training:** models trained at the same compression level as the test set outperform mismatched ones (Nowara et al., BOE 2021). Zhao et al. (2018) found compression hurts the red and blue channels most. STVEN+rPPGNet (ICCV 2019) enhances compressed video — [cited in arXiv 2405.02652](https://arxiv.org/html/2405.02652).
- **Codec type matters, PCA vs CHROM (2026 single-author preprint):**
  - MPEG-4 at 85 kbps: CHROM MAE 27.47 on UBFC-rPPG and 21.80 on MCD.
  - Waveform r collapses to about 0 for CHROM and the PCA variant under MPEG-4 (85–500 kbps).
  - PCA-hybrid beats CHROM under H.264/H.265 (CRF 28) but loses under MPEG-4 with a raw source.
  - Recommends CHROM when SAC ≥ 0.48. POS is within 0–3 BPM of CHROM in all conditions.
  - Source: [arXiv 2606.04198](https://arxiv.org/html/2606.04198).
- Another source in the search results reported CHROM and OMIT handling compression better than GREEN and POS. This conflicts somewhat with the near-equivalence of POS and CHROM above — [search summary; Face2PPG, arXiv 2202.04101](https://arxiv.org/pdf/2202.04101).

### Inferences
- **On a Pi you control the encoder, so avoid compression in the pipeline:** process raw frames from the camera. If the challenge's test videos are compressed, the decisive factors are compression-matched training or augmentation for deep models and SQI gating for classical ones.
- **Low resolution:** a 36–72 px face crop is enough for classical methods. That allows heavy downscaling on the Pi (face ROI at about 72x72) with no measurable accuracy loss for POS/CHROM.
- **Low fps (10–15 fps) is safe for classical methods but risky for deep models trained at 30 fps.** If a Pi forces low fps, either retrain or augment the deep model with temporal resampling, or use POS/CHROM.

### Gaps
- No single study compares POS/CHROM versus modern deep models (FactorizePhys, RhythmMamba) across an H.264 CRF/bitrate sweep at QVGA or VGA. The team would need to run this ablation, which could be an innovation and robustness contribution.
- No quantitative results were found for full-frame QVGA (320x240) capture specifically, as opposed to face-crop size.

---

## Q6. Realistic accuracy targets

### Takeaway
- **Baseline conditions** (still subject, good light, uncompressed, UBFC/PURE-like): about 1 BPM MAE is achievable with tiny deep models cross-dataset; POS/CHROM give about 3–4 BPM, RMSE about 8–12.
- **Hard conditions** (MMPD-like: mobile, motion, varied skin and lighting, compression): expect 9–14 BPM MAE cross-dataset from both deep and classical methods. Only intra-dataset training reaches about 3 BPM.
- **Clinical-style acceptability** is commonly framed as an error within max(5 bpm, 10%).

### Cited Findings
- **Best cross-dataset MAE, easy pairs:** FactorizePhys UBFC→PURE 0.48 and PURE→UBFC 1.04 — [arXiv 2411.01542](https://arxiv.org/html/2411.01542). RhythmFormer PURE→UBFC 0.89 — [arXiv 2404.06483](https://arxiv.org/html/2404.06483).
- **Best cross-dataset MAE on MMPD:** RhythmFormer 8.98–9.08, PhysNet (UBFC-trained) 9.47, POS 12.36 — [arXiv 2404.06483](https://arxiv.org/html/2404.06483); [arXiv 2210.00716](https://arxiv.org/html/2210.00716).
- **Best intra-MMPD:** RhythmFormer 3.07, RhythmMamba 3.16 — [arXiv 2404.06483](https://arxiv.org/html/2404.06483).
- **Window correctness criterion** used in the SQI literature: |error| ≤ max(5 bpm, 10%) — [CinC 2020](https://www.cinc.org/archives/2020/pdf/CinC2020-165.pdf).
- **Edge quantisation:** a Raspberry Pi AI Kit (Hailo-8) port of DeepPhys reached MAE 3.01 BPM after quantisation, with reported latency falling to 28.94 ms. The "before" figure appears as 163,493.80 ms in the search snippet, which looks implausible and is probably a formatting error — [ISBI 2026 paper (via search)](https://informatika.stei.itb.ac.id/~rinaldi.munir/Penelitian/Makalah-ISBI-2026.pdf).

### Inferences
- **Suggested targets:**
  - Clean/still: MAE ≤ 2 BPM, RMSE ≤ 4, r ≥ 0.95.
  - Moderate motion or lighting changes: MAE ≤ 5.
  - Hard, MMPD-like: MAE ≤ 8–10, which would beat all published cross-dataset results.
- Report the percentage of windows within max(5 bpm, 10%) as a clinically interpretable metric alongside MAE, RMSE and r.

### Gaps
- I did not retrieve the npj Digital Medicine 2026 rPPG roadmap full text (paywall redirect) — [npj Digit Med 2026](https://www.nature.com/articles/s41746-026-02715-1). Its recommended clinical thresholds are not captured here.
- I did not directly verify the ANSI/AAMI EC13 HR-monitor accuracy standard text (commonly cited as ±10% or ±5 bpm).

---

## Q7. Open-source code and edge deployment evidence

### Takeaway
- **rPPG-Toolbox** (ubicomplab) is the de facto benchmark framework: 7 unsupervised and 10 supervised methods including FactorizePhys, RhythmFormer and PhysMamba, plus many dataset loaders. It is under a "Responsible AI" behavioural-use licence, which needs checking for competition use.
- **pyVHR** is GPL-3.0 and focused on classical, patch-based, real-time pipelines.
- **On the Pi, face detection and alignment are usually the bottleneck**, not the rPPG model.

### Cited Findings
- **rPPG-Toolbox:**
  - Unsupervised: GREEN, ICA, CHROM, LGI, PBV, POS, OMIT.
  - Supervised: DeepPhys, PhysNet, TS-CAN, EfficientPhys, BigSmall, PhysFormer, iBVPNet, PhysMamba, RhythmFormer, FactorizePhys.
  - Datasets: SCAMPS, UBFC-rPPG, PURE, BP4D+, UBFC-Phys, MMPD, iBVP, plus PhysDrive, SUMS and LADH loaders.
  - Metrics: MAE, RMSE, MAPE, Pearson, SNR, Bland-Altman.
  - Licence: Responsible AI License badge.
  - Source: [GitHub](https://github.com/ubicomplab/rPPG-Toolbox); [paper](https://arxiv.org/html/2210.00716).
- **pyVHR:**
  - Methods: GREEN, ICA, PCA, CHROM, PBV, SSR, POS, LGI, plus MTTS-CAN.
  - Pipelines: holistic and landmark-patch.
  - Speed: holistic under 0.025 s/frame on CPU or GPU (real-time at 30 fps); 100 patches take about 0.12 s/frame on CPU, about 5× faster on GPU.
  - Licence: GPL-3.0.
  - Source: [PMC9044207](https://pmc.ncbi.nlm.nih.gov/articles/PMC9044207).
- **Pi-class evidence:**
  - Raspberry Pi 5 (fan-cooled) streaming front end sustained 34.60 fps, with face processing rather than network inference limiting frame rate. This is a Research Square preprint, so less verified — [rs-10816153](https://www.researchsquare.com/article/rs-10816153).
  - A Face2PPG-based real-time system for low-power devices runs multithreaded at 30 fps — [arXiv 2508.18787](https://arxiv.org/pdf/2508.18787).
  - A Technion student prototype runs at 720p with 52 ms end-to-end latency, 15 fps in the real-time path — [Technion SDA 2025](https://sipl.ece.technion.ac.il/wp-content/uploads/2025/12/2025-SDA-Student-Competition-Remote-PPG-based-Vital-Signs-Monitoring.pdf).
  - An earlier efficient real-time HR/HRV method identifies face finding as the only bottleneck (about 23 fps at 1280x720) — [arXiv 1909.01206](https://arxiv.org/pdf/1909.01206).
- **Face2PPG** (Álvarez Casado & Bordallo López) is an unsupervised pipeline that includes OMIT — [arXiv 2202.04101](https://arxiv.org/pdf/2202.04101).

### Inferences
- **Recommended core pipeline for the Pi:**
  1. A lightweight face/landmark tracker (detect every N frames, track in between).
  2. Timestamp-resampled multi-patch RGB traces.
  3. POS (plus CHROM) per patch.
  4. SNR-weighted fusion and spectral HR tracking with an SQI-gated output.
  5. Optionally, a tiny deep model (FactorizePhys or a quantised RhythmMamba/PhysNet) on a 72x72 face crop as a second estimator, fused by SQI.
- This hits accuracy (deep where conditions are clean), robustness (classical fallback stable to fps and resolution), and complexity (tens of K to about 1 M params).
- **Licensing:** the Responsible AI licence of rPPG-Toolbox and GPL-3.0 of pyVHR may constrain redistributing a combined submission. Check the competition's open-source requirements.

### Gaps
- No published end-to-end HR accuracy plus latency figures were found for FactorizePhys or RhythmMamba on a Raspberry Pi 4/5 CPU (ARM). Benchmarking with ONNX Runtime or TFLite int8 is needed.
- The full text of the rPPG-Toolbox Responsible AI licence was not reviewed.
