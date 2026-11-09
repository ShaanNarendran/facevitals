# Win FaceVitals with honest, quality-gated rPPG

The best route to the FaceVitals final is **not** the most complex deep network. It is a **quality-gated hybrid**: build a classical core first, add a tiny deep model second, and report everything with clinical-grade honesty. The classical core is multi-patch POS/CHROM with timestamp resampling, SNR-weighted patch fusion, harmonic-aware spectral tracking and a signal-quality index (SQI). The deep model is a FactorizePhys- or EfficientPhys-class network with about 50 K–2 M parameters, fused with the classical estimate by SQI. Under the challenge's own constraints, classical methods are the safer foundation:
- **POS holds roughly 2.7 BPM MAE down to a 36-pixel face crop and down to 10 fps**, while some deep models collapse at 10 fps (48 BPM MAE) ([arXiv 2405.01230](https://arxiv.org/html/2405.01230)).
- **POS's error is dominated by occasional gross failures** (RMSE 11.8 vs MAE 3.7 on PURE) that cheap SQI gating and tracking can remove ([RhythmMamba, arXiv 2404.06483](https://arxiv.org/html/2404.06483)).

The innovation story should combine four things. The first two are gaps no published work has closed: (1) a unified ablation of SQI-gated classical-plus-deep fusion and (2) a physics-correct constraint stress-test bench. The other two are what a PhD jury rewards: (3) a target-locked multi-face pipeline with bystander-rejection metrics and (4) a rigorous "beat the constant predictor" evaluation for SpO2 and BP.

Be honest about the vitals beyond HR. RGB-face SpO2 reaches about 1 percentage point (pp) MAE only because test ranges are narrow. A constant predictor ties a trained model at 0.98 pp on 600 subjects ([arXiv 2508.17924](https://arxiv.org/html/2508.17924v1)). Video BP is still only 1–3 mmHg better than predicting the mean ([arXiv 2412.10679](https://arxiv.org/html/2412.10679v3)). The team should therefore frame SpO2 as **desaturation-trend tracking** and BP as an **exploratory estimate**, each always shown next to its baseline.

Three practical rules follow:
- **Data:** train the submitted model only on licence-clean data (MCD-rPPG under CC BY 4.0, the team's own IRB-approved recordings, and the organiser's data). Benchmark on non-commercial sets but do not train on them. This matters because SCAMPS, MMPD, SUMS and iBVP forbid commercial use, and the MathWorks clause in the brief creates that commercial exposure.
- **Round 1 scope:** handle four or five constraints rather than two. Low resolution (C4) and low bandwidth (C5) are nearly free for a classical core. Multi-face (C3) and masks (C7) are pure engineering wins. Skin-tone stratification (C2) is mandatory reporting anyway.
- **Calendar:** start ethics approval, the organiser data agreement and dataset access requests **this week**. Every later phase depends on them.

## A classical core plus a tiny deep model beats either alone

### The evidence: classical methods are robust where the challenge is hard

The rPPG-Toolbox benchmark makes the trade-off concrete:
- **Training-free methods:** POS and CHROM are the best, at about 3.7–4.5 BPM MAE on PURE, UBFC-rPPG and UBFC-Phys, and 12–14 BPM on the hard, compressed, mobile MMPD set. GREEN and ICA are far worse ([rPPG-Toolbox, arXiv 2210.00716](https://arxiv.org/html/2210.00716)).
- **Statistical comparison:** pyVHR's analysis across five datasets finds POS, CHROM, PCA and SSR statistically tied at the top ([pyVHR/HAL](https://hal-lara.archives-ouvertes.fr/LIRFAI/hal-03652383v1)).
- **Deep models, easy cross-dataset pairs:** they reach about 1 BPM. **FactorizePhys** reaches 0.48 BPM UBFC→PURE and 1.04 PURE→UBFC with only **52 K parameters (0.22 MB)**, at about 96 ms per clip on a laptop CPU ([FactorizePhys, arXiv 2411.01542](https://arxiv.org/html/2411.01542)).
- **Deep models, hard data:** every model degrades to about 9–17 BPM on MMPD. The best, RhythmFormer, sits at 9.0, against 12.4 for POS ([arXiv 2404.06483](https://arxiv.org/html/2404.06483)).

The gap between deep and classical methods is therefore large on clean data and small on the messy data the constraints describe.

The robustness study that best matches the challenge's constraints tips the decision:
- **Resolution:** on UBFC-rPPG, POS stays at **2.72–2.74 BPM MAE** whether the face crop is 128, 72 or 36 pixels. Contrast-Phys drifts from 3.14 to 4.91.
- **Frame rate:** POS is flat from 30 to 10 fps (2.74 → 2.69). EfficientPhys roughly doubles (7.73 → 13.27), and Contrast-Phys collapses to **48.4 BPM**.
- **Frame drops:** at 50% random frame loss, timestamp-based interpolation rescues POS from 65.9 to **3.03 BPM**, while deep models got worse after the same mitigation.

All three results come from the same study ([arXiv 2405.01230](https://arxiv.org/html/2405.01230)). C4 (≤640×480, stress 320×240) and C5 (≤10 fps, ≤500 kbps) are therefore problems a classical core already tolerates, and a deep model trained only at 30 fps is a liability.

Compression is the main exception and hurts everything. A deep model trained on raw video climbs from 1.66 to 14.4 BPM cross-dataset as H.264 CRF goes from 0 to 25 ([arXiv 2405.02652](https://arxiv.org/html/2405.02652)). But H.265 at about 0.26 Mbps on PURE cost only about 0.3 BPM ([Gudi et al., arXiv 2012.15846](https://arxiv.org/pdf/2012.15846)). The 500 kbps limit is survivable for a mostly still face if the codec is reasonable.

### The recommended pipeline

The recommended architecture keeps the classical path as the always-on backbone and adds the deep model where it pays off:

| Stage | Choice | Why |
|---|---|---|
| Capture / input | Fixed fps, auto-exposure, white balance and autofocus locked; per-frame sensor timestamps | Auto modes inject step artefacts; timestamps enable resampling |
| Face detection | YuNet every 3–5 frames on a low-res stream | About 6 ms at 160×120 on a Pi 4B ([OpenCV Zoo](https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md)) |
| Target tracking | Kalman + IoU/Hungarian (ByteTrack-style); face-embedding re-ID on events | Needed for C3; under 1 ms per frame for tracking |
| ROI | Landmark-defined skin patches (forehead, glabella, malar/upper cheeks, below-eye, lower cheeks if visible) | Forehead and cheeks are the most accurate ROIs ([PMC12297079](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12297079/)) |
| Trace conditioning | Per-patch RGB means, resampled to a uniform grid from timestamps | Rescues classical methods under frame drops ([arXiv 2405.01230](https://arxiv.org/html/2405.01230)) |
| Pulse extraction A | POS (primary) and CHROM per patch | Best training-free methods; robust to resolution and fps |
| Pulse extraction B | Tiny deep model on a 72×72 target crop, chunked once per second | Adds about 2–3 BPM accuracy on clean data |
| Fusion | Per-patch / per-estimator SNR weighting; MAD across patches as uncertainty | pyVHR-style median + MAD ([PMC9044207](https://pmc.ncbi.nlm.nih.gov/articles/PMC9044207)) |
| HR estimation | Welch/FFT with zero-padding; harmonic-aware Viterbi tracking across windows | Removes harmonic- and motion-peak locking |
| Output | HR + confidence; hold last good value and flag when SQI is low | Targets the RMSE ≫ MAE failure pattern |

### The signal-quality index and the deep-model choice

The SQI design is well grounded. The de Haan–Jeanne SNR compares power within ±5 BPM of the HR peak and its first harmonic with the rest of the band, and needs no reference signal. Ernst et al. combine it with consecutive-segment cross-correlation and a peak-ratio feature in a small decision tree. That combination **detected 81% of erroneous HR windows and raised the accuracy of retained windows from 0.50 to 0.80**, with the largest gains in motion-rich segments ([Ernst et al., CinC 2020](https://www.cinc.org/archives/2020/pdf/CinC2020-165.pdf)). The challenge may penalise missing outputs, so the default behaviour should be to weight and hold rather than drop windows. Report coverage (the fraction of windows with a valid output) alongside error.

For the deep branch, the choice depends on how it will be deployed:
- **Offline accuracy:** FactorizePhys has the best generalisation per parameter.
- **Raspberry Pi CPU:** EfficientPhys-C has a published Pi 4B cost of **about 40 ms per frame** ([EfficientPhys, arXiv 2110.04447](https://ar5iv.labs.arxiv.org/html/2110.04447)).
- **Low-latency streaming:** ME-Flow, a causal streaming model, reports 0.15 GFLOPs and 9.46 ms per frame on a laptop CPU. It scored MAE 1.78/2.53 BPM static/moving in a real-world C920 study ([ME-rPPG, arXiv 2504.01774](https://arxiv.org/html/2504.01774v1)).

Two cautions apply. First, RhythmMamba's size is reported inconsistently (1.07 M parameters in its own paper vs 4.9 M in the ME-rPPG table), so do not rely on either without measuring. Second, any deep model must be trained with fps, codec and low-light augmentation, given how badly 30 fps-trained models behave at 10 fps.

### Four innovation claims the jury can verify

1. **SQI-gated hybrid fusion with calibrated confidence, plus a full ablation.** Literature has multi-ROI gains of 0.5–1 BPM for POS (COHFACE: 5.18 → 4.63 → 4.06 with angle-guided ROIs and graph denoising ([arXiv 2604.11395](https://arxiv.org/pdf/2604.11395))). It also has SQI filtering and Kalman tracking. The research found **no unified quantitative ablation of this combination**.
2. **Harmonic-aware spectral tracking.** No rPPG-specific Viterbi or harmonic-tracking study turned up in the search, so a Viterbi tracker over the spectrogram with a 2×HR consistency prior is a defensible novelty. That absence reflects the search budget, not proof that none exists.
3. **A physics-correct constraint stress-test bench.** No study compares POS/CHROM with modern tiny deep models across an H.264/H.265 bitrate sweep at QVGA. Low-light simulation that adds shot noise in linear RGB is also rarely validated against real low-lux video. The team can do both and publish the curves.
4. **Clinically honest reporting of SpO2 and BP**, covered below. This is precisely the critique a physiological-signal-processing jury will apply.

## Handle five constraints in Round 1, not two

The brief rewards each constraint beyond the minimum two. Several are cheap for this architecture, so the plan is to submit **C3, C4, C5 and C7 with full simulations, plus a C2 skin-tone stratification table** in Round 1. C1 (low light) and C6 (aging) are added in Round 2, once the team has its own recordings. The per-constraint plan is summarised first, then justified.

| Constraint | Evidence strength | Strategy | Simulation for the report | Data |
|---|---|---|---|---|
| C1 low light ≤50 lux | **Thin** at ≤50 lux | Enhance only the detector input (gamma/CLAHE on a low-res copy); extract traces from raw frames; long exposure; at 10 fps allow up to 100 ms exposure | Brief's brightness scaling + `imnoise`, **plus** a linear-RGB shot- and read-noise model | MSLR (3/8/35 lux) for validation; own recordings at 50/20/5 lux |
| C2 skin tone I–VI + vitiligo | Moderate (bias documented) | POS normalisation, multi-patch SNR weighting, longer windows; skin-tone augmentation for the deep branch; per-patch ITA for vitiligo | `rgb2lab` ITA per subject from a well-exposed reference frame; stratified tables | Organiser data (likely Indian IV–VI, unverified); own data; VitalVideos if licensed |
| C3 multi-face | Engineering (no rPPG dataset) | Detector + ByteTrack-style tracker + embedding enrolment of the target; invalidate the buffer on ID loss | Composite or tile single-subject videos; Kalman tracking per brief | Composites of public videos; own multi-person recordings, one oximeter each |
| C4 low res ≤640×480 / 320×240 | **Strong**: POS flat to 36 px crops | Classical core; deep branch at 72×72 | `imresize` sweep 640×480 → 320×240 → 160×120 | Any uncompressed source |
| C5 ≤10 fps, ≤500 kbps | Strong for fps; moderate for bitrate | Timestamp resampling; 15–20 s windows or zero-padding at 10 fps; codec-augmented deep training; SAC compression SQI | `VideoWriter` / ffmpeg re-encode at 500 kbps, fps decimation | Re-encode uncompressed sources (UBFC, own data) |
| C6 aging 60–70+ | **None** found | Same multi-patch SNR weighting; abstention; flag arrhythmia | Brief's `imfilter` wrinkle texture | Own recordings of older adults; organiser clinical data |
| C7 masks ≥40% lower face | Moderate | Patches above the mask line (forehead, periocular, malar); exclude occluded skin; mask-augmented deep training | `insertShape` mask overlays at 40–60% coverage | Synthetic overlays; own masked recordings |

### C4 and C5 are the cheapest credit available

The resolution and frame-rate results above make C4 nearly free. At 320×240 a face of 80–120 pixels still contains thousands of skin pixels, and spatial averaging suppresses quantisation noise.

At 10 fps the Nyquist limit is 300 BPM, so HR is not aliased. The cost is spectral resolution: a 10 s window gives only 6 BPM bins. Use 15–20 s windows, zero-padding, or parabolic peak interpolation, and say explicitly that this trades real-time latency for resolution.

For the 500 kbps half of C5:
- **Codecs:** H.265 is far more forgiving than H.264. Even very small CRF values can hurt depending on the codec ([TBME 2019](https://doi.org/10.1109/tbme.2019.2904326)). MPEG-4 at 85 kbps collapses CHROM to 27 BPM MAE ([arXiv 2606.04198](https://arxiv.org/html/2606.04198)).
- **Encoder settings:** if the team controls the encoder, use H.265, a short GOP and a low QP on the face.
- **Deep branch:** train it with on-the-fly re-encoding at random CRF/bitrate (300 kbps–5 Mbps) and random fps (8–30).
- **Compression SQI:** add the Spatial Artifact Coherence index as a compression-failure detector. In a single-author preprint it explained 93.8% of codec-induced variance ([arXiv 2606.04198](https://arxiv.org/html/2606.04198)). That evidence is thin, so present it as an exploratory SQI.

### C3 and C7 are engineering problems with clean evaluation stories

**Multi-face (C3).** No published Pi system combines ByteTrack-style tracking with face re-identification for a single designated target. The building blocks are cheap, though:
- Kalman + IoU association costs well under 1 ms per frame.
- A MobileFaceNet/SFace embedding costs about 28–69 ms on a Pi 4B CPU ([OpenCV Zoo](https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md)), so it runs only at enrolment, at track birth or overlap, and about once per second.

The design is to enrol the target with 2 s of frontal embeddings, then bind the tracker ID to that identity with cosine-similarity voting and hysteresis. Whenever identity is lost, invalidate the rPPG buffer so a bystander's pulse is never fused in. The evaluation is what separates a strong entry from a demo: report **target-ID precision, ID-switch count, time-to-reacquire after occlusion**, and HR error with a bystander whose HR differs from the target's.

**Masks (C7).**
- **Size of the problem:** cloth masks raised HR MAE by more than 80% for methods designed on unmasked faces. Synthetic mask augmentation shifts models toward the forehead and periocular regions and narrows the gap ([Speth et al., arXiv 2101.04096](https://arxiv.org/abs/2101.04096)).
- **What works:** simply excluding occluded skin improved POS from 4.72 to **2.91 BPM** and beat GAN inpainting ([arXiv 2405.01230](https://arxiv.org/html/2405.01230)).
- **Patch design:** combining patches helps, with diminishing returns beyond about 60 ([PMC12297079](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12297079/)).

A patch grid above the landmark-derived mask line, with SNR weighting, is therefore evidence-aligned and costs almost nothing. Add a minimum-skin-pixel flag for the hard case of a fringe plus a mask.

### C2 needs measurement discipline more than new algorithms

Skin-tone bias is real and large for some models. Models trained on PURE, UBFC or MMPD and tested per Fitzpatrick type gave PhysNet near 0–1.6 BPM for types I–II but **14.65 BPM for type VI** ([VitalVideo benchmark, arXiv 2404.05003](https://arxiv.org/html/2404.05003v1)). Treat the often-quoted "CHROM 5.2 → 14.1 BPM" figure with caution: an audit attributes it to Dasari et al., but that paper never grouped subjects by Fitzpatrick type ([PMC8175478](https://pmc.ncbi.nlm.nih.gov/articles/PMC8175478/); [PMC12491395](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12491395/)).

For the deep branch, the best-supported mitigation is synthetic skin-tone augmentation. Ba et al. report about a **31% HR MAE reduction for dark-skinned subjects** ([arXiv 2106.06007](https://arxiv.org/pdf/2106.06007)). The UCLA project page reports 38% for a revised version ([UCLA](https://visual.ee.ucla.edu/rppg_augmentation.htm)).

The ITA computation the brief asks for has known traps. L* scales with exposure, so an underexposed light-skinned face reads as "dark", and white balance shifts b*. To avoid this:
- Compute ITA once per subject, from a well-exposed, white-balanced reference frame.
- Use median cheek and forehead pixels, excluding speculars, beard and shadow.
- For vitiligo, compute ITA and SNR per patch, because the face is bimodal. No rPPG study on vitiligo was found, so any vitiligo result is novel but should be presented as a case study.

The FDA's draft oximeter guidance already uses forehead ITA thresholds (≤ −50° for the darkest group), which gives the team a regulatory anchor for the stratification bins ([FDA draft guidance](https://www.fda.gov/media/184896/download)).

### C1 and C6 are where the evidence runs out, so measure

The most rigorous low-light study (CHILL) found illumination mattered less than elevated HR ([PMC12678791](https://pmc.ncbi.nlm.nih.gov/articles/PMC12678791/)). Its "dark" condition, however, is reported as 140 lux on a DSLR, and its arXiv version gives only LED settings and pixel means ([arXiv 2503.11697](https://arxiv.org/html/2503.11697)). Either way it says nothing reliable about ≤50 lux on a small Pi sensor.

Xi et al. report the failure modes in genuinely low light ([arXiv 2303.09336](https://arxiv.org/abs/2303.09336)):
- face detection fails;
- the pulse amplitude sinks into noise;
- POS's skin-tone assumption breaks.

**MSLR** is the one verified dataset with measured 3, 8 and 35 lux conditions ([GitHub MSLR](https://github.com/wwenmaositu/MSLR)). Use it to validate the low-light simulator rather than to train, because its data licence is ambiguous.

The brief's `imnoise`/brightness-scaling simulation understates the problem: scaling brightness without adding shot noise leaves the pulse SNR almost intact. Implement both:
- the brief's version, for compliance;
- a physics version: undo gamma, scale by the lux ratio, add Poisson shot noise and Gaussian read noise, re-apply gamma, then quantise.

Show that only the physics version reproduces MSLR's real degradation. That comparison is itself a contribution.

For aging skin (C6) the research found **no age-stratified rPPG validation of adults 60+**. CHILL was 18–32, and HiDURPA had no one over 50 ([EAH Jena](https://www.eah-jena.de/vitawave/publications/videocompressionrppg)). The team's own older-adult recordings, and the organiser's hospital data, are the only route to a real number. Present the hypotheses as hypotheses: lower perfusion, wrinkle micro-shadows, pigment heterogeneity, and arrhythmia breaking the single-peak spectral assumption.

## SpO2 and BP must be framed against their baselines

### SpO2: aim for desaturation tracking and prove the model beats a constant

Published RGB-face SpO2 MAEs of 0.85–1.3 pp are mostly an artefact of narrow test ranges. The best evidence:
- **Range split:** an EfficientNet-B3 STMap model on VIPL-HR scores **0.98 pp MAE for SpO2 ≥ 95% but 3.08 pp below 95%**. A ratio-of-ratios (RoR) baseline scores 1.69 and 4.48 on the same split ([Cheng et al., PMC10968547](https://pmc.ncbi.nlm.nih.gov/articles/PMC10968547)).
- **Constant baseline:** on 600 subjects (MCD-rPPG, SpO2 mean 98.01 ± 1.29) the optimal constant and the trained model **both score 0.98 pp** ([arXiv 2508.17924](https://arxiv.org/html/2508.17924v1)).
- **Low MAE without tracking:** classic RoR can show low MAE with near-zero correlation (MAE 2.08, r ≈ 0.01–0.05) ([Mathew et al., arXiv 2107.05087](https://arxiv.org/html/2107.05087v3)).
- **Physics:** broadband, overlapping RGB filters and the missing IR channel mean a two-channel ratio "yields a trend rather than a calibrated SpO2 value" ([US 10667723](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10667723)). Visible-light calibration also shifts with posture ([TU/e](https://www.es.ele.tue.nl/~sander/publications/pmea19-posture.php)).

The recommended SpO2 pipeline reuses the HR front end:
1. Build multi-ROI RGB DC and AC components (DC low-passed below 0.3 Hz; AC band-passed 0.75–2.5 Hz around the tracked HR).
2. Compute two estimators. The first is a **calibrated multi-channel RoR on R/G and R/B**, an interpretable baseline; multi-channel RoR was 25% better than two-channel in Tian et al. ([arXiv 2107.08528](https://arxiv.org/pdf/2107.08528)). The second is a **small two-stream DC/AC CNN** in the style of Akamatsu et al., with a MobileNet or 1D-CNN backbone replacing ResNet18 ([arXiv 2212.07116](https://arxiv.org/abs/2212.07116)).
3. **Gate SpO2 output on HR SNR**, because SpO2 fails before HR in low light.

Joint HR–SpO2 training gave 7.6–10.6% relative MAE reduction in Summit Vitals, which supports a shared front end ([arXiv 2409.19223](https://arxiv.org/pdf/2409.19223)). Camera settings matter more here than for HR. Samsung's facial RoR study, which reached RMSE 1.93 with r = 0.97 down to 81% SpO2, **disabled camera auto-tuning** ([Samsung Research](https://sra.samsung.com/publications/estimating-spo2-with-deep-oxygen-desaturations-from-facial-video-under-various-lighting-conditions-a-feasibility-study/)). Tang et al. added a colour checker in-frame ([arXiv 2503.01699](https://arxiv.org/html/2503.01699v1)).

**Honest targets:** about 1–1.5 pp MAE overall, 2.5–4 pp below 94%, and within-subject correlation of 0.4–0.7 during breath-holds. These are inferences from the literature above, not guarantees.

The reporting checklist a PhD jury will expect:
- **Baselines and skill score:** a train-mean constant and a last-value baseline, with skill = 1 − MAE_model / MAE_constant.
- **Range-stratified errors:** MAE and RMSE per SpO2 bin (≥95, 90–95, <90), labelled "ARMS vs reference pulse oximeter", never "ARMS vs SaO2".
- **Agreement and tracking:** Bland–Altman, within-subject correlation, nadir error and lag, and AUROC for detecting SpO2 < 94% or a ≥3–4 pp drop.
- **Fairness:** errors by Monk or Fitzpatrick group, with confidence intervals.
- **Oximeter lag:** estimate it by cross-correlation and report results with and without compensation.

Also state the regulatory ceiling plainly. The FDA draft guidance requires ≥150 participants and ≥3,000 arterial SaO2 pairs spread across 70–100%, with ARMS < 3% ([FDA](https://www.fda.gov/media/184896/download)). A student breath-hold study cannot test that. The reference oximeter itself under-detects occult hypoxemia in Black patients: 11.7% vs 3.6% in Sjoding et al. ([AHRQ PSNet](https://psnet.ahrq.gov/issue/racial-bias-pulse-oximetry-measurement)).

### BP: build a feature model in February and present it as exploratory

No published video-BP method meets AAMI/ISO 81060-2 (mean error ≤5 mmHg, SD ≤8 mmHg) under subject-independent, calibration-free testing. The most reliable studies agree:

| Study | Subjects | Mean predictor MAE SBP/DBP | Model MAE SBP/DBP |
|---|---|---|---|
| U-FaceBP ([arXiv 2412.10679](https://arxiv.org/html/2412.10679v3)) | 1,094 | 14.29 / 9.85 | 11.05 / 7.90 |
| MCD-rPPG ([arXiv 2508.17924](https://arxiv.org/html/2508.17924)) | 600 | 13.78 / 7.50 | 12.82 / **8.39 (worse)** |
| Clinical cohort ([arXiv 2503.00890](https://arxiv.org/html/2503.00890v1)) | 134 | 14.96 (clinical features) | 11.35 |
| Schrumpf et al. ([ar5iv 2104.09313](https://ar5iv.labs.arxiv.org/html/2104.09313)) | 14 analysed | 13.4 / 8.9 | 13.02 / 9.81 |

The critique literature explains why:
- **Leakage:** Mehta et al. showed random-window splits shrink SBP SD from 23.1 to 8.0 mmHg.
- **Ill-conditioning:** 33% of MIMIC windows have a near-identical PPG with very different SBP.
- **Information content:** learned features capture only about 9–13% of SBP information ([Mehta et al., Sci Rep 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11306225/)).
- **Benchmark result:** the best subject-disjoint models reach only about 80–94% of the mean predictor's MAE (MASE) ([González et al., Sci Data 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10030661/)).

Because faces reveal age, sex and adiposity, a face-video model can learn demographics instead of haemodynamics. The team should therefore include a **demographics- or appearance-only baseline**.

The recommended BP approach is deliberately small:
- **Features:** beat-averaged pulse-morphology features from the existing rPPG signal (rise time, widths at 25/50/75%, systolic/diastolic area ratio, VPG/APG ratios), HR and HRV, and forehead-vs-cheek phase lag at the HR fundamental with sub-frame interpolation.
- **Regressor:** ridge or LightGBM, trained subject-disjoint on MCD-rPPG, which has cuff BP plus age, sex and BMI ([HF dataset card](https://huggingface.co/datasets/kyegorov/mcd_rppg)).

Feature-based models were competitive with deep ones on small datasets ([González et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10030661/)). Facial phase differences are only milliseconds at 30 fps and not consistent across subjects ([arXiv 2412.10679](https://arxiv.org/pdf/2412.10679)), so treat phase lag as a feature, not a method.

If the organisers allow one cuff reading per subject, report a calibrated mode too, alongside the very strong "repeat the calibration value" baseline (bias 0.79, SD 8.61 mmHg in Mehta et al.).

Expect **SBP MAE about 10–13 and DBP 7–9 mmHg**. For the final, report all of the following:
- ME ± SD with ISO Criterion 1 pass/fail;
- MAE, MASE and r;
- BHS percentages within 5/10/15 mmHg;
- IEEE 1708 grade;
- Bland–Altman and predicted-vs-reference scatter (to expose regression to the mean);
- errors by BP category.

Cite the ESH position that cuffless devices are not recommended for diagnosing or managing hypertension ([Curr Cardiol Rep](https://link.springer.com/article/10.1007/s11886-023-01956-w)). On screen, show a categorical readout (normal / elevated / high) with an uncertainty band rather than a bare number.

## Train only on licence-clean data and start ethics approval now

### The MathWorks clause constrains which data can train the model

The brief's clause that submissions may go to MathWorks for commercial use collides with most rPPG dataset licences:
- **MMPD:** "any commercial usage is banned", and a faculty member must sign ([GitHub](https://github.com/McJackTang/MMPD_rPPG_dataset)).
- **SUMS:** academic only ([GitHub](https://github.com/thuhci/SUMS)).
- **iBVP:** academic research only, with a supervisor-signed EULA ([GitHub](https://github.com/PhysiologicAILab/iBVP-Dataset)).
- **SCAMPS:** its R-UDA licence forbids using the Data **or any Results** in a commercial offering ([LICENSE](https://raw.githubusercontent.com/danmcduff/scampsdataset/main/LICENSE.txt)). This rules out FactorizePhys's attractive synthetic-pretraining route for the submitted model.

The tiers that follow:

| Tier | Datasets | Use in the submission |
|---|---|---|
| Train the submitted model | **MCD-rPPG** (CC BY 4.0; 600 subjects × 2 states × 3 cameras; PPG, ECG, SpO2, BP, age/sex/BMI) ([HF](https://huggingface.co/datasets/kyegorov/mcd_rppg)); own IRB data with commercial-use consent; organiser Sri Ramachandra data (terms to confirm) | Training, validation, test |
| Negotiate | **VitalVideos** (Europe-1: 850 subjects, Fitzpatrick I–VI, 40% over 50; WorldWide: about 6,500 subjects, Fitzpatrick types fairly evenly spread, 18% with SBP >160; commercial R&D licence by email) ([vitalvideos.org](https://www.vitalvideos.org/)) | Training if a commercial licence is obtained; otherwise academic benchmarking |
| Benchmark only | MMPD, SUMS, iBVP, CHILL, SCAMPS, CAST-Phys | Cross-dataset test numbers, as their EULAs allow |
| Ambiguous; ask authors | UBFC-rPPG ("research purpose"), MSLR (MIT on repo), COHFACE, M3PD, MPU-rPPG | Benchmark; train only after written clarification |

Two code licences matter as well:
- rPPG-Toolbox carries a Responsible AI licence, and pyVHR is GPL-3.0 ([rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox); [PMC9044207](https://pmc.ncbi.nlm.nih.gov/articles/PMC9044207)).
- POS, CHROM, the SQI and the tracker are each a few dozen lines, so **reimplement the core** in MATLAB and Python. Use the toolbox only to produce comparison baselines.

Ask the organisers **in writing** whether models trained on non-commercial data are acceptable given the MathWorks clause. The answer could reopen MMPD and SUMS for training.

### Close the coverage gaps with your own recordings and degradation

**What public data cannot cover.** No public dataset has real masks or multiple people with per-person references, and public data is thin at ≤50 lux and for adults 60+. Data for C4 and C5 is best produced by degrading uncompressed masters (UBFC-rPPG, own lossless recordings): downscale, decimate fps and re-encode at ≤500 kbps. The ground truth stays valid because only the video changes.

**What to collect.** A realistic plan is 30–40 subjects over 2–3 weeks:
- **Equipment:**
  - lossless 1080p webcam capture, plus the Pi camera;
  - a CMS50D+/CMS50E-class oximeter logged to PC (about 60 Hz PPG; check the actual rate on the purchased unit);
  - a validated upper-arm cuff before and after each clip;
  - a lux meter at the face plane, a dimmable LED panel and a colour checker.
- **Synchronisation:** one monotonic clock for all streams and an LED flash in frame.
- **Session, about 20 minutes per subject:**
  - rest at 300+ lux;
  - rest at 50, 20 and 5 lux;
  - talking and head motion;
  - surgical and cloth masks;
  - post-exercise;
  - multi-person with an oximeter on each person;
  - three seated breath-holds after normal exhalation, 20–40 s each with ≥60 s recovery, stopping on discomfort or SpO2 below 88–90%.
- **Recruitment:** deliberately recruit Fitzpatrick V–VI and adults 60+.

The breath-hold protocol mirrors ethics-approved precedents ([Akamatsu et al.](https://arxiv.org/pdf/2212.07116); [Mathew et al.](https://arxiv.org/html/2107.05087v3)). Expect nadirs of about 90–95%, so SpO2 claims can cover only about 88–100%. Use tiered consent (internal only / share with organisers / public), as iBVP and CHILL do ([iBVP](https://github.com/PhysiologicAILab/iBVP-Dataset); [CHILL Zenodo](https://zenodo.org/records/14637544)). The consent form must name **use by the organisers and MathWorks, including commercial use**.

Ethics approval rarely takes under four weeks, so Round 1 must stand on public and organiser data. Own data arrives for Round 2.

## Evaluate like a clinical study and report complexity like an embedded one

### Evaluation protocol

This protocol pre-empts every common objection:
1. **Splits:** subject-disjoint splits only (5-fold or leave-one-subject-out), plus at least two cross-dataset directions. Segment-level splits overstate accuracy through leakage ([npj Digit Med roadmap 2026](https://www.nature.com/articles/s41746-026-02715-1)).
2. **HR windows:** 10 s windows with a 1 s step for real-time use, 30 s windows, and whole-clip results comparable to rPPG-Toolbox tables. Extend the band to about 3.5 Hz, because exercise data reaches 141–160 BPM.
3. **Reference HR:** compute it from the reference PPG with the same window and estimator, not from the oximeter's lagged 1 Hz display. Estimate the sync lag per recording by cross-correlation (±2 s search).
4. **Metrics:** MAE, RMSE, Pearson r and MAPE with **bootstrap confidence intervals over subjects**, not windows. Use a repeated-measures Bland–Altman (Bland & Altman 2007 correction, or per-subject averaging). Add the percentage of windows within max(5 BPM, 10%), the correctness criterion used in the SQI literature ([CinC 2020](https://www.cinc.org/archives/2020/pdf/CinC2020-165.pdf)).
5. **Baselines:** a training-mean "dummy" estimator as CHILL uses ([arXiv 2503.11697](https://arxiv.org/html/2503.11697)), POS/CHROM, and one supervised baseline (PhysNet or EfficientPhys).
6. **Stratification:** every metric per constraint and per Fitzpatrick/ITA group, with n subjects and n windows. Flag strata with fewer than about 5 subjects as unreliable.

The `.mat` deliverables map naturally onto this protocol. The `metrics` struct holds MAE, RMSE, r and Bland–Altman bias and limits of agreement. The `params` struct records resolution, fps, lux, distance, Fitzpatrick type and age. Where public datasets lack a field (most lack lux), record it as "not reported" rather than guessing.

The HR accuracy targets that would be competitive:

| Condition | Target MAE | Literature anchor |
|---|---|---|
| Clean, still, uncompressed | ≤2 BPM (RMSE ≤4, r ≥0.95) | FactorizePhys 0.48–1.04 cross-dataset; POS 2.7–4.0 |
| Moderate motion / lighting | ≤5 BPM | ME-Flow 2.53 moving; POS COHFACE about 4–5 |
| MMPD-like hard data | ≤8–10 BPM | Best cross-dataset 9.0 (RhythmFormer); POS 12.4 |

### Edge stack and complexity reporting

**No published Pi 5 CPU benchmark exists for any deep rPPG model or for YuNet or MediaPipe.** Every Pi 5 figure below is extrapolated and must be measured. The ISBI 2026 system shows about **30 fps** is achievable on the Raspberry Pi AI Kit (Hailo-8L) with three components:
- SCRFD-2.5G detection at 18.9 ms;
- a centroid tracker;
- DeepPhys quantised to mixed UINT4/UINT8 at 28.9 ms, with MAE unchanged at 3.01 BPM.

The same paper found that only DeepPhys compiled for Hailo; 3D-conv models failed. Its hardware description is internally inconsistent: it says "Pi 4 with Cortex-A76", but that core belongs to the Pi 5 ([ISBI 2026](https://informatika.stei.itb.ac.id/~rinaldi.munir/Penelitian/Makalah-ISBI-2026.pdf)). Face2PPG-RT shows that the face pipeline, not the rPPG maths, dominates cost: the rPPG step took 0.46 ms of a 9.7 ms frame on a laptop ([arXiv 2508.18787](https://arxiv.org/html/2508.18787v1)).

The recommended CPU-first stack:

| Stage | Choice | Rate | Est. Pi 5 cost (to measure) |
|---|---|---|---|
| Capture | Pi 5 8 GB with active cooler; picamera2; 640×480 main + 320×240 lores; AE/AWB/AF locked after a 2 s settle; `SensorTimestamp` | Every frame | 1–3 ms |
| Detect | YuNet (OpenCV DNN) on lores | Every 3–5 frames | 2–5 ms (Pi 4B: 6.2 ms at 160×120) |
| Track + re-ID | ByteTrack-style; MobileFaceNet/SFace on events | Per frame; about 1 Hz | <1 ms; 12–30 ms per call |
| Landmarks | MediaPipe Face Landmarker on the target crop only | 10–30 Hz | 15–40 ms |
| rPPG | POS/CHROM per patch; deep branch via ONNX Runtime, batched per 1 s chunk | Per frame / per chunk | <1 ms; 15–30 ms per frame-equivalent |
| HR/SpO2/BP + SQI | Welch, Viterbi, RoR, CNN head, GBM | 1 Hz | <5 ms |

**Hailo option.** Keep a Hailo-8L AI Kit as an upgrade path for detection plus a DeepPhys-style 2D network. Verify that the NPU is actually used: one community user found their face pipeline silently running on the CPU ([Hailo community](https://community.hailo.ai/t/a-comprehensive-guide-to-building-a-face-recognition-system/8803/35)).

**Camera.** Run capture, inference and UI in separate processes with latest-frame queues. Lock camera controls via libcamera (`AeEnable`, `AwbEnable`, `ExposureTime`, `AnalogueGain`, `ColourGains`, `FrameDurationLimits`) ([libcamera controls](https://docs.libcamera.org/master/public-api/namespacelibcamera_1_1controls.html)).

**The MATLAB route.** MathWorks officially supports Raspberry Pi 5 ([MathWorks](https://www.mathworks.com/hardware-support/raspberry-pi.html)). However, the ARM Compute Library deep-learning target is being deprecated, and no 3D-conv layer appears in the code-generation support list ([MathWorks codegen layers](https://www.mathworks.com/help/coder/ug/networks-and-layers-supported-for-c-code-generation.html)). The pragmatic split:
- Write `process_round1.m` and all constraint simulations in MATLAB, which satisfies the brief and the sponsor.
- Generate C code for the signal-processing core (band-pass, POS, SQI, Viterbi) with MATLAB Coder.
- Run the live Pi pipeline in Python with ONNX Runtime and call that generated core.

For the "model and time complexity" criterion (one-sixth of the final score), report:
- parameters and FLOPs/MACs per network, and FLOPs per output second for the rPPG model;
- per-stage latency (mean, p50, p95 over ≥1,000 frames);
- glass-to-HR latency;
- sustained fps over 10 minutes with thermal state;
- CPU% and RAM;
- wall power from a USB-C meter;
- the FP32-vs-INT8 accuracy delta;
- bystander-rejection metrics.

ME-rPPG's params/memory/latency/FLOPs table is a ready template ([arXiv 2504.01774](https://arxiv.org/html/2504.01774v1)).

## Thirty weeks from registration to the ICASSP demo

### Roles and phase plan

The plan assumes five or six undergraduates plus a faculty supervisor. The supervisor is needed for ethics approval and to sign dataset agreements: MMPD, SUMS and iBVP accept only faculty or supervisor signatures, so allow 1–4 weeks for those. Roles are split so that no critical path runs through one person.

| Role | Owns | Round 1 focus | Final focus |
|---|---|---|---|
| Signal-processing lead | POS/CHROM, patching, SQI, Viterbi tracking, SpO2 RoR | Classical core + ablations | SpO2 RoR, BP features |
| Deep-learning lead | Deep branch, augmentation (codec/fps/low light/mask/skin tone), fusion | Baseline deep model on MCD-rPPG | SpO2 CNN, BP regressor |
| Vision/edge lead | Detection, landmarks, tracker, re-ID, Pi port, latency | C3 multi-face, ROI geometry | Real-time Pi app, live demo |
| Data & ethics lead | IRB, consent, collection protocol, licences, degradation harness | Public data + C4/C5/C7 simulations | Own data, BP/SpO2 sessions |
| Evaluation & documentation lead | Metrics, bootstrap CIs, Bland–Altman, `.mat` packaging, MATLAB scripts, IEEE report | `process_round1.m`, report | `benchmark_comparison.csv`, report, slides |
| (Optional 6th) Demo & UI | Display, confidence visualisation, rehearsal logistics | — | Presentation, backup video |

| Dates | Phase | Key deliverables / gates |
|---|---|---|
| 20–25 Oct | Launch | Register by 25 Oct. Submit IRB application. Request the organiser data agreement. Email organisers about NC-data training, calibration/demographic inputs and the test-video format. Supervisor signs MMPD/SUMS/iBVP/CHILL EULAs; email VitalVideos about a commercial licence. Download MCD-rPPG, UBFC-rPPG, MSLR. Order Pi 5 8 GB + cooler, Camera Module 3, C920-class webcam, CMS50D+/E oximeter, Omron cuff, lux meter, LED panel, colour checker. Activate MATLAB licences; set up repo. |
| 26 Oct–8 Nov | Baseline | In-house POS/CHROM in MATLAB and Python, YuNet + landmarks, patch ROIs, Welch HR. Mean-predictor and POS baselines on UBFC, MCD-rPPG and the organiser data (if received). Degradation harness (brief's MATLAB functions + physics low-light + ffmpeg codec). `.mat` writer. **Gate: POS within about 1 BPM of published UBFC numbers.** |
| 9–22 Nov | Robust core + constraints | Timestamp resampling, SNR patch fusion, Viterbi tracker, SQI. C3 tracker + embedding enrolment on composites. C4/C5 sweeps. C7 mask overlays + above-mask ROIs. C2 ITA tables. Deep baseline (FactorizePhys/EfficientPhys) trained on MCD-rPPG for comparison. Ablation table. |
| 23–30 Nov | Round 1 freeze | IEEE-format report (innovation, feasibility, clarity, deviation from reference). `process_round1.m`; `Round1_Condition_Constraint.mat` files; `target_video.mp4`, CSVs. **Submit by 30 Nov.** |
| 1–21 Dec | Data + SpO2 | Ethics approval expected (if delayed, see risk register). Pilot, then collect 30–40 subjects including breath-holds, 50/20/5 lux, masks, multi-person, 60+, Fitzpatrick V–VI. SpO2 RoR + DC/AC CNN on MCD-rPPG. Deep HR branch retrained with codec/fps/low-light/mask/skin-tone augmentation. |
| 22 Dec–4 Jan | Holiday buffer | Light-duty: data cleaning, sync-lag estimation, label QA. Pi latency benchmarks of every stage. |
| 5–16 Jan | Round 2 freeze | SQI-gated HR fusion final. SpO2 with constant baseline, per-range errors, desaturation AUROC. C1 (validated against MSLR) and C6 added. Report + `.mat` + code. **Submit by 16 Jan.** |
| 17 Jan–9 Feb | Hedge while finalists are selected | BP baselines on MCD-rPPG (mean, demographics, HR-only, morphology + ridge/LightGBM), subject-disjoint, MASE. Live HR + SpO2 on the Pi at ≥15 fps. Complexity table draft. |
| 10 Feb–9 Mar | BP + final data | (If selected) BP sessions with cuff under IRB; phase-lag features; optional calibrated mode. Choose the two live-demo constraints (recommended C3 + C7, with C5 configured). |
| 10 Mar–6 Apr | Integration | All three vitals on the Pi with confidence display. Thermal soak tests. Hailo upgrade only if CPU misses the target. `benchmark_comparison.csv` against POS/CHROM/PhysNet and mean predictors. |
| 7 Apr–4 May | Final evaluation + writing | Validation/test runs. Final report with full ablations and standards tables. Slides. Three timed dry runs before an outside audience. |
| 5–15 May | Freeze + travel | Spare Pi/camera/oximeter, recorded backup demo, portable LED panel and lux meter. Final rehearsal. |

### Round 1 accuracy and live-demo choices

**Round 1 is judged partly on "least deviation from medical equipment."** If the organiser's hospital dataset arrives in time, headline the numbers on that data, since it most resembles what the judges will compare against.

**For the live final demo**, pick constraints the system controls rather than ones the venue imposes:
- **C3 (multi-face)** shows off target locking with a teammate as bystander.
- **C7 (masks)** is visually obvious and evidence-backed.
- **C5** can be enforced by configuring 10 fps and a 500 kbps encode on the device.

Low light (C1) is the riskiest to demo live, because venue lighting is unknown and the ≤50 lux evidence is thin. Bring a lux meter regardless, so every demo reading is logged with its conditions.

### Risk register

| Risk | Likelihood / impact | Mitigation | Trigger to act |
|---|---|---|---|
| IRB approval slips past December | High / High | Round 1 fully on public + organiser data; submit IRB this week; prepare a minimal-risk "self-recording by team members" amendment | No approval by 1 Dec |
| Organiser dataset agreement delayed or restrictive | Medium / High | Cross-dataset results on MCD-rPPG/UBFC; flag in report | No data by 10 Nov |
| Licence conflict with MathWorks clause | Medium / High | Train only on clean tiers; written query to organisers; document provenance per model | Any organiser reply restricting NC data |
| Faculty-signature delays for MMPD/SUMS/iBVP | Medium / Low | Not on the critical path (benchmarks only) | — |
| Deep branch generalises poorly or fails at 10 fps / low bitrate | High / Medium | Classical core always on; SQI fusion down-weights deep output; augmentation | Deep worse than POS on cross-dataset |
| Pi misses real-time target | Medium / High | Detect every 5 frames; landmarks at 10 Hz; classical-only mode; Hailo-8L upgrade for detection | <15 fps sustained by 1 Mar |
| SpO2 no better than constant | High / Medium | Frame as trend/desaturation detection; report skill score honestly; breath-hold data | Skill ≤0 on own data |
| BP no better than mean or demographics | High / Medium | Exploratory framing; categorical output; MASE; demographics baseline shown | MASE ≥95% |
| ≤50 lux performance on Pi sensor much worse than simulated | Medium / Medium | Long exposure at 10 fps; analogue gain; abstain with confidence; report honestly | Own 20-lux MAE >10 BPM |
| Organiser test videos heavily compressed | Medium / High | Codec-augmented training; SAC compression SQI; CHROM/POS selection by SQI | First look at organiser data |
| Reference sync errors corrupt metrics | Medium / Medium | Optical sync event; per-recording cross-correlation lag; same correction in all conditions | Lag variance >0.5 s |
| Another team from the same institution competes | Unknown / High | Only one finalist per institution: coordinate internally now | Before 25 Oct |
| Team availability over exams and holidays | High / Medium | Front-load Round 2 work into early December; holiday buffer | — |
| Live-demo hardware or lighting failure | Medium / High | Duplicate hardware; recorded backup run; own LED panel | — |
| Travel and visa for ICASSP | Unknown / High | Start visa paperwork immediately after the 9 Feb announcement | 9 Feb |

## Conclusion

The decisive insight from this evidence is that **the challenge's constraints shift the balance toward signal processing and away from large deep models**, and the jury's expertise rewards exactly that. Low resolution, 10 fps and frame drops are where training-free POS stays flat while celebrated deep models collapse. The remaining failure modes are compression, low light and skin tone. Those are best handled by measuring signal quality and abstaining honestly rather than by adding parameters. A team that ships a few hundred lines of well-engineered POS/CHROM, SQI and tracking code, plus a 50 K–2 M parameter deep branch, can credibly claim both accuracy and the lowest complexity in the field. Its ablations will also be the clearest.

The second insight is strategic. **For SpO2 and BP, scientific honesty is a competitive advantage, not a concession.** Published results in both areas largely rest on narrow label ranges, subject leakage or missing baselines. A finalist that shows a constant predictor, MASE, per-range errors, a demographics-only BP baseline and desaturation-tracking metrics will be evaluated as the team that understood the problem.

The places where the team must generate its own evidence are also where it can publish something new:
- Pi 5 latency;
- ≤50 lux on a small sensor;
- adults 60+;
- vitiligo;
- the combined 10 fps / 500 kbps / 320×240 condition.

Planned well, those measurements double as the innovation section of the final report.
