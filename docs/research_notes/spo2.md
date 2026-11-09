# SpO2 from RGB Facial Video: Methods, Accuracy, Physics Limits, Datasets, Honest Evaluation (FaceVitals SP Cup 2027, Round 2)

## Q1. Methods (ratio-of-ratios, regression/ML, deep models) and reported accuracy *with test SpO2 range*

### Takeaway
Every published RGB-face SpO2 method reports MAE of about 0.85 to 2.5 percentage points (pp). Almost all were tested on SpO2 that stays in about 85/93 to 100%, and those numbers are close to what a constant predictor scores. Where results are split by range, error grows sharply below 95% (about 3 pp MAE). The most defensible pipeline is a multi-ROI RGB DC/AC representation (ratio-of-ratios features over R/B and R/G, plus a small CNN on DC/AC spatio-temporal maps). It should be reported against a constant-mean baseline and judged on how well it tracks desaturation, not on absolute accuracy.

### Cited Findings
**Classic ratio-of-ratios (RoR)**
- RoR is the pulse-oximeter principle: SpO2 ≈ A·RoR + B, with RoR = (AC_λ1/DC_λ1)/(AC_λ2/DC_λ2). Contact oximeters use 660 nm (red) and 940 nm (IR), where Hb and HbO2 absorption is roughly equal at 940 nm. RGB-camera methods substitute the blue channel for IR, and A and B are fitted on training data. Earlier RGB methods took DC as the 10 s mean and AC as the std or the mean peak-to-trough. — [Akamatsu et al., ICASSP 2023, arXiv 2212.07116](https://arxiv.org/abs/2212.07116)
- Multi-channel RoR (Tian et al., IEEE JSTSP 2022, hand video, smartphone) uses all three RGB channels with HR-adaptive bandpass filters. It reports 1.26% MAE, which the authors say is 25% better than two-channel RoR. — [Tian et al. (NSF PAR)](https://par.nsf.gov/servlets/purl/10358187); [arXiv 2107.08528](https://arxiv.org/pdf/2107.08528)
- Samsung (Zhu et al., EMBC 2023): facial video RoR with Red/Blue and Red/Green pairs, consumer RGB camera with **auto-tuning disabled**. Desaturation was induced by manually controlling inhaled air through a mouth tube with nose clip, down to **81%**, against a multi-wavelength SpO2 reference, with leave-one-subject-out evaluation. Under warm-colour, normal-brightness light: **RMSE 1.93%, PCC 0.97**. They also tested neutral, cool and dim light, but the abstract page gives no numbers for those. — [Samsung Research America publication page](https://sra.samsung.com/publications/estimating-spo2-with-deep-oxygen-desaturations-from-facial-video-under-various-lighting-conditions-a-feasibility-study/)

**Deep models on DC/AC spatio-temporal maps (Akamatsu et al., NEC, ICASSP 2023)**
- Self-collected data: 50 subjects aged 20s–50s, 3 min each, Logitech C920n at 30 fps, OxyTrue A oximeter (samples every 4 s, interpolated to 1 Hz). Desaturation came from **breath-holding: about 30 s hold and 30 s rest, ×3 cycles**, stopping when uncomfortable, under ethics approval (NEC 21-017). SpO2 labels were scaled over an assumed 85–100% range, and the figure y-axes span about 92–100%. Five-fold subject-independent CV (40 train / 10 test). — [arXiv 2212.07116](https://arxiv.org/pdf/2212.07116)
- Method: 224 ROIs (16×14 blocks below the eyes) × 300 frames × RGB form an STMap. A low-pass filter below 0.3 Hz gives DC and a 0.75–2.5 Hz bandpass gives AC. Two ImageNet-pretrained ResNet18s are fused with MMTM, and the end-to-end variant learns DC/AC with depthwise convolutions. Loss is MSE plus negative Pearson. — [arXiv 2212.07116](https://arxiv.org/pdf/2212.07116)
- Table 1 (MAE / RMSE / r): RoR 1.37/1.66/0.173; LR 1.25/1.51/0.193; SVR 1.25/1.50/0.143; MultiPhys 1.30/1.58/0.145; CNN 1.23/1.47/0.455; CNN-Filter 1.17/1.40/0.474; **CNN-EndToEnd 1.18/1.40/0.496**. Every method sits within about 0.2 pp MAE of the others, and the mean per-subject correlation is below 0.5. In some subjects the ground truth did not drop during breath-holds. — [arXiv 2212.07116](https://arxiv.org/pdf/2212.07116)

**Deep models on VIPL-HR (Cheng et al., Bioengineering 2024, PanopticAI/HKUST)**
- VIPL-HR: 107 subjects, mostly Asian, 2,378 RGB videos. STMap with EfficientNet-B3 (RGB): **MAE 1.274, RMSE 1.710**. Hu et al. (PURE pretrain, then VIPL-HR fine-tune): MAE 1.000, RMSE 1.430. RoR baselines: MAE 3.334 (Casalino) and 1.838 (Bal). — [Cheng et al., PMC10968547](https://pmc.ncbi.nlm.nih.gov/articles/PMC10968547)
- **Split by range**: EfficientNet-B3 has MAE **0.978 for SpO2 ≥ 95%** but **3.077 (RMSE 3.563) for SpO2 < 95%**. RoR [21] scores 1.690 and 4.482 on the same split. The authors note the training data "contains fewer low SpO2 values", that the model may be biased toward Fitzpatrick III–IV, and that trends may be more informative than single readings. — [PMC10968547](https://pmc.ncbi.nlm.nih.gov/articles/PMC10968547)

**CL-SPO2Net (Peng et al., Bioengineering 2024)**
- Contrastive pre-training on unlabeled FaceForensics++, then a spatiotemporal-attention network. Labelled data is MVPD: **13 volunteers, SpO2 range 93–100%**, leave-one-participant-out. Face ROI: MAE **0.85** (stable), **1.13** (lighting fluctuation), **1.20** (face rotation), with ρ 0.76–0.80. The authors acknowledge the 93–100% range must be extended. Kim et al. beats it in the stable condition (MAE 0.54, ρ 0.92). — [PMC10885926](https://pmc.ncbi.nlm.nih.gov/articles/PMC10885926/)

**Mathew et al. (hand video, smartphone; arXiv 2107.05087, v3 2025)**
- 14 participants (Fitzpatrick II–V, no I or VI), iPhone 7 Plus at 30 fps, CONTEC CMS50E reference. Breath-hold protocol: 30–40 s normal breathing, full exhalation, 30–40 s hold, ×3; SpO2 "can drop below 90%". Leave-one-participant-out, best model: **MAE 1.97, RMSE 2.32, r 0.46**. Classic RoR (Scully) reaches comparable MAE (2.08) but **r ≈ 0.01–0.05**, i.e. it gets low MAE without tracking the trend. — [arXiv 2107.05087v3](https://arxiv.org/html/2107.05087v3)

**Tang et al. 2025, "Camera Measurement of Blood Oxygen Saturation" (Tsinghua / UW / TUAT)**
- Real hypoxia was induced two ways: THU used a mask with gas pressure reduced to 0.6 atm (71 people, 110 videos, Sony α7R III DSLR); TUAT used FiO2 11% for 4 min (15 people, DFK23U618 at 640×480, 15 Hz). VC2S is a single-frame forehead CNN that also takes **colour-checker** data and fits a **per-subject linear calibration on the first n frames**. LOSO RMSE: THU 1.69 (baseline 1.71); TUAT V1 3.24 (baseline 5.62; MAE 2.53 vs 4.68); cross-dataset TUAT V1 4.11. **No ARMS reported.** The authors call reliance on calibration a limitation. — [arXiv 2503.01699](https://arxiv.org/html/2503.01699v1)

**Other**
- CCSpO2Net reportedly reaches MAE 1.79 on an ICU facial-video dataset (IEEE 2024). This comes from a search summary only and is not verified. — [search listing via unpaywall](https://unpaywall.org/10.1109%2FTIM.2024.3375409)
- Summit Vitals / SUMS (high altitude): face+finger fusion with HR–SpO2 joint training. Reports SpO2 MAE of about 2.5% and 7.6–10.6% relative MAE reduction from fusion. — [arXiv 2409.19223](https://arxiv.org/pdf/2409.19223)
- For contrast, smartphone *contact* (finger-on-camera+flash) oximetry by Hoffman et al. (npj Digit Med 2022) used a varied-FiO2 protocol for 70–100%. It got MAE **5.00%**, and sensitivity 81% / specificity 79% for SpO2 < 90%, with 6 subjects. **Over a clinically meaningful range, error is much higher** than the ~1 pp figures in the narrow-range rPPG literature. — [UW News](https://washington.edu/news/2022/09/19/smartphone-camera-flash-could-help-people-measure-blood-oxygen-levels-home); [DOAJ record](https://doaj.org/article/2637ce72ffd3438193df03560ee8fbeb)

### Inferences
- The "SOTA" MAE values (0.85–1.3 pp) are not comparable across papers and are mostly explained by narrow label ranges. The two papers that split by range (Cheng et al.) or test 70–100% (Hoffman et al.) show about 3–5 pp error where it matters. Realistic target for a student team on a breath-hold-augmented test set: MAE about 1–1.5 pp overall, about 2.5–4 pp for SpO2 < 94%, and within-subject correlation of 0.4–0.7 during breath-holds.
- Recommended edge pipeline: face landmarks, then multi-ROI (forehead, both cheeks, nose) RGB means, then a DC/AC STMap. Predict with (a) a calibrated RoR on R/B and R/G as an interpretable baseline and (b) a small DC/AC two-stream CNN in the style of Akamatsu, using MobileNet or a 1D CNN in place of ResNet18 for edge use. Share the rPPG/HR front-end so SpO2 benefits from HR-quality gating (Summit Vitals reports joint HR–SpO2 training helps). Output a signal-quality index and abstain on low-quality windows.
- Given the narrow ranges in available data, a "Δ-SpO2 / desaturation detector" framing (trend plus a <94% alarm) is scientifically more honest than claiming absolute accuracy.

### Gaps
- Could not verify numbers or test ranges for Casalino et al. (2020, ISCC), Hu et al. (2023), or Kim et al. beyond the secondary tables above. CCSpO2Net is unverified.
- I found no 2025–2026 RGB-face SpO2 paper that reports ARMS over 70–100% with diverse skin tones. To my knowledge none exists, but my search was not exhaustive.
- Full Samsung EMBC 2023 results for cool, neutral and dim lighting were not accessible (abstract page only).

## Q2. Physics: why RGB SpO2 is hard; illumination, low light, skin tone

### Takeaway
RGB filters are broad and overlap, there is no IR channel, and the AC component is tiny and depth-dependent. The red/green or red/blue ratio therefore depends on illumination spectrum, camera spectral response, white balance, posture/geometry, temperature and melanin, not only on SpO2. RGB can track *trends* but needs calibration (per camera, light and possibly subject) to give absolute values. Even contact pulse oximeters show clinically significant skin-tone bias.

### Cited Findings
- Broadband, overlapping R/G/B passbands mean a two-channel RoR from RGB "yields a trend rather than a calibrated SpO2 value", though it can track changes over time (patent family on video vital-sign monitoring). — [US 10667723](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10667723)
- Broadband RGB channels are less optically selective than the narrowband LEDs in clinical oximeters (Tian et al.). — [arXiv 2107.08528](https://arxiv.org/pdf/2107.08528)
- Red/green visible-light camera oximetry (46 adults; 580/675/840 nm; normoxia, hypoxia and cooling) raises calibratability concerns because visible wavelengths penetrate tissue to different depths. A related study reports visible-light error of **3.0% vs 1.7%** for red/NIR, with red/green errors growing at lower temperature or slightly different skin sites. — [J Clin Monit Comput 2021 (Springer Medicine listing)](https://www.springermedicine.com/pulse-oximetry-based-on-photoplethysmography-imaging-with-red-an/21654008); [doi 10.1007/s10877-019-00449-y](https://www.doi.org/10.1007/s10877-019-00449-y)
- Posture changes shift the red/green ratio. Uncorrected SpO2 errors are estimated above 3% for visible light vs below 1% for red/IR, so visible-light calibrations require a fixed measurement position. — [TU/e, Physiol Meas 2019](https://www.es.ele.tue.nl/~sander/publications/pmea19-posture.php)
- Philips forehead camera oximetry (red/IR, Anesthesia & Analgesia 2017, 41 adults) showed SpO2 can be calibrated across subjects, but **15 of 41 subjects were discarded** for motion or low pulsatile strength. Remote light travels a shorter tissue path than in contact probes, which is why calibratability was doubted. — [MD+DI](https://www.mddionline.com/rd/philips-pulse-ox-without-skin-contact-is-possible)
- Illumination colour temperature: Samsung tested warm, neutral and cool colour temperature plus a dim condition, with auto-tuning disabled. Only warm/normal numbers are public (RMSE 1.93, PCC 0.97). — [SRA](https://sra.samsung.com/publications/estimating-spo2-with-deep-oxygen-desaturations-from-facial-video-under-various-lighting-conditions-a-feasibility-study/)
- Lighting fluctuations raise CL-SPO2Net MAE from 0.85 to 1.13 and roughly double the MAE of several RoR-style baselines (e.g., Kim 0.54 → 1.99; Shao 1.05 → 2.35). — [PMC10885926](https://pmc.ncbi.nlm.nih.gov/articles/PMC10885926/)
- Tang et al. feed a **colour-checker** into the network to normalise illumination and camera response. — [arXiv 2503.01699](https://arxiv.org/html/2503.01699v1)
- Skin tone and rPPG: higher melanin absorbs more light and weakens the camera PPG signal. The UCLA camera+radar fusion (SIGGRAPH/TOG 2022) was designed to reduce this bias. — [UCLA Samueli](https://samueli.ucla.edu/remote-heart-rate-sensors-can-be-biased-against-darker-skin/)
- An rPPG BP screening app in a Nigerian darker-skinned field cohort (BMJ Open 2026) had "inadequate performance" and critically low sensitivity. — [BMJ Open e119311](https://bmjopen.bmj.com/content/16/6/e119311)
- Tang et al. report THU MAE by skin tone (1–2: 1.22; 3–4: 1.49; 5–6: 1.13). This shows no monotonic degradation, but with per-subject calibration and few type 5–6 videos (23). — [arXiv 2503.01699](https://arxiv.org/html/2503.01699v1)
- Mathew et al. found credible skin-type effects with small n (14), and in one leave-one-out setting darker skin performed better. — [arXiv 2107.05087v3](https://arxiv.org/html/2107.05087v3)
- **Contact oximeter bias (Sjoding et al., NEJM 2020)**: among readings of 92–96%, arterial SaO2 < 88% (occult hypoxemia) occurred in **11.7% of Black vs 3.6% of White** patients' measurements (Michigan, 10,789 pairs). The multicenter cohort showed **17% vs 6.2%** (37,308 pairs). — [AHRQ PSNet summary](https://psnet.ahrq.gov/issue/racial-bias-pulse-oximetry-measurement); [Healthcare IT News](https://www.healthcareitnews.com/news/pulse-oximeters-may-be-less-accurate-black-patients)
- In February 2021 FDA said accuracy can be affected by "poor circulation, skin pigmentation, skin thickness, skin temperature, current tobacco use, and use of fingernail polish". — [FDA draft guidance, Jan 2025](https://www.fda.gov/media/184896/download)

### Inferences
- **Implication for the challenge's ground truth:** the reference is a clinical finger pulse oximeter, which is itself biased (overestimating SaO2 in darker skin) and has about ±2–3% ARMS vs arterial blood. A video model trained on SpO2 labels inherits that bias, and its error vs the oximeter has a floor near the oximeter's own resolution and noise (integer readings, 1 Hz or 4 s updates, averaging lag).
- From general optics, which the sources above support only qualitatively: blue light penetrates skin shallowly and has very low pulsatile SNR on the face, especially for darker skin and in low light. That makes R/B RoR especially fragile, so R/G is often the more practical pair. White balance gains rescale R and B relative to G, which changes the DC ratio directly. Auto-exposure and auto-gain inject step changes into DC. Compression (H.264) suppresses the sub-1% AC variations. All camera auto modes should be locked, and each window should be normalised per channel (AC/DC per channel cancels a static gain but not a time-varying one).
- Low light: shot noise dominates the AC term and the R/B ratio becomes noise-limited. Expect SpO2 to fail before HR does, and gate SpO2 output on HR-signal SNR.

### Gaps
- No primary quantitative source was found on how AWB or auto-exposure on/off, or H.264 CRF level, specifically changes RGB SpO2 error. This is a good ablation for the team to run themselves.
- Could not access the full Verkruysse 2017 paper for its ARMS and desaturation range.

## Q3. Validity: baselines, metrics, regulatory standards, desaturation protocols, ethics

### Takeaway
A constant predictor already achieves about 1 pp MAE on typical datasets: MCD-rPPG's 600-subject baseline is literally tied with its model at 0.98. The team must report a train-mean baseline and per-range (≥95 / 90–95 / <90) errors. They should also report within-subject correlation and desaturation-event detection during breath-holds, plus Bland–Altman and bias by Monk skin-tone group, all in FDA/ISO terminology. They must state plainly that FDA-grade ARMS < 3% over 70–100% against arterial SaO2 cannot be tested without a clinical hypoxia lab.

### Cited Findings
- **MCD-rPPG (600 subjects)**: SpO2 distribution mean **98.01, std 1.29, min 86, max 99**. The optimal constant baseline gives **MAE 0.98**, and the trained model also gives **0.98**, i.e. no improvement. — [Egorov et al., arXiv 2508.17924](https://arxiv.org/html/2508.17924v1)
- On NCTUBO, only **1.21%** of instances had SpO2 < 90% despite breath-holding ("data imbalance"). — [Summit Vitals, arXiv 2409.19223](https://arxiv.org/pdf/2409.19223)
- RoR can show low MAE with near-zero correlation (Mathew: MAE 2.08, r ≈ 0.01–0.05), so MAE alone is uninformative. — [arXiv 2107.05087v3](https://arxiv.org/html/2107.05087v3)
- **FDA draft guidance (Jan 2025)**, "Pulse Oximeters for Medical Purposes", requires a controlled desaturation lab study per **ISO 80601-2-61 Annex EE** against **CO-oximetry SaO2**. Its terms:
  - **≥150 healthy participants**, with **≥25% in each Monk Skin Tone group (1–4, 5–7, 8–10)** and ≥50% of MST 8–10 having forehead ITA ≤ −50°.
  - **≥3,000 paired observations**, 20–24 pairs per participant (17–30), and **≥30% of pairs in each SaO2 decile 70–80, 80–90, 90–100%**.
  - Success criteria: **ARMS < 3%** (upper bound of the two-sided 95% CI, over 70–100%). Largest pairwise **MST-group bias difference < 3.5% for SaO2 70–85% and < 1.5% for 85–100%**. Same thresholds per 100° change in ITA.
  - Required plots: Bland–Altman, modified Bland–Altman, QQ and inverse-prediction plots, colour-coded by MST, with 95% limits of agreement.
  - Reporting: ARMS by decile, and mean ± SD of error.
  - ARMS is defined as the RMS of (SpO2 − SaO2) over all paired measures and participants. — [FDA draft guidance PDF](https://www.fda.gov/media/184896/download)
- The draft does not apply to general-wellness or consumer devices. — [RAPS](https://www.raps.org/resource/pulse-oximeters-fda-drafts-guidance-to-address-per.html)
- Desaturation methods in the literature:
  - **Breath-hold**, 30 s hold and 30 s rest ×3 (Akamatsu, ethics-approved): face video SpO2 mostly ≥ 92%. — [arXiv 2212.07116](https://arxiv.org/pdf/2212.07116)
  - **Exhale-then-hold** 30–40 s ×3 (Mathew): occasionally < 90%. — [arXiv 2107.05087v3](https://arxiv.org/html/2107.05087v3)
  - **Controlled inhaled air** via mouth tube and nose clip, to 81% (Samsung). — [SRA](https://sra.samsung.com/publications/estimating-spo2-with-deep-oxygen-desaturations-from-facial-video-under-various-lighting-conditions-a-feasibility-study/)
  - **FiO2 11% for 4 min**, or a **mask at 0.6 atm** (Tang). — [arXiv 2503.01699](https://arxiv.org/html/2503.01699v1)
  - **Varied FiO2, 70–100%** (Hoffman). — [UW News](https://washington.edu/news/2022/09/19/smartphone-camera-flash-could-help-people-measure-blood-oxygen-levels-home)
  - **Real altitude**, 2,274–4,898 m, with exercise and O2 inhalation (SUMS). — [arXiv 2409.19223](https://arxiv.org/pdf/2409.19223)
  - A 2026 breath-hold dataset (exhale fully, then hold) produced reference values concentrated in **91–95%**. — [search summary of 2026 paper; primary not fetched]
- Breath-hold physiology: trained apnea divers reached an average end-apnea SpO2 of **79% ± 12** after about 329 s maximal dry holds, and a single maximal hold caused acute endothelial activation. Untrained people holding about 30–40 s reach much smaller drops. — [QxMD summary of 2017 study](https://read.qxmd.com/read/28464406/sustained-apnea-induces-endothelial-activation)

### Inferences
- **What to report (judge-proof checklist):**
  1. A **constant train-mean/median baseline** and a "last-known-value" baseline. Report Δ vs baseline (skill score = 1 − MAE_model/MAE_const).
  2. MAE, RMSE (= ARMS vs pulse oximeter; call it "ARMS vs reference pulse oximeter", **not** ARMS vs SaO2) and bias ± 1.96 SD (Bland–Altman), **stratified by SpO2 bin** (≥95, 90–95, <90) and by **Monk/Fitzpatrick group**.
  3. **Within-subject** Pearson/concordance correlation, plus ΔSpO2 tracking (nadir error, nadir timing lag) on breath-hold segments.
  4. Desaturation detection: AUROC and sensitivity/specificity for a ≥3–4 pp drop or for SpO2 < 94%.
  5. Error vs illumination (lux, CCT) and vs rPPG SNR, plus coverage (fraction of windows not abstained).
  6. Subject-disjoint splits only, and cross-dataset tests.
- Account for oximeter latency: finger oximeters lag the true change and face perfusion responds earlier than finger, so estimate lag via cross-correlation and report results with and without lag compensation.
- **Ethically feasible for a student team:** short, seated, voluntary breath-holds (e.g., after normal exhalation, 20–40 s, ×3 with ≥60 s recovery). Stop on discomfort or if SpO2 < 88–90%. Screen out cardiopulmonary disease, pregnancy, epilepsy or syncope history. Obtain university ethics or IRB approval and informed consent, and have a supervisor present. Hypoxic-gas or altitude-chamber protocols need clinical supervision and are out of scope. Expect nadirs of about 90–95% in most people, so the team can validate trend tracking over roughly 88–100% but **cannot** validate 70–88%.
- Be explicit to judges: "This is not a medical device. We cannot test ARMS against SaO2 over 70–100%, so our claims are limited to agreement with a reference finger oximeter in the 88–100% range and to tracking desaturation trends."

### Gaps
- ISO 80601-2-61 full text (paywalled) was not read. ARMS details above come via the FDA draft.
- I found no source on the status of the FDA draft guidance after January 2025 (whether it was finalised).
- No authoritative published safety protocol for student breath-hold studies was found. The protocol above is inferred from the ethics-approved Akamatsu and Mathew protocols.

## Q4. Datasets with facial RGB video + SpO2 ground truth (and SpO2 range)

### Takeaway
Large public datasets (MCD-rPPG, VIPL-HR, PURE) have SpO2 almost entirely at 95–100%, which is good for pre-training and for showing the constant-baseline trap. Datasets with real desaturation are small: SUMS (altitude), NCTUBO (breath-hold, open, but under 1.2% of samples below 90%), DDMP-FV (breath-hold, 20 subjects), plus non-public sets (THU/TUAT hypoxia, NEC breath-hold). The team should collect its own breath-hold set with Monk skin-tone annotation.

### Cited Findings
- **MCD-rPPG** (Sber, 2025): 600 subjects × 2 states (rest and after 15 squats) × 3 cameras (Samsung A3s phone, Sony FDR-AX43A, Defender webcam), 640×480 at 24/30 fps, 3,600 videos. SpO2 from a Beurer PO 60 finger oximeter: **mean 98.01 ± 1.29, range 86–99**. CC BY 4.0, Hugging Face. Lighting is not specified. — [arXiv 2508.17924](https://arxiv.org/html/2508.17924v1); [HF dataset](https://huggingface.co/datasets/Bgeorge/mcd_rppg)
- **VIPL-HR**: 107 subjects (mostly Asian), 2,378 RGB + 752 NIR videos, 9 scenarios, 4 devices, has SpO2. The numeric range was not given in the text, and the "abnormal" (<95%) subset is small. — [PMC10968547](https://pmc.ncbi.nlm.nih.gov/articles/PMC10968547)
- **PURE**: listed as having PPG/SpO2 (finger oximeter). It is a low-altitude, resting dataset. — [Summit Vitals Table I, arXiv 2409.19223](https://arxiv.org/pdf/2409.19223)
- **NCTUBO**: 46 subjects at sea level, head-supported sitting, breath-holding. It is open-source, but **only 1.21% of instances < 90%**. — [arXiv 2409.19223](https://arxiv.org/pdf/2409.19223)
- **SSL** (Taiwan, 3,150 m): 14 subjects, static and dynamic SpO2, lying and sitting. Not open. — [arXiv 2409.19223](https://arxiv.org/pdf/2409.19223)
- **SUMS / Summit Vitals**: 10 residents of the Qinghai Plateau (2,274–4,898 m), 80 face+finger videos (Logitech C922 and others). States: rest, post-stair-exercise, and O2 inhalation. CMS50E+ oximeter. Most subjects dropped **below 92%** after exercise, and the histogram spans about 80–100%. Open (GitHub). — [arXiv 2409.19223](https://arxiv.org/pdf/2409.19223)
- **DDMP-FV** (IEEE DataPort): 20 subjects in resting, breath-holding and post-locomotion states, 2 × 30 s each (120 sets). CONTEC CMS50DL SpO2/HR, cuff BP and manual RR. The SpO2 range is not stated. — [IEEE DataPort](https://ieee-dataport.org/documents/ddmp-fv)
- **M³PD** (2025): front camera (face) plus rear camera (fingertip with flash). 13 lab and 47 cardiovascular outpatients, 1280×720 at 30 fps, CMS50E SpO2 at 1 Hz. The SpO2 range is not stated. Available on request (non-commercial), via github.com/Health-HCI-Group/F3Mamba. — [arXiv 2511.02349](https://arxiv.org/html/2511.02349)
- **LADH / FusionVitals** (2025): 21 subjects, 240 synchronized RGB+IR facial videos during personal-care routines at high altitude, with SpO2 ground truth. Open (GitHub McJackTang/FusionVitals). SpO2 range not stated. — [arXiv 2506.09718](https://arxiv.org/abs/2506.09718)
- **MPU-rPPG** (Sci Data 2026): RGB video with SpO2 ground truth, but only a subset is on Figshare. Details could not be fetched (paywall/redirect). — [Nature Sci Data](https://www.nature.com/articles/s41597-026-07310-3)
- **Vital Videos** (Europe / "Worldwide"): 850–893 participants, uncompressed 30 s videos, all Fitzpatrick types, PPG and BP. **No SpO2 ground truth was confirmed.** — [arXiv 2306.11891](https://arxiv.org/pdf/2306.11891); [arXiv 2404.05003](https://arxiv.org/html/2404.05003v1)
- **MVPD** (used by CL-SPO2Net): 13 volunteers, SpO2 93–100%. — [PMC10885926](https://pmc.ncbi.nlm.nih.gov/articles/PMC10885926/)
- **Not public**: NEC 50-subject breath-hold set; THU (71 people, reduced-pressure hypoxia) and TUAT (FiO2 11%); Hoffman finger dataset (70–100%, open but finger-contact, not face). A Brazilian skin-tone-diverse dataset is proposed but appears to be planned work. — [arXiv 2212.07116](https://arxiv.org/pdf/2212.07116); [arXiv 2503.01699](https://arxiv.org/html/2503.01699v1); [SOL-SBC](https://sol.sbc.org.br/index.php/eramiars/article/view/39459)

### Inferences
- Suggested data strategy:
  1. Pre-train the rPPG and DC/AC front-end on MCD-rPPG and VIPL-HR (camera and subject diversity).
  2. Fine-tune and evaluate SpO2 on SUMS, NCTUBO and DDMP-FV (real drops).
  3. Hold out the team's own breath-hold recordings, with Monk annotation and lux/CCT logs, as the honest test set.
  4. Report everything per dataset with the constant baseline.
- The skin-tone diversity of SpO2-labelled data is poor: VIPL-HR and NCTUBO/SUMS are mostly East Asian, and MCD-rPPG is mostly Russian. Any skin-tone fairness claim for SpO2 will rest on the team's own small sample. Present it as a stratified bias table with CIs, not as a claim of equity.

### Gaps
- UBFC-rPPG dataset 1: I could not confirm whether it includes SpO2 labels or their range.
- Exact SpO2 ranges for PURE, VIPL-HR, DDMP-FV, M³PD, LADH and MPU-rPPG were not found in accessible text.
- "VitalVideos-Worldwide" SpO2 availability is unconfirmed.

## Q5. Camera settings that matter

### Takeaway
Lock white balance, exposure, gain/ISO and focus. Record at a fixed frame rate with minimal compression, include a colour reference (gray card or colour checker) where possible, and log illuminance and colour temperature.

### Cited Findings
- Samsung's facial RoR study disabled the camera's "auto-tuning" (auto WB, exposure and similar) to make RoR viable. — [SRA](https://sra.samsung.com/publications/estimating-spo2-with-deep-oxygen-desaturations-from-facial-video-under-various-lighting-conditions-a-feasibility-study/)
- Tang et al. use a colour checker in-frame as a model input, and per-subject calibration from initial frames. — [arXiv 2503.01699](https://arxiv.org/html/2503.01699v1)
- The Vital Videos authors recorded **uncompressed** video, indicating community concern over compression. — [arXiv 2306.11891](https://arxiv.org/pdf/2306.11891)
- Visible-light remote oximetry calibration is posture- and position-dependent, so a fixed measurement geometry is recommended. — [TU/e Physiol Meas 2019](https://www.es.ele.tue.nl/~sander/publications/pmea19-posture.php)

### Inferences
- Practical settings for the team:
  - Manual WB (fixed Kelvin), fixed shutter (e.g., 1/60 s, or 1/50 in 50 Hz mains regions to avoid flicker beating), and fixed ISO/gain set for the darkest test condition.
  - 30 fps, and lossless or high-bitrate recording (or raw YUV) for development.
  - At inference, if the challenge provides compressed video, train with compression augmentation.
  - Per-window per-channel AC/DC normalisation, plus a white-reference or forehead-reference gray-world correction.
  - If auto-modes cannot be controlled (e.g., in the challenge's test videos), detect exposure/WB step changes from frame statistics and mask those windows.

### Gaps
- No quantitative primary study on SpO2 error vs AWB, auto-exposure or compression settings was found. The team should run this ablation as a contribution.
