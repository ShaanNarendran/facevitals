# Blood Pressure (SBP/DBP) Estimation from RGB Facial Video — FaceVitals SP Cup 2027

Research date: 20 Oct 2026. Scope: methods, realistic accuracy, critique literature, standards, datasets, and a feasible plan for a student team (about 3 months, Raspberry Pi target, judged against a calibrated cuff).

Headline for the report writer: no published video-BP method meets the AAMI/ESH/ISO 81060-2 limits (mean error at most 5 mmHg, SD at most 8 mmHg) under subject-independent, calibration-free testing. The best large-cohort results (subject-independent) reach SBP MAE about 11 mmHg against a mean-predictor MAE of about 14 mmHg. DBP gains over the mean predictor are often tiny, and in one recent 600-subject study the model was worse than a constant on DBP. Teams should present BP as an exploratory estimate, always next to a mean/demographic baseline, and report against the standards rather than claim compliance.

---

## Q1. Methods and reported accuracy (PTT, morphology features, deep models; 2019–2026)

### Takeaway
Three families exist: (a) pulse-transit/phase differences between facial regions or face and hand, (b) rPPG waveform-morphology features plus regression, and (c) deep models on rPPG or raw video, often pretrained on contact PPG. Honest subject-independent results on hundreds of subjects cluster at SBP MAE about 11–13 mmHg and DBP MAE about 7–9 mmHg, only 1–3 mmHg better than predicting the training mean. The one striking "meets 5±8" style result (Luo 2019) comes from a normotensive, narrow-range cohort and a company with patents.

### Cited Findings

**Transdermal optical imaging (Luo et al. 2019, Circulation: Cardiovascular Imaging)**
- Luo H, Yang D, Barszczyk A, et al., "Smartphone-Based Blood Pressure Measurement Using Transdermal Optical Imaging Technology," Circ Cardiovasc Imaging 2019;12(8):e008857. The study had 1,328 normotensive adults split 70/15/15 into train/test/validation. Validation bias was 0.39±7.30 mmHg for SBP, −0.20±6.00 for DBP, and 0.52±6.42 for pulse pressure — [TCTMD](https://www.tctmd.com/news/smartphone-selfie-may-reveal-bp-levels-early-results-suggest); [Healio](https://www.healio.com/cardiology/vascular-medicine/news/online/%7Bfac4a70e-fd04-41ec-b2e5-7e1dff760103%7D/novel-technology-with-smartphone-camera-may-measure-bp)
- Weaknesses: the sample included only people with SBP 100–139 mmHg. The authors said future work must test the 5±8 mmHg threshold "on a full range of blood pressures." Two authors hold a patent on the technology. Secondary sources disagree on capture length (2 minutes vs. 30 seconds) — [TCTMD](https://www.tctmd.com/news/smartphone-selfie-may-reveal-bp-levels-early-results-suggest); [Healio](https://cm.healio.com/news/cardiology/20190806/novel-technology-with-smartphone-camera-may-measure-bp)

**Schrumpf et al. 2021 (Sensors 21(18):6022; CVPRW 2021): PPG-pretrained deep models transferred to rPPG**
- The PPG pretraining set was MIMIC-B: 4,000 PPG/ABP record pairs, 3,750 subjects for training and 625 for validation/test, and 1,000,000 training samples — [ar5iv 2104.09313](https://ar5iv.labs.arxiv.org/html/2104.09313)
- The rPPG set came from 50 surgical patients enrolled at Leipzig University Hospital. Only 14 were analysed (12 train / 1 val / 1 test, with leave-one-out fine-tuning) — [ar5iv 2104.09313](https://ar5iv.labs.arxiv.org/html/2104.09313)
- rPPG MAE without personalization: ResNet 13.02/9.81 mmHg (SBP/DBP), AlexNet 15.7/8.27, Slapničar 14.04/8.64, **mean regressor 13.4/8.9**. With personalization (20% of the test subject's data used for training): ResNet 12.51/8.3, mean regressor 13.9/8.5. The deep models were roughly at mean-predictor level — [ar5iv 2104.09313](https://ar5iv.labs.arxiv.org/html/2104.09313)
- The paper found "a strong systematic increase of the prediction error towards less frequent BP values" and stressed subject-aware splits — [arXiv 2104.09313](https://arxiv.org/abs/2104.09313)

**Cheng et al., WACV 2025 (PanopticAI/HKUST): PPG→rPPG transfer learning**
- Pretraining used a MIMIC-III Kaggle subset (12,000 records; the authors note one patient may have multiple records, a possible leakage source) and a PulseDB MIMIC-III subset (calibration-free test split). Fine-tuning used rPPG-LUH, the public 17-subject release of Schrumpf's Leipzig ICU data, with 4-fold subject-disjoint CV — [Cheng et al. WACV 2025 PDF](https://www.ieda.ust.hk/dfaculty/so/pdf/Cheng-et-al-2025-WACV2025.pdf)
- Best result: BiLSTM pretrained on MIMIC-III, fine-tuned with rPPG plus HR, MAE 8.721 mmHg SBP and 8.653 DBP. That is a 36.23% SBP and 28.27% DBP improvement over the mean predictor. It was BHS Grade C for both and did not meet AAMI. rPPG-LUH is skewed toward SBP >130 and DBP >60. AlexNet was worse than the mean predictor for SBP — [Cheng et al. WACV 2025 PDF](https://www.ieda.ust.hk/dfaculty/so/pdf/Cheng-et-al-2025-WACV2025.pdf)
- On the same rPPG-LUH data, Kim et al. reported 6.9/4.43 mmHg using a 20% split "with potential subject overlapping," vs. Schrumpf 13.6/10.3 (leave-two-out). This illustrates how much leakage inflates results — [Cheng et al. WACV 2025 PDF, Table 1](https://www.ieda.ust.hk/dfaculty/so/pdf/Cheng-et-al-2025-WACV2025.pdf)

**U-FaceBP (arXiv 2412.10679; accepted IEEE TIM 2026): the largest subject-independent face-video BP study found**
- FaceBP dataset: 1,094 subjects, 2,812 recording sets, ages 18–91, SBP 88–234 (mean 127.79±18.26), DBP 56–148 (85.40±12.47). Reference: A&D UA-1200BLE cuff. MSPM: 103 subjects, SBP 96–156. Evaluation was 5-fold subject-independent CV — [arXiv 2412.10679v3](https://arxiv.org/html/2412.10679v3)
- MAE (SBP/DBP), FaceBP: **mean regressor 14.29/9.85**; ResNet1D 12.59/8.86; FS-Net 11.72/8.62; **U-FaceBP 11.05/7.90** (r = 0.620/0.597). MSPM: mean regressor 9.10/7.22; U-FaceBP 7.87/6.28 — [arXiv 2412.10679v3](https://arxiv.org/html/2412.10679v3)
- Method: a Bayesian ensemble over rPPG, face-video-derived PPG, and face images. Only BHS Suc10 is reported (Grade C for FaceBP DBP). The authors say it "does not yet reach the validation grades achieved by cuff-based BP monitors." Accuracy was poor for severe hypertension and darker skin tones, and smartphone video was slightly worse than webcam. ME±SD (the AAMI metric) is not reported — [arXiv 2412.10679v3](https://arxiv.org/html/2412.10679v3)

**MCD-rPPG, "Gaze into the Heart" (arXiv 2508.17924)**
- 600 subjects, 3,600 videos (rest and post-exercise, multiple cameras). BP reference was an A&D UA-911BT-C automatic monitor. Ages 23.08±10.90 (range 18–83). SBP 122.45±17.43 (80–202), DBP 73.79±9.25 (50–108) — [arXiv 2508.17924 HTML](https://arxiv.org/html/2508.17924)
- BP MAE: **baseline (optimal constant fitted on training subset) 13.78 SBP / 7.50 DBP vs. model 12.82 / 8.39**. The multitask model beat a constant on SBP by about 1 mmHg and was *worse* than a constant on DBP. The split type (subject-independent or not) is not stated. The model is 3.9 MB and CPU-capable — [arXiv 2508.17924 HTML](https://arxiv.org/html/2508.17924)

**Phase-shifted rPPG (Hwang et al., arXiv 2401.04560): face-vs-acral phase as a PTT proxy**
- BBP-Net results, MAE (RMSE): MMSE-HR (40 subjects, 102 sequences) SBP 10.19 (13.01), DBP 7.09 (8.86). V4V (179 subjects) SBP 13.64, DBP 9.40. Comparison on MMSE-HR: Schrumpf NIBPP 13.60/10.30, BPE-Net 12.35/9.54. On V4V: Schrumpf 31.36/20.62, Hamoud 15.12/11.17 — [arXiv 2401.04560v4](https://arxiv.org/html/2401.04560v4)
- Caveats: the output is clamped by a sigmoid to SBP 85–155 / DBP 45–95, which caps the error on extremes. No mean-predictor baseline is reported. Subject independence of the BP split is not explicit. The authors note too few hypo- and hypertensive samples — [arXiv 2401.04560v4](https://arxiv.org/html/2401.04560v4)

**Facial PTT / phase-difference physiology and limits**
- U-FaceBP observes that cheek pulse waves lead forehead pulse waves, but that "such clear phase differences are not consistently observed across all subjects" — [arXiv 2412.10679](https://arxiv.org/pdf/2412.10679)
- A 2018 CVPRW study of 21 subjects (Moço et al., "Impairing Factors in Remote-PPG Pulse Transit Time Measurements on the Face") found that skin variability threatens phase-shift estimation from collocated facial pixels — [TU/e page](https://www.es.ele.tue.nl/~sander/publications/cvpm18.php); [mlanthology](https://mlanthology.org/cvprw/2018/moco2018cvprw-impairing)
- Cheng et al.'s literature review summarises other rPPG BP approaches: Wu et al. (POS multi-ROI to 2D CNN, plus an InfoGAN for synthetic rPPG with BMI/age as inputs), Iuchi et al. (multi-ROI spatial maps to CNN), Chen et al. (spatiotemporal maps to ResNet-LSTM), Han et al. (cardiovascular-knowledge features to a Bayesian NN), and Chen et al. (Euler magnification of face and hand, PTT features, stacked regressors) — [Cheng et al. WACV 2025 PDF](https://www.ieda.ust.hk/dfaculty/so/pdf/Cheng-et-al-2025-WACV2025.pdf)

**Clinical cohort (arXiv 2503.00890, ambulatory cardiology patients)**
- 134 patients (143 visits), age 69±11.7, 71% hypertensive, 27% AF. Recording used a DSLR at 60 fps, with an oscillometric cuff reference and no per-subject calibration. SBP MAE: clinical-feature baseline 14.96 (r 0.261), rPPG model 11.61 (r 0.499), rPPG+clinical hybrid 11.35 (r 0.541), finger-PPG hybrid 11.84. Facial rPPG was on par with finger PPG. DBP results were not reported in the text read. A hypertension screen (SBP ≥130) reached 67.5% accuracy. The authors call the work exploratory, not a validation — [arXiv 2503.00890](https://arxiv.org/html/2503.00890v1)

**WACV 2026 "Non-Contact Blood Pressure Estimation from Face Videos via Physiology-Aware Contrastive Learning"**
- A dual-branch model (raw rPPG segments plus handcrafted waveform features) with augmentation-free contrastive pretraining on unlabeled facial videos and a distribution-aware loss for label imbalance. It claims "competitive accuracy under both controlled and in-the-wild conditions" and better cross-dataset transfer. Only the abstract was retrieved; no numbers were verified — [WACV 2026 poster page](https://wacv.thecvf.com/virtual/2026/poster/299)

### Inferences
- Across five independent subject-disjoint or large studies (Schrumpf, Cheng, U-FaceBP, MCD-rPPG, the clinical cohort), learned models improve SBP MAE over the mean or clinical baseline by roughly 1–4 mmHg. DBP gains are smaller, sometimes zero or negative. A realistic Round-3 target is SBP MAE about 10–13 and DBP MAE about 7–9 mmHg, with heavy regression to the mean.
- The low published MAEs on small datasets (e.g., MMSE-HR 10.19/7.09, rPPG-LUH 8.7/8.7) largely track narrow label distributions. On MSPM (SBP SD 11.27), even the mean regressor gets 9.10 SBP MAE.
- Facial-region PTT is physiologically motivated, but its timing differences are a few milliseconds at about 30 fps webcam rates. Sub-frame phase estimation is needed, and skin variability confounds it. Treat it as a feature, not a standalone method.
- Luo 2019 is the most-cited "it works" result. Its numbers come from a normotensive, range-restricted cohort with proprietary methods, so it should not be presented as evidence of general accuracy.

### Gaps
- Exact BP numbers for the WACV 2026 contrastive-learning paper were not retrieved (abstract only).
- No published BP results on VitalVideos were found. The only related artifact is a Mendeley release of "IPPG signals and BP from CN-BP and VV-Small" ([Mendeley](https://data.mendeley.com/datasets/3nmj5v5nrh/1)), which was not examined.
- "Wu et al." BP numbers (multi-ROI CNN; InfoGAN augmentation) were not retrieved directly; they are known only through Cheng et al.'s review.
- Whether MCD-rPPG's BP split is subject-independent is not stated in the paper text read.

---

## Q2. Critique literature: mean-predictor baselines, demographics, BP-range and leakage issues; calibration-based vs. calibration-free

### Takeaway
A strong critique literature shows that PPG/rPPG-to-BP is ill-conditioned. Common evaluation practices (segment-level or random splits, range trimming, short-record calibration) inflate results, and naive constant predictors come close to the "AAMI SD" in calibrated settings. Video-BP claims must always be shown next to a mean predictor and, ideally, a demographics-only predictor, with subject-disjoint splits and full-range reporting.

### Cited Findings
- **Mehta, Kwatra, Jain, McDuff (Microsoft Research), "Examining the challenges of blood pressure estimation via photoplethysmogram," Scientific Reports 2024 (doi 10.1038/s41598-024-68862-1)** — [PMC11306225](https://pmc.ncbi.nlm.nih.gov/articles/PMC11306225/):
  - Leakage: on PPG2ABP's SBP task, SD dropped from 23.1 mmHg (no overlap) to 16.2 (same patient in train and test) to 8.01 (overlapping windows split randomly).
  - Range trimming: their reference CNN's SD was 15.8 for SBP 65–200, 14.0 for 75–165, and 12.4 for 75–150 mmHg.
  - Calibration baseline: a naive constant (mean of the first three windows' ground truth) achieved bias 0.79, SD 8.61 mmHg, close to the AAMI SD target. Offset calibration error rises sharply after the first few days.
  - Ill-conditioning: 33.2% of MIMIC windows had a similar-PPG / very-different-SBP intra-patient match (15.0% inter-patient), vs. 0.75% for heart rate.
  - Information: autoencoder features captured about 12.9% (MIMIC) / 8.7% (Aurora-BP) of SBP information vs. about 92–93% for HR. Adding ECG and demographic features raised SBP Info-Fraction only to about 9.9%.
  - Recommendations: patient-level splits, keep extremes, evaluate calibration over realistic intervals, limit aggressive filtering, and check conditioning before training.
- **González, Hsieh, Chen, "A benchmark for machine-learning based non-invasive BP estimation using PPG," Scientific Data 2023** — [PMC10030661](https://pmc.ncbi.nlm.nih.gov/articles/PMC10030661/); [code](https://github.com/inventec-ai-center/bp-benchmark):
  - Four datasets (MIMIC-III "Sensors" 1,195 subjects; UCI/MIMIC-II; BCG 40 subjects; PPG-BP 219 subjects), subject-disjoint stratified splits, no calibration.
  - It introduced MASE: model MAE divided by the MAE of a training-mean predictor, where 100% means no better than the mean.
  - The best SBP MASE was only about 79.8% (PPG-BP), about 88.6% (Sensors), about 92.8% (BCG), and about 94.1% (UCI). Some models were worse than the mean (e.g., SpectroResNet on PPG-BP about 115% SBP, 129% DBP).
  - The authors call treating random-split leakage as "personalization/calibration" a common mistake.
- **Moulaeifard, Charlton, Strodthoff 2025 (arXiv 2502.19167)**: PulseDB-trained models reached in-distribution calibration-free MAE 14.0/8.5 mmHg and with calibration 9.4/6.0. Out-of-distribution, calibration-free MAE was 15.0–25.1 SBP and 7.0–10.4 DBP. "Performance depends strongly on differences in BP distributions between datasets" — [arXiv 2502.19167](https://arxiv.org/abs/2502.19167)
- **Schrumpf 2021**: prediction error grows systematically for less frequent BP values, and rPPG models were near the mean regressor (see Q1) — [arXiv 2104.09313](https://arxiv.org/abs/2104.09313)
- **Demographics vs. PPG (Aktiia, Frontiers Digital Health 2025, 4,823 test users)**:
  - Pearson r vs. cuff: demographics (age, sex, BMI) 0.23 SBP / 0.28 DBP; PPG 0.53 / 0.58; PPG+demographics 0.63 / 0.67. AUC for high BP: demographics 0.633/0.659 vs. PPG 0.842/0.891.
  - Caveats: all authors are Aktiia employees, the device relies on monthly cuff "initializations," and wrist PPG is not facial rPPG. No MAE/SD was reported.
  - [Frontiers fdgth.2025.1518322](https://www.frontiersin.org/articles/10.3389/fdgth.2025.1518322/full)
- **Clinical rPPG cohort**: a clinical-feature-only baseline gave SBP MAE 14.96 vs. 11.35 for the rPPG hybrid — [arXiv 2503.00890](https://arxiv.org/html/2503.00890v1)
- A 2026 preprint argues that mixed-population training lets models "exploit pathological or disease-specific patterns correlated with BP." It proposes healthy-adult benchmarks from VitalDB/MIMIC-III with explicit age/sex/BMI inputs (seen in search snippet only) — [arXiv 2602.04725](https://arxiv.org/pdf/2602.04725)
- Gutierrez et al., "Exploring the limitations of blood pressure estimation using the photoplethysmography signal" (arXiv 2404.16049), reaches similar skeptical conclusions against invasive ABP references (snippet only) — [arXiv 2404.16049](https://arxiv.org/pdf/2404.16049)

### Inferences
- In face video, demographic proxies (apparent age, sex, adiposity) are directly visible, so a CNN on face images can learn them. U-FaceBP explicitly uses face images. Any honest Round-3 presentation should include a "face-appearance or demographics only" baseline to show whether the pulse signal adds anything.
- **Calibration-free vs. calibrated**: calibrated settings make the job easier. A constant equal to the subject's recent cuff value already nears AAMI SD over short horizons, so calibrated results should be compared with a "repeat the calibration value" baseline and evaluated over realistic recalibration intervals. In a competition with a single cuff reading per test subject, calibration-free is the realistic mode. If organisers allow a per-subject initial cuff reading, report both.
- Most judges with biomedical backgrounds will know Mehta 2024 / González 2023. Pre-empting the "you're predicting the mean" critique with MASE and scatter/Bland-Altman plots is likely to be judged well.

### Gaps
- No video-specific paper was found that directly measures how much of face-video BP accuracy comes from visible demographics (e.g., a face-image-only model vs. an rPPG-only model on the same split). U-FaceBP's ablations beyond the first 100k characters were not read.
- The El-Hajj & Kyriacou reviews were not retrieved in this session.

---

## Q3. Standards and how to report (AAMI/ESH/ISO 81060-2, BHS, IEEE 1708, ESH 2023 cuffless, ISO 81060-3)

### Takeaway
Report mean error ± SD (Bland-Altman) for SBP and DBP against the AAMI/ESH/ISO 81060-2 Criterion 1 (|mean| ≤5 mmHg, SD ≤8 mmHg, at least 85 subjects). Add BHS cumulative-percentage grading (within 5/10/15 mmHg), IEEE 1708 MAD grade, MAE, MASE vs. the mean predictor, Pearson r, and performance by BP category. Be explicit that a small competition test is not a validation study, and that the ESH (2022/2023) position is that cuffless devices are not recommended for hypertension diagnosis or management.

### Cited Findings
- ISO 81060-2 Criterion 1: mean difference within ±5.0 mmHg and SD at most 8.0 mmHg, applied separately to SBP and DBP. The 85-subject sample size derives from early AAMI work (98% CI, 95% power) — [ISO 81060-2:2018 study protocol on ClinicalTrials.gov](https://cdn.clinicaltrials.gov/large-docs/14/NCT05654714/Prot_002.pdf)
- ISO 81060-2 Criterion 2 limits the SD of *per-subject* mean differences to a value that depends on the mean difference. For example, SD ≤6.95 mmHg at mean difference 0 and ≤6.76 at ±1.6 mmHg — [search summary of validation reports, e.g. ABPMpro validation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10132455/)
- The standard also requires a BP distribution: for example, at least 5% of readings with DBP ≥100 and at least 20% with DBP ≥85. This came from a 510(k) summary and was not verified against the standard text — [510k summary K163683](https://510k.innolitics.com/device/K163683)
- BHS grading: Grade A requires at least 60/85/95% of errors within 5/10/15 mmHg. U-FaceBP uses the within-10 mmHg (Suc10) thresholds of ≥65% for Grade C and ≥75% for Grade B — [JMIR 2018](https://www.jmir.org/2018/10/e10089); [arXiv 2412.10679v3](https://arxiv.org/html/2412.10679v3)
- IEEE 1708-2014 (with 2019 amendment) is a validation standard specific to cuffless devices. It grades by mean absolute difference: A ≤5 mmHg, B 5–6, C 6–7. Sources differ slightly on boundary handling — [PMC table](https://pmc.ncbi.nlm.nih.gov/articles/PMC6431352/table/Tab3); [Standard Norge listing](https://online.standard.no/en/ieee-1708)
- ESH 2023 recommendations for validating intermittent cuffless devices (Stergiou et al., J Hypertens 2023;41:2074–2087, doi 10.1097/HJH.0000000000003483) define six tests: static (absolute accuracy), device position (hydrostatic), treatment (BP decrease), awake/asleep, exercise (BP increase), and recalibration (stability over time). The tests required depend on whether the device needs per-user calibration — [Macquarie listing](https://researchers.mq.edu.au/en/publications/european-society-of-hypertension-recommendations-for-the-validati/); [Oxford PHC](https://www.phc.ox.ac.uk/publications/1393612)
- The 2022 ESH Working Group review says established cuff protocols are inadequate for cuffless devices. It notes that ESH guidelines "do not recommend cuffless devices for the diagnosis and management of hypertension" and cites IEEE 1708-2014/2019 and ISO 81060-3:2022 (continuous devices) — [Springer Curr Cardiol Rep](https://link.springer.com/article/10.1007/s11886-023-01956-w); [Macquarie listing](https://researchers.mq.edu.au/en/publications/cuffless-blood-pressure-measuring-devices-review-and-statement-by/)
- ISO 81060-3:2022 includes change-tracking criteria. A 2024 prototype met mean-difference accuracy but only partly met change tracking — [Frontiers Med Tech 2024](https://www.frontiersin.org/journals/medical-technology/articles/10.3389/fmedt.2024.1464473/pdf)
- Elgendi et al. (Communications Medicine 2024) review seven commonly used metrics and recommend standardized error evaluation for PPG BP algorithms — [PMC11245506](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11245506/)
- MASE (MAE relative to a training-mean predictor) is a recommended cross-dataset comparison metric — [González et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10030661/)

### Inferences
- Recommended reporting table for the SP Cup: for each of SBP and DBP, report n subjects and n readings, BP range and SD of the reference, ME±SD (with ISO Criterion 1 pass/fail), MAE, MASE (vs. train-mean), r, BHS % within 5/10/15 and grade, and IEEE 1708 MAD grade. Add Bland-Altman and predicted-vs-reference scatter plots with the identity line (to expose regression to the mean), plus error stratified by BP category (<120, 120–139, ≥140 SBP).
- Note a statistical trap: a model that outputs a near-constant can pass "|mean| ≤5" easily, and the SD criterion is passed only if the population SD of BP is small. With a narrow test population (e.g., healthy students, SBP SD about 10), even a mean predictor can come close to SD ≤8–10. Judges should therefore see the reference distribution and MASE alongside AAMI numbers.
- Each test subject should be measured with cuff readings taken close in time (ideally the ISO-style sequential or simultaneous protocol, averaged). Cuff variability itself is several mmHg, which sets a floor on achievable error.

### Gaps
- The exact ISO 81060-2 BP-distribution table and IEEE 1708 full protocol (calibration, sample size, BP-change requirements) were not verified from primary text, since the standards are paywalled.
- Specific ESH 2023 thresholds per test were not retrieved.
- BHS Grade B/C cumulative thresholds at 5 and 15 mmHg were not verified from a primary source in this session.

---

## Q4. Datasets: facial video + BP, and PPG-only BP datasets for pretraining; domain gap

### Takeaway
Usable facial-video BP datasets now range from small (MMSE-HR 40 subjects, rPPG-LUH 17) to large (MCD-rPPG 600, VitalVideos about 850–893, FaceBP 1,094 private, CLBP-300 controlled-access). Most have one cuff reading per session and narrow normotensive distributions. Large contact-PPG datasets (PulseDB about 5.2M segments / 5,361 subjects; MIMIC; PPG-BP 219 subjects) exist for pretraining. ICU arterial-line PPG differs markedly from webcam rPPG in sampling, noise, and population, so transfer gains are modest.

### Cited Findings
**Facial video + BP**
- **MCD-rPPG** (Hugging Face): 600 subjects, 3,600 videos from 3 cameras at rest and post-exercise. It includes 100 Hz PPG, ECG, BP (A&D UA-911BT-C) and 13 biomarkers. SBP 80–202, DBP 50–108, mostly young (mean age 23) — [arXiv 2508.17924](https://arxiv.org/html/2508.17924); [HF dataset](https://huggingface.co/datasets/Bgeorge/mcd_rppg)
- **VitalVideos-Europe**: 850 participants, two 30 s uncompressed videos each, synchronized PPG, and a single BP measurement per participant — [arXiv 2306.11891](https://arxiv.org/pdf/2306.11891). A later analysis describes VitalVideos as 893 subjects across 6 Fitzpatrick types — [paperswithcode summary](https://cs.paperswithcode.com/paper/camera-based-remote-physiology-sensing-for)
- **CLBP-300** (Al-Naji et al., Preprints.org April 2026, doi 10.20944/preprints202604.1079.v1): 300 adults aged 18–65 with SBP/DBP/HR from clinical electronic monitors, indoor and outdoor illuminance, about 58 GB. It is controlled-access (institutional email, signed DUA, support fee), with only a public sample — [Mendeley v2](https://data.mendeley.com/datasets/8xpbdtctpv/2)
- **FaceBP** (U-FaceBP authors): 1,094 subjects, SBP 88–234, ages 18–91. Public availability was not confirmed. **MSPM**: 103 subjects, public — [arXiv 2412.10679v3](https://arxiv.org/html/2412.10679v3)
- **MMSE-HR**: 102 videos from 40 subjects, used for BP by phase-shifted rPPG. **V4V**: 179 subjects, 1,358 sequences — [arXiv 2401.04560v4](https://arxiv.org/html/2401.04560v4)
- **rPPG-LUH** (Leipzig ICU): 17 subjects publicly released, about 2 h recordings at 32 fps with an industrial USB camera and arterial-line reference. SBP skewed above 130 — [Cheng et al. WACV 2025](https://www.ieda.ust.hk/dfaculty/so/pdf/Cheng-et-al-2025-WACV2025.pdf)
- **Clinical ambulatory cohort** (134 patients, DSLR 60 fps, cuff BP); availability not stated — [arXiv 2503.00890](https://arxiv.org/html/2503.00890v1)

**PPG-only BP datasets for pretraining**
- **PulseDB**: about 5.2 million 10 s segments of ECG, PPG, and ABP from 5,361 subjects (MIMIC-III matched subset plus VitalDB). It includes subject IDs and demographics, and has predefined calibration-based ("Calib") and calibration-free subsets — [PulseDB, Frontiers Digital Health 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9944565/)
- **González 2023 benchmark collection**: MIMIC-III "Sensors" (1,195 subjects), UCI (MIMIC-II, no subject IDs), BCG (40 subjects), PPG-BP (219 subjects, cardiovascular conditions). Preprocessed data are on Figshare — [Sci Data benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC10030661/); [Figshare](https://springernature.figshare.com/collections/A_Benchmark_for_Machine-Learning_based_Non-Invasive_Blood_Pressure_Estimation_using_Photoplethysmogram/6150390)
- **Aurora-BP**: 24,650 records from 483 subjects, used by Mehta et al. — [PMC11306225](https://pmc.ncbi.nlm.nih.gov/articles/PMC11306225/)
- **MIMIC-III Kaggle subset**: 12,000 records. Patients can repeat across records, so it is a leakage risk — [Cheng et al. WACV 2025](https://www.ieda.ust.hk/dfaculty/so/pdf/Cheng-et-al-2025-WACV2025.pdf)

**Domain gap**
- Cheng et al. found pretraining on MIMIC significantly improved rPPG-LUH results (paired t-tests p<0.05). They attribute this to large, clean PPG from medical devices vs. small, noisy rPPG, and say combining rPPG datasets is hard because setups differ — [Cheng et al. WACV 2025](https://www.ieda.ust.hk/dfaculty/so/pdf/Cheng-et-al-2025-WACV2025.pdf)
- Phase-shifted rPPG notes that reference PPG in rPPG datasets comes from fingertips, so a site-related time shift exists vs. facial pulse. Training with time-domain losses against finger PPG ignores this — [arXiv 2401.04560](https://arxiv.org/html/2401.04560v4)
- Out-of-distribution calibration-free PPG→BP MAE degrades to 15–25 mmHg SBP across external datasets — [arXiv 2502.19167](https://arxiv.org/abs/2502.19167)

### Inferences
- ICU populations (MIMIC/VitalDB: older, medicated, arterial-line ABP) differ in BP distribution from a young, healthy SP Cup test cohort. Pretraining helps learn waveform features but can shift the prior mean. Fine-tuning or recentering on face-video data with cuff labels is required.
- MCD-rPPG is the most practical public face-video BP dataset for fine-tuning: large, public, with cuff reference, and its BP range resembles a student/young-adult test population. VitalVideos adds skin-tone diversity. MMSE-HR and V4V are useful for comparison with literature numbers.

### Gaps
- DDMP-FV, BP4D+ access terms, and UBFC-Phys BP availability were not verified in this session. My understanding is that UBFC-Phys provides BVP and EDA rather than cuff BP, but this is unconfirmed.
- VitalVideos licensing and whether its BP labels can be used for model training in a competition were not checked.
- FaceBP public availability was not confirmed.

---

## Q5. Feasible, lightweight approach for a student team (~3 months, Raspberry Pi)

### Takeaway
Build on the rPPG pipeline already needed for HR and SpO2. Extract beat-averaged waveform-morphology features plus multi-ROI phase-lag features plus HR (and optionally age/sex/BMI if allowed). Train a small, regularised regressor (ridge or LightGBM, or a tiny 1D-CNN pretrained on PulseDB and fine-tuned on MCD-rPPG). Evaluate subject-disjoint against mean and demographics baselines with full AAMI/BHS/IEEE 1708/MASE reporting. Optionally add a one-cuff-reading calibration mode. Expect MAE about 10–13 / 7–9 mmHg and say so.

### Cited Findings
- Feature-based (classical ML) models were competitive with deep models, especially on smaller datasets. Deep models won only on the largest dataset — [González et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10030661/)
- MCD-rPPG's multitask rPPG model is 3.9 MB and CPU-capable, but its BP head barely beats a constant — [arXiv 2508.17924](https://arxiv.org/html/2508.17924)
- Adding HR as an input improved rPPG BiLSTM SBP prediction in Cheng et al. — [Cheng et al. WACV 2025](https://www.ieda.ust.hk/dfaculty/so/pdf/Cheng-et-al-2025-WACV2025.pdf)
- U-FaceBP authors list real-time mobile inference as future work (lightweight architectures or distillation needed). Their best model is an ensemble of Bayesian networks over three modalities, not Pi-friendly as is — [arXiv 2412.10679v3](https://arxiv.org/html/2412.10679v3)
- Calibration (one subject-specific offset) cut PulseDB MAE from 14.0/8.5 to 9.4/6.0 in-distribution. However, Mehta et al. show calibration benefits decay over days and that a calibrated constant baseline is very strong — [arXiv 2502.19167](https://arxiv.org/abs/2502.19167); [Mehta et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11306225/)

### Inferences (recommended plan; this is synthesis, not sourced fact)
1. **Weeks 1–2: baselines first.** Implement training-mean, demographics-only (if age/sex/BMI collection is permitted), and HR-only regressors on MCD-rPPG with subject-disjoint 5-fold CV. Compute MASE for everything thereafter.
2. **Weeks 2–5: features from the existing rPPG signal.** Use POS/CHROM or the team's HR model output, with 10–30 s windows and beat segmentation into an ensemble-averaged pulse. Features: rise time, pulse width at 25/50/75%, systolic/diastolic area ratio, dicrotic-notch timing where detectable, first and second derivative (VPG/APG a–e wave ratios), HR and HRV. Add forehead-vs-cheek (and optionally face-vs-palm if the setup shows hands) phase lag via cross-spectral phase at the HR fundamental, with sub-frame interpolation. Regress with ridge, LightGBM, or a monotone GBM; these are tiny and fast on a Pi.
3. **Weeks 4–8 (optional): tiny 1D-CNN** (under 1M parameters), pretrained on PulseDB calibration-free subjects (downsampled to camera rates, with band-limiting and noise augmentation to mimic rPPG), then fine-tuned on MCD-rPPG/VitalVideos. Keep it only if it beats the feature model on MASE under subject-disjoint CV.
4. **Calibration mode (optional, clearly labelled):** a one-shot offset from an initial cuff reading. Compare against the "repeat calibration value" baseline.
5. **Output design:** show BP with an uncertainty band (e.g., quantile regression or a CV-derived interval) and a categorical readout (normal / elevated / high) with confidence. Suppress output when signal quality (SNR, motion) is poor.
6. **Presentation:** report the Q3 table and Bland-Altman plot, a predicted-vs-reference scatter showing compression toward the mean, and the cuff's own repeat variability. State plainly that this is not a validated BP device, citing the ESH position.

### Gaps
- No published Raspberry Pi benchmark for video-BP inference was found. Feasibility rests on the small size of feature/GBM models and the 3.9 MB MCD-rPPG model.
- Competition rules (whether demographics may be entered, whether per-subject calibration is permitted, BP range of the final test cohort) were not available here and strongly affect strategy.
