# rPPG Robustness to Deployment Constraints (C1 low light, C2 skin tone, C4 low res, C5 low bandwidth, C6 aging, C7 masks, plus motion)

Scope note: about 20 search/fetch calls were made. Several primary PDFs could not be machine-read (arXiv 2503.03780 PDF was binary; PMC12297079 returned a bot-check page). Where numbers come from a secondary citation, or from background knowledge not re-verified in this session, this is stated explicitly. "Pi-cost" judgements are inferences unless a source is cited.

## C1 Low light (<=50 lux): how badly do POS/CHROM and deep methods degrade, and do enhancement or camera tricks help?

### Takeaway
The best recent controlled study (CHILL, npj Digital Medicine, Dec 2025) found illumination had a small effect relative to elevated heart rate. But its "dark" condition was 140 lux on a DSLR, nearly 3x brighter than the challenge's <=50 lux, so it does not establish robustness at 50 lux. Other work (Xi et al. 2023; arXiv 2503.03780) reports clear failure modes in genuinely low light: face detection fails, pulse amplitude falls into sensor noise, and POS's chrominance/skin-tone assumption breaks. There is no published head-to-head showing that generic low-light enhancers (Zero-DCE, Retinexformer, etc.) improve HR MAE. The one dedicated rPPG enhancement pipeline reports SNR gains, but we did not obtain its numbers.

### Cited Findings
- CHILL dataset: 45 participants, 180 one-minute recordings. Lighting was "dark" = 140 lux and "bright" = 830 lux (luxmeter-measured, windows shielded). Camera was a Canon EOS 550D at 1920x1080, 25 fps. Exposure and gain settings were not reported. Reference was a finger PPG at 1000 Hz. Fitzpatrick I-III only, ages 18-32. — [Acharya et al., npj Digit Med 2025 (PMC12678791)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12678791/)
- CHILL within-dataset, bright vs dark: only GREEN (p=0.045) and ICA (p=0.0028) changed significantly, and both were slightly *better* in the dark. The other six methods (POS, CHROM, DeepPhys, TS-CAN, PhysNet, rPPGNet) were not significant. Five of eight methods degraded significantly at elevated HR. — [PMC12678791](https://pmc.ncbi.nlm.nih.gov/articles/PMC12678791/)
- CHILL within-dataset MAE (bpm, all conditions pooled): POS 1.1, CHROM 1.8, GREEN 2.6, rPPGNet 2.0, TS-CAN 3.2, PhysNet 4.1, ICA 5.4, DeepPhys 9.1. A mean-predictor dummy scored 10.8. Classical POS/CHROM were the best in this static, well-exposed setting. — [PMC12678791](https://pmc.ncbi.nlm.nih.gov/articles/PMC12678791/)
- Cross-dataset results on CHILL (MAE bpm, rest-bright / rest-dark / elevated-bright / elevated-dark):
  - TS-CAN trained on PURE: 0.42 / 0.94 / 1.80 / 1.44.
  - DeepPhys trained on PURE: 0.50 / 0.67 / 1.61 / 2.72.
  - PhysNet trained on PURE: 12.00 / 9.99 / 20.63 / 19.67 (worse than the 10.75 dummy).
  - rPPGNet trained on COHFACE: 14.47 / 27.59 / 9.24 / 20.96.
  - Some deep models fail to generalise regardless of light. — [PMC12678791](https://pmc.ncbi.nlm.nih.gov/articles/PMC12678791/)
- On COHFACE (studio vs natural light), DeepPhys, PhysNet, TS-CAN and GREEN were significantly better under studio light. POS, CHROM, ICA and rPPGNet were not significantly affected. — [PMC12678791](https://pmc.ncbi.nlm.nih.gov/articles/PMC12678791/)
- Press summary from the authors: low lighting has little effect, but accuracy drops sharply at high HR. — [Uni Bielefeld press release, Dec 2025](https://aktuell.uni-bielefeld.de/2025/12/09/contactless-pulse-measurement-falters-at-high-heart-rates/?lang=en)
- Xi et al., "Image Enhancement for rPPG in a Low-Light Environment" (FG 2023, arXiv 2303.09336):
  - Built a large multi-illumination HR database ([GitHub: Large-scale-Multi-illumination-HR-Database](https://github.com/xilin1991/Large-scale-Multi-illumination-HR-Database), also called BUAA-MIHR).
  - In low light, face detection fails, the reduced pulsatile amplitude lets noise dominate, and the chrominance method's skin-tone assumption breaks.
  - GREEN and ICA were less affected than POS in darkness.
  - A proposed low-light enhancement pipeline (enhance, then detect/track face, then extract HR) "improves SNR and pulsatile-signal precision", but the abstract gives no numbers. — [arXiv 2303.09336](https://arxiv.org/abs/2303.09336); [search summary](https://arxiv.org/pdf/2303.09336)
- arXiv 2503.03780 (2025) proposes weighted combination plus singular spectrum analysis for rPPG in low-light environments. It states that previous methods fail under low illumination. We could not extract its numbers. — [arXiv 2503.03780](https://arxiv.org/pdf/2503.03780)
- arXiv 2503.11697 is a preprint version or companion of the CHILL work ("Generalization of video-based HR estimation methods to low illumination and elevated heart rates"). — [arXiv 2503.11697](https://arxiv.org/html/2503.11697v1)
- Rolling-shutter timing can create phase shifts of the same order as physiological ones. Irregular frame timing did not significantly distort PPG spectra. — [Mironenko et al., CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w19/Mironenko_Remote_Photoplethysmography_Rarely_Considered_Factors_CVPRW_2020_paper.pdf)

### Inferences
- These points are inferences, not proven:
  - CHILL's null illumination result is only evidence for >=140 lux with a large-sensor DSLR. At <=50 lux on a small Pi camera sensor, photon shot noise and quantisation are likely much worse, so the team should **measure** degradation on its own captures rather than assume robustness.
  - Mechanistically, global per-pixel enhancement (gamma, histogram equalisation, Zero-DCE-style curves) is applied identically to all frames. Spatial averaging over the ROI then mostly undoes any SNR gain. The more likely benefit of enhancement is **better face detection and landmarking**, not a better pulse signal.
  - Recommended pattern: run enhancement only on the detector's input (e.g., gamma or CLAHE on a downscaled copy), and extract RGB traces from the **raw, un-enhanced** frames. This is cheap on a Pi.
  - Non-linear, temporally varying enhancement (per-frame histogram equalisation, auto-tuned Retinex, learned enhancers with frame-to-frame flicker) risks injecting low-frequency luminance fluctuations into the HR band. Avoid it on the signal path, or use temporally fixed curves.
  - Deep enhancers such as Retinexformer and Zero-DCE are too slow for a Pi at video rate except in tiny variants. Zero-DCE++ is light, but there is no rPPG evidence for it.
  - Temporal video denoising (e.g., recursive averaging) can low-pass the pulse. Spatial averaging over a large ROI is the "free" denoiser.
- Camera-side tricks (engineering reasoning, not validated in the sources found):
  - Lock exposure, gain and white balance (auto-exposure and AWB steps appear as step artefacts).
  - Use the longest exposure the frame rate allows.
  - Prefer analogue gain over digital gain.
  - Capture without heavy in-camera denoise or sharpening.
  - Use raw or YUV output where possible.
  - The Pi camera stack (libcamera/Picamera2) exposes manual exposure, analogue gain and AWB lock, so these are cheap to do.
- Elevated HR (>80-100 bpm) appears to be a larger error driver than light. A test set with exercise-elevated HR would hit both effects at once.

### Gaps
- We could not verify an "MSLR" (multi-scene low-light rPPG) database by that name. The closest match is Xi et al.'s multi-illumination HR database (BUAA-MIHR). Its lux levels and per-lux MAE were not retrieved.
- arXiv 2107.13193 (illumination assessment) was not retrieved in this session.
- We found no study measuring rPPG MAE at <=50 lux on a Raspberry Pi or low-cost CMOS camera.
- We found no paper quantifying Zero-DCE, Retinexformer or LLNet followed by POS/CHROM HR MAE.
- We found no direct evidence on raw vs ISP-processed capture for rPPG.

## C2 Skin tone (Fitzpatrick I-VI plus vitiligo; ITA stratification): bias magnitude and mitigations

### Takeaway
Darker skin (Fitzpatrick V-VI) consistently shows higher rPPG error, because melanin absorbs light and lowers pulsatile amplitude. The commonly quoted magnitude (CHROM 5.2 to 14.1 bpm MAE, DL 6.0 to 9.5 bpm, from I-III to V-VI) comes from a 2025 audit's citation that we could not reconcile with the cited Dasari paper. The best-supported mitigation is synthetic skin-tone augmentation for deep models (about 31-38% MAE reduction for dark skin). Public datasets are heavily skewed toward light skin, and most skin-type labels are subjective.

### Cited Findings
- Nowara et al. meta-analysis (CVPRW 2020): 3 datasets, 73 people, over 400 videos. HR measurement works on all skin types under some conditions, but average performance drops noticeably for the darkest skin type. There is a slight drop for female subjects, and skin type does not affect all algorithms equally (supervised vs unsupervised differ). — [Nowara et al., CVPRW 2020](https://mlanthology.org/cvprw/2020/nowara2020cvprw-metaanalysis)
- Dasari et al., "Evaluation of biases in rPPG methods" (npj Digit Med 2021):
  - 140 subjects (44 India, 96 Sierra Leone), Samsung J7 at about 25 fps, Masimo finger oximeter reference.
  - Skin tone was **not** scored on Fitzpatrick or ITA. Groups were compared via KS tests on colour-space distributions.
  - Bland-Altman mean error / SD (bpm): CHROM -6.40/14.15, POS 9.59/14.14, BKF 0.72/6.99, Spherical Mean 14.92/7.50, DeepPhys -14.11/12.24.
  - Between-country differences were not significant (p=0.08-0.64). The authors acknowledge the design was underpowered.
  - Outliers were driven by motion, poor or changing lighting, and extra people in frame, and came mostly from the Sierra Leone recordings. — [PMC8175478](https://pmc.ncbi.nlm.nih.gov/articles/PMC8175478/)
- A 2025 npj Digital Medicine audit (Bondarenko, Menon, Elgendi) of 100 rPPG studies/datasets cites Dasari et al. as follows: chrominance methods 5.2 bpm MAE on Fitzpatrick I-III vs 14.1 on V-VI (p<0.01); deep learning 6.0 vs 9.5 bpm. It also reports underrepresentation of darker skin and gender imbalance. — [Demographic bias in public rPPG datasets, PMC12491395](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12491395/)
  - **Conflict:** the Dasari paper itself (PMC8175478) uses no Fitzpatrick grouping and reports no per-skin-type MAE. The 5.2/14.1 figures therefore cannot be traced to it and should not be quoted as Dasari's results without checking.
- Ba et al. (UCLA, Kadambi group), "Overcoming Difficulty in Obtaining Dark-skinned Subjects for Remote-PPG by Synthetic Augmentation" (arXiv 2106.06007):
  - Translates light-skinned videos into dark skin tones while preserving the pulse.
  - Reports about a 31% reduction in HR MAE for the dark-skinned group and about 46% improvement in bias mitigation across groups, vs training on real data only.
  - Dataset skew it cites: MMSE-HR about 10% dark-skinned subjects, AFRL 0%, UBFC-rPPG about 5%. — [arXiv 2106.06007](https://arxiv.org/pdf/2106.06007)
  - **Conflict:** the UCLA project page for the bio-realistic style-transfer version reports about 38% MAE reduction and about 49% bias improvement. This may be a revised method or evaluation. — [UCLA project page](https://visual.ee.ucla.edu/rppg_augmentation.htm)
- PhysFlow (BMVC 2024): skin-tone transfer via conditional normalising flows, conditioned on CIELAB skin features, so no skin-tone labels are needed. Validated on UCLA-rPPG and MMPD, with reduced HR error, particularly for dark skin. It also notes that earlier studies (including Nowara) documented the dark-skin drop without fixing it. We did not get exact numbers. — [arXiv 2407.21519](https://arxiv.org/html/2407.21519v1); [BMVC paper](https://bmva-archive.org.uk/bmvc/2024/papers/Paper_136/paper.pdf)
- Xi et al. found that in low light, POS's chrominance and skin-tone assumption no longer holds. This suggests skin tone and low light compound each other. — [arXiv 2303.09336](https://arxiv.org/abs/2303.09336)
- MMPD (EMBC 2023): 33 subjects, about 11 h of mobile-phone video, 4 skin tones (Fitzpatrick III-VI), 4 motion and 4 lighting conditions, compatible with rPPG-Toolbox. We did not retrieve per-skin-tone MAE tables. — [arXiv 2302.03840](https://arxiv.org/pdf/2302.03840); [UW project page](https://ubicomplab.cs.washington.edu/publications/mmpd)
- Vital Videos dataset:
  - Skin types were labelled subjectively before recording, with extra strictness for types 4-6. The authors note a colorimeter (about USD 3,000) would be more objective. — [search summary of the Vital Videos paper](https://arxiv.org/pdf/2306.11891v3)
  - An analysis covering 893 subjects across 6 Fitzpatrick types found a few hundred subjects suffice for training, and that skin-tone diversity matters for cross-dataset evaluation. — [arXiv 2404.05003](https://arxiv.org/html/2404.05003v1)
- The VitalLens technical report (vendor-authored, VV-Medium, 289 participants) reports HR MAE 0.71 bpm. It claims VitalLens is less affected than POS by darker skin, but we found no per-type numbers. — [arXiv 2312.06892](https://arxiv.org/abs/2312.06892)
- Field study in Nigeria (Fitzpatrick V-VI) of an rPPG blood-pressure app: Fitzpatrick VI was associated with higher odds of measurement failure. This concerns BP, not HR. — [BMJ Open 2026, e119311](https://bmjopen.bmj.com/content/16/6/e119311)
- Fitzpatrick is subjective (self- or rater-assigned), not a measurement. — [QUMPHY skin-tone/PPG material](https://www.qumphy.ptb.de/fileadmin/epm2022/qumphy/documents/Educational_Material/skin_tone_classification_with_ppg.pdf)

### Inferences
- **ITA computation** (standard definition, Chardon et al. 1991; not re-sourced in this session):
  - Formula: ITA° = arctan((L* - 50) / b*) x 180/π, computed on CIELAB of skin pixels.
  - Common bins: >55 very light, 41-55 light, 28-41 intermediate, 10-28 tan, -30 to 10 brown, < -30 dark.
- **ITA pitfalls under low light** (inference):
  - L* scales with exposure, so an underexposed light-skinned face reads as "dark", and ITA drops.
  - Camera AWB, colour cast from warm or LED lighting, and gamma/tone curves shift b*.
  - Recommended mitigations:
    - Compute ITA from a well-exposed reference frame, or after grey-world or colour-card white balance.
    - Use median skin pixels from cheeks/forehead, excluding specular highlights, beard and shadow.
    - Report ITA per subject, not per frame.
  - For vitiligo, ITA is bimodal within a face, so compute ITA (and pulse SNR) per ROI patch.
- Physically, melanin lowers the absolute pulsatile amplitude and raises relative noise. Low light and low resolution therefore likely hurt dark skin more (a compounding effect, consistent with Xi et al.'s note).
- Cheap Pi-friendly mitigations:
  - Signal-domain: use POS (relatively insensitive to absolute skin colour because it normalises by the temporal mean), with multi-patch SNR-weighted averaging and longer windows.
  - Training-domain (offline cost only, zero inference cost): skin-tone augmentation (Ba et al., PhysFlow) and physics-based synthetic data such as SCAMPS.
- Vitiligo: no rPPG evidence was found. Patch-wise processing with per-patch SNR weighting would naturally down-weight depigmented or low-SNR patches, or exploit them if their SNR is higher. This is speculative.

### Gaps
- We could not verify the original source of the 5.2 to 14.1 bpm figure.
- No per-skin-type MAE tables were retrieved for MMPD, VitalVideos, or UCLA-rPPG.
- No rPPG study on vitiligo was found.
- We found no paper using ITA (rather than Fitzpatrick) for rPPG stratification, though PhysFlow uses CIELAB conditioning.
- SCAMPS and fairness-aware training losses were not investigated, due to the call budget.

## C4/C5 Low resolution, compression (<=500 kbps) and low frame rate (<=10 fps)

### Takeaway
Compression hurts rPPG mainly through quantisation, 4:2:0 chroma subsampling and inter-frame skip/merge modes, which erase sub-pixel-level colour changes. But H.265 at about 0.26 Mbps (PURE) cost only about 0.3 bpm in one study, so 500 kbps is likely workable for a static face if the codec is tuned. Training on compressed video, or compression-aware networks (STVEN/rPPGNet; Deep Pulse-Signal Magnification), helps deep models. No study tested exactly 10 fps. At 10 fps the Nyquist limit is 5 Hz = 300 bpm, so HR (0.7-3 Hz) is not aliased in principle, but HRV and peak timing degrade.

### Cited Findings
- On PURE, H.265 gave the best accuracy-per-bitrate. Accuracy dropped about 0.3 bpm while bitrate fell two orders of magnitude (26 Mbps to 0.26 Mbps). FFV1 lossless did not affect rPPG accuracy at all. — [Gudi et al., arXiv 2012.15846](https://arxiv.org/pdf/2012.15846)
- Resolution-aware compression study (HiDURPA dataset):
  - Setup: uncompressed 1080p30 4:2:2 masters, 11 adults, 55 clips of 30 s, ECG reference. Codecs H.264, H.265 (4:2:2), H.266 and AV1 (4:2:0), at 1080p, 720p, 480p and 360p.
  - Methods tested: GREEN, ICA, CHROM, LGI, OMIT.
  - Strong compression (aggressive QP, 4:2:0, skip/merge, downscaling) suppresses the pulsatile chrominance.
  - ICA failed numerically on about 4% of sequences: whitening became rank-deficient as channels turned near-collinear.
  - GREEN, CHROM, LGI and OMIT did not fail numerically, but their accuracy still degraded.
  - Numeric thresholds are only in figures we could not read. — [EAH Jena VitaWave](https://www.eah-jena.de/vitawave/publications/videocompressionrppg); [visual-computing.com](https://visual-computing.com/publications/a-study-of-resolution-aware-video-compression-for-remote-photoplethysmography)
- Compression-robust deep methods:
  - Yu et al., ICCV 2019: STVEN video enhancer plus rPPGNet for highly compressed facial video. — [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Remote_Heart_Rate_Measurement_From_Highly_Compressed_Facial_Videos_An_ICCV_2019_paper.pdf)
  - "Deep Pulse-Signal Magnification" (2024) maps compressed video to an uncompressed-like domain with a magnified rPPG signal. — [arXiv 2405.02652](https://arxiv.org/pdf/2405.02652)
  - A separate method treats pulse extraction as single-channel signal separation and reports gains at low bitrates. — [IS&T EI 2024, IPAS-248](https://library.imaging.org/ei/articles/36/10/IPAS-248)
- CHILL used face crops of 72x72 px (signal-processing methods), 36x36 (DeepPhys/TS-CAN) and 128x128 (PhysNet/rPPGNet). DeepPhys and TS-CAN reached about 1 bpm cross-dataset MAE at 36x36. This shows that very low face resolution is not inherently fatal for clean, static video. — [PMC12678791](https://pmc.ncbi.nlm.nih.gov/articles/PMC12678791/)
- Frame timing:
  - Irregular frame rates did not significantly distort PPG spectra. — [Mironenko et al., CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w19/Mironenko_Remote_Photoplethysmography_Rarely_Considered_Factors_CVPRW_2020_paper.pdf)
  - For irregular sampling or dropped frames, cubic interpolation reconstructs most robustly but costs more compute. Linear and filter-based interpolation are cheaper. — [Sensors 2025, PMC11768942](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11768942/)
- A 2025 real-time rPPG acquisition system reports no significant HR accuracy loss at reduced resolution and sampling rate. Exact values were not seen. — [arXiv 2508.18787](https://arxiv.org/pdf/2508.18787)
- An embedded ARM benchmark (Sensors 2023) reports about 96.7% average HR accuracy across platforms, distances and resolutions. It is directly relevant to the Pi. — [Sensors 2023, 23(7):3507](https://www.mdpi.com/1424-8220/23/7/3507/html)

### Inferences
- At 320x240 with a face of about 80-120 px, a face ROI still holds several thousand pixels. Spatial averaging keeps quantisation noise manageable, so resolution alone is probably not the binding constraint. Compression plus low light plus dark skin together probably is.
- At 10 fps:
  - HR up to 300 bpm is theoretically resolvable.
  - FFT bin resolution depends on window length (a 10 s window gives 0.1 Hz = 6 bpm bins). Use longer windows (>=15-20 s), zero-padding, or peak interpolation.
  - Exposure time per frame can be up to 100 ms, which helps low light: a useful synergy for C1 + C5.
  - Deep models trained at 30 fps need retraining or resampling. Upsampling the trace to 30 Hz (cubic) before a trained 1D head is cheap.
- Encoder settings to request or emulate (inference): H.265 or H.264 with a low QP on the face region (ROI-QP), all-intra or short GOP to avoid skip blocks, and 4:2:0 is unavoidable at 500 kbps. Train deep models with on-the-fly H.264/H.265 CRF augmentation at the target bitrate.
- Pi-cost:
  - POS/CHROM/LGI/OMIT over a few patches is essentially free.
  - ICA should be avoided: compute plus numerical fragility under compression.
  - STVEN-style enhancers are too heavy for a Pi. Compression-augmented training of a small network (e.g., TS-CAN or EfficientPhys-sized) is the cheap route.

### Gaps
- No quantitative numbers were found at exactly 10 fps and 500 kbps on 320x240. McDuff et al. (FG 2017) on video compression was not retrieved in this session.
- The specific QP/bitrate breakpoints of the HiDURPA study are in figures and were not extracted.

## C7 Masks (>=40% lower-face occlusion): which ROIs carry the pulse, and what accuracy is achievable?

### Takeaway
Cloth masks increased HR MAE by more than 80% for methods designed on unmasked faces. Training with synthetic mask augmentation pushes models toward the forehead and periocular regions and narrows the gap. A 2025 systematic review (70 studies) finds forehead and cheeks the most accurate ROIs. Multi-patch combination helps, with diminishing returns beyond about 60 patches. Forehead plus upper cheeks plus periocular (below-eye) patches with landmark tracking and SNR weighting is the evidence-aligned, Pi-cheap design.

### Cited Findings
- Speth et al. ("Remote Pulse Estimation in the Presence of Face Masks"):
  - Two public datasets with 86 unmasked and 61 masked subjects.
  - Cloth masks increased HR MAE by more than 80% for methods designed on unmasked faces.
  - Augmenting training videos with synthetic patterned masks forces the model to attend to the periocular and forehead regions and reduces the error gap. — [arXiv 2101.04096](https://arxiv.org/abs/2101.04096)
- Systematic review (Bondarenko et al., npj Digit Med, Jul 2025), 70 studies:
  - Most methods use the whole face, but forehead and cheek ROIs show superior accuracy.
  - Combining multiple patches improves performance, with diminishing returns beyond about 60 patches. — [The role of face regions in rPPG, PMC12297079](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12297079/); [DOAJ record](https://doaj.org/article/7fb67158fac64c7da0a486d35e8187bd)
- A Beijing University of Technology method for partially visible faces (profile or mask) locates the ROI from face and eye landmarks. It reports RMSE <7.64 bpm at 30-45° head deflection. — [BJUT IR](https://ir.bjut.edu.cn/irpui/item/262503)
- arXiv 2410.15851 (2024) proposes head-pose-adaptive ROI selection, switching to the cheek when the forehead is not visible. We did not get accuracy figures. — [arXiv 2410.15851](https://www.arxiv.org/pdf/2410.15851)

### Inferences
- A surgical or cloth mask covers the nose, lower cheeks and chin, leaving the forehead, glabella, periocular area and upper cheeks (malar region above the mask line).
- Practical pipeline:
  1. Run a landmarker (e.g., MediaPipe Face Mesh; the BlazeFace/FaceMesh lite models run on a Pi in real time — engineering knowledge, not sourced here).
  2. Define a grid of small patches above the mask line.
  3. Run POS per patch.
  4. Weight patches by in-band SNR (power at the spectral peak vs the rest of 0.7-4 Hz), or by inter-patch agreement.
  5. Reject patches with hair, eyebrows or glasses glare.
- Mask detection can simply be "skin-colour likelihood below landmark-y threshold". Patches that are not skin produce low SNR and get down-weighted automatically.
- Hair or fringe over the forehead plus a mask can leave very little skin. Recommend a minimum-skin-pixel quality flag and abstain or flag low-confidence output.
- Talking under a mask moves the upper cheeks. The forehead is the most motion-stable region.

### Gaps
- No absolute MAE with vs without surgical masks was extracted (only the "more than 80%" relative increase).
- The patch-level forehead vs cheek vs periocular SNR numbers in the review were not readable (bot-check page).

## C6 Aging skin (60-70+): evidence on rPPG accuracy in older adults

### Takeaway
We found no age-stratified rPPG validation focused on adults aged 60-70+. Benchmark datasets are dominated by young adults (CHILL 18-32; HiDURPA mostly 20-30; a 2025 smartphone pilot with median age 25). This is a genuine evidence gap, and any claims about aging skin must be labelled as hypotheses.

### Cited Findings
- CHILL participants were aged 18-32. The authors list age range as a limitation. — [PMC12678791](https://pmc.ncbi.nlm.nih.gov/articles/PMC12678791/)
- HiDURPA: 8 of 11 participants aged 20-30, none over 50. — [EAH Jena VitaWave](https://www.eah-jena.de/vitawave/publications/videocompressionrppg)
- A 2025 smartphone facial-vPPG pilot (35 volunteers, median age 25) explicitly calls for validation with greater age, skin-tone and lighting diversity. — [Frontiers in Physiology 2025](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1638809/pdf)
- Dasari et al.'s cohorts were older on average (mean age 45.0 in Sierra Leone, 37.0 in India), but results were not stratified by age. — [PMC8175478](https://pmc.ncbi.nlm.nih.gov/articles/PMC8175478/)
- A hospital study of 1,045 pulmonary-function-test patients compared camera HR with ECG and recorded age. We did not see age-subgroup results. — [J Clin Med 2022 (DOAJ)](https://doaj.org/article/1f5427bab9c84dcfb6b61a3c9c8dddb8)
- Contactless HR is motivated by aging populations because wearables are less acceptable to elderly users. — [Sensors 2023, 23(7):3507](https://www.mdpi.com/1424-8220/23/7/3507/html)

### Inferences
- These are physiological hypotheses, not shown for rPPG in the retrieved sources:
  - Reduced dermal perfusion and capillary density and thinner epidermis could lower pulsatile amplitude.
  - Wrinkles create micro-shadows that vary with small motions (motion-artefact sources).
  - Age spots and uneven pigmentation make the skin ROI heterogeneous, much like vitiligo.
  - Arrhythmias (e.g., atrial fibrillation) are more prevalent and break the single-peak spectral assumption.
- Mitigations: the same multi-patch SNR weighting as C2 and C7, longer windows, and a signal-quality index for abstention. Include older subjects in the team's own validation set.

### Gaps
- No quantitative MAE for 60+ subjects was found. The 1,045-patient hospital study may have age data in its full text.
- No studies on wrinkles or perfusion and rPPG SNR were retrieved.

## Motion robustness (talking, head movement) as it interacts with the above

### Takeaway
Motion is consistently the dominant real-world failure source, and it interacts badly with low light, compression and dark skin, all of which shrink the pulse-to-noise margin. Elevated HR also degrades most methods. In-the-wild outliers in Dasari et al. were driven by head or camera motion, lighting changes and extra people.

### Cited Findings
- Dasari et al. removed 9 outlier videos with rapid head or camera movement, changing or poor lighting, or an extra person in frame. Most came from the Sierra Leone recordings, so environmental confounds can masquerade as demographic bias. — [PMC8175478](https://pmc.ncbi.nlm.nih.gov/articles/PMC8175478/)
- Elevated HR significantly degraded 5 of 8 methods in CHILL. Cross-dataset deep models were hit hardest (e.g., PhysNet trained on PURE: -13.55 bpm median difference). — [PMC12678791](https://pmc.ncbi.nlm.nih.gov/articles/PMC12678791/)
- MMPD includes 4 motion conditions (stationary, head rotation, talking, walking) on mobile phones. It is the standard benchmark for combined motion, skin-tone and lighting stress. We did not retrieve its per-condition MAE. — [arXiv 2302.03840](https://arxiv.org/pdf/2302.03840)
- Head-pose-adaptive ROI switching (forehead to cheek) and landmark-based ROI for partial or turned faces. — [arXiv 2410.15851](https://www.arxiv.org/pdf/2410.15851); [BJUT IR](https://ir.bjut.edu.cn/irpui/item/262503)

### Inferences
- POS's projection is designed to suppress specular and intensity (motion) components, and is the cheapest motion-robust baseline.
- Landmark-tracked patches with SNR weighting address both motion and occlusion.
- A quality/abstention flag is important for honest reporting under combined stressors.
- At 10 fps, motion blur and inter-frame displacement are larger, so landmark tracking quality matters more.

### Gaps
- No quantitative MMPD per-motion, per-lighting or per-skin-tone tables were retrieved.
- No study combines all of the challenge's stressors (low light, dark skin, mask, 10 fps, 500 kbps).
