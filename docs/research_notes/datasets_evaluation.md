# Datasets and Evaluation Methodology for Camera-Based Vital Signs (FaceVitals, 2027 IEEE SP Cup)

Research date: 20 Oct 2026. About 40 tool calls. "VERIFIED" means read on the official page or repo this session. "UNVERIFIED" means it comes from background knowledge or secondary sources and must be checked before anyone relies on it. Several official pages (Nature, PMC, PubMed, figshare, UCLA) blocked automated fetches. Details for those come from search snippets and are flagged.

---

## Q1. Dataset catalogue: specs, ground truth, conditions, diversity, access, and licence (MathWorks commercial-use conflict)

### Takeaway
Most established rPPG datasets are academic or research-only. Several state this outright: MMPD, SUMS, iBVP, CHILL, and SCAMPS. Under SCAMPS's R-UDA licence, neither the data nor any *Results* may be used in a commercial offering. These terms can clash with a clause letting MathWorks use submissions commercially. Of the datasets checked, only these appear to allow commercial use:
- **MCD-rPPG**: CC BY 4.0, 600 subjects, with SpO2 and BP.
- **VitalVideos**: a separate commercial licence, negotiated by email.
- **M3PD**: possibly. The arXiv page shows a CC BY 4.0 icon, but the dataset's own terms are unverified.

The team should clear training-data provenance with the organisers. NC data is safer used only for benchmarking than for training the submitted model.

### Cited Findings

**MCD-rPPG ("Gaze into the Heart", ACM MM 2025 Datasets track; HF `kyegorov/mcd_rppg`)**: VERIFIED
- 600 subjects and 3,600 synchronised recordings (600 subjects × 2 states × 3 cameras). Each measurement is a 3-minute video. — [HF dataset card](https://huggingface.co/datasets/kyegorov/mcd_rppg)
- Cameras are a frontal webcam, a FullHD camcorder, and a mobile phone camera. Resolution, fps, and codec are not stated on the card. — [HF card](https://huggingface.co/datasets/kyegorov/mcd_rppg)
- Ground truth: PPG at 100 Hz, synchronised ECG, and 13 biomarkers. These include systolic/diastolic BP, SpO2, temperature, glucose, HbA1c, cholesterol, respiratory rate, arterial stiffness, stress (PSM-25), age, sex, and BMI. — [HF card](https://huggingface.co/datasets/kyegorov/mcd_rppg)
- Conditions are rest and post-exercise (15 squats in 30 s). The card says it "may not cover all real-world scenarios (e.g., different lighting conditions, motion artifacts)." Data was "collected primarily in Russia." No skin-tone breakdown is given. — [HF card](https://huggingface.co/datasets/kyegorov/mcd_rppg)
- **Licence: CC BY 4.0.** It is public on the HF Hub with no approval step. — [HF card](https://huggingface.co/datasets/kyegorov/mcd_rppg); [arXiv 2508.17924](https://arxiv.org/pdf/2508.17924)
- Mirrors and forks exist, e.g. `wengziheng/mcd_rppg`, `surui7-7/mcd_rppg`, `Bgeorge/mcd_rppg`. Use the original kyegorov repo. — [search results](https://huggingface.co/datasets/wengziheng/mcd_rppg)

**VitalVideos (Europe-1, Africa-1, WorldWide; also SpO2 and Longitudinal sets)**: VERIFIED from vitalvideos.org
- **Europe-1** has 850 participants:
  - Video: 30 s, 30 fps, raw, Basler acA1920-40, 1920×1080, uncompressed MP4.
  - Ground truth: PPG from a CMS50D+ at about 60 Hz, HR/SpO2 at 1 Hz, BP from an Omron M7 during recording. No RR waveform and no ECG.
  - Demographics: 50/50 sex, mean age 42, 40% over 50.
  - Fitzpatrick I–VI: 10/35/20/10/7.5/7.5%.
  - — [vitalvideos.org](https://www.vitalvideos.org/)
- **Africa-1** has 1,550 participants:
  - Video: 24–30 s at 40/60 fps, raw. Includes face-only and face+palm video and a "variety of lighting scenario's".
  - Ground truth: HR, SpO2, BP, and RR are listed.
  - — [vitalvideos.org](https://www.vitalvideos.org/)
- **WorldWide** has about 6,500+ participants:
  - Video: 60 s at 60 fps, multicamera. A Basler acA1440-220 records 960×1056 uncompressed MP4, and an iPhone 15 Pro records 1920×1080 slightly compressed MOV.
  - Ground truth: CMS50D+ PPG plus a pulse oximeter at 500 Hz, HR/SpO2 at 1 Hz, BP from a MicroLife B3 before and after recording, a 500 Hz RR waveform, and 3-lead ECG at 500 Hz.
  - Demographics: about 42% male, mean age about 45, 38% over 50, 18% with systolic BP above 160.
  - Fitzpatrick I–VI: 13/13/11/20/23/20%.
  - Consent and IRB review are stated.
  - — [vitalvideos.org](https://www.vitalvideos.org/)
- **Access:**
  - Commercial R&D: "please reach out to pj@vitalvideos.org". No terms or pricing are published.
  - Academic subsets: a signed "Data Transfer and Licensing Agreement - ACADEMIC.pdf", sent from an academic email with the lead researcher's Google Scholar link. — [vitalvideos.org](https://www.vitalvideos.org/)
  - A co-founder's LinkedIn post says academia gets free licences for curated 300- and 900-participant subsets. The official page does not confirm that these are free. — [LinkedIn (P-J Toye)](https://be.linkedin.com/in/pjtoye)
  - The older Europe documentation offered a free subset to non-profits that publish, and a fee for profit-making use. — [arXiv 2306.11891](https://arxiv.org/pdf/2306.11891)
- The VitalVideo benchmark paper (Health-HCI group) used 893 subjects aged 5–95, recorded at 10 indoor locations in Western Europe:
  - Camera: Basler acA1920-40uc, 1920×1200, 30 fps, at 50–70 cm.
  - Lighting: **at least 150 lux**, with a ring light when needed. So it has **no low-light content**.
  - Clips: two 30 s clips per subject.
  - Ground truth: CMS50D+ (PPG/HR/SpO2) and Omron M7.
  - — [arXiv 2404.05003](https://arxiv.org/html/2404.05003v1)

**MMPD (Multi-domain Mobile Video Physiology Dataset)**: VERIFIED from GitHub
- 33 subjects and about 11 h of mobile video (about 1,152K frames). The released full set is 320×240 (370 GB) and the mini set is 80×60 (48 GB). — [GitHub](https://github.com/McJackTang/MMPD_rPPG_dataset)
- Secondary sources describe 660 one-minute videos from a Samsung Galaxy S22 Ultra, with PPG from a pulse oximeter. — [arXiv 2404.05003](https://arxiv.org/html/2404.05003v1); [MMPD paper](https://arxiv.org/pdf/2302.03840)
- Labels:
  - Lighting: LED-low, LED-high, Incandescent, Nature.
  - Motion: Stationary, Rotation, Talking, Walking.
  - Exercise: yes/no.
  - Also skin tone 3–6, gender, glasses, hair cover, and makeup.
  - Ground truth is a PPG waveform only. No SpO2 is listed.
  - — [GitHub](https://github.com/McJackTang/MMPD_rPPG_dataset)
- **Licence/access:** "Any commercial usage is banned." A **faculty member** (not a student) must sign the Data Release Agreement and email it from an institutional address to tjk24@mails.tsinghua.edu.cn, cc yuntaowang@tsinghua.edu.cn. The repo code is MIT, which does not cover the data. — [GitHub](https://github.com/McJackTang/MMPD_rPPG_dataset)

**SUMS / Summit Vitals (high altitude, SpO2)**: VERIFIED from GitHub
- 10 subjects on the Qinghai Plateau (2,274–4,898 m) and 80 synchronised face + finger videos. Two Logitech C922 cameras record 1280×720 at 60 fps. — [GitHub thuhci/SUMS](https://github.com/thuhci/SUMS)
- Ground truth: PPG and SpO2 from a CMS50E+ (PPG at 20 Hz), and RR from an HKH-11C sensor at 50 Hz. — [GitHub](https://github.com/thuhci/SUMS)
- Protocol: rest for 3 min, climb stairs, then 2 min post-exercise. The afternoon set adds portable oxygen, which gives natural **SpO2 variation**, rare among rPPG datasets. — [GitHub](https://github.com/thuhci/SUMS); [arXiv 2409.19223](https://arxiv.org/pdf/2409.19223)
- **Licence:** "This dataset is built for academic use. Any commercial usage is banned." A faculty member must request it. — [GitHub](https://github.com/thuhci/SUMS)

**iBVP (RGB-thermal)**: VERIFIED from GitHub
- 124 sessions of about 3 min each, about 372 min in total. Four conditions: (A) slow breathing/rest, (B) easy math, (C) difficult math, (D) guided head movement. — [GitHub](https://github.com/PhysiologicAILab/iBVP-Dataset)
- Cameras: a Logitech BRIO records RGB at 640×480, 30 fps (lossless .bmp), and a FLIR A65SC records thermal at 640×512, 30 fps. — [GitHub](https://github.com/PhysiologicAILab/iBVP-Dataset)
- Ground truth: ear PPG (BVP) downsampled to 30 fps, manual signal-quality labels SQ1/SQ2, and a perfusion index. Some participants are in a "Confidential_No-media-use" folder. — [GitHub](https://github.com/PhysiologicAILab/iBVP-Dataset)
- **Licence:** released "only for academic research purposes". **Academic supervisors** email a signed EULA. — [GitHub](https://github.com/PhysiologicAILab/iBVP-Dataset)

**SCAMPS (synthetic)**: VERIFIED from GitHub and LICENSE
- 2,800 synthetic videos and 1.68M frames (about 600 frames per video). Raw frames are 320×240, and face crops and masks are 240×240. — [GitHub](https://github.com/danmcduff/scampsdataset)
- Ground truth: frame-aligned PPG and breathing waveforms, HR, BR, IBI, HRV, head pose, action-unit intensities, and segmentation masks. The full set is about 600 GB, and an official train/val/test split is provided. — [GitHub](https://github.com/danmcduff/scampsdataset)
- **Licence: "Research Use of Data Agreement v1.0" (R-UDA).** Use is limited to non-commercial research, and "you may not use the Data or any Results in any commercial offering, including as part of a product or service." **This directly conflicts with commercial transfer of a model trained on SCAMPS.** — [SCAMPS LICENSE.txt](https://raw.githubusercontent.com/danmcduff/scampsdataset/main/LICENSE.txt)

**UBFC-rPPG**: VERIFIED from the official site
- Dataset 1 ("simple") has 8 videos. Dataset 2 ("realistic") recorded 46 videos and shares 42 "for research purpose". — [UBFC-rPPG site](https://sites.google.com/view/ybenezeth/ubfcrppg)
- Recording: a Logitech C920 at 640×480, 30 fps, **uncompressed 8-bit RGB**. Ground truth is PPG waveform and HR from a CMS50E transmissive pulse oximeter. Lighting is indoor with varying sunlight. Subjects sat about 1 m from the camera and played a math game to raise HR. — [UBFC-rPPG site](https://sites.google.com/view/ybenezeth/ubfcrppg)
- **Licence:** none stated beyond "shared for research purpose" and a citation request (Bobbia et al., PRL 2017). Access is an open Google Drive download. Commercial use is unclear and should be treated as research-only unless confirmed. — [UBFC-rPPG site](https://sites.google.com/view/ybenezeth/ubfcrppg)
- A 2025 field study reports UBFC-rPPG at about 5% dark-skinned subjects, MMSE-HR at about 10%, and AFRL at 0%. — [medRxiv 2025 Nigeria study](https://www.medrxiv.org/content/10.1101/2025.10.13.25337878.full.pdf)

**COHFACE (Idiap)**: partially VERIFIED
- 40 subjects (12 F, 28 M) and 160 one-minute videos. A Logitech HD C525 records 640×480 at 20 Hz. Ground truth is BVP and breathing from Thought Technology sensors (BioGraph Infiniti v5). — [Idiap COHFACE page](https://www.idiap.ch/en/scientific-research/data/cohface); [Idiap dataset page](https://idiap.ch/dataset/cohface)
- Access is via a Zenodo "Get data" link, and the page states no licence terms. The bob.db.cohface wrapper is BSD-3 (code only). — [Idiap](https://www.idiap.ch/en/scientific-research/data/cohface); [gitlab bob.db.cohface](https://gitlab.idiap.ch/bob/bob.db.cohface)

**UCLA-rPPG (real) and the UCLA synthetic avatars**: partially VERIFIED via search snippets (the official site failed TLS)
- 104 subjects were recorded and 102 kept, with Fitzpatrick types 1–6 and varied age, gender, and ethnicity. Each has 5 videos of about 1 min, 503 videos after cleaning. — [Diverse rPPG / UCLA (search snippet)](https://visual.ee.ucla.edu/diverse_rppg.htm); [CVPR 2022 paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Synthetic_Generation_of_Face_Videos_With_Plethysmograph_Physiology_CVPR_2022_paper.pdf)
- **Access:** "restricted due to IRB protocol". A Data Request Form is required. No licence text was found. — [UCLA rppg_avatars page (snippet)](https://visual.ee.ucla.edu/rppg_avatars.htm)

**MSLR (Multi-Scene Low-Light rPPG)**: VERIFIED from GitHub
- 42 adults aged 18–40 (32 M, 10 F) and 504 videos of 65 s each, at 640×480 and 30 fps. — [GitHub wwenmaositu/MSLR](https://github.com/wwenmaositu/MSLR)
- **Illumination is 3.0, 8.0, and 35.0 lux, plus NIR synchronised with the 3 lux condition.** Motion conditions are stable, talking, and head movement. Ground truth is in `sN_wave.csv`, with contents not described. — [GitHub](https://github.com/wwenmaositu/MSLR)
- Published in *Machine Vision and Applications* (Springer, 2025) with the LPPE enhancement method. — [Springer](https://link.springer.com/article/10.1007/s00138-025-01719-3)
- **Access:** a public OneDrive link. The repo lists an MIT licence, and it is unclear whether that covers the data. Commercial use is ambiguous, so ask the authors.

**CHILL (Challenging Heartrate and Illumination)**: VERIFIED from arXiv HTML and Zenodo
- 50 participants recruited and 45 analysed (27 F, 17 M as given in the table). **Fitzpatrick 1–3 only.** Age is not reported. — [arXiv 2503.11697 HTML](https://arxiv.org/html/2503.11697)
- Recording: a Canon EOS 550D DSLR at 1920×1080 and 25 fps. Four one-minute scenarios: LowHR-Bright, LowHR-Dark, HighHR-Dark, HighHR-Bright.
  - Lighting is set by LED output (50 vs 5 on a 0–50 scale), **with no lux values**. Mean pixel value is 129.7 bright vs 33.6 dark.
  - Ground truth: Biosignalsplux finger PPG plus 3-lead ECG at 1000 Hz, synchronised by timestamps. HR range is 54–141 bpm.
  - — [arXiv HTML](https://arxiv.org/html/2503.11697)
- Finding: high HR degraded 5 of 8 methods significantly, while low light had a smaller effect. — [npj Digit Med 2025 (search summary)](https://www.nature.com/articles/s41746-025-02192-y)
- **Access:** Zenodo DOI 10.5281/zenodo.14637544. Files are restricted and require an academic affiliation plus a signed EULA emailed to bhargav.acharya@uni-bielefeld.de. Research use only.
  - **The Zenodo record says 23 participants, with faces cropped to 36×36 via rPPG-Toolbox. The paper says frames were downsampled to 128×128 for consenting participants.** These conflict. — [Zenodo](https://zenodo.org/records/14637544); [arXiv HTML](https://arxiv.org/html/2503.11697)

**MPU-rPPG (Macao Polytechnic University; Sci Data 13:1216, May 2026)**: search snippets only (official pages blocked)
- Synchronised RGB video with ground-truth PPG, HR, and SpO2, plus region masks (face, arms, hands, legs) and task annotations. HR ranges from 50 to 160 bpm across rest and high-intensity activity. — [Nature Sci Data (snippet)](https://www.nature.com/articles/s41597-026-07310-3)
- Recording: an MX Brio camera at 60 fps and a CMS60D oximeter. Lighting is indoor and natural. Ethnicity is listed as Asian, African, and European. Snippets say about 20 h total and 15 min per subject. Technical validation used 50 participants (40 train, 10 test). — [Sci Data snippet](https://www.nature.com/articles/s41597-026-07310-3); [PMC13500808](https://pmc.ncbi.nlm.nih.gov/articles/PMC13500808/)
- Metadata includes age, gender, **Fitzpatrick skin tone**, lighting, and activity. — [PubMed 42209535 (snippet)](https://pubmed.ncbi.nlm.nih.gov/42209535/)
- **Access:** a "MPU-rPPG Sample Dataset" on figshare (doi 10.6084/m9.figshare.29377835), which returned 403. It may be only a sample. Licence unverified. — [figshare](https://figshare.com/articles/dataset/MPU-rPPG_Sample_Dataset/29377835)

**M3PD (dual-view smartphone, clinical)**: VERIFIED from the arXiv abstract
- 60 participants including **47 cardiovascular patients**. Facial and fingertip videos were captured simultaneously by the front and rear smartphone cameras in lab and clinical settings. Data and code are at github.com/Health-HCI-Group/F3Mamba. The arXiv page shows a CC BY 4.0 icon, which may apply to the paper rather than the data. — [arXiv 2511.02349](https://arxiv.org/abs/2511.02349)
- A secondary review says the files include HR, RR, and SpO2 labels and that access is "on-request" for patient privacy. — [ResearchGate](https://www.researchgate.net/publication/397279949_M3PD_Dataset_Dual-view_Photoplethysmography_PPG_Using_Front-and-rear_Cameras_of_Smartphones_in_Lab_and_Clinical_Settings); [Moonlight review](https://www.themoonlight.io/en/review/m3pd-dataset-dual-view-photoplethysmography-ppg-using-front-and-rear-cameras-of-smartphones-in-lab-and-clinical-settings)

**CAST-Phys (IEEE DataPort)**: VERIFIED via DataPort and arXiv snippets
- 60 participants and 1,080 facial videos (18 emotional stimuli each). High-resolution **uncompressed** facial video, plus contact PPG, EDA, and respiration. — [IEEE DataPort CAST-Phys](https://ieee-dataport.org/documents/cast-phys); [arXiv 2507.06080](https://arxiv.org/pdf/2507.06080)
- **Access:** DataPort DOI 10.21227/bc86-aa67, under a signed EULA for academic research. — [search summary of DataPort](https://ieee-dataport.org/documents/cast-phys)

**DDMP-FV (IEEE DataPort)**: VERIFIED
- 20 subjects. States are rest, breath-hold rest, and locomotion rest, each recorded twice for 30 s (120 sets). Camera, resolution, and fps are not specified. — [IEEE DataPort DDMP-FV](https://ieee-dataport.org/documents/ddmp-fv)
- Ground truth: HR and BP from a Bioland arm sphygmomanometer, HR and SpO2 from a CONTEC CMS50DL, and RR recorded manually. — [DataPort](https://ieee-dataport.org/documents/ddmp-fv)
- **Access:** "This dataset requires an IEEE DataPort Subscription." No licence is stated, and **the page said "Files have not been uploaded for this dataset."** DOI 10.21227/00tq-3495. — [DataPort](https://ieee-dataport.org/documents/ddmp-fv)

**MPSC-rPPG (IEEE DataPort, UMBC)**: partially VERIFIED
- Facial videos with synchronised BVP, for HR and HRV. Authors: Z. Hasan, S. Ramasamy Ramamurthy, N. Roy (UMBC). DOI 10.21227/ddgz-tx88. Subject count, specs, and licence were not retrieved. — [IEEE DataPort](https://ieee-dataport.org/documents/mpsc-rppg-dataset); [DataPort keyword list](https://ieee-dataport.org/keywords/remote-photoplethysmography-rppg)

**Other notable items found**
- **Simulated_Multispectral_rPPG** (DataPort) reconstructs UBFC-rPPG into 31 bands (400–700 nm) with CMS50E PPG/HR. — [DataPort](https://ieee-dataport.org/keywords/remote-photoplethysmography-rppg)
- **EMPD** (2025) has 83 subjects with event + RGB cameras and a pulse oximeter, at rest and during exercise. Seen only as a table fragment. — [arXiv 2603.26699](https://arxiv.org/pdf/2603.26699)
- **PhysDrive** (in-vehicle) and **LADH** (daily personal care) are supported by rPPG-Toolbox. — [rPPG-Toolbox README](https://github.com/ubicomplab/rPPG-Toolbox)
- **Nigeria multi-site field study** (medRxiv, Oct 2025): rPPG for hypertension screening in darker-skinned populations. Relevant evidence for Fitzpatrick V–VI and BP. — [medRxiv](https://www.medrxiv.org/content/10.1101/2025.10.13.25337878.full.pdf)
- **Bias audit** (Bondarenko, Menon, Elgendi, npj Digit Med, Oct 2025): a review of 100 rPPG studies found most public datasets demographically biased, with dark skin under-represented. The full text was captcha-blocked. — [PMC12491395](https://pmc.ncbi.nlm.nih.gov/articles/PMC12491395/); [DOAJ](https://doaj.org/article/d5ce85512e22454bbb8b485ebd378267)

### Inferences
- **Licence risk tiers for a team whose submission may go to MathWorks for commercial use:**
  - *Low risk (commercial use allowed):* MCD-rPPG (CC BY 4.0, attribution only), the team's own IRB-approved data with commercial-use consent, and possibly the organiser's Sri Ramachandra dataset (depends on the organiser's terms).
  - *Negotiable:* VitalVideos, which offers a paid commercial R&D licence. The academic subset is probably non-commercial.
  - *Explicitly non-commercial or academic-only:* MMPD, SUMS, iBVP, CHILL (research EULA), and SCAMPS (R-UDA bans commercial use of *Results*). CAST-Phys has an academic EULA.
  - *Ambiguous:* UBFC-rPPG ("research purpose"), MSLR (MIT on the repo), COHFACE, M3PD, and MPU-rPPG.
- Practical strategy: train the submitted model on CC BY, own, and organiser data. Use NC datasets only to report benchmark numbers, as allowed by their EULAs, and say so in the report. Ask the SP Cup organisers in writing whether models trained on NC data are acceptable given the MathWorks clause.
- **Student-access blocker:** MMPD, SUMS, and iBVP require a faculty member or supervisor to sign. Allow 1–4 weeks for approval.
- **Best SpO2 sources:** VitalVideos (CMS50D+ 1 Hz), MCD-rPPG, SUMS (hypoxia gives real SpO2 range), MPU-rPPG, M3PD (patients), and DDMP-FV (if files ever appear).
- **Best BP sources:** VitalVideos (Omron/MicroLife, many over 160 systolic), MCD-rPPG, and DDMP-FV. Beat-to-beat BP waveform is only in BP4D+/MMSE-HR (unverified, see Gaps).

### Gaps
The following are UNVERIFIED this session and come from background knowledge. Confirm each on the official page.
- **PURE (TU Ilmenau):**
  - Content: about 10 subjects and 60 one-minute sequences across 6 head-motion setups, 640×480 at 30 fps, stored as lossless PNG. Ground truth is a CMS50E with PPG and SpO2 at 60 Hz.
  - Access: by request from the authors, research use.
  - rPPG-Toolbox confirms "raw data requested from the paper authors" ([README](https://github.com/ubicomplab/rPPG-Toolbox)), and the VitalVideo paper confirms 10 subjects ([arXiv 2404.05003](https://arxiv.org/html/2404.05003v1)).
- **VIPL-HR (ICT CAS):**
  - Content: 107 subjects and about 2,378 visible plus 752 NIR videos from 3 devices (webcam, RealSense, Huawei phone). Nine scenarios include head motion, talking, and **dark/dim lighting**.
  - Ground truth: HR, SpO2, and BVP from a CONTEC CMS60C.
  - Access: release agreement signed by a permanent staff member, academic only. Not verified this session.
- **MAHNOB-HCI:**
  - Content: 27 subjects, emotion elicitation, about 780×580 at 61 fps, **heavily H.264-compressed**. Ground truth is ECG.
  - Access: EULA via the iBUG/Imperial website, academic only. Not verified.
- **MMSE-HR / BP4D+ (Binghamton):**
  - Content: BP4D+ has 140 subjects, and MMSE-HR is a 40-subject subset. Video is 1040×1392 at 25 fps. Biopac continuous BP waveform and HR are recorded.
  - Access: the rPPG-Toolbox says to "email the authors" ([README](https://github.com/ubicomplab/rPPG-Toolbox)). The licence agreement is understood to be non-commercial academic, with a separate commercial licence. No Binghamton licence page was found.
- **UBFC-Phys:**
  - Content: 56 subjects, three tasks (rest, speech, arithmetic), about 1024×1024 at 35 fps. Ground truth is BVP and EDA from an Empatica E4.
  - Access: via the Benezeth Google site. Licence not checked.
- **COHFACE:** compression (believed MPEG-4, heavily compressed) and the two lighting conditions (studio vs natural) are not confirmed. Licence terms sit on the Zenodo record and were not read.
- **NCTUBO** and **AiPEX "rppg_biases"**: no sources found under these names. They may be internal or alternative names. Ask whoever proposed them.
- **Masks and multi-person:** no public rPPG dataset with **face masks** or **multiple people in one frame** was found. MCD-rPPG is multi-*view*, not multi-person.
- The figshare and PMC pages for MPU-rPPG were blocked, so its exact counts, Fitzpatrick distribution, age, and licence are unverified. The VitalVideos academic agreement PDF was not read.

---

## Q2. Which datasets support which challenge constraints (C1–C7)?

### Takeaway
Low light (C1) is covered by MSLR (3/8/35 lux, verified), CHILL (dark LED, no lux values), and MMPD (LED-low). Skin tone (C2) is covered best by VitalVideos-WorldWide, UCLA-rPPG, and MMPD (types 3–6). Older adults (C5) are covered by VitalVideos (38–40% over 50, ages 5–95) and M3PD (patients). Low resolution, low fps, and low bitrate (C3/C4) are best produced by synthetic degradation of uncompressed sources. **No verified public dataset covers masks (C6) or multi-person scenes (C7).** These must come from self-collection, the organiser dataset, or synthetic compositing.

### Cited Findings
- **C1 low light (≤50 lux):**
  - MSLR records 3.0, 8.0, and 35.0 lux plus NIR, with motion. — [GitHub MSLR](https://github.com/wwenmaositu/MSLR)
  - CHILL has bright and dark conditions but reports pixel means (129.7 vs 33.6), not lux. — [arXiv HTML](https://arxiv.org/html/2503.11697)
  - MMPD has an "LED-low" label. — [GitHub](https://github.com/McJackTang/MMPD_rPPG_dataset)
  - A 2025 paper studies dynamic lighting at night. — [arXiv 2507.04306](https://arxiv.org/pdf/2507.04306)
  - VitalVideo used at least 150 lux, so it is not suitable for C1. — [arXiv 2404.05003](https://arxiv.org/html/2404.05003v1)
- **C2 Fitzpatrick I–VI:**
  - VitalVideos-WorldWide is 13/13/11/20/23/20%, and Europe-1 is 10/35/20/10/7.5/7.5%. — [vitalvideos.org](https://www.vitalvideos.org/)
  - UCLA-rPPG covers FP 1–6 with 102 subjects. — [UCLA (snippet)](https://visual.ee.ucla.edu/diverse_rppg.htm)
  - MMPD covers types 3–6. — [GitHub](https://github.com/McJackTang/MMPD_rPPG_dataset)
  - MPU-rPPG includes Asian, African, and European participants with Fitzpatrick metadata. — [Sci Data snippet](https://www.nature.com/articles/s41597-026-07310-3)
  - CHILL is FP 1–3 only. — [arXiv HTML](https://arxiv.org/html/2503.11697)
  - Error rises sharply with darker skin. Models trained on PURE, UBFC, or MMPD and tested on VV100 gave PhysNet MAE of about 0–1.6 bpm for FP1–2 vs 14.65 for FP6 (PURE-trained), and PhysFormer gave 17.23 for FP6. — [arXiv 2404.05003](https://arxiv.org/html/2404.05003v1)
- **C3 low resolution:**
  - MMPD's released full set is 320×240 and its mini set 80×60. — [GitHub](https://github.com/McJackTang/MMPD_rPPG_dataset)
  - SCAMPS frames are 320×240. — [GitHub](https://github.com/danmcduff/scampsdataset)
  - The public CHILL release is 36×36 or 128×128 crops. — [Zenodo](https://zenodo.org/records/14637544)
- **C4 ≤10 fps or ≤500 kbps:**
  - No native dataset was found. Uncompressed or raw sources that can be re-encoded in a controlled way include UBFC-rPPG (uncompressed), VitalVideos (raw/uncompressed MP4), iBVP (bmp), and CAST-Phys (uncompressed). — [UBFC](https://sites.google.com/view/ybenezeth/ubfcrppg); [vitalvideos.org](https://www.vitalvideos.org/); [iBVP](https://github.com/PhysiologicAILab/iBVP-Dataset); [CAST-Phys arXiv](https://arxiv.org/pdf/2507.06080)
  - COHFACE is natively 20 fps. — [Idiap](https://www.idiap.ch/en/scientific-research/data/cohface)
  - A 2026 paper analyses codec robustness for patch-based rPPG. — [arXiv 2606.04198](https://arxiv.org/pdf/2606.04198)
- **C5 age 60+:**
  - VitalVideos Europe-1 has 40% over 50 (mean 42), and WorldWide has 38% over 50 (mean 45). — [vitalvideos.org](https://www.vitalvideos.org/)
  - The VitalVideo paper covers ages 5–95. — [arXiv 2404.05003](https://arxiv.org/html/2404.05003v1)
  - M3PD has 47 cardiovascular patients. — [arXiv 2511.02349](https://arxiv.org/abs/2511.02349)
  - MSLR is ages 18–40 only. — [GitHub](https://github.com/wwenmaositu/MSLR)
- **C6 masks:** no dataset found (see Gaps).
- **C7 multi-person:** no dataset found. MCD-rPPG is multi-view, not multi-person. — [arXiv 2508.17924](https://arxiv.org/pdf/2508.17924)

### Inferences
Coverage matrix (✓ native, ~ partial or derivable, ✗ none):

| Dataset | C1 low light | C2 FP I–VI | C3 low res | C4 low fps/kbps | C5 60+ | C6 mask | C7 multi | SpO2 | BP | Commercial OK? |
|---|---|---|---|---|---|---|---|---|---|---|
| MSLR | ✓ (3/8/35 lux) | ✗ (unreported) | ~ | ~ (re-encode) | ✗ | ✗ | ✗ | ✗ | ✗ | Unclear (MIT repo) |
| CHILL | ~ (no lux) | ✗ (FP1–3) | ✓ (crops) | ✗ | ✗ (not reported) | ✗ | ✗ | ✗ | ✗ | No (research EULA) |
| MMPD | ~ (LED-low) | ~ (3–6) | ✓ | ~ (mobile compressed) | ✗ | ✗ | ✗ | ✗ | ✗ | **No** |
| VitalVideos-WW | ✗ (≥150 lux in VV) | ✓ | ~ | ~ (raw → re-encode) | ✓ | ✗ | ✗ | ✓ | ✓ | Paid licence |
| UCLA-rPPG | ✗ | ✓ | ~ | ~ | ~ (unverified) | ✗ | ✗ | ✗ (unverified) | ✗ | Unknown (IRB form) |
| MCD-rPPG | ✗ | ✗ (mostly Russia) | ~ | ~ | ~ (age provided) | ✗ | ✗ (multi-view) | ✓ | ✓ | **Yes (CC BY)** |
| SUMS | ✗ | ✗ | ~ | ~ | ✗ | ✗ | ✗ | ✓ (hypoxia) | ✗ | **No** |
| MPU-rPPG | ~ (natural) | ✓ (Fitz. metadata) | ~ | ~ | ? | ✗ | ✗ | ✓ | ✗ | Unverified |
| M3PD | ✗ | ? | ~ | ~ (phone) | ✓ (patients, likely) | ✗ | ✗ | ✓ (per review) | ? | Unverified |
| UBFC-rPPG | ✗ | ✗ (~5% dark) | ~ | ~ (uncompressed → re-encode) | ✗ | ✗ | ✗ | ✗ | ✗ | Unclear |
| iBVP | ✗ | ? | ~ | ~ | ✗ | ✗ | ✗ | ✗ | ✗ | **No** |
| SCAMPS | ~ (synthetic lighting) | ~ | ✓ | ~ | ✗ | ✗ | ✗ | ✗ | ✗ | **No (R-UDA)** |

- C3 and C4 are best handled by a **documented degradation pipeline** applied to uncompressed test videos: downscale to e.g. 160×120 and 80×60, drop to 10 fps or less, and re-encode H.264 at 500 kbps or less with ffmpeg `-b:v 500k -maxrate 500k -bufsize 1000k`. Report the original vs degraded deltas. The ground truth stays valid because only the video changes.
- C6 can be approximated by synthetic mask overlays on lower-face ROIs, but this cannot replicate real mask effects (breath condensation, fit, shadowing). Collect real masked recordings.
- C7 can be approximated by tiling or compositing single-subject videos and running a multi-face detector and tracker. Real multi-person footage with per-person references (one pulse oximeter per person) is needed for credible results.

### Gaps
- No verified public dataset with real face masks or multiple simultaneous subjects with per-subject ground truth.
- No dataset found that reports measured lux for every clip except MSLR. CHILL reports only LED settings and pixel means.
- The organiser dataset (Sri Ramachandra hospital, India) was not examined. It probably supplies Indian Fitzpatrick IV–VI and older or clinical subjects, which would fill C2 and C5. Licence terms are unknown.

---

## Q3. Evaluation pitfalls and the protocol top papers and rPPG-Toolbox use

### Takeaway
Use subject-disjoint splits, both intra-dataset and cross-dataset results, and fixed, reported HR windows. Derive HR from the reference PPG using the same window and estimator as the prediction. Report MAE, RMSE, and Pearson r per stratum and overall, together with a "predict the training mean" baseline. Use a repeated-measures Bland-Altman. For SpO2 and BP, compare against the mean-predictor baseline and clinical standards, because a narrow target range makes small MAE values meaningless.

### Cited Findings
- **rPPG-Toolbox protocol** (de facto standard):
  - Metrics: MAE, RMSE, MAPE, Pearson, SNR, and Bland-Altman ("BA") plots.
  - Data split: `BEGIN`/`END` fractions, e.g. the first 80% of subjects for training and the last 20% for validation, with no subject overlap.
  - Cross-dataset: train on PURE or SCAMPS, validate on a held-out part, test on UBFC-rPPG, and pick the best checkpoint by lowest validation loss.
  - `USE_SMALLER_WINDOW` allows a test window shorter than the video, and `CHUNK_LENGTH` is set in frames.
  - Recommended training sets are UBFC-rPPG, PURE, iBVP, and SCAMPS, chosen for synchronisation quality and volume.
  - Code licence is labelled "Responsible AI" (licenses.ai).
  - — [rPPG-Toolbox README](https://github.com/ubicomplab/rPPG-Toolbox)
- **Subject leakage:** the npj rPPG roadmap (2026) warns that splitting at segment level "is highly susceptible to data leakage". Subject-specific features appear in both train and test, so MAE is overstated. — [npj Digit Med roadmap 2026](https://www.nature.com/articles/s41746-026-02715-1)
  - A Frontiers 2026 paper recommends LOSO or subject-excluded stratified k-fold. — [Frontiers Digital Health 2026](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1855425/full)
- **Window length:** published HR MAEs use different windows, which makes comparisons unreliable. One paper reports 15 s, 30 s, and full-video windows for that reason. — [arXiv 2012.15846](https://arxiv.org/pdf/2012.15846)
  - Very short windows (2 s) make even the reference HR unreliable, so one study uses a 10 s reference. — [arXiv 2506.22078](https://arxiv.org/pdf/2506.22078)
- **HR estimator and baseline:** CHILL estimates HR with Welch PSD, taking the frequency of peak power. It reports MAE ± standard error against a **dummy estimator that predicts the training-set mean**. — [CHILL arXiv HTML](https://arxiv.org/html/2503.11697)
- **Cross-dataset and per-stratum reporting:** the VitalVideo paper trains on PURE, UBFC, and MMPD and tests per Fitzpatrick type on VV100. It reports MAE, RMSE, MAPE, and Pearson, and the errors differ greatly by stratum (see Q2). — [arXiv 2404.05003](https://arxiv.org/html/2404.05003v1)
- Dataset bias makes pooled metrics misleading. Most public datasets under-represent dark skin. — [Bondarenko et al. 2025 (DOAJ)](https://doaj.org/article/d5ce85512e22454bbb8b485ebd378267)
- **Reference sensor sampling and sync:**
  - UBFC uses CMS50E. — [UBFC site](https://sites.google.com/view/ybenezeth/ubfcrppg)
  - SUMS samples PPG at 20 Hz. — [SUMS](https://github.com/thuhci/SUMS)
  - CHILL samples at 1000 Hz with timestamp sync. — [CHILL](https://arxiv.org/html/2503.11697)
  - iBVP downsamples BVP to 30 fps. — [iBVP](https://github.com/PhysiologicAILab/iBVP-Dataset)
  - These differences matter for HRV and peak-based HR.

### Inferences
Recommended protocol for the FaceVitals report:
1. **Splits.** Use subject-disjoint 5-fold CV or LOSO within each dataset. Run at least two cross-dataset directions, e.g. train MCD-rPPG + own data → test UBFC/PURE/MSLR/organiser. Keep any test-time hyperparameter tuning off the test subjects.
2. **Windows.** Report HR for **10 s windows with a 1 s step** (real-time use) and **30 s windows**, as well as whole-clip results for comparison with rPPG-Toolbox tables. State the estimator: FFT/Welch peak within 0.7–3.0 Hz (42–180 bpm), with zero-padding to under 0.1 bpm resolution. Exercise data such as CHILL and SUMS reach about 141–160 bpm, so the band must extend to about 3.5 Hz.
3. **Ground-truth HR.** Compute it from the reference PPG with the same window and spectral method. Optionally cross-check with peak detection, and drop windows whose reference SQI is low. Do not use the oximeter's on-screen 1 Hz HR, which is averaged and lagged, unless it is the only reference. If so, say so.
4. **Sync offsets.** Estimate the lag per recording by cross-correlating the predicted rPPG with the reference PPG (search ±2 s). Report whether lag correction was applied, and apply the same correction in every condition. PPG at the finger lags the face by pulse transit time (about 0.1–0.3 s), which matters for waveform metrics but little for windowed HR.
5. **Metrics.**
   - Report MAE, RMSE, and Pearson r with 95% CIs (bootstrap over subjects, not windows), plus MAPE and SNR for HR.
   - Use **Bland-Altman with repeated measures** (multiple windows per subject): compute bias and limits of agreement with the Bland & Altman 2007 correction, or average per subject first. Plot the difference against the mean of the two methods, not against the reference. Check for proportional bias.
6. **Per-stratum tables.** Report every metric for each of C1–C7 and for Fitzpatrick I–II, III–IV, and V–VI, with n subjects and n windows. Flag strata with fewer than about 5 subjects as unreliable.
7. **Baselines.** Include a mean predictor (training mean), a classical unsupervised method (POS or CHROM) via rPPG-Toolbox, and one supervised baseline (PhysNet or EfficientPhys).
8. **Narrow-range targets (SpO2, BP).**
   - SpO2 in healthy subjects sits around 95–99%, so a constant predictor gets MAE of about 1%. Report MAE alongside the mean-predictor MAE, Pearson r (often near zero), and ARMS.
   - Prefer data with real desaturation (SUMS hypoxia, M3PD patients).
   - For BP, report ME ± SD and compare to the AAMI/ISO 81060-2 thresholds (ME ≤ 5 mmHg, SD ≤ 8 mmHg; standard knowledge, unverified here). Also report against a mean or demographic-only predictor using age, sex, and BMI. MCD-rPPG supplies these fields, and they predict BP strongly.

### Gaps
- Exact rPPG-Toolbox defaults (e.g. `EVALUATION_WINDOW` = 10 s, `CHUNK_LENGTH` = 180, face resize 72×72) are from background knowledge, not the README text retrieved. Check the YAML configs in `configs/`.
- The FDA pulse-oximeter ARMS ≤ 3% guidance, the 2025 FDA skin-pigmentation guidance for oximeters, and AAMI/ISO 81060-2 BP thresholds were not fetched this session.
- No source was retrieved that quantifies how much segment-level leakage inflates rPPG MAE.

---

## Q4. Synthetic data and physics-based augmentation

### Takeaway
SCAMPS is the largest synthetic rPPG set (2,800 videos with waveform labels), but its R-UDA licence bars commercial use of Data **or Results**. Other options are the UCLA avatar pipeline (CVPR 2022), PhysFlow skin-tone transfer, MA-rPPG motion augmentation, and simulated multispectral UBFC. The team can probably build its own commercially clean augmentations: skin-tone or melanin scaling, low-light simulation, codec and fps degradation, and mask overlays.

### Cited Findings
- SCAMPS: 2,800 videos with PPG, breathing, AUs, pose, and masks. The R-UDA licence bars any commercial offering that uses the Data or Results. — [GitHub](https://github.com/danmcduff/scampsdataset); [LICENSE](https://raw.githubusercontent.com/danmcduff/scampsdataset/main/LICENSE.txt)
- UCLA "Synthetic Generation of Face Videos with Plethysmograph Physiology" (Wang et al., CVPR 2022) generates synthetic avatars with PPG to improve accuracy on darker skin, alongside the real UCLA-rPPG set. A related US patent (12539047) exists. — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Synthetic_Generation_of_Face_Videos_With_Plethysmograph_Physiology_CVPR_2022_paper.pdf); [USPTO 12539047](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12539047)
- PhysFlow: skin-tone transfer for rPPG via conditional normalising flows. — [arXiv 2407.21519](https://arxiv.org/pdf/2407.21519)
- Motion-augmented versions of UBFC-rPPG, PURE, SCAMPS, and UBFC-Phys are available through the MA-rPPG Video Toolbox. — [rPPG-Toolbox README](https://github.com/ubicomplab/rPPG-Toolbox)
- Simulated multispectral UBFC-rPPG (31 bands, 400–700 nm) is on IEEE DataPort. — [DataPort](https://ieee-dataport.org/keywords/remote-photoplethysmography-rppg)
- Low-light enhancement for rPPG: LPPE (with MSLR) and an image-enhancement method. — [Springer MSLR](https://link.springer.com/article/10.1007/s00138-025-01719-3); [HKUST IEM 2025](https://www.ieda.ust.hk/dfaculty/so/pdf/Chen-et-al-2025-IEM.pdf)

### Inferences
- Synthetic data should augment training, not replace real test data. All headline numbers should come from real video.
- Low-light simulation that respects physics: work in linear RGB (undo gamma). Scale intensity by the lux ratio (e.g. 300 → 30 lux gives ×0.1). Add Poisson shot noise and Gaussian read noise calibrated to the target camera. Re-apply gamma, then quantise to 8 bits.
  - Naive brightness scaling understates noise, which is what actually destroys the pulse signal at under 50 lux.
  - Validate the simulation against MSLR at 3, 8, and 35 lux.
- Skin-tone augmentation: melanin absorption scaling in an optical skin model reduces the pulsatile AC/DC ratio for darker skin. Simple hue shifts do not reproduce this. Validate on real Fitzpatrick V–VI data (VitalVideos, the organiser data, or self-collected).
- Codec augmentation: re-encode on the fly with random CRF or bitrate (300 kbps–5 Mbps), random fps (8–30), and random downscale. This is cheap and commercially clean.

### Gaps
- No dataset or tool called "SynthPPG" was found. It may refer to the UCLA CVPR 2022 avatar work.
- Licence terms for the UCLA avatar data and PhysFlow code were not verified (the UCLA site failed TLS).

---

## Q5. Recommended self-collection protocol for a student team

### Takeaway
Copy the best-documented public protocols. VitalVideos, UBFC, and SUMS all use a **CMS50D+/CMS50E-class finger oximeter logged to a PC**, which gives PPG at about 20–60 Hz and SpO2/HR at 1 Hz, plus an automatic BP cuff. Add a lux meter, a sync event, and a consent form that **explicitly permits commercial use by third parties (MathWorks)**. Target the conditions public data lacks: masks, multi-person scenes, measured lux of 50 or less, Fitzpatrick V–VI, and ages 60+.

### Cited Findings
- Reference devices in public datasets:
  - VitalVideos uses a CMS50D+ at about 60 Hz with HR/SpO2 at 1 Hz, an Omron M7 or MicroLife B3 for BP, and (WorldWide) 3-lead ECG at 500 Hz. — [vitalvideos.org](https://www.vitalvideos.org/)
  - UBFC-rPPG uses a CMS50E. — [UBFC](https://sites.google.com/view/ybenezeth/ubfcrppg)
  - SUMS uses a CMS50E+ (PPG at 20 Hz) and an HKH-11C respiration sensor at 50 Hz. — [SUMS](https://github.com/thuhci/SUMS)
  - DDMP-FV uses a CONTEC CMS50DL and a Bioland arm cuff. — [DataPort](https://ieee-dataport.org/documents/ddmp-fv)
  - MPU-rPPG uses a CMS60D with an MX Brio at 60 fps. — [Sci Data snippet](https://www.nature.com/articles/s41597-026-07310-3)
  - CHILL uses Biosignalsplux PPG and ECG at 1 kHz with timestamp sync. — [CHILL](https://arxiv.org/html/2503.11697)
- Recording setups:
  - VitalVideo: 50–70 cm distance, at least 150 lux with a ring light. — [arXiv 2404.05003](https://arxiv.org/html/2404.05003v1)
  - UBFC: about 1 m with a math game to raise HR. — [UBFC](https://sites.google.com/view/ybenezeth/ubfcrppg)
  - SUMS and MCD-rPPG: exercise (stairs, squats) to widen the HR range. — [SUMS](https://github.com/thuhci/SUMS); [MCD-rPPG](https://huggingface.co/datasets/kyegorov/mcd_rppg)
- Ethics, consent, and release models:
  - VitalVideos-WorldWide states consent and IRB review. — [vitalvideos.org](https://www.vitalvideos.org/)
  - iBVP keeps limited-consent participants in a "No-media-use" folder. — [iBVP](https://github.com/PhysiologicAILab/iBVP-Dataset)
  - CHILL released only consenting participants, as downsampled faces. — [Zenodo](https://zenodo.org/records/14637544)
  - UCLA restricts its data under its IRB protocol. — [UCLA (snippet)](https://visual.ee.ucla.edu/rppg_avatars.htm)

### Inferences
Suggested protocol for about 30–40 subjects in 2–3 weeks:

- **Equipment:**
  - Cameras: a 1080p/30–60 fps webcam (Logitech C920/C922/Brio class) recorded **losslessly**, e.g. FFV1 or raw frames via ffmpeg. Add a mid-range smartphone at fixed settings if possible.
  - Pulse oximeter: CMS50D+ or CMS50E with USB PC logging, PPG at about 60 Hz plus SpO2/HR at 1 Hz. Use the vendor software or an open-source reader. The open-source tools are unverified this session.
  - BP: a validated automatic upper-arm cuff (Omron) before and after each clip.
  - Lux meter at the face plane: log the reading per clip.
  - Dimmable LED panel, and optionally an IR or night-mode camera.
  - Also: chair with head distance marked at 60–100 cm, colour checker card, Fitzpatrick questionnaire, and a colorimeter or skin-tone swatch (e.g. Monk scale) photographed per subject.
- **Sync:** have the recording PC log both streams with one monotonic clock. Add an optical event at start and end, such as an LED flash in frame or a clap plus a finger-tap. Then refine with rPPG–PPG cross-correlation. Save camera frame timestamps (variable frame rate is common on webcams).
- **Session per subject (about 20 min):**
  - Rest at 300+ lux for 60 s.
  - Rest at 50, 20, and 5 lux for 60 s each.
  - Talking or head motion for 60 s.
  - Surgical mask, then cloth or N95, for 60 s each at normal light.
  - Post-exercise (squats or stairs) for 60 s.
  - Multi-person: 2–3 subjects in frame together, **each with their own oximeter**, for 60–120 s.
  - Optional slow-breathing segment to vary RR.
- **Sampling:** deliberately recruit Fitzpatrick V–VI and adults aged 60+ (e.g. family or community volunteers, under ethics approval). Record age, sex, BMI, makeup, glasses, beard, and medications affecting HR.
- **Derived conditions:** create C3/C4 variants (≤10 fps, ≤500 kbps, low-res) offline from the lossless masters, so every condition shares the same ground truth.
- **IRB and consent:** use the university HREC/IRB. The consent form must cover:
  - identifiable facial video;
  - storage and retention;
  - optional public release;
  - **use by the competition organisers and sponsor (MathWorks), including commercial use**;
  - the right to withdraw.
  
  Offer tiered consent (internal use only / share with organisers / public release), like iBVP and CHILL. Older or clinical participants may need extra safeguards. Do not use invasive references.
- **Data hygiene:** pseudonymise IDs, keep consent tiers in metadata, and build subject-disjoint splits before any modelling.

### Gaps
- No primary source was fetched for CMS50D+/CMS50E PC-logging specifics (USB protocol, open-source drivers, actual PPG sample rate). VitalVideos and UBFC report about 60 Hz, while SUMS reports 20 Hz for the CMS50E+. Confirm on the purchased unit.
- Specific IRB requirements of the team's institution, and whether SP Cup rules require proof of IRB approval, were not researched.
- Whether the Sri Ramachandra organiser dataset's terms allow MathWorks commercial use is unknown. Clarify with the organisers.
