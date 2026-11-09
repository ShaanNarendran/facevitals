# Real-time edge deployment of an rPPG pipeline on Raspberry Pi (face detection, landmarks, target tracking, rPPG models, camera, MATLAB route)

Scope note: researched 20 Oct 2026. Pi 5-specific published numbers are scarce. Many figures below are for Pi 4B, x86 laptops, or the Hailo-8L accelerator. Each number says which hardware it was measured on. Numbers marked **[unverified]** come from secondary or anecdotal sources, or from a source with internal inconsistencies.

---

## 1. Face detection / landmarks on Raspberry Pi 4/5 (CPU and Hailo accelerators)

### Takeaway
On CPU, OpenCV YuNet is the cheapest credible detector with 5-point landmarks: about 6 ms per frame at 160x120 on a Pi 4B, so it should be faster on a Pi 5. MediaPipe Face Landmarker (478 points) is the obvious dense-landmark option for skin-ROI definition, but the only published number is about 10 fps for the older Face Mesh on a Pi 4. On the Hailo-8L (AI Kit / AI HAT+ 13 TOPS), SCRFD-2.5G runs at 311 fps raw in the model zoo. Measured end-to-end inside a real rPPG pipeline it took about 19 ms per frame. Pi 5 numbers must be measured by the team.

### Cited Findings
- **YuNet (OpenCV Zoo benchmark, OpenCV DNN CPU backend):** Raspberry Pi 4B, input 160x120: mean 6.23 (fp32) and 6.68 (int8). Units are not stated on the page but are presumably ms. INT8 gives no speed-up on the Pi 4. For comparison: Jetson Nano CPU 5.62, Khadas Edge2 (RK3588, Cortex-A76 cores like the Pi 5) 2.30 — [OpenCV Zoo benchmark README](https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md)
- **SFace (OpenCV face-recognition embedding, 150x150) on Pi 4B:** 68.82 fp32 / 87.42 int8. RK3588: 28.94. Too heavy to run every frame, so use it only for occasional re-identification — [OpenCV Zoo benchmark README](https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md)
- **MobileFaceNet (facial-expression model, 112x112) on Pi 4B:** about 27.8 (fp32). This is a useful proxy for the cost of a MobileFaceNet face embedding on a Pi 4 CPU — [OpenCV Zoo benchmark README](https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md)
- **YuNet model page:** the 2023mar ONNX has a fixed input size. A newer `face_detection_yunet_2026may.onnx` supports dynamic H/W. It detects faces of about 10x10 to 300x300 px. WIDER Face AP is 0.8844/0.8656/0.7503 (easy/medium/hard) in the table, though the intro gives different numbers. The page lists no params/FLOPs or Pi latency — [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **YuNet vs MediaPipe on desktop:** in a 2025 ICITS study on an i5-13600K, YuNet took about 0.03 s per face vs MediaPipe at about 0.04 s, with MediaPipe making fewer detections. **[secondary, from search snippet; desktop, not Pi]** — [Benchmark face detection using DL models and frameworks (TEC)](https://orion.tec.ac.cr/en/publications/benchmark-face-detection-using-deep-learning-models-andframeworks/)
- **MediaPipe on Pi 4:** a project doc claims MediaPipe face detection at about 30–40 fps on a Pi 4 (4 GB), Haar at 45–60 fps, and YOLOv8n at 15–25 fps, detection only. **[unverified: project documentation, not a controlled benchmark]** — [Eulerian-Vitals-Extraction "choosing detector"](https://www.mintlify.com/Daval03/Eulerian-Vitals-Extraction/guides/choosing-detector)
- **MediaPipe Face Mesh (468 landmarks) on Pi 4B:** about 10 fps. **[unverified: tutorial anecdote]** — [Core Electronics: face landmark tracking with Raspberry Pi](https://core-electronics.com.au/videos/pose-estimation-and-face-landmark-tracking-with-raspberry-pi-and-opencv)
- **Pi 5 tutorial:** runs MediaPipe Face Mesh (468 points) with picamera2 capture but reports no fps — [SunFounder AI Lab Kit: mp_1_face](https://docs.sunfounder.com/projects/ai-lab-kit/en/latest/mediapipe/mp_1_face.html)
- **Official MediaPipe Face Landmarker (Python):** uses the `mediapipe` PyPI package plus a `face_landmarker.task` bundle, with IMAGE/VIDEO/LIVE_STREAM running modes. Google points to a dedicated Raspberry Pi sample app — [Google AI Edge: Face Landmarker Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python?hl=zh-tw)
- **Hailo-8L model zoo (WIDER Face, DFC v2.19.0), raw on-chip throughput:**
  - scrfd_500m: 588 fps, 640x640, 0.63 M params, 1.5 GOPS, mAP 68.8 (HW)
  - scrfd_2.5g: 311 fps, 0.82 M params, 6.88 GOPS, mAP 76.4
  - scrfd_10g: 149 fps, 4.23 M params, 26.74 GOPS, mAP 82.1
  - retinaface_mobilenet_v1: 75.6 fps (b1) / 104 fps (b8), 736x1280, 3.49 M params, 25.14 GOPS, mAP 81.2
  - lightface_slim (Ultra-Light-Fast): 3622 fps, 240x320, 0.26 M params, mAP 39.3

  Source: [Hailo Model Zoo – HAILO8L face detection](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_face_detection.rst)
- **Measured on a Raspberry Pi AI Kit (Hailo-8L) via the DeGirum AI Hub models, end-to-end per frame, WIDER Face val AP easy/med/hard:**
  - YOLOv8n-ReLU6-Face: 12.37 ms (87.3/78.9/44.7)
  - SCRFD-0.5G: 20.44 ms (89.1/85.7/63.7)
  - SCRFD-2.5G: 18.91 ms (93.2/90.9/74.1)
  - SCRFD-10G: 26.57 ms (95.1/93.4/80.9)
  - RetinaFace-MobileNet: 69.63 ms (87.3/82.6/61.3)

  The authors chose SCRFD-2.5G because they needed under 20 ms per frame for detection — [Ahsanunnisa et al., ISBI 2026](https://informatika.stei.itb.ac.id/~rinaldi.munir/Penelitian/Makalah-ISBI-2026.pdf)
- **AI Kit hardware:** the AI Kit pairs the M.2 HAT+ with a Hailo-8L at 13 TOPS. Vendors state that landmarking and detection networks run entirely on the Hailo, freeing the Pi 5 CPU — [Electrokit AI Kit](https://electrokit.com/en/raspberry-pi-ai-kit); [Electronic Specifier](https://electronicspecifier.com/artificial-intelligence/raspberry-pi-ai-kit-speeds-vision-applications)
- **Hailo community:** SCRFD is supported in DeGirum PySDK. RetinaFace works with the same code at a different input resolution — [Hailo community: SCRFD in DeGirum PySDK](https://community.hailo.ai/t/scrfd-model-support-in-degirum-pysdk/8457). One user found that a face pipeline was not actually using the NPU on a Pi 5 with the AI HAT+, so verify NPU use before trusting fps numbers — [Hailo community face-recognition guide thread](https://community.hailo.ai/t/a-comprehensive-guide-to-building-a-face-recognition-system/8803/35)
- **In a real-time rPPG system (Face2PPG-RT, laptop i7-6700HQ, C++):** YuNet detection took 1.10 ms per frame vs dlib HOG at 5.48 ms. dlib ERT alignment (68-point) took 3.74–4.48 ms. Faces were detected only every 3 frames — [Álvarez Casado et al., arXiv 2508.18787](https://arxiv.org/html/2508.18787v1)

### Inferences
- **Raspberry Pi 5 estimate:** the Pi 5 uses Cortex-A76 cores, like the RK3588 big cores. YuNet at 160x120 to 320x240 should therefore take roughly 2–5 ms per frame on a Pi 5 CPU, based on RK3588 = 2.3 ms at 160x120. This is an extrapolation, not a measurement.
- MediaPipe Face Landmarker on a Pi 5 CPU should plausibly reach about 15–30 fps for a single face, given about 10 fps for Face Mesh on a Pi 4 and a Pi 5 CPU 2–3x faster. **[extrapolation: must be measured]**
- **Recommended CPU-only stack:**
  - YuNet for detection plus a 5-point coarse ROI every N frames (N = 3–5)
  - MediaPipe Face Landmarker (VIDEO mode, `num_faces` = 1) on the target crop only, for skin-ROI polygons (forehead/cheeks)
  - optional KLT point tracking between landmark runs
- **With an AI Kit / AI HAT+:** SCRFD-2.5G on Hailo for detection leaves the CPU free for landmarks and rPPG.
- dlib 68-point (ERT) is fast enough (about 4 ms on a laptop) but needs a separate detector and is less robust. Prefer MediaPipe or YuNet.

### Gaps
- No controlled Pi 5 CPU benchmarks were found for any of: YuNet, BlazeFace, Face Landmarker, SCRFD, RetinaFace, dlib, or Ultra-Light-Fast-Generic-Face-Detector.
- No published Hailo-10H (AI HAT+ 2) or Hailo-8 (AI HAT+ 26 TOPS) face-model tables were checked. The AI HAT+ 26 TOPS and AI HAT+ 2 specs were not verified in this session.
- The MediaPipe Face Landmarker blendshapes/478-point model cost on ARM was not found.

---

## 2. Multi-face tracking with persistent identity, and target designation (constraint C3)

### Takeaway
No published Pi system combines ByteTrack/SORT with face re-ID for a single designated target. The building blocks are cheap, though: Kalman plus IoU association on 5–10 faces takes well under 1 ms. A MobileFaceNet/SFace embedding costs about 28–70 ms on a Pi 4 CPU, so run it only at enrolment and on track birth or ambiguity, not every frame. The ISBI 2026 Pi rPPG system used a simple centroid tracker.

### Cited Findings
- **Pi + Hailo rPPG pipeline:** uses detection, then a **Centroid Tracker** (Euclidean distance between centroids in consecutive frames), then ROI, rPPG, and HR, chosen for real-time requirements — [Ahsanunnisa et al., ISBI 2026](https://informatika.stei.itb.ac.id/~rinaldi.munir/Penelitian/Makalah-ISBI-2026.pdf)
- **Face2PPG-RT:** detects faces every 3 frames, keeps the latest frame from the capture thread, and resets after 60 consecutive missed detections — [arXiv 2508.18787](https://arxiv.org/html/2508.18787v1)
- **Pi 4 face-recognition attendance system:**
  - Components: MediaPipe BlazeFace, MobileFaceNet (ONNX Runtime), tracking, alignment, quality filtering, cosine-similarity matching, and multi-frame voting
  - Reported results: 24.3 ms latency, 95.8% accuracy, 0.1% FAR, 110–140 MB RAM

  **[thesis-level source; numbers from search summary, not verified in full text]** — [Univ. M'sila: Efficient Raspberry Pi-based face recognition for attendance](https://depot.univ-msila.dz/items/42d819e1-9eef-4ae1-b9cd-5205f88e1370)
- **ByteTrack enhanced with appearance-based track-ID correction:** reduced ID switches during occlusion. Measured on a GPU (A100), not embedded — [arXiv 2505.04616](https://arxiv.org/pdf/2505.04616)
- **Embedding costs on Pi 4B CPU (OpenCV DNN):** SFace (150x150) about 69 ms. MobileFaceNet-based 112x112 model about 28 ms. RK3588 (A76-class cores) is about 2.4x faster — [OpenCV Zoo benchmark README](https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md)
- **MATLAB precedent:** there is a MathWorks Answers thread on deploying a face-tracking algorithm to a Raspberry Pi 4. It shows the question is common but gives no benchmark — [MATLAB Answers 1567048](https://www.mathworks.com/matlabcentral/answers/1567048-how-can-i-deploy-a-face-tracking-algorithm-to-raspberry-pi-4)

### Inferences (proposed design for C3; not a documented result)
- **Per-frame tracker:** ByteTrack- or OC-SORT-style.
  - Constant-velocity Kalman filter on [cx, cy, s, r]
  - IoU / Hungarian association, with a second pass for low-confidence detections (ByteTrack)
  - Track birth after 3 hits; deletion after about 1–2 s of misses
  - Cost is negligible next to detection
- **Target designation (lock-in policy), in priority order:**
  1. **Enrolment by embedding.** At session start, the target looks at the camera for about 2 s. Average 5–10 aligned-crop MobileFaceNet/SFace embeddings into a template. This is the most defensible approach for "reject bystanders".
  2. **Fallback, only if enrolment is not allowed by the protocol:** pick the largest and most central face that persists for at least K frames, and lock it.
  3. **Operator tap/click** for the demo UI.
- **Track-to-identity binding:** the target track keeps its ID as long as IoU association holds.
- **Re-check the embedding** when any of these happens:
  - a new track is born
  - two tracks overlap (IoU > 0.3)
  - the target track is lost
  - a periodic check fires (e.g., every 1–2 s)
- **Accept/reject rule:** accept a track as the target if its cosine similarity to the template exceeds a threshold, tuned on validation data (often about 0.4–0.5 for ArcFace-type models). Add hysteresis and voting over several frames.
- **Occlusion and re-entry:**
  - Keep a short Kalman coast.
  - On re-appearance, any new track is a candidate. It becomes the target only after the embedding matches.
  - The rPPG buffer is marked invalid during the gap (no signal fusion from other faces). The output can hold the last HR with a "quality low" flag.
- **Cost budget:** about 30–70 ms per embedding on Pi 4 means an estimated 12–30 ms on Pi 5 **[extrapolation]**. At about 1 check per second this adds under 5% CPU. Embeddings can also run on Hailo, which was not benchmarked here.
- **MATLAB equivalents:**
  - `vision.CascadeObjectDetector` or a deep detector
  - `configureKalmanFilter` / `vision.KalmanFilter`
  - `assignDetectionsToTracks` (Hungarian)
  - `vision.PointTracker` (KLT) for landmark/ROI propagation
  - `trackerGNN`/`multiObjectTracker` (Sensor Fusion toolbox)

  These function names are from general MathWorks knowledge. **They were not re-verified against current docs in this session.** In particular, check whether a built-in `faceDetector` object exists in the team's MATLAB release.

### Gaps
- No published ID-switch/false-target rates were found for ByteTrack + face-embedding on a Pi.
- No published Pi 5 latency was found for MobileFaceNet/ArcFace-mobile under ONNX Runtime or TFLite.

---

## 3. Running rPPG models on Pi-class hardware (classical and deep; runtimes; quantisation)

### Takeaway
Published edge numbers:
- **EfficientPhys (WACV 2023), Pi 4B:** POS/CHROM about 27–28 ms per frame; EfficientPhys-C about 40 ms; TS-CAN about 63 ms; EfficientPhys-T1 about 300 ms.
- **ISBI 2026, Raspberry Pi AI Kit (Hailo):** quantised DeepPhys (mixed UINT4/UINT8) ran in about 29 ms with unchanged MAE (3.01 bpm), and the pipeline sustained about 30 fps. Only DeepPhys compiled on Hailo; 3D-conv and 5-D reshape models did not.
- **ME-rPPG / "ME-Flow" (2025):** a streaming state-space model at 0.15 GFLOPs, 9.46 ms per frame on a laptop CPU, 3.6 MB memory. A strong candidate for Pi 5 CPU streaming.

### Cited Findings
- **EfficientPhys Table 4, "Preprocessing and model latency on Raspberry Pi 4B per frame" (quad-core Cortex-A72):**

  | Method | Total per frame |
  |---|---|
  | EfficientPhys-C | 40 ms |
  | EfficientPhys-T1 | 300 ms |
  | EfficientPhys-T2 | 40 ms |
  | TS-CAN | 63 ms (3 preprocessing + 60 model) |
  | POS | 27 ms |
  | CHROM | 28 ms |
  | ICA | 31 ms |
  | Dual-GAN | >275 ms |

  Input is 72x72. EfficientPhys needs no face detection or normalisation preprocessing — [Liu et al., EfficientPhys (arXiv 2110.04447)](https://ar5iv.labs.arxiv.org/html/2110.04447); [WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.html). Note: about 27 ms per frame for POS looks high for pure signal processing. It likely reflects their Python implementation or window recomputation **[interpretation]**.
- **ME-rPPG Table 1 (CPU latency, i7-13620H laptop, ONNX export):**

  | Model | Params (K) | Memory (MB) | Latency (ms) | FLOPs (G) |
  |---|---|---|---|---|
  | EfficientPhys | 2,189 | 609 | 371 | 39.0 |
  | TS-CAN | 533 | 228 | 55.2 | 8.07 |
  | PhysNet | 770 | 230 | 62.1 | 6.16 |
  | PhysFormer | 7,395 | 576 | 2330 | 95.7 |
  | RhythmMamba | 4,936 | 347 | 889 | 27.4 |
  | ME-Chunk | 580 | 260 | 526 | 20.6 |
  | ME-Flow | 580 | 3.6 | 9.46 | 0.15 |

  ME-Flow runs frame by frame (causal, recursive moving-average temporal normalisation, constant memory). It reaches UBFC MAE 0.52 and PURE 0.38 frame-wise. In a real-world C920 study (11 subjects) it scored MAE 1.78 static / 2.53 moving, vs PhysNet at 2.21 / 5.03. Latencies are per inference unit (chunk vs frame) on a laptop, not a Pi — [arXiv 2504.01774](https://arxiv.org/html/2504.01774v1)
- **ISBI 2026, "Real-Time Camera-Based Heart Rate Estimation System Using Deep Learning on Edge Device" (ITB):**
  - **Accuracy (rPPG-Toolbox, trained on PURE, tested on UBFC-rPPG), MAE in bpm:**
    - Conventional: POS 3.735, CHROM 3.934, ICA 15.2, GREEN 19.8
    - Deep: DeepPhys 1.109, EfficientPhys 1.381, FactorizePhys 1.130, TS-CAN 1.172, PhysNet 2.658, RhythmFormer 0.837, PhysFormer 1.465, PhysMamba 1.528, iBVPNet 2.992
  - **Hailo compatibility:** only DeepPhys was fully supported. Others failed on "tensor 5D reshape operations, expand functions, and 3D CNN layers".
  - **Quantisation:** calibration on 128 PURE images.
    - Full UINT8: latency from 163,493.80 ms (Pi CPU FP32) to 33.55 ms on Hailo; MAE from 3.01 to 3.28 bpm
    - Mixed 20% UINT4 / 80% UINT8 with equalisation: 28.94 ms, MAE 3.01 bpm
  - **Throughput:** 29.972 fps average end-to-end, with SCRFD-2.5G detection and a centroid tracker.
  - **Pipeline optimisations:** split the 180-frame window into 30-frame chunks; batch inference on the NPU; incremental (running) normalisation statistics.
  - **Threads:** capture, processing, and display, connected by queues.
  - **Input:** pre-recorded UBFC videos fed frame by frame, not a live camera.

  — [Ahsanunnisa et al., ISBI 2026, DOI 10.1109/ISBI61048.2026.11515871](https://informatika.stei.itb.ac.id/~rinaldi.munir/Penelitian/Makalah-ISBI-2026.pdf)
  - **[Caution: internal inconsistencies]** The paper says "Raspberry Pi 4 with an Arm Cortex-A76", but the A76 is the Pi 5 CPU and the AI Kit is a Pi 5 product. It alternates between "Hailo-8" and "Hailo-8L". The 163,493.80 ms FP32 figure is presumably for a whole window or batch, not per frame. Treat the hardware as "Pi 5 + Hailo-8L AI Kit (probably)".
- **Face2PPG-RT (C++, laptop i7-6700HQ):**
  - Total 9.69 ms per frame (103 fps) with YuNet + dlib ERT + CIE-Lab a* channel
  - The rPPG/BVP step itself takes 0.46 ms; filtering and spectral analysis take microseconds
  - The offline POS "Server" configuration was 116–222 ms per frame because of heavier detection/alignment (SSD+DAN)
  - Real-time configurations are less accurate: UBFC2 MAE 4.7–6.7 vs 0.9–1.4 for Server

  This shows the face pipeline, not the rPPG maths, dominates cost — [arXiv 2508.18787](https://arxiv.org/html/2508.18787v1)
- **Technion SDA student competition 2025 (remote-PPG vital signs):** reports about 52 ms end-to-end latency (about 19 fps), judged sufficient for 15 fps processing. **[hardware not verified; from search snippet]** — [Technion SIPL 2025 SDA competition PDF](https://sipl.ece.technion.ac.il/wp-content/uploads/2025/12/2025-SDA-Student-Competition-Remote-PPG-based-Vital-Signs-Monitoring.pdf)
- **Aggregator figure:** a listing shows "371 ms frame latency" for EfficientPhys under mobile/ARM. This actually matches ME-rPPG's laptop-CPU table, so do not cite it as a mobile number — [arXiv 2504.01774](https://arxiv.org/html/2504.01774v1)

### Inferences
- Classical POS/CHROM on an ROI-mean RGB trace is computationally trivial (sub-ms per frame in C++ per Face2PPG). The Pi 4 "27 ms" figure is implementation-bound. On a Pi 5 with NumPy, POS over a sliding 10 s window should be well under 5 ms per update **[estimate]**.
- **Best CPU deep candidates for Pi 5:**
  - ME-Flow-style causal models (0.15 GFLOPs per frame)
  - EfficientPhys-C / TS-CAN at 72x72 (about 40–63 ms per frame on Pi 4, so roughly 15–30 ms on a Pi 5 **[extrapolation]**)
  - Batch the deep model per chunk (e.g., 30 frames once per second) rather than per frame, as the ISBI paper did.
- **Hailo constraints:** avoid 3D convs and 5-D reshapes (PhysNet, PhysFormer, FactorizePhys, RhythmFormer, likely EfficientPhys/TS-CAN's TSM). DeepPhys-style 2D two-branch nets compile.
- **Quantisation:** the only edge rPPG INT8 evidence found (DeepPhys on Hailo) shows a small MAE change (3.01 to 3.28 bpm for full UINT8, recovered with mixed precision + equalisation). Calibrate with in-domain face crops.
- **Runtime choice for CPU (Pi 5):** ONNX Runtime (ARM64 wheels) or LiteRT/TFLite with XNNPACK are the standard options. NCNN is often fastest on ARM for small CNNs. OpenVINO's ARM CPU plugin exists, but no rPPG-on-Pi numbers were found for any of them.
- **Complexity reporting:** report params, FLOPs/MACs per frame (or per window, normalised per frame), peak RAM, per-stage latency (mean/p95), end-to-end fps, and CPU% plus power (USB-C meter). The ME-rPPG table format (Params, Mem, Latency, FLOPs) is a good template.

### Gaps
- No direct Pi 5 CPU latency was found for any deep rPPG model (PhysNet, TS-CAN, EfficientPhys, FactorizePhys, ME-rPPG) under ONNX Runtime, TFLite, NCNN, or OpenVINO.
- MobilePhys smartphone latency and other phone/Jetson rPPG numbers were not retrieved in this session.
- No INT8-on-CPU (non-Hailo) rPPG accuracy study was found.

---

## 4. Camera stack on Pi 5 (Camera Module 3 / USB webcam, locked exposure/WB, timing)

### Takeaway
For rPPG, auto-exposure, AGC, and AWB must be frozen after the camera settles. libcamera/picamera2 expose `AeEnable`, `AwbEnable`, `ExposureTime`, `AnalogueGain`, `ColourGains`, and `FrameDurationLimits`, and each frame's metadata carries `SensorTimestamp`. Use `SensorTimestamp` rather than wall-clock time to build the evenly-sampled signal.

### Cited Findings
- **Setting controls:** picamera2 sets controls via `set_controls({...})`. Example: disable AWB and set `ColourGains` (red, blue) together with `ExposureTime` and `AnalogueGain` — [tessl picamera2 controls doc](https://tessl.io/registry/tessl/pypi-picamera2/files/docs/controls.md)
- **Locking procedure:** the practical approach is to let AE/AWB converge, then turn off `AeEnable`/`AwbEnable` and set fixed exposure, gain, and colour gains — [OctoPrint community: disable AEC on RPi camera](https://community.octoprint.org/t/is-there-a-way-to-disable-auto-exposure-control-on-an-rpi-camera2/59802)
- **libcamera control semantics:**
  - `ColourGains` = red and blue gains, in that order
  - `SensorTimestamp` = time at which the first row of the active array was exposed
  - `FrameDurationLimits` was introduced for frame-duration control, and the sensor may clip the requested limits to what its mode supports

  — [libcamera controls API](https://docs.libcamera.org/master/public-api/namespacelibcamera_1_1controls.html); [libcamera FrameDurationLimits patch](https://lists.libcamera.org/pipermail/libcamera-devel/2020-June/009288.html)
- **Caveat:** the original patch said the limits apply when AE is enabled. When AE is off, frame duration is bounded by the manual `ExposureTime`. Set `FrameDurationLimits=(33333,33333)` for 30 fps and keep `ExposureTime` < 33 ms — [libcamera patch discussion](https://patchwork.libcamera.org/patch/3988/raw/). **[Behaviour on current Pi 5 firmware not re-verified]**
- **Third-party example:** a picamera wrapper sets these controls programmatically — [QVideo Picamera source](https://qvideo.readthedocs.io/en/stable/_modules/QVideo/cameras/Picamera/_camera.html)
- **Pi 5 capture with picamera2 feeding MediaPipe Face Mesh:** a working example exists — [SunFounder AI Lab Kit](https://docs.sunfounder.com/projects/ai-lab-kit/en/latest/mediapipe/mp_1_face.html)

### Inferences
- **Recommended capture settings:**
  - picamera2, video configuration
  - main stream 640x480 or 1280x720 RGB888, plus a low-res "lores" stream (e.g., 320x240) for detection
  - 30 fps fixed via `FrameDurationLimits`
  - after about 2 s settle: `AeEnable=False`, `AwbEnable=False`, with `ExposureTime`/`AnalogueGain`/`ColourGains` read from metadata and then fixed
  - `NoiseReductionMode` off or minimal, to avoid temporal NR smoothing the pulse **[recommendation; NR control name from libcamera/picamera2 general knowledge, not re-verified]**
  - disable Camera Module 3 continuous autofocus (`AfMode` manual) to avoid focus breathing **[recommendation]**
- **Timing:** log `SensorTimestamp` per frame and resample the RGB trace to a uniform grid before filtering/FFT. Dropped frames then do not bias HR.
- **USB webcams (e.g., C920):** UVC exposure/WB locks via `v4l2-ctl` (`exposure_auto`, `white_balance_temperature_auto`). Timestamps are less precise.
- **Threading:** run capture in its own thread that always provides the latest frame. Face2PPG-RT and the ISBI 2026 system both used capture/processing/display threads with queues — [arXiv 2508.18787](https://arxiv.org/html/2508.18787v1); [ISBI 2026](https://informatika.stei.itb.ac.id/~rinaldi.munir/Penelitian/Makalah-ISBI-2026.pdf)

### Gaps
- The official Picamera2 manual PDF could not be fetched (redirect then 404), so the exact picamera2 metadata key names and units (`SensorTimestamp` in ns) were not confirmed from the primary doc.
- No measured frame-rate jitter data for Camera Module 3 on a Pi 5 was found.

---

## 5. MATLAB route (Raspberry Pi support package / Blockset, codegen, deep networks)

### Takeaway
MathWorks officially supports Raspberry Pi 5 (32/64-bit Raspberry Pi OS) and standalone deployment via code generation. However, the ARM Compute Library deep-learning target is being deprecated in favour of generic C/C++ ("none" library). 3D conv layers do not appear in the codegen-supported list. The MATLAB route is therefore practical for classical POS/CHROM, Kalman/Hungarian tracking, and simple 2D CNNs, but riskier for MediaPipe-class landmarkers and 3D-conv rPPG nets. Python is faster to iterate on Pi.

### Cited Findings
- **Platform support:** MathWorks lists Raspberry Pi 5 (plus 4B, CM4, Zero 2 W, …) as supported. Raspberry Pi OS 32- and 64-bit are supported, and MathWorks advises checking the release-specific compatibility table — [MathWorks Raspberry Pi hardware support](https://www.mathworks.com/hardware-support/raspberry-pi.html)
- **Deployment requirements:** standalone executables (run at boot) need Embedded Coder. Edge AI uses MATLAB Coder to generate optimised C/C++. Examples include ResNet-50 classification and a webcam edge-detection example — [MathWorks Raspberry Pi hardware support](https://www.mathworks.com/hardware-support/raspberry-pi.html)
- **ARM Compute Library codegen workflow:**
  - `coder.DeepLearningConfig('arm-compute')` with `ArmArchitecture`/`ArmComputeVersion`
  - an entry-point function using `coder.loadDeepLearningNetwork` + `predict`
  - code is built on the Pi via the support package

  — [MathWorks: Code generation for deep learning networks with ARM Compute Library](https://www.mathworks.com/help/coder/ug/code-generation-for-deep-learning-networks-with-arm-compute-library.html)
- **Deprecation:** "Support for the ARM Compute Library as a target for deep learning code generation will be removed in a future release". MathWorks recommends `coder.DeepLearningConfig` with target library `"none"` (generic C/C++) — [MathWorks: Networks and layers supported for code generation](https://www.mathworks.com/help/coder/ug/networks-and-layers-supported-for-c-code-generation.html)
- **Supported layers:**
  - convolution2d, FC, LSTM/GRU, and most activations/normalisations are supported
  - convolution1d and grouped conv are generic-C only
  - ARM Compute does not support maxUnpooling2d or ShuffleNet
  - custom ONNX layers (e.g., `nnet.onnx.layer.FlattenLayer`) are listed as supported
  - no `convolution3dLayer` entry appears
  - INT8 for ARM is not mentioned on that page

  — [same page](https://www.mathworks.com/help/coder/ug/networks-and-layers-supported-for-c-code-generation.html)
- **Blockset board list:** an older Raspberry Pi Blockset page listed only Pi 3B/2B/1B+. This conflicts with the current hardware-support page that lists Pi 5. Trust the current page and the release-specific table — [MathWorks sp_RASPPI](https://www.mathworks.com/sp_RASPPI) vs [MathWorks hardware support](https://www.mathworks.com/hardware-support/raspberry-pi.html)
- **Support package currency:** the File Exchange entry shows an update dated 12 Nov 2025 with "Run on Target Hardware" deep-learning deployment — [File Exchange 45145](https://mathworks.com/matlabcentral/fileexchange/45145)
- **Common failure mode:** a forum thread on a failed Pi build. The accepted answer says the setup works only with the hardware support package installed — [MATLAB Answers 472926](https://www.mathworks.com/matlabcentral/answers/472926-code-generation-for-deep-learning-networks-with-arm-compute-library_help)

### Inferences
- **Hybrid plan** (keeps sponsor alignment while protecting the fps target):
  1. Prototype and validate in MATLAB (signal processing, POS/CHROM, `vision.KalmanFilter` + `assignDetectionsToTracks`).
  2. Deploy the real-time Pi pipeline in Python/C++ (picamera2 + YuNet/MediaPipe + ONNX Runtime).
  3. Optionally generate C code with MATLAB Coder for the signal-processing core (bandpass, POS, HR spectral peak, SQI) and call it from Python as a shared library.
- A full-MATLAB deep pipeline on the Pi is feasible only for 2D-CNN rPPG (DeepPhys/TS-CAN-like without TSM ops) and a codegen-compatible detector. MediaPipe and Hailo are not reachable from MATLAB codegen **[inference]**.
- Simulink Raspberry Pi Blockset "External mode" is good for a live demo and for parameter tuning. Judges' complexity scoring will still want measured on-device latency.

### Gaps
- No published fps numbers were found for MATLAB-generated face-detection or rPPG code on a Pi 5.
- It is unconfirmed whether `importNetworkFromONNX` networks with 3D conv or TSM can be code-generated for the generic C/C++ target.
- The status of `faceDetector` (if it exists in R2025b/R2026a) and its codegen support was not verified.

---

## 6. Overall recommended architecture (≥10–30 fps end-to-end on Pi 5) and complexity reporting

### Takeaway
A 30 fps target-locked pipeline is realistic on a Pi 5. The plan:
- detect at a reduced rate on a low-res stream
- track every frame with Kalman + IoU (+ optional KLT)
- extract landmarks only on the target crop
- compute rPPG on ROI-mean traces (POS/CHROM) or with a small streaming/chunked deep model
- run re-ID only on events

Published Pi-class evidence supports this: 30 fps on Pi 5 + Hailo-8L with SCRFD + quantised DeepPhys; per-frame costs of about 40–63 ms on Pi 4 for EfficientPhys-C/TS-CAN, which shrink on Pi 5 and when chunked.

### Cited Findings
- **ISBI 2026 Pi AI Kit system:** SCRFD-2.5G (18.91 ms) + centroid tracker + quantised DeepPhys (28.94 ms) with chunking/batching/incremental stats reaches 29.97 fps average. Detection must stay under 20 ms per frame because resources are shared — [ISBI 2026](https://informatika.stei.itb.ac.id/~rinaldi.munir/Penelitian/Makalah-ISBI-2026.pdf)
- **Face2PPG-RT:** detect every 3 frames + multithreading reaches 103 fps on a laptop. Face detection/alignment dominate cost — [arXiv 2508.18787](https://arxiv.org/html/2508.18787v1)
- **ME-rPPG:** reports Params, memory, latency, and FLOPs side by side. This is a ready template for the "model and time complexity" criterion — [arXiv 2504.01774](https://arxiv.org/html/2504.01774v1)
- **EfficientPhys:** reports preprocessing vs model latency separately per frame on a Pi 4B, which is a useful precedent for per-stage reporting — [arXiv 2110.04447](https://ar5iv.labs.arxiv.org/html/2110.04447)

### Inferences (recommended stack; per-stage budgets are estimates to validate on device)

**Recommended CPU-only option (Pi 5, 8 GB, active cooler, 64-bit Bookworm):**

| Stage | Choice | Rate | Est. Pi 5 cost |
|---|---|---|---|
| Capture | picamera2, 30 fps fixed, AE/AWB/AF locked, main 640x480 + lores 320x240, SensorTimestamp | every frame | about 1–3 ms (copy/convert) |
| Detection | YuNet (OpenCV DNN, ONNX) on lores | every 3–5 frames, or on track loss | about 2–5 ms **[extrapolated from RK3588 2.3 ms]** |
| Tracking | ByteTrack/OC-SORT (Kalman + Hungarian/IoU) | every frame | <1 ms |
| Target lock / re-ID | MobileFaceNet or SFace embedding vs enrolment template, cosine + voting | on track birth/overlap/loss + about 1 Hz | about 12–30 ms per call **[extrapolated]** |
| Landmarks / ROI | MediaPipe Face Landmarker on target crop only (or YuNet 5-pt + geometric ROIs); KLT between runs | 10–30 Hz | about 15–40 ms per call **[unverified]** |
| rPPG | POS/CHROM on ROI means (baseline), plus optional small deep model (EfficientPhys-C/TS-CAN 72x72, ONNX Runtime/TFLite XNNPACK), chunked, 30 frames per 1 s | per frame (means), per chunk (deep) | <1 ms (POS); deep about 15–30 ms per frame-equivalent **[extrapolated]** |
| HR/RR estimation + SQI | bandpass 0.7–3 Hz, Welch/FFT peak, SNR-based quality, resample on SensorTimestamp | 1 Hz | <2 ms |

- **Threads:** run capture, inference, and UI/logging in separate threads or processes (multiprocessing to avoid the GIL), with "latest-frame" queues.
- **Accelerated option (AI Kit / AI HAT+ Hailo-8L):**
  - Move detection (SCRFD-2.5G) to Hailo.
  - Optionally move a DeepPhys-style 2D rPPG net to Hailo (INT8 / mixed UINT4).
  - Keep landmarks and tracking on the CPU.

  This is evidenced at about 30 fps end-to-end. Downsides: complexity, and model-architecture constraints (no 3D conv or 5-D reshape).
- **Complexity and performance reporting checklist for the judges:**
  1. Params and FLOPs/MACs for each network (e.g., `fvcore`/`thop`/`onnx-tool`); for the rPPG model, report FLOPs per output second.
  2. Per-stage latency mean/p50/p95 over at least 1000 frames using `time.perf_counter_ns`, plus end-to-end glass-to-HR latency (frame SensorTimestamp to HR output timestamp).
  3. Sustained fps over 5–10 minutes, with thermal state noted (`vcgencmd measure_temp`, `get_throttled`).
  4. CPU% and RAM (psutil).
  5. Power from an inline USB-C meter: idle vs running, in W.
  6. Accuracy deltas FP32 vs INT8.
  7. Bystander-rejection metrics: target-ID precision, ID-switch count, and time-to-reacquire after occlusion.

### Gaps
- No Pi 5 end-to-end CPU-only rPPG pipeline with published fps was found. All Pi 5 CPU stage costs above are extrapolations and must be measured.
- No published power measurements were found for rPPG on Pi 5 or the AI Kit.
- Hailo-10H / AI HAT+ 2 face and rPPG numbers were not found.
