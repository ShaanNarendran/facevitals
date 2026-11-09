# Models

| File | What | Source / licence |
|---|---|---|
| `pulsenet_all.pt` | learned pulse branch, trained on MCD-rPPG subjects 1-60 (deployed) | this project |
| `pulsenet_fold{0-4}.pt` | the same, 5-fold cross-validation models (out-of-fold evaluation) | this project |
| `abstainer_final_dl.pkl`, `abstainer_final.pkl` | quality gate for the fused / classical-only system (scikit-learn pickle) | this project |
| `ablations/` | rejected variants (wider time-stretch, balanced sampling) and the subjects 1-40 model | this project |
| `face_landmarker.task` | MediaPipe Face Landmarker (468-point mesh) | Google MediaPipe, Apache License 2.0 |
| `face_detection_yunet_2023mar.onnx` | YuNet face detector | OpenCV Zoo, MIT License |

Only load the `.pkl` files from this repository: pickle files can run code when loaded.
