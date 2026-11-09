"""Face landmarks, skin patches, target tracking and per-patch colour traces."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")

# MediaPipe 468-point mesh indices. Patches are polygons in landmark order.
PATCHES: dict[str, list[int]] = {
    # Hairline/face-oval top down to the upper edge of both eyebrows.
    "forehead": [54, 103, 67, 109, 10, 338, 297, 332, 284,
                 300, 293, 334, 296, 336, 9, 107, 66, 105, 63, 70],
    # Between the brows, above the nose bridge.
    "glabella": [107, 9, 336, 285, 417, 168, 193, 55],
    # Upper/malar cheeks just below the eyes (survive a surgical mask).
    "cheek_r": [116, 117, 118, 119, 100, 126, 209, 49, 129, 203, 205, 50],
    "cheek_l": [345, 346, 347, 348, 329, 355, 429, 279, 358, 423, 425, 280],
    # Lower cheeks beside the mouth (covered by a mask).
    "lowcheek_r": [205, 207, 187, 147, 213, 192, 214, 212, 216, 206],
    "lowcheek_l": [425, 427, 411, 376, 433, 416, 434, 432, 436, 426],
}
UPPER_PATCHES = ("forehead", "glabella", "cheek_r", "cheek_l")

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]

# Lower-face polygon used to simulate a surgical mask (C7): from mid nose bridge
# (197) across both cheeks (93 / 323) and down the jaw to the chin.
MASK_POLY = [93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379,
             365, 397, 288, 361, 323, 280, 197, 50]

# "bg": non-skin areas beside the face (wall / hair), used as an illumination
# reference to cancel light flicker common to skin and background.
ALL_REGIONS = list(PATCHES) + ["face", "bg"]


def make_landmarker(num_faces: int = 4, live: bool = False):
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=num_faces,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def detect_faces(landmarker, frame_bgr: np.ndarray, t_ms: int) -> list[np.ndarray]:
    """Return a list of (468, 2) pixel landmark arrays, one per detected face."""
    import mediapipe as mp

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
    faces = []
    for lms in res.face_landmarks:
        pts = np.array([[p.x * w, p.y * h] for p in lms[:468]], dtype=np.float32)
        faces.append(pts)
    return faces


def bbox(pts: np.ndarray) -> np.ndarray:
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


@dataclass
class TargetTracker:
    """Keeps one designated face across frames and ignores bystanders (C3).

    The target is chosen on the first detection (largest face, or the face
    nearest a given point). Afterwards each detection round picks the box that
    best overlaps the target's predicted box; constant-velocity prediction
    bridges short misses. If no box is close enough the frame is a gap, never a
    switch to another person.
    """

    select_point: tuple[float, float] | None = None
    max_gap: int = 5
    box: np.ndarray | None = None
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4))
    missed: int = 0

    def update(self, boxes: list[np.ndarray]) -> np.ndarray | None:
        if self.box is None:
            if not boxes:
                return None
            if self.select_point is not None:
                cx, cy = self.select_point
                k = int(np.argmin([np.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy) for b in boxes]))
            else:
                k = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
            self.box = boxes[k]
            return self.box

        pred = self.box + self.vel
        scores = [iou(pred, b) for b in boxes]
        if scores and max(scores) > 0.3:
            new = boxes[int(np.argmax(scores))]
            self.vel = 0.7 * self.vel + 0.3 * (new - self.box)
            self.box, self.missed = new, 0
            return self.box

        self.missed += 1
        self.box = pred
        if self.missed > self.max_gap and boxes:
            # Lost for several detection rounds: re-acquire only a face close to
            # where the target was, so a bystander elsewhere is never adopted.
            c = (self.box[:2] + self.box[2:]) / 2
            d = [np.hypot(*((b[:2] + b[2:]) / 2 - c)) for b in boxes]
            k = int(np.argmin(d))
            if d[k] < 0.75 * (self.box[2] - self.box[0]):
                self.box, self.vel, self.missed = boxes[k], np.zeros(4), 0
                return self.box
        return None

    def refine(self, pts: np.ndarray) -> None:
        """Snap the box to this frame's landmarks (tracking between detections)."""
        self.box = bbox(pts)


YUNET_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_detection_yunet_2023mar.onnx")


class FaceFrontEnd:
    """Two-stage face front end: YuNet finds all faces, the tracker keeps the
    designated target, and MediaPipe landmarks run only on the target's crop.

    Running the landmarker on a crop keeps small/distant faces usable (the
    MediaPipe detector alone misses faces in wide multi-person frames) and its
    cost independent of how many people are in view.
    """

    def __init__(self, select_point=None, detect_every: int = 3, det_scale: float = 0.5,
                 score_threshold: float = 0.6, landmark_every: int = 1):
        self.det = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 240), score_threshold)
        self.lm = make_landmarker(num_faces=1)
        self.tracker = TargetTracker(select_point)
        self.detect_every, self.det_scale = detect_every, det_scale
        self.landmark_every = landmark_every
        self.i = 0
        self.n_faces = 0
        self.boxes: list[np.ndarray] = []  # all faces found in the latest detection round
        self.last_pts: np.ndarray | None = None
        self.last_box: np.ndarray | None = None

    def detect(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        s = self.det_scale
        small = cv2.resize(frame_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else frame_bgr
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, f = self.det.detect(small)
        if f is None:
            return []
        return [np.array([x, y, x + w, y + h]) / s for x, y, w, h in f[:, :4]]

    def process(self, frame_bgr: np.ndarray, t_ms: int) -> np.ndarray | None:
        """Return (468, 2) target landmarks in frame pixels, or None."""
        box = None
        if self.i % self.detect_every == 0 or self.tracker.box is None or self.tracker.missed:
            boxes = self.detect(frame_bgr)
            self.n_faces, self.boxes = len(boxes), boxes
            box = self.tracker.update(boxes)
        else:
            box = self.tracker.box
        self.i += 1
        if box is None:
            self.last_pts = None
            return None
        # Between landmark refreshes, reuse the last mesh shifted with the box.
        if self.last_pts is not None and (self.i - 1) % self.landmark_every:
            shift = ((box[:2] + box[2:]) - (self.last_box[:2] + self.last_box[2:])) / 2
            return self.last_pts + shift.astype(np.float32)

        h, w = frame_bgr.shape[:2]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        side = 1.8 * max(box[2] - box[0], box[3] - box[1])
        x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        x1, y1 = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        faces = detect_faces(self.lm, crop, t_ms)
        if not faces:
            self.last_pts = None
            return None
        pts = faces[0] + np.array([x0, y0], np.float32)
        self.tracker.refine(pts)
        self.last_pts, self.last_box = pts, self.tracker.box.copy()
        return pts

    def close(self):
        self.lm.close()


class PatchSampler:
    """Mean RGB of each skin patch, ignoring clipped pixels."""

    def __init__(self):
        self.mask_buf: np.ndarray | None = None

    def sample(self, frame_bgr: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (R, 3) mean RGB and (R,) pixel counts for ALL_REGIONS."""
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = bbox(pts).astype(int)
        x0, y0 = max(x0 - 2, 0), max(y0 - 2, 0)
        x1, y1 = min(x1 + 2, w), min(y1 + 2, h)
        crop = frame_bgr[y0:y1, x0:x1]
        out = np.full((len(ALL_REGIONS), 3), np.nan, dtype=np.float32)
        cnt = np.zeros(len(ALL_REGIONS), dtype=np.int32)
        if crop.size == 0:
            return out, cnt
        valid = (crop.max(axis=2) < 250) & (crop.min(axis=2) > 3)
        local = pts - np.array([x0, y0], dtype=np.float32)
        mask = np.zeros(crop.shape[:2], np.uint8)
        for i, name in enumerate(ALL_REGIONS):
            if name == "bg":
                out[i], cnt[i] = self._background(frame_bgr, pts)
                continue
            idx = FACE_OVAL if name == "face" else PATCHES[name]
            mask[:] = 0
            cv2.fillPoly(mask, [np.round(local[idx]).astype(np.int32)], 1)
            m = (mask > 0) & valid
            n = int(m.sum())
            if n >= 8:
                out[i] = crop[m].mean(axis=0)[::-1]  # BGR -> RGB
                cnt[i] = n
        return out, cnt


    @staticmethod
    def _background(frame_bgr: np.ndarray, pts: np.ndarray):
        """Mean RGB of two boxes left and right of the face at eye level."""
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = bbox(pts)
        fw, fh = x1 - x0, y1 - y0
        ya, yb = int(max(y0 + 0.1 * fh, 0)), int(min(y0 + 0.6 * fh, h))
        acc, n = np.zeros(3), 0
        for xa, xb in ((x0 - 0.5 * fw, x0 - 0.15 * fw), (x1 + 0.15 * fw, x1 + 0.5 * fw)):
            xa, xb = int(max(xa, 0)), int(min(xb, w))
            if xb - xa < 4 or yb - ya < 4:
                continue
            box = frame_bgr[ya:yb, xa:xb].reshape(-1, 3)
            ok = (box.max(axis=1) < 250) & (box.min(axis=1) > 3)
            acc += box[ok].sum(axis=0)
            n += int(ok.sum())
        if n < 16:
            return np.full(3, np.nan, np.float32), 0
        return (acc / n)[::-1].astype(np.float32), n


def draw_patches(frame_bgr: np.ndarray, pts: np.ndarray, weights: dict[str, float] | None = None) -> np.ndarray:
    img = frame_bgr.copy()
    for name, idx in PATCHES.items():
        wgt = 1.0 if weights is None else weights.get(name, 0.0)
        color = (0, int(80 + 175 * wgt), int(255 * (1 - wgt)))
        cv2.polylines(img, [np.round(pts[idx]).astype(np.int32)], True, color, 1, cv2.LINE_AA)
    return img


def apply_mask_overlay(frame_bgr: np.ndarray, pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paint a light-blue surgical-mask polygon over the lower face (C7 simulation)."""
    img = frame_bgr.copy()
    poly = np.round(pts[MASK_POLY]).astype(np.int32)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [poly], 255)
    tex = np.clip(rng.normal(0, 4, img.shape) + np.array([225, 205, 170]), 0, 255).astype(np.uint8)
    img[m > 0] = tex[m > 0]
    return img


def ita_degrees(rgb_mean: np.ndarray) -> float:
    """Individual Typology Angle from a mean sRGB skin colour (0-255)."""
    px = (np.asarray(rgb_mean, np.float32).reshape(1, 1, 3) / 255.0).astype(np.float32)
    L, _a, b = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0]
    return float(np.degrees(np.arctan2(L - 50.0, b)))


def ita_category(ita: float) -> str:
    """Chardon et al. ITA bands, commonly mapped to Fitzpatrick I-VI."""
    if ita > 55:
        return "very light (I)"
    if ita > 41:
        return "light (II)"
    if ita > 28:
        return "intermediate (III)"
    if ita > 10:
        return "tan (IV)"
    if ita > -30:
        return "brown (V)"
    return "dark (VI)"
