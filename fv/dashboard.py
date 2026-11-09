"""Live dashboard: the video with what the pipeline sees, and each processing step as a live graph.

Layout (1280x720): status chips and the annotated video on the left with the
heart-rate card below; on the right four graphs in pipeline order:
  1 raw skin colour vs background (what the camera measures)
  2 cleaned pulse (after lighting correction and POS colour mixing)
  3 spectrum: standard POS, learned branch and their SNR-weighted combination
  4 heart rate over time, coloured by the quality model's confidence
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, find_peaks, sosfiltfilt

from . import face, hr, pipeline

W, H = 1280, 720
VID = (16, 92, 656, 572)  # x0, y0, x1, y1 of the 640x480 video
CARD = (16, 580, 656, 704)
PANELS = [(672, 56 + k * 164, 1264, 56 + k * 164 + 156) for k in range(4)]

# Dark theme, BGR. Series use the reference categorical order (dark steps);
# status colours (good / warning / critical) only ever mean state.
BG, PANEL, GRID = (20, 19, 18), (34, 32, 31), (62, 60, 58)
TEXT, TEXT2, MUTED = (236, 239, 240), (183, 194, 195), (128, 130, 130)
BLUE, ORANGE, AQUA, VIOLET = (229, 135, 57), (38, 89, 217), (112, 158, 25), (233, 133, 144)
GOOD, WARN, BAD = (12, 163, 12), (25, 178, 250), (59, 59, 208)

_FONT_FILES = [("/System/Library/Fonts/HelveticaNeue.ttc", 0, 1),
               ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", None, None)]
_font_cache: dict = {}


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _font_cache:
        f = None
        for path, reg, b in _FONT_FILES:
            if not os.path.exists(path):
                continue
            try:
                if reg is None:
                    path = path.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf") if bold else path
                    f = ImageFont.truetype(path, size)
                else:
                    f = ImageFont.truetype(path, size, index=b if bold else reg)
                break
            except OSError:
                continue
        _font_cache[key] = f or ImageFont.load_default(size)
    return _font_cache[key]


class Dashboard:
    def __init__(self, source: str, has_reference: bool, learned_on: bool):
        self.source, self.has_ref, self.learned_on = source, has_reference, learned_on
        self.texts: list = []
        self.static = self._static()

    # ------------------------------------------------------------ drawing helpers
    def text(self, xy, s, size=13, color=TEXT2, bold=False, anchor="la"):
        self.texts.append((xy, s, size, color, bold, anchor))

    def _flush_text(self, img):
        pil = Image.fromarray(img)  # BGR array drawn with BGR colours: channels stay consistent
        d = ImageDraw.Draw(pil)
        for xy, s, size, color, bold, anchor in self.texts:
            d.text(xy, s, font=font(size, bold), fill=color, anchor=anchor)
        self.texts = []
        return np.asarray(pil).copy()

    @staticmethod
    def line(img, xs, ys, area, xlim, ylim, color, thick=2, dashed=False):
        x0, y0, x1, y1 = area
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        px = x0 + (xs - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
        py = y1 - (np.clip(ys, *ylim) - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)
        ok = np.isfinite(px) & np.isfinite(py)
        if ok.sum() < 2:
            return
        pts = np.round(np.stack([px[ok], py[ok]], 1) * 16).astype(np.int32)
        if not dashed:
            cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA, shift=4)
            return
        for k in range(0, len(pts) - 1, 4):
            cv2.polylines(img, [pts[k:k + 3]], False, color, thick, cv2.LINE_AA, shift=4)

    @staticmethod
    def to_px(v, lim, a, b):
        return int(round(a + (v - lim[0]) / (lim[1] - lim[0]) * (b - a)))

    def _static(self):
        img = np.full((H, W, 3), BG, np.uint8)
        self.text((20, 14), "FaceVitals", 22, TEXT, True)
        self.text((140, 20), "contactless heart rate from a face video", 15, TEXT2)
        self.text((W - 20, 20), self.source, 14, TEXT2, anchor="ra")
        cv2.rectangle(img, VID[:2], VID[2:], PANEL, -1)
        cv2.rectangle(img, CARD[:2], CARD[2:], PANEL, -1)
        titles = [("1  What the camera measures", "forehead colour vs the background, last 10 s (light flicker above 4 Hz smoothed)"),
                  ("2  Pulse signal after cleaning", "lighting removed, colours mixed to bring out the pulse (POS)"),
                  ("3  Which beat rate is strongest", "spectrum of the last 10 s; the peak is the heart rate"),
                  ("4  Heart rate over time", "each dot is one 10 s window, updated every second")]
        for (x0, y0, x1, y1), (t, sub) in zip(PANELS, titles):
            cv2.rectangle(img, (x0, y0), (x1, y1), PANEL, -1)
            self.text((x0 + 12, y0 + 8), t, 15, TEXT, True)
            self.text((x0 + 12, y0 + 28), sub, 12, MUTED)
        self.text((CARD[0] + 16, CARD[1] + 12), "HEART RATE", 12, MUTED, True)
        self.text((CARD[0] + 268, CARD[1] + 12), "CONFIDENCE (quality model)", 12, MUTED, True)
        if self.has_ref:
            self.text((CARD[0] + 500, CARD[1] + 12), "FINGER SENSOR", 12, MUTED, True)
        return self._flush_text(img)

    @staticmethod
    def plot_area(panel):
        x0, y0, x1, y1 = panel
        return (x0 + 52, y0 + 50, x1 - 14, y1 - 22)

    def grid(self, img, area, xlim, ylim, xticks, yticks, xfmt, yfmt):
        x0, y0, x1, y1 = area
        for v in yticks:
            y = self.to_px(v, ylim, y1, y0)
            cv2.line(img, (x0, y), (x1, y), GRID, 1)
            self.text((x0 - 6, y), yfmt(v), 11, MUTED, anchor="rm")
        for v in xticks:
            x = self.to_px(v, xlim, x0, x1)
            self.text((x, y1 + 5), xfmt(v), 11, MUTED, anchor="ma")

    def legend(self, items, x, y):
        """items: (label, colour), laid out right to left so the last one ends at x."""
        for label, col in reversed(items):
            w = int(font(11).getlength(label))
            self.text((x, y), label, 11, TEXT2, anchor="ra")
            self._legend_marks.append(((x - w - 16, y + 7), (x - w - 4, y + 7), col))
            x -= w + 30

    # ------------------------------------------------------------ main entry
    def render(self, frame, pts, s: dict) -> np.ndarray:
        img = self.static.copy()
        self._legend_marks = []
        self._video(img, frame, pts, s)
        self._chips(img, s)
        self._card(img, s)
        self._raw_panel(img, s)
        self._pulse_panel(img, s)
        self._spectrum_panel(img, s)
        self._history_panel(img, s)
        for a, b, col in self._legend_marks:
            cv2.line(img, a, b, col, 2, cv2.LINE_AA)
        return self._flush_text(img)

    # ------------------------------------------------------------ pieces
    def _video(self, img, frame, pts, s):
        x0, y0, x1, y1 = VID
        fh, fw = frame.shape[:2]
        sx, sy = (x1 - x0) / fw, (y1 - y0) / fh
        v = cv2.resize(frame, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA) if (fw, fh) != (x1 - x0, y1 - y0) else frame.copy()
        d = s.get("d") or {}
        masked = d.get("masked", False)
        target = s.get("target")
        for b in s.get("boxes", []):
            if target is not None and face.iou(b, target) > 0.3:
                continue
            p, q = (int(b[0] * sx), int(b[1] * sy)), (int(b[2] * sx), int(b[3] * sy))
            cv2.rectangle(v, p, q, MUTED, 1, cv2.LINE_AA)
            self.text((x0 + p[0] + 4, y0 + p[1] - 18), "other person: ignored", 12, MUTED)
        if pts is not None:
            P = pts * np.array([sx, sy], np.float32)
            if not masked:
                cv2.polylines(v, [np.round(P[face.FACE_OVAL] * 16).astype(np.int32)], True, BLUE, 2, cv2.LINE_AA, shift=4)
            occ, wts = d.get("occ", {}), d.get("weights", {})
            for name, idx in face.PATCHES.items():
                if occ.get(name):
                    col, th = BAD, 2
                elif masked:
                    col, th = GOOD, 1 + int(round(2 * wts.get(name, 0) / max(max(wts.values(), default=1), 1e-9)))
                else:
                    col, th = TEXT2, 1
                cv2.polylines(v, [np.round(P[idx] * 16).astype(np.int32)], True, col, th, cv2.LINE_AA, shift=4)
        if target is not None:
            b = target
            p, q = (int(b[0] * sx), int(b[1] * sy)), (int(b[2] * sx), int(b[3] * sy))
            cv2.rectangle(v, p, q, TEXT, 1, cv2.LINE_AA)
            cv2.rectangle(v, (p[0], p[1] - 20), (p[0] + 64, p[1]), TEXT, -1)
            self.text((x0 + p[0] + 6, y0 + p[1] - 18), "TARGET", 12, BG, True)
        img[y0:y1, x0:x1] = v
        sim = s.get("sim", "none")
        if sim != "none":
            label = {"mask": "Simulated: surgical mask", "lowlight50": "Simulated: 50 lux (dim room)",
                     "lowlight20": "Simulated: 20 lux (very dim)", "lowres": "Simulated: 320x240 camera",
                     "lowfps": "Simulated: 10 fps stream"}.get(sim, sim)
            w = int(font(13, True).getlength(label)) + 20
            cv2.rectangle(img, (x0 + 10, y0 + 10), (x0 + 10 + w, y0 + 34), WARN, -1)
            self.text((x0 + 20, y0 + 14), label, 13, BG, True)
        cv2.rectangle(img, (x0, y1 - 26), (x1, y1), (0, 0, 0), -1)
        key = ("Blue = whole face (standard method); grey = patches for the AI branch; red = covered"
               if not masked else "Mask found: only uncovered patches used (thicker = more weight)")
        self.text((x0 + 8, y1 - 21), key, 12, TEXT2)
        self.text((x1 - 8, y1 - 21), f"t = {s.get('t', 0):.0f} s   {s.get('ms', 0):.0f} ms/frame", 12, TEXT2, anchor="ra")

    def _chips(self, img, s):
        d = s.get("d") or {}
        found = s.get("pts") is not None
        others = max(0, s.get("n_faces", 0) - 1)
        chips = [("Face found" if found else "Looking for a face", "ok" if found else "warn")]
        if found:
            chips.append(("Target locked" + (f", {others} other ignored" if others else ""), "ok"))
        chips.append(("Mask found: upper face only" if d.get("masked") else "No mask", "warn" if d.get("masked") else "ok"))
        chips.append(("Lighting correction on", "ok"))
        chips.append(("AI branch on" if self.learned_on else "AI branch off", "ok" if self.learned_on else "off"))
        x, y = VID[0], 56
        for label, state in chips:
            col = {"ok": GOOD, "warn": WARN, "off": MUTED}[state]
            w = int(font(12).getlength(label)) + 28
            if x + w > VID[2]:
                break
            cv2.rectangle(img, (x, y), (x + w, y + 26), PANEL, -1)
            cv2.rectangle(img, (x, y), (x + w, y + 26), col, 1, cv2.LINE_AA)
            cv2.circle(img, (x + 11, y + 13), 4, col, -1, cv2.LINE_AA)
            self.text((x + 20, y + 6), label, 12, TEXT)
            x += w + 6

    def _card(self, img, s):
        x0, y0, x1, y1 = CARD
        status = s.get("status")
        if status in ("ok", "low"):
            col = TEXT if status == "ok" else WARN
            self.text((x0 + 16, y0 + 30), f"{s['hr']:.0f}", 54, col, True)
            self.text((x0 + 16 + int(font(54, True).getlength(f"{s['hr']:.0f}")) + 8, y0 + 62), "bpm", 18, TEXT2)
            msg = "Reported" if status == "ok" else "Low confidence: not reported"
            self.text((x0 + 16, y0 + 98), msg, 13, GOOD if status == "ok" else WARN, True)
        elif status == "collect":
            frac = min(1.0, s.get("collect_s", 0) / pipeline.WIN_S)
            self.text((x0 + 16, y0 + 36), "Collecting signal...", 20, TEXT, True)
            cv2.rectangle(img, (x0 + 16, y0 + 74), (x0 + 236, y0 + 84), GRID, -1)
            cv2.rectangle(img, (x0 + 16, y0 + 74), (x0 + 16 + int(220 * frac), y0 + 84), BLUE, -1)
            self.text((x0 + 16, y0 + 92), f"{s.get('collect_s', 0):.0f} of {pipeline.WIN_S:.0f} s needed", 12, MUTED)
        else:
            self.text((x0 + 16, y0 + 40), "Looking for a face...", 20, TEXT, True)
        # Confidence bar with the reporting threshold.
        bx0, bx1, by = x0 + 268, x0 + 468, y0 + 44
        cv2.rectangle(img, (bx0, by), (bx1, by + 14), GRID, -1)
        conf = s.get("conf")
        if conf is not None and status in ("ok", "low"):
            cv2.rectangle(img, (bx0, by), (bx0 + int((bx1 - bx0) * conf), by + 14), GOOD if conf >= s["report_p"] else WARN, -1)
            self.text((bx0, by + 22), f"{100 * conf:.0f}% chance the error is within 5 bpm", 12, TEXT2)
        tx = bx0 + int((bx1 - bx0) * s.get("report_p", 0.8))
        cv2.line(img, (tx, by - 5), (tx, by + 19), TEXT, 1)
        self.text((tx, by - 8), "report", 10, MUTED, anchor="md")
        if self.has_ref:
            ref = s.get("ref")
            if ref is not None and np.isfinite(ref):
                self.text((x0 + 500, y0 + 30), f"{ref:.0f}", 40, VIOLET, True)
                self.text((x0 + 500 + int(font(40, True).getlength(f"{ref:.0f}")) + 6, y0 + 52), "bpm", 15, TEXT2)
                if status in ("ok", "low"):
                    self.text((x0 + 500, y0 + 84), f"difference {s['hr'] - ref:+.0f} bpm", 13, TEXT2)
                self.text((x0 + 500, y0 + 102), "true value (not used)", 11, MUTED)

    def _raw_panel(self, img, s):
        area = self.plot_area(PANELS[0])
        raw = s.get("raw")
        xlim = (-pipeline.WIN_S, 0)
        if raw is None or len(raw[0]) < 10:
            self.text((area[0] + 8, area[1] + 30), "waiting for the face...", 13, MUTED)
            return
        t, f, b = raw
        t = t - s["t"]
        series = []
        fs = 1 / np.median(np.diff(t)) if len(t) > 1 else 30.0
        # Mains flicker aliases to ~10 Hz at 30 fps and can be 10x the pulse; smooth it for display only.
        sos = butter(2, min(4.0 / (fs / 2), 0.9), output="sos")
        for x in (f[:, 1], b.mean(axis=1)):  # forehead: green channel (strongest pulse); background: brightness
            ok = np.isfinite(x)
            y = np.full(len(x), np.nan)
            if ok.sum() > 20:
                y[ok] = 100 * (x[ok] / x[ok].mean() - 1)
                y[ok] -= np.polyval(np.polyfit(t[ok], y[ok], 1), t[ok])
                y[ok] = sosfiltfilt(sos, y[ok])
            series.append(y)
        lim = max(0.3, np.nanpercentile(np.abs(np.concatenate(series)), 98) * 1.15) if np.isfinite(series[0]).any() else 1
        ylim = (-lim, lim)
        tick = max(v for v in (0.1, 0.2, 0.5, 1, 2, 5, 10) if v <= 0.75 * lim)
        self.grid(img, area, xlim, ylim, [-10, -5, 0], [-tick, 0, tick],
                  lambda v: "now" if v == 0 else f"{v:.0f} s", lambda v: f"{v:+.1f}%" if v else "0")
        self.line(img, t, series[1], area, xlim, ylim, MUTED, 1)
        self.line(img, t, series[0], area, xlim, ylim, AQUA, 2)
        self.legend([("forehead skin (green)", AQUA), ("background", MUTED)], PANELS[0][2] - 14, PANELS[0][1] + 8)

    def _pulse_panel(self, img, s):
        area = self.plot_area(PANELS[1])
        d = s.get("d") or {}
        p = d.get("pulse")
        xlim = (-pipeline.WIN_S, 0)
        if p is None:
            self.text((area[0] + 8, area[1] + 30), "needs 10 s of face video", 13, MUTED)
            return
        t = d["t"][-len(p):] - s["t"]
        z = (p - p.mean()) / (p.std() + 1e-9)
        ylim = (-3, 3)
        self.grid(img, area, xlim, ylim, [-10, -5, 0], [-2, 0, 2],
                  lambda v: "now" if v == 0 else f"{v:.0f} s", lambda v: "")
        self.line(img, t, z, area, xlim, ylim, BLUE, 2)
        pk, _ = find_peaks(z, distance=int(pipeline.FS * 60 / 200), prominence=0.8)
        for k in pk:
            x = self.to_px(t[k], xlim, area[0], area[2])
            y = self.to_px(np.clip(z[k], *ylim), ylim, area[3], area[1])
            cv2.circle(img, (x, y), 3, TEXT, -1, cv2.LINE_AA)
        label = "best uncovered patch" if d.get("masked") else "whole face, POS"
        self.legend([(label, BLUE), ("detected beats", TEXT)], PANELS[1][2] - 14, PANELS[1][1] + 8)

    def _spectrum_panel(self, img, s):
        area = self.plot_area(PANELS[2])
        d = s.get("d") or {}
        xlim, ylim = (40, 200), (0, 1.08)
        self.grid(img, area, xlim, ylim, [60, 90, 120, 150, 180], [0, 0.5, 1],
                  lambda v: f"{v:.0f} bpm" if v == 180 else f"{v:.0f}", lambda v: "")
        if d.get("spec") is None:
            self.text((area[0] + 8, area[1] + 30), "needs 10 s of face video", 13, MUTED)
            return
        norm = lambda S: S / (S.max() + 1e-12)
        items = []
        if not d.get("masked"):
            if d.get("S_pos") is not None:
                self.line(img, hr.BPM, norm(d["S_pos"]), area, xlim, ylim, BLUE, 1)
                items.append(("standard (POS)", BLUE))
            if d.get("S_dl") is not None:
                self.line(img, hr.BPM, norm(d["S_dl"]), area, xlim, ylim, ORANGE, 1)
                items.append(("AI branch", ORANGE))
        self.line(img, hr.BPM, norm(d["spec"]), area, xlim, ylim, TEXT, 2)
        items.append(("combined" if not d.get("masked") else "uncovered patches", TEXT))
        if s.get("hr") is not None:
            x = self.to_px(s["hr"], xlim, area[0], area[2])
            cv2.line(img, (x, area[1]), (x, area[3]), TEXT, 1)
            self.text((x + 4, area[1] - 2), f"estimate {s['hr']:.0f}", 11, TEXT, True)
        ref = s.get("ref")
        if ref is not None and np.isfinite(ref):
            x = self.to_px(ref, xlim, area[0], area[2])
            for yy in range(area[1], area[3], 8):
                cv2.line(img, (x, yy), (x, min(yy + 4, area[3])), VIOLET, 1)
            self.text((x + 4, area[1] + 12), f"sensor {ref:.0f}", 11, VIOLET)
        self.legend(items, PANELS[2][2] - 14, PANELS[2][1] + 8)

    def _history_panel(self, img, s):
        area = self.plot_area(PANELS[3])
        hist = [h for h in s.get("history", []) if s["t"] - h["t"] <= 60]
        xlim = (-60, 0)
        vals = [h["hr"] for h in hist] + [h["ref"] for h in hist if h.get("ref") is not None and np.isfinite(h["ref"])]
        lo, hi = (min(vals), max(vals)) if vals else (60, 100)
        mid, half = (lo + hi) / 2, max(15, (hi - lo) / 2 + 8)
        ylim = (max(40, mid - half), min(200, mid + half))
        step = 10 if ylim[1] - ylim[0] <= 60 else 20
        yt = [v for v in range(int(np.ceil(ylim[0] / step) * step), int(ylim[1]) + 1, step)]
        self.grid(img, area, xlim, ylim, [-60, -45, -30, -15, 0], yt,
                  lambda v: "now" if v == 0 else f"{v:.0f} s", lambda v: f"{v:.0f}")
        if not hist:
            self.text((area[0] + 8, area[1] + 30), "first reading after 10 s", 13, MUTED)
            return
        t = np.array([h["t"] for h in hist]) - s["t"]
        if self.has_ref:
            self.line(img, t, [h.get("ref", np.nan) for h in hist], area, xlim, ylim, VIOLET, 2, dashed=True)
        self.line(img, t, [h["hr"] for h in hist], area, xlim, ylim, MUTED, 1)
        for tt, h in zip(t, hist):
            x = self.to_px(tt, xlim, area[0], area[2])
            y = self.to_px(np.clip(h["hr"], *ylim), ylim, area[3], area[1])
            cv2.circle(img, (x, y), 3, GOOD if h["reported"] else WARN, -1, cv2.LINE_AA)
        items = [("reported", GOOD), ("low confidence", WARN)] + ([("finger sensor", VIOLET)] if self.has_ref else [])
        self.legend(items, PANELS[3][2] - 14, PANELS[3][1] + 8)
