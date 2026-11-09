"""Live heart rate from the webcam (or a recorded video), with every processing step on screen.

    python live.py                      # default webcam
    python live.py --simulate mask      # also: lowlight50, lowlight20, lowres, lowfps
    python live.py --log session.csv    # save HR estimates (compare with a watch/oximeter)
    python live.py --video clip.avi --meta clip_meta.txt --ppg clip_ppg.txt
                                        # replay on the true frame times, with the finger-sensor reference
    python live.py --video ... --record demo.mp4 --no-display --schedule "30:mask,50:lowlight20,65:none"
                                        # render a demo video; switch simulations at given seconds

The window (fv/dashboard.py) shows the video with the target face, the skin
patches and any ignored bystanders, the heart-rate card with the quality
model's confidence, and four live graphs in pipeline order: raw skin colour vs
background, the cleaned pulse, the spectrum (standard POS, learned branch,
combined) and the heart rate over time.

Confidence is the offline abstention model (models/abstainer_*.pkl) applied to
the newest window; its "share trend" feature uses the last 11 windows instead
of the 11 centred on the window, which would need the future.

Keys: q quit, m toggle mask simulation, l cycle low-light simulation,
click a face to make it the target (others are ignored).
"""
import argparse
import collections
import os
import subprocess
import time

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np

from fv import dashboard, degrade, extract, face, hr, pipeline, quality

SIM_KEYS = ["none", "mask", "lowlight50", "lowlight20", "lowres", "lowfps"]


def parse_schedule(text: str) -> list[tuple[float, str]]:
    out = []
    for item in filter(None, (p.strip() for p in text.split(","))):
        t, sim = item.split(":")
        if sim not in SIM_KEYS:
            raise SystemExit(f"--schedule: unknown simulation {sim!r} (use {', '.join(SIM_KEYS)})")
        out.append((float(t), sim))
    return sorted(out)


def recorder(path: str, fps: float):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{dashboard.W}x{dashboard.H}",
           "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="replay a video file instead of the camera")
    ap.add_argument("--meta", help="MCD-rPPG meta file with the video's real frame times")
    ap.add_argument("--ppg", help="contact PPG, one sample per video frame (first column), shown as the reference")
    ap.add_argument("--label", help="source description shown in the header")
    ap.add_argument("--simulate", default="none", choices=SIM_KEYS)
    ap.add_argument("--schedule", default="", help='switch simulations at given seconds, e.g. "30:mask,50:none"')
    ap.add_argument("--log", help="CSV file for time, HR, SNR, confidence")
    ap.add_argument("--record", help="write the dashboard to this .mp4 (file replays are then processed frame by frame)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until q or the end)")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--no-learned", action="store_true", help="classical branch only")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. On macOS, allow camera access for the app you run this from "
                         "in System Settings > Privacy & Security > Camera, then retry.")
    stamps = extract.load_stamps(args.meta) if args.meta else None
    src_fps = (1 / np.median(np.diff(stamps[:300]))) if stamps is not None else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    ppg = np.array([float(l.split()[0]) for l in open(args.ppg) if l.strip()]) if args.ppg else None
    if ppg is not None and stamps is None:
        stamps = np.arange(len(ppg)) / src_fps

    # Face detection + landmarks at ~10 Hz in real time, whatever the frame rate.
    every = max(1, round((src_fps / 3 if args.simulate == "lowfps" else src_fps) / 10))
    front = face.FaceFrontEnd(landmark_every=every, detect_every=every)
    clean_lm = face.make_landmarker(1)
    sampler = face.PatchSampler()
    regions = face.ALL_REGIONS
    buf_t, buf_rgb = collections.deque(), collections.deque()
    tracker = hr.OnlineTracker(hop_s=pipeline.HOP_S)
    branch = None
    if os.path.exists("models/pulsenet_all.pt") and not args.no_learned:
        from fv import learned
        branch = learned.Branch(["models/pulsenet_all.pt"])
    abst_path = f"models/abstainer_{'final_dl' if branch else 'final'}.pkl"
    abstainer = quality.Abstainer.load(abst_path) if os.path.exists(abst_path) else None
    rng = np.random.default_rng(0)
    sim = args.simulate
    schedule = parse_schedule(args.schedule)
    sims = ["none", "lowlight50", "lowlight20"]

    source = args.label or (f"Replay: {os.path.basename(args.video)}" if args.video else f"Live webcam {args.camera}")
    dash = dashboard.Dashboard(source, has_reference=ppg is not None, learned_on=branch is not None)
    st = dict(hr=None, conf=None, status="noface", d=None, history=[], ref=None, report_p=quality.REPORT_P)
    shares: list = []
    clicked = {}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (x, y)

    if not args.no_display:
        cv2.namedWindow("FaceVitals", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FaceVitals", dashboard.W, dashboard.H)
        cv2.setMouseCallback("FaceVitals", on_mouse)
    rec = recorder(args.record, src_fps) if args.record else None
    log = open(args.log, "w") if args.log else None
    if log:
        log.write("t_s,hr_bpm,peak_bpm,ref_bpm,snr_db,confidence,reported\n")

    t0 = time.perf_counter()
    last_est = -1.0
    i = 0
    proc_ms = collections.deque(maxlen=60)
    while True:
        ok, frame = cap.read()
        if not ok or (stamps is not None and i >= len(stamps)):
            break
        if stamps is not None:
            now = float(stamps[i])
        elif args.video:
            now = i / src_fps
        else:
            now = time.perf_counter() - t0
        if args.video and not args.record:  # pace file playback like a camera
            wait = now - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
        i += 1
        if args.seconds and now > args.seconds:
            break
        while schedule and now >= schedule[0][0]:
            sim = schedule.pop(0)[1]
        if sim == "lowfps" and (i % 3):
            continue  # ~10 fps
        tp = time.perf_counter()
        t_ms = int(now * 1000) + i

        if "pt" in clicked:  # re-designate the target (click position in dashboard -> frame pixels)
            x, y = clicked.pop("pt")
            fh, fw = frame.shape[:2]
            vx0, vy0, vx1, vy1 = dashboard.VID
            if vx0 <= x < vx1 and vy0 <= y < vy1:
                front.tracker = face.TargetTracker(select_point=((x - vx0) * fw / (vx1 - vx0), (y - vy0) * fh / (vy1 - vy0)))
                buf_t.clear(); buf_rgb.clear(); shares.clear()
                tracker = hr.OnlineTracker(hop_s=pipeline.HOP_S)

        if sim == "mask":
            c = face.detect_faces(clean_lm, frame, t_ms)
            if c:
                frame = face.apply_mask_overlay(frame, c[0], rng)
        elif sim.startswith("lowlight"):
            frame = degrade.low_light(frame, rng, (50 if sim == "lowlight50" else 20) / 300)
        elif sim == "lowres":
            frame = cv2.resize(degrade.low_res(frame), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        pts = front.process(frame, t_ms)
        rgb = sampler.sample(frame, pts)[0] if pts is not None else np.full((len(regions), 3), np.nan, np.float32)
        buf_t.append(now)
        buf_rgb.append(rgb)
        while buf_t and now - buf_t[0] > pipeline.WIN_S + 2:
            buf_t.popleft(); buf_rgb.popleft()

        if now - last_est >= pipeline.HOP_S and len(buf_t) > 30:
            last_est = now
            d = pipeline.live_window(np.array(buf_t), np.array(buf_rgb), regions, learned=branch)
            st["d"] = d
            if d is not None and d["spec"] is not None:
                est = tracker.step(d["spec"])
                snr = hr.snr_db(d["spec"], est)
                conf = float(abstainer.p_good(quality.live_features(d, est, shares))[0]) if abstainer else np.nan
                reported = conf >= quality.REPORT_P if abstainer else snr >= -3.0
                ref = None
                if ppg is not None:
                    n = min(len(ppg), len(stamps))
                    ref = float(pipeline.reference_hr(stamps[:n], ppg[:n], np.array([now]))[0])
                st.update(hr=est, conf=conf, status="ok" if reported else "low", ref=ref)
                st["history"].append(dict(t=now, hr=est, reported=reported, ref=ref))
                if log:
                    peak = hr.BPM[np.argmax(d["spec"])]
                    log.write(f"{now:.2f},{est:.1f},{peak:.0f},{ref if ref is not None else np.nan:.0f},{snr:.2f},{conf:.3f},{int(reported)}\n"); log.flush()
                if args.no_display and not args.record:
                    print(f"t={now:5.1f}s  HR {est:5.1f} bpm  confidence {100 * conf:3.0f}%  "
                          f"{'reported' if reported else 'low confidence'}", flush=True)
        if st["hr"] is None:
            st["status"] = "collect" if pts is not None else "noface"
            st["collect_s"] = buf_t[-1] - buf_t[0] if buf_t else 0
        proc_ms.append((time.perf_counter() - tp) * 1000)

        if args.no_display and not rec:
            continue
        tb = np.array(buf_t)
        rb = np.array(buf_rgb)
        recent = tb > now - pipeline.WIN_S
        st.update(t=now, ms=np.mean(proc_ms), sim=sim, pts=pts, n_faces=front.n_faces, boxes=front.boxes,
                  target=front.tracker.box, raw=(tb[recent], rb[recent, regions.index("forehead")], rb[recent, regions.index("bg")]))
        view = dash.render(frame, pts, st)
        if rec:
            rec.stdin.write(view.tobytes())
        if not args.no_display:
            cv2.imshow("FaceVitals", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("m"):
                sim = "none" if sim == "mask" else "mask"
            if k == ord("l"):
                sim = sims[(sims.index(sim) + 1) % len(sims)] if sim in sims else "lowlight50"
    if log:
        log.close()
    if rec:
        rec.stdin.close()
        rec.wait()
        print(f"saved {args.record}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
