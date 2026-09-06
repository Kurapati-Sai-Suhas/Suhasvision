"""
phase2_latency.py — per-stage latency, measured cleanly.

Warmup is excluded (first call loads weights, allocates CUDA context and
triggers cuDNN autotuning; including it would misattribute one-off setup to
per-frame cost). Stages are timed separately so the report can say where the
time goes rather than only what the total is.
"""

import argparse
import json
import os
import statistics
import time

import cv2

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_MODULE_DIR, "phase2_results")

WARMUP = 10


def _sample_frames(video_dir, n_clips=12, per_clip=8):
    import glob
    frames = []
    for p in sorted(glob.glob(os.path.join(video_dir, "*.mp4")))[:n_clips * 2:2]:
        cap = cv2.VideoCapture(p)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n < per_clip:
            cap.release()
            continue
        for i in range(per_clip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * (n - 1) / max(per_clip - 1, 1)))
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
        if len(frames) >= n_clips * per_clip:
            break
    return frames


def time_stage(fn, frames, warmup=WARMUP):
    for f in frames[:warmup]:
        fn(f)
    times = []
    for f in frames:
        t0 = time.perf_counter()
        fn(f)
        times.append(time.perf_counter() - t0)
    times.sort()
    return {"mean_ms": round(1000 * statistics.mean(times), 2),
            "median_ms": round(1000 * statistics.median(times), 2),
            "p90_ms": round(1000 * times[int(len(times) * 0.9)], 2),
            "n": len(times)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "phase2_latency.json"))
    args = ap.parse_args()

    from phase2_detectors import build_detector
    import phase2_tracking as T

    frames = _sample_frames(args.videos)
    print(f"timing on {len(frames)} real frames (warmup {WARMUP} excluded)\n")

    out = {}

    mp_det = build_detector("mediapipe")
    out["B0_mediapipe_detection"] = time_stage(lambda f: mp_det.detect(f), frames)

    yolo = build_detector("yolo11m@640")
    out["yolo11m@640_detection"] = time_stage(lambda f: yolo.detect(f), frames)

    model = T.build_tracking_model("yolo11m.pt")
    model.track(frames[0], persist=True, tracker="bytetrack.yaml", classes=[0],
                imgsz=640, conf=0.10, verbose=False)
    out["yolo11m@640_detection_plus_bytetrack"] = time_stage(
        lambda f: model.track(f, persist=True, tracker="bytetrack.yaml", classes=[0],
                              imgsz=640, conf=0.10, verbose=False), frames)

    # Pose on a crop: the top-down path B5 tests. Crop is a fixed central
    # region so this measures pose cost, not detection cost.
    import zero_storage_pipeline as zsp

    def pose_on_crop(f):
        h, w = f.shape[:2]
        crop = f[int(0.1 * h):int(0.9 * h), int(0.3 * w):int(0.7 * w)]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with zsp._DETECTOR_LOCK:
            return zsp._detect_candidates(zsp._get_shared_detector_locked(), [rgb])

    out["mediapipe_pose_on_crop"] = time_stage(pose_on_crop, frames)

    d = out["yolo11m@640_detection"]["median_ms"]
    dt = out["yolo11m@640_detection_plus_bytetrack"]["median_ms"]
    out["bytetrack_overhead_ms"] = round(dt - d, 2)
    out["recommended_total_median_ms"] = round(dt + out["mediapipe_pose_on_crop"]["median_ms"], 2)
    out["baseline_B0_total_median_ms"] = out["B0_mediapipe_detection"]["median_ms"]

    print(f"{'stage':<40} {'mean':>8} {'median':>8} {'p90':>8}")
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"{k:<40} {v['mean_ms']:>8.2f} {v['median_ms']:>8.2f} {v['p90_ms']:>8.2f}")
    print(f"\nByteTrack association overhead : {out['bytetrack_overhead_ms']:.2f} ms/frame")
    print(f"Recommended stack (det+track+pose): {out['recommended_total_median_ms']:.2f} ms/frame")
    print(f"B0 baseline (MediaPipe detect)   : {out['baseline_B0_total_median_ms']:.2f} ms/frame")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
