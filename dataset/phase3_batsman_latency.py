"""
phase3_batsman_latency.py — added cost of each S0-S4 evidence group.

Reported per SAMPLED FRAME PER TRACK where the work is per-track (pose), and
per sampled frame where it is shared (bat detection), because that is how the
cost actually scales: a clip with three candidate tracks pays the pose cost
three times but the bat-detection cost once.

Warmup excluded — the first call loads weights and allocates the CUDA
context, which is one-off setup, not per-frame cost.
"""

import json
import os
import statistics
import time

import cv2
import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")
WARMUP = 8


def timed(fn, items):
    for x in items[:WARMUP]:
        fn(x)
    ts = []
    for x in items:
        t0 = time.perf_counter()
        fn(x)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"mean_ms": round(1000 * statistics.mean(ts), 3),
            "median_ms": round(1000 * statistics.median(ts), 3),
            "p90_ms": round(1000 * ts[int(len(ts) * 0.9)], 3), "n": len(ts)}


def main():
    from phase2_cache_tracks import load_tracks
    from phase2_batsman_score import track_features
    from phase3_semantic_features import (persistence_features, skeleton_features,
                                          equipment_features, action_features,
                                          _pose_metrics, sample_frames, COCO_BAT)
    import zero_storage_pipeline as zsp
    from ultralytics import YOLO

    tracks = load_tracks(os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))

    # Collect real frames + real track boxes to time against.
    frames, crops, obs_list, ctxs = [], [], [], []
    for cid, c in list(tracks["clips"].items())[:8]:
        path = os.path.join(_MODULE_DIR, "raw_videos", cid)
        if not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = set(sample_frames(n, 8))
        grabbed, i = {}, -1
        while i < max(idxs):
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            if i in idxs:
                grabbed[i] = fr
        cap.release()
        ctx = {"frame_w": c["frame_w"], "frame_h": c["frame_h"],
               "frames_processed": c["frames_processed"]}
        for tid, obs in list(c["tracks"].items())[:2]:
            obs_list.append((obs, ctx, n))
            by_frame = {f: b for f, b, _ in obs}
            for f, fr in grabbed.items():
                frames.append(fr)
                box = by_frame.get(f)
                if box is None:
                    continue
                h, w = fr.shape[:2]
                x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
                if x2 > x1 and y2 > y1:
                    crops.append((fr[y1:y2, x1:x2], (x1, y1, x2, y2)))
    frames = frames[:60]
    crops = crops[:60]
    print(f"timing on {len(frames)} frames, {len(crops)} track crops, "
          f"{len(obs_list)} tracks (warmup {WARMUP} excluded)\n")

    out = {}

    # S0 / S1: pure arithmetic on cached tracks.
    out["S0_geometry_features_per_track"] = timed(
        lambda t: track_features(t[0], t[1]["frame_w"], t[1]["frame_h"],
                                 t[1]["frames_processed"]), obs_list)
    out["S1_persistence_features_per_track"] = timed(
        lambda t: persistence_features(t[0], t[1]["frames_processed"], t[2]), obs_list)

    # S2: pose on the track crop — the dominant per-track cost.
    def pose_on_crop(item):
        crop, box = item
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with zsp._DETECTOR_LOCK:
            cands = zsp._detect_candidates(zsp._get_shared_detector_locked(), [rgb])
        return _pose_metrics(cands[0][0] if cands[0] else None, box)

    out["S2_pose_on_crop_per_frame_per_track"] = timed(pose_on_crop, crops)

    seqs = [[_pose_metrics(None, (0, 0, 10, 10))] * 8]
    out["S2_skeleton_math_per_track"] = timed(lambda s: skeleton_features(s), seqs * 30)

    # S3: bat detection — shared across tracks in a frame.
    bat = YOLO("yolo11x.pt")
    out["S3_bat_detection_per_frame_shared"] = timed(
        lambda fr: bat.predict(fr, imgsz=960, classes=[COCO_BAT], conf=0.05,
                               verbose=False, device=0), frames)

    ev = [{"any": True, "on_person": True, "near_hand": True, "conf": 0.4}] * 20
    out["S3_association_math_per_track"] = timed(
        lambda e: equipment_features(e, [], 20), [ev] * 30)
    out["S4_action_math_per_track"] = timed(lambda s: action_features(s), seqs * 30)

    # Cumulative per-clip estimate at the ablation's own sampling rate.
    N_SAMPLE, TRACKS = 20, 5
    pose = out["S2_pose_on_crop_per_frame_per_track"]["median_ms"]
    batd = out["S3_bat_detection_per_frame_shared"]["median_ms"]
    per_clip = {
        "assumptions": f"{N_SAMPLE} sampled frames, {TRACKS} candidate tracks per clip",
        "S0_S1_ms": round((out["S0_geometry_features_per_track"]["median_ms"]
                           + out["S1_persistence_features_per_track"]["median_ms"]) * TRACKS, 1),
        "S2_added_ms": round(pose * N_SAMPLE * TRACKS, 1),
        "S3_added_ms": round(batd * N_SAMPLE, 1),
        "S4_added_ms": round(out["S4_action_math_per_track"]["median_ms"] * TRACKS, 1),
    }
    per_clip["S2_total_ms"] = round(per_clip["S0_S1_ms"] + per_clip["S2_added_ms"], 1)
    per_clip["S3_total_ms"] = round(per_clip["S2_total_ms"] + per_clip["S3_added_ms"], 1)
    per_clip["S4_total_ms"] = round(per_clip["S3_total_ms"] + per_clip["S4_added_ms"], 1)
    out["per_clip_estimate"] = per_clip

    print(f"{'stage':<44}{'mean':>9}{'median':>9}{'p90':>9}")
    for k, v in out.items():
        if isinstance(v, dict) and "median_ms" in v:
            print(f"{k:<44}{v['mean_ms']:>9.2f}{v['median_ms']:>9.2f}{v['p90_ms']:>9.2f}")

    print(f"\nPer-clip estimate ({per_clip['assumptions']}):")
    for k in ("S0_S1_ms", "S2_added_ms", "S3_added_ms", "S4_added_ms",
              "S2_total_ms", "S3_total_ms", "S4_total_ms"):
        print(f"  {k:<16}{per_clip[k]:>10.1f} ms")

    path = os.path.join(P3, "phase3_batsman_latency.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
