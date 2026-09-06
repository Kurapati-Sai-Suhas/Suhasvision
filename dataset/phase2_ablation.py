"""
phase2_ablation.py — B0..B5, each arm adding exactly one thing.

    B0  MediaPipe candidates          + Phase-1 size-dominance selection
    B1  YOLO11m@640 candidates        + Phase-1 size-dominance selection
    B2  YOLO11m@640 + ByteTrack       + Phase-1 size-dominance over TRACKS
    B3  B2 + geometry-only fitted scorer   (crease/stability features)
    B4  B3 + full fitted scorer            (adds size, centrality, coverage)
    B5  B4 + MediaPipe pose on the selected crop (top-down), pose success only

B0 -> B1 isolates the DETECTOR. B1 -> B2 isolates TRACKING. B2 -> B3 isolates
CRICKET GEOMETRY replacing size dominance. B3 -> B4 isolates the remaining
appearance/position features. B5 tests whether the existing pose model is
sufficient once detection is fixed, before considering RTMPose (§19).

B0 and B1 share one association routine so that arm difference is purely the
detector. It is a faithful port of Phase 1's rule: link candidates across
frames by centre proximity gated in body-height units, require coverage, then
demand size dominance or refuse.

EVALUATION
Per clip: is the selected subject the annotated batsman? Measured on the same
6 annotated frames for every arm. Refusals are counted separately from wrong
answers throughout, because "rejected the clip" and "silently scored the
bowler" are not the same outcome and averaging them hides the whole point.
"""

import argparse
import json
import math
import os
import time

import cv2

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_MODULE_DIR, "phase2_results")

IOU_MATCH = 0.5
GATE_BODY_HEIGHTS = 2.0     # Phase-1 association gate, in body heights


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def associate(dets_per_frame, frame_numbers):
    """Phase-1-style greedy association of boxes into tracks.

    Deliberately the SAME logic for B0 and B1 so the only difference between
    those arms is which detector produced the boxes.
    """
    tracks = {}
    next_id = 0
    for fn, dets in zip(frame_numbers, dets_per_frame):
        used = set()
        for d in dets:
            box = d.as_xyxy()
            cx, cy = d.center
            h = max(d.height, 1e-6)
            best, best_dist = None, None
            for tid, obs in tracks.items():
                if tid in used:
                    continue
                lf, lbox, _ = obs[-1]
                lcx, lcy = (lbox[0] + lbox[2]) / 2, (lbox[1] + lbox[3]) / 2
                lh = max(lbox[3] - lbox[1], 1e-6)
                gap = max(1, frame_numbers.index(fn) - frame_numbers.index(lf))
                gate = GATE_BODY_HEIGHTS * gap * max(h, lh)
                dist = math.hypot(cx - lcx, cy - lcy)
                if dist <= gate and (best_dist is None or dist < best_dist):
                    best, best_dist = tid, dist
            if best is None:
                tracks[next_id] = [(fn, box, d.conf)]
                used.add(next_id)
                next_id += 1
            else:
                tracks[best].append((fn, box, d.conf))
                used.add(best)
    return tracks


def gt_track(tracks, gt_frames):
    counts = {}
    for frec in gt_frames:
        box = frec["batsman_bbox"]
        if not box:
            continue
        best, best_iou = None, 0.0
        for tid, obs in tracks.items():
            for f, tbox, _ in obs:
                if f == frec["frame"]:
                    v = _iou(tbox, box)
                    if v > best_iou:
                        best, best_iou = tid, v
                    break
        if best is not None and best_iou >= IOU_MATCH:
            counts[best] = counts.get(best, 0) + 1
    return (max(counts, key=counts.get), max(counts.values())) if counts else (None, 0)


def _outcome(pick, gt_tid):
    """Classify one clip's selection.

    CRITICAL: when the detector never produced the batsman at all, there IS
    no ground-truth track — and committing to some other person is a
    WRONG-PERSON result, not an excluded clip. Bucketing those as "no ground
    truth" is exactly what would hide the Phase-1 failure this ablation
    exists to measure: an arm that never sees the batsman would score a
    perfect 0% wrong-person rate. Only a refusal is not a wrong answer.
    """
    if pick is None:
        return "refused"
    if gt_tid is not None and pick == gt_tid:
        return "correct"
    return "wrong"


def run(gt, video_dir, tracks_cache, model_full, model_geom, split):
    from phase2_detectors import build_detector
    from phase2_batsman_score import select_batsman, baseline_size_dominance

    clips = [c for c in gt["clips"] if c["split"] == split]
    arms = ["B0", "B1", "B2", "B3", "B4", "B5"]
    tally = {a: {"correct": 0, "wrong": 0, "refused": 0, "no_gt": 0} for a in arms}
    recall = {a: [0, 0] for a in arms}          # [hits, total gt frames]
    timing = {a: [0.0, 0] for a in arms}
    pose_stats = {"attempted": 0, "succeeded": 0}
    detail = []

    mp_det = build_detector("mediapipe")
    yolo_det = build_detector("yolo11m@640")

    for clip in clips:
        path = os.path.join(video_dir, clip["clip_id"])
        if not os.path.exists(path):
            continue
        gt_frames = [f for f in clip["frames"] if f["batsman_visible"] and f["batsman_bbox"]]
        if not gt_frames:
            continue
        fnums = [f["frame"] for f in clip["frames"]]

        cap = cv2.VideoCapture(path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        for i in fnums:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            frames.append(fr if ok else None)
        cap.release()

        ctx_sparse = {"frame_w": W, "frame_h": H, "frames_processed": len(fnums)}
        row = {"clip_id": clip["clip_id"]}

        # ---- B0 / B1: detector + Phase-1 association + size dominance ----
        for arm, det in (("B0", mp_det), ("B1", yolo_det)):
            t0 = time.perf_counter()
            dets = det.detect_batch(frames)
            trk = associate(dets, fnums)
            pick = baseline_size_dominance(trk, ctx_sparse)
            timing[arm][0] += time.perf_counter() - t0
            timing[arm][1] += len([f for f in frames if f is not None])

            g, _ = gt_track(trk, gt_frames)
            hits = sum(1 for frec in gt_frames
                       if any(_iou(d.as_xyxy(), frec["batsman_bbox"]) >= IOU_MATCH
                              for d in dets[fnums.index(frec["frame"])]))
            recall[arm][0] += hits
            recall[arm][1] += len(gt_frames)
            tally[arm][_outcome(pick, g)] += 1
            if g is None and pick is not None:
                tally[arm]["no_gt"] += 1        # diagnostic only, also counted wrong
            row[arm] = {"pick": pick, "gt": g}

        # ---- B2..B5: cached ByteTrack tracks ----
        c = tracks_cache["clips"].get(clip["clip_id"])
        if c is None:
            detail.append(row)
            continue
        ctx = {"frame_w": c["frame_w"], "frame_h": c["frame_h"],
               "frames_processed": c["frames_processed"]}
        trk = c["tracks"]
        g, _ = gt_track(trk, gt_frames)

        hits = 0
        for frec in gt_frames:
            best = 0.0
            for obs in trk.values():
                for f, tbox, _ in obs:
                    if f == frec["frame"]:
                        best = max(best, _iou(tbox, frec["batsman_bbox"]))
                        break
            if best >= IOU_MATCH:
                hits += 1

        picks = {
            "B2": baseline_size_dominance(trk, ctx),
            "B3": None, "B4": None,
        }
        sel3 = select_batsman(trk, ctx, model_geom)
        sel4 = select_batsman(trk, ctx, model_full)
        picks["B3"] = sel3["track_id"]
        picks["B4"] = sel4["track_id"]
        picks["B5"] = sel4["track_id"]          # same selection; B5 adds pose

        for arm in ("B2", "B3", "B4", "B5"):
            recall[arm][0] += hits
            recall[arm][1] += len(gt_frames)
            p = picks[arm]
            tally[arm][_outcome(p, g)] += 1
            if g is None and p is not None:
                tally[arm]["no_gt"] += 1        # diagnostic only, also counted wrong
            row[arm] = {"pick": p, "gt": g}
        row["B4_verdict"] = sel4["verdict"]
        row["B4_conf"] = sel4.get("confidence")

        # ---- B5: MediaPipe pose on the SELECTED crop (top-down) ----
        # §19: test YOLO -> crop -> existing MediaPipe pose BEFORE reaching
        # for RTMPose, so the pose model is only replaced on evidence.
        if picks["B5"] is not None:
            import zero_storage_pipeline as zsp
            import mediapipe as mp
            for frec in gt_frames:
                box = next((b for f, b, _ in trk[picks["B5"]] if f == frec["frame"]), None)
                fr = frames[fnums.index(frec["frame"])]
                if box is None or fr is None:
                    continue
                pad = 0.15
                bw, bh = box[2] - box[0], box[3] - box[1]
                x1 = max(0, int(box[0] - pad * bw)); y1 = max(0, int(box[1] - pad * bh))
                x2 = min(W, int(box[2] + pad * bw)); y2 = min(H, int(box[3] + pad * bh))
                crop = fr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                pose_stats["attempted"] += 1
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                with zsp._DETECTOR_LOCK:
                    cands = zsp._detect_candidates(zsp._get_shared_detector_locked(), [rgb])
                if cands[0]:
                    pose_stats["succeeded"] += 1

        detail.append(row)

    out = {"split": split, "clips": len(clips), "arms": {}, "pose_on_crop": pose_stats}
    for a in arms:
        t = tally[a]
        decided = t["correct"] + t["wrong"]
        out["arms"][a] = {
            **t,
            "batsman_candidate_recall": round(recall[a][0] / recall[a][1], 4) if recall[a][1] else None,
            "wrong_person_rate": round(t["wrong"] / decided, 4) if decided else None,
            "correct_rate_all_clips": round(t["correct"] / max(len(clips), 1), 4),
            "seconds_per_frame": (round(timing[a][0] / timing[a][1], 4)
                                  if timing[a][1] else None),
        }
    out["detail"] = detail
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--tracks", default=os.path.join(RESULTS, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--model", default=os.path.join(RESULTS, "phase2_batsman_model.json"))
    ap.add_argument("--geom-model", default=os.path.join(RESULTS, "phase2_batsman_model_geom.json"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "phase2_ablation.json"))
    args = ap.parse_args()

    from phase2_cache_tracks import load_tracks
    from phase2_detector_benchmark import load_gt

    gt = load_gt()
    tracks_cache = load_tracks(args.tracks)
    with open(args.model, encoding="utf-8") as f:
        model_full = json.load(f)["model"]
    with open(args.geom_model, encoding="utf-8") as f:
        model_geom = json.load(f)["model"]

    report = {}
    for split in ("dev", "eval"):
        r = run(gt, args.videos, tracks_cache, model_full, model_geom, split)
        report[split] = r
        print(f"\n=== {split.upper()} ({r['clips']} clips) ===")
        print(f"  {'arm':<4} {'recall':>7} {'correct':>8} {'wrong':>6} {'refused':>8} "
              f"{'wrongrate':>10} {'correct/all':>12} {'s/frame':>9}")
        for a, s in r["arms"].items():
            print(f"  {a:<4} {str(s['batsman_candidate_recall']):>7} {s['correct']:>8} "
                  f"{s['wrong']:>6} {s['refused']:>8} {str(s['wrong_person_rate']):>10} "
                  f"{s['correct_rate_all_clips']:>12} {str(s['seconds_per_frame']):>9}")
        p = r["pose_on_crop"]
        if p["attempted"]:
            print(f"  B5 pose-on-crop success: {p['succeeded']}/{p['attempted']} "
                  f"= {p['succeeded']/p['attempted']:.3f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
