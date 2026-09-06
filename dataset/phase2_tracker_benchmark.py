"""
phase2_tracker_benchmark.py — compare ByteTrack and BoT-SORT on the ability
to hold ONE consistent track on the batsman.

METRICS, AND WHAT THEY HONESTLY ARE
The ground truth is SPARSE: 6 annotated frames per clip. That supports some
metrics exactly and forbids others, and the two are kept apart.

Computable exactly at annotated frames:
  batsman_track_recall     fraction of annotated batsman frames covered by
                           SOME track (IoU >= 0.5)
  batsman_track_coverage   fraction covered by the clip's SINGLE BEST track
                           -- the track a downstream selector would pick
  observed_id_switches     times the best-matching track id CHANGES between
                           consecutive annotated frames. A lower bound: a
                           switch that occurs and reverts between two
                           annotated frames is invisible here.

NOT computed: MOTA, HOTA, IDF1. Those need every person identity-labelled in
every frame. Reporting them from sparse single-person annotation would be
fabrication, so they are absent rather than approximated.
"""

import argparse
import json
import os
import time

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_MODULE_DIR, "phase2_results")

IOU_MATCH = 0.5


def evaluate_tracker(tracker_cfg, gt, video_dir, weights="yolo11m.pt",
                     imgsz=640, conf=0.10, split=None):
    from phase2_detectors import Detection, iou
    import phase2_tracking as T

    model = T.build_tracking_model(weights)
    clips = [c for c in gt["clips"] if split is None or c["split"] == split]

    n_gt = n_any = n_best = 0
    n_switches = 0
    n_switch_opportunities = 0
    per_clip, struct = [], []
    t_total, frames_done = 0.0, 0

    for clip in clips:
        path = os.path.join(video_dir, clip["clip_id"])
        if not os.path.exists(path):
            continue
        gt_frames = [f for f in clip["frames"] if f["batsman_visible"] and f["batsman_bbox"]]
        must = [f["frame"] for f in clip["frames"]]

        t0 = time.perf_counter()
        tracks, meta = T.track_clip(path, model, tracker_cfg, must, imgsz=imgsz, conf=conf)
        t_total += time.perf_counter() - t0
        frames_done += meta.get("frames_processed", 0)

        # Which track best matches the batsman at each annotated frame?
        matched = []          # (frame_no, track_id or None)
        for frec in gt_frames:
            box = frec["batsman_bbox"]
            best_tid, best_iou = None, 0.0
            for tid, tbox, _ in T.tracks_at_frame(tracks, frec["frame"]):
                d = Detection(*tbox, conf=1.0)
                v = iou(d, box)
                if v > best_iou:
                    best_tid, best_iou = tid, v
            matched.append((frec["frame"], best_tid if best_iou >= IOU_MATCH else None))

        n_gt += len(gt_frames)
        n_any += sum(1 for _, t in matched if t is not None)

        # The single best track = the one covering the most annotated frames.
        counts = {}
        for _, tid in matched:
            if tid is not None:
                counts[tid] = counts.get(tid, 0) + 1
        best_track_cover = max(counts.values()) if counts else 0
        n_best += best_track_cover

        # Identity switches between CONSECUTIVE annotated frames where both
        # were matched. Frames with no match are skipped rather than counted
        # as switches -- a miss is a detection failure, not an id change.
        seq = [t for _, t in matched if t is not None]
        n_switches += sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        n_switch_opportunities += max(0, len(seq) - 1)

        s = T.track_stats(tracks, meta.get("frames_processed", 1))
        struct.append(s)
        per_clip.append({
            "clip_id": clip["clip_id"], "split": clip["split"],
            "gt_frames": len(gt_frames), "any_track": sum(1 for _, t in matched if t),
            "best_track": best_track_cover,
            "n_tracks": s["n_tracks"],
            "switches": sum(1 for a, b in zip(seq, seq[1:]) if a != b),
            "subsampled": meta.get("subsampled"),
        })

    def pct(a, b):
        return round(a / b, 4) if b else None

    n = max(len(struct), 1)
    return {
        "tracker": tracker_cfg.replace(".yaml", ""),
        "detector": f"{weights.replace('.pt', '')}@{imgsz}",
        "split": split or "all",
        "gt_batsman_frames": n_gt,
        "batsman_track_recall": pct(n_any, n_gt),
        "batsman_track_coverage": pct(n_best, n_gt),
        "observed_id_switches": n_switches,
        "id_switch_rate": pct(n_switches, n_switch_opportunities),
        "switch_opportunities": n_switch_opportunities,
        "mean_tracks_per_clip": round(sum(s["n_tracks"] for s in struct) / n, 2),
        "mean_track_fragmentation": round(sum(s["mean_fragmentation"] or 0 for s in struct) / n, 3),
        "mean_best_track_coverage_fraction": round(
            sum(s.get("max_coverage_fraction") or 0 for s in struct) / n, 3),
        "seconds_per_frame": round(t_total / frames_done, 4) if frames_done else None,
        "note": "MOTA/HOTA/IDF1 not computed: sparse single-subject GT cannot support them.",
        "per_clip": per_clip,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--trackers", nargs="+", default=["bytetrack.yaml", "botsort.yaml"])
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=os.path.join(RESULTS, "phase2_tracker_results.json"))
    args = ap.parse_args()

    from phase2_detector_benchmark import load_gt
    gt = load_gt()

    report = {"iou_match": IOU_MATCH, "results": {}}
    print(f"{'tracker':<12} {'split':<5} {'recall':>7} {'cover':>7} {'switch':>7} "
          f"{'trk/clip':>9} {'frag':>7} {'s/frame':>8}")
    for tcfg in args.trackers:
        for split in ("dev", "eval"):
            try:
                r = evaluate_tracker(tcfg, gt, args.videos, args.weights, args.imgsz, split=split)
            except Exception as exc:  # noqa: BLE001
                print(f"{tcfg:<12} {split:<5} FAILED {type(exc).__name__}: {exc}")
                report["results"][f"{tcfg}|{split}"] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            report["results"][f"{tcfg}|{split}"] = r
            print(f"{r['tracker']:<12} {split:<5} {r['batsman_track_recall']:>7.3f} "
                  f"{r['batsman_track_coverage']:>7.3f} {r['observed_id_switches']:>7d} "
                  f"{r['mean_tracks_per_clip']:>9.2f} {r['mean_track_fragmentation']:>7.2f} "
                  f"{r['seconds_per_frame']:>8.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
