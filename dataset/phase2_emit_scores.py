"""
phase2_emit_scores.py — write one JSONL record per (clip, track) with the
batsman score, its components, the clip-level verdict, and whether that track
is the annotated batsman.

This is the audit trail for §10-11 of the Phase-2 spec: every selection can be
re-examined after the fact without re-running detection or tracking, and the
separate confidence concepts stay separate in the record rather than being
collapsed into one number.
"""

import argparse
import json
import os

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_MODULE_DIR, "phase2_results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=os.path.join(RESULTS, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--model", default=os.path.join(RESULTS, "phase2_batsman_model_geom.json"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "phase2_batsman_scores.jsonl"))
    args = ap.parse_args()

    from phase2_cache_tracks import load_tracks
    from phase2_detector_benchmark import load_gt
    from phase2_batsman_score import select_batsman
    from phase2_fit_batsman_model import gt_track_for_clip

    data = load_tracks(args.tracks)
    gt = {c["clip_id"]: c for c in load_gt()["clips"]}
    with open(args.model, encoding="utf-8") as f:
        payload = json.load(f)
    model = payload["model"]

    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for cid, c in data["clips"].items():
            clip_gt = gt.get(cid)
            if not clip_gt:
                continue
            gt_frames = [f for f in clip_gt["frames"]
                         if f["batsman_visible"] and f["batsman_bbox"]]
            gt_tid, n_match = gt_track_for_clip(c["tracks"], gt_frames)

            ctx = {"frame_w": c["frame_w"], "frame_h": c["frame_h"],
                   "frames_processed": c["frames_processed"]}
            sel = select_batsman(c["tracks"], ctx, model)

            for s in sel["scores"]:
                out.write(json.dumps({
                    "clip_id": cid,
                    "split": c["split"],
                    "track_id": s["track_id"],
                    "batsman_score": s["score"],
                    "batsman_confidence": s["confidence"],
                    "components": s["components"],
                    "n_observations": len(c["tracks"][s["track_id"]]),
                    "mean_detection_confidence": s["components"].get("mean_conf"),
                    "clip_verdict": sel["verdict"],
                    "clip_selected_track": sel.get("track_id"),
                    "clip_margin": sel.get("margin"),
                    "is_annotated_batsman": (gt_tid is not None and s["track_id"] == gt_tid),
                    "gt_track_matched_frames": n_match if s["track_id"] == gt_tid else 0,
                }) + "\n")
                n += 1

    print(f"wrote {args.out} ({n} track records, model={payload.get('feature_set')})")


if __name__ == "__main__":
    main()
