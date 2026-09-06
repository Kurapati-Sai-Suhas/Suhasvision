"""
phase2_cache_tracks.py — run detector+tracker once per clip and persist the
tracks, so batsman-model fitting and the B0-B5 ablation reuse identical
tracking rather than re-running it (and rather than drifting apart).

Caching is a correctness measure as much as a speed one: fitting a selection
model on one tracking run and evaluating it on another would silently compare
different inputs.
"""

import argparse
import json
import os
import time

import cv2

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_MODULE_DIR, "phase2_results")


def cache(video_dir, weights="yolo11m.pt", imgsz=640, conf=0.10,
          tracker="bytetrack.yaml", out=None):
    import phase2_tracking as T
    from phase2_detector_benchmark import load_gt

    gt = load_gt()
    model = T.build_tracking_model(weights)
    out = out or os.path.join(
        RESULTS, f"tracks_{weights.replace('.pt','')}@{imgsz}_{tracker.replace('.yaml','')}.json")

    clips = {}
    t0 = time.perf_counter()
    for i, clip in enumerate(gt["clips"], 1):
        path = os.path.join(video_dir, clip["clip_id"])
        if not os.path.exists(path):
            continue
        must = [f["frame"] for f in clip["frames"]]
        tracks, meta = T.track_clip(path, model, tracker, must, imgsz=imgsz, conf=conf)

        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        clips[clip["clip_id"]] = {
            "split": clip["split"],
            "frame_w": w, "frame_h": h,
            "frames_processed": meta.get("frames_processed", 0),
            "stride": meta.get("stride", 1),
            "subsampled": meta.get("subsampled", False),
            "n_untracked_detections": meta.get("n_untracked_detections", 0),
            # track id -> [[frame, x1, y1, x2, y2, conf], ...]
            "tracks": {str(tid): [[f, *[round(v, 2) for v in box], round(c, 4)]
                                  for f, box, c in obs]
                       for tid, obs in tracks.items()},
        }
        if i % 10 == 0:
            print(f"  {i}/{len(gt['clips'])} clips, {time.perf_counter()-t0:.0f}s")

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"detector": f"{weights.replace('.pt','')}@{imgsz}",
                   "tracker": tracker.replace(".yaml", ""),
                   "conf": conf, "clips": clips}, f)
    print(f"wrote {out}  ({len(clips)} clips, {time.perf_counter()-t0:.0f}s)")
    return out


def load_tracks(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for c in data["clips"].values():
        c["tracks"] = {int(t): [(int(o[0]), (o[1], o[2], o[3], o[4]), o[5]) for o in obs]
                       for t, obs in c["tracks"].items()}
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.10)
    args = ap.parse_args()
    cache(args.videos, args.weights, args.imgsz, args.conf, args.tracker)


if __name__ == "__main__":
    main()
