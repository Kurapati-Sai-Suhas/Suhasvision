"""
phase2_detector_benchmark.py — measure whether each detector actually puts
the batsman in the candidate set.

THE PRIMARY METRIC
    Batsman Candidate Recall = (GT batsman frames where some detection has
                                IoU >= 0.5 with the GT batsman box)
                             / (GT batsman frames)

This is the metric that matters, because Phase 1 established that selection
cannot pick a batsman the detector never produced. Everything else here is
secondary.

FRAMES WHERE THE PROPOSAL DETECTOR ITSELF MISSED
Those frames have batsman_visible=true and batsman_bbox=null. They stay in
the DENOMINATOR and no detector can score them -- including the proposal
detector. Without this, ground truth anchored to a detector would report that
detector as perfect by construction.

THE CIRCULARITY THAT REMAINS
The GT boxes were chosen from yolo11x@1280's proposals, so that exact
configuration is NOT a fair contender: its recall is ~1.0 by construction and
is reported only as the annotation ceiling. Every other configuration --
including other YOLO sizes and resolutions -- is scored against boxes a
DIFFERENT model produced, which is a real measurement. MediaPipe is
architecturally independent, so the MediaPipe-vs-YOLO comparison, which is
the one this phase turns on, is unaffected.

FALSE POSITIVES
A true FP rate needs every person in every frame annotated, which this
benchmark does not have. Reporting one would be inventing it. Instead two
measurable proxies are reported:
    duplicate_rate   detections overlapping another detection of the SAME
                     frame at IoU > 0.7 -- redundant boxes on one person,
                     unambiguously spurious
    tiny_rate        detections under 24px tall, below the size at which a
                     usable pose can be extracted from this footage
"""

import argparse
import json
import os
import time

import cv2

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_MODULE_DIR, "phase2_results")

IOU_MATCH = 0.5          # standard detection-match threshold
DUP_IOU = 0.7            # above this, two boxes in one frame are redundant
TINY_PX = 24             # below this height, no usable pose from this footage

DEFAULT_SPECS = [
    "mediapipe",
    "yolo11n@640", "yolo11n@960",
    "yolo11m@640", "yolo11m@960", "yolo11m@1280",
    "yolo11x@960", "yolo11x@1280",
]


def load_gt(path=None):
    with open(path or os.path.join(RESULTS, "phase2_groundtruth.json"), encoding="utf-8") as f:
        return json.load(f)


def _read_frames(video_path, indices):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [None] * len(indices)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        frames.append(f if ok else None)
    cap.release()
    return frames


def evaluate(spec, gt, video_dir, split=None):
    from phase2_detectors import build_detector, iou

    det = build_detector(spec)
    clips = [c for c in gt["clips"] if split is None or c["split"] == split]

    n_gt = n_hit = n_miss_proposal = 0
    n_det = n_frames = n_dup = n_tiny = 0
    per_clip = []
    gt_heights, hit_heights, missed = [], [], []
    t_total = 0.0

    for clip in clips:
        video_path = os.path.join(video_dir, clip["clip_id"])
        if not os.path.exists(video_path):
            continue
        indices = [f["frame"] for f in clip["frames"]]
        frames = _read_frames(video_path, indices)

        t0 = time.perf_counter()
        dets_per_frame = det.detect_batch(frames)
        t_total += time.perf_counter() - t0

        clip_gt = clip_hit = 0
        for frec, dets in zip(clip["frames"], dets_per_frame):
            if frames[indices.index(frec["frame"])] is None:
                continue
            n_frames += 1
            n_det += len(dets)
            for i, d in enumerate(dets):
                if d.height < TINY_PX:
                    n_tiny += 1
                if any(iou(d, o) > DUP_IOU for j, o in enumerate(dets) if j > i):
                    n_dup += 1

            if not frec["batsman_visible"]:
                continue
            n_gt += 1
            clip_gt += 1
            box = frec["batsman_bbox"]
            if box is None:                      # proposal miss: nobody can score it
                n_miss_proposal += 1
                missed.append((clip["clip_id"], frec["frame"], None))
                continue
            gt_h = box[3] - box[1]
            gt_heights.append(gt_h)
            best = max((iou(d, box) for d in dets), default=0.0)
            if best >= IOU_MATCH:
                n_hit += 1
                clip_hit += 1
                hit_heights.append(gt_h)
            else:
                missed.append((clip["clip_id"], frec["frame"], round(gt_h, 1)))

        per_clip.append({"clip_id": clip["clip_id"], "split": clip["split"],
                         "gt": clip_gt, "hit": clip_hit})

    det.close()

    def pct(a, b):
        return round(a / b, 4) if b else None

    # Recall on small subjects specifically -- the Phase-1 failure mode.
    small_gt = [h for h in gt_heights if h < 160]
    small_hit = [h for h in hit_heights if h < 160]

    return {
        "detector": spec,
        "split": split or "all",
        "gt_batsman_frames": n_gt,
        "batsman_candidate_recall": pct(n_hit, n_gt),
        "recall_small_subjects": pct(len(small_hit), len(small_gt)),
        "n_small_gt": len(small_gt),
        "unscorable_proposal_misses": n_miss_proposal,
        "frames": n_frames,
        "mean_people_per_frame": round(n_det / n_frames, 3) if n_frames else None,
        "duplicate_rate": pct(n_dup, n_det),
        "tiny_detection_rate": pct(n_tiny, n_det),
        "seconds_per_frame": round(t_total / n_frames, 4) if n_frames else None,
        "misses": missed[:25],
        "per_clip": per_clip,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--specs", nargs="+", default=DEFAULT_SPECS)
    ap.add_argument("--split", default=None, choices=["dev", "eval"])
    ap.add_argument("--out", default=os.path.join(RESULTS, "phase2_detector_benchmark.json"))
    args = ap.parse_args()

    gt = load_gt()
    report = {"iou_match": IOU_MATCH, "proposal_detector": gt["proposal_detector"],
              "results": {}}

    print(f"{'detector':<16} {'split':<5} {'recall':>7} {'small':>7} {'ppl/f':>6} "
          f"{'dup':>6} {'tiny':>6} {'s/frame':>8}")
    for spec in args.specs:
        for split in (["dev", "eval"] if args.split is None else [args.split]):
            try:
                r = evaluate(spec, gt, args.videos, split)
            except Exception as exc:  # noqa: BLE001 - one detector failing must not end the run
                print(f"{spec:<16} {split:<5} FAILED {type(exc).__name__}: {exc}")
                report["results"][f"{spec}|{split}"] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            report["results"][f"{spec}|{split}"] = r
            print(f"{spec:<16} {split:<5} {r['batsman_candidate_recall']:>7.3f} "
                  f"{(r['recall_small_subjects'] if r['recall_small_subjects'] is not None else -1):>7.3f} "
                  f"{r['mean_people_per_frame']:>6.2f} {r['duplicate_rate']:>6.3f} "
                  f"{r['tiny_detection_rate']:>6.3f} {r['seconds_per_frame']:>8.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
