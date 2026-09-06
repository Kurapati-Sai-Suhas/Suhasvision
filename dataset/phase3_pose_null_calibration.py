"""
phase3_pose_null_calibration.py — what does RTMPose output when there is no
person there?

WHY THIS EXISTS
The first run of the pose benchmark reported pose_availability = 1.0000 for
every RTMPose variant on every clip, including the seven clips that currently
fail for pose reasons. Taken at face value that says RTMPose eliminates 100%
of pose failures. It does not. RTMPose is a TOP-DOWN regressor: handed a box,
it always emits 26 keypoints. It has no way to answer "nobody is here".

A direct check confirmed it: on a flat grey image with no content whatsoever,
RTMPose still returns a complete skeleton (mean joint score 0.141), while
MediaPipe correctly returns nothing. So "availability" is not a property the
two models share, and comparing them on it is meaningless -- one model is
being credited for being unable to decline.

WHAT THIS SCRIPT DOES
Builds the NULL distribution of RTMPose's confidence on boxes known to
contain no person, then sets the gate at the null's upper tail. A frame
counts as "pose available" only if its mean joint score clears that gate.
This gives RTMPose an operating point comparable to MediaPipe's implicit one.

The gate is calibrated ONLY against no-person patches. It never sees the
benchmark's labels or outcomes, so it does not tune on the data it is later
scored on.
"""

import argparse
import json
import os
import random

import cv2
import numpy as np

import phase3_pose_models as PM

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

NULL_PERCENTILE = 95     # gate at the 95th percentile of the no-person null


def person_free_boxes(frame_shape, person_boxes, n=4, rng=None):
    """Random boxes of person-like shape that overlap no person.

    `person_boxes` must come from a LOW-THRESHOLD detector sweep, not from the
    tracks. A first version of this script excluded only tracked persons on
    frame 0 and produced a null whose 95th percentile (0.738) sat above
    RTMPose's mean confidence on real batsmen (0.617) -- an impossible result
    that gave the contamination away. Cricket frames are full of fielders,
    umpires and crowd that are never tracked, so "not a track" is nowhere near
    "not a person".
    """
    rng = rng or random
    h, w = frame_shape[:2]
    out, tries = [], 0
    while len(out) < n and tries < 60:
        tries += 1
        bh = rng.randint(int(0.25 * h), int(0.8 * h))
        bw = max(20, int(bh * rng.uniform(0.3, 0.6)))
        if bw >= w or bh >= h:
            continue
        x1 = rng.randint(0, w - bw); y1 = rng.randint(0, h - bh)
        box = (x1, y1, x1 + bw, y1 + bh)
        # Overlap is measured against the NULL box's own area, not IoU: a
        # small box sitting entirely inside a large person has a low IoU but
        # is completely full of person.
        if all(_contained(box, p) < 0.02 for p in person_boxes):
            out.append(box)
    return out


def _contained(a, b):
    """Fraction of box `a` covered by box `b`."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return inter / area if area > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--tracks", default=os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--variants", default="rtmpose-s@192x256,rtmpose-m@192x256,rtmpose-x@288x384")
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--frames-per-clip", type=int, default=1)
    ap.add_argument("--boxes-per-frame", type=int, default=4)
    args = ap.parse_args()

    from phase2_cache_tracks import load_tracks
    from ultralytics import YOLO
    rng = random.Random(0)

    # conf=0.05 deliberately: for building a person-FREE null, a false person
    # only costs us a candidate box, while a missed person poisons the null.
    det = YOLO("yolo11x.pt")

    tracks = load_tracks(args.tracks)
    models = [PM.RTMPosePose(variant=v.strip()) for v in args.variants.split(",")]
    mp = PM.MediaPipePose()

    null = {m.name: [] for m in models}
    null_mp_ok = 0
    null_n = 0

    cids = sorted(tracks["clips"])[:args.clips]
    for n, cid in enumerate(cids, 1):
        path = os.path.join(args.videos, cid)
        if not os.path.exists(path):
            continue
        c = tracks["clips"][cid]
        nf = int(c.get("frames_processed") or 0)
        want = sorted({int(i * max(1, nf - 1) / max(1, args.frames_per_clip))
                       for i in range(args.frames_per_clip)}) or [0]
        cap = cv2.VideoCapture(path)
        i, grabbed = -1, {}
        while i < max(want):
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            if i in want:
                grabbed[i] = fr
        cap.release()

        for fidx, fr in grabbed.items():
            # Every person the detector can find at a permissive threshold,
            # plus every tracked box on this frame, is off limits.
            r0 = det.predict(fr, classes=[0], conf=0.05, verbose=False)[0]
            pboxes = [tuple(float(v) for v in b) for b in r0.boxes.xyxy.cpu().numpy()]
            pboxes += [b for obs in c["tracks"].values() for f, b, _ in obs if f == fidx]
            for box in person_free_boxes(fr.shape, pboxes, n=args.boxes_per_frame, rng=rng):
                null_n += 1
                if mp.infer(fr, box).ok:
                    null_mp_ok += 1
                for m in models:
                    r = m.infer(fr, box)
                    if r.ok and r.scores:
                        null[m.name].append(float(np.mean(list(r.scores.values()))))
        print(f"[{n}/{len(cids)}] {cid}: null boxes so far {null_n}")

    out = {"n_null_boxes": null_n,
           "null_percentile_used": NULL_PERCENTILE,
           "mediapipe_false_pose_rate_on_null": (round(null_mp_ok / null_n, 4)
                                                 if null_n else None),
           "note": ("MediaPipe's rate above is its own false-positive rate on "
                    "person-free boxes and is the operating point RTMPose's "
                    "gate is being made comparable to."),
           "models": {}}
    for m in models:
        v = null[m.name]
        if not v:
            continue
        out["models"][m.name] = {
            "n": len(v),
            "null_mean_score_mean": round(float(np.mean(v)), 4),
            "null_mean_score_p50": round(float(np.percentile(v, 50)), 4),
            "null_mean_score_p95": round(float(np.percentile(v, 95)), 4),
            "null_mean_score_max": round(float(np.max(v)), 4),
            "gate": round(float(np.percentile(v, NULL_PERCENTILE)), 4),
            "raw_null_scores": [round(x, 4) for x in v],
        }

    with open(os.path.join(P3, "phase3_pose_null_calibration.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(f"\nnull boxes: {null_n}   MediaPipe false-pose rate on them: "
          f"{out['mediapipe_false_pose_rate_on_null']}")
    for k, v in out["models"].items():
        print(f"{k:<20} null mean={v['null_mean_score_mean']:.3f}  "
              f"p95={v['null_mean_score_p95']:.3f}  max={v['null_mean_score_max']:.3f}"
              f"   -> gate {v['gate']:.3f}")
    print(f"\nwrote {os.path.join(P3, 'phase3_pose_null_calibration.json')}")


if __name__ == "__main__":
    main()
