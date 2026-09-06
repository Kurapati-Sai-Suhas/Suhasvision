"""
phase3_pose_diagnosis.py — is the "pose failure" diagnosis actually a pose
failure?

THE PROBLEM THIS FOUND
The seven clips that motivated the RTMPose evaluation were labelled
"pose failure on the batsman crop (small/occluded subject)" by
phase3_batsman_analysis.categorise_failure(), on the test

    gt["k_pose_rate"] < 0.4

k_pose_rate comes from phase3_semantic_features.extract(), which samples 20
frames uniformly ACROSS THE WHOLE CLIP and then, for the batsman track, does:

    box = by_frame.get(f)
    if box is None or fr is None:
        seq.append(None)          # <-- counted exactly like a pose failure

So a sampled frame where the batsman TRACK SIMPLY DOES NOT EXIST is recorded
identically to a frame where the pose model was called and failed. The pose
model is never invoked on those frames at all.

    k_pose_rate = (frames with a pose) / 20
                = coverage_on_sampled_frames x P(pose | box exists)

Only the second factor is a pose problem. The first is a tracking/coverage
problem, and no pose model can fix it.

A SECOND, PURELY LOGICAL CONSEQUENCE
categorise_failure() tests, in this order:

    if k_pose_rate < 0.4:  return "pose failure on the batsman crop"
    if coverage   < 0.4:   return "tracking failure - batsman track too fragmented"

Because a pose requires a box, k_pose_rate <= coverage always. So whenever
coverage < 0.4, k_pose_rate < 0.4 is also true and the POSE branch returns
first. The tracking-failure branch is unreachable for exactly the clips it
was written to catch: every fragmented-track failure is reported as a pose
failure.

This script separates the two factors, using the pipeline's own uniform
20-frame sampling so the numbers are directly comparable to k_pose_rate.
"""

import argparse
import json
import os

import cv2

import phase3_pose_models as PM

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

N_SAMPLE = 20
THRESH = 0.40


def sample_frames(n_frames, k=N_SAMPLE):
    """Identical to phase3_semantic_features.sample_frames."""
    if n_frames <= 1:
        return [0]
    return sorted({int(round(i * (n_frames - 1) / (k - 1))) for i in range(k)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--tracks", default=os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--variant", default="rtmpose-m@192x256")
    args = ap.parse_args()

    from phase2_cache_tracks import load_tracks
    from phase2_detector_benchmark import load_gt
    from phase3_pose_benchmark import gt_track, POSE_FAILURE_CLIPS

    tracks = load_tracks(args.tracks)
    gt = {c["clip_id"]: c for c in load_gt()["clips"]}
    mp = PM.MediaPipePose()
    rtm = PM.RTMPosePose(variant=args.variant)

    gate = None
    sp = os.path.join(P3, "phase3_pose_separability.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            s = json.load(f)
        m = s.get("models", {}).get(rtm.name, {}).get("matched_to_mediapipe_fpr")
        gate = m["gate"] if m else None

    rows = []
    clips = sorted(set(tracks["clips"]) & set(gt))
    for n, cid in enumerate(clips, 1):
        c = tracks["clips"][cid]
        gframes = [f for f in gt[cid]["frames"]
                   if f.get("batsman_visible") and f.get("batsman_bbox")]
        if not gframes:
            continue
        tid = gt_track(c["tracks"], gframes)
        if tid is None or tid not in c["tracks"]:
            continue
        nf = int(c.get("frames_processed") or 0)
        idxs = sample_frames(nf)
        by_frame = {f: b for f, b, _ in c["tracks"][tid]}

        path = os.path.join(args.videos, cid)
        if not os.path.exists(path):
            continue
        want = set(idxs)
        cap = cv2.VideoCapture(path)
        grabbed, i = {}, -1
        while i < max(want):
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            if i in want:
                grabbed[i] = fr
        cap.release()

        n_box = n_mp = n_rtm = n_rtm_gated = 0
        for f in idxs:
            box, fr = by_frame.get(f), grabbed.get(f)
            if box is None or fr is None:
                continue
            n_box += 1
            if mp.infer(fr, box).ok:
                n_mp += 1
            r = rtm.infer(fr, box)
            if r.ok:
                n_rtm += 1
                if gate is not None and r.scores:
                    mc = sum(r.scores.values()) / len(r.scores)
                    if mc >= gate:
                        n_rtm_gated += 1
        k = len(idxs)
        row = {
            "clip_id": cid,
            "is_labelled_pose_failure": cid in POSE_FAILURE_CLIPS,
            "n_sampled": k,
            "n_frames_track_present": n_box,
            "coverage_on_sampled": round(n_box / k, 4) if k else 0.0,
            "k_pose_rate_mediapipe": round(n_mp / k, 4) if k else 0.0,
            "pose_given_box_mediapipe": round(n_mp / n_box, 4) if n_box else None,
            "k_pose_rate_rtmpose_ungated": round(n_rtm / k, 4) if k else 0.0,
            "k_pose_rate_rtmpose_gated": round(n_rtm_gated / k, 4) if k else 0.0,
            "pose_given_box_rtmpose_gated": (round(n_rtm_gated / n_box, 4)
                                             if n_box else None),
        }
        rows.append(row)
        if row["is_labelled_pose_failure"]:
            print(f"[{n}/{len(clips)}] {cid:<28} coverage={row['coverage_on_sampled']:.2f} "
                  f"k_pose(mp)={row['k_pose_rate_mediapipe']:.2f} "
                  f"pose|box(mp)={row['pose_given_box_mediapipe']}")

    fails = [r for r in rows if r["is_labelled_pose_failure"]]
    ceiling = [r for r in fails
               if r["coverage_on_sampled"] < THRESH]
    genuine = [r for r in fails
               if r["pose_given_box_mediapipe"] is not None
               and r["pose_given_box_mediapipe"] < THRESH]

    out = {
        "threshold": THRESH,
        "n_clips": len(rows),
        "n_labelled_pose_failures": len(fails),
        "decomposition": ("k_pose_rate = coverage_on_sampled x "
                          "P(pose | box exists); only the second factor is a "
                          "pose-model problem"),
        "labelled_failures_whose_coverage_alone_is_below_threshold": {
            "n": len(ceiling),
            "clips": [r["clip_id"] for r in ceiling],
            "meaning": ("For these clips the batsman track is absent from more "
                        "than 60% of the sampled frames, so k_pose_rate cannot "
                        "reach 0.4 no matter which pose model is used. These "
                        "are tracking failures reported as pose failures."),
        },
        "labelled_failures_that_are_genuinely_pose_limited": {
            "n": len(genuine),
            "clips": [r["clip_id"] for r in genuine],
            "meaning": ("MediaPipe fails on more than 60% of frames where the "
                        "batsman box DOES exist. Only these can be fixed by a "
                        "better pose model."),
        },
        "unreachable_branch_bug": (
            "phase3_batsman_analysis.categorise_failure() tests k_pose_rate<0.4 "
            "before coverage<0.4. Since a pose requires a box, "
            "k_pose_rate <= coverage, so coverage<0.4 always implies "
            "k_pose_rate<0.4 and the pose branch returns first. The "
            "'tracking failure - batsman track too fragmented' branch is "
            "unreachable."),
        "clips": rows,
    }
    with open(os.path.join(P3, "phase3_pose_diagnosis.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(f"\n{'clip':<30}{'cover':>8}{'k_pose':>8}{'pose|box':>10}"
          f"{'rtm gated':>11}")
    for r in fails:
        pb = r["pose_given_box_mediapipe"]
        print(f"{r['clip_id']:<30}{r['coverage_on_sampled']:>8.2f}"
              f"{r['k_pose_rate_mediapipe']:>8.2f}"
              f"{(f'{pb:.2f}' if pb is not None else '-'):>10}"
              f"{r['k_pose_rate_rtmpose_gated']:>11.2f}")
    print(f"\ncoverage alone below {THRESH} (no pose model can fix): "
          f"{len(ceiling)} -> {[r['clip_id'] for r in ceiling]}")
    print(f"genuinely pose-limited (pose|box < {THRESH}): "
          f"{len(genuine)} -> {[r['clip_id'] for r in genuine]}")
    print(f"\nwrote {os.path.join(P3, 'phase3_pose_diagnosis.json')}")


if __name__ == "__main__":
    main()
