"""
phase3_pose_benchmark.py — is pose the binding bottleneck, and does RTMPose fix it?

EXPERIMENTAL DISCIPLINE
Everything upstream of the pose model is held identical:

  * the same cached ByteTrack output (tracks_yolo11m@640_bytetrack.json),
  * the same clip list,
  * the same sampled frame indices,
  * the same bounding box per frame.

Both models are asked to do the same job on the same pixels for the same
person, so a difference in the numbers is attributable to the pose model and
not to detection, tracking, sampling, or identity.

WHICH PERSON: the ANNOTATED batsman track, not the model-selected one. Phase 3
already showed that anchoring on the predicted batsman silently evaluates the
wrong person on the feeder clips. Using the annotation removes identity from
this experiment entirely, which is the point -- we are asking whether pose is
the bottleneck ONCE identity is correct.

Frames are decoded sequentially. Random cap.set() seeking forces H.264
keyframe re-decode and made an earlier tool ~100x slower than it needed to be.
"""

import argparse
import json
import os
import statistics
import time

import cv2

import phase3_pose_models as PM

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

N_SAMPLE = 20
VIS_THRESH = 0.30

# The clips whose extraction currently fails for POSE reasons, taken from the
# repo's own analysis rather than from memory. These are the cases that decide
# the question: a model that helps everywhere except here has not helped.
POSE_FAILURE_CLIPS = (
    "kohli_side_12.mp4", "kohli_side_16.mp4", "pro_player_front_06.mp4",
    "pro_player_front_10.mp4", "pro_player_front_18.mp4", "rishi_front_2.mp4",
    "sanjay_front _1.mp4",
)


def gt_track(clip_tracks, gt_frames):
    """The track that matches the annotated batsman box most often."""
    counts = {}
    for frec in gt_frames:
        box = frec.get("batsman_bbox")
        if not box:
            continue
        best, best_iou = None, 0.0
        for tid, obs in clip_tracks.items():
            for f, tbox, _ in obs:
                if f == frec["frame"]:
                    v = PM_iou(tbox, box)
                    if v > best_iou:
                        best, best_iou = tid, v
        if best is not None and best_iou >= 0.3:
            counts[best] = counts.get(best, 0) + 1
    return max(counts, key=counts.get) if counts else None


def PM_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def clip_plan(clip_tracks, tid, n_sample=N_SAMPLE):
    """Sampled (frame_index, box) pairs on the batsman track."""
    obs = sorted(clip_tracks[tid], key=lambda o: o[0])
    if not obs:
        return []
    if len(obs) <= n_sample:
        return [(f, b) for f, b, _ in obs]
    step = (len(obs) - 1) / (n_sample - 1)
    picks = sorted({int(round(i * step)) for i in range(n_sample)})
    return [(obs[i][0], obs[i][1]) for i in picks]


def run_clip(video_path, plan, models):
    """One sequential decode; every model sees the identical frame and box."""
    want = {f for f, _ in plan}
    if not want:
        return {m.name: [] for m in models}
    boxes = dict(plan)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    out = {m.name: [] for m in models}
    timings = {m.name: [] for m in models}
    i, last = -1, max(want)
    while i < last:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i not in want:
            continue
        for m in models:
            t0 = time.perf_counter()
            r = m.infer(fr, boxes[i])
            timings[m.name].append(time.perf_counter() - t0)
            out[m.name].append(r)
    cap.release()
    return out, timings


def summarise_sequence(seq):
    """Per-clip quality for one model.

    A CAVEAT THAT CHANGES HOW THESE NUMBERS SHOULD BE READ
    `mean_confidence` and `mean_visible_fraction` are NOT comparable across
    the two models. MediaPipe's "visibility" and RTMPose's SimCC peak score
    are different quantities on different scales; neither is calibrated, and
    a shared threshold does not mean the same thing to both. They are
    reported per model for within-model use (a fallback trigger needs one)
    and must not be used to declare a winner.

    The comparable metrics are the threshold-free ones -- bone-length CV,
    jitter, and the downstream feature checks -- because they are computed
    from geometry the two models both have to get right, not from a score
    each model defines for itself.
    """
    got = [p for p in seq if p and p.ok]
    n = len(seq)
    if not n:
        return None
    vis = [p.visible(VIS_THRESH) / len(PM.CANONICAL) for p in got]
    confs = [s for p in got for s in p.scores.values()]
    bone = PM.bone_length_cv(seq)
    jit = PM.jitter(seq)
    return {
        "n_frames": n,
        "n_pose": len(got),
        "pose_availability": round(len(got) / n, 4),
        "mean_visible_fraction_UNCALIBRATED": round(sum(vis) / len(vis), 4) if vis else None,
        "mean_confidence_UNCALIBRATED": round(sum(confs) / len(confs), 4) if confs else None,
        "bone_length_cv": round(bone, 4) if bone is not None else None,
        "jitter_torso_units": round(jit, 4) if jit is not None else None,
        **PM.feature_quality(seq),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--tracks", default=os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", default="rtmpose-m@192x256",
                    help="comma-separated keys of PM.RTMPOSE_VARIANTS")
    args = ap.parse_args()

    from phase2_cache_tracks import load_tracks
    from phase2_detector_benchmark import load_gt

    tracks = load_tracks(args.tracks)
    gt = {c["clip_id"]: c for c in load_gt()["clips"]}

    models = [PM.MediaPipePose()]
    for spec in args.variants.split(","):
        models.append(PM.RTMPosePose(variant=spec.strip()))
    print("models:", ", ".join(m.name for m in models))

    clips = sorted(set(tracks["clips"]) & set(gt))
    if args.limit:
        clips = clips[:args.limit]

    rows, lat = [], {m.name: [] for m in models}
    for n, cid in enumerate(clips, 1):
        ct = tracks["clips"][cid]["tracks"]
        gframes = [f for f in gt[cid]["frames"]
                   if f.get("batsman_visible") and f.get("batsman_bbox")]
        if not gframes:
            continue
        tid = gt_track(ct, gframes)
        if tid is None or tid not in ct:
            rows.append({"clip_id": cid, "skipped": "no_track_matches_annotated_batsman"})
            print(f"[{n}/{len(clips)}] {cid}: no track matches the annotated batsman")
            continue
        plan = clip_plan(ct, tid)
        path = os.path.join(args.videos, cid)
        if not os.path.exists(path):
            continue
        res = run_clip(path, plan, models)
        if res is None:
            continue
        seqs, timings = res
        # How large is the batsman in SOURCE pixels? A top-down model resizes
        # the crop to its fixed input, so if the person is already smaller
        # than 192x256 the model is upsampling and extra input resolution
        # cannot recover detail that was never captured. This decides whether
        # "use a higher-resolution model" is even the right lever.
        bh = [b[3] - b[1] for _, b in plan]
        bw = [b[2] - b[0] for _, b in plan]
        med_h, med_w = statistics.median(bh), statistics.median(bw)
        row = {"clip_id": cid, "track_id": tid, "n_sampled": len(plan),
               "is_pose_failure_clip": cid in POSE_FAILURE_CLIPS,
               "batsman_crop_px": {"median_w": round(med_w, 1),
                                   "median_h": round(med_h, 1),
                                   "smaller_than_model_input": bool(med_h < 256)},
               "models": {}}
        for m in models:
            row["models"][m.name] = summarise_sequence(seqs[m.name])
            lat[m.name] += timings[m.name]
        row["mediapipe_z_contribution"] = PM.z_contribution(seqs[models[0].name])

        # PAIRED-SUBSET QUALITY.
        # The headline quality numbers are not measured on the same frames:
        # MediaPipe's are computed only where it managed to return a pose
        # (~72% of frames, and by construction the EASIER ones), while
        # RTMPose's cover every frame including the hard ones it alone
        # attempted. That biases the comparison in MediaPipe's favour.
        # Restricting to frames where every model produced a pose removes the
        # bias and makes the quality comparison like-for-like.
        L = min(len(seqs[m.name]) for m in models)
        keep = [k for k in range(L)
                if all(seqs[m.name][k] and seqs[m.name][k].ok for m in models)]
        row["n_paired_frames"] = len(keep)
        row["paired"] = {}
        if len(keep) >= 3:
            for m in models:
                sub = [seqs[m.name][k] for k in keep]
                b, j = PM.bone_length_cv(sub), PM.jitter(sub)
                row["paired"][m.name] = {
                    "bone_length_cv": round(b, 4) if b is not None else None,
                    "jitter_torso_units": round(j, 4) if j is not None else None,
                    **PM.feature_quality(sub),
                }

        # Per-frame paired outcomes. The hybrid question -- "when MediaPipe
        # fails, does RTMPose succeed on THAT frame?" -- cannot be answered
        # from per-clip averages: two models can post identical availability
        # while failing on disjoint frames (fallback helps a lot) or on
        # exactly the same frames (fallback helps not at all).
        frames = []
        for k, (fidx, _) in enumerate(plan):
            rec = {"frame": fidx}
            for m in models:
                s = seqs[m.name]
                p = s[k] if k < len(s) else None
                rec[m.name] = {
                    "ok": bool(p and p.ok),
                    "mean_conf": (round(sum(p.scores.values()) / len(p.scores), 4)
                                  if p and p.ok and p.scores else None),
                }
            frames.append(rec)
        row["frames"] = frames
        rows.append(row)
        mp = row["models"][models[0].name]
        rt = row["models"][models[1].name]
        flag = "  <-- known pose failure" if cid in POSE_FAILURE_CLIPS else ""
        print(f"[{n}/{len(clips)}] {cid:<28} mp_avail={mp['pose_availability']:.2f} "
              f"rtm_avail={rt['pose_availability']:.2f}{flag}")

    os.makedirs(P3, exist_ok=True)
    with open(os.path.join(P3, "phase3_pose_comparison.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def agg(subset, key):
        out = {}
        for m in models:
            vals = [r["models"][m.name][key] for r in subset
                    if "models" in r and r["models"].get(m.name)
                    and r["models"][m.name].get(key) is not None]
            out[m.name] = round(statistics.mean(vals), 4) if vals else None
        return out

    scored = [r for r in rows if "models" in r]
    fails = [r for r in scored if r["is_pose_failure_clip"]]
    rest = [r for r in scored if not r["is_pose_failure_clip"]]

    # Ordered so the comparable metrics come first and the uncalibrated
    # per-model scores sit at the bottom where they cannot be mistaken for
    # the headline result.
    keys = ("pose_availability", "bone_length_cv", "jitter_torso_units",
            "invalid_rate", "implausible_rate", "velocity_discontinuity",
            "angle_smoothness", "mean_visible_fraction_UNCALIBRATED",
            "mean_confidence_UNCALIBRATED")
    summary = {
        "n_clips": len(scored),
        "n_pose_failure_clips": len(fails),
        "vis_threshold": VIS_THRESH,
        "all_clips": {k: agg(scored, k) for k in keys},
        "known_pose_failures": {k: agg(fails, k) for k in keys},
        "other_clips": {k: agg(rest, k) for k in keys},
        "latency_ms_per_frame_per_track": {
            m.name: {
                "median": round(1000 * statistics.median(lat[m.name]), 2),
                "mean": round(1000 * statistics.mean(lat[m.name]), 2),
                "n": len(lat[m.name]),
            } for m in models if lat[m.name]
        },
        "note_latency": "CPU for both models; no CUDAExecutionProvider available here.",
    }
    zs = [r["mediapipe_z_contribution"]["median_deg"] for r in scored
          if r.get("mediapipe_z_contribution")]
    summary["mediapipe_z_contribution_median_deg"] = (
        round(statistics.mean(zs), 2) if zs else None)

    def crop_stats(subset):
        hs = [r["batsman_crop_px"]["median_h"] for r in subset if "batsman_crop_px" in r]
        if not hs:
            return None
        return {"median_person_height_px": round(statistics.median(hs), 1),
                "min_px": round(min(hs), 1), "max_px": round(max(hs), 1),
                "n_below_model_input_256px": sum(1 for h in hs if h < 256),
                "n": len(hs)}
    summary["crop_resolution"] = {
        "all_clips": crop_stats(scored),
        "known_pose_failures": crop_stats(fails),
        "note": ("RTMPose resizes each crop to its fixed input height (256 or "
                 "384). Persons already shorter than that are UPSAMPLED, so a "
                 "higher-resolution model adds no real detail for them."),
    }
    def hybrid(subset, other):
        """Paired frame-level contingency: MediaPipe vs one RTMPose variant.

        `mp_fail_rtm_ok` is the ceiling on what a fallback can recover, and
        `both_fail` is the floor no choice of model fixes. If both_fail
        dominates, the bottleneck is upstream of the pose model.
        """
        c = {"both_ok": 0, "mp_fail_rtm_ok": 0, "mp_ok_rtm_fail": 0, "both_fail": 0}
        for r in subset:
            for fr in r.get("frames", []):
                a = fr[models[0].name]["ok"]
                b = fr[other]["ok"]
                c["both_ok" if (a and b) else
                  "mp_fail_rtm_ok" if (not a and b) else
                  "mp_ok_rtm_fail" if (a and not b) else "both_fail"] += 1
        n = sum(c.values())
        if n:
            c["n_frames"] = n
            c["fallback_recoverable_pct"] = round(100 * c["mp_fail_rtm_ok"] / n, 2)
            c["irreducible_pct"] = round(100 * c["both_fail"] / n, 2)
        return c

    def agg_paired(subset, key):
        out = {}
        for m in models:
            vals = [r["paired"][m.name][key] for r in subset
                    if r.get("paired", {}).get(m.name)
                    and r["paired"][m.name].get(key) is not None]
            out[m.name] = round(statistics.mean(vals), 4) if vals else None
        return out

    pkeys = ("bone_length_cv", "jitter_torso_units", "invalid_rate",
             "implausible_rate", "velocity_discontinuity", "angle_smoothness")
    summary["paired_subset"] = {
        "note": ("Computed only on frames where EVERY model returned a pose, "
                 "so all models are scored on identical frames. The headline "
                 "table is not: MediaPipe is scored there only on the frames "
                 "it could do, which are the easier ones."),
        "n_paired_frames": sum(r.get("n_paired_frames", 0) for r in scored),
        "all_clips": {k: agg_paired(scored, k) for k in pkeys},
        "known_pose_failures": {k: agg_paired(fails, k) for k in pkeys},
    }
    summary["hybrid_fallback"] = {
        m.name: {"all_clips": hybrid(scored, m.name),
                 "known_pose_failures": hybrid(fails, m.name)}
        for m in models[1:]
    }
    summary["confounds"] = {
        "rtmpose-x@288x384": ("Changes BOTH capacity (m->x) and input "
                              "resolution vs rtmpose-m@192x256; no halpe26 "
                              "rtmpose-m exists at 384x288, so these cannot "
                              "be separated with published weights. Treat as "
                              "an upper bound on RTMPose, not a resolution "
                              "result. rtmpose-s@192x256 vs rtmpose-m@192x256 "
                              "isolates capacity at fixed resolution."),
        "mediapipe_input": ("MediaPipe runs on a crop we cut with 12% padding; "
                            "RTMPose runs top-down on the original frame with "
                            "the same box. One resampling step differs."),
    }

    with open(os.path.join(P3, "phase3_pose_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    print(f"\n{'metric':<26}" + "".join(f"{m.name[:22]:>24}" for m in models))
    for scope, label in ((scored, "ALL"), (fails, "POSE-FAILURE CLIPS")):
        print(f"\n-- {label} (n={len(scope)})")
        for k in keys:
            a = agg(scope, k)
            print(f"{k:<26}" + "".join(
                f"{(f'{a[m.name]:.4f}' if a[m.name] is not None else '-'):>24}"
                for m in models))
    print(f"\n-- PAIRED SUBSET (identical frames for every model, "
          f"n={summary['paired_subset']['n_paired_frames']} frames)")
    for k in pkeys:
        a = agg_paired(scored, k)
        print(f"{k:<26}" + "".join(
            f"{(f'{a[m.name]:.4f}' if a[m.name] is not None else '-'):>24}"
            for m in models))
    print(f"\n-- PAIRED SUBSET, pose-failure clips only")
    for k in pkeys:
        a = agg_paired(fails, k)
        print(f"{k:<26}" + "".join(
            f"{(f'{a[m.name]:.4f}' if a[m.name] is not None else '-'):>24}"
            for m in models))
    print(f"\nwrote {os.path.join(P3, 'phase3_pose_comparison.json')}")


if __name__ == "__main__":
    main()
