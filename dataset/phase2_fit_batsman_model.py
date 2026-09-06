"""
phase2_fit_batsman_model.py — fit the track-level batsman scorer on DEV and
report it on EVAL, alongside the rules it is meant to replace.

STATISTICAL DISCIPLINE
The model is fitted on DEV clips ONLY. EVAL clips are touched exactly once,
to produce the reported numbers. Thresholds are not tuned against EVAL.

WHAT COUNTS AS THE GROUND-TRUTH TRACK
The annotated batsman box exists on 6 sampled frames per clip. A track is the
batsman track if it matches (IoU >= 0.5) the annotated box on more sampled
frames than any other track, and on at least one. Clips where no track
matches any annotated frame have no positive label and are excluded from
FITTING — but they are kept in EVALUATION as unavoidable failures, so the
reported accuracy is not inflated by quietly dropping the hard clips.
"""

import argparse
import json
import math
import os

import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_MODULE_DIR, "phase2_results")

IOU_MATCH = 0.5


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_track_for_clip(clip_tracks, gt_frames):
    """Which track is the batsman? The one matching the annotated box on the
    most sampled frames."""
    counts = {}
    for frec in gt_frames:
        box = frec["batsman_bbox"]
        if not box:
            continue
        best, best_iou = None, 0.0
        for tid, obs in clip_tracks.items():
            for f, tbox, _ in obs:
                if f == frec["frame"]:
                    v = _iou(tbox, box)
                    if v > best_iou:
                        best, best_iou = tid, v
                    break
        if best is not None and best_iou >= IOU_MATCH:
            counts[best] = counts.get(best, 0) + 1
    if not counts:
        return None, 0
    tid = max(counts, key=counts.get)
    return tid, counts[tid]


def build_dataset(tracks_data, gt, feature_names=None):
    from phase2_batsman_score import track_features, FEATURE_NAMES
    feature_names = feature_names or FEATURE_NAMES

    gt_by_clip = {c["clip_id"]: c for c in gt["clips"]}
    rows = []
    for cid, cdata in tracks_data["clips"].items():
        clip_gt = gt_by_clip.get(cid)
        if not clip_gt:
            continue
        gt_frames = [f for f in clip_gt["frames"] if f["batsman_visible"] and f["batsman_bbox"]]
        gt_tid, n_match = gt_track_for_clip(cdata["tracks"], gt_frames)

        ctx = {"frame_w": cdata["frame_w"], "frame_h": cdata["frame_h"],
               "frames_processed": cdata["frames_processed"]}
        for tid, obs in cdata["tracks"].items():
            feats = track_features(obs, ctx["frame_w"], ctx["frame_h"], ctx["frames_processed"])
            rows.append({
                "clip_id": cid, "split": cdata["split"], "track_id": tid,
                "label": 1 if (gt_tid is not None and tid == gt_tid) else 0,
                "x": [feats[k] for k in feature_names],
                "features": feats,
            })
        gt_by_clip[cid]["_gt_track"] = gt_tid
        gt_by_clip[cid]["_gt_match_frames"] = n_match
    return rows, gt_by_clip


def fit_logistic(X, y, l2=1.0, iters=4000, lr=0.5):
    """Plain gradient-descent logistic regression.

    Deliberately dependency-free: sklearn is not installed here and adding it
    for eight features and a few hundred rows would be a heavier dependency
    than the model. L2 keeps coefficients from exploding on the separable
    directions that a small dev split can easily contain.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        gw = Xs.T @ (p - y) / n + (l2 / n) * w
        gb = float((p - y).mean())
        w -= lr * gw
        b -= lr * gb
    # Unstandardise so the stored coefficients apply to raw features.
    coef = w / sd
    intercept = b - float((w * mu / sd).sum())
    return coef, intercept


def evaluate_selection(rows, gt_by_clip, model, split):
    """Per-clip: does the chosen track equal the ground-truth batsman track?"""
    from phase2_batsman_score import (select_batsman, baseline_largest,
                                      baseline_size_dominance, baseline_most_central,
                                      baseline_most_persistent)

    by_clip = {}
    for r in rows:
        if r["split"] != split:
            continue
        by_clip.setdefault(r["clip_id"], []).append(r)

    stats = {k: {"correct": 0, "wrong": 0, "refused": 0}
             for k in ("model", "largest", "size_dominance", "most_central", "most_persistent")}
    verdicts = {"CONFIDENT_BATSMAN": 0, "AMBIGUOUS": 0, "NO_BATSMAN": 0}
    detail = []

    for cid, rs in by_clip.items():
        gt_tid = gt_by_clip[cid].get("_gt_track")
        # Rebuild the observation lists the baselines need.
        obs_by_tid = {r["track_id"]: r["_obs"] for r in rs}
        ctx = rs[0]["_ctx"]

        sel = select_batsman(obs_by_tid, ctx, model)
        verdicts[sel["verdict"]] += 1
        if sel["verdict"] == "CONFIDENT_BATSMAN":
            key = "correct" if sel["track_id"] == gt_tid else "wrong"
        else:
            key = "refused"
        stats["model"][key] += 1

        for name, fn in (("largest", baseline_largest),
                         ("size_dominance", baseline_size_dominance),
                         ("most_central", baseline_most_central),
                         ("most_persistent", baseline_most_persistent)):
            pick = fn(obs_by_tid, ctx)
            if pick is None:
                stats[name]["refused"] += 1
            elif gt_tid is not None and pick == gt_tid:
                stats[name]["correct"] += 1
            else:
                stats[name]["wrong"] += 1

        detail.append({"clip_id": cid, "gt_track": gt_tid,
                       "model_verdict": sel["verdict"],
                       "model_pick": sel.get("best_track_id"),
                       "model_conf": sel.get("confidence"),
                       "correct": sel.get("track_id") == gt_tid})

    n = len(by_clip)
    out = {"split": split, "clips": n, "verdicts": verdicts, "rules": {}}
    for name, s in stats.items():
        decided = s["correct"] + s["wrong"]
        out["rules"][name] = {
            **s,
            # Accuracy among clips where the rule COMMITTED to an answer.
            "precision_when_committed": round(s["correct"] / decided, 4) if decided else None,
            # Accuracy over ALL clips: refusing counts as not-correct, which
            # is the number that matters for how much usable data survives.
            "accuracy_all_clips": round(s["correct"] / n, 4) if n else None,
        }
    out["detail"] = detail
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=os.path.join(RESULTS, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "phase2_batsman_model.json"))
    ap.add_argument("--features", default="full", choices=["full", "geometry"],
                    help="'geometry' fits only the crease/stability features, "
                         "isolating the cricket-geometric prior from size and position")
    args = ap.parse_args()

    from phase2_cache_tracks import load_tracks
    from phase2_detector_benchmark import load_gt
    from phase2_batsman_score import FEATURE_NAMES

    GEOMETRY_ONLY = ("coverage", "foot_stability", "scale_stability",
                     "centroid_stability", "vertical_position")
    feature_names = FEATURE_NAMES if args.features == "full" else GEOMETRY_ONLY

    tracks_data = load_tracks(args.tracks)
    gt = load_gt()
    rows, gt_by_clip = build_dataset(tracks_data, gt, feature_names)

    # Attach observations/context for the baselines and the selector.
    for r in rows:
        c = tracks_data["clips"][r["clip_id"]]
        r["_obs"] = c["tracks"][r["track_id"]]
        r["_ctx"] = {"frame_w": c["frame_w"], "frame_h": c["frame_h"],
                     "frames_processed": c["frames_processed"]}

    dev = [r for r in rows if r["split"] == "dev"]
    dev_fit = [r for r in dev if gt_by_clip[r["clip_id"]].get("_gt_track") is not None]
    X = [r["x"] for r in dev_fit]
    y = [r["label"] for r in dev_fit]
    print(f"dev tracks: {len(dev)} ({sum(r['label'] for r in dev)} positive); "
          f"fitting on {len(dev_fit)} from clips with a GT track")

    coef, intercept = fit_logistic(X, y)
    model = {"coef": {k: float(v) for k, v in zip(feature_names, coef)},
             "intercept": float(intercept)}

    print("\nfitted coefficients (positive => raises batsman score):")
    for k in sorted(model["coef"], key=lambda k: -abs(model["coef"][k])):
        print(f"  {k:<20} {model['coef'][k]:+8.3f}")
    print(f"  {'(intercept)':<20} {model['intercept']:+8.3f}")

    report = {"model": model, "feature_set": args.features,
              "feature_names": list(feature_names), "splits": {}}
    for split in ("dev", "eval"):
        r = evaluate_selection(rows, gt_by_clip, model, split)
        report["splits"][split] = r
        print(f"\n=== {split.upper()} ({r['clips']} clips) ===")
        print(f"  model verdicts: {r['verdicts']}")
        print(f"  {'rule':<18} {'correct':>7} {'wrong':>6} {'refused':>8} "
              f"{'prec|commit':>12} {'acc|all':>9}")
        for name, s in r["rules"].items():
            print(f"  {name:<18} {s['correct']:>7} {s['wrong']:>6} {s['refused']:>8} "
                  f"{str(s['precision_when_committed']):>12} {str(s['accuracy_all_clips']):>9}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
