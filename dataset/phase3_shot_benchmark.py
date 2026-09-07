"""
phase3_shot_benchmark.py — L0-L3 shot localization against the human
annotations.

SPLIT DISCIPLINE
`alpha` (boundary threshold), `tau` (rejection prominence) and the duration
clamp are selected on DEV by grid search and applied unchanged to EVAL. Eval
is scored once. Nothing here is chosen by looking at eval.

IDENTITY IS KEPT SEPARATE FROM LOCALIZATION
Two questions, reported separately, because an identity error must not be
allowed to masquerade as a localization error:

  A. was the correct batsman selected?   (S2, already measured)
  B. given that track, was the shot localized?

The batsman box feeding L2/L3 comes from S2's OWN selection, so the
"all clips" numbers are honestly end-to-end. The identity-conditioned numbers
restrict to clips where S2 picked the annotated batsman, which is the clean
measurement of the localization component itself.

INVALID CLIPS ARE NOT GIVEN FAKE INTERVALS
NO_SHOT / MULTIPLE_SHOTS / SCENE_CUT / AMBIGUOUS / EXCESSIVE_DURATION /
INSUFFICIENT_ACTION clips have no annotated single-shot boundaries, so IoU is
not computed for them. They are scored on rejection behaviour instead.

Multi-shot contamination note: the annotations mark a clip as MULTIPLE_SHOTS
but do not carry per-shot boundaries, so "does the interval contain more than
one batting event" is NOT computable. The honest proxy reported here is
whether the method declined the clip at all.
"""

import argparse
import json
import os
import statistics
import time

import numpy as np

import phase3_shot_localization as SL

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

VALID = "VALID_SINGLE_SHOT"
IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def s2_selection(tracks_path, features_path):
    """Which track S2 picks per clip, and whether that is the annotated
    batsman. Uses the committed MediaPipe-only features (H1 was not adopted)."""
    import phase3_batsman_ablation as AB
    import phase3_hybrid_identity as HI
    rows = AB.load_everything(tracks_path, features_path)
    feats = AB.SYSTEMS["S2_skeleton"]
    model = HI.fit_on_dev(rows, feats)
    out = {}
    for split in ("dev", "eval"):
        for d in AB.evaluate(rows, feats, model, split)["detail"]:
            out[d["clip_id"]] = {
                "picked_track": d["picked_track"],
                "gt_track": d["gt_track"],
                "identity_correct": (d["picked_track"] is not None
                                     and d["picked_track"] == d["gt_track"]),
                "identity_outcome": d["outcome"],
            }
    return out


def gather(video_dir, tracks_path, features_path, max_frames=400):
    """Decode each clip ONCE and cache both motion signals.

    One decode per clip, shared by every method and every parameter setting in
    the dev grid search — otherwise the search would re-decode 51 videos for
    each grid point.
    """
    from phase2_cache_tracks import load_tracks
    from phase3_annotations import build_records

    tracks = load_tracks(tracks_path)
    recs = {r["video_id"]: r for r in build_records()}
    sel = s2_selection(tracks_path, features_path)

    # build_records() does not carry the split; phase3_build_annotations
    # attaches it from the Phase-2 ground truth. Do the same here rather than
    # letting every clip default to split=None, which silently empties the
    # eval tables.
    with open(os.path.join(P2, "phase2_groundtruth.json"), encoding="utf-8") as f:
        split_of = {c["clip_id"]: c["split"] for c in json.load(f)["clips"]}

    clips = []
    for cid, rec in recs.items():
        path = os.path.join(video_dir, cid)
        if not os.path.exists(path):
            continue
        tc = tracks["clips"].get(cid)
        s = sel.get(cid, {})
        tid = s.get("picked_track")
        by_frame = None
        if tc and tid is not None and tid in tc["tracks"]:
            by_frame = {f: b for f, b, _ in tc["tracks"][tid]}
        t0 = time.perf_counter()
        n, g, l = SL.motion_signals(path, by_frame, max_frames=max_frames)
        decode_s = time.perf_counter() - t0
        if n == 0 or g is None:
            continue
        clips.append({
            "clip_id": cid,
            "split": split_of.get(cid),
            "coherence": rec["coherence"],
            "gt_start": rec["shot_start_frame"],
            "gt_end": rec["shot_end_frame"],
            "scene_cut_frames": rec.get("scene_cut_frames") or [],
            "n_frames": n,
            "g": g, "l": l,
            "has_track": by_frame is not None,
            "decode_s": decode_s,
            **{k: s.get(k) for k in ("picked_track", "gt_track",
                                     "identity_correct", "identity_outcome")},
        })
    return clips


def evaluate(clips, method, params, subset=None):
    """Score one method over a clip list."""
    use = [c for c in clips if subset is None or subset(c)]
    valid = [c for c in use if c["coherence"] == VALID
             and c["gt_start"] is not None]
    invalid = [c for c in use if c["coherence"] != VALID]

    ious, se, ee, per_clip = [], [], [], []
    n_rejected_valid = 0
    for c in valid:
        p = SL.predict(method, c["n_frames"], c["g"], c["l"], **params)
        gt = (c["gt_start"], c["gt_end"])
        if p["rejected"]:
            n_rejected_valid += 1
            iou = 0.0
            per_clip.append({"clip_id": c["clip_id"], "split": c["split"],
                             "gt": list(gt), "pred": None, "iou": 0.0,
                             "rejected": True,
                             "identity_correct": c["identity_correct"]})
            ious.append(0.0)
            continue
        pr = (p["start"], p["end"])
        iou = SL.temporal_iou(pr, gt)
        ious.append(iou)
        se.append(abs(pr[0] - gt[0]))
        ee.append(abs(pr[1] - gt[1]))
        per_clip.append({
            "clip_id": c["clip_id"], "split": c["split"], "gt": list(gt),
            "pred": list(pr), "iou": round(iou, 4),
            "start_err": abs(pr[0] - gt[0]), "end_err": abs(pr[1] - gt[1]),
            "rejected": False, "prominence": p["prominence"],
            "identity_correct": c["identity_correct"],
            "crosses_scene_cut": SL.crosses(pr, c["scene_cut_frames"]),
        })

    # Invalid clips: rejection behaviour only. No IoU, no fabricated interval.
    inv_detail, n_false_shot = [], 0
    sc_total = sc_crossed = 0
    for c in invalid:
        p = SL.predict(method, c["n_frames"], c["g"], c["l"], **params)
        emitted = not p["rejected"]
        if emitted:
            n_false_shot += 1
        pr = (p["start"], p["end"]) if emitted else None
        if c["scene_cut_frames"]:
            sc_total += 1
            if emitted and SL.crosses(pr, c["scene_cut_frames"]):
                sc_crossed += 1
        inv_detail.append({"clip_id": c["clip_id"], "split": c["split"],
                           "coherence": c["coherence"],
                           "pred": list(pr) if pr else None,
                           "emitted_interval": emitted,
                           "crosses_scene_cut": (SL.crosses(pr, c["scene_cut_frames"])
                                                 if emitted else False)})

    n_multi = sum(1 for c in invalid if c["coherence"] == "MULTIPLE_SHOTS")
    n_multi_emitted = sum(1 for d in inv_detail
                          if d["coherence"] == "MULTIPLE_SHOTS"
                          and d["emitted_interval"])

    def rate(th):
        return round(sum(1 for i in ious if i >= th) / len(ious), 4) if ious else None

    return {
        "method": method, "n_valid": len(valid), "n_invalid": len(invalid),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "median_iou": round(float(np.median(ious)), 4) if ious else None,
        "recall@0.3": rate(0.3), "recall@0.5": rate(0.5), "recall@0.7": rate(0.7),
        "median_start_err": round(float(np.median(se)), 2) if se else None,
        "median_end_err": round(float(np.median(ee)), 2) if ee else None,
        "mean_start_err": round(float(np.mean(se)), 2) if se else None,
        "mean_end_err": round(float(np.mean(ee)), 2) if ee else None,
        "rejected_valid_clips": n_rejected_valid,
        "localization_recall": round(1 - n_rejected_valid / len(valid), 4) if valid else None,
        "false_shot_rate": round(n_false_shot / len(invalid), 4) if invalid else None,
        "n_false_shot": n_false_shot,
        "multi_shot_emitted": n_multi_emitted, "n_multi_shot": n_multi,
        "multi_shot_contamination": round(n_multi_emitted / n_multi, 4) if n_multi else None,
        "scene_cut_clips_with_cut_frames": sc_total,
        "scene_cut_crossed": sc_crossed,
        "scene_cut_contamination": round(sc_crossed / sc_total, 4) if sc_total else None,
        "detail": per_clip, "invalid_detail": inv_detail,
    }


def fit_on_dev(clips, method):
    """Grid-search the shared windowing parameters on DEV only.

    Objective is mean IoU on dev valid clips minus a penalty for emitting
    intervals on dev invalid clips, so the search cannot buy IoU by never
    rejecting anything.
    """
    if method == "L0_whole_clip":
        return {}, None
    dev = lambda c: c["split"] == "dev"
    best, best_score = None, -1e9
    for alpha in (0.20, 0.30, 0.35, 0.45, 0.55, 0.65):
        for tau in (1.2, 1.4, 1.6, 1.8, 2.2):
            for mx in (30, 40, 50):
                p = {"alpha": alpha, "tau": tau, "min_dur": 12, "max_dur": mx}
                r = evaluate(clips, method, p, subset=dev)
                if r["mean_iou"] is None:
                    continue
                score = r["mean_iou"] - 0.25 * (r["false_shot_rate"] or 0.0)
                if score > best_score:
                    best, best_score = p, score
    return (best or {}), round(best_score, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--tracks", default=os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--features", default=os.path.join(P3, "phase3_semantic_features.json"))
    ap.add_argument("--out", default=os.path.join(P3, "phase3_shot_localization.json"))
    args = ap.parse_args()

    clips = gather(args.videos, args.tracks, args.features)
    print(f"gathered {len(clips)} clips")

    out = {
        "n_clips": len(clips),
        "iou_thresholds": list(IOU_THRESHOLDS),
        "protocol": ("Windowing parameters fit on dev by grid search, applied "
                     "unchanged to eval. L2/L3 use S2's own batsman selection, "
                     "so 'all clips' is end-to-end; identity-conditioned "
                     "results restrict to clips where S2 picked the annotated "
                     "batsman."),
        "l0_note": ("L0 is the whole clip: the production pipeline has no "
                    "intra-clip shot localizer -- select_phase_frames is handed "
                    "start/end from the external ingestion step."),
        "multi_shot_note": ("MULTIPLE_SHOTS clips carry no per-shot boundaries, "
                            "so 'interval contains >1 event' is not computable. "
                            "Reported proxy: did the method emit an interval."),
        "methods": {},
    }

    for m in SL.METHODS:
        params, dev_score = fit_on_dev(clips, m)
        entry = {"params_fit_on_dev": params, "dev_objective": dev_score}
        for split, sub in (("dev", lambda c: c["split"] == "dev"),
                           ("eval", lambda c: c["split"] == "eval"),
                           ("all", None)):
            entry[split] = evaluate(clips, m, params, subset=sub)
        # Identity-conditioned: only clips where S2 picked the right batsman.
        entry["eval_identity_correct"] = evaluate(
            clips, m, params,
            subset=lambda c: c["split"] == "eval" and c["identity_correct"])
        entry["all_identity_correct"] = evaluate(
            clips, m, params, subset=lambda c: c["identity_correct"])
        out["methods"][m] = entry

    n_id_ok = sum(1 for c in clips if c["identity_correct"])
    out["identity"] = {
        "clips_with_correct_identity": n_id_ok,
        "clips_total": len(clips),
        "valid_clips_with_correct_identity": sum(
            1 for c in clips if c["identity_correct"] and c["coherence"] == VALID),
        "valid_clips_total": sum(1 for c in clips if c["coherence"] == VALID),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    for split in ("eval", "all", "all_identity_correct"):
        print(f"\n=== {split} ===")
        print(f"{'method':<20}{'mIoU':>7}{'medIoU':>8}{'@0.3':>7}{'@0.5':>7}"
              f"{'@0.7':>7}{'sErr':>7}{'eErr':>7}{'reject':>8}{'falseShot':>11}")
        for m in SL.METHODS:
            r = out["methods"][m][split]
            f_ = lambda v, d=2: ("-" if v is None else f"{v:.{d}f}")
            print(f"{m:<20}{f_(r['mean_iou'],3):>7}{f_(r['median_iou'],3):>8}"
                  f"{f_(r['recall@0.3']):>7}{f_(r['recall@0.5']):>7}"
                  f"{f_(r['recall@0.7']):>7}{f_(r['median_start_err'],1):>7}"
                  f"{f_(r['median_end_err'],1):>7}"
                  f"{r['rejected_valid_clips']:>8}{f_(r['false_shot_rate']):>11}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
