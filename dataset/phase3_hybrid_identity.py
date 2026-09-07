"""
phase3_hybrid_identity.py — does the RTMPose fallback improve S2 IDENTITY
decisions, or merely produce more poses?

    H0   S2 + MediaPipe only
    H1   S2 + MediaPipe, RTMPose-s where MediaPipe returns no pose

This is deliberately NOT the previous pose experiment. That one anchored on
the annotated batsman track to isolate the pose variable, which was right for
measuring pose. Here the question is downstream and end-to-end, so the system
runs on the ACTUAL candidate tracks and picks the batsman itself. Ground truth
is used only to score the decision, never to make it.

Everything except the pose backend is held identical: same detections, same
ByteTrack tracks, same sampled frames, same bat detections, same feature code,
same fitting routine, same decision thresholds, same dev/eval split.

WHY BOTH A REFIT AND A FROZEN-MODEL ARM
Refitting on dev per condition matches how S2 is defined (every ablation arm
is fitted on dev and reported on eval), but it means two things change at
once: the features AND the coefficients. The frozen arm applies the
H0-fitted model to H1 features, isolating the effect of the recovered poses
alone. Neither is sufficient by itself; disagreement between them is
informative rather than embarrassing.

THE FAILURE MODE THIS IS WATCHING FOR
RTMPose returns a skeleton for ANY box it is given -- it cannot decline. So a
fallback can manufacture skeleton evidence for a bad candidate and convert a
correct refusal into a confident wrong answer. More pose is not the goal;
better identity decisions are. Transitions are therefore classified
individually, and `correct->wrong` / `refused->wrong` are called out as
regressions no aggregate improvement excuses.
"""

import argparse
import json
import os

import phase3_batsman_ablation as AB
import phase3_failure_taxonomy as FT

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

SYSTEM = "S2_skeleton"

# Transitions that are strictly worse, regardless of what the aggregate does.
REGRESSIONS = ("correct->wrong", "correct->refused", "refused->wrong")
IMPROVEMENTS = ("refused->correct", "wrong->correct", "wrong->refused")


def fit_on_dev(rows, features):
    dev = [r for r in rows if r["split"] == "dev"]
    X = [[r["tracks"][t].get(k, 0.0) for k in features]
         for r in dev for t in r["tracks"] if r["gt_track"] is not None]
    y = [1 if t == r["gt_track"] else 0
         for r in dev for t in r["tracks"] if r["gt_track"] is not None]
    coef, inter = AB.fit_logistic(X, y)
    return {"coef": dict(zip(features, map(float, coef))), "intercept": float(inter)}


def pose_recovery(row, tid):
    """Frames the fallback recovered for one track, from the extraction counts."""
    d = (row.get("diagnostics") or {}).get(tid) or {}
    return d


def compare(h0_rows, h1_rows, features, model_h0, model_h1, split):
    """Per-clip H0 vs H1 outcomes plus the transition each clip made."""
    e0 = AB.evaluate(h0_rows, features, model_h0, split)
    e1 = AB.evaluate(h1_rows, features, model_h1, split)
    d0 = {d["clip_id"]: d for d in e0["detail"]}
    d1 = {d["clip_id"]: d for d in e1["detail"]}

    h1_by_clip = {r["clip_id"]: r for r in h1_rows}
    h0_by_clip = {r["clip_id"]: r for r in h0_rows}

    changed, transitions = [], {}
    for cid, a in d0.items():
        b = d1.get(cid)
        if not b:
            continue
        if a["outcome"] == b["outcome"] and a["picked_track"] == b["picked_track"]:
            continue
        t = f"{a['outcome']}->{b['outcome']}"
        transitions[t] = transitions.get(t, 0) + 1

        r0, r1 = h0_by_clip.get(cid), h1_by_clip.get(cid)
        # What actually changed for the tracks involved.
        tracks = {}
        for tid in {a["picked_track"], b["picked_track"], a["gt_track"]}:
            if tid is None:
                continue
            f0 = (r0["tracks"].get(tid) or {}) if r0 else {}
            f1 = (r1["tracks"].get(tid) or {}) if r1 else {}
            dg0, dg1 = pose_recovery(r0 or {}, tid), pose_recovery(r1 or {}, tid)
            deltas = {k: [round(f0.get(k, 0.0), 4), round(f1.get(k, 0.0), 4)]
                      for k in AB.SKEL + AB.EQUIP
                      if abs(f1.get(k, 0.0) - f0.get(k, 0.0)) > 1e-6}
            tracks[str(tid)] = {
                "is_gt": tid == a["gt_track"],
                "picked_by": ("H0" if tid == a["picked_track"] else "") +
                             ("H1" if tid == b["picked_track"] else ""),
                "H0_score": a["scores"].get(str(tid)),
                "H1_score": b["scores"].get(str(tid)),
                "pose_frames_H0": dg0.get("n_pose"),
                "pose_frames_H1": dg1.get("n_pose"),
                "pose_frames_recovered": (
                    (dg1.get("n_pose") or 0) - (dg0.get("n_pose") or 0)),
                "coverage": dg0.get("coverage"),
                "failure_type_H0": dg0.get("failure_type"),
                "failure_type_H1": dg1.get("failure_type"),
                "features_changed": deltas,
            }
        changed.append({
            "clip_id": cid, "split": split, "transition": t,
            "is_regression": t in REGRESSIONS,
            "gt_track": a["gt_track"],
            "H0": {"outcome": a["outcome"], "verdict": a["verdict"],
                   "picked_track": a["picked_track"],
                   "confidence": a["confidence"], "margin": a["margin"]},
            "H1": {"outcome": b["outcome"], "verdict": b["verdict"],
                   "picked_track": b["picked_track"],
                   "confidence": b["confidence"], "margin": b["margin"]},
            "tracks": tracks,
        })
    return e0, e1, changed, transitions


def headline(e):
    return {k: e[k] for k in ("clips", "correct", "wrong", "refused",
                              "batsman_precision", "wrong_person_rate",
                              "refusal_rate", "accuracy_all_clips")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h0", required=True, help="H0 features json")
    ap.add_argument("--h1", required=True, help="H1 features json")
    ap.add_argument("--tracks", default=os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--out", default=os.path.join(P3, "phase3_hybrid_identity.json"))
    args = ap.parse_args()

    features = AB.SYSTEMS[SYSTEM]
    h0_rows = AB.load_everything(args.tracks, args.h0)
    h1_rows = AB.load_everything(args.tracks, args.h1)

    model_h0 = fit_on_dev(h0_rows, features)
    model_h1 = fit_on_dev(h1_rows, features)

    out = {
        "system": SYSTEM,
        "features": list(features),
        "protocol": ("Actual candidate tracks; the system selects the batsman "
                     "itself. Ground truth scores the decision only. Fitted on "
                     "dev, reported on eval."),
        "conditions": {
            "H0": "S2 + MediaPipe only",
            "H1": "S2 + MediaPipe, RTMPose-s fallback on MediaPipe refusal",
        },
        "status": "EXPERIMENTAL — H1 is not production",
    }

    for split in ("dev", "eval"):
        e0, e1, changed, trans = compare(h0_rows, h1_rows, features,
                                         model_h0, model_h1, split)
        out[split] = {
            "H0": headline(e0), "H1": headline(e1),
            "transitions": trans,
            "n_changed": len(changed),
            "n_regressions": sum(1 for c in changed if c["is_regression"]),
            "changed_clips": changed,
        }

    # Frozen-model arm: H0's coefficients applied to H1's features, so the
    # recovered poses are the only thing that differs.
    for split in ("dev", "eval"):
        ef = AB.evaluate(h1_rows, features, model_h0, split)
        out[split]["H1_frozen_H0_model"] = headline(ef)

    out["candidate_recall"] = AB.candidate_recall(h0_rows, args.tracks)
    out["candidate_recall_note"] = ("Identical for H0 and H1 by construction — "
                                    "both consume the same tracks. Pose cannot "
                                    "change which candidates exist.")
    out["models"] = {"H0": model_h0, "H1": model_h1}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    for split in ("dev", "eval"):
        s = out[split]
        print(f"\n=== {split.upper()} ===")
        print(f"{'':<26}{'correct':>9}{'wrong':>8}{'refused':>9}"
              f"{'precision':>11}{'wrong-rate':>12}{'refusal':>9}")
        for name in ("H0", "H1", "H1_frozen_H0_model"):
            h = s[name]
            fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"{name:<26}{h['correct']:>9}{h['wrong']:>8}{h['refused']:>9}"
                  f"{fmt(h['batsman_precision']):>11}"
                  f"{fmt(h['wrong_person_rate']):>12}"
                  f"{fmt(h['refusal_rate']):>9}")
        print(f"changed decisions: {s['n_changed']}  "
              f"(regressions: {s['n_regressions']})")
        for t, n in sorted(s["transitions"].items()):
            flag = "  <-- REGRESSION" if t in REGRESSIONS else ""
            print(f"    {t:<22} {n}{flag}")
    print(f"\ncandidate recall (identical both arms): {out['candidate_recall']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
