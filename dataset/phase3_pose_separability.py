"""
phase3_pose_separability.py — can a confidence gate make RTMPose able to say
"nobody is here"?

RTMPose cannot decline: handed any box it emits 26 keypoints, including on a
flat grey image. That is why its raw pose_availability is 1.0 everywhere and
why that number must not be read as "it found the batsman".

A gate on confidence is the obvious repair. Whether the repair WORKS is an
empirical question: it works only if the score separates real batsmen from
person-free background. This script measures that separation directly as the
AUC between

    positives = per-frame mean joint score on the annotated batsman box
    negatives = per-frame mean joint score on YOLO-verified person-free boxes

AUC 0.5 means the score carries no information about whether a person is
there, and no threshold can help. AUC near 1.0 means a gate is viable and the
p95-of-null gate is a reasonable operating point.

MediaPipe is included for reference using its own implicit gate: it either
returns a pose or it does not, which is a single operating point rather than
a curve.
"""

import json
import os

import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.join(_MODULE_DIR, "phase3_results")


def auc(pos, neg):
    """Mann-Whitney U / rank-based AUC, ties counted at half."""
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    with open(os.path.join(P3, "phase3_pose_null_calibration.json"),
              encoding="utf-8") as f:
        cal = json.load(f)
    rows = []
    with open(os.path.join(P3, "phase3_pose_comparison.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "frames" in r:
                rows.append(r)

    out = {"n_clips": len(rows), "models": {}}
    for name, c in cal["models"].items():
        neg = c.get("raw_null_scores") or []
        pos = [fr[name]["mean_conf"] for r in rows for fr in r["frames"]
               if fr.get(name, {}).get("mean_conf") is not None]
        if not (pos and neg):
            continue
        a = auc(pos, neg)
        gate = c["gate"]
        kept = sum(1 for p in pos if p >= gate) / len(pos)
        fp = sum(1 for n in neg if n >= gate) / len(neg)
        # The comparison that is actually fair. MediaPipe sits at ONE
        # operating point (it declines by itself); RTMPose only becomes
        # comparable once gated to the SAME false-positive rate on
        # person-free boxes. Comparing raw availability against a model that
        # cannot decline is not a comparison at all.
        mp_fpr = cal.get("mediapipe_false_pose_rate_on_null")
        matched = None
        if mp_fpr is not None and neg:
            g = float(np.percentile(neg, 100 * (1 - mp_fpr)))
            matched = {
                "matched_fpr": round(mp_fpr, 4),
                "gate": round(g, 4),
                "real_batsman_frames_kept": round(
                    sum(1 for p in pos if p >= g) / len(pos), 4),
            }
        sweep = {}
        for f in (0.01, 0.02, 0.05, 0.10):
            g = float(np.percentile(neg, 100 * (1 - f)))
            sweep[f"fpr_{f:.2f}"] = {
                "gate": round(g, 4),
                "kept": round(sum(1 for p in pos if p >= g) / len(pos), 4)}

        out["models"][name] = {
            "matched_to_mediapipe_fpr": matched,
            "recall_at_fixed_fpr": sweep,
            "n_positive_frames": len(pos), "n_null_boxes": len(neg),
            "positive_mean": round(float(np.mean(pos)), 4),
            "positive_p05": round(float(np.percentile(pos, 5)), 4),
            "null_mean": round(float(np.mean(neg)), 4),
            "null_p95": round(float(np.percentile(neg, 95)), 4),
            "auc_person_vs_background": round(a, 4),
            "gate_from_null_p95": gate,
            "real_batsman_frames_kept_at_gate": round(kept, 4),
            "null_boxes_passing_gate": round(fp, 4),
        }

    # MediaPipe's single operating point, for reference.
    mp_pos = sum(1 for r in rows for fr in r["frames"]
                 if fr.get("mediapipe_heavy", {}).get("ok"))
    mp_tot = sum(1 for r in rows for fr in r["frames"] if "mediapipe_heavy" in fr)
    out["mediapipe_operating_point"] = {
        "real_batsman_frames_with_pose": round(mp_pos / mp_tot, 4) if mp_tot else None,
        "false_pose_rate_on_person_free_boxes":
            cal.get("mediapipe_false_pose_rate_on_null"),
        "note": ("MediaPipe declines on its own, so it has one operating point, "
                 "not a tunable curve."),
    }

    with open(os.path.join(P3, "phase3_pose_separability.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(f"{'model':<20}{'AUC':>8}{'pos mean':>10}{'null mean':>11}"
          f"{'gate':>8}{'kept@gate':>11}{'null pass':>11}")
    for k, v in out["models"].items():
        print(f"{k:<20}{v['auc_person_vs_background']:>8.3f}"
              f"{v['positive_mean']:>10.3f}{v['null_mean']:>11.3f}"
              f"{v['gate_from_null_p95']:>8.3f}"
              f"{v['real_batsman_frames_kept_at_gate']:>11.3f}"
              f"{v['null_boxes_passing_gate']:>11.3f}")
    m = out["mediapipe_operating_point"]
    print(f"\nMediaPipe: pose on {m['real_batsman_frames_with_pose']:.3f} of real "
          f"batsman frames, false pose on "
          f"{m['false_pose_rate_on_person_free_boxes']:.3f} of person-free boxes")

    print(f"\nMATCHED-FPR COMPARISON (all gated to MediaPipe's own "
          f"{m['false_pose_rate_on_person_free_boxes']:.3f} false-pose rate)")
    print(f"{'model':<20}{'gate':>8}{'batsman frames kept':>22}")
    print(f"{'mediapipe_heavy':<20}{'n/a':>8}"
          f"{m['real_batsman_frames_with_pose']:>22.3f}")
    for k, v in out["models"].items():
        mt = v.get("matched_to_mediapipe_fpr")
        if mt:
            print(f"{k:<20}{mt['gate']:>8.3f}{mt['real_batsman_frames_kept']:>22.3f}")

    print(f"\nRecall at fixed false-positive rates:")
    fprs = ["fpr_0.01", "fpr_0.02", "fpr_0.05", "fpr_0.10"]
    print(f"{'model':<20}" + "".join(f"{f:>12}" for f in fprs))
    for k, v in out["models"].items():
        print(f"{k:<20}" + "".join(
            f"{v['recall_at_fixed_fpr'][f]['kept']:>12.3f}" for f in fprs))
    print(f"\nwrote {os.path.join(P3, 'phase3_pose_separability.json')}")


if __name__ == "__main__":
    main()
