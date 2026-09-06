"""
phase3_pose_impact.py — how many pipeline failures actually disappear?

The counterfactual is asked against the pipeline's OWN failure definition:

    phase3_batsman_analysis.categorise_failure() calls a clip a
    "pose failure on the batsman crop" when k_pose_rate < 0.4,
    k_pose_rate = (sampled frames with a pose) / (sampled frames)

and it is computed on the pipeline's own uniform 20-frame sampling, taken
from phase3_pose_diagnosis.json. That matters: the pose benchmark samples
ALONG the batsman track, which makes coverage 1.0 by construction and would
quietly answer a different question.

TWO COUNTERFACTUALS, BECAUSE ONE ALONE MISLEADS

  ungated  RTMPose returns a pose for every box, so its k_pose_rate equals
           the track's coverage. This is the right number FOR THIS PIPELINE,
           where every box already comes from a YOLO+ByteTrack person
           detection and nothing is asking the pose model to decide whether a
           person is present.

  gated    RTMPose's own confidence, thresholded at MediaPipe's false-pose
           rate on person-free boxes. This is what would happen if RTMPose
           had to self-verify, and it shows that it cannot.

Reporting only the first would overstate RTMPose; reporting only the second
would understate it by charging it for a job the pipeline never gives it.
"""

import json
import os

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.join(_MODULE_DIR, "phase3_results")

POSE_RATE_THRESHOLD = 0.40      # phase3_batsman_analysis.categorise_failure


def main():
    with open(os.path.join(P3, "phase3_pose_diagnosis.json"), encoding="utf-8") as f:
        diag = json.load(f)
    rows = diag["clips"]
    T = POSE_RATE_THRESHOLD

    def bucket(r):
        mp = r["k_pose_rate_mediapipe"]
        un = r["k_pose_rate_rtmpose_ungated"]
        ga = r["k_pose_rate_rtmpose_gated"]
        return mp, un, ga

    fixed_un, remain_un, fixed_ga = [], [], []
    for r in rows:
        mp, un, ga = bucket(r)
        if mp < T:
            (fixed_un if un >= T else remain_un).append(r)
            if ga >= T:
                fixed_ga.append(r)

    below = [r for r in rows if r["k_pose_rate_mediapipe"] < T]
    labelled = [r for r in rows if r["is_labelled_pose_failure"]]

    out = {
        "threshold": T,
        "basis": ("phase3_pose_diagnosis.json — pipeline's own uniform "
                  "20-frame sampling, annotated batsman track"),
        "n_clips": len(rows),
        "n_below_threshold_under_mediapipe": len(below),
        "ungated": {
            "n_fixed": len(fixed_un),
            "fixed": [r["clip_id"] for r in fixed_un],
            "n_still_failing": len(remain_un),
            "still_failing": [r["clip_id"] for r in remain_un],
            "why_still_failing": ("RTMPose's k_pose_rate equals the track's "
                                  "coverage, so a clip whose batsman track "
                                  "covers under 40% of sampled frames stays "
                                  "below threshold no matter the pose model. "
                                  "These are tracking failures."),
        },
        "gated": {
            "n_fixed": len(fixed_ga),
            "fixed": [r["clip_id"] for r in fixed_ga],
            "meaning": ("Using RTMPose's own confidence as a presence check "
                        "at MediaPipe's false-pose rate. See "
                        "phase3_pose_separability.json."),
        },
        "labelled_pose_failures": {
            "n": len(labelled),
            "genuinely_pose_limited": [
                r["clip_id"] for r in labelled
                if (r["pose_given_box_mediapipe"] or 0) < T],
            "coverage_limited_only": [
                r["clip_id"] for r in labelled
                if r["coverage_on_sampled"] < T],
            "not_failing_by_this_measure": [
                r["clip_id"] for r in labelled
                if r["k_pose_rate_mediapipe"] >= T],
        },
        "clips": rows,
    }
    with open(os.path.join(P3, "phase3_pose_impact.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(f"clips: {len(rows)}   below k_pose_rate {T} under MediaPipe: {len(below)}")
    print(f"\n{'clip':<30}{'cover':>8}{'mp':>8}{'rtm un':>9}{'rtm gated':>11}  verdict")
    for r in sorted(below, key=lambda r: r["k_pose_rate_mediapipe"]):
        mp, un, ga = bucket(r)
        v = "FIXED (ungated)" if un >= T else "still failing — coverage"
        print(f"{r['clip_id']:<30}{r['coverage_on_sampled']:>8.2f}{mp:>8.2f}"
              f"{un:>9.2f}{ga:>11.2f}  {v}")

    print(f"\nungated : fixes {len(fixed_un)} of {len(below)}; "
          f"{len(remain_un)} remain (tracking, not pose)")
    print(f"gated   : fixes {len(fixed_ga)} of {len(below)} "
          f"— RTMPose cannot self-verify at MediaPipe's operating point")
    L = out["labelled_pose_failures"]
    print(f"\nOf the {L['n']} clips labelled 'pose failure':")
    print(f"  genuinely pose-limited : {L['genuinely_pose_limited']}")
    print(f"  coverage-limited only  : {L['coverage_limited_only']}")
    print(f"  not failing at all     : {L['not_failing_by_this_measure']}")
    print(f"\nwrote {os.path.join(P3, 'phase3_pose_impact.json')}")


if __name__ == "__main__":
    main()
