"""
phase3_failure_table.py — the corrected corpus diagnostic.

Replaces the old seven-clip "pose failure" list, which was produced by a
statistic that could not tell a missing track from a failed pose. The table
here reports the two factors separately for the ANNOTATED batsman track of
every clip:

    coverage                 fraction of sampled frames where the track exists
    pose_success_given_box   of those, fraction where the pose model succeeded
    effective_pose_rate      their product — the old `k_pose_rate`
    failure_type             TRACKING / POSE / JOINT / NO_FAILURE

Reported on the annotated batsman track because the question is "did the
pipeline fail on this clip", and the batsman is the track it needed.

The old list is loaded and diffed against the corrected verdicts so the
mislabelling is explicit rather than quietly replaced.
"""

import argparse
import json
import os

import phase3_failure_taxonomy as FT

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

# The list this table supersedes: the union of clips ever labelled
# "pose failure on the batsman crop" across S0-S4 by the old, conflated test.
OLD_POSE_FAILURE_LIST = (
    "kohli_side_12.mp4", "kohli_side_16.mp4", "pro_player_front_06.mp4",
    "pro_player_front_10.mp4", "pro_player_front_18.mp4", "rishi_front_2.mp4",
    "sanjay_front _1.mp4",
)


def build(features_path, tracks_path):
    from phase2_cache_tracks import load_tracks
    from phase2_detector_benchmark import load_gt
    from phase2_fit_batsman_model import gt_track_for_clip

    with open(features_path, encoding="utf-8") as f:
        feats = json.load(f)
    tracks = load_tracks(tracks_path)
    gt = {c["clip_id"]: c for c in load_gt()["clips"]}

    rows = []
    for cid, fc in feats["clips"].items():
        tc = tracks["clips"].get(cid)
        clip_gt = gt.get(cid)
        if not tc or not clip_gt:
            continue
        gt_frames = [f for f in clip_gt["frames"]
                     if f["batsman_visible"] and f["batsman_bbox"]]
        gt_tid, _ = gt_track_for_clip(tc["tracks"], gt_frames)
        if gt_tid is None:
            rows.append({"clip_id": cid, "split": fc.get("split"),
                         "gt_track": None, "coverage": None,
                         "pose_success_given_box": None,
                         "effective_pose_rate": None,
                         "failure_type": "NO_TRACK_MATCHES_BATSMAN",
                         "was_labelled_pose_failure": cid in OLD_POSE_FAILURE_LIST})
            continue
        t = fc["tracks"].get(str(gt_tid)) or {}
        d = t.get("diagnostic")
        if not d:
            continue
        rows.append({
            "clip_id": cid, "split": fc.get("split"), "gt_track": gt_tid,
            "n_sampled": d["n_sampled"], "n_box": d["n_box"], "n_pose": d["n_pose"],
            "coverage": d["coverage"],
            "pose_success_given_box": d["pose_success_given_box"],
            "effective_pose_rate": d["effective_pose_rate"],
            "failure_type": d["failure_type"],
            "was_labelled_pose_failure": cid in OLD_POSE_FAILURE_LIST,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(P3, "phase3_semantic_features.json"))
    ap.add_argument("--tracks", default=os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--out", default=os.path.join(P3, "phase3_failure_table.json"))
    ap.add_argument("--label", default="H0_mediapipe_only")
    args = ap.parse_args()

    rows = build(args.features, args.tracks)
    by_type = {}
    for r in rows:
        by_type.setdefault(r["failure_type"], []).append(r["clip_id"])

    old = set(OLD_POSE_FAILURE_LIST)
    corrected_pose = {r["clip_id"] for r in rows
                      if r["failure_type"] == FT.POSE_FAILURE}
    mislabelled = sorted(old - corrected_pose)
    newly_found = sorted(corrected_pose - old)

    out = {
        "condition": args.label,
        "n_clips": len(rows),
        "thresholds": {"coverage": FT.COVERAGE_THRESHOLD,
                       "pose_success_given_box": FT.POSE_THRESHOLD},
        "semantics": FT.SEMANTICS,
        "counts": {k: len(v) for k, v in sorted(by_type.items())},
        "by_type": {k: sorted(v) for k, v in sorted(by_type.items())},
        "supersedes": {
            "old_list": sorted(old),
            "old_list_definition": ("clips labelled 'pose failure on the "
                                    "batsman crop' by the conflated "
                                    "k_pose_rate < 0.4 test"),
            "mislabelled_by_old_test": mislabelled,
            "genuine_pose_failures_the_old_test_missed": newly_found,
        },
        "clips": sorted(rows, key=lambda r: (r["failure_type"],
                                             r["effective_pose_rate"]
                                             if r["effective_pose_rate"] is not None
                                             else -1)),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(f"condition: {args.label}   clips: {len(rows)}")
    print(f"\n{'clip':<30}{'cover':>7}{'pose|box':>10}{'effective':>11}"
          f"  {'failure_type':<26}old?")
    for r in out["clips"]:
        if r["failure_type"] == FT.NO_FAILURE:
            continue
        pb = r["pose_success_given_box"]
        print(f"{r['clip_id']:<30}"
              f"{(r['coverage'] if r['coverage'] is not None else 0):>7.2f}"
              f"{(f'{pb:.2f}' if pb is not None else '-'):>10}"
              f"{(r['effective_pose_rate'] if r['effective_pose_rate'] is not None else 0):>11.2f}"
              f"  {r['failure_type']:<26}"
              f"{'yes' if r['was_labelled_pose_failure'] else ''}")
    print("\ncounts:", out["counts"])
    print(f"\nold list mislabelled: {mislabelled}")
    print(f"genuine pose failures the old test missed: {newly_found}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
