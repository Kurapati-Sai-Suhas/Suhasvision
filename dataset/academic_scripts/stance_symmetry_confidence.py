"""
stance_symmetry_confidence.py
Stance-Symmetry-Aware Temporal Confidence (Milestone 3).

WHY THIS EXISTS
kinematic_validator.py's bone-length CV filter catches one failure mode: a
joint whose estimated bone length drifts across a session's frames --
evidence of a bad per-frame pose estimate, since a real person's bone
lengths don't change. It does NOT catch a second, different failure mode:
a joint whose bone length stays self-consistent but whose FRAME-TO-FRAME
TRAJECTORY is implausible (a position jump no real body produces in that
time), and it uses no signal specific to the batting stance itself.

Per the frozen V2 architecture spec (Two complementary, both-mandatory
filters gate every session before it reaches training data), this module
adds that second signal:

  1. Trajectory smoothness -- for the same 12 joints kinematic_validator.py
     already tracks (the endpoints of its BONES dict), how implausible is
     each frame's jump relative to that joint's OWN overall motion in the
     session. A joint that's supposed to move a lot during the swing isn't
     penalized on an absolute scale -- only a jump disproportionate to its
     own path is.
  2. Stance-phase symmetry -- at the stance/trigger phases only (the static
     setup, before the swing becomes intentionally asymmetric), how close
     bilateral hip/knee/ankle angles are to the range empirically observed
     on real, already-kinematically-valid stance/trigger frames. These are
     the same three joints zero_storage_pipeline.py's own validate_pose
     already treats as "high value" for phase_index in [0, 1].

Unlike kinematic_valid, this produces a CONTINUOUS per-frame
`temporal_confidence` score in (0, 1], not a boolean -- kept as a real
column (not thresholded away) because Milestone 5 (Confidence-Gated
Attention, not yet implemented) will consume the continuous score at
inference time, not just a pass/fail flag. merge.py still uses it as a
second mandatory gate (frames below MIN_TEMPORAL_CONFIDENCE are dropped),
matching how it already enforces kinematic_valid.

The angle math (calculate_angle) is imported from feature_engineering.py,
not reimplemented -- the joint-triple definitions here mirror the ones
already written there for angle_knee_{L,R}, angle_hip_{L,R}, and
angle_ankle_{L,R}.

USAGE
    python stance_symmetry_confidence.py --input keypoints_validated.csv \
        --output keypoints_validated.csv --report symmetry_confidence_report.csv

Run this AFTER kinematic_validator.py, on its output -- both filters read
and write the same keypoints_validated.csv, each contributing their own
column (kinematic_valid, temporal_confidence).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_academic_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_dataset_dir = os.path.dirname(_academic_scripts_dir)
for _p in (_dataset_dir, _academic_scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from feature_engineering import calculate_angle  # noqa: E402
from kinematic_validator import BONES as _BONES  # noqa: E402

# ---------------------------------------------------------------------------
# The joints kinematic_validator.py's BONES dict already tracks (its unique
# endpoints), IMPORTED rather than re-typed -- a hardcoded copy would
# silently drift out of sync if BONES ever gains or loses a joint. sorted()
# only for deterministic iteration order; membership is what matters.
# ---------------------------------------------------------------------------
TRAJECTORY_JOINTS = sorted({joint for pair in _BONES.values() for joint in pair})

# Bilateral angle triples for the stance-symmetry prior, matching
# feature_engineering.py's angle_knee_{L,R}/angle_hip_{L,R}/angle_ankle_{L,R}
# definitions exactly -- these are the same three joints
# zero_storage_pipeline.py's validate_pose() already treats as "high value"
# for phase_index in [0, 1] (hips, knees, ankles).
STANCE_SYMMETRY_ANGLES = {
    "knee": (
        ("left_hip", "left_knee", "left_ankle"),
        ("right_hip", "right_knee", "right_ankle"),
    ),
    "hip": (
        ("left_shoulder", "left_hip", "left_knee"),
        ("right_shoulder", "right_hip", "right_knee"),
    ),
    "ankle": (
        ("left_knee", "left_ankle", "left_foot_index"),
        ("right_knee", "right_ankle", "right_foot_index"),
    ),
}

# Frame positions (0-indexed, after sorting by frame_name) corresponding to
# "01_stance" and "02_trigger" -- the static setup phases, before the swing
# becomes intentionally asymmetric. Matches zero_storage_pipeline.py's own
# `phase_index in [0, 1]` special-casing.
STANCE_PHASE_INDICES = {0, 1}

# (mean_diff, std_diff) in degrees for (left_angle - right_angle) at
# stance/trigger phases, fit empirically against the real, already
# kinematically-valid dataset (keypoints_validated.csv, kinematic_valid ==
# True, frame_name in {frame_01_stance, frame_02_trigger}; n=613 frames
# across 798 sessions as of 2026-07-18). Tune these against a larger dataset
# as more sessions are collected -- a reasonable, real-data-grounded
# starting point, not a permanently fixed constant.
STANCE_SYMMETRY_REFERENCE = {
    "knee": (0.1357, 20.1016),
    "hip": (-0.9096, 24.1464),
    "ankle": (-1.7846, 18.7112),
}

# Below this combined score, a frame is dropped by merge.py's gate (see
# merge.py). Fit against the real distribution of combined scores on the
# same dataset above: this flags roughly the bottom 2.5% of real frames as
# an outlier catcher, not a bulk rejector -- a reasonable starting point,
# not a validated constant. Tune alongside MAX_BONE_CV /
# MAX_SYMMETRY_RATIO_DEV in kinematic_validator.py once more data exists.
MIN_TEMPORAL_CONFIDENCE = 0.5


def _xyz(df, name):
    return df[[f"{name}_x", f"{name}_y", f"{name}_z"]].values.astype(float)


def compute_trajectory_smoothness(session_df):
    """
    Returns a numpy array of per-frame trajectory-smoothness scores in
    (0, 1], aligned to session_df's row order (caller's responsibility to
    have already sorted by frame_name).

    For each tracked joint: the discrete second difference (jerk) at
    interior frames, normalized by that joint's own total path length
    across the session. Edge frames have no defined second difference and
    inherit their nearest interior frame's score. Sessions with fewer than
    3 frames have no interior frame to compute a jerk from at all -- jerk
    stays 0 (no evidence of a problem, not a crash) rather than guessing.
    """
    n = len(session_df)
    if n == 0:
        return np.array([])
    per_joint_scores = []
    for joint in TRAJECTORY_JOINTS:
        pos = _xyz(session_df, joint)
        path_length = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))) if n > 1 else 0.0
        jerks = np.zeros(n)
        for i in range(1, n - 1):
            accel = pos[i - 1] - 2 * pos[i] + pos[i + 1]
            jerks[i] = np.linalg.norm(accel)
        if n > 2:
            jerks[0] = jerks[1]
            jerks[-1] = jerks[-2]
        normalized_jerk = jerks / (path_length + 1e-6)
        per_joint_scores.append(np.exp(-normalized_jerk))
    return np.mean(per_joint_scores, axis=0)


def compute_stance_symmetry(session_df):
    """
    Returns a numpy array of per-frame stance-symmetry scores, aligned to
    session_df's row order. Only frames at STANCE_PHASE_INDICES get a real
    score (mean of exp(-0.5 * z^2) across the 3 bilateral angle pairs,
    z-scored against STANCE_SYMMETRY_REFERENCE). Every other frame is NaN
    -- deliberately, not a neutral placeholder, so the caller can average
    only the components that actually apply to a given frame rather than
    diluting the trajectory signal with a meaningless constant for phases
    where bilateral symmetry isn't a real prior (the swing itself).
    """
    n = len(session_df)
    scores = np.full(n, np.nan)
    for i in range(n):
        if i not in STANCE_PHASE_INDICES:
            continue
        row = session_df.iloc[[i]]
        pair_scores = []
        for pair_name, (left_triple, right_triple) in STANCE_SYMMETRY_ANGLES.items():
            angle_l = calculate_angle(row, *left_triple)[0]
            angle_r = calculate_angle(row, *right_triple)[0]
            diff = angle_l - angle_r
            mean_diff, std_diff = STANCE_SYMMETRY_REFERENCE[pair_name]
            z = (diff - mean_diff) / (std_diff + 1e-6)
            pair_scores.append(np.exp(-0.5 * z * z))
        scores[i] = float(np.mean(pair_scores))
    return scores


def compute_session_confidence(session_df):
    """
    Returns (confidence: pd.Series indexed like session_df, report: dict).

    Sorts by frame_name internally (jerk and phase-symmetry both require
    temporal order) but returns the confidence Series aligned to
    session_df's ORIGINAL index, so callers can assign it back with
    `df.loc[session_df.index, "temporal_confidence"] = confidence` the same
    way kinematic_validator.py assigns `kinematic_valid`.

    A real, currently-occurring data issue (verified against the live
    dataset: 42 sessions today) is a session with every frame_name
    duplicated -- e.g. two "01_stance" rows before the first "02_trigger"
    row. Sorting such a session and treating row position as phase index
    would silently score a duplicate stance frame as if it were the
    trigger frame, corrupting both the symmetry prior (wrong phase
    compared) and the jerk calculation (an artificial zero-then-jump
    pattern from the duplicate pairs, not real motion). Rather than guess
    at de-duplication here, sessions with a duplicate frame_name get NaN
    confidence for every frame -- merge.py's `>= MIN_TEMPORAL_CONFIDENCE`
    gate then drops them the same way it would drop any low-confidence
    frame, instead of silently reporting a number that doesn't mean what
    it claims to.
    """
    ordered = session_df.sort_values("frame_name")
    if ordered["frame_name"].duplicated().any():
        n = len(ordered)
        confidence = pd.Series(np.full(n, np.nan), index=ordered.index)
        report = {
            "mean_trajectory_score": float("nan"), "mean_symmetry_score": float("nan"),
            "mean_temporal_confidence": float("nan"), "min_temporal_confidence": float("nan"),
            # 0, not n: these frames were never scored at all (NaN), which
            # is a different thing from being scored and found below
            # threshold. duplicate_frame_names=True is what actually marks
            # them. Keeping this at n would double-count them if anyone
            # sums n_below_threshold across report_df -- it wouldn't match
            # main()'s own top-level "below" count, which already treats
            # NaN and below-threshold as separate buckets.
            "n_below_threshold": 0, "duplicate_frame_names": True,
        }
        return confidence, report

    trajectory_scores = compute_trajectory_smoothness(ordered)
    symmetry_scores = compute_stance_symmetry(ordered)
    combined = np.where(np.isnan(symmetry_scores), trajectory_scores, (trajectory_scores + symmetry_scores) / 2.0)
    confidence = pd.Series(combined, index=ordered.index)

    valid_symmetry = symmetry_scores[~np.isnan(symmetry_scores)]
    report = {
        "mean_trajectory_score": float(np.mean(trajectory_scores)) if len(trajectory_scores) else float("nan"),
        "mean_symmetry_score": float(np.mean(valid_symmetry)) if len(valid_symmetry) else float("nan"),
        "mean_temporal_confidence": float(np.mean(combined)) if len(combined) else float("nan"),
        "min_temporal_confidence": float(np.min(combined)) if len(combined) else float("nan"),
        "n_below_threshold": int(np.sum(combined < MIN_TEMPORAL_CONFIDENCE)),
        "duplicate_frame_names": False,
    }
    return confidence, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="keypoints_validated.csv")
    ap.add_argument("--output", default="keypoints_validated.csv")
    ap.add_argument("--report", default="symmetry_confidence_report.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["temporal_confidence"] = np.nan
    reports = []

    for session_name, session_df in df.groupby("session_name"):
        confidence, report = compute_session_confidence(session_df)
        df.loc[confidence.index, "temporal_confidence"] = confidence.values
        report["session_name"] = session_name
        report["n_frames"] = len(session_df)
        reports.append(report)

    report_df = pd.DataFrame(reports).set_index("session_name")
    report_df.to_csv(args.report)
    df.to_csv(args.output, index=False)

    total = len(df)
    # NaN confidence (duplicate-frame_name sessions, see compute_session_
    # confidence) fails both "< threshold" and ">= threshold" -- count it
    # separately so it isn't silently missing from either bucket. merge.py's
    # `>= MIN_TEMPORAL_CONFIDENCE` gate drops both groups the same way.
    nan_count = int(df["temporal_confidence"].isna().sum())
    below = int((df["temporal_confidence"] < MIN_TEMPORAL_CONFIDENCE).sum())
    dropped = below + nan_count
    print(f"Scored {total} frames across {df['session_name'].nunique()} sessions.")
    print(f"{below} frames ({below/total:.1%}) fall below MIN_TEMPORAL_CONFIDENCE ({MIN_TEMPORAL_CONFIDENCE}).")
    if nan_count:
        print(f"{nan_count} additional frames ({nan_count/total:.1%}) scored NaN (duplicate frame_name within "
              f"the session -- phase order can't be trusted) and will also be dropped by merge.py's gate.")
    print(f"Total frames merge.py's gate will drop: {dropped} ({dropped/total:.1%}).")
    print(f"Per-session confidence report saved to {args.report}")
    print(f"Full annotated dataset saved to {args.output}")


if __name__ == "__main__":
    main()
