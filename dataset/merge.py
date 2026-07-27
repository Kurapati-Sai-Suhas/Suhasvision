import os
import sys

import pandas as pd

# Milestone 3 (Stance-Symmetry-Aware Temporal Confidence): MIN_TEMPORAL_CONFIDENCE
# is imported, not re-hardcoded here, so the gate threshold below and the
# threshold documented/tuned in stance_symmetry_confidence.py can never
# silently drift apart.
_dataset_dir = os.path.dirname(os.path.abspath(__file__))
_academic_scripts_dir = os.path.join(_dataset_dir, "academic_scripts")
if _academic_scripts_dir not in sys.path:
    sys.path.insert(0, _academic_scripts_dir)
from stance_symmetry_confidence import MIN_TEMPORAL_CONFIDENCE  # noqa: E402


def main():
    print("Loading keypoints_validated.csv and labels.csv...")
    try:
        kp = pd.read_csv("keypoints_validated.csv")
        lb = pd.read_csv("labels.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Drop duplicates in labels to prevent cartesian explosion
    lb = lb.drop_duplicates(subset=["session_name"], keep="last")

    # Confirmed real bug (2026-07-18, fixed in nvidia_client.py): a missing
    # score used to silently default to 50, so a session where the AI
    # labelling response didn't include a usable "scores" object still got
    # a full label row with all four scores exactly 50 -- indistinguishable
    # from a genuine assessment unless checked directly. Verified against
    # real labels.csv: 90 of 496 rows had all four scores == 50 (the exact
    # fallback signature; a 5th score-identical row was 60/60/60/60, a
    # different value, not excluded here since it doesn't match the
    # confirmed fallback and could be a genuine assessment). New labelling
    # calls can no longer produce this row shape (nvidia_client.py rejects
    # incomplete scores instead of guessing), but these are already in
    # labels.csv from before that fix -- not deleted from the file (keeping
    # the raw label history intact), just excluded here, the same
    # choke-point pattern as the kinematic_valid/temporal_confidence
    # filters below.
    score_cols = ["score_balance", "score_power", "score_technique", "score_defence"]
    if all(c in lb.columns for c in score_cols):
        before = len(lb)
        is_fallback_default = (lb[score_cols] == 50).all(axis=1)
        lb = lb[~is_fallback_default]
        print(f"fallback-default-score filter: {before} -> {len(lb)} label rows (dropped {is_fallback_default.sum()} rows where all 4 scores were exactly 50, the confirmed AI-labelling-failure signature)")

    # kinematic_validator.py flags but never deletes frames -- until now,
    # nothing downstream actually excluded flagged frames (the only script
    # that did, train_lstm_model.py, was superseded). A session with any
    # invalid frame among its fixed 7-frame sequence is an incomplete/corrupt
    # sample, so drop invalid frames here, the single choke point everything
    # else (dataset.csv, dataset_angles.csv, training) passes through.
    if "kinematic_valid" in kp.columns:
        before = kp["session_name"].nunique()
        kp = kp[kp["kinematic_valid"] == True]
        after = kp["session_name"].nunique()
        print(f"kinematic_valid filter: {before} -> {after} sessions with at least one valid frame remaining")

    # Milestone 3 (Stance-Symmetry-Aware Temporal Confidence): the second,
    # complementary mandatory filter (frozen V2 spec, Section 6.2) alongside
    # kinematic_valid above -- same choke-point pattern, dropping frames
    # below MIN_TEMPORAL_CONFIDENCE. Unlike kinematic_valid, the continuous
    # temporal_confidence column is NOT dropped after gating -- it is kept
    # in dataset.csv on purpose, since Milestone 5 (Confidence-Gated
    # Attention, not yet implemented) will consume the continuous score at
    # inference time, not just this pass/fail cut.
    if "temporal_confidence" in kp.columns:
        before = kp["session_name"].nunique()
        kp = kp[kp["temporal_confidence"] >= MIN_TEMPORAL_CONFIDENCE]
        after = kp["session_name"].nunique()
        print(f"temporal_confidence filter (>= {MIN_TEMPORAL_CONFIDENCE}): {before} -> {after} sessions with at least one valid frame remaining")

    # Merge on session_name
    merged = kp.merge(lb, on="session_name", how="inner")
    
    # Save dataset.csv
    merged.to_csv("dataset.csv", index=False)
    
    print(f"dataset.csv successfully created!")
    print(f"Total rows: {len(merged)}")
    print(f"Unique sessions mapped: {merged['session_name'].nunique()}")

if __name__ == "__main__":
    main()
