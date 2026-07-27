"""
ablation_stance_symmetry_filter.py
The required ablation for Milestone 3 (Stance-Symmetry-Aware Temporal
Confidence): the roadmap's own stated completion criterion is that this
milestone "is not considered complete until [a] comparison exists" between
dataset yield and downstream MAE with the new filter on vs. off, under the
Milestone 2 unified evaluation protocol, with a significance test.

This is a NEW, separate script rather than an edit to ablation_study.py
(which ablates ARCHITECTURE choices, a different question) -- per the
milestone rule not to modify unrelated modules.

WHAT IT COMPARES
  Variant A ("filter off"): keypoints_validated.csv gated by kinematic_valid
  only -- the pre-Milestone-3 behavior.
  Variant B ("filter on"): additionally gated by temporal_confidence >=
  MIN_TEMPORAL_CONFIDENCE -- the new, current merge.py behavior.

Both variants go through the identical downstream path: merge with labels,
compute the same 15 joint angles + velocities (reusing feature_engineering
.py's calculate_angle/calculate_angle_with_vertical, not reimplementing
them), build 7-frame sequences, and train the SAME architecture
(Conv1D + BiLSTM + TemporalAttention, matching the production model and
ablation_study.py's winning architecture).

FOLD METHODOLOGY (important, fixed in review): folds are built ONCE from
variant A's (the larger, unfiltered) identity set via
evaluation_protocol.make_folds, and the SAME held-out identities are then
used to evaluate variant B. Calling make_folds independently for each
variant -- the original implementation -- produces two UNRELATED sets of
held-out identities per "fold index" (verified against the real dataset:
fold 1 of each variant held out almost entirely different people), which
would make paired_significance_test's pairing meaningless even though it
would still silently return a p-value. A fold whose held-out identities
have zero surviving sessions in the filtered variant is dropped from both
sides rather than guessed at.

USAGE
    python ablation_stance_symmetry_filter.py --keypoints keypoints_validated.csv \
        --labels labels.csv --n_folds 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

_academic_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_dataset_dir = os.path.dirname(_academic_scripts_dir)
for _p in (_dataset_dir, _academic_scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from schema import SEQ_LEN, FEATURE_BASE_NAMES, CANONICAL_FRAME_NAMES, EXPECTED_FEATURES as FEATURE_COLS  # noqa: E402
from model_layers import TemporalAttention  # noqa: E402
from feature_engineering import calculate_angle, calculate_angle_with_vertical, normalize_frame_name  # noqa: E402
from stance_symmetry_confidence import MIN_TEMPORAL_CONFIDENCE  # noqa: E402
from evaluation_protocol import (  # noqa: E402
    extract_batsman_name,
    make_folds,
    assert_no_leakage,
    compute_metrics,
    paired_significance_test,
    SCORE_NAMES,
)

SCORE_COLS = ["score_balance", "score_power", "score_technique", "score_defence"]

# The same 7 canonical phase-ordered frame names feature_engineering.py
# pads sessions to -- required so both variants build sequences the same
# way regardless of how many raw frames a session happened to keep. Since
# Milestone 3 (audit H6) the list itself lives once, in schema.py -- an
# independent stale copy here is exactly how the reindex bug that wiped
# metadata for one naming convention was duplicated into this file before.
# list(), not the tuple: pandas reindex below gets a list indexer.
CANONICAL_FRAMES = list(CANONICAL_FRAME_NAMES)


def build_angle_features(merged_df):
    """
    Computes the same 15 joint angles + 15 angular velocities as
    feature_engineering.py's main(), on an already keypoints+labels-merged
    dataframe. Duplicated orchestration (not duplicated math -- calculate_
    angle/calculate_angle_with_vertical are imported, not reimplemented)
    because feature_engineering.py's main() reads/writes fixed CSV
    filenames and isn't itself an importable function; extracting one
    would mean editing that file, out of scope for this milestone.
    """
    df = merged_df.copy()
    angles_df = pd.DataFrame()
    meta_cols = ["session_name", "frame_name"] + SCORE_COLS
    for col in meta_cols:
        if col in df.columns:
            angles_df[col] = df[col]

    angles_df["angle_knee_L"] = calculate_angle(df, "left_hip", "left_knee", "left_ankle")
    angles_df["angle_knee_R"] = calculate_angle(df, "right_hip", "right_knee", "right_ankle")
    angles_df["angle_hip_L"] = calculate_angle(df, "left_shoulder", "left_hip", "left_knee")
    angles_df["angle_hip_R"] = calculate_angle(df, "right_shoulder", "right_hip", "right_knee")
    angles_df["angle_elbow_L"] = calculate_angle(df, "left_shoulder", "left_elbow", "left_wrist")
    angles_df["angle_elbow_R"] = calculate_angle(df, "right_shoulder", "right_elbow", "right_wrist")
    angles_df["angle_shoulder_L"] = calculate_angle(df, "left_hip", "left_shoulder", "left_elbow")
    angles_df["angle_shoulder_R"] = calculate_angle(df, "right_hip", "right_shoulder", "right_elbow")
    angles_df["angle_ankle_L"] = calculate_angle(df, "left_knee", "left_ankle", "left_foot_index")
    angles_df["angle_ankle_R"] = calculate_angle(df, "right_knee", "right_ankle", "right_foot_index")
    angles_df["angle_trunk_L"] = calculate_angle_with_vertical(df, "left_hip", "left_shoulder")
    angles_df["angle_trunk_R"] = calculate_angle_with_vertical(df, "right_hip", "right_shoulder")
    angles_df["angle_arm_L"] = calculate_angle_with_vertical(df, "left_shoulder", "left_elbow")
    angles_df["angle_arm_R"] = calculate_angle_with_vertical(df, "right_shoulder", "right_elbow")
    df["mid_shoulder_x"] = (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2
    df["mid_shoulder_y"] = (df["left_shoulder_y"] + df["right_shoulder_y"]) / 2
    df["mid_shoulder_z"] = (df["left_shoulder_z"] + df["right_shoulder_z"]) / 2
    angles_df["angle_head_tilt"] = calculate_angle_with_vertical(df, "mid_shoulder", "nose")

    angle_cols = list(FEATURE_BASE_NAMES)
    angles_df[angle_cols] = angles_df[angle_cols] / 180.0

    def pad_group(group):
        session_id = group.name
        group = group.copy()
        group["frame_name"] = group["frame_name"].apply(normalize_frame_name)
        group = group.drop_duplicates(subset=["frame_name"])
        group = group.set_index("frame_name").reindex(CANONICAL_FRAMES)
        group = group.ffill().bfill().reset_index()
        group["session_name"] = session_id
        return group

    angles_df = angles_df.groupby("session_name", group_keys=False).apply(pad_group).reset_index(drop=True)
    angles_df = angles_df.sort_values(["session_name", "frame_name"])
    for col in angle_cols:
        angles_df[col + "_vel"] = angles_df.groupby("session_name")[col].diff().fillna(0)
    return angles_df


def build_sequences(df):
    session_ids, X, y = [], [], []
    for session_name, group in df.groupby("session_name"):
        if len(group) != SEQ_LEN or group[SCORE_COLS].isna().any().any():
            continue
        group = group.sort_values("frame_name")
        X.append(group[FEATURE_COLS].values.astype(np.float32))
        y.append(group.iloc[0][SCORE_COLS].values.astype(np.float32) / 100.0)
        session_ids.append(session_name)
    return np.array(X), np.array(y), np.array(session_ids)


def build_model(seq_len, n_features):
    inputs = layers.Input(shape=(seq_len, n_features))
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = TemporalAttention()(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def load_variant(kp_variant, labels_df):
    merged = kp_variant.merge(labels_df, on="session_name", how="inner")
    angles_df = build_angle_features(merged)
    X, y, session_ids = build_sequences(angles_df)
    groups = np.array([extract_batsman_name(sid) for sid in session_ids])
    return X, y, groups


def derive_paired_split(groups_off, idx_val_off, groups_on):
    """
    Given one fold's held-out identities (from groups_off's own split),
    derives the matching train/val index split into groups_on -- the same
    held-out PEOPLE, scored from the (possibly smaller) filtered variant's
    own sessions. Returns (held_out_identities, idx_train_on, idx_val_on),
    or (held_out_identities, None, None) if the filtered variant has no
    sessions for the held-out identities, or no sessions left to train on
    (fold can't be paired -- caller should drop it from both sides).

    Pure index arithmetic, no model training -- kept separate from
    train_and_eval specifically so this can be unit-tested without
    TensorFlow. This is the exact logic a real bug lived in: comparing
    fold i of two INDEPENDENTLY computed splits instead of deriving both
    from the same held-out identities (see module docstring).
    """
    held_out_identities = set(groups_off[idx_val_off])
    idx_val_on = np.where(np.isin(groups_on, list(held_out_identities)))[0]
    idx_train_on = np.where(~np.isin(groups_on, list(held_out_identities)))[0]
    if len(idx_val_on) == 0 or len(idx_train_on) == 0:
        return held_out_identities, None, None
    return held_out_identities, idx_train_on, idx_val_on


def train_and_eval(X, y, idx_train, idx_val, epochs):
    model = build_model(SEQ_LEN, len(FEATURE_COLS))
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
    model.fit(X[idx_train], y[idx_train], validation_data=(X[idx_val], y[idx_val]),
              epochs=epochs, batch_size=16, callbacks=callbacks, verbose=0)
    preds = model.predict(X[idx_val], verbose=0) * 100.0
    metrics = compute_metrics(y[idx_val] * 100.0, preds, score_names=SCORE_NAMES)
    return float(np.mean([metrics[s]["mae"] for s in SCORE_NAMES]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoints", default="keypoints_validated.csv")
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    kp = pd.read_csv(args.keypoints)
    labels_df = pd.read_csv(args.labels).drop_duplicates(subset=["session_name"], keep="last")

    if "kinematic_valid" not in kp.columns:
        raise SystemExit("keypoints file has no kinematic_valid column -- run kinematic_validator.py first.")
    if "temporal_confidence" not in kp.columns:
        raise SystemExit("keypoints file has no temporal_confidence column -- run stance_symmetry_confidence.py first.")

    kp_valid = kp[kp["kinematic_valid"] == True]
    variant_off = kp_valid
    variant_on = kp_valid[kp_valid["temporal_confidence"] >= MIN_TEMPORAL_CONFIDENCE]

    X_off, y_off, groups_off = load_variant(variant_off, labels_df)
    X_on, y_on, groups_on = load_variant(variant_on, labels_df)
    n_groups_off = len(set(groups_off))
    print(f"Yield without new filter: {len(X_off)} complete 7-frame sessions, {n_groups_off} unique identities")
    print(f"Yield with new filter:    {len(X_on)} complete 7-frame sessions, {len(set(groups_on))} unique identities")

    if n_groups_off < 2:
        raise SystemExit("Too few identities in the unfiltered variant for cross-validation.")

    # Folds are built ONCE, on the (larger) filter-OFF identity set, and the
    # SAME held-out identities are then used to evaluate filter-ON. This is
    # required for the significance test below to be a genuine PAIRED
    # comparison: two independent make_folds() calls (one per variant) would
    # each pick their own arbitrary held-out identities -- verified directly
    # against the real dataset that "fold 1" of each would then hold out
    # almost entirely different people, making a paired test meaningless
    # even though it would still silently produce a p-value.
    n_folds = min(args.n_folds, n_groups_off)
    folds = make_folds(groups_off.tolist(), n_splits=n_folds)

    fold_maes_off, fold_maes_on = [], []
    for fold, (idx_train_off, idx_val_off) in enumerate(folds, 1):
        assert_no_leakage(groups_off.tolist(), idx_train_off, idx_val_off, fold_label=f"filter-OFF fold {fold}")
        held_out_identities, idx_train_on, idx_val_on = derive_paired_split(groups_off, idx_val_off, groups_on)

        mae_off = train_and_eval(X_off, y_off, idx_train_off, idx_val_off, args.epochs)
        fold_maes_off.append(mae_off)
        print(f"[filter OFF] fold {fold}/{n_folds} (held out {sorted(held_out_identities)}): MAE = {mae_off:.2f} pts")

        if idx_val_on is None:
            print(f"[filter ON]  fold {fold}/{n_folds}: none of this fold's held-out identities have any "
                  f"surviving sessions in the filtered variant -- this fold can't be paired, skipping it "
                  f"for filter ON (and its filter-OFF result above is excluded from the paired test too).")
            fold_maes_off.pop()
            continue

        mae_on = train_and_eval(X_on, y_on, idx_train_on, idx_val_on, args.epochs)
        fold_maes_on.append(mae_on)
        print(f"[filter ON]  fold {fold}/{n_folds} (same held-out identities): MAE = {mae_on:.2f} pts")

    print("\n" + "=" * 60)
    print("STANCE-SYMMETRY-AWARE TEMPORAL CONFIDENCE -- ABLATION RESULT")
    print("=" * 60)
    lost = len(X_off) - len(X_on)
    print(f"Sessions lost to the new filter: {lost} ({lost / max(len(X_off), 1):.1%} of pre-filter yield)")
    print(f"Paired folds usable for comparison: {len(fold_maes_on)} / {n_folds}")

    if len(fold_maes_off) >= 2 and len(fold_maes_off) == len(fold_maes_on):
        print(f"\nMean MAE without new filter: {np.mean(fold_maes_off):.2f} +/- {np.std(fold_maes_off):.2f} pts ({len(fold_maes_off)} folds)")
        print(f"Mean MAE with new filter:    {np.mean(fold_maes_on):.2f} +/- {np.std(fold_maes_on):.2f} pts ({len(fold_maes_on)} folds)")
        sig = paired_significance_test(fold_maes_on, fold_maes_off)
        print("\nPaired Wilcoxon signed-rank test (filter ON vs. OFF, SAME held-out identities per fold):")
        if sig["note"]:
            print(f"  {sig['note']}")
        else:
            print(f"  statistic={sig['statistic']:.3f}, p={sig['p_value']:.4f} (n={sig['n_pairs']} folds)")
    else:
        print("\nToo few genuinely paired folds survived (need >= 2) -- no significance test run. "
              "This usually means the filter removed an entire identity's sessions in most folds; "
              "re-run with more labeled data.")


if __name__ == "__main__":
    main()
