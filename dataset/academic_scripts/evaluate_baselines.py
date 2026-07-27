import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# Milestone 1 (deferred here): SEQ_LEN/FEATURE_COLS/SCORE_COLS used to be
# redefined independently in this file. Milestone 2 (unified evaluation
# protocol): this file's Leave-One-Out loop held out one SESSION at a time,
# not one IDENTITY at a time. With 13 sessions likely covering fewer than
# 13 distinct people, a left-out session's same-person sibling could still
# be in the training fold -- a real leakage risk, not just a style
# inconsistency. Routing this through extract_batsman_name / make_folds
# (LeaveOneGroupOut) fixes it the same way ablation_study.py was fixed.
_academic_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_dataset_dir = os.path.dirname(_academic_scripts_dir)
for _p in (_dataset_dir, _academic_scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from schema import SEQ_LEN, EXPECTED_FEATURES as FEATURE_COLS  # noqa: E402
from evaluation_protocol import (  # noqa: E402
    extract_batsman_name,
    make_folds,
    assert_no_leakage,
    compute_metrics,
    constant_mean_baseline_predict,
    SCORE_NAMES,
)

SCORE_COLS = ["score_balance", "score_power", "score_technique", "score_defence"]
# CONSENTED_PLAYERS was defined here previously and never used anywhere --
# removed rather than carried forward as dead code.


def load_clean_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # The dataset_angles.csv currently contains youtube_dataset names.
    # To simulate the 13 consented videos, we will pick exactly 13 unique non-flipped videos.
    unique_sessions = [s for s in df["session_name"].unique() if not s.endswith("_flipped")]
    target_sessions = unique_sessions[:13]

    df_clean = df[df["session_name"].isin(target_sessions)]
    print(f"Filtered dataset from {len(df)} to {len(df_clean)} rows (13 clean videos).")

    session_ids = []
    X = []
    y = []

    for session_name, group in df_clean.groupby("session_name"):
        if len(group) != SEQ_LEN:
            continue
        group = group.sort_values("frame_name")

        seq = group[FEATURE_COLS].values.astype(np.float32)
        scores = group.iloc[0][SCORE_COLS].values.astype(np.float32)
        scores = np.where(scores > 100, scores / 10.0, scores)  # Fix typos
        scores = np.clip(scores, 0, 100) / 100.0

        X.append(seq)
        y.append(scores)
        session_ids.append(session_name)

    return np.array(X), np.array(y), np.array(session_ids)


def augment_training_data(X_train, y_train, num_variants=4, noise_std=0.02):
    """
    Augments the training set by adding Gaussian noise to the angles.
    noise_std=0.02 corresponds to ~3.6 degrees of noise (since angles are 0-1 mapped to 0-180).
    """
    X_aug = [X_train]
    y_aug = [y_train]

    for _ in range(num_variants):
        noise = np.random.normal(0, noise_std, X_train.shape)
        X_noisy = np.clip(X_train + noise, 0.0, 1.0)
        X_aug.append(X_noisy)
        y_aug.append(y_train)

    return np.vstack(X_aug), np.vstack(y_aug)


# ================================
# MODELS
# ================================

class RuleBasedModel:
    """
    A naive heuristic model: Finds the single highest correlated joint angle for each metric
    in the training set, and uses a simple linear equation y = mx + b to predict.
    No deep learning.
    """
    def __init__(self):
        self.rules = []

    def fit(self, X_train, y_train):
        self.rules = []
        # Flatten time dimension by taking the mean angle across the 7 frames
        X_mean = np.mean(X_train, axis=1)  # Shape: (samples, 30 features)

        for score_idx in range(4):
            y_target = y_train[:, score_idx]
            best_corr = 0
            best_feat_idx = 0
            best_m, best_b = 0, 0

            for feat_idx in range(len(FEATURE_COLS)):
                feat_vals = X_mean[:, feat_idx]
                # Avoid constant features
                if np.std(feat_vals) < 1e-6:
                    continue

                corr, _ = pearsonr(feat_vals, y_target)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_feat_idx = feat_idx
                    # Calculate simple y = mx + b
                    m = corr * (np.std(y_target) / np.std(feat_vals))
                    b = np.mean(y_target) - m * np.mean(feat_vals)
                    best_m, best_b = m, b

            self.rules.append((best_feat_idx, best_m, best_b))

    def predict(self, X_test):
        X_mean = np.mean(X_test, axis=1)
        preds = []
        for i in range(len(X_test)):
            sample_preds = []
            for score_idx in range(4):
                feat_idx, m, b = self.rules[score_idx]
                val = m * X_mean[i, feat_idx] + b
                sample_preds.append(np.clip(val, 0.0, 1.0))
            preds.append(sample_preds)
        return np.array(preds)


def build_mlp():
    inputs = layers.Input(shape=(SEQ_LEN, len(FEATURE_COLS)))
    x = layers.Flatten()(inputs)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_cnn():
    inputs = layers.Input(shape=(SEQ_LEN, len(FEATURE_COLS)))
    x = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_simplified_bilstm():
    inputs = layers.Input(shape=(SEQ_LEN, len(FEATURE_COLS)))
    # Massively reduced hidden state to prevent memorization
    x = layers.Bidirectional(layers.LSTM(8))(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ================================
# EVALUATION LOOP
# ================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "..", "dataset_angles.csv"),
                     help="Path to the angles CSV. Defaults to the real dataset -- override for testing "
                          "so a smoke test never has to write to (or risk overwriting) real data.")
    args = ap.parse_args()

    X, y, session_ids = load_clean_dataset(args.input)

    n_samples = len(X)
    if n_samples == 0:
        print("Error: No data found.")
        return

    groups = [extract_batsman_name(sid) for sid in session_ids]
    n_identities = len(set(groups))
    folds = make_folds(groups, n_splits=None)  # LeaveOneGroupOut -- one fold per identity
    print(f"\nStarting identity-grouped Leave-One-Out CV: {n_samples} videos across "
          f"{n_identities} unique batsmen ({len(folds)} folds)...")
    if n_identities < n_samples:
        print(f"Note: {n_samples} videos but only {n_identities} unique identities -- "
              f"this is fewer, stricter folds than a naive per-session LOO would run, "
              f"specifically because it now refuses to leak a person's other session into training.")

    # Track absolute errors for each model across all folds
    errors_rule, errors_mlp, errors_cnn, errors_lstm, errors_baseline = [], [], [], [], []
    preds_rule_all, preds_mlp_all, preds_cnn_all, preds_lstm_all, preds_baseline_all = [], [], [], [], []
    y_test_all = []

    for fold_idx, (train_indices, test_indices) in enumerate(folds, 1):
        assert_no_leakage(groups, train_indices, test_indices, fold_label=f"Fold {fold_idx}")

        X_test, y_test = X[test_indices], y[test_indices]
        X_train_clean, y_train_clean = X[train_indices], y[train_indices]
        test_ids = np.asarray(session_ids)[test_indices]

        # Augment (only the training fold)
        X_train_aug, y_train_aug = augment_training_data(X_train_clean, y_train_clean)

        print(f"Fold {fold_idx}/{len(folds)} | Test: {list(test_ids)} | Train Size: {len(X_train_aug)} (Augmented)")

        # --- Rule-Based ---
        model_rule = RuleBasedModel()
        model_rule.fit(X_train_aug, y_train_aug)
        preds_rule = model_rule.predict(X_test)
        errors_rule.append(mean_absolute_error(y_test, preds_rule))

        # --- Simple MLP ---
        model_mlp = build_mlp()
        model_mlp.fit(X_train_aug, y_train_aug, epochs=50, verbose=0)
        preds_mlp = model_mlp.predict(X_test, verbose=0)
        errors_mlp.append(mean_absolute_error(y_test, preds_mlp))

        # --- 1D-CNN ---
        model_cnn = build_cnn()
        model_cnn.fit(X_train_aug, y_train_aug, epochs=50, verbose=0)
        preds_cnn = model_cnn.predict(X_test, verbose=0)
        errors_cnn.append(mean_absolute_error(y_test, preds_cnn))

        # --- Simplified Bi-LSTM ---
        model_lstm = build_simplified_bilstm()
        model_lstm.fit(X_train_aug, y_train_aug, epochs=50, verbose=0)
        preds_lstm = model_lstm.predict(X_test, verbose=0)
        errors_lstm.append(mean_absolute_error(y_test, preds_lstm))

        # --- Constant-mean baseline (the trivial floor every model above should beat) ---
        preds_baseline = constant_mean_baseline_predict(y_train_clean, len(test_indices))
        errors_baseline.append(mean_absolute_error(y_test, preds_baseline))

        y_test_all.append(y_test)
        preds_rule_all.append(preds_rule)
        preds_mlp_all.append(preds_mlp)
        preds_cnn_all.append(preds_cnn)
        preds_lstm_all.append(preds_lstm)
        preds_baseline_all.append(preds_baseline)

    # Output Results
    print("\n" + "=" * 50)
    print(f"FINAL {len(folds)}-FOLD IDENTITY-GROUPED LOO-CV RESULTS (Mean Absolute Error on 1-100 scale)")
    print("=" * 50)
    print(f"Naive Rule-Based        : {np.mean(errors_rule) * 100:.2f} points (Std: {np.std(errors_rule)*100:.2f})")
    print(f"Simple MLP              : {np.mean(errors_mlp) * 100:.2f} points (Std: {np.std(errors_mlp)*100:.2f})")
    print(f"1D-CNN                  : {np.mean(errors_cnn) * 100:.2f} points (Std: {np.std(errors_cnn)*100:.2f})")
    print(f"Shrunk Bi-LSTM          : {np.mean(errors_lstm) * 100:.2f} points (Std: {np.std(errors_lstm)*100:.2f})")
    print(f"Constant-Mean Baseline  : {np.mean(errors_baseline) * 100:.2f} points (Std: {np.std(errors_baseline)*100:.2f})")
    print("=" * 50)

    # The MAE figures above are identity-fair by construction: mean_absolute_error
    # is computed once per fold (= once per identity, since folds are now
    # identity-grouped), then averaged across folds, so an identity with 3
    # held-out sessions counts the same as one with 1. Pooling raw sessions
    # for rank correlation below does NOT have that property -- an identity
    # with more sessions contributes proportionally more data points to the
    # correlation. At this sample size (13 videos across a handful of
    # people) that's a real, not just theoretical, difference in what the
    # two reported numbers actually represent -- flagged explicitly so
    # "identity-fair MAE" and "session-weighted correlation" aren't read as
    # the same kind of average.
    sessions_per_identity = pd.Series(groups).value_counts()
    print(f"\nRank correlation (Spearman) per model, pooled across all {n_samples} held-out sessions "
          f"({n_identities} identities, {sessions_per_identity.min()}-{sessions_per_identity.max()} sessions each).")
    print("Note: unlike the identity-fair MAE above, this pools individual sessions, so an identity "
          "with more held-out clips has proportionally more influence on the correlation figure.")
    y_true_pooled = np.concatenate(y_test_all) * 100.0
    for label, preds_all in [
        ("Naive Rule-Based", preds_rule_all), ("Simple MLP", preds_mlp_all),
        ("1D-CNN", preds_cnn_all), ("Shrunk Bi-LSTM", preds_lstm_all),
        ("Constant-Mean Baseline", preds_baseline_all),
    ]:
        preds_pooled = np.concatenate(preds_all) * 100.0
        metrics = compute_metrics(y_true_pooled, preds_pooled, score_names=SCORE_NAMES)
        rho_vals = [metrics[n]["spearman"][0] for n in SCORE_NAMES if not np.isnan(metrics[n]["spearman"][0])]
        rho_str = f"{np.mean(rho_vals):.3f}" if rho_vals else "undefined (no variance)"
        print(f"  {label:<24}: mean Spearman rho across 4 scores = {rho_str}")


if __name__ == "__main__":
    main()
