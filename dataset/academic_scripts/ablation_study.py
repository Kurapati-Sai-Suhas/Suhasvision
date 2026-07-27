import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# Milestone 1 (deferred here): this script used to redefine SEQ_LEN,
# FEATURE_COLS, and TemporalAttention independently instead of importing
# the canonical versions. Milestone 2 (unified evaluation protocol): its
# leakage prevention used to group by de-flipped SESSION NAME, not by
# batsman IDENTITY -- two different recordings of the same real person
# ("PlayerA_side_0s_01" and "PlayerA_side_0s_02") were treated as two
# different groups, so the model could see one of a person's sessions in
# training and be tested on another session of that same person. Routing
# this through extract_batsman_name / make_folds (same as cross_validate.py
# and train_advanced_model.py) fixes both.
_academic_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_dataset_dir = os.path.dirname(_academic_scripts_dir)
for _p in (_dataset_dir, _academic_scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from schema import SEQ_LEN, EXPECTED_FEATURES as FEATURE_COLS  # noqa: E402
from model_layers import TemporalAttention  # noqa: E402
from evaluation_protocol import (  # noqa: E402
    extract_batsman_name,
    make_folds,
    assert_no_leakage,
    compute_metrics,
    constant_mean_baseline_predict,
    paired_significance_test,
    breakdown_by_column,
    SCORE_NAMES,
)

SCORE_COLS = ["score_balance", "score_power", "score_technique", "score_defence"]


def build_sequences(df):
    session_ids, X, y = [], [], []
    for session_name, group in df.groupby("session_name"):
        if len(group) != SEQ_LEN:
            continue
        group = group.sort_values("frame_name")
        X.append(group[FEATURE_COLS].values.astype(np.float32))
        y.append(group.iloc[0][SCORE_COLS].values.astype(np.float32) / 100.0)
        session_ids.append(session_name)
    return np.array(X), np.array(y), np.array(session_ids)


def build_model(use_conv, use_attention, seq_len, n_features):
    inputs = layers.Input(shape=(seq_len, n_features))
    x = inputs

    if use_conv:
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=use_attention))(x)
    x = layers.Dropout(0.3)(x)

    if use_attention:
        x = TemporalAttention()(x)

    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset_angles.csv")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    X, y, session_ids = build_sequences(df)
    groups = [extract_batsman_name(sid) for sid in session_ids]

    folds = make_folds(groups, n_splits=args.n_folds)
    print(f"Starting {len(folds)}-Fold Cross Validation across 4 architectures "
          f"(identity-grouped -- {len(set(groups))} unique batsmen)...")

    architectures = [
        ("Base LSTM", False, False),
        ("LSTM + Attn", False, True),
        ("Conv1D + LSTM", True, False),
        ("Conv1D + LSTM + Attn", True, True),
    ]
    WINNER = "Conv1D + LSTM + Attn"

    results = {name: [] for name, _, _ in architectures}
    baseline_maes = []
    final_model_per_metric = {name: {"mae": [], "spearman": []} for name in SCORE_NAMES}
    all_test_session_ids = []
    all_test_session_errors = []  # winning architecture only

    for fold, (idx_train, idx_val) in enumerate(folds, 1):
        assert_no_leakage(groups, idx_train, idx_val, fold_label=f"Fold {fold}")
        print(f"\n--- Fold {fold}/{len(folds)} ---")

        X_train, y_train = X[idx_train], y[idx_train]
        X_val, y_val = X[idx_val], y[idx_val]

        # Calculate weights for this fold
        shot_types = df.drop_duplicates("session_name").set_index("session_name")["shot_type"].to_dict()
        train_sids = [session_ids[i] for i in idx_train]
        seq_shot_types = [shot_types.get(sid, "Unknown") for sid in train_sids]
        unique_shots, counts = np.unique(seq_shot_types, return_counts=True)
        weight_dict = {shot: len(seq_shot_types) / (len(unique_shots) * count) for shot, count in zip(unique_shots, counts)}
        w_train = np.array([weight_dict[shot] for shot in seq_shot_types])

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

        # The trivial baseline, on this same fold, for a fair comparison
        baseline_preds = constant_mean_baseline_predict(y_train * 100.0, len(idx_val)) / 100.0
        baseline_maes.append(float(np.mean(np.abs(baseline_preds - y_val))) * 100.0)

        for name, use_conv, use_attention in architectures:
            model = build_model(use_conv, use_attention, SEQ_LEN, len(FEATURE_COLS))
            model.fit(X_train, y_train, validation_data=(X_val, y_val), sample_weight=w_train,
                      epochs=args.epochs, batch_size=16, callbacks=callbacks, verbose=0)

            _, val_mae = model.evaluate(X_val, y_val, verbose=0)
            results[name].append(val_mae * 100.0)  # 0-100 scale
            print(f"{name}: Fold {fold} MAE = {val_mae * 100.0:.2f} pts")

            if name == WINNER:
                preds = model.predict(X_val, verbose=0) * 100.0
                y_val_100 = y_val * 100.0
                metrics = compute_metrics(y_val_100, preds, score_names=SCORE_NAMES)
                for score_name in SCORE_NAMES:
                    final_model_per_metric[score_name]["mae"].append(metrics[score_name]["mae"])
                    final_model_per_metric[score_name]["spearman"].append(metrics[score_name]["spearman"][0])
                per_session_abs_err = np.mean(np.abs(preds - y_val_100), axis=1)
                all_test_session_ids.extend(np.asarray(session_ids)[idx_val].tolist())
                all_test_session_errors.extend(per_session_abs_err.tolist())

    print("\n\n" + "=" * 50)
    print(f"TABLE 2: ABLATION STUDY RESULTS ({len(folds)}-Fold CV MAE)")
    print("=" * 50)
    for name in results:
        mean_mae = np.mean(results[name])
        std_mae = np.std(results[name])
        print(f"{name:<25}: {mean_mae:.2f} +/- {std_mae:.2f} pts")
    print(f"{'Constant-mean baseline':<25}: {np.mean(baseline_maes):.2f} +/- {np.std(baseline_maes):.2f} pts")

    sig = paired_significance_test(results[WINNER], baseline_maes)
    print(f"\n{WINNER} vs. constant-mean baseline (Wilcoxon signed-rank):")
    if sig["note"]:
        print(f"  {sig['note']}")
    else:
        print(f"  statistic={sig['statistic']:.3f}, p={sig['p_value']:.4f} (n={sig['n_pairs']} folds)")

    print("\n" + "=" * 50)
    print(f"TABLE 3: PER-METRIC MAE AND RANK CORRELATION FOR {WINNER}")
    print("=" * 50)
    for name in SCORE_NAMES:
        mean_mae = np.mean(final_model_per_metric[name]["mae"])
        std_mae = np.std(final_model_per_metric[name]["mae"])
        rho_vals = [r for r in final_model_per_metric[name]["spearman"] if not np.isnan(r)]
        rho_str = f"{np.mean(rho_vals):.3f}" if rho_vals else "undefined (no variance in any fold)"
        print(f"{name:<25}: MAE {mean_mae:.2f} +/- {std_mae:.2f} pts | mean Spearman rho: {rho_str}")

    print("\n" + "=" * 50)
    print(f"TABLE 4: {WINNER} -- BREAKDOWN BY SKILL LEVEL AND SHOT TYPE")
    print("=" * 50)
    print("By skill level:")
    for level, stats in breakdown_by_column(df, all_test_session_ids, all_test_session_errors, "skill_level").items():
        print(f"  {level}: n={stats['n']}, mean per-session error={stats['mean']:.2f}")
    print("By shot type:")
    for shot, stats in breakdown_by_column(df, all_test_session_ids, all_test_session_errors, "shot_type").items():
        print(f"  {shot}: n={stats['n']}, mean per-session error={stats['mean']:.2f}")


if __name__ == "__main__":
    main()
