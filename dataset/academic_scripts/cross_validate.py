import numpy as np
import pandas as pd
import tensorflow as tf

from train_advanced_model import build_tcn_attention_model, build_sequences, FEATURE_COLS, SCORE_COLS, SEQ_LEN
# Milestone 2 (unified evaluation protocol): fold generation, leakage
# checking, metrics (MAE + rank correlation), the constant-mean baseline,
# the paired significance test, and the skill-level/shot-type breakdown all
# used to be either hand-rolled here or simply absent. They now come from
# evaluation_protocol.py, the same module ablation_study.py and
# evaluate_baselines.py use.
from evaluation_protocol import (
    extract_batsman_name,
    make_folds,
    assert_no_leakage,
    split_inner_validation,
    compute_metrics,
    constant_mean_baseline_predict,
    paired_significance_test,
    breakdown_by_column,
    SCORE_NAMES,
)


def run_cv(csv_path="dataset_angles.csv", n_splits=3):
    print(f"Loading {csv_path} for {n_splits}-Fold Grouped Cross Validation...")
    df = pd.read_csv(csv_path)

    X, y, session_ids = build_sequences(df)
    groups = [extract_batsman_name(sid) for sid in session_ids]
    session_ids = np.array(session_ids)

    unique_batsmen = set(groups)
    print(f"Total sessions: {len(X)}")
    print(f"Unique batsmen (groups): {len(unique_batsmen)}")

    folds = make_folds(groups, n_splits=n_splits)

    fold_metrics = {name: {"mae": [], "spearman": [], "kendall": []} for name in SCORE_NAMES}
    overall_maes = []
    baseline_maes = []
    all_test_session_ids = []
    all_test_session_errors = []

    print("\n--- Starting Cross-Validation ---")
    for fold, (train_idx, test_idx) in enumerate(folds, 1):
        assert_no_leakage(groups, train_idx, test_idx, fold_label=f"Fold {fold}")
        test_groups = {groups[i] for i in test_idx}

        print(f"\nFold {fold}/{n_splits} | Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"  Test Batsmen: {test_groups}")

        # 1. Build and Train Model
        tf.keras.backend.clear_session()
        model = build_tcn_attention_model(SEQ_LEN, len(FEATURE_COLS))

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )

        # CRITICAL (fixed 2026-07-28): early stopping must NEVER see the test
        # fold. This previously passed validation_data=(X[test_idx], ...) with
        # restore_best_weights=True and then scored the model on that SAME
        # fold -- so the returned weights were explicitly SELECTED to minimise
        # loss on the data being reported. That is not a generalisation
        # estimate, it is model selection on the test set.
        #
        # MEASURED impact of this bug on the real production dataset (130
        # sessions / 42 identities, 5 folds, seed 42): reported MAE 10.85
        # +/- 1.95 was optimistic by +4.23 points -- the honest number under
        # this corrected protocol is 15.08 +/- 3.03 (+39%). Mean Spearman
        # likewise fell 0.279 -> 0.145. Anything citing the old number is
        # citing a test-set-selected result.
        inner_train_idx, inner_val_idx = split_inner_validation(
            groups, train_idx, val_fraction=0.25, seed=42
        )
        assert_no_leakage(groups, inner_train_idx, test_idx, fold_label=f"Fold {fold} inner-train vs test")
        assert_no_leakage(groups, inner_val_idx, test_idx, fold_label=f"Fold {fold} inner-val vs test")

        # We don't use sample_weights in CV to evaluate raw generalization
        model.fit(
            X[inner_train_idx], y[inner_train_idx],
            epochs=80, batch_size=16,
            validation_data=(X[inner_val_idx], y[inner_val_idx]),
            callbacks=[early_stopping],
            verbose=0
        )

        # 2. Evaluate on Held-Out Test Set (0-100 scale for reporting)
        preds = model.predict(X[test_idx], verbose=0) * 100.0
        y_true_100 = y[test_idx] * 100.0

        metrics = compute_metrics(y_true_100, preds, score_names=SCORE_NAMES)
        overall_mae = float(np.mean([metrics[name]["mae"] for name in SCORE_NAMES]))
        overall_maes.append(overall_mae)
        for name in SCORE_NAMES:
            fold_metrics[name]["mae"].append(metrics[name]["mae"])
            fold_metrics[name]["spearman"].append(metrics[name]["spearman"][0])
            fold_metrics[name]["kendall"].append(metrics[name]["kendall"][0])

        # 3. The trivial baseline, on the SAME fold, for a fair per-fold comparison
        baseline_preds = constant_mean_baseline_predict(y[train_idx] * 100.0, len(test_idx))
        baseline_metrics = compute_metrics(y_true_100, baseline_preds, score_names=SCORE_NAMES)
        baseline_maes.append(float(np.mean([baseline_metrics[name]["mae"] for name in SCORE_NAMES])))

        # 4. Per-session error, for the skill-level / shot-type breakdown below
        per_session_abs_err = np.mean(np.abs(preds - y_true_100), axis=1)
        all_test_session_ids.extend(session_ids[test_idx].tolist())
        all_test_session_errors.extend(per_session_abs_err.tolist())

        print(f"  Held-Out MAE: {overall_mae:.2f} (constant-mean baseline: {baseline_maes[-1]:.2f})")
        for name in SCORE_NAMES:
            rho = metrics[name]["spearman"][0]
            print(f"    - {name}: MAE {metrics[name]['mae']:.2f}, Spearman rho {rho:.3f}" if not np.isnan(rho)
                  else f"    - {name}: MAE {metrics[name]['mae']:.2f}, Spearman rho undefined (no variance this fold)")

    print("\n=== FINAL CROSS-VALIDATION RESULTS ===")
    print(f"Overall Held-Out MAE: {np.mean(overall_maes):.2f} +/- {np.std(overall_maes):.2f}")
    print(f"Constant-Mean Baseline MAE: {np.mean(baseline_maes):.2f} +/- {np.std(baseline_maes):.2f}")

    sig = paired_significance_test(overall_maes, baseline_maes)
    if sig["note"]:
        print(f"Significance vs. baseline: {sig['note']}")
    else:
        print(f"Significance vs. baseline (Wilcoxon signed-rank): statistic={sig['statistic']:.3f}, "
              f"p={sig['p_value']:.4f} (n={sig['n_pairs']} folds)")

    print("\nPer-metric MAE and rank correlation:")
    for name in SCORE_NAMES:
        mae_mean = np.mean(fold_metrics[name]["mae"])
        mae_std = np.std(fold_metrics[name]["mae"])
        rho_vals = [r for r in fold_metrics[name]["spearman"] if not np.isnan(r)]
        rho_str = f"{np.mean(rho_vals):.3f}" if rho_vals else "undefined (no variance in any fold)"
        print(f"{name}: MAE {mae_mean:.2f} +/- {mae_std:.2f} | mean Spearman rho: {rho_str}")

    print("\nBreakdown by skill level:")
    for level, stats in breakdown_by_column(df, all_test_session_ids, all_test_session_errors, "skill_level").items():
        print(f"  {level}: n={stats['n']}, mean per-session error={stats['mean']:.2f}")

    print("\nBreakdown by shot type:")
    for shot, stats in breakdown_by_column(df, all_test_session_ids, all_test_session_errors, "shot_type").items():
        print(f"  {shot}: n={stats['n']}, mean per-session error={stats['mean']:.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset_angles.csv")
    ap.add_argument("--n_splits", type=int, default=3)
    args = ap.parse_args()
    run_cv(csv_path=args.input, n_splits=args.n_splits)
