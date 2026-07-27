"""
evaluation_protocol.py -- single source of truth for cross-validation
methodology, shared across cross_validate.py, ablation_study.py, and
evaluate_baselines.py (Milestone 2: Unified Evaluation Protocol).

Before this module existed:
  - extract_batsman_name was independently redefined in train_advanced_model.py
    and cross_validate.py (identical logic, duplicated).
  - ablation_study.py grouped by de-flipped SESSION NAME, not by batsman
    IDENTITY -- two different recordings of the same real person
    ("PlayerA_side_0s_01" and "PlayerA_side_0s_02") were treated as two
    different groups, so the model could see one of a person's sessions in
    training and be tested on another session of that SAME person. That's a
    real leakage risk for a generalization claim, not just a style
    inconsistency -- routing this file through extract_batsman_name fixes it.
  - evaluate_baselines.py's Leave-One-Out loop held out one SESSION at a
    time, not one IDENTITY at a time. With 13 sessions likely covering
    fewer than 13 distinct people, the same leakage risk applied: a
    left-out session's same-person sibling could still be in the training
    fold. Routing this through LeaveOneGroupOut fixes it the same way.
  - None of the three scripts reported rank correlation, ran a significance
    test against a baseline, or broke results down by skill level / shot
    type. This module adds all of that in one place instead of four times.
"""
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from scipy.stats import spearmanr, kendalltau, wilcoxon

SCORE_NAMES = ["Balance", "Power", "Technique", "Defence"]


def extract_batsman_name(session_name):
    """
    Extracts the base batsman identity so held-out validation never puts a
    batsman's flipped/jittered clone in both train and validation. Kept
    byte-identical to the pre-Milestone-2 versions in train_advanced_model.py
    and cross_validate.py -- this is a relocation, not a rewrite.
    """
    base = session_name.replace("_flipped", "")
    parts = base.split('_')
    if base.startswith("youtube_dataset"):
        return "_".join(parts[:3])
    return parts[0]


def make_folds(groups, n_splits=None):
    """
    Returns a list of (train_idx, test_idx) index arrays, grouped by
    identity so no batsman's sessions -- including flip/jitter clones --
    ever split across train and test within the same fold.

    n_splits=None (default): LeaveOneGroupOut, one fold per unique identity.
    Appropriate for small-sample studies (e.g. evaluate_baselines.py).
    n_splits=<int>: GroupKFold with that many folds (e.g. cross_validate.py's
    3-fold, ablation_study.py's 5-fold). Raises ValueError if n_splits
    exceeds the number of unique identities, since GroupKFold can't honor
    that request.
    """
    groups = np.asarray(groups)
    n_groups = len(set(groups))
    dummy_X = np.zeros(len(groups))
    if n_splits is None:
        return list(LeaveOneGroupOut().split(dummy_X, groups=groups))
    if n_splits > n_groups:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of unique identities "
            f"({n_groups}); GroupKFold requires n_splits <= n_groups."
        )
    return list(GroupKFold(n_splits=n_splits).split(dummy_X, groups=groups))


def assert_no_leakage(groups, train_idx, test_idx, fold_label=""):
    """Raises AssertionError if any identity appears in both splits."""
    groups = np.asarray(groups)
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups & test_groups
    label = f" in {fold_label}" if fold_label else ""
    assert not overlap, f"Leakage{label}: identities {overlap} appear in both train and test."


def compute_metrics(y_true, y_pred, score_names=SCORE_NAMES):
    """
    y_true, y_pred: (n_samples, n_scores) arrays on the SAME scale (caller's
    responsibility -- pass both as 0-100 or both as 0-1, consistently).

    Returns {score_name: {"mae": float, "spearman": (rho, p), "kendall": (tau, p)}}.
    Rank correlations are NaN, not a coincidental 0.0, when a score has no
    variance within this fold -- there's no rank to compute, and reporting
    0.0 would misleadingly read as "no correlation" rather than "undefined".
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    results = {}
    for i, name in enumerate(score_names):
        mae = float(np.mean(np.abs(y_pred[:, i] - y_true[:, i])))
        if np.std(y_true[:, i]) == 0 or np.std(y_pred[:, i]) == 0:
            spearman = (float("nan"), float("nan"))
            kendall = (float("nan"), float("nan"))
        else:
            rho, p_s = spearmanr(y_true[:, i], y_pred[:, i])
            tau, p_k = kendalltau(y_true[:, i], y_pred[:, i])
            spearman = (float(rho), float(p_s))
            kendall = (float(tau), float(p_k))
        results[name] = {"mae": mae, "spearman": spearman, "kendall": kendall}
    return results


def constant_mean_baseline_predict(y_train, n_test):
    """
    The simplest possible baseline: predict the training-fold mean for
    every test sample, regardless of input. Any real model that can't beat
    this on held-out data isn't learning anything from the input features
    -- this is the floor every reported MAE should be compared against.
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    mean_per_score = y_train.mean(axis=0)
    return np.tile(mean_per_score, (n_test, 1))


def paired_significance_test(scores_a, scores_b):
    """
    Wilcoxon signed-rank test on paired per-fold scores (e.g. per-fold MAE
    for architecture A vs. architecture B, or model vs. baseline) -- tests
    whether the paired differences are systematically non-zero, which is
    the actual question "is A better than B" requires, not just comparing
    two mean-of-folds numbers with overlapping-looking error bars.

    Returns {"statistic", "p_value", "n_pairs", "note"}. Only two cases are
    genuinely too degenerate to test: fewer than 2 folds, or every paired
    difference being exactly zero (no signal to rank at all -- scipy itself
    only special-cases this one, not "differences that happen to be a
    constant non-zero value", which is a real, valid, informative result --
    the most extreme, consistent margin a paired test can show -- and must
    not be reported as "not enough data".
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    if len(scores_a) != len(scores_b):
        raise ValueError("paired_significance_test requires equal-length paired arrays.")
    diffs = scores_a - scores_b
    if len(scores_a) < 2 or np.all(diffs == 0):
        return {
            "statistic": float("nan"), "p_value": float("nan"), "n_pairs": len(scores_a),
            "note": "not enough non-tied paired folds to run Wilcoxon signed-rank",
        }
    statistic, p_value = wilcoxon(scores_a, scores_b)
    return {"statistic": float(statistic), "p_value": float(p_value), "n_pairs": len(scores_a), "note": None}


def breakdown_by_column(labels_df, session_ids, per_session_values, column):
    """
    Groups a per-session scalar (typically per-session absolute error) by a
    labels.csv column such as "skill_level" or "shot_type", returning
    {value: {"n": count, "mean": float}} -- the skill-level / shot-type
    breakdown the frozen spec requires alongside the aggregate MAE, so a
    reported number can't be hiding a model that only works on one bucket.
    """
    lookup = labels_df.drop_duplicates("session_name").set_index("session_name")[column].to_dict()
    buckets = {}
    for sid, value in zip(session_ids, per_session_values):
        key = lookup.get(sid, "Unknown")
        buckets.setdefault(key, []).append(value)
    return {key: {"n": len(vals), "mean": float(np.mean(vals))} for key, vals in buckets.items()}
