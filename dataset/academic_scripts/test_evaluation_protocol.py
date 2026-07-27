import unittest
import numpy as np
import pandas as pd

from evaluation_protocol import (
    extract_batsman_name,
    make_folds,
    assert_no_leakage,
    compute_metrics,
    constant_mean_baseline_predict,
    paired_significance_test,
    breakdown_by_column,
)


class TestExtractBatsmanName(unittest.TestCase):
    def test_youtube_dataset_pattern(self):
        self.assertEqual(
            extract_batsman_name("youtube_dataset_10_1_frontview_10s_02"),
            "youtube_dataset_10",
        )

    def test_plain_player_pattern(self):
        self.assertEqual(extract_batsman_name("PlayerA_side_0s_01"), "PlayerA")

    def test_flipped_clone_groups_with_original(self):
        self.assertEqual(
            extract_batsman_name("PlayerA_side_0s_01"),
            extract_batsman_name("PlayerA_side_0s_01_flipped"),
        )

    def test_different_sessions_of_same_real_person_group_together(self):
        # This is the exact leakage gap ablation_study.py's old de-flip-only
        # grouping missed: two distinct recordings of the same person must
        # land in the same identity group, not two different ones.
        self.assertEqual(
            extract_batsman_name("PlayerA_side_0s_01"),
            extract_batsman_name("PlayerA_side_0s_02"),
        )

    def test_jittered_clone_groups_with_original(self):
        self.assertEqual(
            extract_batsman_name("PlayerA_side_0s_01"),
            extract_batsman_name("PlayerA_side_0s_01_jitterA"),
        )


class TestMakeFolds(unittest.TestCase):
    def setUp(self):
        # 4 identities, uneven session counts per identity -- deliberately
        # not a clean multiple, to catch off-by-one group-count bugs.
        self.groups = (
            ["p1"] * 3 + ["p2"] * 2 + ["p3"] * 4 + ["p4"] * 1
        )

    def test_leave_one_group_out_produces_one_fold_per_identity(self):
        folds = make_folds(self.groups, n_splits=None)
        self.assertEqual(len(folds), 4)

    def test_group_kfold_produces_requested_fold_count(self):
        folds = make_folds(self.groups, n_splits=3)
        self.assertEqual(len(folds), 3)

    def test_group_kfold_rejects_too_many_splits(self):
        with self.assertRaises(ValueError):
            make_folds(self.groups, n_splits=10)  # only 4 unique identities

    def test_no_fold_ever_leaks_an_identity_across_train_and_test(self):
        # The actual safety property this whole module exists to guarantee.
        for n_splits in (None, 2, 3, 4):
            folds = make_folds(self.groups, n_splits=n_splits)
            for train_idx, test_idx in folds:
                assert_no_leakage(self.groups, train_idx, test_idx)  # must not raise


class TestAssertNoLeakage(unittest.TestCase):
    def test_raises_on_overlap(self):
        groups = ["p1", "p1", "p2", "p2"]
        with self.assertRaises(AssertionError):
            assert_no_leakage(groups, train_idx=[0, 2], test_idx=[1, 3])  # p1 and p2 both split

    def test_passes_on_clean_split(self):
        groups = ["p1", "p1", "p2", "p2"]
        assert_no_leakage(groups, train_idx=[0, 1], test_idx=[2, 3])  # no exception


class TestComputeMetrics(unittest.TestCase):
    def test_mae_matches_hand_computation(self):
        y_true = np.array([[10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0]])
        y_pred = np.array([[12.0, 20.0, 25.0, 40.0], [8.0, 20.0, 35.0, 40.0]])
        # Balance: |12-10|, |8-10| -> mean 2.0. Power: 0. Technique: |25-30|,|35-30| -> mean 5.0. Defence: 0.
        metrics = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["Balance"]["mae"], 2.0, places=6)
        self.assertAlmostEqual(metrics["Power"]["mae"], 0.0, places=6)
        self.assertAlmostEqual(metrics["Technique"]["mae"], 5.0, places=6)
        self.assertAlmostEqual(metrics["Defence"]["mae"], 0.0, places=6)

    def test_perfect_rank_correlation_on_monotonic_data(self):
        y_true = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_pred = np.array([[10.0], [20.0], [30.0], [40.0]])  # same rank order, different scale
        metrics = compute_metrics(y_true, y_pred, score_names=["OnlyScore"])
        rho, _ = metrics["OnlyScore"]["spearman"]
        tau, _ = metrics["OnlyScore"]["kendall"]
        self.assertAlmostEqual(rho, 1.0, places=6)
        self.assertAlmostEqual(tau, 1.0, places=6)

    def test_zero_variance_gives_nan_not_zero(self):
        y_true = np.array([[5.0], [5.0], [5.0]])  # constant -- no meaningful rank
        y_pred = np.array([[1.0], [2.0], [3.0]])
        metrics = compute_metrics(y_true, y_pred, score_names=["OnlyScore"])
        rho, p = metrics["OnlyScore"]["spearman"]
        self.assertTrue(np.isnan(rho))
        self.assertTrue(np.isnan(p))


class TestConstantMeanBaseline(unittest.TestCase):
    def test_predicts_training_mean_for_every_test_row(self):
        y_train = np.array([[10.0, 50.0], [20.0, 70.0], [30.0, 60.0]])  # mean = [20, 60]
        preds = constant_mean_baseline_predict(y_train, n_test=3)
        self.assertEqual(preds.shape, (3, 2))
        for row in preds:
            np.testing.assert_allclose(row, [20.0, 60.0])


class TestPairedSignificanceTest(unittest.TestCase):
    def test_detects_a_real_paired_difference(self):
        # Model A consistently beats Model B by a clear, non-tied margin.
        scores_a = [5.0, 6.0, 4.5, 5.5, 6.5, 5.2]
        scores_b = [9.0, 10.0, 8.5, 9.5, 10.5, 9.2]
        result = paired_significance_test(scores_a, scores_b)
        self.assertLess(result["p_value"], 0.05)
        self.assertEqual(result["n_pairs"], 6)
        self.assertIsNone(result["note"])

    def test_reports_insufficient_data_instead_of_crashing(self):
        result = paired_significance_test([5.0], [9.0])  # only 1 fold
        self.assertTrue(np.isnan(result["p_value"]))
        self.assertIsNotNone(result["note"])

    def test_constant_nonzero_difference_is_a_valid_result_not_insufficient_data(self):
        # Regression test for a real bug found in review: a constant non-zero
        # per-fold difference (model A beats B by exactly the same margin in
        # every fold) is the MOST extreme, most consistent evidence a paired
        # test can show -- scipy computes it without complaint (verified
        # directly against wilcoxon([5,6,7],[7,8,9]) -> p=0.25, not an error).
        # The original implementation wrongly treated "all differences equal"
        # as "not enough data", silently hiding a real result.
        result = paired_significance_test([5.0, 6.0, 7.0], [7.0, 8.0, 9.0])  # constant diff of -2.0
        self.assertFalse(np.isnan(result["p_value"]))
        self.assertIsNone(result["note"])
        self.assertAlmostEqual(result["p_value"], 0.25, places=6)

    def test_reports_insufficient_data_when_all_differences_are_zero(self):
        # The one genuinely degenerate case: no signal to rank at all.
        result = paired_significance_test([5.0, 6.0, 7.0], [5.0, 6.0, 7.0])
        self.assertTrue(np.isnan(result["p_value"]))
        self.assertIsNotNone(result["note"])

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            paired_significance_test([1.0, 2.0], [1.0, 2.0, 3.0])


class TestBreakdownByColumn(unittest.TestCase):
    def test_groups_and_averages_correctly(self):
        labels_df = pd.DataFrame({
            "session_name": ["s1", "s2", "s3", "s4"],
            "skill_level": ["amateur", "amateur", "professional", "professional"],
        })
        session_ids = ["s1", "s2", "s3", "s4"]
        per_session_errors = [2.0, 4.0, 10.0, 20.0]
        result = breakdown_by_column(labels_df, session_ids, per_session_errors, "skill_level")
        self.assertEqual(result["amateur"]["n"], 2)
        self.assertAlmostEqual(result["amateur"]["mean"], 3.0)
        self.assertEqual(result["professional"]["n"], 2)
        self.assertAlmostEqual(result["professional"]["mean"], 15.0)

    def test_missing_session_falls_back_to_unknown_bucket(self):
        labels_df = pd.DataFrame({"session_name": ["s1"], "skill_level": ["amateur"]})
        result = breakdown_by_column(labels_df, ["s1", "s_not_in_labels"], [1.0, 2.0], "skill_level")
        self.assertIn("Unknown", result)
        self.assertEqual(result["Unknown"]["n"], 1)


if __name__ == '__main__':
    unittest.main()
