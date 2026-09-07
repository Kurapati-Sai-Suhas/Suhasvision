"""
Tests for the corrected failure taxonomy.

The two cases at the top are the regressions that matter: they are exactly the
confusions the old single-statistic diagnostic could not express, and they are
what sent a whole investigation after the pose model when the real problem was
tracking.
"""

import unittest

import phase3_failure_taxonomy as FT


class TestTheTwoRegressions(unittest.TestCase):
    """The specific misclassifications the old code produced."""

    def test_low_coverage_but_perfect_pose_is_a_TRACKING_failure(self):
        # sanjay_front _1: the batsman track exists on 1 of 20 sampled frames,
        # and on that frame the pose model succeeds. The old code called this
        # a pose failure because k_pose_rate = 0.05 < 0.4.
        d = FT.decompose(n_sampled=20, n_box=1, n_pose=1)
        self.assertEqual(d["coverage"], 0.05)
        self.assertEqual(d["pose_success_given_box"], 1.0)
        self.assertEqual(d["effective_pose_rate"], 0.05)
        self.assertEqual(d["failure_type"], FT.TRACKING_FAILURE)

    def test_high_coverage_but_low_pose_success_is_a_POSE_failure(self):
        # pro_player_front_12: coverage 1.00, MediaPipe still fails on 80% of
        # frames. This is the genuine article.
        d = FT.decompose(n_sampled=20, n_box=20, n_pose=4)
        self.assertEqual(d["coverage"], 1.0)
        self.assertEqual(d["pose_success_given_box"], 0.2)
        self.assertEqual(d["failure_type"], FT.POSE_FAILURE)


class TestUnreachableBranchIsFixed(unittest.TestCase):
    def test_tracking_failure_is_reachable_at_all(self):
        # The old ordering made this branch dead code: because
        # k_pose_rate <= coverage, coverage<0.4 always tripped the pose test
        # first. At least one input must now yield TRACKING_FAILURE.
        self.assertEqual(FT.decompose(20, 2, 2)["failure_type"],
                         FT.TRACKING_FAILURE)

    def test_low_coverage_never_alone_produces_POSE_FAILURE(self):
        # Sweep every count combination with bad coverage and good pose. None
        # may be labelled a pose failure.
        for n_box in range(1, 8):            # coverage < 0.4 for n_box < 8
            for n_pose in range(n_box + 1):
                d = FT.decompose(20, n_box, n_pose)
                self.assertNotEqual(d["failure_type"], FT.POSE_FAILURE,
                                    msg=f"n_box={n_box} n_pose={n_pose}")

    def test_joint_failure_is_named_not_forced_into_one_bucket(self):
        d = FT.decompose(n_sampled=20, n_box=5, n_pose=1)   # 0.25 and 0.20
        self.assertEqual(d["failure_type"], FT.JOINT_FAILURE)

    def test_healthy_clip_is_no_failure(self):
        self.assertEqual(FT.decompose(20, 20, 19)["failure_type"], FT.NO_FAILURE)


class TestDecomposition(unittest.TestCase):
    def test_effective_rate_is_the_product_of_the_two_factors(self):
        # The stored factors are rounded to 4dp for output, so the identity
        # reconstructs to within that rounding rather than to machine epsilon.
        for n_box in range(1, 21):
            for n_pose in range(n_box + 1):
                d = FT.decompose(20, n_box, n_pose)
                self.assertAlmostEqual(
                    d["effective_pose_rate"],
                    FT.effective_pose_rate(d["coverage"],
                                           d["pose_success_given_box"]),
                    delta=1e-4, msg=f"n_box={n_box} n_pose={n_pose}")

    def test_effective_rate_equals_the_old_k_pose_rate(self):
        # The replacement must measure the same headline quantity, so the
        # corrected taxonomy stays comparable with prior numbers.
        self.assertAlmostEqual(FT.decompose(20, 11, 4)["effective_pose_rate"],
                               4 / 20, places=6)

    def test_effective_rate_never_exceeds_coverage(self):
        # The inequality that made the old ordering broken, asserted directly.
        for n_box in range(0, 21):
            for n_pose in range(n_box + 1):
                d = FT.decompose(20, n_box, n_pose)
                self.assertLessEqual(d["effective_pose_rate"], d["coverage"])

    def test_a_pose_cannot_exist_without_a_box(self):
        with self.assertRaises(ValueError):
            FT.decompose(n_sampled=20, n_box=3, n_pose=4)

    def test_boxes_cannot_exceed_sampled_frames(self):
        with self.assertRaises(ValueError):
            FT.decompose(n_sampled=20, n_box=21, n_pose=0)

    def test_track_with_no_boxes_reports_no_pose_evidence(self):
        d = FT.decompose(n_sampled=20, n_box=0, n_pose=0)
        self.assertEqual(d["coverage"], 0.0)
        self.assertIsNone(d["pose_success_given_box"])
        self.assertEqual(d["failure_type"], FT.TRACKING_FAILURE)


class TestClassifyDirectly(unittest.TestCase):
    def test_boundary_values_are_not_failures(self):
        # Thresholds are strict "<", so exactly 0.40 passes.
        self.assertEqual(FT.classify(0.40, 0.40), FT.NO_FAILURE)

    def test_just_below_boundary_fails(self):
        self.assertEqual(FT.classify(0.399, 0.99), FT.TRACKING_FAILURE)
        self.assertEqual(FT.classify(0.99, 0.399), FT.POSE_FAILURE)

    def test_thresholds_are_overridable(self):
        self.assertEqual(FT.classify(0.5, 0.9, coverage_threshold=0.6),
                         FT.TRACKING_FAILURE)

    def test_every_verdict_has_documented_semantics(self):
        for t in FT.FAILURE_TYPES:
            self.assertIn(t, FT.SEMANTICS)
            self.assertTrue(FT.SEMANTICS[t].strip())

    def test_missing_coverage_is_not_reported_as_a_failure(self):
        self.assertEqual(FT.classify(None, None), FT.NO_FAILURE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
