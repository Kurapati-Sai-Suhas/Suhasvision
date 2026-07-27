import math
import unittest

import numpy as np
import pandas as pd

from stance_symmetry_confidence import (
    TRAJECTORY_JOINTS,
    STANCE_SYMMETRY_REFERENCE,
    STANCE_PHASE_INDICES,
    MIN_TEMPORAL_CONFIDENCE,
    compute_trajectory_smoothness,
    compute_stance_symmetry,
    compute_session_confidence,
)

# A neutral base position for every joint either compute_trajectory_smoothness
# (the 12 TRAJECTORY_JOINTS) or compute_stance_symmetry (adds foot_index,
# needed for the ankle pair) looks at, so tests only need to override the
# joints they actually care about instead of re-specifying all of them.
_BASE = {joint: (float(i), float(i) * 2, 0.0) for i, joint in enumerate(TRAJECTORY_JOINTS)}
_BASE["left_foot_index"] = (0.0, 0.0, 1.0)
_BASE["right_foot_index"] = (1.0, 0.0, 1.0)


def _frame_row(session_name, frame_name, positions):
    """positions: {joint_name: (x, y, z)} overriding _BASE for this frame."""
    row = {"session_name": session_name, "frame_name": frame_name}
    merged = dict(_BASE)
    merged.update(positions)
    for joint, (x, y, z) in merged.items():
        row[f"{joint}_x"] = x
        row[f"{joint}_y"] = y
        row[f"{joint}_z"] = z
    return row


def _static_session(session_name, n_frames, frame_prefix="frame_0"):
    """Every joint at the exact same position in every frame -- zero path
    length, zero jerk, smoothness == 1.0 everywhere by construction."""
    rows = [_frame_row(session_name, f"{frame_prefix}{i+1}", {}) for i in range(n_frames)]
    return pd.DataFrame(rows)


class TestTrajectorySmoothnessStaticSession(unittest.TestCase):
    def test_no_motion_gives_perfect_smoothness(self):
        df = _static_session("s1", 7)
        scores = compute_trajectory_smoothness(df)
        np.testing.assert_allclose(scores, np.ones(7))

    def test_single_frame_session_does_not_crash(self):
        df = _static_session("s1", 1)
        scores = compute_trajectory_smoothness(df)
        np.testing.assert_allclose(scores, [1.0])

    def test_empty_session_returns_empty(self):
        df = _static_session("s1", 0)
        scores = compute_trajectory_smoothness(df)
        self.assertEqual(len(scores), 0)


class TestTrajectorySmoothnessExactJerk(unittest.TestCase):
    def test_hand_computed_single_joint_jerk(self):
        # 3 frames, every joint static EXCEPT left_shoulder, which moves
        # (0,0,0) -> (1,0,0) -> (3,0,0): a real acceleration (non-constant
        # velocity), hand-computable exactly.
        #   accel_1 = pos0 - 2*pos1 + pos2 = (0,0,0) - (2,0,0) + (3,0,0) = (1,0,0)
        #   |accel_1| = 1.0; path_length = |1-0| + |3-1| = 1 + 2 = 3
        #   normalized_jerk = 1/3 -> smoothness_joint = exp(-1/3)
        # Edge frames inherit the (only) interior frame's jerk, so all 3
        # frames get the same per-joint smoothness for this joint.
        joint = "left_shoulder"
        rows = [
            _frame_row("s1", "frame_01", {joint: (0.0, 0.0, 0.0)}),
            _frame_row("s1", "frame_02", {joint: (1.0, 0.0, 0.0)}),
            _frame_row("s1", "frame_03", {joint: (3.0, 0.0, 0.0)}),
        ]
        df = pd.DataFrame(rows)
        scores = compute_trajectory_smoothness(df)

        expected_joint_smoothness = math.exp(-1.0 / 3.0)
        n_joints = len(TRAJECTORY_JOINTS)
        expected_combined = ((n_joints - 1) * 1.0 + expected_joint_smoothness) / n_joints

        np.testing.assert_allclose(scores, [expected_combined] * 3, rtol=1e-6)

    def test_injected_jump_lowers_only_nearby_frames(self):
        # Every joint moves at constant velocity (zero jerk) across 7
        # frames, EXCEPT left_wrist gets a single large displacement at
        # frame index 3 -- a real, isolated pose-estimation glitch.
        n = 7
        rows = []
        for i in range(n):
            positions = {j: (float(i) * 0.1, 0.0, 0.0) for j in TRAJECTORY_JOINTS}
            if i == 3:
                positions["left_wrist"] = (5.0, 5.0, 5.0)  # way off the line
            rows.append(_frame_row("s1", f"frame_0{i+1}", positions))
        df = pd.DataFrame(rows)
        scores = compute_trajectory_smoothness(df)

        # Frames 0, 1, 5, 6 are never adjacent to the perturbed frame 3 in
        # any acceleration window -- exactly 1.0, unaffected.
        self.assertAlmostEqual(scores[0], 1.0, places=9)
        self.assertAlmostEqual(scores[1], 1.0, places=9)
        self.assertAlmostEqual(scores[5], 1.0, places=9)
        self.assertAlmostEqual(scores[6], 1.0, places=9)
        # Frames 2, 3, 4 all sit inside the perturbed joint's acceleration
        # window and must be strictly less smooth than the untouched frames.
        self.assertLess(scores[2], 1.0)
        self.assertLess(scores[3], 1.0)
        self.assertLess(scores[4], 1.0)


class TestStanceSymmetry(unittest.TestCase):
    @staticmethod
    def _right_angle_stance(session_name, frame_names, right_side_skew=0.0):
        """
        Builds one row per frame_name where left_{shoulder,hip,knee,ankle,
        foot_index} form a zigzag of exact 90 degree turns (hand-verifiable:
        each consecutive segment is orthogonal to the last), and the right
        side is an identical shape (diff = 0) unless right_side_skew != 0,
        which tilts the right ankle->foot_index segment OUT OF the straight
        line (a perpendicular offset, not just a longer segment in the same
        direction -- angle depends on direction, not magnitude) to break the
        90 degree angle at the ankle only.
        """
        left = {
            "left_shoulder": (0.0, 0.0, 0.0),
            "left_hip": (0.0, 1.0, 0.0),
            "left_knee": (1.0, 1.0, 0.0),
            "left_ankle": (1.0, 2.0, 0.0),
            "left_foot_index": (2.0, 2.0, 0.0),
        }
        right = {
            "right_shoulder": (10.0, 0.0, 0.0),
            "right_hip": (10.0, 1.0, 0.0),
            "right_knee": (11.0, 1.0, 0.0),
            "right_ankle": (11.0, 2.0, 0.0),
            "right_foot_index": (12.0, 2.0 + right_side_skew, 0.0),
        }
        positions = {**left, **right}
        rows = [_frame_row(session_name, name, positions) for name in frame_names]
        return pd.DataFrame(rows)

    def test_symmetric_stance_hand_computed_score(self):
        # All three bilateral angles (knee, hip, ankle) are exactly 90/90
        # on both sides -> diff = 0 for every pair. Independently
        # hand-compute the expected score from the public reference
        # constants (not by calling any internal helper).
        df = self._right_angle_stance("s1", ["frame_01", "frame_02", "frame_03"])
        scores = compute_stance_symmetry(df)

        expected_pair_scores = []
        for mean_diff, std_diff in STANCE_SYMMETRY_REFERENCE.values():
            z = (0.0 - mean_diff) / (std_diff + 1e-6)
            expected_pair_scores.append(math.exp(-0.5 * z * z))
        expected = float(np.mean(expected_pair_scores))

        self.assertAlmostEqual(scores[0], expected, places=6)
        self.assertAlmostEqual(scores[1], expected, places=6)

    def test_frames_outside_stance_phase_are_nan(self):
        df = self._right_angle_stance("s1", ["frame_01", "frame_02", "frame_03"])
        scores = compute_stance_symmetry(df)
        self.assertEqual(set(STANCE_PHASE_INDICES), {0, 1})
        self.assertTrue(np.isnan(scores[2]))
        self.assertFalse(np.isnan(scores[0]))
        self.assertFalse(np.isnan(scores[1]))

    def test_asymmetric_stance_scores_lower_than_symmetric(self):
        symmetric = self._right_angle_stance("s1", ["frame_01", "frame_02"], right_side_skew=0.0)
        asymmetric = self._right_angle_stance("s2", ["frame_01", "frame_02"], right_side_skew=1.5)

        sym_scores = compute_stance_symmetry(symmetric)
        asym_scores = compute_stance_symmetry(asymmetric)

        self.assertLess(asym_scores[0], sym_scores[0])


class TestComputeSessionConfidence(unittest.TestCase):
    def test_combines_trajectory_and_symmetry_at_stance_frames(self):
        # Static (zero-jerk, smoothness == 1.0 everywhere) session with a
        # symmetric stance -- combined score at stance frames should be the
        # average of 1.0 and the symmetry score; combined score at
        # non-stance frames should equal the trajectory score alone (1.0),
        # not be diluted by a fake neutral symmetry value.
        left = {
            "left_shoulder": (0.0, 0.0, 0.0), "left_hip": (0.0, 1.0, 0.0),
            "left_knee": (1.0, 1.0, 0.0), "left_ankle": (1.0, 2.0, 0.0),
            "left_foot_index": (2.0, 2.0, 0.0),
        }
        right = {
            "right_shoulder": (10.0, 0.0, 0.0), "right_hip": (10.0, 1.0, 0.0),
            "right_knee": (11.0, 1.0, 0.0), "right_ankle": (11.0, 2.0, 0.0),
            "right_foot_index": (12.0, 2.0, 0.0),
        }
        positions = {**left, **right}
        rows = [_frame_row("s1", f"frame_0{i+1}", positions) for i in range(7)]
        df = pd.DataFrame(rows)

        confidence, report = compute_session_confidence(df)

        expected_pair_scores = []
        for mean_diff, std_diff in STANCE_SYMMETRY_REFERENCE.values():
            z = (0.0 - mean_diff) / (std_diff + 1e-6)
            expected_pair_scores.append(math.exp(-0.5 * z * z))
        expected_symmetry = float(np.mean(expected_pair_scores))
        expected_stance_combined = (1.0 + expected_symmetry) / 2.0

        values = confidence.values
        self.assertAlmostEqual(values[0], expected_stance_combined, places=6)
        self.assertAlmostEqual(values[1], expected_stance_combined, places=6)
        for i in range(2, 7):
            self.assertAlmostEqual(values[i], 1.0, places=6)

        self.assertAlmostEqual(report["mean_trajectory_score"], 1.0, places=6)
        self.assertAlmostEqual(report["mean_symmetry_score"], expected_symmetry, places=6)

    def test_returned_series_stays_aligned_to_original_index_when_input_is_scrambled(self):
        # Real keypoints.csv rows are not guaranteed to already be sorted by
        # frame_name -- compute_session_confidence must sort internally but
        # still hand back a Series that lines up with the CALLER's original
        # row index, not the sorted position.
        joint = "left_shoulder"
        ordered_rows = {
            "frame_01": _frame_row("s1", "frame_01", {joint: (0.0, 0.0, 0.0)}),
            "frame_02": _frame_row("s1", "frame_02", {joint: (1.0, 0.0, 0.0)}),
            "frame_03": _frame_row("s1", "frame_03", {joint: (3.0, 0.0, 0.0)}),
        }
        # Deliberately scrambled row order with a non-trivial pandas index.
        scrambled = pd.DataFrame(
            [ordered_rows["frame_03"], ordered_rows["frame_01"], ordered_rows["frame_02"]],
            index=[42, 7, 13],
        )
        confidence, _ = compute_session_confidence(scrambled)

        naturally_ordered = pd.DataFrame(
            [ordered_rows["frame_01"], ordered_rows["frame_02"], ordered_rows["frame_03"]]
        )
        expected_confidence, _ = compute_session_confidence(naturally_ordered)

        # Row index 7 held frame_01 (position 0 once sorted), 13 held
        # frame_02 (position 1), 42 held frame_03 (position 2).
        self.assertAlmostEqual(confidence.loc[7], expected_confidence.iloc[0], places=9)
        self.assertAlmostEqual(confidence.loc[13], expected_confidence.iloc[1], places=9)
        self.assertAlmostEqual(confidence.loc[42], expected_confidence.iloc[2], places=9)


class TestConstants(unittest.TestCase):
    def test_min_temporal_confidence_is_a_valid_score_bound(self):
        self.assertGreater(MIN_TEMPORAL_CONFIDENCE, 0.0)
        self.assertLess(MIN_TEMPORAL_CONFIDENCE, 1.0)

    def test_stance_phase_indices_matches_zero_storage_pipeline_convention(self):
        # zero_storage_pipeline.py's validate_pose() treats phase_index in
        # [0, 1] (stance, trigger) as the lower-body-critical phases. This
        # module's stance-symmetry prior must apply to exactly those two.
        self.assertEqual(STANCE_PHASE_INDICES, {0, 1})

    def test_trajectory_joints_matches_kinematic_validator_bones_endpoints(self):
        # Regression test for a real review finding: TRAJECTORY_JOINTS used
        # to be a hand-retyped copy of kinematic_validator.py's BONES
        # endpoints, which could silently drift out of sync if BONES ever
        # changed. It is now derived from BONES directly.
        from kinematic_validator import BONES
        expected = sorted({joint for pair in BONES.values() for joint in pair})
        self.assertEqual(sorted(TRAJECTORY_JOINTS), expected)


class TestDuplicateFrameNameGuard(unittest.TestCase):
    def test_session_with_duplicated_frame_names_gets_nan_confidence_not_a_silently_wrong_score(self):
        # Regression test for a real bug found against the live dataset: 42
        # real sessions today have every frame_name duplicated (e.g. two
        # "01_stance" rows before the first "02_trigger" row). Treating
        # post-sort row POSITION as phase index in that case would silently
        # score a duplicate stance frame as if it were the trigger frame --
        # wrong, not a crash. Such sessions must get NaN, not a number that
        # looks valid but isn't.
        left = {
            "left_shoulder": (0.0, 0.0, 0.0), "left_hip": (0.0, 1.0, 0.0),
            "left_knee": (1.0, 1.0, 0.0), "left_ankle": (1.0, 2.0, 0.0),
            "left_foot_index": (2.0, 2.0, 0.0),
        }
        right = {
            "right_shoulder": (10.0, 0.0, 0.0), "right_hip": (10.0, 1.0, 0.0),
            "right_knee": (11.0, 1.0, 0.0), "right_ankle": (11.0, 2.0, 0.0),
            "right_foot_index": (12.0, 2.0, 0.0),
        }
        positions = {**left, **right}
        # frame_01 duplicated, frame_02 present once -- 3 rows, a duplicate.
        rows = [
            _frame_row("s1", "frame_01", positions),
            _frame_row("s1", "frame_01", positions),
            _frame_row("s1", "frame_02", positions),
        ]
        df = pd.DataFrame(rows)

        confidence, report = compute_session_confidence(df)

        self.assertTrue(np.all(np.isnan(confidence.values)))
        self.assertTrue(report["duplicate_frame_names"])
        # 0, not 3 (the frame count): these frames were never scored at
        # all, which is different from being scored and found below
        # threshold -- regression test for a review finding where this
        # field was set to n, silently double-counting NaN sessions if
        # anyone summed n_below_threshold across report_df.
        self.assertEqual(report["n_below_threshold"], 0)

    def test_session_without_duplicates_is_unaffected_and_flag_is_false(self):
        df = _static_session("s1", 7)
        confidence, report = compute_session_confidence(df)
        self.assertFalse(np.any(np.isnan(confidence.values)))
        self.assertFalse(report["duplicate_frame_names"])


if __name__ == "__main__":
    unittest.main()
