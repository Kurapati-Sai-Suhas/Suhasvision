"""
Tests for the pose-comparison maths.

These cover the parts that decide the RTMPose experiment's numbers and that
would fail silently if wrong: the canonical joint mapping (a wrong index maps
a skeleton onto the wrong limbs and still produces plausible angles), the
15-angle reconstruction, and the ground-truth-free quality measures.
"""

import math
import unittest

import phase3_pose_models as PM


class FakePose(PM.PoseResult):
    pass


def upright(scale=1.0, dx=0.0, dy=0.0, z=0.0):
    """A simple anatomically sane standing figure, in pixels."""
    P = {
        "nose": (0, -80), "l_shoulder": (-20, -60), "r_shoulder": (20, -60),
        "l_elbow": (-20, -30), "r_elbow": (20, -30),
        "l_wrist": (-20, 0), "r_wrist": (20, 0),
        "l_hip": (-15, 0), "r_hip": (15, 0),
        "l_knee": (-15, 40), "r_knee": (15, 40),
        "l_ankle": (-15, 80), "r_ankle": (15, 80),
        "l_foot_index": (-5, 90), "r_foot_index": (5, 90),
    }
    pts = {k: (v[0] * scale + dx, v[1] * scale + dy) for k, v in P.items()}
    p3 = {k: (v[0], v[1], z) for k, v in pts.items()}
    return PM.PoseResult(pts, {k: 0.9 for k in pts}, p3)


class TestJointMapping(unittest.TestCase):
    def test_both_models_expose_the_same_canonical_joints(self):
        # If these ever diverge, the two models are silently being compared
        # on different feature sets.
        self.assertEqual(set(PM.MEDIAPIPE_IDX), set(PM.CANONICAL))
        self.assertEqual(set(PM.HALPE26_IDX), set(PM.CANONICAL))

    def test_halpe26_indices_are_within_26_keypoints(self):
        for name, i in PM.HALPE26_IDX.items():
            self.assertLess(i, 26, name)

    def test_halpe26_carries_feet(self):
        # The whole reason halpe26 was chosen over COCO-17: angle_ankle_L/R
        # needs foot_index, which COCO-17 does not have.
        self.assertIn("l_foot_index", PM.HALPE26_IDX)
        self.assertIn("r_foot_index", PM.HALPE26_IDX)

    def test_variants_all_declare_an_input_size(self):
        for k, (url, size) in PM.RTMPOSE_VARIANTS.items():
            self.assertTrue(url.endswith(".zip"), k)
            self.assertEqual(len(size), 2, k)


class TestFifteenAngles(unittest.TestCase):
    def test_produces_exactly_fifteen_named_angles(self):
        a = PM.fifteen_angles(upright())
        self.assertEqual(len(a), 15)

    def test_names_match_production_feature_engineering(self):
        expected = {
            "angle_knee_L", "angle_knee_R", "angle_hip_L", "angle_hip_R",
            "angle_elbow_L", "angle_elbow_R", "angle_shoulder_L",
            "angle_shoulder_R", "angle_ankle_L", "angle_ankle_R",
            "angle_trunk_L", "angle_trunk_R", "angle_arm_L", "angle_arm_R",
            "angle_head_tilt"}
        self.assertEqual(set(PM.fifteen_angles(upright())), expected)

    def test_straight_leg_is_about_180_degrees(self):
        a = PM.fifteen_angles(upright())
        self.assertAlmostEqual(a["angle_knee_L"], 180.0, delta=1.0)

    def test_bent_elbow_is_about_90_degrees(self):
        p = upright()
        p.pts["l_wrist"] = (-50, -30)      # forearm horizontal
        a = PM.fifteen_angles(p)
        self.assertAlmostEqual(a["angle_elbow_L"], 90.0, delta=1.0)

    def test_angles_are_scale_and_translation_invariant(self):
        # Not EXACTLY invariant, and deliberately so: production's
        # calculate_angle() divides by (mag1*mag2 + 1e-6), and that epsilon is
        # absolute, so its relative effect shrinks as the subject gets bigger.
        # The residual is ~0.001 deg at these scales -- far below anything the
        # feature set cares about, and reproducing production exactly matters
        # more than mathematical purity here.
        a = PM.fifteen_angles(upright())
        b = PM.fifteen_angles(upright(scale=2.5, dx=300, dy=-120))
        for k in a:
            self.assertAlmostEqual(a[k], b[k], delta=0.01, msg=k)

    def test_missing_joint_yields_none_not_a_wrong_number(self):
        p = upright()
        del p.pts["l_foot_index"]
        a = PM.fifteen_angles(p)
        self.assertIsNone(a["angle_ankle_L"])
        self.assertIsNotNone(a["angle_ankle_R"])

    def test_no_pose_returns_none(self):
        self.assertIsNone(PM.fifteen_angles(PM.PoseResult()))
        self.assertIsNone(PM.fifteen_angles(None))


class TestQualityMeasures(unittest.TestCase):
    def test_rigid_body_has_near_zero_bone_variation(self):
        seq = [upright(dx=10 * i) for i in range(8)]
        self.assertLess(PM.bone_length_cv(seq), 1e-6)

    def test_scaling_alone_does_not_inflate_bone_cv(self):
        # Bones are normalised by torso length, so a subject walking toward
        # the camera must not register as landmark noise.
        seq = [upright(scale=1.0 + 0.1 * i) for i in range(8)]
        self.assertLess(PM.bone_length_cv(seq), 1e-6)

    def test_jittered_body_scores_worse_than_still_body(self):
        import random
        rng = random.Random(0)
        still = [upright() for _ in range(8)]
        noisy = []
        for _ in range(8):
            p = upright()
            p.pts = {k: (v[0] + rng.uniform(-6, 6), v[1] + rng.uniform(-6, 6))
                     for k, v in p.pts.items()}
            noisy.append(p)
        self.assertLess(PM.bone_length_cv(still), PM.bone_length_cv(noisy))
        self.assertLess(PM.jitter(still), PM.jitter(noisy))

    def test_jitter_is_zero_for_a_motionless_sequence(self):
        self.assertAlmostEqual(PM.jitter([upright() for _ in range(5)]), 0.0,
                               places=9)

    def test_translation_registers_as_jitter_in_torso_units(self):
        seq = [upright(dx=0), upright(dx=60)]   # torso is 60px in `upright`
        self.assertAlmostEqual(PM.jitter(seq), 1.0, delta=0.05)


class TestFeatureQuality(unittest.TestCase):
    def test_clean_sequence_has_no_invalid_or_implausible_angles(self):
        q = PM.feature_quality([upright() for _ in range(6)])
        self.assertEqual(q["n_frames_with_angles"], 6)
        self.assertEqual(q["invalid_rate"], 0.0)
        self.assertEqual(q["velocity_discontinuity"], 0.0)

    def test_missing_joints_are_counted_as_invalid(self):
        seq = []
        for _ in range(6):
            p = upright()
            del p.pts["l_foot_index"]
            seq.append(p)
        self.assertGreater(PM.feature_quality(seq)["invalid_rate"], 0.0)

    def test_large_angle_jumps_are_counted_as_discontinuities(self):
        a = upright()
        b = upright()
        b.pts["l_wrist"] = (-20, -60)      # elbow snaps fully closed
        q = PM.feature_quality([a, b, a, b])
        self.assertGreater(q["velocity_discontinuity"], 0.0)

    def test_empty_sequence_reports_nothing_rather_than_crashing(self):
        q = PM.feature_quality([None, None])
        self.assertEqual(q["n_frames_with_angles"], 0)
        self.assertIsNone(q["invalid_rate"])


class TestZContribution(unittest.TestCase):
    def test_planar_pose_has_no_z_contribution(self):
        seq = [upright(z=0.0) for _ in range(4)]
        self.assertAlmostEqual(PM.z_contribution(seq)["median_deg"], 0.0,
                               places=6)

    def test_single_out_of_plane_joint_moves_the_max_not_the_median(self):
        p = upright()
        # Push one wrist out of the image plane; the 2D angle cannot see it.
        p.pts3["l_wrist"] = (p.pts3["l_wrist"][0], p.pts3["l_wrist"][1], 60.0)
        z = PM.z_contribution([p])
        # Only angle_elbow_L involves that joint, so 9 of the 10 tracked
        # angles are unchanged and the median stays at 0. That is the metric
        # behaving correctly -- it is deliberately robust to a single joint.
        self.assertGreater(z["max_deg"], 0.0)
        self.assertEqual(z["median_deg"], 0.0)

    def test_broadly_out_of_plane_pose_moves_the_median(self):
        # Giving every joint the SAME depth does nothing: the triangles stay
        # planar and parallel to the image plane, so the 3D angle equals the
        # 2D one. Depth only changes an angle when the three points differ in
        # z, so each joint gets its own depth here.
        p = upright()
        for n, j in enumerate(sorted(p.pts3)):
            x, y, _ = p.pts3[j]
            p.pts3[j] = (x, y, 18.0 * (n % 5))
        self.assertGreater(PM.z_contribution([p])["median_deg"], 0.0)

    def test_returns_none_when_the_model_has_no_depth(self):
        # RTMPose is 2D: pts3 is empty, so there is nothing to compare.
        flat = PM.PoseResult(upright().pts, {}, None)
        self.assertIsNone(PM.z_contribution([flat]))


class TestPoseResult(unittest.TestCase):
    def test_visible_counts_only_joints_above_threshold(self):
        p = upright()
        p.scores["l_wrist"] = 0.1
        self.assertEqual(p.visible(0.3), len(PM.CANONICAL) - 1)

    def test_torso_length_is_shoulder_to_hip_midpoint_distance(self):
        self.assertAlmostEqual(upright().torso(), 60.0, delta=1e-6)

    def test_empty_result_is_not_ok(self):
        self.assertFalse(PM.PoseResult().ok)
        self.assertEqual(PM.PoseResult().torso(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
