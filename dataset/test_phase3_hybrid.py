"""
Tests for the H0/H1 hybrid: the fallback rule, the landmark adapter that
feeds RTMPose output into the existing semantic-feature code, and the
null-box safeguards that must survive the change.
"""

import unittest

import phase3_pose_models as PM
import phase3_semantic_features as SF


class FakeRTM:
    """Stands in for RTMPosePose. Records how often it was asked."""

    def __init__(self, result=None):
        self.calls = 0
        self.result = result

    def infer(self, frame, box):
        self.calls += 1
        return self.result if self.result is not None else PM.PoseResult()


def upright_pose(x0=100, y0=100, scale=1.0):
    """A plausible skeleton in absolute FRAME pixels."""
    base = {
        "nose": (0, -80), "l_shoulder": (-20, -60), "r_shoulder": (20, -60),
        "l_elbow": (-20, -30), "r_elbow": (20, -30),
        "l_wrist": (-20, 0), "r_wrist": (20, 0),
        "l_hip": (-15, 0), "r_hip": (15, 0),
        "l_knee": (-15, 40), "r_knee": (15, 40),
        "l_ankle": (-15, 80), "r_ankle": (15, 80),
        "l_foot_index": (-5, 90), "r_foot_index": (5, 90),
    }
    pts = {k: (x0 + v[0] * scale, y0 + v[1] * scale) for k, v in base.items()}
    return PM.PoseResult(pts, {k: 0.8 for k in pts})


class TestFallbackExecution(unittest.TestCase):
    def test_fallback_runs_when_mediapipe_returns_nothing(self):
        rtm = FakeRTM(upright_pose())
        st = SF.PoseFallbackStats()
        pm = SF.resolve_pose(None, rtm, "frame", (0, 0, 200, 200),
                             (0, 0, 200, 200), st)
        self.assertEqual(rtm.calls, 1)
        self.assertEqual(st.n_fallback_calls, 1)
        self.assertEqual(st.n_fallback_success, 1)
        self.assertIsNotNone(pm)

    def test_fallback_does_NOT_run_when_mediapipe_succeeds(self):
        # The whole point of the refusal trigger: no extra cost, and no
        # second opinion, when the primary model answered.
        rtm = FakeRTM(upright_pose())
        st = SF.PoseFallbackStats()
        primary = {"torso": 60.0}
        pm = SF.resolve_pose(primary, rtm, "frame", (0, 0, 200, 200),
                             (0, 0, 200, 200), st)
        self.assertEqual(rtm.calls, 0)
        self.assertEqual(st.n_fallback_calls, 0)
        self.assertIs(pm, primary)

    def test_h0_never_calls_the_fallback(self):
        # rtm=None is the H0 condition; it must be a no-op, not a crash.
        st = SF.PoseFallbackStats()
        self.assertIsNone(SF.resolve_pose(None, None, "f", (0, 0, 1, 1),
                                          (0, 0, 1, 1), st))
        self.assertEqual(st.n_fallback_calls, 0)

    def test_failed_fallback_counts_the_call_but_not_a_success(self):
        rtm = FakeRTM(PM.PoseResult())          # RTMPose returned nothing
        st = SF.PoseFallbackStats()
        pm = SF.resolve_pose(None, rtm, "f", (0, 0, 200, 200),
                             (0, 0, 200, 200), st)
        self.assertIsNone(pm)
        self.assertEqual(st.n_fallback_calls, 1)
        self.assertEqual(st.n_fallback_success, 0)


class TestFallbackOutputFeedsSemanticFeatures(unittest.TestCase):
    """The adapter must produce something _pose_metrics can actually consume,
    otherwise the fallback silently contributes nothing."""

    def test_adapter_output_produces_usable_pose_metrics(self):
        box = (50, 20, 150, 220)
        lms = PM.to_mediapipe_landmarks(upright_pose(), box)
        pm = SF._pose_metrics(lms, box)
        self.assertIsNotNone(pm)
        for key in ("torso", "wrist_mid", "lwr", "rwr", "hip_mid",
                    "ankle_mid", "sh_angle", "wrist_lift", "vis"):
            self.assertIn(key, pm)
        self.assertGreater(pm["torso"], 0.0)

    def test_adapter_places_joints_at_mediapipe_indices(self):
        box = (0, 0, 200, 200)
        pose = upright_pose(x0=100, y0=100)
        lms = PM.to_mediapipe_landmarks(pose, box)
        self.assertEqual(len(lms), 33)
        # l_shoulder is MediaPipe index 11 and sits at (80, 40) in the frame.
        self.assertAlmostEqual(lms[11].x, 80 / 200, places=6)
        self.assertAlmostEqual(lms[11].y, 40 / 200, places=6)

    def test_adapter_reports_scores_as_visibility(self):
        lms = PM.to_mediapipe_landmarks(upright_pose(), (0, 0, 200, 200))
        self.assertAlmostEqual(lms[11].visibility, 0.8, places=6)

    def test_adapter_returns_none_for_a_refused_pose(self):
        self.assertIsNone(PM.to_mediapipe_landmarks(PM.PoseResult(),
                                                    (0, 0, 10, 10)))
        self.assertIsNone(PM.to_mediapipe_landmarks(None, (0, 0, 10, 10)))

    def test_adapter_returns_none_for_a_degenerate_box(self):
        self.assertIsNone(PM.to_mediapipe_landmarks(upright_pose(),
                                                    (10, 10, 10, 10)))

    def test_rtmpose_is_2d_so_adapter_reports_zero_depth(self):
        # Guards the constraint that keeps RTMPose out of the 3D scoring path.
        lms = PM.to_mediapipe_landmarks(upright_pose(), (0, 0, 200, 200))
        self.assertEqual(lms[11].z, 0.0)


class TestNullBoxSafeguardsRetained(unittest.TestCase):
    """RTMPose returns a skeleton for ANY box, so the safeguards that stop
    that becoming evidence must not have been lost in the refactor."""

    def test_pose_metrics_still_rejects_a_degenerate_box(self):
        lms = PM.to_mediapipe_landmarks(upright_pose(), (0, 0, 200, 200))
        self.assertIsNone(SF._pose_metrics(lms, (5, 5, 5, 5)))

    def test_pose_metrics_still_rejects_a_collapsed_skeleton(self):
        # All joints coincident -> zero torso -> unusable, not a divide-by-zero.
        flat = PM.PoseResult({k: (10.0, 10.0) for k in PM.CANONICAL},
                             {k: 0.9 for k in PM.CANONICAL})
        lms = PM.to_mediapipe_landmarks(flat, (0, 0, 100, 100))
        self.assertIsNone(SF._pose_metrics(lms, (0, 0, 100, 100)))

    def test_pose_metrics_rejects_none_landmarks(self):
        self.assertIsNone(SF._pose_metrics(None, (0, 0, 100, 100)))


class TestBatAssociationUnchangedByRefactor(unittest.TestCase):
    def test_no_bats_gives_an_empty_event(self):
        e = SF._bat_event([], (0, 0, 100, 200), None, 0, 0)
        self.assertEqual(e, {"any": False, "on_person": False,
                             "near_hand": False, "conf": 0.0})

    def test_bat_off_the_person_is_seen_but_not_associated(self):
        e = SF._bat_event([((500, 500, 520, 560), 0.9)], (0, 0, 100, 200),
                          None, 0, 0)
        self.assertTrue(e["any"])
        self.assertFalse(e["on_person"])

    def test_bat_on_person_without_pose_cannot_be_near_hand(self):
        # near_hand needs wrists; with no pose it must stay False rather than
        # defaulting to True.
        e = SF._bat_event([((10, 10, 90, 190), 0.7)], (0, 0, 100, 200),
                          None, 0, 0)
        self.assertTrue(e["on_person"])
        self.assertFalse(e["near_hand"])
        self.assertAlmostEqual(e["conf"], 0.7)

    def test_bat_at_the_wrist_is_near_hand(self):
        box = (0, 0, 200, 200)
        pm = SF._pose_metrics(PM.to_mediapipe_landmarks(upright_pose(), box), box)
        # upright_pose puts wrists at frame (80,100) and (120,100).
        e = SF._bat_event([((70, 90, 90, 110), 0.8)], box, pm, 0, 0)
        self.assertTrue(e["on_person"])
        self.assertTrue(e["near_hand"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
