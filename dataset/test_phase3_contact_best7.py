"""
Tests for the Phase-3D contact signals, event search region and Best-7
plumbing.
"""

import unittest

import numpy as np

import phase3_best7_benchmark as B7
import phase3_contact_signals as CS


def frame(i, pose=None, bat=None, gm=1.0, lm=1.0, box=(0, 0, 50, 120), blur=100.0):
    return {"frame": i, "t": i / 30.0, "global_motion": gm, "local_motion": lm,
            "has_box": box is not None, "box": list(box) if box else None,
            "blur": blur, "pose": pose, "bat": bat}


def pose_at(lwr, rwr, torso=40.0, lift=0.0, vis=0.9, minvis=0.8):
    return {"torso": torso, "lwr": list(lwr), "rwr": list(rwr),
            "msh": [0, 0], "mhip": [0, 40], "mank": [0, 80],
            "wrist_lift": lift, "pose_visibility": vis, "min_visibility": minvis}


class TestGapHandling(unittest.TestCase):
    def test_missing_values_are_interpolated_not_zeroed(self):
        # A missing box is an absence of evidence, not evidence of stillness.
        out = CS._fill(np.array([2.0, np.nan, 4.0]))
        self.assertAlmostEqual(out[1], 3.0)

    def test_all_missing_returns_none(self):
        self.assertIsNone(CS._fill(np.array([np.nan, np.nan])))

    def test_normalise_handles_constant_signal(self):
        self.assertTrue(np.all(CS._norm(np.array([3.0, 3.0])) == 0.0))


class TestWristVelocity(unittest.TestCase):
    def test_zero_for_a_still_batsman(self):
        fs = [frame(i, pose_at((10, 10), (20, 10))) for i in range(6)]
        v = CS.wrist_velocity(fs)
        self.assertAlmostEqual(float(np.nanmax(v[1:])), 0.0, places=6)

    def test_peaks_where_the_wrist_moves_fastest(self):
        fs = [frame(i, pose_at((10, 10), (20, 10))) for i in range(6)]
        fs[3]["pose"] = pose_at((50, 10), (20, 10))     # big jump into frame 3
        v = CS.wrist_velocity(fs)
        self.assertEqual(int(np.nanargmax(v)), 3)

    def test_normalised_by_torso_so_camera_distance_does_not_matter(self):
        near = [frame(i, pose_at((0, 0), (0, 0), torso=80.0)) for i in range(4)]
        far = [frame(i, pose_at((0, 0), (0, 0), torso=40.0)) for i in range(4)]
        near[2]["pose"] = pose_at((16, 0), (0, 0), torso=80.0)
        far[2]["pose"] = pose_at((8, 0), (0, 0), torso=40.0)
        self.assertAlmostEqual(float(CS.wrist_velocity(near)[2]),
                               float(CS.wrist_velocity(far)[2]), places=6)


class TestBatAssociation(unittest.TestCase):
    def test_no_bat_gives_zero(self):
        fs = [frame(i) for i in range(4)]
        self.assertTrue(np.all(CS.bat_association(fs) == 0.0))

    def test_bat_off_the_person_scores_below_bat_on_the_person(self):
        off = [frame(0, bat={"conf": 0.9, "on_person_frac": 0.0, "cx": 900, "cy": 900})]
        on = [frame(0, bat={"conf": 0.9, "on_person_frac": 1.0, "cx": 10, "cy": 10})]
        self.assertLess(CS.bat_association(off)[0], CS.bat_association(on)[0])

    def test_bat_near_the_hand_scores_above_bat_far_from_it(self):
        p = pose_at((10, 10), (20, 10))
        near = [frame(0, p, {"conf": 0.9, "on_person_frac": 1.0, "cx": 10, "cy": 10})]
        far = [frame(0, p, {"conf": 0.9, "on_person_frac": 1.0, "cx": 300, "cy": 300})]
        self.assertGreater(CS.bat_association(near)[0], CS.bat_association(far)[0])


class TestSkeletonDynamics(unittest.TestCase):
    def test_rising_hands_produce_no_downswing_signal(self):
        fs = [frame(i, pose_at((0, 0), (0, 0), lift=i * 0.2)) for i in range(6)]
        self.assertAlmostEqual(float(np.max(CS.skeleton_dynamics(fs))), 0.0, places=6)

    def test_falling_hands_produce_a_positive_downswing_signal(self):
        fs = [frame(i, pose_at((0, 0), (0, 0), lift=1.0 - i * 0.2)) for i in range(6)]
        self.assertGreater(float(np.max(CS.skeleton_dynamics(fs))), 0.0)


class TestCombinedEvent(unittest.TestCase):
    def test_has_no_free_parameters_and_returns_a_full_series(self):
        fs = [frame(i, pose_at((i * 2, 0), (0, 0))) for i in range(8)]
        s = CS.combined_event(fs)
        self.assertEqual(len(s), 8)

    def test_estimate_returns_a_frame_inside_the_clip(self):
        fs = [frame(i, pose_at((i * 2, 0), (0, 0))) for i in range(8)]
        e = CS.estimate("C6_combined_event", fs)
        self.assertIsNotNone(e["frame"])
        self.assertTrue(0 <= e["frame"] < 8)

    def test_restrict_confines_the_search(self):
        fs = [frame(i, pose_at((0, 0), (0, 0)), gm=1.0, lm=1.0) for i in range(20)]
        fs[17]["global_motion"] = 50.0
        free = CS.estimate("C4_global_motion", fs)
        held = CS.estimate("C4_global_motion", fs, restrict=(0, 8))
        # A single-frame spike smoothed by a symmetric 3-window gives an exact
        # tie at 16/17/18, and argmax takes the lowest index — so assert
        # proximity, which is the property that matters, not the exact tie.
        self.assertLessEqual(abs(free["frame"] - 17), 1)
        self.assertLessEqual(held["frame"], 8)


class TestEventRegion(unittest.TestCase):
    def setUp(self):
        self.clip = {"n_frames": 120, "frames": [frame(i) for i in range(120)]}

    def test_region_is_the_union_and_is_wider_than_l3(self):
        l3 = {"rejected": False, "start": 40, "end": 60}
        lo, hi, src = B7.event_region(self.clip, l3, contact=90)
        self.assertLessEqual(lo, 40)
        self.assertGreaterEqual(hi, 90)
        self.assertEqual(src, "l3_union_contact")

    def test_contact_only_still_produces_a_region(self):
        lo, hi, _ = B7.event_region(self.clip, None, contact=50)
        self.assertLess(lo, 50)
        self.assertGreater(hi, 50)

    def test_l3_only_still_produces_a_region(self):
        lo, hi, _ = B7.event_region(self.clip, {"rejected": False, "start": 30,
                                                "end": 55}, contact=None)
        self.assertLessEqual(lo, 30)
        self.assertGreaterEqual(hi, 55)

    def test_no_evidence_falls_back_to_the_whole_clip(self):
        lo, hi, src = B7.event_region(self.clip, {"rejected": True}, None)
        self.assertEqual((lo, hi), (0, 119))
        self.assertEqual(src, "whole_clip_fallback")

    def test_region_never_leaves_the_clip(self):
        lo, hi, _ = B7.event_region(self.clip, {"rejected": False, "start": 0,
                                                "end": 119}, contact=119)
        self.assertGreaterEqual(lo, 0)
        self.assertLessEqual(hi, 119)

    def test_a_degenerate_region_is_widened_to_something_usable(self):
        lo, hi, _ = B7.event_region(self.clip, {"rejected": False, "start": 50,
                                                "end": 51}, contact=None)
        self.assertGreaterEqual(hi - lo, 14)


class TestCandidatePool(unittest.TestCase):
    def test_pool_excludes_frames_without_a_batsman_box(self):
        frames = [frame(i, box=(0, 0, 50, 120) if i % 2 == 0 else None)
                  for i in range(40)]
        clip = {"n_frames": 40, "frames": frames}
        pool = B7.build_pool(clip, (0, 39))
        self.assertTrue(all(f % 2 == 0 for f in (c.frame for c in pool)))

    def test_pool_is_capped_at_the_target(self):
        clip = {"n_frames": 200, "frames": [frame(i) for i in range(200)]}
        self.assertLessEqual(len(B7.build_pool(clip, (0, 199), target=40)), 40)

    def test_quality_and_motion_are_separate_fields(self):
        # A sharp, still frame and a blurred, fast one must be distinguishable.
        frames = [frame(0, pose_at((0, 0), (0, 0)), blur=500.0, lm=0.0),
                  frame(1, pose_at((0, 0), (0, 0)), blur=5.0, lm=90.0)]
        clip = {"n_frames": 2, "frames": frames}
        pool = B7.build_pool(clip, (0, 1))
        self.assertGreater(pool[0].sharpness, pool[1].sharpness)
        self.assertLess(pool[0].motion, pool[1].motion)

    def test_occlusion_uses_minimum_visibility_not_the_mean(self):
        frames = [frame(0, pose_at((0, 0), (0, 0), vis=0.9, minvis=0.1))]
        pool = B7.build_pool({"n_frames": 1, "frames": frames}, (0, 0))
        self.assertAlmostEqual(pool[0].occlusion, 0.9, places=6)

    def test_empty_region_gives_an_empty_pool(self):
        clip = {"n_frames": 10, "frames": [frame(i, box=None) for i in range(10)]}
        self.assertEqual(B7.build_pool(clip, (0, 9)), [])


class TestSequenceMetrics(unittest.TestCase):
    def _pool(self, idx):
        import phase3_frame_selection as FS
        return [FS.Candidate(frame=i, t=i / 30.0, pose_quality=0.8,
                             extra={"has_pose": True}) for i in idx]

    def test_detects_duplicates(self):
        m = B7.sequence_metrics(self._pool([1, 2, 2, 4, 5, 6, 7]), 4, (0, 20), 30)
        self.assertEqual(m["duplicates"], 1)

    def test_chronology_and_min_gap(self):
        m = B7.sequence_metrics(self._pool([0, 5, 10, 15, 20, 25, 30]), 15, (0, 40), 30)
        self.assertTrue(m["chronological"])
        self.assertEqual(m["min_gap"], 5)

    def test_pre_and_post_contact_counts(self):
        m = B7.sequence_metrics(self._pool([0, 2, 4, 10, 16, 18, 20]), 10, (0, 30), 30)
        self.assertEqual(m["pre_contact"], 3)
        self.assertEqual(m["post_contact"], 3)

    def test_contact_adjacency_uses_a_three_frame_window(self):
        m = B7.sequence_metrics(self._pool([0, 5, 9, 10, 11, 20, 30]), 10, (0, 40), 30)
        self.assertEqual(m["contact_adjacent"], 3)

    def test_flags_frames_outside_the_region(self):
        m = B7.sequence_metrics(self._pool([0, 5, 10, 15, 20, 25, 99]), 10, (0, 30), 30)
        self.assertFalse(m["inside_region"])


class TestHumanReference(unittest.TestCase):
    def test_reference_is_chronological_and_inside_the_clip(self):
        import phase3_best7_audit as AU
        ref = AU.reference_sequence(10, 40, 25, 120)
        self.assertEqual(len(ref), 7)
        self.assertTrue(all(b > a for a, b in zip(ref, ref[1:])))
        self.assertTrue(all(0 <= f < 120 for f in ref))

    def test_reference_needs_a_contact_frame(self):
        import phase3_best7_audit as AU
        self.assertIsNone(AU.reference_sequence(10, 40, None, 120))

    def test_distance_is_zero_for_an_identical_sequence(self):
        import phase3_best7_audit as AU
        s = [1, 5, 9, 13, 17, 21, 25]
        self.assertEqual(AU.seq_distance(s, s), 0.0)

    def test_coverage_counts_matches_within_tolerance(self):
        import phase3_best7_audit as AU
        self.assertAlmostEqual(AU.coverage_overlap([1, 5, 9], [2, 6, 40], tol=3),
                               2 / 3, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
