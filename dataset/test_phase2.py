import unittest

from phase2_detectors import Detection, iou
from phase2_batsman_score import (track_features, select_batsman, score_batsman_track,
                                  baseline_largest, baseline_size_dominance,
                                  baseline_most_central, baseline_most_persistent,
                                  CONFIDENT_BATSMAN, AMBIGUOUS, NO_BATSMAN)
import phase2_ablation as ABL
from phase2_annotate import split_for, sample_frame_indices


def obs(frames, box, conf=0.9):
    """Track observations with a fixed box."""
    return [(f, box, conf) for f in frames]


def moving(frames, x0, y0, w, h, dx=0.0, dy=0.0, dh=0.0):
    """Track that translates and/or changes scale over time."""
    out = []
    for i, f in enumerate(frames):
        hh = h + dh * i
        out.append((f, (x0 + dx * i, y0 + dy * i, x0 + dx * i + w, y0 + dy * i + hh), 0.9))
    return out


CTX = {"frame_w": 640, "frame_h": 360, "frames_processed": 6}
FRAMES = [0, 1, 2, 3, 4, 5]


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self):
        d = Detection(0, 0, 10, 10, 1.0)
        self.assertAlmostEqual(iou(d, d), 1.0)

    def test_disjoint_boxes(self):
        self.assertEqual(iou(Detection(0, 0, 10, 10, 1.0), (20, 20, 30, 30)), 0.0)

    def test_accepts_plain_tuple_ground_truth(self):
        d = Detection(0, 0, 10, 10, 1.0)
        self.assertAlmostEqual(iou(d, (0, 0, 10, 10)), 1.0)

    def test_half_overlap(self):
        # Union 150, intersection 50 -> 1/3
        self.assertAlmostEqual(iou(Detection(0, 0, 10, 10, 1.0), (5, 0, 15, 10)), 1 / 3, places=5)


class TestTrackFeatures(unittest.TestCase):
    def test_stationary_track_is_maximally_stable(self):
        f = track_features(obs(FRAMES, (100, 100, 140, 200)), 640, 360, 6)
        self.assertAlmostEqual(f["foot_stability"], 1.0)
        self.assertAlmostEqual(f["scale_stability"], 1.0)
        self.assertAlmostEqual(f["centroid_stability"], 1.0)

    def test_approaching_subject_loses_scale_stability(self):
        """The Phase-1 bowler failure: someone running at the camera grows.
        Scale stability must fall for them and not for a batsman."""
        batsman = track_features(obs(FRAMES, (100, 100, 140, 200)), 640, 360, 6)
        bowler = track_features(moving(FRAMES, 100, 60, 40, 60, dh=30), 640, 360, 6)
        self.assertLess(bowler["scale_stability"], batsman["scale_stability"])

    def test_traversing_subject_loses_foot_stability(self):
        stayer = track_features(obs(FRAMES, (100, 100, 140, 200)), 640, 360, 6)
        walker = track_features(moving(FRAMES, 100, 100, 40, 100, dy=25), 640, 360, 6)
        self.assertLess(walker["foot_stability"], stayer["foot_stability"])

    def test_features_are_scale_invariant(self):
        """Same motion in own-body units must score the same near or far --
        otherwise every feature becomes a proxy for camera distance, which is
        the exact error Phase 1 made."""
        near = track_features(moving(FRAMES, 0, 0, 80, 200, dy=20), 640, 360, 6)
        far = track_features(moving(FRAMES, 0, 0, 40, 100, dy=10), 640, 360, 6)
        self.assertAlmostEqual(near["foot_stability"], far["foot_stability"], places=5)
        self.assertAlmostEqual(near["centroid_stability"], far["centroid_stability"], places=5)

    def test_coverage_is_bounded(self):
        f = track_features(obs(FRAMES * 3, (0, 0, 10, 10)), 640, 360, 6)
        self.assertLessEqual(f["coverage"], 1.0)

    def test_empty_track_does_not_crash(self):
        f = track_features([], 640, 360, 6)
        self.assertEqual(f["coverage"], 0.0)


class TestBaselines(unittest.TestCase):
    def setUp(self):
        # t0 = small distant batsman, t1 = large near bowler.
        self.tracks = {
            0: obs(FRAMES, (300, 120, 330, 220)),
            1: obs(FRAMES, (20, 40, 180, 350)),
        }

    def test_largest_picks_the_near_bowler(self):
        """Documents the failure, so a regression here is visible: 'largest'
        is exactly the rule that selects the wrong person in front view."""
        self.assertEqual(baseline_largest(self.tracks, CTX), 1)

    def test_most_persistent_is_a_tie_here(self):
        self.assertIn(baseline_most_persistent(self.tracks, CTX), (0, 1))

    def test_size_dominance_refuses_without_a_clear_winner(self):
        equal = {0: obs(FRAMES, (0, 0, 100, 200)), 1: obs(FRAMES, (200, 0, 300, 200))}
        self.assertIsNone(baseline_size_dominance(equal, CTX))

    def test_size_dominance_commits_when_one_dominates(self):
        self.assertEqual(baseline_size_dominance(self.tracks, CTX), 1)

    def test_most_central(self):
        self.assertEqual(baseline_most_central(self.tracks, CTX), 0)

    def test_baselines_handle_empty(self):
        for fn in (baseline_largest, baseline_size_dominance,
                   baseline_most_central, baseline_most_persistent):
            self.assertIsNone(fn({}, CTX))


class TestSelectionRefusesWithoutModel(unittest.TestCase):
    def test_unfitted_model_refuses_rather_than_guessing(self):
        r = select_batsman({0: obs(FRAMES, (0, 0, 10, 10))}, CTX, None)
        self.assertEqual(r["verdict"], AMBIGUOUS)
        self.assertIsNone(r["track_id"])

    def test_score_without_model_returns_components_and_no_score(self):
        s = score_batsman_track(obs(FRAMES, (0, 0, 10, 10)), CTX, 0, None)
        self.assertIsNone(s["score"])
        self.assertIn("coverage", s["components"])

    def test_no_tracks_is_no_batsman(self):
        self.assertEqual(select_batsman({}, CTX, None)["verdict"], NO_BATSMAN)


class TestSelectionWithModel(unittest.TestCase):
    """A model that keys entirely on foot_stability, to exercise the verdict
    logic without depending on the fitted coefficients."""
    MODEL = {"coef": {"foot_stability": 20.0}, "intercept": -15.0}

    def test_confident_when_one_track_clearly_wins(self):
        tracks = {0: obs(FRAMES, (300, 120, 330, 220)),
                  1: moving(FRAMES, 20, 40, 160, 200, dy=40)}
        r = select_batsman(tracks, CTX, self.MODEL)
        self.assertEqual(r["verdict"], CONFIDENT_BATSMAN)
        self.assertEqual(r["track_id"], 0)

    def test_ambiguous_when_two_tracks_score_alike(self):
        tracks = {0: obs(FRAMES, (300, 120, 330, 220)),
                  1: obs(FRAMES, (20, 120, 50, 220))}
        r = select_batsman(tracks, CTX, self.MODEL)
        self.assertEqual(r["verdict"], AMBIGUOUS)
        self.assertIsNone(r["track_id"], "an ambiguous verdict must not name a track")

    def test_low_scores_give_no_batsman(self):
        tracks = {0: moving(FRAMES, 0, 0, 40, 100, dy=60)}
        r = select_batsman(tracks, CTX, self.MODEL)
        self.assertEqual(r["verdict"], NO_BATSMAN)

    def test_model_uses_only_declared_features(self):
        """A geometry-only model declares a subset; scoring must not require
        the full feature list."""
        s = score_batsman_track(obs(FRAMES, (0, 0, 10, 10)), CTX, 0, self.MODEL)
        self.assertIsNotNone(s["score"])


class TestAblationOutcomeAccounting(unittest.TestCase):
    """The bug this guards against materially changed the reported result:
    bucketing 'detector never found the batsman' as excluded rather than
    wrong made the MediaPipe baseline look like it had a 0% wrong-person
    rate."""

    def test_pick_with_no_gt_track_is_wrong_not_excluded(self):
        self.assertEqual(ABL._outcome(pick=3, gt_tid=None), "wrong")

    def test_refusal_is_not_wrong(self):
        self.assertEqual(ABL._outcome(pick=None, gt_tid=None), "refused")
        self.assertEqual(ABL._outcome(pick=None, gt_tid=2), "refused")

    def test_correct_and_wrong_picks(self):
        self.assertEqual(ABL._outcome(pick=2, gt_tid=2), "correct")
        self.assertEqual(ABL._outcome(pick=1, gt_tid=2), "wrong")


class TestAnnotationSplit(unittest.TestCase):
    def test_split_is_stable_and_deterministic(self):
        self.assertEqual(split_for("a.mp4"), split_for("a.mp4"))
        self.assertIn(split_for("a.mp4"), ("dev", "eval"))

    def test_split_uses_both_buckets(self):
        splits = {split_for(f"clip_{i}.mp4") for i in range(50)}
        self.assertEqual(splits, {"dev", "eval"})

    def test_sampled_frames_are_sorted_unique_and_in_range(self):
        idx = sample_frame_indices(90, 6)
        self.assertEqual(len(idx), 6)
        self.assertEqual(idx, sorted(idx))
        self.assertTrue(all(0 <= i < 90 for i in idx))

    def test_very_short_clip_does_not_crash(self):
        self.assertTrue(all(i >= 0 for i in sample_frame_indices(1, 6)))


class TestAssociation(unittest.TestCase):
    def test_stationary_person_forms_one_track(self):
        dets = [[Detection(100, 100, 140, 200, 0.9)] for _ in FRAMES]
        tracks = ABL.associate(dets, FRAMES)
        self.assertEqual(len(tracks), 1)

    def test_two_separated_people_form_two_tracks(self):
        dets = [[Detection(0, 0, 40, 100, 0.9), Detection(500, 200, 540, 300, 0.9)]
                for _ in FRAMES]
        tracks = ABL.associate(dets, FRAMES)
        self.assertEqual(len(tracks), 2)

    def test_empty_detections_yield_no_tracks(self):
        self.assertEqual(ABL.associate([[] for _ in FRAMES], FRAMES), {})


if __name__ == "__main__":
    unittest.main()
