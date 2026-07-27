import unittest
from collections import namedtuple

from subject_selection import (
    GATE_TORSOS_MAX,
    GATE_TORSOS_PER_FRAME,
    MAX_POSE_CANDIDATES,
    MIN_TRACK_COVERAGE,
    SIZE_DOMINANCE_RATIO,
    build_tracks,
    pose_center_and_scale,
    select_subject,
)

Landmark = namedtuple("Landmark", ["x", "y", "z", "visibility", "presence"])

N_FRAMES = 7


def make_pose(cx, cy, torso=0.2, visibility=1.0):
    """A 33-landmark pose with hips straddling (cx, cy), shoulders `torso`
    above, and knees/ankles below in valid topological order (so these poses
    also pass validate_pose's stance-phase Y-ordering in integration tests)."""
    lms = [Landmark(cx, cy, 0.0, visibility, 1.0) for _ in range(33)]
    lms[11] = Landmark(cx - 0.05, cy - torso, 0.0, visibility, 1.0)  # left_shoulder
    lms[12] = Landmark(cx + 0.05, cy - torso, 0.0, visibility, 1.0)  # right_shoulder
    lms[23] = Landmark(cx - 0.05, cy, 0.0, visibility, 1.0)          # left_hip
    lms[24] = Landmark(cx + 0.05, cy, 0.0, visibility, 1.0)          # right_hip
    lms[25] = Landmark(cx - 0.05, cy + torso * 0.75, 0.0, visibility, 1.0)  # left_knee
    lms[26] = Landmark(cx + 0.05, cy + torso * 0.75, 0.0, visibility, 1.0)  # right_knee
    lms[27] = Landmark(cx - 0.05, cy + torso * 1.5, 0.0, visibility, 1.0)   # left_ankle
    lms[28] = Landmark(cx + 0.05, cy + torso * 1.5, 0.0, visibility, 1.0)   # right_ankle
    return lms


def _selected_center_x(selected, frame_idx=0):
    pose = selected[frame_idx]
    return (pose[23].x + pose[24].x) / 2.0


class TestPoseDescriptor(unittest.TestCase):
    def test_center_and_scale(self):
        cx, cy, torso = pose_center_and_scale(make_pose(0.5, 0.6, torso=0.2))
        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.6)
        self.assertAlmostEqual(torso, 0.2)


class TestConstantsLinkage(unittest.TestCase):
    def test_coverage_threshold_matches_pipeline_survival_rule(self):
        # apply_pipeline_rules hard-rejects sessions with >= 4 of 7 frames
        # missing, i.e. survivors have >= 4 present. Selecting a subject the
        # rules would reject anyway is pointless — the two must stay in sync.
        self.assertEqual(MIN_TRACK_COVERAGE, 4)

    def test_candidate_cap_allows_multi_person_scenes(self):
        self.assertGreaterEqual(MAX_POSE_CANDIDATES, 2)


class TestSingleSubject(unittest.TestCase):
    def test_one_visible_player_selected_every_frame(self):
        frames = [[make_pose(0.5, 0.5)] for _ in range(N_FRAMES)]
        selected, report = select_subject(frames)
        self.assertIsNotNone(selected)
        self.assertEqual(report["reason"], "selected")
        self.assertEqual(report["n_tracks"], 1)
        self.assertFalse(report["multi_track"])
        self.assertTrue(all(pose is not None for pose in selected))

    def test_confidence_fluctuations_do_not_affect_selection(self):
        # Selection is geometric (center + torso), deliberately independent
        # of visibility scores, which oscillate frame-to-frame in real
        # footage — a subject must not be dropped because MediaPipe's
        # confidence dipped.
        visibilities = [0.9, 0.2, 0.7, 0.1, 0.8, 0.3, 0.6]
        frames = [[make_pose(0.5, 0.5, visibility=v)] for v in visibilities]
        selected, report = select_subject(frames)
        self.assertIsNotNone(selected)
        self.assertEqual(report["winner_coverage"], N_FRAMES)

    def test_moving_subject_within_gate_stays_one_track(self):
        # A real swing moves the hips between phases — drift of 0.15/frame
        # with a 0.2 torso is inside the 2-torso gate and must not split
        # into multiple tracks.
        frames = [[make_pose(0.2 + 0.15 * i, 0.5)] for i in range(N_FRAMES)]
        selected, report = select_subject(frames)
        self.assertIsNotNone(selected)
        self.assertEqual(report["n_tracks"], 1)


class TestTemporalContinuity(unittest.TestCase):
    def test_temporary_pose_loss_keeps_slots_not_switches(self):
        # Subject missing in frames 3-4 (motion blur): one track, coverage 5,
        # None in exactly those slots — handed to the existing interpolation
        # rules, NOT replaced by some other candidate.
        frames = [[make_pose(0.5, 0.5)] if i not in (3, 4) else [] for i in range(N_FRAMES)]
        selected, report = select_subject(frames)
        self.assertIsNotNone(selected)
        self.assertEqual(report["n_tracks"], 1)
        self.assertEqual(report["winner_coverage"], 5)
        self.assertIsNone(selected[3])
        self.assertIsNone(selected[4])

    def test_gate_widens_across_gaps_so_subject_reassociates(self):
        # Missing 2 frames then reappearing 0.55 away: per-frame gate would
        # be 0.4 (2 torsos x 0.2), but the 3-frame gap widens it to
        # min(2*3, 4) * 0.2 = 0.8 — same track, not a new one.
        frames = [
            [make_pose(0.3, 0.5)], [make_pose(0.3, 0.5)], [make_pose(0.3, 0.5)],
            [], [],
            [make_pose(0.85, 0.5)], [make_pose(0.85, 0.5)],
        ]
        selected, report = select_subject(frames)
        self.assertIsNotNone(selected)
        self.assertEqual(report["n_tracks"], 1)
        self.assertEqual(report["winner_coverage"], 5)

    def test_teleport_beyond_capped_gate_is_a_new_track(self):
        # 5-frame gap, but the cap (4 torsos = 0.8) still applies: a
        # reappearance 0.9 away is a DIFFERENT subject, and neither track
        # alone is persistent enough — reject rather than stitch them.
        frames = [
            [make_pose(0.05, 0.5)], [], [], [], [],
            [make_pose(0.95, 0.5)], [make_pose(0.95, 0.5)],
        ]
        selected, report = select_subject(frames)
        self.assertIsNone(selected)
        self.assertEqual(report["n_tracks"], 2)
        self.assertIn("persistent", report["reason"])


class TestMultiPerson(unittest.TestCase):
    def _two_person_frames(self, torso_a=0.25, torso_b=0.15, frames_b=range(N_FRAMES)):
        frames = []
        for i in range(N_FRAMES):
            frame = [make_pose(0.3, 0.5, torso=torso_a)]
            if i in frames_b:
                frame.append(make_pose(0.75, 0.5, torso=torso_b))
            frames.append(frame)
        return frames

    def test_two_players_dominant_subject_wins_consistently(self):
        # Both fully persistent; A is 1.67x B's size (>= 1.25 dominance).
        selected, report = select_subject(self._two_person_frames())
        self.assertIsNotNone(selected)
        self.assertEqual(report["qualified"], 2)
        for i in range(N_FRAMES):
            self.assertAlmostEqual(_selected_center_x(selected, i), 0.3)  # A, every frame

    def test_equally_valid_subjects_are_rejected_not_guessed(self):
        # Same persistence, sizes within the dominance ratio (0.2 vs 0.18):
        # genuinely ambiguous — refuse rather than silently pick.
        selected, report = select_subject(self._two_person_frames(torso_a=0.2, torso_b=0.18))
        self.assertIsNone(selected)
        self.assertIn("refusing to guess", report["reason"])

    def test_late_entrant_smaller_than_subject_does_not_displace(self):
        # B walks into frame for the last 4 frames, smaller than A: A keeps
        # max coverage AND size dominance — selected.
        selected, report = select_subject(
            self._two_person_frames(torso_a=0.25, torso_b=0.15, frames_b=range(3, 7)))
        self.assertIsNotNone(selected)
        for i in range(N_FRAMES):
            self.assertAlmostEqual(_selected_center_x(selected, i), 0.3)

    def test_large_late_entrant_forces_rejection_not_a_switch(self):
        # B enters late but LARGER (a foreground walk-through): B lacks max
        # coverage, A lacks size dominance — nobody wins, and crucially the
        # pipeline does NOT silently switch subjects mid-session.
        selected, report = select_subject(
            self._two_person_frames(torso_a=0.2, torso_b=0.32, frames_b=range(3, 7)))
        self.assertIsNone(selected)
        self.assertIn("refusing to guess", report["reason"])

    def test_subject_missing_one_frame_while_rival_persists_rejects(self):
        # The adversarial documented case: batsman (large) lost at contact
        # blur for 2 frames while the feeder (small) is detected in all 7.
        # Feeder has more coverage, batsman has more size — no Pareto winner:
        # reject. Scoring the feeder here is exactly audit C1's defect.
        frames = []
        for i in range(N_FRAMES):
            frame = [make_pose(0.7, 0.5, torso=0.12)]  # feeder, always present
            if i not in (4, 5):
                frame.append(make_pose(0.35, 0.5, torso=0.28))  # batsman, blurred out at 4-5
            frames.append(frame)
        selected, report = select_subject(frames)
        self.assertIsNone(selected)
        self.assertIn("refusing to guess", report["reason"])

    def test_spurious_single_frame_detections_are_ignored(self):
        # Equipment-bag-style one-frame false positives at scattered spots
        # never accumulate coverage and never affect the subject.
        frames = []
        spurious_spots = {1: (0.9, 0.1), 3: (0.1, 0.9), 5: (0.9, 0.9)}
        for i in range(N_FRAMES):
            frame = [make_pose(0.4, 0.5, torso=0.22)]
            if i in spurious_spots:
                sx, sy = spurious_spots[i]
                frame.append(make_pose(sx, sy, torso=0.05))
            frames.append(frame)
        selected, report = select_subject(frames)
        self.assertIsNotNone(selected)
        self.assertEqual(report["winner_coverage"], N_FRAMES)
        for i in range(N_FRAMES):
            self.assertAlmostEqual(_selected_center_x(selected, i), 0.4)

    def test_detector_candidate_order_shuffle_does_not_change_subject(self):
        # MediaPipe does not guarantee candidate ordering between frames.
        # Association is by location, so alternating the order per frame must
        # select the same physical subject every time.
        a = lambda: make_pose(0.3, 0.5, torso=0.25)  # noqa: E731 - tiny local factories
        b = lambda: make_pose(0.75, 0.5, torso=0.15)  # noqa: E731
        frames = [[a(), b()] if i % 2 == 0 else [b(), a()] for i in range(N_FRAMES)]
        selected, report = select_subject(frames)
        self.assertIsNotNone(selected)
        for i in range(N_FRAMES):
            self.assertAlmostEqual(_selected_center_x(selected, i), 0.3)

    def test_two_candidates_in_one_frame_cannot_share_a_track(self):
        # Two nearby candidates in the SAME frame: the second must open a
        # new track, never overwrite the first's slot.
        frames = [[make_pose(0.5, 0.5), make_pose(0.55, 0.5)]] + [[make_pose(0.5, 0.5)] for _ in range(6)]
        tracks = build_tracks(frames)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len(tracks[0]["poses"]), 7)
        self.assertEqual(len(tracks[1]["poses"]), 1)


class TestRejection(unittest.TestCase):
    def test_no_candidates_anywhere_rejects(self):
        selected, report = select_subject([[] for _ in range(N_FRAMES)])
        self.assertIsNone(selected)
        self.assertEqual(report["reason"], "no pose candidates in any frame")

    def test_insufficient_persistence_rejects(self):
        # Only 3 frames of detections — below MIN_TRACK_COVERAGE.
        frames = [[make_pose(0.5, 0.5)] if i < 3 else [] for i in range(N_FRAMES)]
        selected, report = select_subject(frames)
        self.assertIsNone(selected)
        self.assertIn("no persistent subject", report["reason"])
        self.assertIn("3/7", report["reason"])


class TestDeterminism(unittest.TestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        frames = [[make_pose(0.3, 0.5, torso=0.25), make_pose(0.75, 0.5, torso=0.15)]
                  for _ in range(N_FRAMES)]
        first_selected, first_report = select_subject(frames)
        second_selected, second_report = select_subject(frames)
        self.assertEqual(first_report, second_report)
        for pose_a, pose_b in zip(first_selected, second_selected):
            self.assertIs(pose_a, pose_b)  # the exact same objects, not lookalikes

    def test_gate_constants_are_sane(self):
        # The per-frame gate must not exceed its own cap at gap 1, and the
        # cap must genuinely bound multi-frame gaps.
        self.assertLessEqual(GATE_TORSOS_PER_FRAME, GATE_TORSOS_MAX)
        self.assertGreater(SIZE_DOMINANCE_RATIO, 1.0)


if __name__ == "__main__":
    unittest.main()
