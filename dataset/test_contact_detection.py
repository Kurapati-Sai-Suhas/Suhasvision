import unittest
from collections import namedtuple

import numpy as np

import contact_detection as cd
from contact_detection import detect_contact, wrist_speed_series, _smooth

Landmark = namedtuple("Landmark", ["x", "y", "z", "visibility", "presence"])


def make_pose(cx=0.5, cy=0.5, torso=0.2, lw=(0.4, 0.5), rw=(0.6, 0.5)):
    """33-landmark pose with controllable wrist positions, so a synthetic
    velocity profile can be constructed exactly."""
    lms = [Landmark(cx, cy, 0.0, 1.0, 1.0) for _ in range(33)]
    lms[11] = Landmark(cx - 0.05, cy - torso, 0.0, 1.0, 1.0)  # left_shoulder
    lms[12] = Landmark(cx + 0.05, cy - torso, 0.0, 1.0, 1.0)  # right_shoulder
    lms[23] = Landmark(cx - 0.05, cy, 0.0, 1.0, 1.0)          # left_hip
    lms[24] = Landmark(cx + 0.05, cy, 0.0, 1.0, 1.0)          # right_hip
    lms[cd.LEFT_WRIST_IDX] = Landmark(lw[0], lw[1], 0.0, 1.0, 1.0)
    lms[cd.RIGHT_WRIST_IDX] = Landmark(rw[0], rw[1], 0.0, 1.0, 1.0)
    return lms


def swing_samples(n=12, peak_at=8, peak_speed=0.30, base_speed=0.005, bilateral=True):
    """Synthetic two-handed swing: wrists creep, then jump at `peak_at`."""
    samples, lx, rx = [], 0.30, 0.32
    for i in range(n):
        step = peak_speed if i == peak_at else base_speed
        lx += step
        rx += step if bilateral else base_speed
        samples.append((100 + i, make_pose(lw=(lx, 0.5), rw=(rx, 0.5))))
    return samples


class TestWristSpeedSeries(unittest.TestCase):
    def test_speed_is_scale_invariant(self):
        # Same motion as a fraction of body size must score the same whether
        # the subject is camera-near or camera-far.
        near = [(0, make_pose(torso=0.4, lw=(0.2, 0.5), rw=(0.2, 0.5))),
                (1, make_pose(torso=0.4, lw=(0.6, 0.5), rw=(0.6, 0.5)))]
        far = [(0, make_pose(torso=0.2, lw=(0.2, 0.5), rw=(0.2, 0.5))),
               (1, make_pose(torso=0.2, lw=(0.4, 0.5), rw=(0.4, 0.5)))]
        _, nl, _ = wrist_speed_series(near)
        _, fl, _ = wrist_speed_series(far)
        self.assertAlmostEqual(nl[0], fl[0], places=5)

    def test_pairs_spanning_a_gap_are_skipped_not_bridged(self):
        # A displacement measured across a missing detection is not a speed;
        # bridging it would manufacture a peak exactly where tracking failed.
        samples = [(0, make_pose(lw=(0.1, 0.5))), None, (2, make_pose(lw=(0.9, 0.5)))]
        frames, left, right = wrist_speed_series(samples)
        self.assertEqual(len(frames), 0)

    def test_frame_number_is_the_later_frame_of_the_pair(self):
        samples = [(10, make_pose()), (11, make_pose(lw=(0.5, 0.5)))]
        frames, _, _ = wrist_speed_series(samples)
        self.assertEqual(frames, [11])


class TestSmoothing(unittest.TestCase):
    def test_preserves_length(self):
        self.assertEqual(len(_smooth([1.0, 5.0, 1.0, 1.0, 1.0])), 5)

    def test_too_short_series_returned_unchanged(self):
        np.testing.assert_array_equal(_smooth([1.0, 2.0], window=3), np.array([1.0, 2.0]))

    def test_reduces_single_sample_spike(self):
        spiky = [1.0, 1.0, 9.0, 1.0, 1.0]
        self.assertLess(max(_smooth(spiky)), 9.0)


class TestDetectContact(unittest.TestCase):
    def test_locates_a_clear_two_handed_swing(self):
        event = detect_contact(swing_samples(), start_frame=100, end_frame=111)
        self.assertTrue(event["valid"], event["reason"])
        # Smoothing may shift the reported peak by one sample; that is within
        # the +/-1 frame this method honestly claims.
        self.assertLessEqual(abs(event["frame_index"] - 108), 1)
        self.assertGreater(event["peak_prominence"], cd.MIN_PEAK_PROMINENCE_RATIO)

    def test_flat_velocity_curve_is_rejected(self):
        # Constant creep: argmax still exists, but there is no event.
        flat = swing_samples(n=12, peak_at=-1, base_speed=0.02)
        event = detect_contact(flat, 100, 111)
        self.assertFalse(event["valid"])
        self.assertIn("flat velocity", event["reason"])

    def test_one_handed_motion_is_rejected(self):
        event = detect_contact(swing_samples(bilateral=False), 100, 111)
        self.assertFalse(event["valid"])
        self.assertIn("one-handed", event["reason"])

    def test_peak_at_window_edge_is_rejected(self):
        event = detect_contact(swing_samples(n=12, peak_at=1), 100, 111)
        self.assertFalse(event["valid"])
        self.assertIn("window edge", event["reason"])

    def test_too_few_samples_is_reported_not_crashed(self):
        event = detect_contact([(0, make_pose()), (1, make_pose())], 0, 10)
        self.assertFalse(event["valid"])
        self.assertIn("usable speed samples", event["reason"])
        self.assertIsNone(event["frame_index"])

    def test_all_missing_detections_is_handled(self):
        event = detect_contact([None] * 10, 0, 10)
        self.assertFalse(event["valid"])
        self.assertEqual(event["n_speed_samples"], 0)

    def test_returns_a_scored_event_not_a_bare_index(self):
        event = detect_contact(swing_samples(), 100, 111)
        for key in ("frame_index", "confidence", "peak_prominence",
                    "bilateral_agreement", "peak_fraction", "method",
                    "valid", "reason", "n_speed_samples"):
            self.assertIn(key, event)
        self.assertEqual(event["method"], "wrist_speed")
        self.assertGreaterEqual(event["confidence"], 0.0)
        self.assertLessEqual(event["confidence"], 1.0)

    def test_determinism(self):
        a = detect_contact(swing_samples(), 100, 111)
        b = detect_contact(swing_samples(), 100, 111)
        self.assertEqual(a, b)

    def test_frame_index_reported_even_when_invalid(self):
        # A2 (reordering without quality gating) accepts on frame_index, so
        # an invalid event must still carry its located index rather than
        # None -- otherwise the A2/A3 ablation cannot be run.
        event = detect_contact(swing_samples(bilateral=False), 100, 111)
        self.assertFalse(event["valid"])
        self.assertIsNotNone(event["frame_index"])


if __name__ == "__main__":
    unittest.main()
