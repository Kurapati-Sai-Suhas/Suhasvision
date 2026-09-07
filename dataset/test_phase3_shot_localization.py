"""
Tests for L0-L3 shot localization.

The metric definitions and the windowing rule are what every number in the
benchmark rests on, so they are pinned here rather than trusted.
"""

import unittest

import numpy as np

import phase3_shot_localization as SL


def bump(n=120, centre=60, width=10, height=8.0, base=1.0, noise=0.0, seed=0):
    """A motion signal with one clean peak."""
    t = np.arange(n)
    s = base + height * np.exp(-0.5 * ((t - centre) / width) ** 2)
    if noise:
        s = s + np.random.RandomState(seed).normal(0, noise, n)
    return s


class TestTemporalIoU(unittest.TestCase):
    def test_identical_intervals_score_one(self):
        self.assertAlmostEqual(SL.temporal_iou((10, 30), (10, 30)), 1.0)

    def test_disjoint_intervals_score_zero(self):
        self.assertEqual(SL.temporal_iou((0, 10), (20, 30)), 0.0)

    def test_touching_intervals_score_zero(self):
        self.assertEqual(SL.temporal_iou((0, 10), (10, 20)), 0.0)

    def test_half_overlap(self):
        # [0,20] vs [10,30]: intersection 10, union 30.
        self.assertAlmostEqual(SL.temporal_iou((0, 20), (10, 30)), 10 / 30)

    def test_contained_interval(self):
        # [10,20] inside [0,40]: intersection 10, union 40.
        self.assertAlmostEqual(SL.temporal_iou((10, 20), (0, 40)), 0.25)

    def test_is_symmetric(self):
        self.assertAlmostEqual(SL.temporal_iou((3, 19), (11, 40)),
                               SL.temporal_iou((11, 40), (3, 19)))

    def test_none_scores_zero_rather_than_raising(self):
        self.assertEqual(SL.temporal_iou(None, (1, 2)), 0.0)
        self.assertEqual(SL.temporal_iou((1, 2), None), 0.0)


class TestCrosses(unittest.TestCase):
    def test_detects_a_cut_inside_the_interval(self):
        self.assertTrue(SL.crosses((10, 40), [25]))

    def test_ignores_a_cut_outside_the_interval(self):
        self.assertFalse(SL.crosses((10, 40), [80]))

    def test_boundary_frames_count_as_crossing(self):
        self.assertTrue(SL.crosses((10, 40), [10]))
        self.assertTrue(SL.crosses((10, 40), [40]))

    def test_no_cuts_is_not_a_crossing(self):
        self.assertFalse(SL.crosses((10, 40), []))
        self.assertFalse(SL.crosses(None, [5]))


class TestLocalize(unittest.TestCase):
    def test_finds_an_interval_around_a_clean_peak(self):
        r = SL.localize(bump(centre=60), n_frames=120)
        self.assertFalse(r["rejected"])
        self.assertLess(r["start"], 60)
        self.assertGreater(r["end"], 60)

    def test_interval_tracks_the_peak_when_it_moves(self):
        a = SL.localize(bump(centre=30), n_frames=120)
        b = SL.localize(bump(centre=90), n_frames=120)
        self.assertLess(a["start"], b["start"])
        self.assertLess(a["end"], b["end"])

    def test_flat_signal_is_rejected_not_given_an_arbitrary_window(self):
        # This is what makes the false-shot rate meaningful.
        r = SL.localize(np.ones(120), n_frames=120)
        self.assertTrue(r["rejected"])
        self.assertIsNone(r["start"])

    def test_low_prominence_peak_is_rejected(self):
        r = SL.localize(bump(height=0.05), n_frames=120, tau=1.6)
        self.assertTrue(r["rejected"])
        self.assertEqual(r["reason"], "peak_not_prominent")

    def test_lower_tau_accepts_what_higher_tau_rejects(self):
        s = bump(height=0.6)
        self.assertTrue(SL.localize(s, 120, tau=3.0)["rejected"])
        self.assertFalse(SL.localize(s, 120, tau=1.05)["rejected"])

    def test_wider_alpha_gives_a_narrower_interval(self):
        s = bump(width=12)
        lo = SL.localize(s, 120, alpha=0.15)
        hi = SL.localize(s, 120, alpha=0.75)
        self.assertGreaterEqual(lo["end"] - lo["start"], hi["end"] - hi["start"])

    def test_duration_is_clamped_to_max(self):
        # The bump must stay a small fraction of the clip: a very wide bump
        # lifts the median baseline, which correctly drops prominence below
        # tau and makes the method reject rather than clamp.
        r = SL.localize(bump(n=400, centre=200, width=20, height=20),
                        n_frames=400, max_dur=25)
        self.assertFalse(r["rejected"])
        self.assertLessEqual(r["end"] - r["start"], 26)

    def test_a_bump_spanning_most_of_the_clip_is_rejected(self):
        # Pins the behaviour the previous case exposed. Prominence is measured
        # against the MEDIAN, so what matters is the bump's width relative to
        # the clip: at width/n = 1/3 the median is itself elevated, prominence
        # falls to ~1.3, and the method correctly declines. When almost
        # everything is moving there is no distinguishable event to localize.
        r = SL.localize(bump(n=120, centre=60, width=40, height=20),
                        n_frames=120)
        self.assertTrue(r["rejected"])
        self.assertEqual(r["reason"], "peak_not_prominent")

    def test_duration_is_padded_to_min(self):
        r = SL.localize(bump(width=1, height=20), n_frames=200, min_dur=18)
        self.assertGreaterEqual(r["end"] - r["start"], 17)

    def test_interval_stays_inside_the_clip(self):
        for c in (2, 60, 118):
            r = SL.localize(bump(centre=c), n_frames=120)
            if not r["rejected"]:
                self.assertGreaterEqual(r["start"], 0)
                self.assertLessEqual(r["end"], 119)
                self.assertLess(r["start"], r["end"])

    def test_too_short_signal_is_rejected(self):
        self.assertTrue(SL.localize(np.array([1.0, 2.0]), 3)["rejected"])
        self.assertTrue(SL.localize(None, 100)["rejected"])


class TestMethods(unittest.TestCase):
    def test_l0_returns_the_whole_clip_and_never_rejects(self):
        r = SL.predict("L0_whole_clip", 90, bump(), bump())
        self.assertEqual(r["start"], 0)
        self.assertEqual(r["end"], 89)
        self.assertFalse(r["rejected"])

    def test_l0_ignores_the_motion_signal_entirely(self):
        a = SL.predict("L0_whole_clip", 90, bump(centre=10), None)
        b = SL.predict("L0_whole_clip", 90, bump(centre=80), None)
        self.assertEqual((a["start"], a["end"]), (b["start"], b["end"]))

    def test_l1_uses_global_and_ignores_local(self):
        g, l = bump(centre=30), bump(centre=90)
        a = SL.predict("L1_global_motion", 120, g, l)
        b = SL.predict("L1_global_motion", 120, g, None)
        self.assertEqual((a["start"], a["end"]), (b["start"], b["end"]))

    def test_l2_uses_local_and_ignores_global(self):
        g, l = bump(centre=30), bump(centre=90)
        r = SL.predict("L2_batsman_motion", 120, g, l)
        self.assertGreater(r["start"], 60)      # follows the local peak

    def test_l2_without_a_track_is_rejected(self):
        r = SL.predict("L2_batsman_motion", 120, bump(), None)
        self.assertTrue(r["rejected"])

    def test_l3_requires_both_signals(self):
        self.assertTrue(SL.predict("L3_hybrid", 120, bump(), None)["rejected"])

    def test_l3_suppresses_a_peak_present_in_only_one_signal(self):
        # Global peaks at 30, local at 90. The product should not endorse
        # either exclusive peak as strongly as an agreeing one.
        g = bump(centre=30, height=8)
        l = bump(centre=90, height=8)
        sig = SL.build_signal("L3_hybrid", g, l)
        self.assertLess(sig[30], 0.5)
        self.assertLess(sig[90], 0.5)

    def test_l3_endorses_an_agreeing_peak(self):
        g = bump(centre=60, height=8)
        l = bump(centre=60, height=8)
        sig = SL.build_signal("L3_hybrid", g, l)
        self.assertGreater(sig[60], 0.9)


class TestGapFilling(unittest.TestCase):
    def test_missing_track_frames_are_interpolated_not_zeroed(self):
        # A zero would assert "no motion here", which a missing box does not
        # support. Interpolation keeps the signal neutral instead.
        s = np.array([1.0, np.nan, 3.0])
        out = SL._fill_gaps(s)
        self.assertAlmostEqual(out[1], 2.0)

    def test_all_missing_returns_none(self):
        self.assertIsNone(SL._fill_gaps(np.array([np.nan, np.nan])))

    def test_none_input_returns_none(self):
        self.assertIsNone(SL._fill_gaps(None))


class TestNormalise(unittest.TestCase):
    def test_maps_to_unit_range(self):
        out = SL._normalise(np.array([2.0, 4.0, 6.0]))
        self.assertAlmostEqual(out.min(), 0.0)
        self.assertAlmostEqual(out.max(), 1.0)

    def test_constant_signal_does_not_divide_by_zero(self):
        out = SL._normalise(np.array([5.0, 5.0, 5.0]))
        self.assertTrue(np.all(out == 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
