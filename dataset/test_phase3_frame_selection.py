import unittest

from phase3_frame_selection import (
    Candidate, select_best_seven, frame_quality, phase_affinity,
    normalised_time, contact_relevance, redundancy, STRATEGIES,
    N_PHASES, PHASE_ANCHORS,
)


def pool(n=40, start=0, step=3, **kw):
    """A well-behaved candidate pool: usable pose everywhere."""
    d = dict(pose_quality=0.8, sharpness=0.7, identity_conf=0.9,
             visibility=0.8, occlusion=0.0, temporal_stability=0.9)
    d.update(kw)
    return [Candidate(frame=start + i * step, motion=0.5, **d) for i in range(n)]


class TestFrameQuality(unittest.TestCase):
    def test_quality_is_bounded(self):
        best = Candidate(0, pose_quality=1, sharpness=1, identity_conf=1,
                         visibility=1, temporal_stability=1, occlusion=0)
        worst = Candidate(0, pose_quality=0, sharpness=0, identity_conf=0,
                          visibility=0, temporal_stability=0, occlusion=1)
        self.assertLessEqual(frame_quality(best), 1.0)
        self.assertGreaterEqual(frame_quality(worst), 0.0)
        self.assertGreater(frame_quality(best), frame_quality(worst))

    def test_occlusion_reduces_quality(self):
        clear = Candidate(0, pose_quality=.8, sharpness=.8, identity_conf=.8,
                          visibility=.8, temporal_stability=.8, occlusion=0.0)
        blocked = Candidate(0, pose_quality=.8, sharpness=.8, identity_conf=.8,
                            visibility=.8, temporal_stability=.8, occlusion=0.9)
        self.assertLess(frame_quality(blocked), frame_quality(clear))

    def test_motion_does_not_raise_quality(self):
        """Motion must not be a quality term: the blurriest frame in a swing
        is often the highest-motion one. This guards the design decision."""
        still = Candidate(0, pose_quality=.8, sharpness=.8, identity_conf=.8,
                          visibility=.8, temporal_stability=.8, motion=0.0)
        fast = Candidate(0, pose_quality=.8, sharpness=.8, identity_conf=.8,
                         visibility=.8, temporal_stability=.8, motion=1.0)
        self.assertAlmostEqual(frame_quality(still), frame_quality(fast))


class TestPhaseModel(unittest.TestCase):
    def test_normalised_time_anchors(self):
        self.assertAlmostEqual(normalised_time(10, 10, 50, 90), 0.0)
        self.assertAlmostEqual(normalised_time(50, 10, 50, 90), 1.0)
        self.assertGreater(normalised_time(90, 10, 50, 90), 1.0)

    def test_each_phase_peaks_at_its_own_anchor(self):
        for k in range(N_PHASES):
            best = max(range(N_PHASES),
                       key=lambda j: phase_affinity(PHASE_ANCHORS[k], j, 0.5))
            self.assertEqual(best, k, f"phase {k} anchor should favour phase {k}")

    def test_contact_relevance_only_applies_to_contact_phase(self):
        self.assertGreater(contact_relevance(100, 100, 5, 30.0), 0.9)
        for k in (0, 1, 2, 3, 4, 6):
            self.assertEqual(contact_relevance(100, 100, k, 30.0), 0.0)

    def test_redundancy_decays_with_time(self):
        a, b, c = Candidate(0), Candidate(1), Candidate(60)
        self.assertGreater(redundancy(a, b, 30.0), redundancy(a, c, 30.0))


class TestBestSevenConstraints(unittest.TestCase):
    """The hard constraints from the Phase-3 spec, as executable checks."""

    def setUp(self):
        self.r = select_best_seven(pool(), onset=0, contact=60, end=117, fps=30.0)

    def test_returns_exactly_seven(self):
        self.assertTrue(self.r["ok"], self.r.get("reason"))
        self.assertEqual(len(self.r["frames"]), N_PHASES)

    def test_strictly_chronological(self):
        f = self.r["frames"]
        self.assertEqual(f, sorted(f))
        self.assertEqual(len(set(f)), N_PHASES, "frames must be distinct")

    def test_respects_minimum_gap(self):
        self.assertGreaterEqual(self.r["min_gap_frames"], 2)

    def test_does_not_collapse_into_a_cluster(self):
        """The failure mode the optimizer exists to prevent: seven frames
        crammed around the motion peak, carrying one frame of information."""
        self.assertGreater(self.r["temporal_span_s"], 1.0)

    def test_every_phase_is_assigned_once_in_order(self):
        phases = [p["phase"] for p in self.r["per_frame"]]
        self.assertEqual(len(set(phases)), N_PHASES)
        frames = [p["frame"] for p in self.r["per_frame"]]
        self.assertEqual(frames, sorted(frames))

    def test_contact_phase_lands_near_the_contact_estimate(self):
        contact_row = self.r["per_frame"][5]
        self.assertLess(abs(contact_row["frame"] - 60), 12)


class TestBestSevenRefusal(unittest.TestCase):
    def test_refuses_when_pool_too_small(self):
        r = select_best_seven(pool(n=4), 0, 6, 12, 30.0)
        self.assertFalse(r["ok"])
        self.assertIsNone(r["frames"])

    def test_refuses_when_pose_quality_floor_removes_everything(self):
        r = select_best_seven(pool(pose_quality=0.05), 0, 60, 117, 30.0)
        self.assertFalse(r["ok"])
        self.assertIn("pose-quality floor", r["reason"])

    def test_refuses_rather_than_violating_min_gap(self):
        """Seven candidates crammed into six frames cannot satisfy a 2-frame
        gap; refusing beats emitting a degenerate sequence."""
        tight = [Candidate(frame=i, pose_quality=.8, sharpness=.7,
                           identity_conf=.9, visibility=.8, motion=.5,
                           temporal_stability=.9) for i in range(7)]
        r = select_best_seven(tight, 0, 3, 6, 30.0, min_gap=2)
        self.assertFalse(r["ok"])

    def test_low_quality_frames_are_excluded_from_the_pool(self):
        mixed = pool(n=30)
        for c in mixed[:10]:
            c.pose_quality = 0.05
        r = select_best_seven(mixed, 0, 45, 87, 30.0)
        self.assertTrue(r["ok"], r.get("reason"))
        self.assertNotIn(mixed[0].frame, r["frames"])


class TestOptimizerPrefersBetterFrames(unittest.TestCase):
    def test_prefers_the_sharper_of_two_equivalent_frames(self):
        cands = pool(n=40)
        # Two frames sit at nearly the same time; one is much sharper.
        blurry = Candidate(frame=60, pose_quality=.8, sharpness=0.05,
                           identity_conf=.9, visibility=.8, motion=.5,
                           temporal_stability=.9)
        sharp = Candidate(frame=61, pose_quality=.8, sharpness=0.99,
                          identity_conf=.9, visibility=.8, motion=.5,
                          temporal_stability=.9)
        cands = [c for c in cands if c.frame not in (60, 61)] + [blurry, sharp]
        r = select_best_seven(cands, 0, 61, 117, 30.0)
        self.assertTrue(r["ok"], r.get("reason"))
        if 60 in r["frames"] or 61 in r["frames"]:
            self.assertIn(61, r["frames"], "should prefer the sharper frame")

    def test_score_is_finite_and_positive(self):
        r = select_best_seven(pool(), 0, 60, 117, 30.0)
        self.assertGreater(r["total_score"], 0)


class TestDPIsExactForItsObjective(unittest.TestCase):
    """The DP claims exactness for a first-order decomposable objective.
    Verify it against brute force on instances small enough to enumerate.

    This checks the claim actually made -- exact for THIS objective -- rather
    than the stronger claim (optimal for any best-7 objective) which is not
    made and would not hold.
    """

    @staticmethod
    def _brute_force(pool_list, onset, contact, end, fps, min_gap):
        """Enumerate every feasible chronological 7-assignment."""
        import itertools
        from phase3_frame_selection import (
            frame_quality, phase_affinity, normalised_time,
            contact_relevance, redundancy, W_QUALITY, W_PHASE, W_CONTACT,
            REDUNDANCY_WEIGHT, MIN_POSE_QUALITY,
        )
        pool = sorted([c for c in pool_list if c.pose_quality >= MIN_POSE_QUALITY],
                      key=lambda c: c.frame)
        n = len(pool)
        motions = [c.motion for c in pool]
        lo, hi = min(motions), max(motions)
        span = (hi - lo) or 1.0
        mn = [(m - lo) / span for m in motions]
        us = [normalised_time(c.frame, onset, contact, end) for c in pool]
        qs = [frame_quality(c) for c in pool]

        def gain(i, k):
            return (W_QUALITY * qs[i]
                    + W_PHASE * phase_affinity(us[i], k, mn[i])
                    + W_CONTACT * contact_relevance(pool[i].frame, contact, k, fps))

        best, best_combo = float("-inf"), None
        for combo in itertools.combinations(range(n), N_PHASES):
            ok = all(pool[b].frame - pool[a].frame >= min_gap
                     for a, b in zip(combo, combo[1:]))
            if not ok:
                continue
            tot = sum(gain(i, k) for k, i in enumerate(combo))
            tot -= REDUNDANCY_WEIGHT * sum(
                redundancy(pool[a], pool[b], fps) for a, b in zip(combo, combo[1:]))
            if tot > best:
                best, best_combo = tot, combo
        return best, ([pool[i].frame for i in best_combo] if best_combo else None)

    def _check(self, cands, onset, contact, end, fps=30.0, min_gap=2):
        dp = select_best_seven(cands, onset, contact, end, fps, min_gap=min_gap)
        bf_score, bf_frames = self._brute_force(cands, onset, contact, end, fps, min_gap)
        self.assertTrue(dp["ok"], dp.get("reason"))
        # select_best_seven rounds total_score to 4dp for readability, so
        # compare at that precision rather than the raw float.
        self.assertAlmostEqual(dp["total_score"], round(bf_score, 4), places=4,
                               msg=f"DP {dp['total_score']} != brute force {bf_score}")
        self.assertEqual(dp["frames"], bf_frames,
                         "DP selected a different frame set from the exhaustive optimum")

    def test_matches_brute_force_small_uniform_pool(self):
        self._check(pool(n=11, step=5), onset=0, contact=25, end=50)

    def test_matches_brute_force_with_varied_quality(self):
        cands = pool(n=12, step=4)
        for i, c in enumerate(cands):
            c.pose_quality = 0.4 + 0.05 * (i % 7)
            c.sharpness = 0.3 + 0.06 * ((i * 3) % 9)
            c.motion = (i % 5) / 4.0
        self._check(cands, onset=0, contact=22, end=44)

    def test_matches_brute_force_with_tight_gap_constraint(self):
        cands = pool(n=10, step=3)
        for i, c in enumerate(cands):
            c.motion = ((i * 7) % 11) / 10.0
        self._check(cands, onset=0, contact=15, end=27, min_gap=3)

    def test_matches_brute_force_when_contact_is_late(self):
        cands = pool(n=11, step=4)
        for i, c in enumerate(cands):
            c.sharpness = ((i * 5) % 13) / 12.0
        self._check(cands, onset=0, contact=36, end=40)


class TestBaselineStrategies(unittest.TestCase):
    def test_all_strategies_return_seven_chronological_frames(self):
        cands = pool(n=40)
        for name, fn in STRATEGIES.items():
            r = fn(cands, 0, 60, 117, 30.0)
            self.assertTrue(r["ok"], f"{name}: {r.get('reason')}")
            self.assertEqual(len(r["frames"]), N_PHASES, name)
            self.assertEqual(r["frames"], sorted(r["frames"]), name)

    def test_motion_top_can_cluster_and_that_is_the_point(self):
        """F2 is included as a baseline BECAUSE it clusters. If this ever
        stops clustering, the comparison has lost its contrast."""
        cands = pool(n=40)
        for c in cands:
            c.motion = 0.1
        for c in cands[20:27]:
            c.motion = 0.99          # one tight burst of motion
        r = STRATEGIES["F2_motion_top"](cands, 0, 60, 117, 30.0)
        span = r["frames"][-1] - r["frames"][0]
        self.assertLess(span, 30, "motion-top should collapse onto the burst")

    def test_uniform_ignores_contact(self):
        cands = pool(n=40)
        a = STRATEGIES["F0_uniform"](cands, 0, 30, 117, 30.0)["frames"]
        b = STRATEGIES["F0_uniform"](cands, 0, 90, 117, 30.0)["frames"]
        self.assertEqual(a, b, "uniform must not depend on the contact estimate")


if __name__ == "__main__":
    unittest.main()
