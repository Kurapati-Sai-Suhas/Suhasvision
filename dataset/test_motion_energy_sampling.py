import unittest

import numpy as np

from zero_storage_pipeline import motion_energy_phase_indices, N_FRAMES


def _frame(value, size=8):
    """A uniform grayscale frame; differences between two of these are just
    the absolute difference of their fill values."""
    return np.full((size, size), value, dtype=np.uint8)


class TestMotionEnergyPhaseIndices(unittest.TestCase):
    def test_returns_none_when_too_few_readable_frames(self):
        self.assertIsNone(motion_energy_phase_indices([None, None, _frame(0)], 0, 100))

    def test_returns_none_on_a_completely_static_window(self):
        # No motion anywhere -> uniform sampling is as good as anything, and
        # the cumulative distribution would be undefined.
        frames = [_frame(50) for _ in range(10)]
        self.assertIsNone(motion_energy_phase_indices(frames, 0, 100))

    def test_uniform_motion_approximates_uniform_sampling(self):
        # Motion spread evenly across the window: cumulative-motion sampling
        # should land close to plain linear spacing.
        frames = [_frame(i * 10) for i in range(11)]  # constant per-step delta
        indices = motion_energy_phase_indices(frames, 0, 100, n_phases=N_FRAMES)
        self.assertIsNotNone(indices)
        uniform = [int(100 * i / (N_FRAMES - 1)) for i in range(N_FRAMES)]
        for got, exp in zip(indices, uniform):
            self.assertLess(abs(got - exp), 20, f"{indices} deviates far from uniform {uniform}")

    def test_late_burst_of_motion_pulls_samples_later(self):
        # Static first half, then a burst -- the real cricket case (slow
        # stance/trigger, explosive downswing/contact). Sampling should
        # concentrate in the second half rather than spreading evenly.
        frames = [_frame(0) for _ in range(6)] + [_frame(40 * i) for i in range(1, 6)]
        indices = motion_energy_phase_indices(frames, 0, 100, n_phases=N_FRAMES)
        self.assertIsNotNone(indices)
        uniform_mid = 50
        # More than half the phases should sit past the midpoint of the window.
        late = sum(1 for i in indices if i > uniform_mid)
        self.assertGreater(late, N_FRAMES // 2,
                           f"expected sampling to shift toward the motion burst, got {indices}")

    def test_indices_are_monotonic_and_in_bounds(self):
        rng = np.random.default_rng(0)
        frames = [_frame(int(v)) for v in rng.integers(0, 255, size=12)]
        indices = motion_energy_phase_indices(frames, 30, 130, n_phases=N_FRAMES)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), N_FRAMES)
        self.assertEqual(indices, sorted(indices), "phase indices must never go backwards")
        self.assertGreaterEqual(indices[0], 30)
        self.assertLessEqual(indices[-1], 130)

    def test_missing_frames_are_tolerated(self):
        frames = [_frame(0), None, _frame(30), _frame(60), None, _frame(200), _frame(210), _frame(215)]
        indices = motion_energy_phase_indices(frames, 0, 100, n_phases=N_FRAMES)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), N_FRAMES)
        self.assertEqual(indices, sorted(indices))

    def test_mismatched_frame_shapes_reject_rather_than_crash(self):
        frames = [_frame(0, size=8), _frame(50, size=16), _frame(100, size=8)]
        self.assertIsNone(motion_energy_phase_indices(frames, 0, 100))


if __name__ == "__main__":
    unittest.main()
