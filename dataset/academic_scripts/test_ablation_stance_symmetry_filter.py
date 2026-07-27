import unittest

import numpy as np

from ablation_stance_symmetry_filter import derive_paired_split


class TestDerivePairedSplit(unittest.TestCase):
    def test_val_on_contains_exactly_the_held_out_identities_sessions(self):
        groups_off = np.array(["p1", "p1", "p2", "p2", "p3", "p3"])
        idx_val_off = np.array([2, 3])  # p2 held out
        groups_on = np.array(["p1", "p1", "p2", "p3", "p3"])  # p2 lost 1 session

        held_out, idx_train_on, idx_val_on = derive_paired_split(groups_off, idx_val_off, groups_on)

        self.assertEqual(held_out, {"p2"})
        self.assertEqual(sorted(idx_val_on.tolist()), [2])  # the one surviving p2 session
        self.assertEqual(sorted(idx_train_on.tolist()), [0, 1, 3, 4])  # p1 and p3 sessions

    def test_train_and_val_are_always_disjoint_and_cover_every_session(self):
        # The invariant the whole function exists to guarantee: no identity
        # can appear on both sides for the filtered variant either.
        groups_off = np.array(["p1", "p2", "p3", "p4", "p5"])
        idx_val_off = np.array([1, 3])
        groups_on = np.array(["p1", "p2", "p2", "p3", "p4", "p5", "p5"])

        _, idx_train_on, idx_val_on = derive_paired_split(groups_off, idx_val_off, groups_on)

        train_set, val_set = set(idx_train_on.tolist()), set(idx_val_on.tolist())
        self.assertEqual(train_set & val_set, set())
        self.assertEqual(train_set | val_set, set(range(len(groups_on))))
        # Every group-on identity assigned to val must actually be held out.
        held_out_ids = set(groups_off[idx_val_off])
        self.assertTrue(all(groups_on[i] in held_out_ids for i in val_set))
        self.assertTrue(all(groups_on[i] not in held_out_ids for i in train_set))

    def test_fold_is_dropped_when_held_out_identity_has_no_surviving_sessions(self):
        groups_off = np.array(["p1", "p1", "p2", "p2"])
        idx_val_off = np.array([2, 3])  # p2 held out
        groups_on = np.array(["p1", "p1"])  # p2 entirely filtered out

        held_out, idx_train_on, idx_val_on = derive_paired_split(groups_off, idx_val_off, groups_on)

        self.assertEqual(held_out, {"p2"})
        self.assertIsNone(idx_train_on)
        self.assertIsNone(idx_val_on)

    def test_fold_is_dropped_when_filtered_variant_has_nothing_left_to_train_on(self):
        # Degenerate but real-shape case: the held-out identity survives in
        # the filtered variant, but nobody ELSE does (nothing to train on).
        groups_off = np.array(["p1", "p2", "p2"])
        idx_val_off = np.array([1, 2])  # p2 held out
        groups_on = np.array(["p2", "p2"])  # only p2 survived filtering at all

        held_out, idx_train_on, idx_val_on = derive_paired_split(groups_off, idx_val_off, groups_on)

        self.assertIsNone(idx_train_on)
        self.assertIsNone(idx_val_on)

    def test_regression_two_independent_fold_calls_would_have_disagreed(self):
        # Reproduces the shape of the real bug found in review: naive
        # make_folds() calls on groups_off and groups_on independently pick
        # unrelated held-out identities for "fold 1" of each. This function
        # must instead force groups_on's split to agree with groups_off's,
        # even though groups_on's own natural grouping would differ.
        groups_off = np.array(["a", "a", "b", "b", "c", "c"])
        idx_val_off = np.array([0, 1])  # "a" held out in this fold

        # groups_on's identity composition differs from groups_off's (as it
        # would in practice, since filtering removes sessions unevenly) --
        # an independent make_folds() call on groups_on could easily pick
        # "b" or "c" as its own "first" held-out identity instead of "a".
        groups_on = np.array(["b", "b", "c", "c", "a"])

        held_out, idx_train_on, idx_val_on = derive_paired_split(groups_off, idx_val_off, groups_on)

        self.assertEqual(held_out, {"a"})
        # Regardless of groups_on's own composition, the val split must be
        # exactly "a"'s sessions -- not whatever an independent fold call
        # over groups_on would have chosen.
        self.assertEqual(idx_val_on.tolist(), [4])
        self.assertEqual(sorted(idx_train_on.tolist()), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
