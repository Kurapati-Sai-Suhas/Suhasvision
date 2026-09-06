import unittest

from phase3_annotation_schema import (
    blank_record, validate_record, summarise,
    VALID_SINGLE_SHOT, NO_SHOT, MULTIPLE_SHOTS, SCENE_CUT, AMBIGUOUS,
    CONTACT_EXACT, CONTACT_ADJACENT, CONTACT_AMBIGUOUS, CONTACT_NOT_VISIBLE,
    CONTACT_TOLERANCE, CONTACT_WITH_FRAME, COHERENCE_USABLE,
    PHASE_LABELS, PIPELINE_PHASE_TO_ANNOTATED,
    CONF_HIGH, CONF_LOW,
)
import phase3_annotations as A


def rec(**kw):
    r = blank_record("x.mp4")
    r.update(coherence=VALID_SINGLE_SHOT, shot_start_frame=10, shot_end_frame=40,
             contact_frame=25, contact_type=CONTACT_EXACT,
             shot_confidence=CONF_HIGH, contact_confidence=CONF_HIGH)
    r.update(kw)
    return r


class TestSchemaValidation(unittest.TestCase):
    def test_a_consistent_record_passes(self):
        self.assertEqual(validate_record(rec()), [])

    def test_missing_required_field_is_caught(self):
        self.assertTrue(any("coherence" in e for e in validate_record(rec(coherence=None))))

    def test_valid_single_shot_requires_boundaries(self):
        errs = validate_record(rec(shot_start_frame=None, shot_end_frame=None,
                                   contact_frame=None, contact_type=CONTACT_NOT_VISIBLE))
        self.assertTrue(any("requires shot_start_frame" in e for e in errs))

    def test_non_single_shot_must_not_carry_boundaries(self):
        """A window that is not one shot has no defensible boundaries;
        inventing them would put fabricated targets into the localization
        metrics."""
        errs = validate_record(rec(coherence=MULTIPLE_SHOTS,
                                   contact_frame=None, contact_type=CONTACT_AMBIGUOUS))
        self.assertTrue(any("must not carry shot boundaries" in e for e in errs))

    def test_reversed_boundaries_are_caught(self):
        errs = validate_record(rec(shot_start_frame=40, shot_end_frame=10,
                                   contact_frame=None, contact_type=CONTACT_AMBIGUOUS))
        self.assertTrue(any("must exceed" in e for e in errs))

    def test_ambiguous_contact_must_not_carry_a_frame(self):
        """Forcing a number for a contact nobody can see injects noise that
        looks like data."""
        errs = validate_record(rec(contact_type=CONTACT_AMBIGUOUS, contact_frame=25))
        self.assertTrue(any("must not carry a contact_frame" in e for e in errs))

    def test_exact_contact_requires_a_frame(self):
        errs = validate_record(rec(contact_frame=None))
        self.assertTrue(any("requires a contact_frame" in e for e in errs))

    def test_contact_outside_shot_window_is_caught(self):
        errs = validate_record(rec(contact_frame=99))
        self.assertTrue(any("outside shot" in e for e in errs))

    def test_scene_cut_requires_a_frame(self):
        errs = validate_record(rec(coherence=SCENE_CUT, shot_start_frame=None,
                                   shot_end_frame=None, contact_frame=None,
                                   contact_type=CONTACT_NOT_VISIBLE,
                                   scene_cut=True, scene_cut_frames=[]))
        self.assertTrue(any("scene_cut_frames" in e for e in errs))

    def test_unknown_phase_label_is_rejected(self):
        errs = validate_record(rec(phases={"NOT_A_PHASE": [12, 15]}))
        self.assertTrue(any("not in" in e for e in errs))

    def test_phase_span_outside_shot_is_caught(self):
        errs = validate_record(rec(phases={PHASE_LABELS[0]: [5, 8]}))
        self.assertTrue(any("outside shot" in e for e in errs))


class TestNoFalsePrecision(unittest.TestCase):
    """The schema deliberately offers five phases, not the pipeline's seven."""

    def test_exactly_five_annotatable_phases(self):
        self.assertEqual(len(PHASE_LABELS), 5)

    def test_all_seven_pipeline_phases_map_to_annotatable_ones(self):
        self.assertEqual(len(PIPELINE_PHASE_TO_ANNOTATED), 7)
        for v in PIPELINE_PHASE_TO_ANNOTATED.values():
            self.assertIn(v, PHASE_LABELS)

    def test_stance_and_trigger_collapse_together(self):
        """Not separable by eye at 30fps — the mapping must reflect that."""
        self.assertEqual(PIPELINE_PHASE_TO_ANNOTATED["01_stance"],
                         PIPELINE_PHASE_TO_ANNOTATED["02_trigger"])

    def test_backlift_phases_collapse_together(self):
        self.assertEqual(PIPELINE_PHASE_TO_ANNOTATED["03_backlift_start"],
                         PIPELINE_PHASE_TO_ANNOTATED["04_full_backlift"])

    def test_exact_contact_has_tighter_tolerance_than_adjacent(self):
        self.assertLess(CONTACT_TOLERANCE[CONTACT_EXACT],
                        CONTACT_TOLERANCE[CONTACT_ADJACENT])

    def test_only_frame_bearing_contact_types_have_a_tolerance(self):
        self.assertEqual(set(CONTACT_TOLERANCE), set(CONTACT_WITH_FRAME))


class TestRealAnnotations(unittest.TestCase):
    """Guards on the actual labelled data, so a future edit cannot silently
    break the benchmark."""

    def setUp(self):
        self.records = A.build_records()

    def test_every_record_validates(self):
        bad = {r["video_id"]: validate_record(r) for r in self.records
               if validate_record(r)}
        self.assertEqual(bad, {}, f"invalid records: {bad}")

    def test_all_51_benchmark_clips_are_annotated(self):
        self.assertEqual(len(A.ANNOTATIONS), 51)
        self.assertEqual(len(set(A.ANNOTATIONS)), 51)

    def test_proposal_and_human_label_are_separate_fields(self):
        """Never merged — that separation is what makes proposal quality
        measurable rather than self-graded."""
        r = blank_record("x.mp4")
        self.assertIn("proposed_contact_frame", r)
        self.assertIn("contact_frame", r)

    def test_usable_subsets_are_non_empty_and_consistent(self):
        s = summarise(self.records)
        self.assertEqual(s["total_clips"], 51)
        self.assertGreater(s["usable_for_shot_localization"], 0)
        self.assertEqual(s["usable_for_shot_localization"], s["valid_single_shot"])
        self.assertEqual(s["usable_for_phase_metrics"], 0,
                         "no per-phase spans were annotated; metrics must stay uncomputable")

    def test_coherence_counts_sum_to_total(self):
        s = summarise(self.records)
        self.assertEqual(sum(s["coherence_distribution"].values()), 51)

    def test_contact_counts_sum_to_total(self):
        s = summarise(self.records)
        self.assertEqual(sum(s["contact_type_distribution"].values()), 51)

    def test_double_annotation_subset_is_a_subset(self):
        for cid in A.DOUBLE_ANNOTATION:
            self.assertIn(cid, A.ANNOTATIONS)

    def test_double_annotation_is_large_enough_to_be_informative(self):
        self.assertGreaterEqual(len(A.DOUBLE_ANNOTATION), 10)


if __name__ == "__main__":
    unittest.main()
