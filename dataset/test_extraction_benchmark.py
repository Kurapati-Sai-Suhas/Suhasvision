import json
import os
import tempfile
import unittest

import extraction_benchmark as bench


def kp(hip=(0.5, 0.5), torso=0.2, lw=(0.4, 0.5), rw=(0.6, 0.5)):
    """One raw keypoint list: 33 landmarks of [x, y, z, visibility, presence]."""
    lms = [[hip[0], hip[1], 0.0, 1.0, 1.0] for _ in range(33)]
    lms[11] = [hip[0] - 0.05, hip[1] - torso, 0.0, 1.0, 1.0]   # left_shoulder
    lms[12] = [hip[0] + 0.05, hip[1] - torso, 0.0, 1.0, 1.0]   # right_shoulder
    lms[23] = [hip[0] - 0.05, hip[1], 0.0, 1.0, 1.0]           # left_hip
    lms[24] = [hip[0] + 0.05, hip[1], 0.0, 1.0, 1.0]           # right_hip
    lms[15] = [lw[0], lw[1], 0.0, 1.0, 1.0]                    # left_wrist
    lms[16] = [rw[0], rw[1], 0.0, 1.0, 1.0]                    # right_wrist
    return lms


class TestSequenceQuality(unittest.TestCase):
    def test_stationary_subject_is_perfectly_smooth(self):
        q = bench.sequence_quality([kp() for _ in range(7)])
        self.assertEqual(q["smoothness"], 0.0)
        self.assertEqual(q["max_center_jump_torsos"], 0.0)
        self.assertEqual(q["n_valid_frames"], 7)

    def test_constant_velocity_is_smooth_despite_movement(self):
        # Smoothness is a SECOND difference: steady drift must not be
        # penalised, or every camera pan would look like an identity switch.
        seq = [kp(hip=(0.3 + 0.02 * i, 0.5)) for i in range(7)]
        q = bench.sequence_quality(seq)
        self.assertAlmostEqual(q["smoothness"], 0.0, places=6)
        self.assertGreater(q["max_center_jump_torsos"], 0.0)

    def test_teleport_produces_a_large_jump(self):
        seq = [kp(hip=(0.2, 0.5)), kp(hip=(0.2, 0.5)), kp(hip=(0.9, 0.5)),
               kp(hip=(0.9, 0.5))]
        q = bench.sequence_quality(seq)
        # 0.7 normalized units over a ~0.2 torso = ~3.5 torso lengths.
        self.assertGreater(q["max_center_jump_torsos"], 3.0)

    def test_missing_frames_are_skipped_not_imputed(self):
        q = bench.sequence_quality([kp(), None, kp(), None])
        self.assertEqual(q["n_valid_frames"], 2)

    def test_too_few_frames_returns_nulls_not_zeros(self):
        q = bench.sequence_quality([kp()])
        self.assertIsNone(q["smoothness"])
        self.assertIsNone(q["max_center_jump_torsos"])

    def test_empty_input_is_handled(self):
        q = bench.sequence_quality(None)
        self.assertEqual(q["n_valid_frames"], 0)

    def test_scale_invariance(self):
        # The same motion by a camera-near and a camera-far subject must score
        # the same, or every metric becomes a proxy for subject distance.
        near = [kp(hip=(0.5 + 0.4 * i, 0.5), torso=0.4) for i in range(3)]
        far = [kp(hip=(0.5 + 0.2 * i, 0.5), torso=0.2) for i in range(3)]
        self.assertAlmostEqual(bench.sequence_quality(near)["max_center_jump_torsos"],
                               bench.sequence_quality(far)["max_center_jump_torsos"],
                               places=4)


class TestGtMetricsHonesty(unittest.TestCase):
    def test_no_annotations_reports_unavailable_not_zero(self):
        rows = [{"video_id": "a.mp4", "accepted": True, "selected_frames": [1, 2, 3]}]
        gt = bench.gt_metrics(rows, [])
        self.assertFalse(gt["available"])
        self.assertIsNone(gt["e2e_vser"])
        self.assertIsNone(gt["temporal_iou_mean"])
        self.assertIn("NOT computable", gt["note"])

    def test_temporal_iou_basic(self):
        self.assertEqual(bench.temporal_iou(0, 10, 0, 10), 1.0)
        self.assertEqual(bench.temporal_iou(0, 10, 20, 30), 0.0)


class TestVisualAudit(unittest.TestCase):
    def _write(self, records):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f)
        self.addCleanup(os.remove, path)
        return path

    def test_wrong_person_rate(self):
        path = self._write([
            {"video_id": "a.mp4", "config": "A3", "subject_correct": True},
            {"video_id": "b.mp4", "config": "A3", "subject_correct": False},
            {"video_id": "c.mp4", "config": "A3", "subject_correct": None},
        ])
        m = bench.visual_audit_metrics(bench.load_visual_audit(path), "A3")
        self.assertEqual(m["n_audited"], 3)
        self.assertEqual(m["n_decided"], 2)
        self.assertEqual(m["n_undecidable"], 1)
        self.assertEqual(m["wrong_person_rate"], 0.5)

    def test_undecidable_does_not_count_as_correct(self):
        path = self._write([{"video_id": "a.mp4", "config": "A3", "subject_correct": None}])
        m = bench.visual_audit_metrics(bench.load_visual_audit(path), "A3")
        self.assertIsNone(m["wrong_person_rate"])

    def test_missing_field_is_rejected(self):
        path = self._write([{"video_id": "a.mp4", "config": "A3"}])
        with self.assertRaises(ValueError):
            bench.load_visual_audit(path)

    def test_unaudited_config_reports_unavailable(self):
        path = self._write([{"video_id": "a.mp4", "config": "A3", "subject_correct": True}])
        m = bench.visual_audit_metrics(bench.load_visual_audit(path), "A0")
        self.assertFalse(m["available"])


class TestSplitDiscipline(unittest.TestCase):
    def test_split_filter_excludes_other_split(self):
        rows = [{"video_id": "a.mp4", "accepted": True},
                {"video_id": "b.mp4", "accepted": True}]
        anns = [{"video_id": "a.mp4", "shot_start_frame": 0, "shot_end_frame": 10, "split": "dev"},
                {"video_id": "b.mp4", "shot_start_frame": 0, "shot_end_frame": 10, "split": "eval"}]
        self.assertEqual(bench.summarize(rows, anns, split="dev")["gt_free"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
