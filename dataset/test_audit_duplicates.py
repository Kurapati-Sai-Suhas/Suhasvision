import unittest

import pandas as pd

from audit_duplicates import analyze_keypoints, report_is_clean
from schema import CANONICAL_FRAME_NAMES, SEQ_LEN

# list() for slicing convenience in the tests below; the canonical source is
# schema.CANONICAL_FRAME_NAMES (audit H6).
CANONICAL_FRAMES = list(CANONICAL_FRAME_NAMES)


def _session_rows(name, frames, nose_x=0.5):
    return [{"session_name": name, "frame_name": frame, "nose_x": nose_x} for frame in frames]


class TestAnalyzeKeypoints(unittest.TestCase):
    def test_clean_file_reports_clean(self):
        df = pd.DataFrame(_session_rows("clean_a", CANONICAL_FRAMES)
                          + _session_rows("clean_b", CANONICAL_FRAMES))
        report = analyze_keypoints(df)
        self.assertTrue(report_is_clean(report))
        self.assertEqual(report["total_rows"], 2 * SEQ_LEN)
        self.assertEqual(report["unique_sessions"], 2)
        self.assertEqual(report["duplicated_frame_names"], {})
        self.assertEqual(report["exact_duplicate_row_count"], 0)
        self.assertEqual(report["overlong_sessions"], {})
        self.assertEqual(report["incomplete_sessions"], {})

    def test_reingested_session_flagged_in_every_relevant_bucket(self):
        # A straight re-ingestion append: the same 7 rows twice. This is the
        # exact shape idempotent ingestion (audit H1) now prevents.
        df = pd.DataFrame(_session_rows("clean", CANONICAL_FRAMES)
                          + _session_rows("dup", CANONICAL_FRAMES)
                          + _session_rows("dup", CANONICAL_FRAMES))
        report = analyze_keypoints(df)
        self.assertFalse(report_is_clean(report))
        self.assertEqual(sorted(report["duplicated_frame_names"]), ["dup"])
        self.assertEqual(report["duplicated_frame_names"]["dup"]["01_stance"], 2)
        self.assertEqual(report["exact_duplicate_row_count"], SEQ_LEN)
        self.assertEqual(report["exact_duplicate_row_sessions"], ["dup"])
        self.assertEqual(report["overlong_sessions"], {"dup": 2 * SEQ_LEN})
        self.assertNotIn("clean", report["overlong_sessions"])

    def test_incomplete_session_is_flagged(self):
        df = pd.DataFrame(_session_rows("short", CANONICAL_FRAMES[:3]))
        report = analyze_keypoints(df)
        self.assertFalse(report_is_clean(report))
        self.assertEqual(report["incomplete_sessions"], {"short": 3})

    def test_duplicated_frame_with_different_values_is_not_an_exact_duplicate(self):
        # A re-extraction with slightly different landmark values duplicates
        # the frame_name but not the full row -- both buckets must exist
        # separately so this shape isn't mistaken for a plain double-append.
        df = pd.DataFrame(_session_rows("varied", CANONICAL_FRAMES)
                          + _session_rows("varied", ["01_stance"], nose_x=0.9))
        report = analyze_keypoints(df)
        self.assertFalse(report_is_clean(report))
        self.assertEqual(report["exact_duplicate_row_count"], 0)
        self.assertEqual(report["duplicated_frame_names"]["varied"]["01_stance"], 2)
        self.assertEqual(report["overlong_sessions"], {"varied": SEQ_LEN + 1})


if __name__ == "__main__":
    unittest.main()
