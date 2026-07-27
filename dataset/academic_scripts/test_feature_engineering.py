import unittest

from feature_engineering import normalize_frame_name


class TestNormalizeFrameName(unittest.TestCase):
    def test_strips_frame_prefix_and_jpg_suffix(self):
        self.assertEqual(normalize_frame_name("frame_01_stance.jpg"), "01_stance")

    def test_bare_current_pipeline_format_is_unchanged(self):
        self.assertEqual(normalize_frame_name("01_stance"), "01_stance")

    def test_both_conventions_normalize_to_the_same_value(self):
        # Regression test for a real, severe bug: feature_engineering.py's
        # canonical_frames list only matched ONE of these two real,
        # coexisting conventions, so reindexing against it silently wiped
        # ALL metadata (scores, bowling_type) to NaN for every session
        # using the other -- confirmed against real data to affect 83 of
        # 197 sessions (42%) before this fix.
        old_style = normalize_frame_name("frame_07_followthrough.jpg")
        new_style = normalize_frame_name("07_followthrough")
        self.assertEqual(old_style, new_style)
        self.assertEqual(old_style, "07_followthrough")

    def test_case_insensitive_extension(self):
        self.assertEqual(normalize_frame_name("frame_03_backlift_start.JPG"), "03_backlift_start")


if __name__ == "__main__":
    unittest.main()
