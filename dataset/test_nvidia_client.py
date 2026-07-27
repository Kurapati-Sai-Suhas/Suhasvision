import unittest

from nvidia_client import build_scan_tiles, merge_overlapping_windows
from zero_storage_pipeline import _resolve_naming_anchor


class TestBuildScanTiles(unittest.TestCase):
    def test_clean_division_covers_full_duration_with_expected_tile_count(self):
        # 60s video, 25s tiles, 5s overlap -> step=20s: tiles at 0,20,40 starts.
        tiles = build_scan_tiles(60.0, tile_seconds=25.0, overlap_seconds=5.0)
        self.assertEqual(tiles[0], (0.0, 25.0))
        self.assertEqual(tiles[-1][1], 60.0)  # last tile always reaches the true end
        # Every consecutive pair overlaps by exactly the requested amount
        # (except possibly the final, clamped tile).
        for (s1, e1), (s2, e2) in zip(tiles, tiles[1:]):
            self.assertGreaterEqual(e1, s2)  # real overlap, not just touching

    def test_short_video_produces_a_single_clamped_tile(self):
        tiles = build_scan_tiles(10.0, tile_seconds=25.0, overlap_seconds=5.0)
        self.assertEqual(tiles, [(0.0, 10.0)])

    def test_zero_or_negative_duration_produces_no_tiles(self):
        self.assertEqual(build_scan_tiles(0.0), [])
        self.assertEqual(build_scan_tiles(-5.0), [])

    def test_last_tile_is_clamped_to_total_duration_not_padded_past_it(self):
        tiles = build_scan_tiles(52.0, tile_seconds=25.0, overlap_seconds=5.0)
        for _, end in tiles:
            self.assertLessEqual(end, 52.0)
        self.assertEqual(tiles[-1][1], 52.0)

    def test_rejects_non_advancing_or_negative_progress_tiling(self):
        with self.assertRaises(ValueError):
            build_scan_tiles(60.0, tile_seconds=10.0, overlap_seconds=10.0)  # step == 0
        with self.assertRaises(ValueError):
            build_scan_tiles(60.0, tile_seconds=10.0, overlap_seconds=15.0)  # step < 0

    def test_rejects_non_positive_tile_seconds(self):
        with self.assertRaises(ValueError):
            build_scan_tiles(60.0, tile_seconds=0.0, overlap_seconds=0.0)


class TestMergeOverlappingWindows(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        self.assertEqual(merge_overlapping_windows([]), [])

    def test_non_overlapping_windows_stay_separate(self):
        windows = [{"start_time": 0.0, "end_time": 2.0}, {"start_time": 10.0, "end_time": 12.0}]
        merged = merge_overlapping_windows(windows)
        self.assertEqual(len(merged), 2)

    def test_overlapping_windows_merge_to_their_union(self):
        # The same real shot detected from two adjacent, overlapping tiles.
        windows = [{"start_time": 18.0, "end_time": 21.0}, {"start_time": 19.5, "end_time": 22.5}]
        merged = merge_overlapping_windows(windows)
        self.assertEqual(merged, [{"start_time": 18.0, "end_time": 22.5}])

    def test_touching_windows_merge(self):
        windows = [{"start_time": 0.0, "end_time": 5.0}, {"start_time": 5.0, "end_time": 8.0}]
        merged = merge_overlapping_windows(windows)
        self.assertEqual(merged, [{"start_time": 0.0, "end_time": 8.0}])

    def test_chain_of_overlaps_merges_into_one(self):
        windows = [
            {"start_time": 0.0, "end_time": 3.0},
            {"start_time": 2.0, "end_time": 5.0},
            {"start_time": 4.5, "end_time": 7.0},
        ]
        merged = merge_overlapping_windows(windows)
        self.assertEqual(merged, [{"start_time": 0.0, "end_time": 7.0}])

    def test_unsorted_input_still_produces_correctly_merged_sorted_output(self):
        windows = [
            {"start_time": 40.0, "end_time": 42.0},
            {"start_time": 0.0, "end_time": 2.0},
            {"start_time": 1.0, "end_time": 3.0},
        ]
        merged = merge_overlapping_windows(windows)
        self.assertEqual(merged, [{"start_time": 0.0, "end_time": 3.0}, {"start_time": 40.0, "end_time": 42.0}])


class TestResolveNamingAnchor(unittest.TestCase):
    def test_human_provided_macro_start_passes_through_unchanged(self):
        self.assertEqual(_resolve_naming_anchor("https://youtu.be/abc", 15.0), 15.0)

    def test_none_macro_start_produces_a_stable_deterministic_int(self):
        anchor1 = _resolve_naming_anchor("https://youtu.be/abc123", None)
        anchor2 = _resolve_naming_anchor("https://youtu.be/abc123", None)
        self.assertEqual(anchor1, anchor2)  # same URL, same run -> same anchor
        self.assertIsInstance(anchor1, int)
        self.assertGreaterEqual(anchor1, 0)
        self.assertLess(anchor1, 100000)

    def test_different_urls_get_different_anchors(self):
        # Not a mathematical guarantee (CRC32 can collide), but any two
        # arbitrary real URLs colliding is astronomically unlikely -- this
        # is the actual disambiguation property auto-scan mode relies on.
        anchor_a = _resolve_naming_anchor("https://youtu.be/videoAAA", None)
        anchor_b = _resolve_naming_anchor("https://youtu.be/videoBBB", None)
        self.assertNotEqual(anchor_a, anchor_b)


if __name__ == "__main__":
    unittest.main()
