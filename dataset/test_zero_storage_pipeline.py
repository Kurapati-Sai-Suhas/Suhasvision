import contextlib
import csv
import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from test_subject_selection import make_pose

import zero_storage_pipeline as zsp
from extraction_config import get_config
from zero_storage_pipeline import (
    N_FRAMES,
    WRIST_SPEED_SAMPLING_ENABLED,
    _expected_keypoints_header,
    _parse_time,
    _peak_speed_frame,
    collect_phase_frames,
    ensure_output_csv_ready,
    extract_features_from_image_array,
    load_ingested_sessions,
    redistribute_phase_indices,
    sessions_matching_prefix,
)


class TestWristSpeedSamplingDisabledByDefault(unittest.TestCase):
    def test_disabled_by_default(self):
        # Regression guard, not a design opinion: confirmed via real
        # footage + landmark overlay (2026-07-18 review) that the coarse
        # scan can lock onto a non-batsman person in frame (MediaPipe's
        # num_poses=1 has no guarantee of tracking the batsman), and a
        # candidate mitigation (wrist-speed vs. hip-speed ratio) was tested
        # against that real failure case and found unable to tell a real
        # swing apart from someone casually swinging an arm while walking.
        # No validated fix exists -- this must stay off until one does.
        # If you're flipping this to True, you have a validated fix; update
        # this test (and re-verify against real multi-person footage, not
        # just unit tests) rather than deleting it.
        self.assertFalse(WRIST_SPEED_SAMPLING_ENABLED)


class TestDistinctPhaseFrameGuard(unittest.TestCase):
    """select_phase_frames must never return a phase list containing the same
    frame twice. Two phases sharing one frame produce identical landmarks, so
    every velocity feature between them is exactly zero -- a stillness that
    never happened, fed straight into the (7, 30) tensor.

    Found by the Phase-1 benchmark: duplicate frames on 16/102 clips under A2
    (contact anchored near a window edge) and 6/102 under A1 (motion energy
    concentrated in one interval). Both producers are monotonic but neither
    guaranteed distinctness."""

    def setUp(self):
        self.cfg_a2 = get_config("A2")
        self.cfg_a1 = get_config("A1")

    def test_contact_anchor_at_window_start_falls_back_instead_of_duplicating(self):
        # Contact detected 2 frames into a 100-frame window: the five
        # pre-contact phases would collapse onto ~2 distinct frames.
        collapsed = zsp.redistribute_phase_indices(0, 100, 2, n_phases=7)
        self.assertLess(len(set(collapsed)), 7, "precondition: this must collapse")

        with mock.patch.object(zsp, "collect_coarse_scan",
                               return_value=([0], [[]], [None], 0)), \
             mock.patch.object(zsp, "select_subject",
                               return_value=(["pose"], {"reason": "selected",
                                                        "winner_track": {"poses": {0: "pose"}},
                                                        "n_tracks": 1, "multi_track": False})), \
             mock.patch.object(zsp.contact_detection, "detect_contact",
                               return_value={"frame_index": 2, "valid": True, "confidence": 0.9,
                                             "peak_fraction": 0.02, "peak_prominence": 3.0,
                                             "bilateral_agreement": 0.8, "reason": "ok"}), \
             mock.patch.object(zsp, "motion_energy_phase_indices", return_value=None):
            indices, method = zsp.select_phase_frames(None, 0, 100, "t", self.cfg_a2)

        self.assertNotEqual(method, "contact_anchored")
        self.assertEqual(len(set(indices)), 7)

    def test_degenerate_motion_energy_falls_back_to_uniform(self):
        with mock.patch.object(zsp, "collect_coarse_scan",
                               return_value=([0], [[]], [None], 0)), \
             mock.patch.object(zsp, "motion_energy_phase_indices",
                               return_value=[0, 0, 0, 5, 5, 5, 100]):
            indices, method = zsp.select_phase_frames(None, 0, 100, "t", self.cfg_a1)

        self.assertEqual(method, "uniform")
        self.assertEqual(len(set(indices)), 7)

    def test_distinct_motion_energy_is_still_used(self):
        good = [0, 10, 20, 30, 40, 50, 100]
        with mock.patch.object(zsp, "collect_coarse_scan",
                               return_value=([0], [[]], [None], 0)), \
             mock.patch.object(zsp, "motion_energy_phase_indices", return_value=good):
            indices, method = zsp.select_phase_frames(None, 0, 100, "t", self.cfg_a1)

        self.assertEqual(method, "motion_energy")
        self.assertEqual(indices, good)

    def test_a0_uniform_path_is_untouched(self):
        indices, method = zsp.select_phase_frames(None, 100, 400, "t", get_config("A0"))
        self.assertEqual(method, "uniform")
        self.assertEqual(indices, [100, 150, 200, 250, 300, 350, 400])


class TestPeakSpeedFrame(unittest.TestCase):
    def test_finds_the_fastest_consecutive_pair(self):
        # Frames 10, 11, 12, 13 -- a big jump between 11 and 12 should win.
        samples = [
            (10, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
            (11, np.array([0.01, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),  # small left move
            (12, np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),   # big left move
            (13, np.array([0.51, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),  # small again
        ]
        self.assertEqual(_peak_speed_frame(samples), 12)

    def test_right_wrist_can_win_over_left(self):
        samples = [
            (5, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
            (6, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),  # right wrist jumps
        ]
        self.assertEqual(_peak_speed_frame(samples), 6)

    def test_skips_over_missing_detections(self):
        # A None in the middle breaks that one pairing but shouldn't crash
        # or corrupt comparisons across the gap.
        samples = [
            (1, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
            None,
            (3, np.array([2.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
            (4, np.array([2.01, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        ]
        # Only the (3, 4) pair is comparable (1 is isolated by the None on
        # both sides) -- its tiny speed is the only candidate, so it wins
        # by being the only valid pair, not because it's fast.
        self.assertEqual(_peak_speed_frame(samples), 4)

    def test_all_missing_returns_none(self):
        self.assertIsNone(_peak_speed_frame([None, None, None]))

    def test_single_sample_returns_none(self):
        self.assertIsNone(_peak_speed_frame([(1, np.zeros(3), np.zeros(3))]))

    def test_empty_returns_none(self):
        self.assertIsNone(_peak_speed_frame([]))


class TestRedistributePhaseIndices(unittest.TestCase):
    def test_preserves_phase_count(self):
        indices = redistribute_phase_indices(100, 200, 170, n_phases=7)
        self.assertEqual(len(indices), 7)

    def test_contact_lands_exactly_on_peak_frame(self):
        indices = redistribute_phase_indices(100, 200, 170, n_phases=7)
        self.assertEqual(indices[5], 170)  # index 5 = "06_contact"

    def test_followthrough_lands_exactly_on_end_frame(self):
        indices = redistribute_phase_indices(100, 200, 170, n_phases=7)
        self.assertEqual(indices[6], 200)  # index 6 = "07_followthrough"

    def test_stance_lands_exactly_on_start_frame(self):
        indices = redistribute_phase_indices(100, 200, 170, n_phases=7)
        self.assertEqual(indices[0], 100)  # index 0 = "01_stance"

    def test_indices_are_non_decreasing(self):
        indices = redistribute_phase_indices(100, 253, 211, n_phases=7)
        self.assertEqual(indices, sorted(indices))

    def test_matches_uniform_formula_exactly_when_peak_is_at_the_uniform_contact_fraction(self):
        # Hand-derived property: if the detected peak happens to sit at the
        # SAME fraction (5/6 of the window) the old uniform formula would
        # already have placed "contact" at, the two formulas must produce
        # byte-identical indices -- adaptive sampling shouldn't change
        # anything when there's nothing to adapt to.
        start_frame, end_frame = 100, 700  # window_frames = 600, divisible by 6
        window_frames = end_frame - start_frame
        uniform = [int(start_frame + (window_frames * i / (N_FRAMES - 1))) for i in range(N_FRAMES)]
        peak_frame = uniform[5]  # where uniform sampling put "contact"

        adaptive = redistribute_phase_indices(start_frame, end_frame, peak_frame, n_phases=N_FRAMES)

        self.assertEqual(adaptive, uniform)

    def test_hand_computed_example(self):
        # start=0, end=60, peak=50 (contact happens near the end of the
        # window -- a fast, late swing). 5 "before" phases spread linearly
        # across [0, 50] at fractions 0, 1/5, 2/5, 3/5, 4/5:
        #   0, 10, 20, 30, 40
        # then contact=50, followthrough=60.
        indices = redistribute_phase_indices(0, 60, 50, n_phases=7)
        self.assertEqual(indices, [0, 10, 20, 30, 40, 50, 60])

    def test_peak_at_start_frame_does_not_crash(self):
        # Degenerate: detected peak sits at the very start of the window.
        # All "before" phases collapse onto start_frame -- a tight cluster,
        # not a crash.
        indices = redistribute_phase_indices(100, 200, 100, n_phases=7)
        self.assertEqual(indices, [100, 100, 100, 100, 100, 100, 200])

    def test_peak_at_end_frame_does_not_crash(self):
        indices = redistribute_phase_indices(100, 200, 200, n_phases=7)
        self.assertEqual(indices[-2:], [200, 200])

    def test_peak_outside_window_gets_clamped(self):
        # A caller passing an out-of-range peak (shouldn't happen from
        # find_wrist_speed_peak_frame, which only samples within the
        # window, but defended against anyway) must not silently produce
        # indices outside [start_frame, end_frame].
        below = redistribute_phase_indices(100, 200, 50, n_phases=7)
        above = redistribute_phase_indices(100, 200, 999, n_phases=7)
        self.assertTrue(all(100 <= i <= 200 for i in below))
        self.assertTrue(all(100 <= i <= 200 for i in above))
        self.assertEqual(below[5], 100)  # clamped to start_frame
        self.assertEqual(above[5], 200)  # clamped to end_frame


class _FakeCap:
    """Duck-typed stand-in for cv2.VideoCapture: set() records the requested
    frame position, read() returns whatever (ret, frame) was programmed for
    it -- lets the frame-collection loop be tested without a real video."""

    def __init__(self, results):
        self._results = results
        self._pos = 0

    def set(self, prop, value):
        self._pos = int(value)

    def read(self):
        return self._results.get(self._pos, (False, None))


class TestCollectPhaseFrames(unittest.TestCase):
    """Regression tests for audit C3: a failed cap.read() used to `continue`,
    silently shrinking the frame list and shifting every later frame onto the
    wrong phase label. Failed reads must keep their slot as None."""

    def setUp(self):
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.indices = [0, 10, 20, 30, 40, 50, 60]

    def test_all_reads_ok_returns_one_frame_per_index(self):
        cap = _FakeCap({i: (True, self.frame.copy()) for i in self.indices})
        collected = collect_phase_frames(cap, self.indices, "test_session_ok")
        self.assertEqual(len(collected), N_FRAMES)
        self.assertTrue(all(f is not None for f in collected))

    def test_failed_read_keeps_its_slot_as_none(self):
        results = {i: (True, self.frame.copy()) for i in self.indices}
        results[30] = (False, None)  # phase 4 (full_backlift) fails to decode
        collected = collect_phase_frames(_FakeCap(results), self.indices, "test_session_gap")
        # Positional integrity is the whole point: length unchanged, ONLY
        # slot 3 missing -- before the fix, slots 3..5 would have held the
        # frames for phases 5..7 and slot 6 wouldn't exist at all.
        self.assertEqual(len(collected), N_FRAMES)
        self.assertIsNone(collected[3])
        for i in (0, 1, 2, 4, 5, 6):
            self.assertIsNotNone(collected[i], f"slot {i} should still hold its own phase's frame")

    def test_multiple_failed_reads_all_keep_their_slots(self):
        results = {i: (True, self.frame.copy()) for i in self.indices}
        results[0] = (False, None)
        results[60] = (False, None)
        collected = collect_phase_frames(_FakeCap(results), self.indices, "test_session_edges")
        self.assertEqual(len(collected), N_FRAMES)
        self.assertIsNone(collected[0])
        self.assertIsNone(collected[6])


class TestLabellingSkippedForUndecodableFrames(unittest.TestCase):
    """Follow-up to audit C3 (verification pass): when a frame read fails but
    keypoint interpolation SUCCEEDS, frames_rgb still holds a None slot.
    Passing that to label_session_frames would crash on Image.fromarray(None)
    AFTER the keypoints were already written, killing the video's remaining
    shots -- so labelling must be skipped for such sessions (no pixels exist
    for that phase, and the label prompt hardcodes a 7-frame phase mapping)."""

    def test_label_call_is_skipped_but_keypoints_still_written(self):
        import zero_storage_pipeline as zsp

        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        class _Cap:
            """timestamps 0..1s at 30fps -> indices [0, 5, 10, 15, 20, 25, 30];
            the read at frame 15 (phase 4, full_backlift) fails to decode."""
            def __init__(self):
                self._pos = 0

            def get(self, prop):
                return 30.0

            def set(self, prop, value):
                self._pos = int(value)

            def read(self):
                if self._pos == 15:
                    return False, None
                return True, frame.copy()

            def release(self):
                pass

        fake_keypoints = [[[0.5, 0.5, 0.0, 1.0, 1.0]] * 33 for _ in range(N_FRAMES)]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = os.path.join(tmpdir, "keypoints.csv")  # does not exist yet
            # redirect_stdout: the pipeline's user-facing prints contain
            # emoji, which crash under consoles whose codepage can't encode
            # them (cp1252) -- tests must not depend on console encoding.
            with mock.patch.object(zsp.cv2, "VideoCapture", return_value=_Cap()), \
                 mock.patch.object(zsp, "OUTPUT_CSV", out_csv), \
                 mock.patch.object(zsp, "extract_features_from_image_array",
                                   return_value=(True, fake_keypoints, ["phase_4"])), \
                 mock.patch.object(zsp, "label_session_frames") as label_mock, \
                 contextlib.redirect_stdout(io.StringIO()):
                frames_saved, interpolated = zsp.extract_keypoints_in_memory(
                    "unused.mp4", {"start_time": 0, "end_time": 1},
                    "testbatsman", "frontview", 1, 0)

            self.assertEqual(frames_saved, N_FRAMES)
            self.assertEqual(interpolated, ["phase_4"])
            label_mock.assert_not_called()
            with open(out_csv, newline="", encoding="utf-8") as f:
                written_rows = list(csv.reader(f))
            # ensure_output_csv_ready (called at the write site since audit
            # H10 moved it out of import time) wrote the header, then the 7
            # keypoint rows were appended.
            self.assertEqual(len(written_rows), N_FRAMES + 1)
            self.assertEqual(written_rows[0], _expected_keypoints_header())


class TestExtractFeaturesHandlesMissingFrames(unittest.TestCase):
    def test_none_frames_flow_through_as_missing_detections(self):
        # None slots (undecodable frames, audit C3) must be treated exactly
        # like failed pose detections, not crashed on. Since Milestone 4,
        # all-7-missing is rejected by subject selection ("no pose candidates
        # in any frame") before the interpolation rules would have hard-
        # rejected it anyway -- same outcome, earlier and better-explained.
        success, payload, interpolated = extract_features_from_image_array(
            [None] * N_FRAMES, "test_session_all_none"
        )
        self.assertFalse(success)
        self.assertIsInstance(payload, str)
        self.assertIsNone(interpolated)


class TestParseTime(unittest.TestCase):
    """_parse_time (audit M6): module-level now, with narrowed exception types
    instead of the old bare `except:` that swallowed everything."""

    def test_plain_values(self):
        self.assertEqual(_parse_time(2.5), 2.5)
        self.assertEqual(_parse_time("3.25"), 3.25)
        self.assertEqual(_parse_time(0), 0.0)

    def test_list_takes_first_element(self):
        self.assertEqual(_parse_time([4.5, 9.0]), 4.5)

    def test_malformed_values_degrade_to_zero(self):
        self.assertEqual(_parse_time([]), 0.0)          # IndexError path
        self.assertEqual(_parse_time({"t": 1}), 0.0)    # dict special-case
        self.assertEqual(_parse_time(None), 0.0)
        self.assertEqual(_parse_time("garbage"), 0.0)   # ValueError path
        self.assertEqual(_parse_time(["garbage"]), 0.0)


class TestEnsureOutputCsvReady(unittest.TestCase):
    """ensure_output_csv_ready (audit H10): the header bootstrap/validation
    that used to run as an import-time side effect in the current working
    directory."""

    def test_creates_missing_file_with_expected_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kp.csv")
            ensure_output_csv_ready(path)
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
        self.assertEqual(rows, [_expected_keypoints_header()])

    def test_noop_when_header_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kp.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(_expected_keypoints_header())
                writer.writerow(["some_session", "01_stance"] + ["0"] * (len(_expected_keypoints_header()) - 2))
            with open(path, "rb") as f:
                before = f.read()
            ensure_output_csv_ready(path)
            with open(path, "rb") as f:
                after = f.read()
        self.assertEqual(before, after)  # existing data untouched

    def test_raises_on_header_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kp.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["wrong", "header"])
            with self.assertRaises(RuntimeError):
                ensure_output_csv_ready(path)


class TestLoadIngestedSessions(unittest.TestCase):
    def test_missing_file_returns_empty_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(load_ingested_sessions(os.path.join(tmpdir, "nope.csv")), set())

    def test_reads_unique_session_names_skipping_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kp.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["session_name", "frame_name", "nose_x"])
                writer.writerow(["a_front_10s_01", "01_stance", "0.5"])
                writer.writerow(["a_front_10s_01", "02_trigger", "0.5"])
                writer.writerow(["b_front_30s_01", "01_stance", "0.5"])
            self.assertEqual(load_ingested_sessions(path),
                             {"a_front_10s_01", "b_front_30s_01"})


class TestSessionsMatchingPrefix(unittest.TestCase):
    def test_matches_only_this_rows_sessions(self):
        sessions = {"a_front_10s_01", "a_front_10s_02", "a_front_105s_01", "b_front_10s_01"}
        self.assertEqual(sessions_matching_prefix("a_front_10s_", sessions),
                         ["a_front_10s_01", "a_front_10s_02"])

    def test_digit_prefix_collision_is_impossible(self):
        # The trailing "s_" terminator is what makes prefix matching safe:
        # macro_start 10 must never match sessions from macro_start 105 (or
        # 1 match 10) just because the digits share a prefix.
        self.assertEqual(sessions_matching_prefix("a_front_10s_", {"a_front_105s_01"}), [])
        self.assertEqual(sessions_matching_prefix("a_front_1s_", {"a_front_10s_01"}), [])


class TestRunPipelineIdempotency(unittest.TestCase):
    """Audit H1 regression tests: re-running a batch must skip rows whose
    sessions already exist in keypoints.csv (unless --force), and a row listed
    twice in one batch file must never be processed twice. process_single_row
    is mocked out, so no network/MediaPipe work happens."""

    BATCH_HEADER = "url,batsman_name,angle,macro_start_sec,macro_end_sec,bowling_type\n"
    ROW1 = "https://yt/v1,playerone,frontview,10,25,fast\n"
    ROW2 = "https://yt/v2,playertwo,frontview,30,45,spin\n"

    def _run(self, batch_rows, existing_sessions, force=False):
        """Runs run_zero_storage_pipeline against temp CSV_FILE/OUTPUT_CSV,
        with process_single_row mocked and the inter-row sleep removed.
        Returns the process_single_row mock."""
        import zero_storage_pipeline as zsp

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_path = os.path.join(tmpdir, "batch_urls.csv")
            with open(batch_path, "w", encoding="utf-8") as f:
                f.write(self.BATCH_HEADER)
                f.writelines(batch_rows)

            keypoints_path = os.path.join(tmpdir, "keypoints.csv")
            header = _expected_keypoints_header()
            with open(keypoints_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for session in existing_sessions:
                    writer.writerow([session, "01_stance"] + ["0"] * (len(header) - 3) + [""])

            # redirect_stdout: the batch runner's prints contain emoji, which
            # crash under consoles whose codepage can't encode them (cp1252)
            # -- tests must not depend on console encoding.
            with mock.patch.object(zsp, "CSV_FILE", batch_path), \
                 mock.patch.object(zsp, "OUTPUT_CSV", keypoints_path), \
                 mock.patch.object(zsp, "process_single_row") as proc, \
                 mock.patch.object(zsp.time, "sleep"), \
                 contextlib.redirect_stdout(io.StringIO()):
                zsp.run_zero_storage_pipeline(force=force)
        return proc

    def test_already_ingested_row_is_skipped_on_rerun(self):
        proc = self._run([self.ROW1, self.ROW2],
                         existing_sessions=["playerone_frontview_10s_01"])
        self.assertEqual(proc.call_count, 1)
        self.assertEqual(proc.call_args[0][1], "playertwo")  # only the new row ran

    def test_fresh_rows_all_process(self):
        proc = self._run([self.ROW1, self.ROW2], existing_sessions=[])
        self.assertEqual(proc.call_count, 2)

    def test_force_reingests_already_ingested_rows(self):
        proc = self._run([self.ROW1, self.ROW2],
                         existing_sessions=["playerone_frontview_10s_01"], force=True)
        self.assertEqual(proc.call_count, 2)

    def test_duplicate_batch_rows_processed_once(self):
        proc = self._run([self.ROW1, self.ROW1], existing_sessions=[])
        self.assertEqual(proc.call_count, 1)

    def test_duplicate_batch_rows_still_deduped_under_force(self):
        # --force means "re-ingest despite history", not "ingest twice in one run"
        proc = self._run([self.ROW1, self.ROW1], existing_sessions=[], force=True)
        self.assertEqual(proc.call_count, 1)

    def test_session_at_different_macro_start_does_not_block(self):
        # playerone at 105s exists; the row is for 10s -- must still process
        proc = self._run([self.ROW1], existing_sessions=["playerone_frontview_105s_01"])
        self.assertEqual(proc.call_count, 1)


class TestModulePathsAreAnchored(unittest.TestCase):
    def test_data_paths_are_absolute_and_module_anchored(self):
        # Audit H10: every data file this module touches must resolve inside
        # dataset/ regardless of the importer's current working directory.
        import zero_storage_pipeline as zsp

        module_dir = os.path.dirname(os.path.abspath(zsp.__file__))
        for value in (zsp.OUTPUT_CSV, zsp.CSV_FILE, zsp.LABELS_CSV,
                      zsp.COOKIES_TXT, zsp.FULL_YT_VIDEO, zsp.AI_CHUNK_VIDEO,
                      zsp.model_path):
            self.assertTrue(os.path.isabs(value), f"{value} is not absolute")
            self.assertEqual(os.path.dirname(value), module_dir)


class _FakeDetectionResult:
    def __init__(self, poses):
        self.pose_landmarks = poses


class _FakeMultiPoseDetector:
    """Stand-in for MediaPipe's PoseLandmarker (injected via the `detector`
    parameter): returns the scripted candidate list for each successive
    detect() call."""

    def __init__(self, per_frame_candidates):
        self._per_frame = list(per_frame_candidates)
        self._calls = 0

    def detect(self, mp_image):
        poses = self._per_frame[self._calls]
        self._calls += 1
        return _FakeDetectionResult(poses)


class TestSubjectSelectionIntegration(unittest.TestCase):
    """Milestone 4 (audit C1) end-to-end through
    extract_features_from_image_array, with the detector faked so the
    multi-person decision path runs without real footage."""

    def _run_pipeline(self, per_frame_candidates, session_name):
        import zero_storage_pipeline as zsp
        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(N_FRAMES)]
        # Injected via the detector parameter (Milestone 5) — the shared
        # singleton is bypassed entirely, no patching of MediaPipe needed.
        fake = _FakeMultiPoseDetector(per_frame_candidates)
        return zsp.extract_features_from_image_array(frames, session_name, detector=fake)

    def test_multi_person_scene_selects_dominant_subject_end_to_end(self):
        per_frame = [[make_pose(0.3, 0.5, torso=0.25), make_pose(0.75, 0.5, torso=0.15)]
                     for _ in range(N_FRAMES)]
        success, raw_keypoints, interpolated = self._run_pipeline(per_frame, "test_multi_person")
        self.assertTrue(success)
        self.assertEqual(interpolated, [])
        for frame_kp in raw_keypoints:
            hip_x = (frame_kp[23][0] + frame_kp[24][0]) / 2.0
            self.assertAlmostEqual(hip_x, 0.3)  # subject A in EVERY frame — never the rival

    def test_ambiguous_scene_rejects_end_to_end(self):
        per_frame = [[make_pose(0.3, 0.5, torso=0.2), make_pose(0.75, 0.5, torso=0.18)]
                     for _ in range(N_FRAMES)]
        success, payload, interpolated = self._run_pipeline(per_frame, "test_ambiguous")
        self.assertFalse(success)
        self.assertIn("No single trackable subject", payload)
        self.assertIsNone(interpolated)

    def test_single_person_scene_still_succeeds(self):
        per_frame = [[make_pose(0.5, 0.5, torso=0.2)] for _ in range(N_FRAMES)]
        success, raw_keypoints, interpolated = self._run_pipeline(per_frame, "test_single_person")
        self.assertTrue(success)
        self.assertEqual(len(raw_keypoints), N_FRAMES)


class TestSharedDetectorLifecycle(unittest.TestCase):
    """Milestone 5 (audits H5/H10): ONE PoseLandmarker per process, created
    lazily, reused by every extraction, closed only by reset_shared_detector.
    All tests here use a patched constructor so no real MediaPipe model is
    built; setUp/tearDown reset the singleton so fake and real detectors can
    never leak into each other's tests."""

    def setUp(self):
        import zero_storage_pipeline as zsp
        self.zsp = zsp
        zsp.reset_shared_detector()
        self.addCleanup(zsp.reset_shared_detector)

    def _patched_constructor(self, delay=0.0):
        constructed = []

        def fake_create(options):
            if delay:
                import time
                time.sleep(delay)
            detector = mock.MagicMock(name=f"fake_detector_{len(constructed)}")
            constructed.append(detector)
            return detector

        patcher = mock.patch.object(self.zsp.vision.PoseLandmarker, "create_from_options",
                                    side_effect=fake_create)
        self.addCleanup(patcher.stop)
        patcher.start()
        return constructed

    def test_singleton_identity_and_single_construction(self):
        constructed = self._patched_constructor()
        with self.zsp._DETECTOR_LOCK:
            first = self.zsp._get_shared_detector_locked()
            second = self.zsp._get_shared_detector_locked()
        self.assertIs(first, second)
        self.assertEqual(len(constructed), 1)

    def test_reset_closes_and_allows_recreation(self):
        constructed = self._patched_constructor()
        with self.zsp._DETECTOR_LOCK:
            first = self.zsp._get_shared_detector_locked()
        self.zsp.reset_shared_detector()
        first.close.assert_called_once()
        with self.zsp._DETECTOR_LOCK:
            second = self.zsp._get_shared_detector_locked()
        self.assertIsNot(first, second)
        self.assertEqual(len(constructed), 2)

    def test_concurrent_extractions_construct_exactly_one_detector(self):
        import threading
        constructed = self._patched_constructor(delay=0.05)
        errors = []

        def run_extraction():
            try:
                # All-None frames: exercises the real public locking path
                # (detector acquired, zero detect calls) and returns the
                # no-candidates rejection.
                success, _, _ = self.zsp.extract_features_from_image_array(
                    [None] * N_FRAMES, "test_concurrent_singleton")
                assert success is False
            except Exception as e:  # pragma: no cover - surfaced via errors list
                errors.append(e)

        threads = [threading.Thread(target=run_extraction) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(constructed), 1)

    def test_explicit_detector_parameter_bypasses_singleton(self):
        constructed = self._patched_constructor()
        fake = _FakeMultiPoseDetector([[make_pose(0.5, 0.5)] for _ in range(N_FRAMES)])
        success, _, _ = self.zsp.extract_features_from_image_array(
            [np.zeros((8, 8, 3), dtype=np.uint8)] * N_FRAMES,
            "test_explicit_detector", detector=fake)
        self.assertTrue(success)
        self.assertEqual(len(constructed), 0)  # shared singleton never touched


_REAL_FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames", "Sai_front_1")


@unittest.skipUnless(os.path.isdir(_REAL_FRAMES_DIR),
                     f"real frames not present at {_REAL_FRAMES_DIR} (gitignored, not committed)")
class TestRealFootageSubjectSelection(unittest.TestCase):
    """Empirical evidence on real batting frames (runs only where the
    gitignored frames/ directory exists — this machine).

    Composite methodology, validated before being baked in: MediaPipe will
    not detect a person occupying too small a fraction of the image (a 55%
    copy pasted beside the FULL original frame was never detected), so the
    two-person scenes below are built from person-CROPPED tiles, where both
    figures are large enough to detect. At tile scale 0.75 both persons are
    detected in 5/7 frames (size ratio 1.33 >= dominance 1.25); at scale 1.0
    both are detected in 7/7 frames with equal size (genuine ambiguity)."""

    @classmethod
    def setUpClass(cls):
        import cv2
        import zero_storage_pipeline as zsp
        cls._zsp = zsp
        frames = []
        for i in range(1, 8):
            bgr = cv2.imread(os.path.join(_REAL_FRAMES_DIR, f"frame_0{i}.jpg"))
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cls.frames = frames
        # One single-pose bootstrap detection per frame to crop each person
        # tile (cached for all tests in this class).
        cls.crops = []
        options = zsp.vision.PoseLandmarkerOptions(
            base_options=zsp.python.BaseOptions(model_asset_path=zsp.model_path),
            output_segmentation_masks=False, num_poses=1)
        with zsp.vision.PoseLandmarker.create_from_options(options) as detector:
            for frame in frames:
                pose = cls._detect_with(detector, frame)[0]
                cls.crops.append(cls._person_crop(frame, pose))

    @staticmethod
    def _detect_with(detector, frame_rgb):
        import mediapipe as mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return list(detector.detect(image).pose_landmarks or [])

    def _detect(self, frame_rgb, num_poses):
        zsp = self._zsp
        options = zsp.vision.PoseLandmarkerOptions(
            base_options=zsp.python.BaseOptions(model_asset_path=zsp.model_path),
            output_segmentation_masks=False, num_poses=num_poses)
        with zsp.vision.PoseLandmarker.create_from_options(options) as detector:
            return self._detect_with(detector, frame_rgb)

    @staticmethod
    def _person_crop(frame, pose, margin=0.12):
        h, w = frame.shape[:2]
        xs = [lm.x for lm in pose]
        ys = [lm.y for lm in pose]
        x0 = max(0, int((min(xs) - margin) * w))
        x1 = min(w, int((max(xs) + margin) * w))
        y0 = max(0, int((min(ys) - margin) * h))
        y1 = min(h, int((max(ys) + margin) * h))
        return frame[y0:y1, x0:x1]

    def _two_person_composites(self, scale):
        """Person crop full-size on the left, the same crop at `scale` pasted
        feet-aligned on the right of a gray canvas."""
        import cv2
        composites = []
        for crop in self.crops:
            ch, cw = crop.shape[:2]
            small = cv2.resize(crop, (int(cw * scale), int(ch * scale)))
            canvas = np.full((ch, cw + small.shape[1] + 30, 3), 128, dtype=np.uint8)
            canvas[:, :cw] = crop
            canvas[ch - small.shape[0]:, cw + 30:] = small
            composites.append(canvas)
        return composites

    def test_single_person_frame_identical_under_multi_pose_config(self):
        # The "single-person behaves identically to Milestone 3" claim,
        # measured: on a real single-batsman frame, num_poses=4 must find the
        # same one person with (near-)identical landmarks to num_poses=1.
        from subject_selection import MAX_POSE_CANDIDATES
        frame = self.frames[0]
        single = self._detect(frame, 1)
        multi = self._detect(frame, MAX_POSE_CANDIDATES)
        self.assertEqual(len(single), 1)
        self.assertEqual(len(multi), 1)
        max_delta = max(
            max(abs(a.x - b.x), abs(a.y - b.y))
            for a, b in zip(single[0], multi[0])
        )
        self.assertLess(max_delta, 0.01)

    def test_real_single_person_session_passes_end_to_end(self):
        # The full pipeline (multi-pose detection + selection + validation +
        # rules) on a REAL, known-good single-person session must succeed
        # with the subject in every frame — no regression vs. the
        # num_poses=1 era that originally ingested this exact session.
        success, raw_keypoints, interpolated = self._zsp.extract_features_from_image_array(
            list(self.frames), "test_real_single_person")
        self.assertTrue(success, f"real single-person session rejected: {raw_keypoints}")
        self.assertEqual(interpolated, [])
        self.assertEqual(len(raw_keypoints), N_FRAMES)

    def test_real_two_person_scene_selects_dominant_subject_deterministically(self):
        # Both persons detected (validated: 2 candidates in 5/7 frames at
        # this scale); the full-size left figure has max coverage AND 1.33x
        # size — it must be selected in EVERY frame, on repeated runs.
        composites = self._two_person_composites(scale=0.75)
        left_fraction = self.crops[0].shape[1] / composites[0].shape[1]

        hip_xs_per_run = []
        for run in range(2):
            success, raw_keypoints, _ = self._zsp.extract_features_from_image_array(
                list(composites), f"test_real_dominant_run{run + 1}")
            self.assertTrue(success, f"dominant-subject scene rejected on run {run + 1}: {raw_keypoints}")
            hip_xs = [(kp[23][0] + kp[24][0]) / 2.0 for kp in raw_keypoints]
            for hip_x in hip_xs:
                self.assertLess(hip_x, left_fraction,
                                "selected subject is not the larger, left-side person")
            hip_xs_per_run.append(hip_xs)

        # Determinism: both runs picked the same physical subject everywhere.
        np.testing.assert_allclose(hip_xs_per_run[0], hip_xs_per_run[1], atol=0.05,
                                   err_msg="subject choice differed between identical runs")

    def test_real_equal_size_two_person_scene_is_rejected_as_ambiguous(self):
        # At scale 1.0 both figures are detected in all 7 frames with equal
        # size — genuinely indistinguishable subjects. The pipeline must
        # refuse to guess (this rejection is also proof that BOTH persons
        # were really detected: the ambiguity branch is unreachable with
        # fewer than two persistent tracks).
        composites = self._two_person_composites(scale=1.0)
        success, payload, interpolated = self._zsp.extract_features_from_image_array(
            list(composites), "test_real_ambiguous")
        self.assertFalse(success)
        self.assertIn("refusing to guess", payload)
        self.assertIsNone(interpolated)


if __name__ == "__main__":
    unittest.main()
