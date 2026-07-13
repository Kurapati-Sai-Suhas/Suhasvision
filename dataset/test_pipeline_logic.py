import unittest
from zero_storage_pipeline import apply_pipeline_rules
import os

class TestPipelineLogic(unittest.TestCase):
    def setUp(self):
        # Create a dummy clean frame
        # 33 landmarks, each with 5 values (x,y,z,vis,pres)
        self.clean_frame = [[0.5, 0.5, 0.0, 1.0, 1.0] for _ in range(33)]
        # Make a slightly different frame for edge interpolation math test
        self.shifted_frame = [[0.6, 0.6, 0.0, 1.0, 1.0] for _ in range(33)]
        
        # Ensure log file is clean
        if os.path.exists("pipeline_rejections.log"):
            os.remove("pipeline_rejections.log")

    def test_interpolation_success(self):
        # Test 1: Phase 2 fails, should interpolate from 1 and 3
        keypoints = [self.clean_frame, None, self.clean_frame, self.clean_frame, self.clean_frame, self.clean_frame, self.clean_frame]
        success, interpolated = apply_pipeline_rules(keypoints, "test_session_1")
        self.assertTrue(success)
        self.assertIn("phase_2", interpolated)
        # Verify the None was replaced with actual data
        self.assertIsNotNone(keypoints[1])
        
    def test_cascading_failure(self):
        # Test 4: 3 consecutive frames fail -> Hard Reject Cascading
        keypoints = [self.clean_frame, self.clean_frame, None, None, None, self.clean_frame, self.clean_frame]
        success, interpolated = apply_pipeline_rules(keypoints, "test_session_2")
        self.assertFalse(success)
        self.assertEqual(len(interpolated), 0)
        
    def test_contact_phase_failure(self):
        # Test 5: Only Phase 6 (Contact) fails -> Should interpolate successfully (strict check removed)
        keypoints = [self.clean_frame, self.clean_frame, self.clean_frame, self.clean_frame, self.clean_frame, None, self.clean_frame]
        success, interpolated = apply_pipeline_rules(keypoints, "test_session_3")
        self.assertTrue(success)
        self.assertIn("phase_6", interpolated)

    def test_count_threshold_failure(self):
        # 3 non-consecutive frames fail -> Hard Reject
        keypoints = [None, self.clean_frame, None, self.clean_frame, None, self.clean_frame, self.clean_frame]
        success, interpolated = apply_pipeline_rules(keypoints, "test_session_4")
        self.assertFalse(success)

    def test_edge_extrapolation_success(self):
        # Test 6: Frame 1 fails, Frame 2 and 3 are clean -> Extrapolate backwards
        # To bypass cascading reject, Frame 2 must be clean. 
        # To actually hit the edge case instead of count/cascading, we need exactly 1 or 2 isolated failures at the edges.
        keypoints = [None, self.clean_frame, self.shifted_frame, self.clean_frame, self.clean_frame, self.clean_frame, self.clean_frame]
        success, interpolated = apply_pipeline_rules(keypoints, "test_session_5")
        self.assertTrue(success)
        self.assertIn("phase_1", interpolated)
        # Check math: p0 = p1 - (p2 - p1)
        # p1 = 0.5, p2 = 0.6 -> p0 = 0.5 - 0.1 = 0.4
        self.assertAlmostEqual(keypoints[0][0][0], 0.4)

    def test_cascading_intercepts_edge(self):
        # Test 7: Frame 1 and 2 fail -> Hits Cascading Reject BEFORE Edge Reject
        keypoints = [None, None, self.clean_frame, self.clean_frame, self.clean_frame, self.clean_frame, self.clean_frame]
        success, interpolated = apply_pipeline_rules(keypoints, "test_session_6")
        self.assertFalse(success)
        
    def test_true_edge_reject(self):
        # Test 8: Frame 1 fails, Frame 2 clean, Frame 3 fails -> Bypasses cascading, hits true Edge Reject
        keypoints = [None, self.clean_frame, None, self.clean_frame, self.clean_frame, self.clean_frame, self.clean_frame]
        success, interpolated = apply_pipeline_rules(keypoints, "test_session_7")
        self.assertFalse(success)
        
    def test_validate_pose_integration(self):
        # Test 9: Verify validate_pose actually rejects a bad topology and feeds None to the rule engine
        from zero_storage_pipeline import validate_pose
        import collections
        
        # Mock a MediaPipe landmark
        Landmark = collections.namedtuple('Landmark', ['x', 'y', 'z', 'visibility', 'presence'])
        bad_pose = [Landmark(0,0,0,1,1) for _ in range(33)]
        
        # Invert topology (ankle absurdly above hip). y=0 is top.
        bad_pose[23] = Landmark(0, 0.5, 0, 1, 1) # L Hip
        bad_pose[24] = Landmark(0, 0.5, 0, 1, 1) # R Hip
        bad_pose[27] = Landmark(0, 0.1, 0, 1, 1) # L Ankle at 0.1
        bad_pose[28] = Landmark(0, 0.1, 0, 1, 1) # R Ankle at 0.1 
        
        # Pass to validate_pose (Phase 4 = Downswing, uses loose bounds)
        is_valid = validate_pose(bad_pose, 4, "test_session_8", "frame_05")
        self.assertFalse(is_valid)

if __name__ == '__main__':
    unittest.main()
