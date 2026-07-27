import unittest
from unittest import mock

import inference_service


class TestPredictScoresFailureArity(unittest.TestCase):
    """Regression test for audit C4: predict_scores' exception path used to
    return a 3-tuple where the success path returns 4 values, so every
    inference failure crashed the caller (app.py unpacks 4) with a
    ValueError that replaced the real error message."""

    def test_failure_path_returns_four_elements_like_success_path(self):
        with mock.patch.object(
            inference_service, "get_model",
            side_effect=RuntimeError("model load failed (test)"),
        ):
            result = inference_service.predict_scores(object())

        self.assertEqual(len(result), 4)
        success, error_message, variance, explanations = result
        self.assertFalse(success)
        self.assertIn("model load failed (test)", error_message)
        self.assertIsNone(variance)
        self.assertIsNone(explanations)


if __name__ == "__main__":
    unittest.main()
