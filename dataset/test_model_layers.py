import unittest
import subprocess
import sys
import os
import numpy as np
import tensorflow as tf

import schema
from model_layers import TemporalAttention
from academic_scripts import train_advanced_model
import inference_service


class TestSchema(unittest.TestCase):
    def test_expected_features_is_base_names_plus_velocities(self):
        self.assertEqual(len(schema.FEATURE_BASE_NAMES), 15)
        self.assertEqual(len(schema.EXPECTED_FEATURES), 30)
        expected_vel_names = [c + "_vel" for c in schema.FEATURE_BASE_NAMES]
        self.assertEqual(schema.EXPECTED_FEATURES, list(schema.FEATURE_BASE_NAMES) + expected_vel_names)

    def test_feature_base_names_is_immutable(self):
        # FEATURE_BASE_NAMES is imported by reference into every consumer
        # module -- it must be a tuple so an accidental in-place mutation in
        # any one of them can't silently corrupt the shared schema for all
        # the others. This test exists specifically to catch a future
        # accidental revert of that to a list.
        self.assertIsInstance(schema.FEATURE_BASE_NAMES, tuple)
        with self.assertRaises(AttributeError):
            schema.FEATURE_BASE_NAMES.append("should_not_be_possible")

    def test_expected_features_is_a_list(self):
        # Deliberately NOT a tuple, unlike FEATURE_BASE_NAMES -- consumers
        # pass this directly to pandas column selection (df[EXPECTED_FEATURES]),
        # which is list-safe but not guaranteed tuple-safe.
        self.assertIsInstance(schema.EXPECTED_FEATURES, list)

    def test_seq_len_is_seven(self):
        # Matches the fixed 7-phase session design (stance..follow-through)
        # documented in FEATURE_BASE_NAMES' consumers.
        self.assertEqual(schema.SEQ_LEN, 7)

    def test_canonical_frame_names_shape_and_immutability(self):
        # One name per phase, immutable for the same reason as
        # FEATURE_BASE_NAMES (imported by reference everywhere).
        self.assertEqual(len(schema.CANONICAL_FRAME_NAMES), schema.SEQ_LEN)
        self.assertIsInstance(schema.CANONICAL_FRAME_NAMES, tuple)

    def test_canonical_frame_names_are_bare_convention(self):
        # Audit H6: the canonical names must be fixed points of
        # normalize_frame_name (the bare convention), and the old file-style
        # convention must normalize TO them -- otherwise the two-conventions
        # split that once NaN'd 42% of the dataset comes back.
        from academic_scripts.feature_engineering import normalize_frame_name
        for name in schema.CANONICAL_FRAME_NAMES:
            self.assertEqual(normalize_frame_name(name), name)
        self.assertEqual(normalize_frame_name("frame_01_stance.jpg"),
                         schema.CANONICAL_FRAME_NAMES[0])

    def test_schema_import_does_not_pull_in_tensorflow_or_mediapipe(self):
        # This is a stated design constraint in schema.py's own module
        # docstring: importing it must not trigger the TF/MediaPipe import
        # chain. Verified in a fresh subprocess so an already-imported
        # tensorflow elsewhere in this same test run can't mask a violation.
        code = (
            "import sys; import schema; "
            "assert 'tensorflow' not in sys.modules, 'schema.py pulled in tensorflow'; "
            "assert 'mediapipe' not in sys.modules, 'schema.py pulled in mediapipe'; "
            "print('OK')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("OK", result.stdout)


class TestTemporalAttention(unittest.TestCase):
    def test_is_registered_keras_serializable(self):
        # register_keras_serializable() is required for tf.keras.models.load_model's
        # custom_objects mechanism to resolve this class by name from a saved .keras file.
        self.assertTrue(hasattr(TemporalAttention, "get_config"))
        config = tf.keras.utils.get_registered_object(
            "Custom>TemporalAttention"
        ) or tf.keras.utils.get_registered_object("TemporalAttention")
        self.assertIsNotNone(config, "TemporalAttention is not resolvable via the Keras serialization registry")

    def test_output_shape(self):
        layer = TemporalAttention()
        x = tf.constant(np.random.randn(2, 7, 128).astype(np.float32))
        out = layer(x)
        self.assertEqual(out.shape, (2, 128))

    def test_call_matches_hand_computed_attention(self):
        # A tiny, hand-checkable case: 1 batch, 2 timesteps, 1 feature.
        # With W and b both zero, e = tanh(0) = 0 for every timestep, so
        # softmax over 2 equal logits gives alpha = [0.5, 0.5] at every step
        # -- the context should be exactly the mean of the two timesteps.
        layer = TemporalAttention()
        x = tf.constant([[[2.0], [8.0]]], dtype=tf.float32)  # (1, 2, 1)
        layer.build(x.shape)
        layer.W.assign(tf.zeros_like(layer.W))
        layer.b.assign(tf.zeros_like(layer.b))
        context = layer(x)
        self.assertAlmostEqual(float(context.numpy()[0][0]), 5.0, places=5)  # mean(2, 8) == 5


class TestSingleSourceOfTruth(unittest.TestCase):
    """Milestone 1: asserts every consumer imports the SAME object, not a
    separately-defined lookalike -- the actual guarantee this milestone
    exists to establish, not just that each file happens to behave the same."""

    def test_train_advanced_model_imports_canonical_attention(self):
        self.assertIs(train_advanced_model.TemporalAttention, TemporalAttention)

    def test_inference_service_imports_canonical_attention(self):
        self.assertIs(inference_service.TemporalAttention, TemporalAttention)

    def test_train_advanced_model_imports_canonical_schema(self):
        self.assertIs(train_advanced_model.FEATURE_COLS, schema.EXPECTED_FEATURES)
        self.assertEqual(train_advanced_model.SEQ_LEN, schema.SEQ_LEN)

    def test_inference_service_imports_canonical_schema(self):
        self.assertIs(inference_service.EXPECTED_FEATURES, schema.EXPECTED_FEATURES)

    def test_frame_name_consumers_import_canonical_frame_names(self):
        # Audit H6 (Milestone 3): every module that writes or pads by the 7
        # canonical phase names must use the ONE schema.py tuple -- an
        # independent stale copy in the tensor log is exactly how the
        # mixed-naming-convention corruption started.
        import zero_storage_pipeline
        from academic_scripts import feature_engineering
        self.assertIs(inference_service.CANONICAL_FRAME_NAMES, schema.CANONICAL_FRAME_NAMES)
        self.assertIs(zero_storage_pipeline.CANONICAL_FRAME_NAMES, schema.CANONICAL_FRAME_NAMES)
        self.assertIs(feature_engineering.CANONICAL_FRAME_NAMES, schema.CANONICAL_FRAME_NAMES)


if __name__ == '__main__':
    unittest.main()
