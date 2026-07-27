# model_layers.py — Single source of truth for shared Keras layer definitions.
#
# Before this module existed, TemporalAttention was independently redefined in
# at least four places (backend/backend/api/ml_service.py,
# dataset/academic_scripts/train_advanced_model.py, dataset/inference_service.py,
# dataset/academic_scripts/eval_uncertainty_correlation.py) with no guarantee
# any two of them stayed in sync. Every consumer of this layer now imports it
# from here instead.
#
# Kept separate from schema.py deliberately: schema.py is explicitly dependency-free
# (importable without triggering the TensorFlow/MediaPipe import chain just to read
# a list of strings). This module does import TensorFlow, so it must stay out of
# schema.py's import path.

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class TemporalAttention(layers.Layer):
    """Additive attention pooling over the time axis of a (batch, time, features)
    sequence, producing a (batch, features) context vector. This is the exact
    formulation train_advanced_model.py trained the currently-deployed model
    weights with -- verified numerically identical (max abs diff 0.0 across a
    random test tensor) to the tf.keras.backend-based formulation previously
    used independently in inference_service.py before this module existed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, time, features)
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        alpha = tf.keras.activations.softmax(e, axis=1)  # (batch, time, 1)
        context = tf.reduce_sum(x * alpha, axis=1)  # (batch, features)
        return context
