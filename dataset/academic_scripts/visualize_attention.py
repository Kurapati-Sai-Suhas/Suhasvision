import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os

@tf.keras.utils.register_keras_serializable()
class TemporalAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        alpha = tf.keras.activations.softmax(e, axis=1)
        return tf.reduce_sum(x * alpha, axis=1)

def extract_alpha_numpy(x, W, b):
    """Manually computes the attention weights (alpha) for visualization."""
    # x shape: (1, 7, 128) if BiLSTM was 64 units
    # dot product with W (128, 1) -> (1, 7, 1)
    e = np.tanh(np.dot(x, W) + b)
    # Softmax over time (axis=1)
    exp_e = np.exp(e - np.max(e, axis=1, keepdims=True))
    alpha = exp_e / np.sum(exp_e, axis=1, keepdims=True)
    return alpha

def main():
    print("Loading Advanced TCN-Attention Model...")
    try:
        model = tf.keras.models.load_model("cricket_stance_advanced_v2.keras", 
                                           custom_objects={"TemporalAttention": TemporalAttention})
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Find the TemporalAttention layer and BiLSTM layer
    attn_layer = None
    lstm_layer_name = None
    for layer in model.layers:
        if isinstance(layer, TemporalAttention):
            attn_layer = layer
        elif isinstance(layer, tf.keras.layers.Bidirectional):
            lstm_layer_name = layer.name
            
    if attn_layer is None:
        print("Could not find TemporalAttention layer!")
        return
        
    W, b = attn_layer.get_weights()
    
    # We need a sub-model to get the output of the BiLSTM just before Attention
    sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(lstm_layer_name).output)
    
    # Load 1 sample from dataset
    df = pd.read_csv("dataset_angles.csv")
    feature_cols = [c for c in df.columns if c.startswith("angle_")]
    
    valid_session = None
    for session_name, group in df.groupby("session_name"):
        if len(group) == 7:
            valid_session = session_name
            break
            
    if valid_session is None:
        print("No sessions with 7 frames found.")
        return
        
    sample_data = df[df["session_name"] == valid_session].sort_values("frame_name")
        
    X_sample = np.expand_dims(sample_data[feature_cols].values.astype(np.float32), axis=0)
    
    # Get LSTM features
    lstm_out = sub_model.predict(X_sample, verbose=0)
    
    # Calculate Alpha
    alpha = extract_alpha_numpy(lstm_out, W, b)[0, :, 0] # shape (7,)
    
    frames = ["1_Stance", "2_Backlift", "3_Downswing", "4_Impact", 
              "5_Follow_Early", "6_Follow_Mid", "7_Follow_Late"]
              
    print("\n==================================================")
    print("TEMPORAL ATTENTION HEATMAP (For a single stroke)")
    print("==================================================")
    print("How much the AI 'focuses' on each frame:")
    
    for i in range(7):
        weight = alpha[i] * 100
        bar = "#" * int(weight / 2)
        print(f"{frames[i]:<15} | {weight:05.2f}% | {bar}")
        
    # Save to CSV for the paper
    out_df = pd.DataFrame({"Frame": frames, "Attention_Weight_Percent": alpha * 100})
    out_df.to_csv("attention_heatmap.csv", index=False)
    print("\nSaved numerical heatmap to attention_heatmap.csv")

if __name__ == "__main__":
    main()
