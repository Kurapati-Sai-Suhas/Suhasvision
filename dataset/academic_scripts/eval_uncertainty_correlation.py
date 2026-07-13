import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from scipy.stats import pearsonr
from mc_dropout_inference import mc_dropout_predict, uncertainty_error_correlation

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
        context = tf.reduce_sum(x * alpha, axis=1) 
        return context

def load_data():
    try:
        df_labels = pd.read_csv('dataset/labels.csv')
        df_angles = pd.read_csv('dataset/dataset_angles.csv')
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return None, None
        
    session_groups = df_angles.groupby('session_name')
    X_list = []
    Y_list = []
    
    for session_name, group in session_groups:
        if len(group) == 7:
            # Drop metadata columns (everything before angle_knee_L + session_name)
            meta_cols = ['frame_name', 'shot_type', 'batting_hand', 'skill_level', 'strength', 'weakness', 'change_drill', 'score_balance', 'score_power', 'score_technique', 'score_defence', 'session_name']
            features = group.drop(columns=[c for c in meta_cols if c in group.columns]).values
            # Get labels
            label_row = df_labels[df_labels['session_name'] == session_name]
            if not label_row.empty:
                y = label_row[['score_balance', 'score_power', 'score_technique', 'score_defence']].values[0]
                X_list.append(features)
                Y_list.append(y)
    
    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)

def main():
    print("Loading data...")
    X, Y = load_data()
    if X is None or len(X) == 0:
        print("Failed to load valid sequences.")
        return
        
    print(f"Loaded {len(X)} valid sessions of shape (7, 30).")
    
    print("Loading model...")
    try:
        model = tf.keras.models.load_model('dataset/cricket_stance_advanced_v2.keras', compile=False, custom_objects={'TemporalAttention': TemporalAttention})
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Running MC Dropout Inference on all sessions in batch...")
    means, stds, _ = mc_dropout_predict(model, X, n_passes=30)
        
    print("Computing Pearson Correlations and P-values...")
    correlations = uncertainty_error_correlation(means, stds, Y)
    
    print("\n=== Uncertainty-Error Correlation Results ===")
    for metric, (corr, p_value) in correlations.items():
        print(f"{metric}: {corr:.3f} (p={p_value:.3f})")
        
    print("\nInterpretation:")
    print("A positive, statistically significant correlation (p < 0.05) indicates that when the model's")
    print("prediction standard deviation (uncertainty) is high, the absolute error is also high.")
    print("A lack of correlation suggests the model's uncertainty bounds are stochastic or uncalibrated.")

if __name__ == "__main__":
    main()
