import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from mc_dropout_inference import mc_dropout_predict, uncertainty_error_correlation

# Milestone 1 (deferred here): TemporalAttention used to be redefined
# independently in this file. Now imported from the canonical module.
_dataset_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dataset_dir not in sys.path:
    sys.path.insert(0, _dataset_dir)
from model_layers import TemporalAttention  # noqa: E402

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
    import argparse
    ap = argparse.ArgumentParser()
    # Was hardcoded to cricket_stance_advanced_v2.keras -- stale even before
    # this milestone (v4 has been the deployed model since before Milestone 1
    # started), meaning this script's own uncertainty-calibration claim was
    # silently about the wrong model. Defaults to the same env var
    # ml_service.py uses, so "which model is this evaluating" has one answer.
    ap.add_argument("--model", default=os.environ.get('CRICKET_MODEL_FILENAME', 'cricket_stance_advanced_v5.keras'))
    args = ap.parse_args()

    print("Loading data...")
    X, Y = load_data()
    if X is None or len(X) == 0:
        print("Failed to load valid sequences.")
        return

    print(f"Loaded {len(X)} valid sessions of shape (7, 30).")

    model_path = os.path.join('dataset', args.model)
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'TemporalAttention': TemporalAttention})
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
