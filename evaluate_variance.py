import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "dataset"))
from academic_scripts.train_advanced_model import TemporalAttention

def build_sequences(df):
    session_ids = []
    X = []
    y = []
    FEATURE_COLS = [
        "angle_knee_L", "angle_knee_R", "angle_hip_L", "angle_hip_R",
        "angle_elbow_L", "angle_elbow_R", "angle_shoulder_L", "angle_shoulder_R",
        "angle_ankle_L", "angle_ankle_R", "angle_trunk_L", "angle_trunk_R",
        "angle_arm_L", "angle_arm_R", "angle_head_tilt"
    ]
    FEATURE_COLS += [c + "_vel" for c in FEATURE_COLS]
    SCORE_COLS = ["score_balance", "score_power", "score_technique", "score_defence"]

    for session_name, group in df.groupby("session_name"):
        if len(group) != 7:
            continue
        group = group.sort_values("frame_name")
        seq = group[FEATURE_COLS].values.astype(np.float32)
        
        scores = group.iloc[0][SCORE_COLS].values.astype(np.float32)
        scores = np.where(scores > 100, scores / 10.0, scores)
        scores = np.clip(scores, 0, 100)
        scores = scores / 100.0
        
        X.append(seq)
        y.append(scores)
        session_ids.append(session_name)
    return np.array(X), np.array(y)

def main():
    df = pd.read_csv("dataset/dataset_angles.csv")
    X, y = build_sequences(df)
    
    model = tf.keras.models.load_model("dataset/cricket_stance_advanced_v2.keras", compile=False)
    y_pred = model.predict(X, verbose=0)
    
    # Scale back to 0-100
    y_true_100 = y * 100
    y_pred_100 = y_pred * 100
    
    print("=== MODEL EVALUATION ===")
    print(f"True Labels Range: Min {np.min(y_true_100):.1f} | Max {np.max(y_true_100):.1f}")
    print(f"True Labels Std Dev: {np.std(y_true_100):.2f}")
    
    print(f"\nModel Predictions Range: Min {np.min(y_pred_100):.1f} | Max {np.max(y_pred_100):.1f}")
    print(f"Model Predictions Std Dev: {np.std(y_pred_100):.2f}")
    
    mae = np.mean(np.abs(y_true_100 - y_pred_100))
    print(f"\nModel MAE: {mae:.2f} points")
    
    # Dummy Regressor Evaluation (Predicts the mean every time)
    mean_prediction = np.mean(y_true_100, axis=0)
    y_dummy = np.tile(mean_prediction, (len(y), 1))
    dummy_mae = np.mean(np.abs(y_true_100 - y_dummy))
    print(f"Dummy Regressor (Mean Predictor) MAE: {dummy_mae:.2f} points")
    
    if dummy_mae - mae > 2.0:
        print("\nCONCLUSION: The model is learning useful features and outperforming a mean guesser!")
    else:
        print("\nCONCLUSION: The model has collapsed and is just predicting the mean.")

if __name__ == "__main__":
    main()
