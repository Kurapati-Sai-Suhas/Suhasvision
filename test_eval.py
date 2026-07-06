import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import os

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
        scores = group.iloc[0][SCORE_COLS].values.astype(np.float32) / 100.0
        X.append(seq)
        y.append(scores)
        session_ids.append(session_name)
    return np.array(X), np.array(y)

df = pd.read_csv("dataset/dataset_angles.csv")
X, y = build_sequences(df)

model = tf.keras.models.load_model("dataset/cricket_stance_advanced_v2.keras", compile=False)
model.compile(loss='mse', metrics=['mae'])
res = model.evaluate(X, y, verbose=0)
print(f"MSE: {res[0]} | MAE: {res[1]}")

y_pred = model.predict(X, verbose=0)
print("Pred mean:", np.mean(y_pred, axis=0))
print("Actual mean:", np.mean(y, axis=0))

mse = np.mean(np.square(y - y_pred))
mae = np.mean(np.abs(y - y_pred))
print(f"Manual MSE: {mse:.4f} | Manual MAE: {mae:.4f}")
