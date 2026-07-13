"""
train_lstm_model.py
Bidirectional LSTM trained on phase-ordered 3D keypoint sequences to
predict stance quality_score. This supersedes the static single-frame
dense model in the original SRS -- since your pipeline already captures
shot_phase per frame, a sequence model can learn backlift -> downswing ->
impact -> follow-through dynamics that a single-frame model cannot see.
This is also the stronger, more novel claim for the IEEE paper.

DATA CONTRACT
- keypoints.csv: should include a kinematic_valid column (run
  kinematic_validator.py first). Frames where this is False are excluded.
- labels.csv: must include quality_score (1-100) per session_id.

USAGE
    python train_lstm_model.py --keypoints keypoints_validated.csv --labels labels.csv
"""

import argparse
import numpy as np
import pandas as pd
# Removed sklearn import to bypass Windows App Control DLL block
from tensorflow import keras
from tensorflow.keras import layers

LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
FEATURE_COLS = [f"{lm}_{axis}" for lm in LANDMARKS for axis in ("x", "y", "z")]
SEQ_LEN = 7   # match our exact extraction pipeline (7 frames), no wasting timesteps on padding


def normalize_session(seq: np.ndarray) -> np.ndarray:
    """Center on hip midpoint and scale by shoulder width, per frame, so the
    model learns pose SHAPE rather than the player's distance from camera."""
    n_frames = seq.shape[0]
    seq = seq.reshape(n_frames, len(LANDMARKS), 3).copy()
    hip_l = seq[:, LANDMARKS.index("left_hip"), :2]
    hip_r = seq[:, LANDMARKS.index("right_hip"), :2]
    hip_mid = (hip_l + hip_r) / 2
    sh_l = seq[:, LANDMARKS.index("left_shoulder"), :2]
    sh_r = seq[:, LANDMARKS.index("right_shoulder"), :2]
    scale = np.linalg.norm(sh_l - sh_r, axis=1, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    seq[:, :, 0] -= hip_mid[:, 0:1]
    seq[:, :, 1] -= hip_mid[:, 1:2]
    seq[:, :, 0:2] /= scale[:, :, None]
    return seq.reshape(n_frames, -1)


def pad_or_truncate(seq: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    if len(seq) >= seq_len:
        return seq[:seq_len]
    pad = np.zeros((seq_len - len(seq), seq.shape[1]), dtype=np.float32)
    return np.vstack([seq, pad])


def build_sequences(keypoints: pd.DataFrame, labels: pd.DataFrame):
    X, y, session_ids = [], [], []
    
    # Extract the 4 AI coaching metrics as separate targets
    score_cols = ["score_balance", "score_power", "score_technique", "score_defence"]
    
    # Drop duplicates in case the pipeline was restarted and appended the same session twice
    labels = labels.drop_duplicates(subset=["session_name"], keep="last")
    label_map = labels.set_index("session_name")[score_cols].to_dict('index')

    for session_name, group in keypoints.groupby("session_name"):
        if session_name not in label_map:
            continue
        valid = group[group.get("kinematic_valid", True) == True]
        if len(valid) < 5:
            continue
        valid = valid.sort_values("frame_name")
        seq = valid[FEATURE_COLS].values.astype(np.float32)
        seq = normalize_session(seq)
        seq = pad_or_truncate(seq)

        X.append(seq)
        y.append([label_map[session_name][col] / 100.0 for col in score_cols])  # scale all 4 to 0-1
        session_ids.append(session_name)

    return np.array(X), np.array(y), session_ids


def build_model(seq_len: int, n_features: int) -> keras.Model:
    inp = keras.Input(shape=(seq_len, n_features))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(4, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoints", default="dataset.csv")
    ap.add_argument("--labels", default="dataset.csv")
    ap.add_argument("--model_out", default="cricket_stance_lstm_v1.keras")
    ap.add_argument("--predictions_out", default="lstm_predictions.csv")
    args = ap.parse_args()

    keypoints = pd.read_csv(args.keypoints)
    labels = pd.read_csv(args.labels)

    X, y, session_ids = build_sequences(keypoints, labels)
    print(f"Built {len(X)} session sequences, shape {X.shape}")

    if len(X) < 30:
        print("WARNING: fewer than 30 labelled sessions. Treat results as a "
              "pipeline smoke test, not a reportable result, until your "
              "dataset grows past roughly 150-200 sessions.")

    np.random.seed(42)
    idx = np.random.permutation(len(X))
    n_train = int(len(X) * 0.70)
    n_val = int(len(X) * 0.15)
    
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train+n_val]
    idx_test = idx[n_train+n_val:]

    # Calculate class weights manually to address the 95% Cover Drive imbalance
    shot_types = labels.set_index("session_name")["shot_type"].to_dict()
    seq_shot_types = [shot_types.get(sid, "Unknown") for sid in session_ids]
    
    unique_shots, counts = np.unique(seq_shot_types, return_counts=True)
    weight_dict = {shot: len(seq_shot_types) / (len(unique_shots) * count) for shot, count in zip(unique_shots, counts)}
    sample_weights = np.array([weight_dict[shot] for shot in seq_shot_types])

    w_train = sample_weights[idx_train]
    w_val = sample_weights[idx_val]

    model = build_model(SEQ_LEN, len(FEATURE_COLS))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(args.model_out, save_best_only=True, monitor="val_loss"),
    ]

    model.fit(
        X[idx_train], y[idx_train],
        validation_data=(X[idx_val], y[idx_val]),
        sample_weight=w_train,
        epochs=200, batch_size=16,
        callbacks=callbacks, verbose=2,
    )

    test_loss, test_mae = model.evaluate(X[idx_test], y[idx_test], verbose=0)
    print(f"\nTest MSE: {test_loss:.4f} | Test MAE: {test_mae*100:.2f} points (0-100 scale)")
    print(f"Model saved to {args.model_out}")

    # Save predictions on the test set for compare_methods.py
    preds = model.predict(X[idx_test]) * 100
    pred_df = pd.DataFrame(preds, columns=[f"pred_{col}" for col in ["balance", "power", "technique", "defence"]])
    pred_df.insert(0, "session_name", [session_ids[i] for i in idx_test])
    pred_df.to_csv(args.predictions_out, index=False)
    print(f"Test-set predictions saved to {args.predictions_out}")


if __name__ == "__main__":
    main()
