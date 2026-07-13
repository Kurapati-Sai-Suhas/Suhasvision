import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import GroupShuffleSplit

SEQ_LEN = 7
FEATURE_COLS = [
    "angle_knee_L", "angle_knee_R", "angle_hip_L", "angle_hip_R",
    "angle_elbow_L", "angle_elbow_R", "angle_shoulder_L", "angle_shoulder_R",
    "angle_ankle_L", "angle_ankle_R", "angle_trunk_L", "angle_trunk_R",
    "angle_arm_L", "angle_arm_R", "angle_head_tilt"
]
FEATURE_COLS += [c + "_vel" for c in FEATURE_COLS]

SCORE_COLS = ["score_balance", "score_power", "score_technique", "score_defence"]

# Custom Temporal Attention Layer
@tf.keras.utils.register_keras_serializable()
class TemporalAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, time, features)
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        alpha = tf.keras.activations.softmax(e, axis=1) # (batch, time, 1)
        context = tf.reduce_sum(x * alpha, axis=1) # (batch, features)
        return context

def extract_batsman_name(session_name):
    """
    Extracts the base batsman identity so held-out validation never puts a
    batsman's flipped clone in both train and validation (matches the grouping
    logic in cross_validate.py / ablation_study.py).
    """
    base = session_name.replace("_flipped", "")
    parts = base.split('_')
    if base.startswith("youtube_dataset"):
        return "_".join(parts[:3])
    return parts[0]

def build_sequences(df):
    """Groups rows by session_name into (7, 15) tensors."""
    session_ids = []
    X = []
    y = []
    
    # We assume dataset_angles.csv has exactly 7 frames per session ordered by frame_name
    for session_name, group in df.groupby("session_name"):
        if len(group) != SEQ_LEN:
            continue
        # Sort by frame_name to ensure Stance -> Follow-through order
        group = group.sort_values("frame_name")
        
        seq = group[FEATURE_COLS].values.astype(np.float32)
        scores = group.iloc[0][SCORE_COLS].values.astype(np.float32)
        # Fix annotation typos (e.g., 700 instead of 70)
        scores = np.where(scores > 100, scores / 10.0, scores)
        scores = np.clip(scores, 0, 100)
        scores = scores / 100.0
        
        X.append(seq)
        y.append(scores)
        session_ids.append(session_name)
        
    return np.array(X), np.array(y), session_ids

def build_tcn_attention_model(seq_len, n_features):
    inputs = layers.Input(shape=(seq_len, n_features))
    
    # TCN Block: Conv1D extracts localized temporal jerks/shifts
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Sequential processing
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    # Attention Mechanism forces the model to weigh frames differently
    context = TemporalAttention()(x)
    
    # Regression Head
    x = layers.Dense(32, activation='relu')(context)
    outputs = layers.Dense(4, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset_angles.csv")
    ap.add_argument("--model_out", default="cricket_stance_advanced_v2.keras")
    args = ap.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)

    X, y, session_ids = build_sequences(df)
    print(f"Built {len(X)} session sequences, shape {X.shape}")

    # Held-out validation split, grouped by batsman identity so a flipped clone
    # of a validation batsman never leaks into training (same grouping rule as
    # cross_validate.py / ablation_study.py).
    groups = [extract_batsman_name(sid) for sid in session_ids]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))
    assert not set(groups[i] for i in train_idx) & set(groups[i] for i in val_idx), \
        "Leakage: a batsman appears in both train and validation."
    print(f"Train sessions: {len(train_idx)} | Held-out validation sessions: {len(val_idx)}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print("Training production model with a real held-out validation split...")

    # Class weighting (computed on the training split only)
    shot_types = df.drop_duplicates("session_name").set_index("session_name")["shot_type"].to_dict()
    seq_shot_types = [shot_types.get(sid, "Unknown") for sid in session_ids]
    train_shot_types = [seq_shot_types[i] for i in train_idx]
    unique_shots, counts = np.unique(train_shot_types, return_counts=True)
    weight_dict = {shot: len(train_shot_types) / (len(unique_shots) * count) for shot, count in zip(unique_shots, counts)}
    w_train = np.array([weight_dict[shot] for shot in train_shot_types])

    model = build_tcn_attention_model(SEQ_LEN, len(FEATURE_COLS))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ]

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        validation_data=(X_val, y_val),
        epochs=80, batch_size=16,
        callbacks=callbacks, verbose=2
    )

    print(f"Saving model to {args.model_out}")
    model.save(args.model_out)

    # Report both: training-set fit AND the real held-out generalization number.
    # These are two different numbers on purpose — only the second one tells you
    # how the model performs on players it hasn't seen.
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Training MSE: {train_loss:.4f} | Final Training MAE: {train_mae*100:.2f} points (0-100 scale)")
    print(f"Held-Out Validation MSE: {val_loss:.4f} | Held-Out Validation MAE: {val_mae*100:.2f} points (0-100 scale)")

if __name__ == "__main__":
    main()
