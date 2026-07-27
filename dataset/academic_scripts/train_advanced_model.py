import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import GroupShuffleSplit

# Milestone 1 (ML single-source-of-truth consolidation): SEQ_LEN and the
# feature list used to be defined independently here. They now come from
# schema.py (the canonical source, also used by ml_service.py,
# inference_service.py, and active_learning_retrain.py), and TemporalAttention
# comes from model_layers.py -- this script is the one that trained the
# currently-deployed model weights, so its former definitions are exactly
# what those two shared modules now contain.
_academic_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_dataset_dir = os.path.dirname(_academic_scripts_dir)
for _p in (_dataset_dir, _academic_scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from schema import SEQ_LEN, EXPECTED_FEATURES as FEATURE_COLS  # noqa: E402
from model_layers import TemporalAttention  # noqa: E402
# Milestone 2 (unified evaluation protocol): extract_batsman_name used to be
# defined here and separately re-defined in cross_validate.py. It now lives
# in evaluation_protocol.py, the single shared home for cross-validation
# methodology used by this script and every evaluation script.
#
# _academic_scripts_dir is explicitly added to sys.path above (not just
# _dataset_dir) because this is a same-directory import -- Python only adds
# a script's own directory to sys.path automatically when that script is
# the one being *run directly*. When this module is instead *imported* as
# academic_scripts.train_advanced_model from a different working directory
# (as the Milestone 1 test suite does), that auto-add doesn't happen, and
# `from evaluation_protocol import ...` would fail without this line.
from evaluation_protocol import extract_batsman_name  # noqa: E402

SCORE_COLS = ["score_balance", "score_power", "score_technique", "score_defence"]

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
