import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import time

SEQ_LEN = 7
FEATURE_COLS = [
    "angle_knee_L", "angle_knee_R", "angle_hip_L", "angle_hip_R",
    "angle_elbow_L", "angle_elbow_R", "angle_shoulder_L", "angle_shoulder_R",
    "angle_ankle_L", "angle_ankle_R", "angle_trunk_L", "angle_trunk_R",
    "angle_arm_L", "angle_arm_R", "angle_head_tilt"
]
# Add velocity columns
FEATURE_COLS += [c + "_vel" for c in FEATURE_COLS]

SCORE_COLS = ["score_balance", "score_power", "score_technique", "score_defence"]

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

def build_sequences(df):
    session_ids, X, y = [], [], []
    for session_name, group in df.groupby("session_name"):
        if len(group) != SEQ_LEN:
            continue
        group = group.sort_values("frame_name")
        X.append(group[FEATURE_COLS].values.astype(np.float32))
        y.append(group.iloc[0][SCORE_COLS].values.astype(np.float32) / 100.0)
        session_ids.append(session_name)
    return np.array(X), np.array(y), np.array(session_ids)

def build_model(use_conv, use_attention, seq_len, n_features):
    inputs = layers.Input(shape=(seq_len, n_features))
    x = inputs
    
    if use_conv:
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=use_attention))(x)
    x = layers.Dropout(0.3)(x)
    
    if use_attention:
        x = TemporalAttention()(x)
        
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset_angles.csv")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    X, y, session_ids = build_sequences(df)
    
    # --- CRITICAL FIX: PREVENT DATA LEAKAGE ---
    # Extract base session names (removing '_flipped')
    base_sessions = np.array([sid.replace("_flipped", "") for sid in session_ids])
    unique_base_sessions = np.unique(base_sessions)
    
    n_folds = 5
    n_base = len(unique_base_sessions)
    fold_size = n_base // n_folds
    
    # Random permutation of BASE sessions, not all sessions
    idx_base = np.random.RandomState(42).permutation(n_base)
    unique_base_shuffled = unique_base_sessions[idx_base]

    
    architectures = [
        ("Base LSTM", False, False),
        ("LSTM + Attn", False, True),
        ("Conv1D + LSTM", True, False),
        ("Conv1D + LSTM + Attn", True, True)
    ]
    
    results = {name: [] for name, _, _ in architectures}
    final_model_per_metric_mae = {col: [] for col in SCORE_COLS}
    
    print(f"Starting {n_folds}-Fold Cross Validation across {len(architectures)} architectures...")
    
    for fold in range(n_folds):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_base
        
        val_base_sessions = unique_base_shuffled[val_start:val_end]
        train_base_sessions = np.concatenate((unique_base_shuffled[:val_start], unique_base_shuffled[val_end:]))
        
        # Get indices of sequences that belong to the base sessions in train/val
        idx_train = np.where(np.isin(base_sessions, train_base_sessions))[0]
        idx_val = np.where(np.isin(base_sessions, val_base_sessions))[0]
        
        X_train, y_train = X[idx_train], y[idx_train]
        X_val, y_val = X[idx_val], y[idx_val]
        
        # Calculate weights for this fold
        shot_types = df.drop_duplicates("session_name").set_index("session_name")["shot_type"].to_dict()
        train_sids = [session_ids[i] for i in idx_train]
        seq_shot_types = [shot_types.get(sid, "Unknown") for sid in train_sids]
        unique_shots, counts = np.unique(seq_shot_types, return_counts=True)
        weight_dict = {shot: len(seq_shot_types) / (len(unique_shots) * count) for shot, count in zip(unique_shots, counts)}
        w_train = np.array([weight_dict[shot] for shot in seq_shot_types])
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
        
        for name, use_conv, use_attention in architectures:
            model = build_model(use_conv, use_attention, SEQ_LEN, len(FEATURE_COLS))
            model.fit(X_train, y_train, validation_data=(X_val, y_val), sample_weight=w_train, 
                      epochs=args.epochs, batch_size=16, callbacks=callbacks, verbose=0)
            
            _, val_mae = model.evaluate(X_val, y_val, verbose=0)
            results[name].append(val_mae * 100.0) # 0-100 scale
            print(f"{name}: Fold {fold+1} MAE = {val_mae * 100.0:.2f} pts")
            
            # If this is the final architecture, calculate per-metric MAE
            if name == "Conv1D + LSTM + Attn":
                preds = model.predict(X_val, verbose=0)
                for i, col in enumerate(SCORE_COLS):
                    metric_mae = np.mean(np.abs(preds[:, i] - y_val[:, i])) * 100.0
                    final_model_per_metric_mae[col].append(metric_mae)
                    
    print("\n\n" + "="*50)
    print("TABLE 2: ABLATION STUDY RESULTS (5-Fold CV MAE)")
    print("="*50)
    for name in results:
        mean_mae = np.mean(results[name])
        std_mae = np.std(results[name])
        print(f"{name:<25}: {mean_mae:.2f} ± {std_mae:.2f} pts")
        
    print("\n" + "="*50)
    print("TABLE 3: PER-METRIC MAE FOR TCN-ATTENTION MODEL")
    print("="*50)
    for col in SCORE_COLS:
        mean_mae = np.mean(final_model_per_metric_mae[col])
        std_mae = np.std(final_model_per_metric_mae[col])
        print(f"{col.replace('score_', '').capitalize():<25}: {mean_mae:.2f} ± {std_mae:.2f} pts")

if __name__ == "__main__":
    main()
