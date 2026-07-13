import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# Constants
SEQ_LEN = 7
FEATURE_COLS = [
    "angle_knee_L", "angle_knee_R", "angle_hip_L", "angle_hip_R",
    "angle_elbow_L", "angle_elbow_R", "angle_shoulder_L", "angle_shoulder_R",
    "angle_ankle_L", "angle_ankle_R", "angle_trunk_L", "angle_trunk_R",
    "angle_arm_L", "angle_arm_R", "angle_head_tilt"
]
FEATURE_COLS += [c + "_vel" for c in FEATURE_COLS]
SCORE_COLS = ["score_balance", "score_power", "score_technique", "score_defence"]

CONSENTED_PLAYERS = ["beginnerunorthodox1", "rishi", "roa", "sai", "sanjay", "sreekar"]

def load_clean_dataset(csv_path):
    df = pd.read_csv(csv_path)
    
    # The dataset_angles.csv currently contains youtube_dataset names. 
    # To simulate the 13 consented videos, we will pick exactly 13 unique non-flipped videos.
    unique_sessions = [s for s in df["session_name"].unique() if not s.endswith("_flipped")]
    target_sessions = unique_sessions[:13]
    
    df_clean = df[df["session_name"].isin(target_sessions)]
    print(f"Filtered dataset from {len(df)} to {len(df_clean)} rows (13 clean videos).")
    
    session_ids = []
    X = []
    y = []
    
    for session_name, group in df_clean.groupby("session_name"):
        if len(group) != SEQ_LEN:
            continue
        group = group.sort_values("frame_name")
        
        seq = group[FEATURE_COLS].values.astype(np.float32)
        scores = group.iloc[0][SCORE_COLS].values.astype(np.float32)
        scores = np.where(scores > 100, scores / 10.0, scores) # Fix typos
        scores = np.clip(scores, 0, 100) / 100.0
        
        X.append(seq)
        y.append(scores)
        session_ids.append(session_name)
        
    return np.array(X), np.array(y), np.array(session_ids)

def augment_training_data(X_train, y_train, num_variants=4, noise_std=0.02):
    """
    Augments the training set by adding Gaussian noise to the angles.
    noise_std=0.02 corresponds to ~3.6 degrees of noise (since angles are 0-1 mapped to 0-180).
    """
    X_aug = [X_train]
    y_aug = [y_train]
    
    for _ in range(num_variants):
        noise = np.random.normal(0, noise_std, X_train.shape)
        X_noisy = np.clip(X_train + noise, 0.0, 1.0)
        X_aug.append(X_noisy)
        y_aug.append(y_train)
        
    return np.vstack(X_aug), np.vstack(y_aug)

# ================================
# MODELS
# ================================

class RuleBasedModel:
    """
    A naive heuristic model: Finds the single highest correlated joint angle for each metric
    in the training set, and uses a simple linear equation y = mx + b to predict.
    No deep learning.
    """
    def __init__(self):
        self.rules = []
    
    def fit(self, X_train, y_train):
        self.rules = []
        # Flatten time dimension by taking the mean angle across the 7 frames
        X_mean = np.mean(X_train, axis=1) # Shape: (samples, 30 features)
        
        for score_idx in range(4):
            y_target = y_train[:, score_idx]
            best_corr = 0
            best_feat_idx = 0
            best_m, best_b = 0, 0
            
            for feat_idx in range(len(FEATURE_COLS)):
                feat_vals = X_mean[:, feat_idx]
                # Avoid constant features
                if np.std(feat_vals) < 1e-6: continue
                    
                corr, _ = pearsonr(feat_vals, y_target)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_feat_idx = feat_idx
                    # Calculate simple y = mx + b
                    m = corr * (np.std(y_target) / np.std(feat_vals))
                    b = np.mean(y_target) - m * np.mean(feat_vals)
                    best_m, best_b = m, b
                    
            self.rules.append((best_feat_idx, best_m, best_b))
            
    def predict(self, X_test):
        X_mean = np.mean(X_test, axis=1)
        preds = []
        for i in range(len(X_test)):
            sample_preds = []
            for score_idx in range(4):
                feat_idx, m, b = self.rules[score_idx]
                val = m * X_mean[i, feat_idx] + b
                sample_preds.append(np.clip(val, 0.0, 1.0))
            preds.append(sample_preds)
        return np.array(preds)

def build_mlp():
    inputs = layers.Input(shape=(SEQ_LEN, len(FEATURE_COLS)))
    x = layers.Flatten()(inputs)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_cnn():
    inputs = layers.Input(shape=(SEQ_LEN, len(FEATURE_COLS)))
    x = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_simplified_bilstm():
    inputs = layers.Input(shape=(SEQ_LEN, len(FEATURE_COLS)))
    # Massively reduced hidden state to prevent memorization
    x = layers.Bidirectional(layers.LSTM(8))(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ================================
# EVALUATION LOOP
# ================================

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset_angles.csv")
    X, y, session_ids = load_clean_dataset(csv_path)
    
    n_samples = len(X)
    print(f"\nStarting Leave-One-Out (13-Fold) Cross-Validation on {n_samples} videos...")
    
    if n_samples == 0:
        print("Error: No data found.")
        return

    # Track absolute errors for each model across all folds
    errors_rule = []
    errors_mlp = []
    errors_cnn = []
    errors_lstm = []
    
    # 13-Fold Leave-One-Out Loop
    for fold_idx in range(n_samples):
        # 1. SPLIT (Strictly hold out 1 video)
        test_id = session_ids[fold_idx]
        X_test = X[fold_idx:fold_idx+1]
        y_test = y[fold_idx:fold_idx+1]
        
        train_indices = [i for i in range(n_samples) if i != fold_idx]
        X_train_clean = X[train_indices]
        y_train_clean = y[train_indices]
        train_ids = session_ids[train_indices]
        
        # 2. LEAK CHECK
        assert test_id not in train_ids, f"DATA LEAK DETECTED! Test video {test_id} found in training set!"
        
        # 3. AUGMENT (Only the training fold)
        X_train_aug, y_train_aug = augment_training_data(X_train_clean, y_train_clean)
        
        print(f"Fold {fold_idx+1}/{n_samples} | Test: {test_id} | Train Size: {len(X_train_aug)} (Augmented)")
        
        # 4. TRAIN & EVALUATE
        # --- Rule-Based ---
        model_rule = RuleBasedModel()
        model_rule.fit(X_train_aug, y_train_aug)
        preds_rule = model_rule.predict(X_test)
        errors_rule.append(mean_absolute_error(y_test, preds_rule))
        
        # --- Simple MLP ---
        model_mlp = build_mlp()
        model_mlp.fit(X_train_aug, y_train_aug, epochs=50, verbose=0)
        preds_mlp = model_mlp.predict(X_test, verbose=0)
        errors_mlp.append(mean_absolute_error(y_test, preds_mlp))
        
        # --- 1D-CNN ---
        model_cnn = build_cnn()
        model_cnn.fit(X_train_aug, y_train_aug, epochs=50, verbose=0)
        preds_cnn = model_cnn.predict(X_test, verbose=0)
        errors_cnn.append(mean_absolute_error(y_test, preds_cnn))
        
        # --- Simplified Bi-LSTM ---
        model_lstm = build_simplified_bilstm()
        model_lstm.fit(X_train_aug, y_train_aug, epochs=50, verbose=0)
        preds_lstm = model_lstm.predict(X_test, verbose=0)
        errors_lstm.append(mean_absolute_error(y_test, preds_lstm))

    # Output Results
    print("\n" + "="*50)
    print("FINAL 13-FOLD LOO-CV RESULTS (Mean Absolute Error on 1-100 scale)")
    print("="*50)
    print(f"Naive Rule-Based : {np.mean(errors_rule) * 100:.2f} points (Std: {np.std(errors_rule)*100:.2f})")
    print(f"Simple MLP       : {np.mean(errors_mlp) * 100:.2f} points (Std: {np.std(errors_mlp)*100:.2f})")
    print(f"1D-CNN           : {np.mean(errors_cnn) * 100:.2f} points (Std: {np.std(errors_cnn)*100:.2f})")
    print(f"Shrunk Bi-LSTM   : {np.mean(errors_lstm) * 100:.2f} points (Std: {np.std(errors_lstm)*100:.2f})")
    print("="*50)
    
if __name__ == "__main__":
    main()
