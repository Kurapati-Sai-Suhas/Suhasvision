import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

from train_advanced_model import build_tcn_attention_model, build_sequences, FEATURE_COLS, SCORE_COLS, SEQ_LEN

def extract_batsman_name(session_name):
    """
    Extracts the base batsman identity to ensure clones/flips stay in the same group.
    session_name format: 
    - youtube_dataset_10_1_frontview_10s_02
    - youtube_dataset_10_1_frontview_10s_02_flipped
    - PlayerA_side_0s_01
    """
    # 1. Strip _flipped
    base = session_name.replace("_flipped", "")
    
    # 2. Extract core identity
    parts = base.split('_')
    if base.startswith("youtube_dataset"):
        # e.g., youtube_dataset_10_1
        return "_".join(parts[:3])
    else:
        # e.g., PlayerA
        return parts[0]

def check_leakage(train_groups, test_groups, fold):
    intersection = set(train_groups).intersection(set(test_groups))
    assert len(intersection) == 0, f"LEAKAGE DETECTED in Fold {fold}! Batsmen {intersection} are in both train and test."

def run_cv(csv_path="dataset_angles.csv", n_splits=3):
    print(f"Loading {csv_path} for {n_splits}-Fold Grouped Cross Validation...")
    df = pd.read_csv(csv_path)
    
    X, y, session_ids = build_sequences(df)
    groups = [extract_batsman_name(sid) for sid in session_ids]
    
    unique_batsmen = set(groups)
    print(f"Total sessions: {len(X)}")
    print(f"Unique batsmen (groups): {len(unique_batsmen)}")
    
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_maes = {name: [] for name in SCORE_COLS}
    overall_maes = []
    
    print("\n--- Starting Cross-Validation ---")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        # 1. Leakage Check
        train_groups = [groups[i] for i in train_idx]
        test_groups = [groups[i] for i in test_idx]
        check_leakage(train_groups, test_groups, fold)
        
        print(f"\nFold {fold}/{n_splits} | Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"  Test Batsmen: {set(test_groups)}")
        
        # 2. Build and Train Model
        tf.keras.backend.clear_session()
        model = build_tcn_attention_model(SEQ_LEN, len(FEATURE_COLS))
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        # We don't use sample_weights in CV to evaluate raw generalization
        model.fit(
            X[train_idx], y[train_idx],
            epochs=80, batch_size=16,
            validation_data=(X[test_idx], y[test_idx]),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # 3. Evaluate on Held-Out Test Set
        preds = model.predict(X[test_idx], verbose=0)
        
        # MAE per output (converting back to 0-100 scale)
        mae_per_output = np.mean(np.abs(preds - y[test_idx]), axis=0) * 100
        overall_mae = np.mean(mae_per_output)
        
        overall_maes.append(overall_mae)
        for i, name in enumerate(SCORE_COLS):
            fold_maes[name].append(mae_per_output[i])
            
        print(f"  Held-Out MAE: {overall_mae:.2f}")
        for i, name in enumerate(SCORE_COLS):
            print(f"    - {name}: {mae_per_output[i]:.2f}")
            
    print("\n=== FINAL CROSS-VALIDATION RESULTS ===")
    print(f"Overall Held-Out MAE: {np.mean(overall_maes):.2f} ± {np.std(overall_maes):.2f}")
    for name in SCORE_COLS:
        mean_val = np.mean(fold_maes[name])
        std_val = np.std(fold_maes[name])
        print(f"{name}: {mean_val:.2f} ± {std_val:.2f}")

if __name__ == "__main__":
    run_cv()
