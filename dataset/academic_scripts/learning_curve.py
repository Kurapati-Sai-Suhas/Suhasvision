import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

from train_advanced_model import build_tcn_attention_model, build_sequences, FEATURE_COLS, SCORE_COLS, SEQ_LEN
from cross_validate import extract_batsman_name, check_leakage

def run_learning_curve(csv_path="dataset_angles.csv", n_splits=5):
    print(f"Loading {csv_path} for Learning Curve Analysis...")
    df = pd.read_csv(csv_path)
    
    X, y, session_ids = build_sequences(df)
    groups = np.array([extract_batsman_name(sid) for sid in session_ids])
    
    fractions = [0.25, 0.50, 0.75, 1.0]
    results = {frac: [] for frac in fractions}
    
    gkf = GroupKFold(n_splits=n_splits)
    
    print("\n--- Starting Learning Curve Evaluation ---")
    for frac in fractions:
        print(f"\nEvaluating Training Data Fraction: {frac*100:.0f}%")
        fold_overall_maes = []
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
            # Subsample the training indices
            # We must be careful not to introduce leakage, so we just take a subset of train_idx
            num_train = int(len(train_idx) * frac)
            subset_train_idx = train_idx[:num_train]
            
            # 1. Leakage Check
            train_groups = [groups[i] for i in subset_train_idx]
            test_groups = [groups[i] for i in test_idx]
            check_leakage(train_groups, test_groups, fold)
            
            # 2. Build and Train Model
            tf.keras.backend.clear_session()
            model = build_tcn_attention_model(SEQ_LEN, len(FEATURE_COLS))
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            )
            
            model.fit(
                X[subset_train_idx], y[subset_train_idx],
                epochs=80, batch_size=16,
                validation_data=(X[test_idx], y[test_idx]),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # 3. Evaluate on Held-Out Test Set
            preds = model.predict(X[test_idx], verbose=0)
            mae_per_output = np.mean(np.abs(preds - y[test_idx]), axis=0) * 100
            fold_overall_maes.append(np.mean(mae_per_output))
            
        frac_mae = np.mean(fold_overall_maes)
        results[frac] = frac_mae
        print(f"  -> Mean Held-Out MAE at {frac*100:.0f}% data: {frac_mae:.2f}")

    print("\n=== FINAL LEARNING CURVE ===")
    for frac, mae in results.items():
        print(f"{frac*100:3.0f}% Training Data -> {mae:.2f} MAE")

if __name__ == "__main__":
    run_learning_curve()
