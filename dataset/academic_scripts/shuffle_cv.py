import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from train_advanced_model import build_tcn_attention_model, build_sequences, FEATURE_COLS, SCORE_COLS, SEQ_LEN
from cross_validate import extract_batsman_name

def check_leakage(train_groups, test_groups, fold):
    intersection = set(train_groups).intersection(set(test_groups))
    assert len(intersection) == 0, f"LEAKAGE DETECTED in Fold {fold}!"
    print(f"    - Leakage Check PASS: 0 overlapping batsmen")

def run_shuffled_cv(csv_path="../dataset_angles.csv", runs=3):
    df = pd.read_csv(csv_path)
    X, y, session_ids = build_sequences(df)
    groups = np.array([extract_batsman_name(sid) for sid in session_ids])
    
    unique_batsmen = list(set(groups))
    print(f"Total Unique Batsmen: {len(unique_batsmen)}")
    
    run_maes = []
    
    for run in range(1, runs + 1):
        print(f"\n{'='*40}")
        print(f"=== SHUFFLE RUN {run} ===")
        print(f"{'='*40}")
        
        # Shuffle the 12 batsmen
        random.seed(42 + run) # Different seed per run
        shuffled_batsmen = unique_batsmen.copy()
        random.shuffle(shuffled_batsmen)
        
        # Split into 3 folds manually (4 batsmen each to ensure balance of subjects)
        # We know we have 12 batsmen. So 3 folds of 4 batsmen.
        fold_assignments = {
            1: shuffled_batsmen[0:4],
            2: shuffled_batsmen[4:8],
            3: shuffled_batsmen[8:12]
        }
        
        fold_overall_maes = []
        
        for fold in range(1, 4):
            test_batsmen = fold_assignments[fold]
            train_batsmen = [b for b in unique_batsmen if b not in test_batsmen]
            
            # Get indices
            test_idx = [i for i, g in enumerate(groups) if g in test_batsmen]
            train_idx = [i for i, g in enumerate(groups) if g in train_batsmen]
            
            print(f"\nFold {fold}/3 | Train Seq: {len(train_idx)}, Test Seq: {len(test_idx)}")
            print(f"  Test Batsmen: {test_batsmen}")
            
            # Leakage Check
            train_g = [groups[i] for i in train_idx]
            test_g = [groups[i] for i in test_idx]
            check_leakage(train_g, test_g, fold)
            
            if len(test_idx) == 0 or len(train_idx) == 0:
                print("Skipping fold due to 0 sequences for these batsmen...")
                continue
                
            tf.keras.backend.clear_session()
            model = build_tcn_attention_model(SEQ_LEN, len(FEATURE_COLS))
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            )
            
            model.fit(
                X[train_idx], y[train_idx],
                epochs=80, batch_size=16,
                validation_data=(X[test_idx], y[test_idx]),
                callbacks=[early_stopping],
                verbose=0
            )
            
            preds = model.predict(X[test_idx], verbose=0)
            mae_per_output = np.mean(np.abs(preds - y[test_idx]), axis=0) * 100
            overall_mae = np.mean(mae_per_output)
            fold_overall_maes.append(overall_mae)
            
            print(f"  Held-Out MAE: {overall_mae:.2f}")
            
        run_mean = np.mean(fold_overall_maes)
        run_std = np.std(fold_overall_maes)
        run_maes.append(run_mean)
        print(f"\nRun {run} Overall Held-Out MAE: {run_mean:.2f} ± {run_std:.2f}")
        
    print("\n=== FINAL STABILITY CHECK ===")
    print(f"Mean across all 3 shuffled runs: {np.mean(run_maes):.2f}")
    for i, run_mae in enumerate(run_maes, 1):
        print(f" - Run {i}: {run_mae:.2f}")

if __name__ == "__main__":
    run_shuffled_cv()
