import random
import numpy as np
import pandas as pd
import tensorflow as tf

from train_advanced_model import build_tcn_attention_model, build_sequences, FEATURE_COLS, SCORE_COLS, SEQ_LEN
from cross_validate import extract_batsman_name

def run_shuffled_lc(csv_path="../dataset_angles.csv", runs=3):
    df = pd.read_csv(csv_path)
    X, y, session_ids = build_sequences(df)
    groups = np.array([extract_batsman_name(sid) for sid in session_ids])
    
    unique_batsmen = list(set(groups))
    
    fractions = [0.25, 0.50, 0.75, 1.0]
    # Store all fold MAEs for pooled stats
    pooled_results = {frac: [] for frac in fractions}
    
    for run in range(1, runs + 1):
        print(f"\n{'='*40}")
        print(f"=== LC SHUFFLE RUN {run} ===")
        print(f"{'='*40}")
        
        # Shuffle batsmen for CV
        random.seed(100 + run)
        shuffled_batsmen = unique_batsmen.copy()
        random.shuffle(shuffled_batsmen)
        
        fold_assignments = {
            1: shuffled_batsmen[0:4],
            2: shuffled_batsmen[4:8],
            3: shuffled_batsmen[8:12]
        }
        
        for fold in range(1, 4):
            test_batsmen = fold_assignments[fold]
            train_batsmen = [b for b in unique_batsmen if b not in test_batsmen]
            
            test_idx = [i for i, g in enumerate(groups) if g in test_batsmen]
            base_train_idx = [i for i, g in enumerate(groups) if g in train_batsmen]
            
            if len(test_idx) == 0 or len(base_train_idx) == 0:
                continue
                
            for frac in fractions:
                # Randomly subsample train_idx for this fraction
                np.random.seed(200 + run * 10 + fold)
                shuffled_train = list(base_train_idx)
                np.random.shuffle(shuffled_train)
                num_train = int(len(shuffled_train) * frac)
                subset_train_idx = shuffled_train[:num_train]
                
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
                
                preds = model.predict(X[test_idx], verbose=0)
                mae_per_output = np.mean(np.abs(preds - y[test_idx]), axis=0) * 100
                overall_mae = np.mean(mae_per_output)
                
                pooled_results[frac].append(overall_mae)
                print(f"Fold {fold} @ {frac*100:3.0f}%: {overall_mae:.2f}")

    print("\n=== POOLED LEARNING CURVE RESULTS (9 FOLDS PER FRACTION) ===")
    for frac in fractions:
        folds = pooled_results[frac]
        mean_val = np.mean(folds)
        std_val = np.std(folds)
        min_val = np.min(folds)
        max_val = np.max(folds)
        print(f"{frac*100:3.0f}% Data: {mean_val:.2f} ± {std_val:.2f} (Range: {min_val:.2f} - {max_val:.2f})")

if __name__ == "__main__":
    run_shuffled_lc()
