import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from train_advanced_model import build_tcn_attention_model, build_sequences, FEATURE_COLS, SCORE_COLS, SEQ_LEN
from cross_validate import extract_batsman_name

def verify_dataset():
    print("=== 1. Batsman Name Parsing Verification ===")
    df = pd.read_csv('../dataset_angles.csv')
    X, y, session_ids = build_sequences(df)
    
    # Track the mapping from batsman to sessions to visually verify parsing
    batsman_map = {}
    for sid in session_ids:
        b_name = extract_batsman_name(sid)
        if b_name not in batsman_map:
            batsman_map[b_name] = []
        batsman_map[b_name].append(sid)
        
    unique_batsmen = list(batsman_map.keys())
    print(f"Unique batsmen count: {len(unique_batsmen)}")
    for b in sorted(unique_batsmen):
        print(f" - {b} (e.g. {batsman_map[b][0]})")
        
    print(f"\nTotal full sessions returned by build_sequences: {len(session_ids)}")

    print("\n=== 2. Train-on-All / Eval-on-All (Apples-to-Apples Check) ===")
    # Train on 100% of data, evaluate on 100% of data
    tf.keras.backend.clear_session()
    model = build_tcn_attention_model(SEQ_LEN, len(FEATURE_COLS))
    
    # We will just train for 80 epochs without early stopping (since there is no val set)
    print("Training on all data...")
    model.fit(X, y, epochs=80, batch_size=16, verbose=0)
    
    # Evaluate on all data
    preds = model.predict(X, verbose=0)
    mae_per_output = np.mean(np.abs(preds - y), axis=0) * 100
    overall_mae = np.mean(mae_per_output)
    
    print(f"Train-on-All / Eval-on-All Overall MAE: {overall_mae:.2f}")
    for i, name in enumerate(SCORE_COLS):
        print(f"  - {name}: {mae_per_output[i]:.2f}")

if __name__ == "__main__":
    verify_dataset()
