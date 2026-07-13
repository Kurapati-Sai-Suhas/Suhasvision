import numpy as np
import pandas as pd
import tensorflow as tf

from train_advanced_model import build_tcn_attention_model, FEATURE_COLS, SCORE_COLS, SEQ_LEN

def build_sequences_buggy(df):
    """Groups rows by session_name, WITHOUT the >100 scaling fix."""
    session_ids = []
    X = []
    y = []
    
    for session_name, group in df.groupby("session_name"):
        if len(group) != SEQ_LEN:
            continue
        group = group.sort_values("frame_name")
        
        seq = group[FEATURE_COLS].values.astype(np.float32)
        scores = group.iloc[0][SCORE_COLS].values.astype(np.float32)
        
        # INTENTIONALLY REMOVING THE BUG FIX:
        # scores = np.where(scores > 100, scores / 10.0, scores)
        
        # We still clip and scale to 0-1 as the model expects sigmoid output
        scores = np.clip(scores, 0, 100)
        scores = scores / 100.0
        
        X.append(seq)
        y.append(scores)
        session_ids.append(session_name)
        
    return np.array(X), np.array(y), session_ids

def run_buggy_eval():
    print("=== Replicating 15.45 MAE (Buggy Labels) ===")
    df = pd.read_csv('../dataset_angles.csv')
    X, y, _ = build_sequences_buggy(df)
    
    tf.keras.backend.clear_session()
    model = build_tcn_attention_model(SEQ_LEN, len(FEATURE_COLS))
    
    print("Training on all data with BUGGY unscaled labels...")
    model.fit(X, y, epochs=80, batch_size=16, verbose=0)
    
    preds = model.predict(X, verbose=0)
    mae_per_output = np.mean(np.abs(preds - y), axis=0) * 100
    overall_mae = np.mean(mae_per_output)
    
    print(f"Buggy Train-on-All / Eval-on-All Overall MAE: {overall_mae:.2f}")

if __name__ == "__main__":
    run_buggy_eval()
