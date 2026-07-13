import pandas as pd
import numpy as np
import os
import subprocess
import glob
import re
import shutil
from datetime import datetime

# Gap C fix: Import from lightweight schema.py (no TF/MediaPipe dependency).
# inference_service.py also imports from schema.py — one source of truth, no heavy side effects.
from schema import EXPECTED_FEATURES

LOG_TAG = "[active_learning_retrain]"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {LOG_TAG} {msg}")

def get_latest_model_version():
    """Returns (latest_version_int, latest_model_path) or (None, None) if no valid model found."""
    models = glob.glob("cricket_stance_advanced_v*.keras")
    latest_version = None
    latest_path = None
    for m in models:
        match = re.search(r"v(\d+)\.keras", m)
        if not match:
            continue
        v = int(match.group(1))
        # Bug Fix 5: Validate the file is a real Keras model before treating it as latest.
        # Partial/corrupt files from crashed saves should be skipped.
        try:
            import tensorflow as tf
            tf.keras.models.load_model(m, compile=False)
            if latest_version is None or v > latest_version:
                latest_version = v
                latest_path = m
        except Exception as e:
            log(f"WARNING: Skipping corrupt model file {m}: {e}")
    return latest_version, latest_path

def main():
    log("Starting Active Learning Retraining Loop...")
    
    if not os.path.exists("coach_overrides.csv"):
        log("No coach overrides found. Nothing to train on.")
        return
        
    if not os.path.exists("anonymized_inference_tensors.csv"):
        log("No anonymized inference tensors found.")
        return
    
    if not os.path.exists("dataset_angles.csv"):
        log("ERROR: Master dataset dataset_angles.csv not found.")
        return
        
    # --- Bug Fix 3: Parse timestamp as datetime before sort ---
    overrides = pd.read_csv("coach_overrides.csv")
    try:
        overrides["timestamp"] = pd.to_datetime(overrides["timestamp"], errors="coerce")
        # Drop rows where timestamp failed to parse (malformed rows)
        bad_rows = overrides["timestamp"].isna().sum()
        if bad_rows > 0:
            log(f"WARNING: Dropped {bad_rows} override rows with unparseable timestamps.")
        overrides = overrides.dropna(subset=["timestamp"])
    except Exception as e:
        log(f"WARNING: Could not parse timestamps: {e}. Proceeding without sort safety.")

    # Take the latest override for each session in case of duplicates
    overrides = overrides.sort_values("timestamp").drop_duplicates("session_id", keep="last")
    
    tensors = pd.read_csv("anonymized_inference_tensors.csv")
    
    # Rename override columns for merging
    overrides = overrides.rename(columns={
        "session_id": "session_name",
        "true_balance": "score_balance",
        "true_power": "score_power",
        "true_technique": "score_technique",
        "true_defence": "score_defence"
    })
    
    # Inner join on session_name: only include sessions the pipeline has actually processed
    merged = pd.merge(tensors, overrides, on="session_name", how="inner")
    
    if len(merged) == 0:
        log("No matching tensors found for the coach overrides. Have the sessions been analyzed in the UI first?")
        return
        
    # Fill in metadata columns that training expects
    meta_cols = ["shot_type", "batting_hand", "skill_level", "strength", "weakness", "change_drill"]
    for col in meta_cols:
        merged[col] = "Unknown"
        
    master_df = pd.read_csv("dataset_angles.csv")
    
    # --- Bug Fix 4: Assert no NaN in feature columns before dataset write ---
    for col in master_df.columns:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[master_df.columns]
    
    nan_feature_rows = merged[EXPECTED_FEATURES].isna().any(axis=1).sum()
    if nan_feature_rows > 0:
        bad_cols = merged[EXPECTED_FEATURES].columns[merged[EXPECTED_FEATURES].isna().any()].tolist()
        log(f"ERROR: {nan_feature_rows} rows have NaN in feature columns after merge.")
        log(f"       Columns with NaN: {bad_cols}")
        log("Aborting to prevent corrupting training data. Check that tensor log schema matches master dataset.")
        return
    
    # Filter out sessions already in the master dataset (idempotent re-runs)
    existing_sessions = set(master_df["session_name"].unique())
    new_data = merged[~merged["session_name"].isin(existing_sessions)]
    
    if len(new_data) == 0:
        log("All overrides are already in the master dataset. Nothing new to add.")
        return
        
    n_new = len(new_data["session_name"].unique())
    log(f"Found {n_new} new session(s) to incorporate.")
    
    # Gap 5 fix: Smarter upweight formula.
    # Early in project (<50 sessions): always upweight at 5x to help the model learn quickly.
    # As dataset grows: taper proportionally so coach data doesn't swamp the original distribution.
    n_master = len(master_df["session_name"].unique()) if "session_name" in master_df.columns else len(master_df) // 7
    if n_master < 50:
        upweight = 5  # Critical early-learning phase
    else:
        upweight = max(1, min(5, n_master // max(1, n_new * 10)))
    log(f"Upweighting new sessions by {upweight}x (master: {n_master} sessions, new: {n_new} sessions).")
    upweighted_new_data = pd.concat([new_data] * upweight, ignore_index=True)
    
    # Gap 6 fix: Timestamped backups with retention limit (keep last 3).
    # This lets you roll back to any of the last 3 dataset states, not just the immediate prior.
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"dataset_angles.bak_{ts}.csv"
    shutil.copy2("dataset_angles.csv", backup_path)
    log(f"Backed up dataset_angles.csv -> {backup_path}")
    
    # Prune old backups — keep only the 3 most recent
    all_backups = sorted(glob.glob("dataset_angles.bak_*.csv"), reverse=True)
    for old_backup in all_backups[3:]:
        try:
            os.remove(old_backup)
            log(f"Pruned old backup: {old_backup}")
        except Exception:
            pass
    
    updated_master = pd.concat([master_df, upweighted_new_data], ignore_index=True)
    updated_master.to_csv("dataset_angles.csv", index=False)
    log(f"Updated dataset_angles.csv (was {len(master_df)} rows, now {len(updated_master)} rows).")
    
    # Determine the next model version
    current_version, _ = get_latest_model_version()
    next_version = (current_version + 1) if current_version is not None else 3
    new_model_name = f"cricket_stance_advanced_v{next_version}.keras"
    
    log(f"Retraining -> {new_model_name}")
    cmd = ["python", "academic_scripts/train_advanced_model.py", "--input", "dataset_angles.csv", "--model_out", new_model_name]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log(f"ERROR: Training subprocess failed: {e}")
        log(f"Rolling back dataset_angles.csv from {backup_path}...")
        shutil.copy2(backup_path, "dataset_angles.csv")
        log("Rollback complete. Previous model remains active.")
        return
    
    # Risk 3 Fix: Validate the newly saved model is loadable before declaring success
    try:
        import tensorflow as tf
        tf.keras.models.load_model(new_model_name, compile=False)
        log(f"Model validation passed: {new_model_name} loaded successfully.")
    except Exception as e:
        log(f"ERROR: New model {new_model_name} failed validation after training: {e}")
        log("Rolling back dataset_angles.csv from backup. Previous model remains active.")
        shutil.copy2(backup_path, "dataset_angles.csv")
        if os.path.exists(new_model_name):
            os.remove(new_model_name)
        return
    
    log(f"Retraining complete! {new_model_name} is now the active production model.")
    log(f"The app will automatically pick it up on the next inference call (hot-swap via get_model()).")

    # --- Risk 1 Fix: Archive inference tensor log if it's getting large ---
    tensor_log = "anonymized_inference_tensors.csv"
    if os.path.exists(tensor_log):
        size_mb = os.path.getsize(tensor_log) / (1024 * 1024)
        if size_mb > 50:
            archive_name = f"anonymized_inference_tensors_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            shutil.copy2(tensor_log, archive_name)
            # Keep only the header in the active log (reset for next batch)
            df_tensor = pd.read_csv(tensor_log, nrows=0)
            df_tensor.to_csv(tensor_log, index=False)
            log(f"Tensor log exceeded 50MB. Archived to {archive_name} and reset active log.")

if __name__ == "__main__":
    main()
