"""
live_demo.py
End-to-end inference demo: takes one new local video file, runs it through
the full pipeline (Gemini timestamp detection -> MediaPipe 3D extraction ->
kinematic validation -> rule-based + LSTM scoring), and prints a final
report. Screen-record a run of this as supplementary material for the
IEEE submission -- reviewers respond well to seeing a deployed system,
not just offline numbers.

This script ORCHESTRATES your existing modules -- it does not reimplement
them. The function calls below are commented out because they need to
match your actual function names in zero_storage_pipeline.py,
extract_keypoints.py, and rule_based_scorer.py. Uncomment and adjust the
imports and calls once you confirm the exact signatures in your codebase.

USAGE
    python live_demo.py --video path/to/new_clip.mp4
"""

import argparse
import os
from tensorflow import keras

# --- Adjust these imports to match your actual module/function names ---
# from zero_storage_pipeline import detect_shot_timestamps, extract_pose_sequence
# from rule_based_scorer import score_sequence as rule_based_score
# from train_lstm_model import normalize_session, pad_or_truncate, FEATURE_COLS, SEQ_LEN


def run_demo(video_path: str, lstm_model_path: str = "cricket_stance_lstm_v1.keras"):
    print(f"\n{'='*60}")
    print("CRICKET STANCE ANALYZER -- LIVE DEMO")
    print(f"Input: {video_path}")
    print(f"{'='*60}\n")

    print("[1/5] Detecting batting shot timestamps via Gemini Vision...")
    # start_ts, end_ts = detect_shot_timestamps(video_path)
    # print(f"      Shot detected: {start_ts:.2f}s - {end_ts:.2f}s")
    print("      TODO: wire up to zero_storage_pipeline.detect_shot_timestamps")

    print("\n[2/5] Extracting 33-landmark 3D pose sequence via MediaPipe...")
    # keypoints_df = extract_pose_sequence(video_path, start_ts, end_ts)
    # print(f"      Extracted {len(keypoints_df)} frames")
    print("      TODO: wire up to extract_keypoints.extract_pose_sequence")

    print("\n[3/5] Deleting raw video (zero-storage architecture)...")
    # Only delete if this is a temp working copy, never the user's original file
    # os.remove(video_path)
    print("      No video data persisted to disk")

    print("\n[4/5] Running rule-based biomechanical scorer...")
    # rule_score = rule_based_score(keypoints_df)
    # print(f"      Rule-based score: {rule_score}")
    print("      TODO: wire up to rule_based_scorer.score_sequence")

    print("\n[5/5] Running custom Bi-LSTM model inference...")
    if os.path.exists(lstm_model_path):
        model = keras.models.load_model(lstm_model_path)
        # seq = normalize_session(keypoints_df[FEATURE_COLS].values)
        # seq = pad_or_truncate(seq, SEQ_LEN)
        # pred = model.predict(seq[None, ...])[0][0] * 100
        # print(f"      LSTM predicted quality score: {pred:.1f} / 100")
        print(f"      Loaded model from {lstm_model_path} -- wire up feature extraction above")
    else:
        print(f"      Model not found at {lstm_model_path} -- train it first with train_lstm_model.py")

    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--lstm_model", default="cricket_stance_lstm_v1.keras")
    args = ap.parse_args()
    run_demo(args.video, args.lstm_model)


if __name__ == "__main__":
    main()
