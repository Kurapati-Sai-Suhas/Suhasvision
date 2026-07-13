"""
Diagnostic: reports how many of the 7 canonical phases each session in dataset.csv
actually has real captured frames for, versus how many get silently duplicate-padded
by feature_engineering.py's ffill/bfill step.

Run this right after merge.py on every new batch of extracted video (e.g. after
processing the new front-view YouTube collection) — BEFORE running
feature_engineering.py / augment.py / training — so a low-yield batch gets caught
immediately instead of discovered only once a model is already trained on it.

Usage: python check_frame_yield.py
"""
import pandas as pd


def main():
    df = pd.read_csv("dataset.csv")
    counts = df.groupby("session_name").size()

    print(f"Sessions: {counts.shape[0]}")
    print(f"Mean real frames/session: {counts.mean():.2f}  (target: 7.00)")
    print()
    print("Distribution of real-frame-count per session:")
    print(counts.value_counts().sort_index().to_string())
    print()

    full = (counts == 7).sum()
    partial = ((counts > 0) & (counts < 7)).sum()
    print(f"Sessions with all 7 real phases  : {full} ({full / len(counts) * 100:.1f}%)")
    print(f"Sessions with 1-6 real phases    : {partial} ({partial / len(counts) * 100:.1f}%) "
          f"— these get duplicate-padded by feature_engineering.py, not truly interpolated")

    if full / len(counts) < 0.5:
        print(
            "\n[WARNING] Fewer than half of sessions have all 7 real captured phases. "
            "The model is learning mostly from duplicated frames for most sessions, "
            "which flattens angular velocity to ~0 for the padded phases. Investigate "
            "the phase-timestamp detection step (auto_label_gemini.py) and the "
            "kinematic_validator rejection thresholds before training on this batch."
        )


if __name__ == "__main__":
    main()
