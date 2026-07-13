"""
dataset_diversity_report.py
Characterizes dataset composition for the Dataset section of the IEEE
paper -- reviewers want to know not just N, but the breakdown across
shot type, batting hand, and skill level before they trust generalization
claims.

NOTE: this script expects labels.csv to eventually include two columns
that are NOT in your current schema -- `batting_hand` and `skill_level`.
The script will tell you exactly what's missing and print the one-line
addition needed in labelling_tool.py to start capturing them going
forward. Run this script now anyway to see what you already have.

USAGE
    python dataset_diversity_report.py --labels labels.csv --keypoints keypoints.csv
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--keypoints", default="keypoints.csv")
    ap.add_argument("--out_dir", default="diversity_report")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    keypoints = pd.read_csv(args.keypoints)

    if os.path.exists(args.labels):
        labels = pd.read_csv(args.labels)
        n_sessions = labels["session_name"].nunique() if "session_name" in labels.columns else len(labels)
    else:
        print(f"\n[!] Warning: '{args.labels}' not found (you haven't labeled any data yet).")
        labels = pd.DataFrame()
        n_sessions = keypoints["session_name"].nunique() if "session_name" in keypoints.columns else 0
    n_frames = len(keypoints)
    avg_frames_per_session = n_frames / max(n_sessions, 1)

    print(f"Total sessions: {n_sessions}")
    print(f"Total frames: {n_frames}")
    print(f"Average frames per session: {avg_frames_per_session:.1f}\n")

    summary_lines = [
        "DATASET DIVERSITY REPORT",
        f"Total sessions: {n_sessions}",
        f"Total frames: {n_frames}",
        f"Average frames per session: {avg_frames_per_session:.1f}\n",
    ]

    def report_categorical(col, title):
        if col not in labels.columns:
            msg = f"[MISSING] '{col}' column not found in labels.csv -- add it before submission."
            print(msg)
            summary_lines.append(msg)
            return
        counts = labels[col].value_counts()
        pct = (counts / counts.sum() * 100).round(1)
        print(f"\n{title}:")
        for cat, n in counts.items():
            print(f"  {cat}: {n} ({pct[cat]}%)")
            summary_lines.append(f"  {title} -- {cat}: {n} ({pct[cat]}%)")

        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot(kind="bar", ax=ax, color="#2471A3")
        ax.set_title(title)
        ax.set_ylabel("Sessions")
        plt.tight_layout()
        fig.savefig(f"{args.out_dir}/{col}_distribution.png", dpi=150)
        plt.close(fig)

    report_categorical("shot_type", "Shot type distribution")
    report_categorical("batting_hand", "Batting hand distribution")
    report_categorical("skill_level", "Skill level distribution")

    scores = ["score_balance", "score_power", "score_technique", "score_defence"]
    for score in scores:
        if score in labels.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            labels[score] = pd.to_numeric(labels[score], errors='coerce')
            labels[score].plot(kind="hist", bins=20, ax=ax, color="#117A65")
            ax.set_title(f"{score.replace('_', ' ').title()} Distribution")
            ax.set_xlabel("Score (0-100)")
            plt.tight_layout()
            fig.savefig(f"{args.out_dir}/{score}_distribution.png", dpi=150)
            plt.close(fig)
            summary_lines.append(
                f"\n{score.title()}: mean={labels[score].mean():.1f}, "
                f"std={labels[score].std():.1f}, "
                f"min={labels[score].min()}, max={labels[score].max()}"
            )

    with open(f"{args.out_dir}/summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nFull report and charts saved to {args.out_dir}/")
    print("\nTo add batting_hand and skill_level to your labelling workflow, "
          "add two fields in labelling_tool.py and include them in the row "
          "you append to labels.csv:")
    print('  batting_hand = st.selectbox("Batting hand", ["Right-handed", "Left-handed"])')
    print('  skill_level = st.selectbox("Skill level", '
          '["Amateur/Club", "Youth/Academy", "Professional"])')


if __name__ == "__main__":
    main()
