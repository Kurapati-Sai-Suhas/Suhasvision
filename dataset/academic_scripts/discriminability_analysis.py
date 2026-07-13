"""
discriminability_analysis.py
Tests whether your stance-quality score actually separates professional
players from amateur/club players -- a simple, intuitive validity check
that's easy for any reviewer to grasp, regardless of their ML background.

PREREQUISITE
labels.csv must include a skill_level column. See dataset_diversity_report.py
for the one-line addition to labelling_tool.py needed to start capturing it.

USAGE
    python discriminability_analysis.py --scores_col quality_score
    python discriminability_analysis.py --scores_file lstm_predictions.csv --scores_col predicted_score
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--scores_file", default=None,
                     help="Optional separate CSV with session_id + score column, "
                          "e.g. lstm_predictions.csv. Defaults to labels.csv itself.")
    ap.add_argument("--scores_col", default="quality_score")
    ap.add_argument("--professional_label", default="Professional")
    ap.add_argument("--amateur_labels", nargs="+", default=["Amateur/Club", "Youth/Academy"])
    ap.add_argument("--output_plot", default="discriminability_roc.png")
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    if "skill_level" not in labels.columns:
        raise SystemExit(
            "labels.csv has no 'skill_level' column. Add it to your labelling "
            "tool first -- see dataset_diversity_report.py for instructions."
        )

    if args.scores_file:
        scores = pd.read_csv(args.scores_file).set_index("session_id")[args.scores_col]
        df = labels.set_index("session_id").join(scores.rename("score"), how="inner")
    else:
        df = labels.set_index("session_id").rename(columns={args.scores_col: "score"})

    df = df[df["skill_level"].isin([args.professional_label] + args.amateur_labels)]
    y_true = (df["skill_level"] == args.professional_label).astype(int)
    y_score = df["score"].astype(float)

    if y_true.nunique() < 2:
        raise SystemExit("Need both professional and amateur sessions to compute AUC.")

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    print(f"Sessions: {len(df)} ({int(y_true.sum())} professional, {int((1-y_true).sum())} amateur)")
    print(f"AUC (score separates professional vs amateur): {auc:.3f}")
    if auc >= 0.80:
        verdict = "strong discriminative validity"
    elif auc >= 0.70:
        verdict = "moderate discriminative validity"
    elif auc >= 0.60:
        verdict = "weak but present discriminative validity"
    else:
        verdict = "little to no discriminative validity -- investigate scoring rubric"
    print(f"Interpretation: {verdict}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#2471A3", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Professional vs amateur discriminability")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(args.output_plot, dpi=150)
    print(f"ROC curve saved to {args.output_plot}")


if __name__ == "__main__":
    main()
