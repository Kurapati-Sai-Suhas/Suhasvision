"""
agreement_calculator.py
Inter-annotator agreement for the Cricket Stance Analyzer labelling pipeline.

WHY THIS EXISTS
A single labeller's quality_score is not defensible as ground truth in a
peer-reviewed paper. This script compares two (or more) independent
labelling passes over the SAME sessions and reports the agreement
statistics reviewers expect: Cohen's kappa for the categorical shot_type
field, plus quadratic-weighted kappa and ICC(2,1) for the continuous
quality_score field.

WORKFLOW
1. Have a second person (or you, on a different day, blind to your first
   pass) run labelling_tool.py independently over the same sample of
   sessions, saving to a differently-named file, e.g.:
     labels_labeller1.csv
     labels_labeller2.csv
   (Just point the Streamlit tool's save path at a labeller-specific
   filename before each pass.)
2. Run:
     python agreement_calculator.py --files labels_labeller1.csv labels_labeller2.csv
3. Read agreement_report.txt. If kappa/ICC < 0.60, your scoring rubric or
   labelling tool's instructions need to be clearer before you scale up
   labelling -- fix that BEFORE generating more labels, not after.
"""

import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y1, y2, n_bins=10, score_range=(1, 100)):
    """Bins continuous 1-100 scores into n_bins ordinal categories, then
    computes quadratic-weighted Cohen's kappa -- the standard way to assess
    ordinal agreement without a dedicated ICC library installed."""
    lo, hi = score_range
    edges = np.linspace(lo, hi, n_bins + 1)
    b1 = np.digitize(y1, edges[1:-1])
    b2 = np.digitize(y2, edges[1:-1])
    return cohen_kappa_score(b1, b2, weights="quadratic")


def icc_2_1(ratings: pd.DataFrame) -> float:
    """
    Two-way random-effects, single-rater ICC -- ICC(2,1) -- computed from
    scratch via the standard ANOVA formulation. `ratings` should have one
    column per labeller, one row per session.
    """
    data = ratings.dropna().values
    n, k = data.shape  # n sessions, k raters
    if n < 2 or k < 2:
        return float("nan")

    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ss_total = ((data - grand_mean) ** 2).sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else np.nan

    icc = (ms_rows - ms_error) / (
        ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
    )
    return float(icc)


def interpret(k: float) -> str:
    if pd.isna(k):
        return "undefined (insufficient data)"
    if k < 0.20:
        return "poor"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                     help="Two or more labels_*.csv files, one per labeller")
    ap.add_argument("--output", default="agreement_report.txt")
    args = ap.parse_args()

    labellers = {}
    for path in args.files:
        name = path.replace(".csv", "")
        labellers[name] = pd.read_csv(path).set_index("session_id")

    common_sessions = sorted(set.intersection(*[set(df.index) for df in labellers.values()]))
    print(f"Found {len(common_sessions)} sessions labelled by all {len(labellers)} labellers.")

    lines = [
        "INTER-ANNOTATOR AGREEMENT REPORT",
        f"Labellers: {list(labellers.keys())}",
        f"Sessions in common: {len(common_sessions)}\n",
    ]

    # --- shot_type agreement (categorical) ---
    for (n1, df1), (n2, df2) in itertools.combinations(labellers.items(), 2):
        if "shot_type" in df1.columns and "shot_type" in df2.columns:
            y1 = df1.loc[common_sessions, "shot_type"]
            y2 = df2.loc[common_sessions, "shot_type"]
            k = cohen_kappa_score(y1, y2)
            lines.append(f"shot_type Cohen's kappa ({n1} vs {n2}): {k:.3f} ({interpret(k)})")

    # --- quality_score agreement (continuous / ordinal) ---
    score_df = pd.DataFrame({
        name: df.loc[common_sessions, "quality_score"]
        for name, df in labellers.items()
        if "quality_score" in df.columns
    })
    if len(score_df.columns) >= 2:
        icc = icc_2_1(score_df)
        lines.append(f"\nquality_score ICC(2,1) across all labellers: {icc:.3f} ({interpret(icc)})")

        for (n1, df1), (n2, df2) in itertools.combinations(labellers.items(), 2):
            y1 = df1.loc[common_sessions, "quality_score"]
            y2 = df2.loc[common_sessions, "quality_score"]
            qwk = quadratic_weighted_kappa(y1.values, y2.values)
            mad = (y1 - y2).abs().mean()
            lines.append(
                f"quality_score quadratic-weighted kappa ({n1} vs {n2}): {qwk:.3f}  "
                f"| mean absolute difference: {mad:.1f} points"
            )

    lines.append(
        "\nRule of thumb for the paper: kappa/ICC >= 0.60 is generally "
        "considered acceptable for publication; >= 0.75 is strong. If you "
        "land below 0.60, revise the labelling rubric and relabel a small "
        "batch before scaling up -- don't keep labelling on a noisy rubric."
    )

    report = "\n".join(lines)
    print("\n" + report)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
