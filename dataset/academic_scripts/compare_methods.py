"""
compare_methods.py
Builds the core results table for the IEEE paper: rule-based scorer vs
Gemini zero-shot vs custom Bi-LSTM vs human ground truth, evaluated on the
same sessions. This single table is the paper's central evidence.

USAGE
    python compare_methods.py \
        --labels labels.csv \
        --rule_based rule_based_scores.csv \
        --gemini gemini_scores.csv \
        --lstm_predictions lstm_predictions.csv

Each input CSV must have a session_id column and a recognizable score
column. Produces comparison_table.csv.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_scores(path: str, score_col_candidates=("overall_score", "quality_score")):
    df = pd.read_csv(path)
    for col in score_col_candidates:
        if col in df.columns:
            return df.set_index("session_id")[col]
    raise ValueError(f"No recognizable score column in {path}. Looked for {score_col_candidates}.")


def evaluate(human: pd.Series, method: pd.Series, method_name: str) -> dict:
    common = human.index.intersection(method.index)
    h = human.loc[common].astype(float)
    m = method.loc[common].astype(float)

    mae = mean_absolute_error(h, m)
    rmse = np.sqrt(mean_squared_error(h, m))
    r, p = pearsonr(h, m)

    return {
        "method": method_name,
        "n_sessions": len(common),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "pearson_r": round(r, 3),
        "p_value": f"{p:.2e}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--rule_based", default="rule_based_scores.csv")
    ap.add_argument("--gemini", default="gemini_scores.csv")
    ap.add_argument("--lstm_predictions", default=None,
                     help="CSV with session_id, predicted_score columns")
    ap.add_argument("--output", default="comparison_table.csv")
    args = ap.parse_args()

    human = load_scores(args.labels, ("quality_score",))
    results = []

    try:
        rule_based = load_scores(args.rule_based, ("overall_score", "rule_based_score"))
        results.append(evaluate(human, rule_based, "Rule-based scorer"))
    except (FileNotFoundError, ValueError) as e:
        print(f"Skipping rule-based: {e}")

    try:
        gemini = load_scores(args.gemini, ("overall_score",))
        results.append(evaluate(human, gemini, "Gemini 2.0 Flash (zero-shot)"))
    except (FileNotFoundError, ValueError) as e:
        print(f"Skipping Gemini: {e}")

    if args.lstm_predictions:
        try:
            lstm = pd.read_csv(args.lstm_predictions).set_index("session_id")["predicted_score"]
            results.append(evaluate(human, lstm, "Custom Bi-LSTM"))
        except FileNotFoundError as e:
            print(f"Skipping LSTM: {e}")

    table = pd.DataFrame(results)
    table.to_csv(args.output, index=False)

    print("\n" + "=" * 70)
    print("COMPARISON TABLE (lower MAE/RMSE better, higher r better)")
    print("=" * 70)
    print(table.to_string(index=False))
    print(f"\nSaved to {args.output}")
    print("\nNote: with very few sessions, p-values are unreliable -- treat "
          "pearson_r as descriptive until n_sessions exceeds roughly 30 per method.")


if __name__ == "__main__":
    main()
