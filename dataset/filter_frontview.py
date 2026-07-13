"""
Filters dataset.csv down to genuinely front-view sessions only, so the model can be
retrained on a single consistent camera geometry instead of a front+side+back blend.

Usage: python filter_frontview.py
Produces: dataset_frontview.csv (same schema as dataset.csv, front-view sessions only)

Run academic_scripts/feature_engineering.py afterwards against dataset_frontview.csv
(pass --input if you add CLI args, or temporarily swap dataset.csv) to get a matching
dataset_angles_frontview.csv for training.
"""
import pandas as pd


def view_of(session_name: str) -> str:
    n = session_name.lower()
    if "frontview" in n or "_front" in n:
        return "front"
    if "backview" in n or "_back" in n:
        return "back"
    if "sideview" in n or "_side" in n:
        return "side"
    return "unknown"


def main():
    df = pd.read_csv("dataset.csv")
    df["_view"] = df["session_name"].map(view_of)

    counts = df.drop_duplicates("session_name")["_view"].value_counts()
    print("Session counts by view:")
    print(counts.to_string())

    front_df = df[df["_view"] == "front"].drop(columns=["_view"])
    n_sessions = front_df["session_name"].nunique()
    print(f"\nFront-view sessions: {n_sessions}")

    front_df.to_csv("dataset_frontview.csv", index=False)
    print("Wrote dataset_frontview.csv")


if __name__ == "__main__":
    main()
