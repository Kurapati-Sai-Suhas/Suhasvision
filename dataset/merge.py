import pandas as pd

def main():
    print("Loading keypoints_validated.csv and labels.csv...")
    try:
        kp = pd.read_csv("keypoints_validated.csv")
        lb = pd.read_csv("labels.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Drop duplicates in labels to prevent cartesian explosion
    lb = lb.drop_duplicates(subset=["session_name"], keep="last")

    # Merge on session_name
    merged = kp.merge(lb, on="session_name", how="inner")
    
    # Save dataset.csv
    merged.to_csv("dataset.csv", index=False)
    
    print(f"dataset.csv successfully created!")
    print(f"Total rows: {len(merged)}")
    print(f"Unique sessions mapped: {merged['session_name'].nunique()}")

if __name__ == "__main__":
    main()
