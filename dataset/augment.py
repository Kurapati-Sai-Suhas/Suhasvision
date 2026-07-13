import pandas as pd
import numpy as np

# Std-dev of the injected noise, in the same normalized [0,1] MediaPipe coordinate
# space as x/y/z. ~0.5% of the frame is small enough to simulate pose-estimation
# jitter without moving a joint into a physically different position.
JITTER_STD = 0.005


def jitter_keypoints(df_in, seed, suffix):
    """
    Adds small Gaussian noise to every x/y/z keypoint coordinate, simulating the
    frame-to-frame pose-estimation jitter MediaPipe actually produces on real video.
    This is the augmentation the SRS has always claimed exists; until now, flipping
    left/right for handedness was the only augmentation actually implemented.
    """
    rng = np.random.default_rng(seed)
    df = df_in.copy()
    coord_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    noise = rng.normal(loc=0.0, scale=JITTER_STD, size=(len(df), len(coord_cols)))
    df[coord_cols] = df[coord_cols].to_numpy() + noise
    if "session_name" in df.columns:
        df["session_name"] = df["session_name"] + suffix
    return df


def flip_keypoints(df_in):
    df = df_in.copy()
    # 1. Flip X coordinates (MediaPipe normalizes X to 0.0 - 1.0)
    x_cols = [c for c in df.columns if c.endswith("_x")]
    df[x_cols] = 1.0 - df[x_cols]

    # 2. Swap Left and Right column names
    rename_map = {}
    for col in df.columns:
        if "left_" in col:
            rename_map[col] = col.replace("left_", "right_")
        elif "right_" in col:
            rename_map[col] = col.replace("right_", "left_")
    
    # Actually swap the data by renaming columns
    df = df.rename(columns=rename_map)
    
    # 3. Rename session to avoid duplicates
    if "session_name" in df.columns:
        df["session_name"] = df["session_name"] + "_flipped"
        
    return df

def main():
    print("Loading dataset.csv...")
    try:
        df = pd.read_csv("dataset.csv")
    except FileNotFoundError:
        print("dataset.csv not found! Run merge.py first.")
        return

    # This script derives augmented data from the BASE sessions only (i.e. not
    # from sessions this same script already produced), so it's safe to re-run
    # without compounding flips-of-flips or jitter-of-jitter on every call.
    is_augmented = df["session_name"].str.contains("_flipped|_jitter", regex=True)
    base_df = df[~is_augmented].copy()
    print(f"Base (non-augmented) rows: {len(base_df)}")

    flipped_df = flip_keypoints(base_df)
    jitter1_df = jitter_keypoints(base_df, seed=42, suffix="_jitterA")
    jitter2_df = jitter_keypoints(flipped_df, seed=43, suffix="_jitterB")

    augmented_df = pd.concat([base_df, flipped_df, jitter1_df, jitter2_df], ignore_index=True)

    augmented_df.to_csv("dataset.csv", index=False)

    print(f"Augmented dataset saved! New row count: {len(augmented_df)}")
    print(f"Unique sessions mapped: {augmented_df['session_name'].nunique()}")

if __name__ == "__main__":
    main()
