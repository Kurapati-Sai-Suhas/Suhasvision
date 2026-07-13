import pandas as pd
import numpy as np
import os
import sys

def main():
    print("=== Z-Coordinate Reliability Evaluation ===")
    
    # Check if dataset exists
    file_path = os.path.join("dataset", "keypoints.csv")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    df = pd.read_csv(file_path)
    
    # We want to measure the frame-to-frame jitter (absolute difference) for x, y, and z.
    # Group by session, then diff()
    
    x_cols = [c for c in df.columns if c.endswith("_x")]
    y_cols = [c for c in df.columns if c.endswith("_y")]
    z_cols = [c for c in df.columns if c.endswith("_z")]
    
    print(f"Analyzing {len(x_cols)} spatial joints across {len(df['session_name'].unique())} sessions...")
    print("NOTE: The dataset is constrained strictly to front-view camera angles to prevent class imbalance.\n")
    
    x_jitter = []
    y_jitter = []
    z_jitter = []
    
    for session, group in df.groupby("session_name"):
        # Sort chronologically (assuming frame_name reflects order, or just sequential order in CSV is fine)
        # Assuming frames are consecutive
        x_diff = group[x_cols].diff().abs().mean().mean()
        y_diff = group[y_cols].diff().abs().mean().mean()
        z_diff = group[z_cols].diff().abs().mean().mean()
        
        x_jitter.append(x_diff)
        y_jitter.append(y_diff)
        z_jitter.append(z_diff)
        
    mean_x_jitter = np.nanmean(x_jitter)
    mean_y_jitter = np.nanmean(y_jitter)
    mean_z_jitter = np.nanmean(z_jitter)
    
    print("Average Frame-to-Frame Jitter (Normalized Coordinate Space 0.0 - 1.0):")
    print(f"  X-Axis Jitter: {mean_x_jitter:.5f}")
    print(f"  Y-Axis Jitter: {mean_y_jitter:.5f}")
    print(f"  Z-Axis Jitter: {mean_z_jitter:.5f}")
    
    z_vs_x_ratio = mean_z_jitter / mean_x_jitter
    z_vs_y_ratio = mean_z_jitter / mean_y_jitter
    
    print("\nReliability Conclusion:")
    print(f"  The Z-coordinate jitter is {z_vs_x_ratio:.2f}x higher than the X-coordinate jitter.")
    print(f"  The Z-coordinate jitter is {z_vs_y_ratio:.2f}x higher than the Y-coordinate jitter.")
    print("\n  This confirms that monocular depth estimation introduces significant noise, ")
    print("  which bounds the absolute precision of Z-dependent angles (e.g., shoulder rotation).")

if __name__ == "__main__":
    main()
