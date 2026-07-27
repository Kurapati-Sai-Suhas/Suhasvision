import os
import re
import sys

import pandas as pd
import numpy as np

# Same bootstrap as stance_symmetry_confidence.py: schema.py lives one
# directory up, and when this file is run directly from academic_scripts/
# (python feature_engineering.py ...), Python only auto-adds THIS directory
# to sys.path — the dataset dir must be added explicitly.
_academic_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_dataset_dir = os.path.dirname(_academic_scripts_dir)
if _dataset_dir not in sys.path:
    sys.path.insert(0, _dataset_dir)
from schema import CANONICAL_FRAME_NAMES  # noqa: E402


def normalize_frame_name(name):
    """
    Real data has two coexisting frame_name conventions: the current
    zero_storage_pipeline.py writes bare "01_stance"; older data used
    "frame_01_stance.jpg". Strips both down to the same canonical form
    ("01_stance") so sessions from either era pad/reindex correctly --
    confirmed via real data that a canonical list matching only ONE
    convention silently wipes ALL metadata (scores, bowling_type, etc. to
    NaN) for every session using the other one, not just frame_name.
    """
    name = re.sub(r'^frame_', '', str(name))
    name = re.sub(r'\.(jpg|jpeg|png)$', '', name, flags=re.IGNORECASE)
    return name

def calculate_angle(df, p1_name, p2_name, p3_name):
    """
    Calculates the 3D angle (in degrees) at p2 given points p1, p2, p3.
    """
    # Extract coordinates
    p1 = df[[f"{p1_name}_x", f"{p1_name}_y", f"{p1_name}_z"]].values
    p2 = df[[f"{p2_name}_x", f"{p2_name}_y", f"{p2_name}_z"]].values
    p3 = df[[f"{p3_name}_x", f"{p3_name}_y", f"{p3_name}_z"]].values
    
    # Vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Cosine of angle. Epsilon handling here must match
    # ml_service.py's calculate_angle() exactly (+1e-6 on the denominator)
    # -- this is the training-time feature computation, and any divergence
    # from the live-inference version is a silent train/serve skew.
    dot_prod = np.sum(v1 * v2, axis=1)
    mag1 = np.linalg.norm(v1, axis=1)
    mag2 = np.linalg.norm(v2, axis=1)
    cos_angle = dot_prod / (mag1 * mag2 + 1e-6)
    # Clip to avoid numerical issues outside [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angles = np.degrees(np.arccos(cos_angle))
    return angles

def calculate_angle_with_vertical(df, p1_name, p2_name):
    """
    Calculates the angle between the vector (p1 -> p2) and the global vertical (Y-axis down).
    In MediaPipe, Y goes down, so vertical vector is [0, 1, 0].
    """
    p1 = df[[f"{p1_name}_x", f"{p1_name}_y", f"{p1_name}_z"]].values
    p2 = df[[f"{p2_name}_x", f"{p2_name}_y", f"{p2_name}_z"]].values
    
    v1 = p2 - p1
    v2 = np.array([0.0, 1.0, 0.0]) # Vertical down
    
    dot_prod = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1, axis=1)
    mag2 = 1.0

    # Same epsilon convention as calculate_angle() above / ml_service.py.
    cos_angle = dot_prod / (mag1 * mag2 + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angles = np.degrees(np.arccos(cos_angle))
    return angles

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset.csv")
    ap.add_argument("--output", default="dataset_angles.csv")
    args = ap.parse_args()

    print(f"Loading {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"{args.input} not found!")
        return
    
    angles_df = pd.DataFrame()
    
    # Keep metadata
    meta_cols = ["session_name", "frame_name", "shot_type", "batting_hand", "bowling_type", "skill_level",
                 "strength", "weakness", "change_drill",
                 "score_balance", "score_power", "score_technique", "score_defence"]
    for col in meta_cols:
        if col in df.columns:
            angles_df[col] = df[col]
            
    print("Calculating 15 Biomechanical Joint Angles...")
    
    # 1-2. Knee Flexion
    angles_df["angle_knee_L"] = calculate_angle(df, "left_hip", "left_knee", "left_ankle")
    angles_df["angle_knee_R"] = calculate_angle(df, "right_hip", "right_knee", "right_ankle")
    
    # 3-4. Hip Flexion
    angles_df["angle_hip_L"] = calculate_angle(df, "left_shoulder", "left_hip", "left_knee")
    angles_df["angle_hip_R"] = calculate_angle(df, "right_shoulder", "right_hip", "right_knee")
    
    # 5-6. Elbow Extension
    angles_df["angle_elbow_L"] = calculate_angle(df, "left_shoulder", "left_elbow", "left_wrist")
    angles_df["angle_elbow_R"] = calculate_angle(df, "right_shoulder", "right_elbow", "right_wrist")
    
    # 7-8. Shoulder Rotation (Hip - Shoulder - Elbow)
    angles_df["angle_shoulder_L"] = calculate_angle(df, "left_hip", "left_shoulder", "left_elbow")
    angles_df["angle_shoulder_R"] = calculate_angle(df, "right_hip", "right_shoulder", "right_elbow")
    
    # 9-10. Ankle Dorsiflexion
    angles_df["angle_ankle_L"] = calculate_angle(df, "left_knee", "left_ankle", "left_foot_index")
    angles_df["angle_ankle_R"] = calculate_angle(df, "right_knee", "right_ankle", "right_foot_index")
    
    # 11-12. Trunk Lean (Angle from vertical)
    angles_df["angle_trunk_L"] = calculate_angle_with_vertical(df, "left_hip", "left_shoulder")
    angles_df["angle_trunk_R"] = calculate_angle_with_vertical(df, "right_hip", "right_shoulder")
    
    # 13-14. Arm Elevation (Angle from vertical)
    angles_df["angle_arm_L"] = calculate_angle_with_vertical(df, "left_shoulder", "left_elbow")
    angles_df["angle_arm_R"] = calculate_angle_with_vertical(df, "right_shoulder", "right_elbow")
    
    # 15. Head Tilt (Nose to mid-shoulder against vertical)
    # create mid-shoulder temporarily
    df["mid_shoulder_x"] = (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2
    df["mid_shoulder_y"] = (df["left_shoulder_y"] + df["right_shoulder_y"]) / 2
    df["mid_shoulder_z"] = (df["left_shoulder_z"] + df["right_shoulder_z"]) / 2
    angles_df["angle_head_tilt"] = calculate_angle_with_vertical(df, "mid_shoulder", "nose")
    
    # Normalize angles to [0, 1] range to feed into Neural Network
    # Angles are [0, 180], so divide by 180.0
    angle_cols = [c for c in angles_df.columns if c.startswith("angle_")]
    angles_df[angle_cols] = angles_df[angle_cols] / 180.0
    
    # Ensure every session has exactly 7 frames (Pad missing frames).
    # Canonical names come from schema.py (audit H6) — an independent stale
    # copy of this list is exactly how the 42%-NaN reindex bug happened.
    # list(), not the tuple itself: pandas reindex gets a list indexer.
    print("Padding sequences to exactly 7 frames...")
    canonical_frames = list(CANONICAL_FRAME_NAMES)

    def pad_group(group):
        session_id = group.name
        group = group.copy()
        group["frame_name"] = group["frame_name"].apply(normalize_frame_name)
        group = group.drop_duplicates(subset=["frame_name"])
        group = group.set_index("frame_name").reindex(canonical_frames)
        group = group.ffill().bfill().reset_index()
        group["session_name"] = session_id
        # For non-numeric columns like shot_type, ffill/bfill works too.
        return group
        
    angles_df = angles_df.groupby("session_name", group_keys=False).apply(pad_group).reset_index(drop=True)
    
    # Calculate Angular Velocity (Delta Angle)
    print("Calculating Temporal Derivatives (Angular Velocities)...")
    angles_df = angles_df.sort_values(["session_name", "frame_name"])
    for col in angle_cols:
        angles_df[col + "_vel"] = angles_df.groupby("session_name")[col].diff().fillna(0)
    
    angles_df.to_csv(args.output, index=False)
    print(f"Successfully generated {args.output} with {len(angles_df)} rows and {len(angles_df.columns) - len(meta_cols)} features.")

if __name__ == "__main__":
    main()
