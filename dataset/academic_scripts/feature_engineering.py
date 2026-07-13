import pandas as pd
import numpy as np

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
    
    # Cosine of angle
    dot_prod = np.sum(v1 * v2, axis=1)
    mag1 = np.linalg.norm(v1, axis=1)
    mag2 = np.linalg.norm(v2, axis=1)
    
    # Avoid division by zero
    mag_prod = mag1 * mag2
    mag_prod[mag_prod == 0] = 1e-10
    
    cos_angle = dot_prod / mag_prod
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
    
    mag_prod = mag1 * mag2
    mag_prod[mag_prod == 0] = 1e-10
    
    cos_angle = dot_prod / mag_prod
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angles = np.degrees(np.arccos(cos_angle))
    return angles

def main():
    print("Loading dataset.csv...")
    try:
        df = pd.read_csv("dataset.csv")
    except FileNotFoundError:
        print("dataset.csv not found!")
        return
    
    angles_df = pd.DataFrame()
    
    # Keep metadata
    meta_cols = ["session_name", "frame_name", "shot_type", "batting_hand", "skill_level", 
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
    
    # Ensure every session has exactly 7 frames (Pad missing frames)
    print("Padding sequences to exactly 7 frames...")
    canonical_frames = [
        'frame_01_stance.jpg', 'frame_02_trigger.jpg', 'frame_03_backlift_start.jpg', 
        'frame_04_full_backlift.jpg', 'frame_05_downswing.jpg', 'frame_06_contact.jpg', 
        'frame_07_followthrough.jpg'
    ]
    
    def pad_group(group):
        session_id = group.name
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
    
    angles_df.to_csv("dataset_angles.csv", index=False)
    print(f"Successfully generated dataset_angles.csv with {len(angles_df)} rows and {len(angles_df.columns) - len(meta_cols)} features.")

if __name__ == "__main__":
    main()
