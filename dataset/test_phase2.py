from datetime import datetime
import pandas as pd
import numpy as np
import os
import subprocess

EXPECTED_FEATURES = [
    "angle_knee_L", "angle_knee_R", "angle_hip_L", "angle_hip_R", 
    "angle_elbow_L", "angle_elbow_R", "angle_shoulder_L", "angle_shoulder_R", 
    "angle_ankle_L", "angle_ankle_R", "angle_trunk_L", "angle_trunk_R", 
    "angle_arm_L", "angle_arm_R", "angle_head_tilt",
    "angle_knee_L_vel", "angle_knee_R_vel", "angle_hip_L_vel", "angle_hip_R_vel", 
    "angle_elbow_L_vel", "angle_elbow_R_vel", "angle_shoulder_L_vel", "angle_shoulder_R_vel", 
    "angle_ankle_L_vel", "angle_ankle_R_vel", "angle_trunk_L_vel", "angle_trunk_R_vel", 
    "angle_arm_L_vel", "angle_arm_R_vel", "angle_head_tilt_vel"
]

def mock_flow():
    session_id = f"test_mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 1. Mock the UI generating inference tensors (like inference_service does)
    import csv
    log_file = "anonymized_inference_tensors.csv"
    file_exists = os.path.isfile(log_file)
    canonical_frames = [
        'frame_01_stance.jpg', 'frame_02_trigger.jpg', 'frame_03_backlift_start.jpg', 
        'frame_04_full_backlift.jpg', 'frame_05_downswing.jpg', 'frame_06_contact.jpg', 
        'frame_07_followthrough.jpg'
    ]
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["session_name", "frame_name"] + EXPECTED_FEATURES)
        for i in range(7):
            mock_tensor_row = np.random.uniform(0.1, 0.9, 30).tolist()
            writer.writerow([session_id, canonical_frames[i]] + mock_tensor_row)
            
    print("Mock inference tensors written.")
    
    # 2. Mock coach override from UI
    df = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "ai_balance": 50,
        "ai_power": 50,
        "ai_technique": 50,
        "ai_defence": 50,
        "true_balance": 99,
        "true_power": 98,
        "true_technique": 97,
        "true_defence": 96
    }])
    df.to_csv("coach_overrides.csv", mode='a', header=not os.path.exists("coach_overrides.csv"), index=False)
    print("Mock coach override written.")
    
    # 3. Call retrain script
    print("Triggering Active Learning Retraining Loop...")
    subprocess.run(["python", "active_learning_retrain.py"], check=True)

if __name__ == '__main__':
    mock_flow()
