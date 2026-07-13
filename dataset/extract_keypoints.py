import os
import cv2
import csv
from pathlib import Path
from zero_storage_pipeline import extract_features_from_image_array

def run():
    dataset_dir = Path("frames")
    annotated_dir = Path("frames_annotated")
    annotated_dir.mkdir(exist_ok=True)
    
    csv_file = open("keypoints.csv", "w", newline="")
    writer = csv.writer(csv_file)
    
    # 33 landmarks in MediaPipe Pose
    landmarks_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", 
        "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", 
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", 
        "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", 
        "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", 
        "right_heel", "left_foot_index", "right_foot_index"
    ]
    
    header = ["session_name", "frame_name"]
    for name in landmarks_names:
        header.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_v", f"{name}_presence"])
    
    writer.writerow(header)
    
    total_sessions = 0
    successful_sessions = 0
    
    print("Starting MediaPipe Pose extraction using unified pipeline rules...")
    
    if not dataset_dir.exists():
        print(f"Error: {dataset_dir} directory not found.")
        return
        
    for session_dir in dataset_dir.iterdir():
        if not session_dir.is_dir():
            continue
            
        session_name = session_dir.name
        print(f"Processing session: {session_name}")
        
        # Expecting exactly 7 frames as per pipeline rules
        img_paths = sorted(list(session_dir.glob("*.jpg")))
        if len(img_paths) != 7:
            print(f"  [!] Session {session_name} does not have exactly 7 frames. Skipping.")
            continue
            
        total_sessions += 1
        
        frames_rgb = []
        frame_names = []
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                frames_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                frame_names.append(img_path.name)
            else:
                print(f"  [!] Failed to read {img_path.name}")
        
        if len(frames_rgb) != 7:
            continue
            
        success, raw_keypoints_or_error, _interpolated = extract_features_from_image_array(frames_rgb, session_name)
        
        if not success:
            print(f"  [!] Pipeline rejected session {session_name}: {raw_keypoints_or_error}")
            continue
            
        raw_keypoints = raw_keypoints_or_error
        successful_sessions += 1
        
        # Write validated and interpolated keypoints to CSV
        for i, kp in enumerate(raw_keypoints):
            if kp is None:
                # Should not happen if pipeline rules pass, but safeguard
                continue
                
            row = [session_name, frame_names[i]]
            for lm in kp:
                row.extend(lm)
            writer.writerow(row)
                
    csv_file.close()
    
    print(f"\nExtraction complete!")
    print(f"Processed {total_sessions} valid sessions.")
    print(f"Successfully extracted and validated {successful_sessions} sessions.")
    print(f"Keypoints saved to 'keypoints.csv'.")

if __name__ == "__main__":
    run()