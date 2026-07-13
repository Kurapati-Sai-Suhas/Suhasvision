import pandas as pd
import numpy as np
import os

def calculate_angle(a, b, c):
    """
    Calculates the 2D angle between three points (a, b, c).
    b is the vertex (e.g. knee), a and c are the endpoints (e.g. hip, ankle).
    Uses x and y coordinates (index 0 and 1).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def score_from_keypoint_df(df):
    """
    Real-time, single-session fallback scorer (SRS FR-ERR-001). Takes a DataFrame
    of raw MediaPipe keypoint columns — one row per phase, same schema
    ml_service.extract_landmarks produces — and returns a 4-metric score dict
    shaped like the ML inference output, for use when the trained model or
    MC-Dropout inference throws.

    Deliberately conservative: the underlying rules only actually measure knee
    flexion and elbow extension, so only Balance and Technique are rule-derived;
    Power and Defence are reported as a neutral midpoint rather than a confident
    number these two rules can't back.
    """
    min_left_knee = 180.0
    min_right_knee = 180.0
    max_left_elbow = 0.0
    max_right_elbow = 0.0

    for _, row in df.iterrows():
        min_left_knee = min(min_left_knee, calculate_angle(
            [row['left_hip_x'], row['left_hip_y']],
            [row['left_knee_x'], row['left_knee_y']],
            [row['left_ankle_x'], row['left_ankle_y']],
        ))
        min_right_knee = min(min_right_knee, calculate_angle(
            [row['right_hip_x'], row['right_hip_y']],
            [row['right_knee_x'], row['right_knee_y']],
            [row['right_ankle_x'], row['right_ankle_y']],
        ))
        max_left_elbow = max(max_left_elbow, calculate_angle(
            [row['left_shoulder_x'], row['left_shoulder_y']],
            [row['left_elbow_x'], row['left_elbow_y']],
            [row['left_wrist_x'], row['left_wrist_y']],
        ))
        max_right_elbow = max(max_right_elbow, calculate_angle(
            [row['right_shoulder_x'], row['right_shoulder_y']],
            [row['right_elbow_x'], row['right_elbow_y']],
            [row['right_wrist_x'], row['right_wrist_y']],
        ))

    front_knee_flexion = min(min_left_knee, min_right_knee)
    front_elbow_extension = max(max_left_elbow, max_right_elbow)

    if 110 <= front_knee_flexion <= 150:
        balance_score, balance_note = 80, "Good weight transfer (optimal knee bend)."
    elif front_knee_flexion < 110:
        balance_score, balance_note = 45, "Knee bent too much (loss of balance)."
    else:
        balance_score, balance_note = 35, "Stiff front leg (poor weight transfer)."

    if 150 <= front_elbow_extension <= 180:
        technique_score, technique_note = 85, "Excellent front arm extension (straight bat)."
    elif 120 <= front_elbow_extension < 150:
        technique_score, technique_note = 60, "Front arm slightly bent."
    else:
        technique_score, technique_note = 35, "Front arm too bent (loss of control/power)."

    power_score = 50
    defence_score = 50
    overall = round((balance_score + power_score + technique_score + defence_score) / 4)

    return {
        "balance_score": balance_score,
        "power_score": power_score,
        "technique_score": technique_score,
        "defence_score": defence_score,
        "overall_score": overall,
        "primary_weakness": balance_note if balance_score < technique_score else technique_note,
        "primary_strength": technique_note if balance_score < technique_score else balance_note,
        "is_fallback": True,
    }


def process_keypoints():
    csv_path = "keypoints.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    print(f"Loaded dataset with {len(df)} frames.")
    
    results = []
    
    # Group by session (video)
    for session_name, session_data in df.groupby('session_name'):
        
        # We will track the minimum (most bent) knee angle and maximum (straightest) elbow angle
        min_left_knee = 180.0
        min_right_knee = 180.0
        max_left_elbow = 0.0
        max_right_elbow = 0.0
        
        for _, row in session_data.iterrows():
            # Left Knee
            l_hip = [row['left_hip_x'], row['left_hip_y']]
            l_knee = [row['left_knee_x'], row['left_knee_y']]
            l_ankle = [row['left_ankle_x'], row['left_ankle_y']]
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            min_left_knee = min(min_left_knee, l_knee_angle)
            
            # Right Knee
            r_hip = [row['right_hip_x'], row['right_hip_y']]
            r_knee = [row['right_knee_x'], row['right_knee_y']]
            r_ankle = [row['right_ankle_x'], row['right_ankle_y']]
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            min_right_knee = min(min_right_knee, r_knee_angle)
            
            # Left Elbow
            l_shldr = [row['left_shoulder_x'], row['left_shoulder_y']]
            l_elbow = [row['left_elbow_x'], row['left_elbow_y']]
            l_wrist = [row['left_wrist_x'], row['left_wrist_y']]
            l_elbow_angle = calculate_angle(l_shldr, l_elbow, l_wrist)
            max_left_elbow = max(max_left_elbow, l_elbow_angle)
            
            # Right Elbow
            r_shldr = [row['right_shoulder_x'], row['right_shoulder_y']]
            r_elbow = [row['right_elbow_x'], row['right_elbow_y']]
            r_wrist = [row['right_wrist_x'], row['right_wrist_y']]
            r_elbow_angle = calculate_angle(r_shldr, r_elbow, r_wrist)
            max_right_elbow = max(max_right_elbow, r_elbow_angle)

        # Assuming the front knee is the one that bends the most during a drive
        front_knee_flexion = min(min_left_knee, min_right_knee)
        
        # Assuming the front elbow is the one that stays the straightest
        front_elbow_extension = max(max_left_elbow, max_right_elbow)
        
        # --- RULE BASED SCORING ---
        score = 0
        
        # Rule 1: Front Knee Flexion (Weight transfer)
        # Optimal bend for a drive is between 110 and 150 degrees.
        if 110 <= front_knee_flexion <= 150:
            score += 40
            knee_feedback = "Good weight transfer (optimal knee bend)."
        elif front_knee_flexion < 110:
            score += 20
            knee_feedback = "Knee bent too much (loss of balance)."
        else:
            score += 10
            knee_feedback = "Stiff front leg (poor weight transfer)."
            
        # Rule 2: Front Elbow Extension (High elbow / straight bat)
        # Optimal is a straight or nearly straight front arm (150-180 degrees)
        if 150 <= front_elbow_extension <= 180:
            score += 40
            elbow_feedback = "Excellent front arm extension (straight bat)."
        elif 120 <= front_elbow_extension < 150:
            score += 25
            elbow_feedback = "Front arm slightly bent."
        else:
            score += 10
            elbow_feedback = "Front arm too bent (loss of control/power)."
            
        # Add 20 base points for just making contact
        score += 20
        
        results.append({
            'session_name': session_name,
            'front_knee_flexion': round(front_knee_flexion, 2),
            'front_elbow_extension': round(front_elbow_extension, 2),
            'score': score,
            'knee_feedback': knee_feedback,
            'elbow_feedback': elbow_feedback
        })

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv("rule_based_scores.csv", index=False)
    
    print("\nScoring Complete!")
    print(results_df[['session_name', 'score']].head(10))
    print("\nFull results saved to 'rule_based_scores.csv'")

if __name__ == "__main__":
    process_keypoints()
