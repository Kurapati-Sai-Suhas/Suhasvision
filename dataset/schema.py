# schema.py — Single source of truth for the feature schema.
# Import from here in ANY script that needs EXPECTED_FEATURES.
# This avoids triggering TensorFlow/MediaPipe on import just to read a list of strings.

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
