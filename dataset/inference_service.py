import os
import csv
import re
import glob
import shutil
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import zero storage logic and academic script math
from zero_storage_pipeline import extract_features_from_image_array
from academic_scripts.mc_dropout_inference import mc_dropout_predict
from academic_scripts.feature_engineering import calculate_angle, calculate_angle_with_vertical

# Gap C fix: Import EXPECTED_FEATURES from the lightweight schema module.
# This avoids triggering TF+MediaPipe chain just to read a list of strings.
from schema import EXPECTED_FEATURES

# 1. Provide custom layer just in case we load the legacy advanced model instead of simple MLP
@tf.keras.utils.register_keras_serializable()
class TemporalAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# Track the version of the loaded model for accurate sidebar display (Gap F fix)
_loaded_version = None

def reload_model():
    """Force-reloads the model cache. Call after active_learning_retrain.py completes."""
    global _model, _loaded_version
    _model = None
    _loaded_version = None
    return get_model()

def get_current_model_version():
    """Returns the version of the model CURRENTLY LOADED IN MEMORY.
    Gap F fix: returns what's actually loaded, not what's on disk."""
    return _loaded_version

def get_model():
    """Loads the highest-version valid .keras model. Tries candidates from highest to lowest."""
    global _model, _loaded_version
    if _model is None:
        models = glob.glob("cricket_stance_advanced_v*.keras")
        if not models:
            raise FileNotFoundError("No model files (cricket_stance_advanced_v*.keras) found.")

        # Sort by version descending — highest first
        versioned = []
        for m in models:
            match = re.search(r"v(\d+)\.keras", m)
            if match:
                versioned.append((int(match.group(1)), m))
        versioned.sort(key=lambda x: x[0], reverse=True)

        for version, model_path in versioned:
            try:
                _model = tf.keras.models.load_model(
                    model_path, custom_objects={'TemporalAttention': TemporalAttention}
                )
                _loaded_version = version
                print(f"[inference_service] Loaded model: {model_path} (v{version})")
                break
            except Exception as load_err:
                print(f"[inference_service] WARNING: Skipping corrupt model {model_path}: {load_err}")

        if _model is None:
            raise RuntimeError("All .keras model files failed to load. Check for corrupt files.")
    return _model

def extract_video_tensor(video_path, session_name, start_frame, end_frame):
    """
    Extracts 7 frames within the specified bounds, runs MediaPipe Pose, applies kinematics rules,
    computes the 15 angles and 15 velocities. Returns a (1, 7, 30) numpy tensor.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Failed to open video.", None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Gap D fix: clamp and validate frame range before use
    # Prevents reversed timestamps (LLaVA hallucination) and zero-duration windows
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(0, min(end_frame, total_frames - 1))
    if end_frame <= start_frame:
        # Ensure at least a 15-frame span to get meaningful motion
        end_frame = min(start_frame + 15, total_frames - 1)
    if end_frame <= start_frame:
        cap.release()
        return False, f"Video too short ({total_frames} frames) to extract a valid action window.", None

    frame_indices = np.linspace(start_frame, end_frame, 7, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
    cap.release()
    
    if len(frames) != 7:
        return False, "Failed to extract exactly 7 frames for analysis.", None
        
    # Run MediaPipe and apply kinematic rules via shared function
    success, raw_keypoints_or_error, interpolated = extract_features_from_image_array(frames, session_name)

    if not success:
        return False, raw_keypoints_or_error, None
        
    raw_keypoints = raw_keypoints_or_error
        
    # Convert validated keypoints to DataFrame for Feature Engineering
    landmarks_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", 
        "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", 
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", 
        "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", 
        "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", 
        "right_heel", "left_foot_index", "right_foot_index"
    ]
    
    df_data = []
    for i, kp in enumerate(raw_keypoints):
        # Guard: apply_pipeline_rules fills None slots via interpolation.
        # This assert catches any edge-case where a slot is still None after all rules pass.
        if kp is None:
            return False, f"Internal error: keypoint slot {i} is still None after pipeline rules. Session: {session_name}", None
        row = {"frame_name": f"frame_0{i+1}"}
        for j, name in enumerate(landmarks_names):
            row[f"{name}_x"] = kp[j][0]
            row[f"{name}_y"] = kp[j][1]
            row[f"{name}_z"] = kp[j][2]
        df_data.append(row)
        
    df = pd.DataFrame(df_data)
    
    # Run Feature Engineering calculations
    angles_df = pd.DataFrame()
    angles_df["angle_knee_L"] = calculate_angle(df, "left_hip", "left_knee", "left_ankle")
    angles_df["angle_knee_R"] = calculate_angle(df, "right_hip", "right_knee", "right_ankle")
    angles_df["angle_hip_L"] = calculate_angle(df, "left_shoulder", "left_hip", "left_knee")
    angles_df["angle_hip_R"] = calculate_angle(df, "right_shoulder", "right_hip", "right_knee")
    angles_df["angle_elbow_L"] = calculate_angle(df, "left_shoulder", "left_elbow", "left_wrist")
    angles_df["angle_elbow_R"] = calculate_angle(df, "right_shoulder", "right_elbow", "right_wrist")
    angles_df["angle_shoulder_L"] = calculate_angle(df, "left_hip", "left_shoulder", "left_elbow")
    angles_df["angle_shoulder_R"] = calculate_angle(df, "right_hip", "right_shoulder", "right_elbow")
    angles_df["angle_ankle_L"] = calculate_angle(df, "left_knee", "left_ankle", "left_foot_index")
    angles_df["angle_ankle_R"] = calculate_angle(df, "right_knee", "right_ankle", "right_foot_index")
    angles_df["angle_trunk_L"] = calculate_angle_with_vertical(df, "left_hip", "left_shoulder")
    angles_df["angle_trunk_R"] = calculate_angle_with_vertical(df, "right_hip", "right_shoulder")
    angles_df["angle_arm_L"] = calculate_angle_with_vertical(df, "left_shoulder", "left_elbow")
    angles_df["angle_arm_R"] = calculate_angle_with_vertical(df, "right_shoulder", "right_elbow")
    
    df["mid_shoulder_x"] = (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2
    df["mid_shoulder_y"] = (df["left_shoulder_y"] + df["right_shoulder_y"]) / 2
    df["mid_shoulder_z"] = (df["left_shoulder_z"] + df["right_shoulder_z"]) / 2
    angles_df["angle_head_tilt"] = calculate_angle_with_vertical(df, "mid_shoulder", "nose")
    
    # Normalize angles
    angle_cols = [c for c in angles_df.columns if c.startswith("angle_")]
    angles_df[angle_cols] = angles_df[angle_cols] / 180.0
    
    # Calculate Velocities
    for col in angle_cols:
        # Avoid fillna(0) mimicking zero motion if train script didn't. 
        # Actually feature_engineering.py DOES use `.diff().fillna(0)`
        angles_df[col + "_vel"] = angles_df[col].diff().fillna(0)
        
    # EXPECTED_FEATURES is now a module-level constant (defined at top of file).
    # No local redefinition needed.
    

    # Strip metadata columns to isolate features
    feature_cols = [c for c in angles_df.columns if c in EXPECTED_FEATURES]
    angles_df = angles_df[feature_cols]
    
    if list(angles_df.columns) != EXPECTED_FEATURES:
        return False, f"Feature column mismatch! Model expects {EXPECTED_FEATURES}, got {list(angles_df.columns)}", None
        
    tensor_2d = angles_df.values
    
    EXPECTED_HEADER = ["session_name", "frame_name"] + EXPECTED_FEATURES

    # --- Phase 2: Active Learning Kinematics Logging ---
    log_file = "anonymized_inference_tensors.csv"
    file_exists = os.path.isfile(log_file)

    # Gap 8 fix: if file exists, verify its header matches current EXPECTED_FEATURES.
    # A schema change (adding/removing an angle) would otherwise silently corrupt the log.
    if file_exists:
        with open(log_file, 'r', newline='') as fcheck:
            existing_header = next(csv.reader(fcheck), None)
        if existing_header != EXPECTED_HEADER:
            import shutil as _shutil
            archive = log_file.replace(".csv", f"_schema_mismatch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            _shutil.move(log_file, archive)
            print(f"[inference_service] WARNING: Tensor log schema mismatch. Archived old log -> {archive}")
            file_exists = False  # Will trigger fresh header write below

    canonical_frames = [
        'frame_01_stance.jpg', 'frame_02_trigger.jpg', 'frame_03_backlift_start.jpg',
        'frame_04_full_backlift.jpg', 'frame_05_downswing.jpg', 'frame_06_contact.jpg',
        'frame_07_followthrough.jpg'
    ]
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(EXPECTED_HEADER)
        for i in range(7):
            writer.writerow([session_name, canonical_frames[i]] + list(tensor_2d[i]))
    # ---------------------------------------------------
    
    return True, np.expand_dims(tensor_2d, axis=0), interpolated

def explain_prediction(model, tensor):
    """
    Computes Saliency Maps (gradients) to find the exact features that most 
    influenced the AI's prediction. This is true Deep Learning XAI.
    """
    import tensorflow as tf
    from schema import EXPECTED_FEATURES
    import numpy as np
    
    input_tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
    explanations = {}
    categories = ["Balance", "Power", "Technique", "Defence"]
    
    for idx, category in enumerate(categories):
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            preds = model(input_tensor, training=False)
            target_pred = preds[0, idx]
            
        grads = tape.gradient(target_pred, input_tensor)
        # Average absolute gradients across the 7 frames to get overall feature importance
        feature_importance = tf.reduce_mean(tf.abs(grads), axis=1)[0].numpy()
        
        # Get top 3 indices
        top_indices = np.argsort(feature_importance)[-3:][::-1]
        top_features = [EXPECTED_FEATURES[i].replace('_vel', ' Velocity').replace('angle_', '').replace('_L', ' (Left)').replace('_R', ' (Right)').title() for i in top_indices]
        explanations[category] = top_features
        
    return explanations

def predict_scores(tensor):
    """
    Runs model inference using MC Dropout for uncertainty.
    Returns scores scaled to 0-100 range, variance, and XAI explanations.
    """
    try:
        model = get_model()
        # Gap B fix: scale_to_100=True so the [0,1] sigmoid output is multiplied
        # by 100 before returning. Without this, scores are ~0.5 instead of ~50.
        mean_scores, std_scores, _ = mc_dropout_predict(model, tensor, n_passes=30, scale_to_100=True)

        # Clamp to [0, 100] in case of numerical noise
        scores_dict = {
            "Balance":   min(100.0, max(0.0, float(mean_scores[0]))),
            "Power":     min(100.0, max(0.0, float(mean_scores[1]))),
            "Technique": min(100.0, max(0.0, float(mean_scores[2]))),
            "Defence":   min(100.0, max(0.0, float(mean_scores[3])))
        }

        # std_scores are also on 0-100 scale now; mean gives overall confidence spread
        variance = float(np.mean(std_scores))
        
        # Compute True XAI Saliency
        explanations = explain_prediction(model, tensor)

        return True, scores_dict, variance, explanations
    except Exception as e:
        return False, str(e), None

def create_annotated_video(input_path, output_path):
    """
    Reads an MP4, applies MediaPipe Pose to every frame, draws the skeleton,
    and writes it to a new MP4.
    """
    import urllib.request
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    import cv2
    import numpy as np

    model_asset_path = 'pose_landmarker_heavy.task'
    if not os.path.exists(model_asset_path):
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
        urllib.request.urlretrieve(url, model_asset_path)

    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with vision.PoseLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            
            if detection_result.pose_landmarks:
                pose_landmarks = detection_result.pose_landmarks[0]
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])
                
                solutions.drawing_utils.draw_landmarks(
                    frame,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style())
                    
            out.write(frame)
            
    cap.release()
    out.release()
    
    # In some environments, mp4v isn't fully supported by browsers. Streamlit uses h264.
    # To fix this, we can optionally use ffmpeg if available, but for MVP mp4v works in most local envs.
    return True
