import os
import sys
import logging
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
from django.conf import settings

logger = logging.getLogger(__name__)

_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "dataset")
if _dataset_dir not in sys.path:
    sys.path.insert(0, _dataset_dir)
from rule_based_scorer import score_from_keypoint_df  # noqa: E402 (path must be set up first)

# 1. Custom Attention Layer Definition (Required for loading the model)
@tf.keras.utils.register_keras_serializable()
class TemporalAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        alpha = tf.keras.activations.softmax(e, axis=1) 
        context = tf.reduce_sum(x * alpha, axis=1) 
        return context

# Globals to hold models in memory
_extractor = None
_att_layer = None
_detector = None

def get_models():
    global _extractor, _att_layer, _detector
    if _extractor is not None and _detector is not None:
        return _extractor, _att_layer, _detector

    # Paths. MODEL_FILENAME is pinned explicitly rather than auto-selected (unlike
    # dataset/inference_service.py's glob-highest-version approach) so a production
    # deploy always knows exactly which trained model is serving requests.
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    model_filename = os.environ.get('CRICKET_MODEL_FILENAME', 'cricket_stance_advanced_v4.keras')
    model_path = os.path.join(base_dir, "dataset", model_filename)
    mp_task_path = os.path.join(base_dir, 'pose_landmarker_heavy.task')
    
    if not os.path.exists(mp_task_path):
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
        urllib.request.urlretrieve(url, mp_task_path)
        
    # Load MediaPipe
    base_options = mp_python.BaseOptions(model_asset_path=mp_task_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    _detector = vision.PoseLandmarker.create_from_options(options)
    
    # Load Keras Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    model = tf.keras.models.load_model(model_path, custom_objects={'TemporalAttention': TemporalAttention})
    bilstm_output = None
    att_layer = None
    for layer in model.layers:
        if isinstance(layer, layers.Bidirectional):
            bilstm_output = layer.output
        if isinstance(layer, TemporalAttention):
            att_layer = layer
            
    _extractor = tf.keras.Model(inputs=model.inputs, outputs=[model.outputs[0], bilstm_output])
    _att_layer = att_layer
    
    return _extractor, _att_layer, _detector

# Feature Engineering Utilities
landmarks_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", 
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", 
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", 
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", 
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", 
    "right_heel", "left_foot_index", "right_foot_index"
]

def calculate_angle(df, point1, point2, point3):
    p1 = df[[f"{point1}_x", f"{point1}_y", f"{point1}_z"]].values
    p2 = df[[f"{point2}_x", f"{point2}_y", f"{point2}_z"]].values
    p3 = df[[f"{point3}_x", f"{point3}_y", f"{point3}_z"]].values
    v1, v2 = p1 - p2, p3 - p2
    cosine_angle = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_angle_with_vertical(df, point1, point2):
    p1 = df[[f"{point1}_x", f"{point1}_y", f"{point1}_z"]].values
    p2 = df[[f"{point2}_x", f"{point2}_y", f"{point2}_z"]].values
    v1 = p2 - p1
    v_vertical = np.array([[0, 1, 0]] * len(df))
    cosine_angle = np.sum(v1 * v_vertical, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v_vertical, axis=1) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_landmarks(frame, detector):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector.detect(mp_image)
    if not results.pose_landmarks:
        return None
    row = {}
    for i, lm in enumerate(results.pose_landmarks[0]):
        name = landmarks_names[i]
        row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"], row[f"{name}_v"] = lm.x, lm.y, lm.z, lm.visibility
    return pd.DataFrame([row])

def run_advanced_inference(video_path):
    extractor, att_layer, detector = get_models()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, 7, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            frames.append(frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8))
    cap.release()
    
    dfs = []
    for f in frames:
        df = extract_landmarks(f, detector)
        dfs.append(df)
        
    # Impute missing frames
    for i in range(7):
        if dfs[i] is None:
            for j in range(i-1, -1, -1):
                if dfs[j] is not None: dfs[i] = dfs[j].copy(); break
            if dfs[i] is None:
                for j in range(i+1, 7):
                    if dfs[j] is not None: dfs[i] = dfs[j].copy(); break
    
    # If video is completely black / empty
    if all(d is None for d in dfs):
        raise ValueError("Could not extract any landmarks from video")

    combined_df = pd.concat(dfs, ignore_index=True)

    # Reliability requirement (SRS FR-ERR-001): if the ML model/inference path
    # fails for any reason, fall back to the rule-based scorer on the same
    # extracted keypoints rather than surfacing a 500 to the user.
    try:
        return _run_model_inference(combined_df, extractor)
    except Exception:
        logger.exception("ML inference failed; falling back to rule-based scorer")
        return score_from_keypoint_df(combined_df)


def _run_model_inference(combined_df, extractor):
    angles_df = pd.DataFrame()
    angles_df["angle_knee_L"] = calculate_angle(combined_df, "left_hip", "left_knee", "left_ankle")
    angles_df["angle_knee_R"] = calculate_angle(combined_df, "right_hip", "right_knee", "right_ankle")
    angles_df["angle_hip_L"] = calculate_angle(combined_df, "left_shoulder", "left_hip", "left_knee")
    angles_df["angle_hip_R"] = calculate_angle(combined_df, "right_shoulder", "right_hip", "right_knee")
    angles_df["angle_elbow_L"] = calculate_angle(combined_df, "left_shoulder", "left_elbow", "left_wrist")
    angles_df["angle_elbow_R"] = calculate_angle(combined_df, "right_shoulder", "right_elbow", "right_wrist")
    angles_df["angle_shoulder_L"] = calculate_angle(combined_df, "left_hip", "left_shoulder", "left_elbow")
    angles_df["angle_shoulder_R"] = calculate_angle(combined_df, "right_hip", "right_shoulder", "right_elbow")
    angles_df["angle_ankle_L"] = calculate_angle(combined_df, "left_knee", "left_ankle", "left_foot_index")
    angles_df["angle_ankle_R"] = calculate_angle(combined_df, "right_knee", "right_ankle", "right_foot_index")
    angles_df["angle_trunk_L"] = calculate_angle_with_vertical(combined_df, "left_hip", "left_shoulder")
    angles_df["angle_trunk_R"] = calculate_angle_with_vertical(combined_df, "right_hip", "right_shoulder")
    angles_df["angle_arm_L"] = calculate_angle_with_vertical(combined_df, "left_shoulder", "left_elbow")
    angles_df["angle_arm_R"] = calculate_angle_with_vertical(combined_df, "right_shoulder", "right_elbow")
    combined_df["mid_shoulder_x"] = (combined_df["left_shoulder_x"] + combined_df["right_shoulder_x"]) / 2
    combined_df["mid_shoulder_y"] = (combined_df["left_shoulder_y"] + combined_df["right_shoulder_y"]) / 2
    combined_df["mid_shoulder_z"] = (combined_df["left_shoulder_z"] + combined_df["right_shoulder_z"]) / 2
    angles_df["angle_head_tilt"] = calculate_angle_with_vertical(combined_df, "mid_shoulder", "nose")

    angle_cols = angles_df.columns.tolist()
    angles_df[angle_cols] = angles_df[angle_cols] / 180.0
    for col in angle_cols:
        angles_df[col + "_vel"] = angles_df[col].diff().fillna(0)

    tensor = angles_df.values.astype(np.float32)
    X_input = np.expand_dims(tensor, axis=0)

    mc_scores = []
    for _ in range(30):
        s, _ = extractor(X_input, training=True)
        mc_scores.append(s.numpy())

    scores_array = np.vstack(mc_scores)
    mean_scores = np.mean(scores_array, axis=0) * 100.0
    # Std across the 30 MC-Dropout passes = the model's own uncertainty in each
    # score. Previously computed nowhere, discarded everywhere.
    std_scores = np.std(scores_array, axis=0) * 100.0

    return {
        "balance_score": int(mean_scores[0]),
        "power_score": int(mean_scores[1]),
        "technique_score": int(mean_scores[2]),
        "defence_score": int(mean_scores[3]),
        "overall_score": int(np.mean(mean_scores)),
        "confidence_variance": {
            "balance": round(float(std_scores[0]), 2),
            "power": round(float(std_scores[1]), 2),
            "technique": round(float(std_scores[2]), 2),
            "defence": round(float(std_scores[3]), 2),
        },
        "primary_weakness": "Requires Review",
        "primary_strength": "Great Form",
        "is_fallback": False,
    }
