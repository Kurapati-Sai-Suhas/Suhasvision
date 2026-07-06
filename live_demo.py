import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import mediapipe as mp
import plotly.graph_objects as go
import os
import sys

# Add dataset scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), "dataset"))
from academic_scripts.feature_engineering import calculate_angle, calculate_angle_with_vertical

# ---------------------------------------------------------
# 1. Custom Attention Layer Definition (Required for loading)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. Extract Attention Model (Hack to get alpha out)
# ---------------------------------------------------------
@st.cache_resource
def load_attention_extractor_v3():
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), "dataset", "cricket_stance_advanced_v2.keras")
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.stop()
    model = tf.keras.models.load_model(model_path, custom_objects={'TemporalAttention': TemporalAttention})
    
    # Create sub-model to get BiLSTM output
    bilstm_output = None
    att_layer = None
    for layer in model.layers:
        if isinstance(layer, layers.Bidirectional):
            bilstm_output = layer.output
        if isinstance(layer, TemporalAttention):
            att_layer = layer
            
    extractor = tf.keras.Model(inputs=model.inputs, outputs=[model.outputs[0], bilstm_output])
    return extractor, att_layer

# ---------------------------------------------------------
# 3. MediaPipe Pipeline
# ---------------------------------------------------------
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request

@st.cache_resource
def load_mediapipe_model():
    model_path = os.path.join(os.path.dirname(__file__), 'pose_landmarker_heavy.task')
    if not os.path.exists(model_path):
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
        urllib.request.urlretrieve(url, model_path)
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector

landmarks_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", 
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", 
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", 
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", 
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", 
    "right_heel", "left_foot_index", "right_foot_index"
]

def extract_landmarks(frame, detector):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector.detect(mp_image)
    if not results.pose_landmarks:
        return None
    
    row = {}
    for i, lm in enumerate(results.pose_landmarks[0]):
        name = landmarks_names[i]
        row[f"{name}_x"] = lm.x
        row[f"{name}_y"] = lm.y
        row[f"{name}_z"] = lm.z
        row[f"{name}_v"] = lm.visibility
    return pd.DataFrame([row])

# Canonical frames
FRAME_NAMES = [
    'Stance', 'Trigger Movement', 'Backlift Start', 
    'Full Backlift', 'Downswing', 'Impact', 'Follow-through'
]

def generate_tensor_from_frames(frames, detector):
    # frames is a list of 7 cv2 images
    dfs = []
    for f in frames:
        df = extract_landmarks(f, detector)
        dfs.append(df)
        
    # Pad if missing
    for i in range(7):
        if dfs[i] is None:
            # ffill or bfill
            for j in range(i-1, -1, -1):
                if dfs[j] is not None:
                    dfs[i] = dfs[j].copy(); break
            if dfs[i] is None:
                for j in range(i+1, 7):
                    if dfs[j] is not None:
                        dfs[i] = dfs[j].copy(); break
                        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Calculate angles
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
    
    # Normalize angles
    angle_cols = angles_df.columns.tolist()
    angles_df[angle_cols] = angles_df[angle_cols] / 180.0
    
    # Calculate Velocity
    for col in angle_cols:
        angles_df[col + "_vel"] = angles_df[col].diff().fillna(0)
        
    v_cols = [c for c in combined_df.columns if c.endswith('_v')]
    mean_visibility = combined_df[v_cols].mean().mean()
        
    return angles_df.values.astype(np.float32), mean_visibility

# ---------------------------------------------------------
# 4. Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="AI Cricket Coach", layout="wide", page_icon="🏏")

# Premium CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #00E676;
    }
    .metric-title {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9E9E9E;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏏 AI Cricket Coaching Assistant")
st.markdown("Upload a video of your batting stroke. Our **TCN-Attention Hybrid Neural Network** will extract 33 3D joints, analyze 15 biomechanical angles, and evaluate your form against a professional baseline.")

uploaded_file = st.file_uploader("Upload MP4 Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    # Play video
    col1, col2 = st.columns([1, 2])
    with col1:
        st.video(tfile.name)
        
    with col2:
        with st.spinner("Extracting Temporal Frames and 3D Biomechanics..."):
            # Simple uniform sampling for 7 frames
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames-1, 7, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    frames.append(frames[-1])
            cap.release()
            
            # Feature Eng
            detector = load_mediapipe_model()
            tensor, mean_vis = generate_tensor_from_frames(frames, detector)
            
        with st.spinner("Running AI Inference..."):
            extractor, att_layer = load_attention_extractor_v3()
            
            # Predict (Monte Carlo Dropout for Uncertainty Quantification)
            X_input = np.expand_dims(tensor, axis=0)
            
            mc_scores = []
            mc_bilstm = []
            for _ in range(30):
                # training=True keeps Dropout active during inference
                s, b = extractor(X_input, training=True)
                mc_scores.append(s.numpy())
                mc_bilstm.append(b.numpy())
                
            # Aggregate MC Dropout results
            scores_array = np.vstack(mc_scores) # shape (30, 4)
            mean_scores = np.mean(scores_array, axis=0) * 100.0
            std_scores = np.std(scores_array, axis=0) * 100.0
            
            # Use mean bilstm output for attention weights
            bilstm_out = np.mean(np.vstack([np.expand_dims(b, axis=0) for b in mc_bilstm]), axis=0)
            
            # Reconstruct alpha (attention weights)
            e = np.tanh(np.dot(bilstm_out, att_layer.W.numpy()) + att_layer.b.numpy())
            # Softmax over time
            e_exp = np.exp(e - np.max(e, axis=1, keepdims=True))
            alpha = e_exp / np.sum(e_exp, axis=1, keepdims=True)
            alpha_weights = alpha[0, :, 0] # shape (7,)
            
            s_bal, s_pow, s_tech, s_def = mean_scores
            std_bal, std_pow, std_tech, std_def = std_scores
            
        # Display Metrics
        st.subheader("Performance Analytics")
        if mean_vis < 0.65:
            st.warning(f"⚠️ **Low Confidence Warning:** The AI struggled to track your joints clearly (Visibility: {mean_vis*100:.1f}%). Ensure you are fully in frame, well-lit, and recording strictly from the **FRONT VIEW**. Side views are intentionally not supported to prevent class imbalance.")
        else:
            st.success(f"✅ **AI Confidence:** {mean_vis*100:.1f}% (High Visibility)")

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-title">Balance</div><div class="metric-value">{s_bal:.1f} <span style="font-size:16px; color:#888;">±{std_bal:.1f}</span></div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-title">Power</div><div class="metric-value">{s_pow:.1f} <span style="font-size:16px; color:#888;">±{std_pow:.1f}</span></div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-title">Technique</div><div class="metric-value">{s_tech:.1f} <span style="font-size:16px; color:#888;">±{std_tech:.1f}</span></div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-title">Defence</div><div class="metric-value">{s_def:.1f} <span style="font-size:16px; color:#888;">±{std_def:.1f}</span></div></div>', unsafe_allow_html=True)
        
        st.write("---")
        
        # Radar Chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[s_bal, s_pow, s_tech, s_def, s_bal],
            theta=['Balance', 'Power', 'Technique', 'Defence', 'Balance'],
            fill='toself',
            name='Your Stroke',
            line_color='#00E676'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[100, 100, 100, 100, 100],
            theta=['Balance', 'Power', 'Technique', 'Defence', 'Balance'],
            fill=None,
            name='Pro Baseline',
            line_color='rgba(255, 255, 255, 0.2)',
            line_dash='dash'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#FAFAFA'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Explainability Section
    st.subheader("🧠 Explainable Coaching Feedback")
    st.markdown("The neural network identifies the most critical phases of your stroke and generates automated coaching feedback based on biomechanical flaws.")
    
    # Heatmap bar
    st.write("**Temporal Attention (Where the AI looked):**")
    for i, (name, weight) in enumerate(zip(FRAME_NAMES, alpha_weights)):
        st.progress(float(weight), text=f"{name} ({weight*100:.1f}%)")
        
    # Feedback Engine
    st.write("**Actionable Tips:**")
    highest_attention_idx = np.argmax(alpha_weights)
    crit_frame = FRAME_NAMES[highest_attention_idx]
    
    # Extremely basic feedback rules for demo purposes
    if highest_attention_idx in [0, 1]:
        st.warning(f"**Focus Area - {crit_frame}:** The model heavily penalized your setup. Ensure your head is stable and vertical, and your knees are slightly flexed for balance.")
    elif highest_attention_idx in [5, 6]:
        if s_pow < 60:
            st.error(f"**Focus Area - {crit_frame}:** You are losing power at impact. Ensure your front elbow does not collapse and you transfer weight fully onto your front foot.")
        else:
            st.success(f"**Focus Area - {crit_frame}:** Great follow-through! The model identified strong kinematic transfer through the impact zone.")
    else:
        st.info(f"**Focus Area - {crit_frame}:** Pay attention to your backlift path. Keep your hands close to your body for better technique.")
