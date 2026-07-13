import streamlit as st
import cv2
import tempfile
import re as _re
import os
import csv
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from zero_storage_pipeline import log_rejection
from ingest_video import get_ai_timestamps
from inference_service import extract_video_tensor, predict_scores, create_annotated_video, reload_model, get_current_model_version

# --- Setup ---
st.set_page_config(page_title="Cricket Stance Analyzer", page_icon="🏏", layout="wide")

# --- Sidebar: Model Management ---
with st.sidebar:
    st.header("⚙️ Model Management")
    current_v = get_current_model_version()
    if current_v:
        st.info(f"Active model: **v{current_v}**")
    else:
        st.warning("Model not loaded yet.")
    # Gap 9 fix: Expose reload_model() so a coach can hot-swap after retraining
    if st.button("🔄 Reload Model (after retrain)"):
        with st.spinner("Loading latest model..."):
            try:
                reload_model()
                st.success(f"Model reloaded! Now using v{get_current_model_version()}.")
            except Exception as e:
                st.error(f"Failed to reload model: {e}")

def get_cached_timestamps(video_path, file_id):
    if 'action_window' in st.session_state and st.session_state.get('action_window_file_id') == file_id:
        return st.session_state['action_window']
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0, 30.0, 100
        
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    macro_length = total_frames / fps
    try:
        shots = get_ai_timestamps(video_path, macro_length, retries=1)
        if shots and isinstance(shots, list) and len(shots) > 0:
            start_s = float(shots[0].get('start_time', 0))
            end_s = float(shots[0].get('end_time', start_s + 2.0))
        else:
            start_s = max(0, (total_frames / fps) / 2.0 - 1.0)
            end_s = start_s + 2.0
    except Exception as e:
        print(f"[app] Timestamp fallback triggered: {e}")
        # Fallback if Gemini is offline or fails
        start_s = max(0, (total_frames / fps) / 2.0 - 1.0)
        end_s = start_s + 2.0
        
    st.session_state['action_window'] = (start_s, end_s, fps, total_frames)
    st.session_state['action_window_file_id'] = file_id
    return start_s, end_s, fps, total_frames

def detect_blur(video_path, file_id):
    """Basic OpenCV variance of Laplacian for blur detection within the action window."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Failed to open video"
    
    start_s, end_s, fps, total_frames = get_cached_timestamps(video_path, file_id)

    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    variances = []
    frames_checked = 0
    
    while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        variances.append(fm)
        frames_checked += 1
        if frames_checked > 60: # Limit check to max 60 frames for responsiveness
            break
            
    cap.release()
    
    if not variances:
        log_rejection("UI_Upload", "BLUR_REJECTED", "No frames found in action window")
        return False, "No frames to analyze in the action window."
        
    avg_variance = np.mean(variances)
    if avg_variance < 50: # Threshold for blur (adjust based on empirical testing)
        log_rejection("UI_Upload", "BLUR_REJECTED", f"Action window variance: {avg_variance:.2f}")
        return False, f"The batting action is too blurry (Variance: {avg_variance:.2f}). Please upload a clearer video."
    
    return True, "Video clarity is acceptable."


# --- UI Layout ---
st.title("🏏 Cricket Stance Analyzer AI")
st.markdown("Upload a video of your batting stroke to get instant biomechanical feedback.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Upload Video")
    uploaded_file = st.file_uploader("Choose an MP4 video", type=["mp4", "mov"])

    if uploaded_file is not None:
        # Guard tempfile churn: only write if it's a new file
        if 'uploaded_file_id' not in st.session_state or st.session_state['uploaded_file_id'] != uploaded_file.file_id:
            # Clean up old zero-storage temp files
            if 'video_path' in st.session_state and os.path.exists(st.session_state['video_path']):
                try:
                    os.remove(st.session_state['video_path'])
                except Exception as e:
                    print(f"[app] Could not delete old temp video: {e}")
            if 'annotated_video_path' in st.session_state and os.path.exists(
                st.session_state.get('annotated_video_path', '')
            ):
                try:
                    os.remove(st.session_state['annotated_video_path'])
                except Exception as e:
                    # Gap G fix: log instead of silently swallowing (bare except hides PermissionError on Windows)
                    print(f"[app] Could not delete old annotated video: {e}")
                
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()  # Gap 2 fix: close before OpenCV opens it (Windows file lock)
            st.session_state['video_path'] = tfile.name
            st.session_state['uploaded_file_id'] = uploaded_file.file_id
            st.session_state['video_name'] = uploaded_file.name
            # Gap 7 fix: sanitize session_name — strip special chars to make it a safe join key
            raw_name = uploaded_file.name
            safe_name = _re.sub(r'[^A-Za-z0-9_]', '_', raw_name)[:50]
            st.session_state['session_name'] = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        video_path = st.session_state['video_path']
        
        show_skeleton = st.checkbox("Show Biomechanics (Skeleton Overlay)")
        if show_skeleton:
            if 'annotated_video_path' not in st.session_state or st.session_state.get('annotated_file_id') != uploaded_file.file_id:
                with st.spinner("Generating skeleton overlay... This may take a moment."):
                    anno_path = video_path.replace(".mp4", "_annotated.mp4")
                    create_annotated_video(video_path, anno_path)
                    st.session_state['annotated_video_path'] = anno_path
                    st.session_state['annotated_file_id'] = uploaded_file.file_id
            st.video(st.session_state['annotated_video_path'])
        else:
            st.video(video_path)
        
        st.header("2. Video Quality Check")
        with st.spinner("Checking video clarity..."):
            is_clear, blur_msg = detect_blur(video_path, st.session_state['uploaded_file_id'])
            
        if not is_clear:
            st.error(blur_msg)
        else:
            st.success("✅ Video quality is good.")
            
            # Button to trigger analysis
            if st.button("Analyze Biomechanics"):
                with st.spinner("Extracting kinematics and running inference..."):
                    
                    start_s, end_s, fps, total_frames = get_cached_timestamps(video_path, st.session_state['uploaded_file_id'])
                    start_frame = int(start_s * fps)
                    end_frame = int(end_s * fps)
                    
                    success, tensor_or_err, interpolated = extract_video_tensor(video_path, st.session_state['session_name'], start_frame, end_frame)
                    
                    if not success:
                        st.error(f"Extraction Error: {tensor_or_err}")
                    else:
                        p_success, scores_or_err, variance, explanations = predict_scores(tensor_or_err)
                        
                        if not p_success:
                            st.error(f"Inference Error: {scores_or_err}")
                        else:
                            st.session_state['scores'] = scores_or_err
                            st.session_state['variance'] = variance
                            st.session_state['interpolated'] = interpolated
                            st.session_state['explanations'] = explanations
                            st.success("Analysis Complete!")

with col2:
    st.header("3. AI Analytics")
    
    if 'scores' in st.session_state:
        scores = st.session_state['scores']
        variance = st.session_state['variance']
        explanations = st.session_state.get('explanations', {})
        
        st.subheader("Performance Radar")
        df = pd.DataFrame(dict(
            r=[scores['Balance'], scores['Technique'], scores['Power'], scores['Defence']],
            theta=['Balance', 'Technique', 'Power', 'Defence']
        ))
        
        fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r=[0,100])
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Confidence Score")
        if variance > 5.0:
            st.warning(f"⚠️ **Low Confidence Warning:** The AI detected potential inconsistencies (Variance: {variance:.2f}). The predicted scores may be inaccurate due to motion blur or occlusion during the swing. We recommend human coach review.")
        else:
            st.info(f"✅ AI Confidence is high (Variance: {variance:.2f}).")
            
        interpolated = st.session_state.get('interpolated', [])
        if interpolated:
            st.info(f"ℹ️ **Note:** Due to minor occlusions, the AI interpolated frames {interpolated} to salvage the analysis without rejecting the video.")
            
        st.subheader("Biomechanical Feedback (Explainable AI)")
        st.markdown("The neural network computed a **Saliency Map** (gradients) to find the exact biomechanical features that drove your scores. Here are the top 3 features the AI looked at for each category:")
        
        if explanations:
            for category, features in explanations.items():
                st.markdown(f"**{category}** (Score: {scores[category]:.0f})")
                for f in features:
                    st.markdown(f"- 🎯 `{f}`")
        else:
            st.warning("XAI Saliency not available for this run.")
            
        st.divider()
        st.subheader("Coach Override (Ground Truth Capture)")
        st.markdown("Are you a certified coach? If the AI scores are incorrect, please provide the true scores below to help improve the model. Note: Authenticated login will be required in production.")
        
        with st.form("override_form"):
            col_a, col_b = st.columns(2)
            # Gap 3 fix: cast float scores to int for st.number_input value=
            # Streamlit raises ValueError if value type doesn't match min/max type
            with col_a:
                bal_override = st.number_input("True Balance Score", min_value=0, max_value=100, value=int(scores['Balance']))
                pow_override = st.number_input("True Power Score", min_value=0, max_value=100, value=int(scores['Power']))
            with col_b:
                tech_override = st.number_input("True Technique Score", min_value=0, max_value=100, value=int(scores['Technique']))
                def_override = st.number_input("True Defence Score", min_value=0, max_value=100, value=int(scores['Defence']))
            submit_override = st.form_submit_button("Submit True Scores")
            
            if submit_override:
                override_file = "coach_overrides.csv"
                # Gap E fix: Safe header write — check the actual file content, not just existence.
                # This prevents duplicate headers if two users submit at the same moment.
                needs_header = True
                if os.path.isfile(override_file):
                    try:
                        with open(override_file, 'r', newline='', encoding='utf-8') as fcheck:
                            first_line = fcheck.readline().strip()
                        needs_header = first_line != "timestamp,session_id,ai_balance,ai_power,ai_technique,ai_defence,true_balance,true_power,true_technique,true_defence"
                    except Exception:
                        needs_header = False  # File exists, assume header is present
                with open(override_file, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if needs_header:
                        writer.writerow(["timestamp", "session_id", "ai_balance", "ai_power", "ai_technique", "ai_defence", "true_balance", "true_power", "true_technique", "true_defence"])
                    writer.writerow([
                        datetime.now().isoformat(),
                        st.session_state.get('session_name', 'unknown'),
                        scores['Balance'], scores['Power'], scores['Technique'], scores['Defence'],
                        bal_override, pow_override, tech_override, def_override
                    ])
                st.success("Override captured and saved! This data will be used to retrain the model.")
        
    else:
        st.info("Upload and analyze a video to see your biomechanical scores.")
