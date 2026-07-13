import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Cricket Digital Twin", layout="wide")

st.title("🏏 Cricket Digital Twin Labelling Tool")

# Data loading
DATA_FILE = "keypoints.csv"
LABELS_FILE = "labels.csv"

@st.cache_data
def load_data():
    # Fallback to keypoints1.csv if user renamed it
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    elif os.path.exists("keypoints1.csv"):
        return pd.read_csv("keypoints1.csv")
    else:
        return pd.DataFrame()

def load_labels():
    if not os.path.exists(LABELS_FILE):
        return pd.DataFrame(columns=["session_name", "shot_type", "batting_hand", "skill_level", "strength", "weakness", "change_drill", "score_balance", "score_power", "score_technique", "score_defence"])
    return pd.read_csv(LABELS_FILE)

def save_label(session_name, shot_type, batting_hand, skill_level, strength, weakness, change_drill, score_balance, score_power, score_technique, score_defence):
    labels_df = load_labels()
    # Remove existing label for this session if it exists
    labels_df = labels_df[labels_df["session_name"] != session_name]
    new_row = pd.DataFrame([{
        "session_name": session_name,
        "shot_type": shot_type,
        "batting_hand": batting_hand,
        "skill_level": skill_level,
        "strength": strength,
        "weakness": weakness,
        "change_drill": change_drill,
        "score_balance": score_balance,
        "score_power": score_power,
        "score_technique": score_technique,
        "score_defence": score_defence
    }])
    labels_df = pd.concat([labels_df, new_row], ignore_index=True)
    labels_df.to_csv(LABELS_FILE, index=False)

df = load_data()
if df.empty:
    st.error("No keypoints dataset found in the current directory!")
    st.stop()

labels_df = load_labels()
labeled_sessions = labels_df["session_name"].tolist()

all_sessions = df["session_name"].unique().tolist()
unlabeled_sessions = [s for s in all_sessions if s not in labeled_sessions]

st.sidebar.header("Dataset Progress")
st.sidebar.write(f"**Labeled:** {len(labeled_sessions)} / {len(all_sessions)}")
st.sidebar.progress(len(labeled_sessions) / len(all_sessions) if len(all_sessions) > 0 else 0.0)

if not unlabeled_sessions:
    st.success("🎉 All shots have been labeled! You are ready to train the Neural Network.")
    st.stop()

# Auto-select the first unlabeled session
session_to_label = st.sidebar.selectbox("Select Session to Label", unlabeled_sessions + labeled_sessions)

session_data = df[df["session_name"] == session_to_label].reset_index(drop=True)

st.subheader(f"Session: {session_to_label}")

# Slider to scrub through frames
if len(session_data) > 0:
    frame_idx = st.slider("Scrub Frames (Stance -> Follow-through)", 0, len(session_data)-1, 0)
    frame_row = session_data.iloc[frame_idx]
    
    # Plotly Skeleton
    connections = [
        ("nose", "left_shoulder"), ("nose", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle")
    ]
    
    fig = go.Figure()
    
    # Plot connections
    for joint1, joint2 in connections:
        try:
            x1, y1 = frame_row[f"{joint1}_x"], frame_row[f"{joint1}_y"]
            x2, y2 = frame_row[f"{joint2}_x"], frame_row[f"{joint2}_y"]
            
            # Flip Y axis because image coordinates start 0 at top
            fig.add_trace(go.Scatter(
                x=[x1, x2],
                y=[-y1, -y2],
                mode="lines",
                line=dict(color="cyan", width=3),
                showlegend=False
            ))
        except KeyError:
            pass # Joint not in dataset
            
    # Plot joints
    joints_list = [
        "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
        "left_wrist", "right_wrist", "left_hip", "right_hip", 
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    for joint in joints_list:
        try:
            x, y = frame_row[f"{joint}_x"], frame_row[f"{joint}_y"]
            fig.add_trace(go.Scatter(
                x=[x], y=[-y], mode="markers+text", 
                marker=dict(color="red", size=8), 
                text=[joint.replace("_", " ")], 
                textposition="top center",
                textfont=dict(size=8, color="rgba(255,255,255,0.5)"),
                showlegend=False
            ))
        except KeyError:
            pass
            
    # Try to dynamically scale the plot to fit the player
    try:
        min_x, max_x = session_data[[f"{j}_x" for j in joints_list]].min().min(), session_data[[f"{j}_x" for j in joints_list]].max().max()
        min_y, max_y = session_data[[f"{j}_y" for j in joints_list]].min().min(), session_data[[f"{j}_y" for j in joints_list]].max().max()
        
        # Add padding
        x_pad = (max_x - min_x) * 0.2
        y_pad = (max_y - min_y) * 0.2
        
        range_x = [max(0, min_x - x_pad), min(1, max_x + x_pad)]
        range_y = [-min(1, max_y + y_pad), -max(0, min_y - y_pad)]
    except:
        range_x = [0, 1]
        range_y = [-1, 0]

    fig.update_layout(
        title=f"Frame: {frame_row.get('frame_name', frame_idx)}",
        xaxis=dict(visible=False, range=range_x),
        yaxis=dict(visible=False, range=range_y),
        width=600, height=600,
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font=dict(color="white")
    )
    
    # --- YOUTUBE VIDEO EMBED ---
    try:
        batch_urls = pd.read_csv("batch_urls.csv")
        import re
        # session_to_label is formatted like: youtube_dataset_1_frontview_03
        match = re.match(r"(.*)_(frontview|sideview|backview)_\d+", session_to_label)
        if match:
            batsman_name = match.group(1)
            url_row = batch_urls[batch_urls["batsman_name"] == batsman_name]
            if not url_row.empty:
                video_url = url_row.iloc[0]["url"]
                start_sec = int(url_row.iloc[0]["macro_start_sec"])
                st.markdown("### 🎥 Original Source Video")
                st.write(f"*(Note: Video starts at **{start_sec}s**, you may need to scrub slightly to find this exact shot)*")
                st.video(video_url, start_time=start_sec)
                st.divider()
    except Exception as e:
        pass
    # ---------------------------
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### Label This Shot (4-Factor Output)")
        st.write("Scrub the slider to watch the digital twin swing the bat, then label it below.")
        with st.form("label_form"):
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            with col_meta1:
                shot_type = st.selectbox("Shot Type", ["Cover Drive", "Pull Shot", "Cut Shot", "Straight Drive", "Flick", "Forward Defense", "Other"])
            with col_meta2:
                batting_hand = st.selectbox("Batting hand", ["Right-handed", "Left-handed"])
            with col_meta3:
                skill_level = st.selectbox("Skill level", ["Amateur/Club", "Youth/Academy", "Professional"])
                
            st.markdown("#### Coaching Factors")
            strength = st.text_input("Strength (e.g. Good head position)")
            weakness = st.text_input("Weakness (e.g. Front foot planted late)")
            change_drill = st.text_input("Actionable Change (e.g. Keep weight forward longer)")
            
            st.markdown("#### The 4 Scores (0-100)")
            colA, colB = st.columns(2)
            with colA:
                score_balance = st.slider("Balance", 0, 100, 50)
                score_power = st.slider("Power", 0, 100, 50)
            with colB:
                score_technique = st.slider("Technique", 0, 100, 50)
                score_defence = st.slider("Defence", 0, 100, 50)
            
            submitted = st.form_submit_button("Save Label & Next")
            if submitted:
                save_label(session_to_label, shot_type, batting_hand, skill_level, strength, weakness, change_drill, score_balance, score_power, score_technique, score_defence)
                st.rerun()
