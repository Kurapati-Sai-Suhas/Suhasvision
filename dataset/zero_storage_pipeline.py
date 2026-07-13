import os
import glob
import time
import csv
import cv2
import urllib.request
import yt_dlp
from moviepy import VideoFileClip
from dotenv import load_dotenv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from nvidia_client import find_shot_windows, label_session_frames, CreditsExhaustedError

def log_rejection(session_name, category, details=""):
    log_file = "pipeline_rejections.log"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {session_name} | {category} | {details}\n")

# ==========================================
# 1. INITIALIZATION & CONFIG
# ==========================================
load_dotenv(override=True)

FULL_YT_VIDEO = "temp_full_youtube.mp4"
AI_CHUNK_VIDEO = "temp_ai_chunk.mp4"
CSV_FILE = "batch_urls.csv"
OUTPUT_CSV = "keypoints.csv"
N_FRAMES = 7

# Initialize MediaPipe Pose Tasks API
model_path = 'pose_landmarker_heavy.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe Pose model...")
    url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

# Ensure CSV has headers
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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
        header.append("interpolated_frames")
        writer.writerow(header)

LABELS_CSV = "labels.csv"
LABELS_HEADER = ["session_name", "shot_type", "batting_hand", "skill_level", "strength", "weakness",
                 "change_drill", "score_balance", "score_power", "score_technique", "score_defence"]


def save_label_row(label_row):
    """Appends one row (from nvidia_client.label_session_frames) to labels.csv."""
    file_exists = os.path.exists(LABELS_CSV)
    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(LABELS_HEADER)
        writer.writerow([label_row.get(col, "") for col in LABELS_HEADER])


def extract_features_from_image_array(frames_rgb, session_name="unknown"):
    """
    Core shared extraction logic. Takes a list of 7 RGB numpy arrays.
    Runs MediaPipe, validates pose topology/visibility, applies pipeline interpolation rules.
    Returns: (success_bool, list_of_raw_keypoints_or_error_string, interpolated_frame_indices)
    """
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import os

    model_asset_path = 'pose_landmarker_heavy.task'
    if not os.path.exists(model_asset_path):
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
        urllib.request.urlretrieve(url, model_asset_path)

    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    
    raw_keypoints = []
    with vision.PoseLandmarker.create_from_options(options) as detector:
        for i, frame_rgb in enumerate(frames_rgb):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = detector.detect(mp_image)

            if not detection_result.pose_landmarks:
                raw_keypoints.append(None)
            else:
                landmarks = detection_result.pose_landmarks[0]
                frame_name = f"frame_0{i+1}"
                
                if validate_pose(landmarks, i, session_name, frame_name):
                    kp_list = [[lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.0), getattr(lm, 'presence', 0.0)] for lm in landmarks]
                    raw_keypoints.append(kp_list)
                else:
                    raw_keypoints.append(None)
                    
    success, interpolated = apply_pipeline_rules(raw_keypoints, session_name)
    if not success:
        return False, "Pose rejected by kinematic validation rules.", None

    return True, raw_keypoints, interpolated

def validate_pose(pose_landmarks, phase_index, session_name, frame_name):
    """Validates topology and visibility based on Phase rules."""
    # Critical joints mapping based on MediaPipe indices
    # 23=left_hip, 24=right_hip, 25=left_knee, 26=right_knee, 27=left_ankle, 28=right_ankle
    # 11=left_shoulder, 12=right_shoulder, 13=left_elbow, 14=right_elbow, 15=left_wrist, 16=right_wrist
    
    # Phase 6 (Contact) is heavily wrist/elbow dependent
    # Phases 1-2 (Stance) are heavily lower-body dependent
    high_value_joints = []
    if phase_index in [0, 1]:
        high_value_joints = [23, 24, 25, 26, 27, 28] # Hips, Knees, Ankles
    elif phase_index in [4, 5]: # Downswing, Contact
        high_value_joints = [13, 14, 15, 16] # Elbows, Wrists
        
    for idx, lm in enumerate(pose_landmarks):
        # MediaPipe sometimes doesn't emit visibility. We assume 0 if missing.
        vis = getattr(lm, 'visibility', 0.0)
        
        # We only check visibility for joints 11-28 (shoulders to ankles) to avoid face rejection
        if 11 <= idx <= 28:
            # In a profile sport like cricket, the far-side arm/leg is naturally occluded.
            # MediaPipe visibility for occluded joints drops near 0.01, but it still accurately 
            # infers their 3D position. We log this for debugging, but we DO NOT reject the pose.
            # The downstream AI model will handle uncertainty, and topology checks will catch broken poses.
            threshold = 0.2 if idx in high_value_joints else 0.05
            if vis < threshold:
                log_rejection(session_name, "LOW_VISIBILITY_WARNING (IGNORED)", f"frame={frame_name}, joint={idx}, vis={vis:.2f}")
                # We no longer `return False` here. We let the 3D inferred joints through!



    # Topology check
    # In MediaPipe, y=0 is top, y=1 is bottom
    l_hip, r_hip = pose_landmarks[23].y, pose_landmarks[24].y
    l_knee, r_knee = pose_landmarks[25].y, pose_landmarks[26].y
    l_ankle, r_ankle = pose_landmarks[27].y, pose_landmarks[28].y
    
    avg_hip = (l_hip + r_hip) / 2.0
    avg_knee = (l_knee + r_knee) / 2.0
    avg_ankle = (l_ankle + r_ankle) / 2.0
    
    if phase_index in [0, 1]: # Strict ordering for stance/trigger
        if avg_ankle < avg_knee or avg_knee < avg_hip:
            log_rejection(session_name, "TOPOLOGY_INVERTED", f"frame={frame_name} failed strict Y ordering")
            return False
    else: # Loose bounds for swing
        # Just ensure the ankle isn't absurdly above the hip (allowing for some leg lift)
        if avg_ankle < avg_hip - 0.2:
            log_rejection(session_name, "TOPOLOGY_INVERTED", f"frame={frame_name} failed loose bounds (ankle above hip)")
            return False
            
    return True


def apply_pipeline_rules(raw_keypoints, session_name):
    status = [1 if kp is not None else 0 for kp in raw_keypoints]
    
    # 1. Contact Phase Check (REMOVED)
    # The moment of contact (frame 5) has the highest motion blur. 
    # Hard-rejecting here prevents our interpolation logic from saving the video.
    # We now let the standard interpolation block reconstruct it if it's missing.

        
    # 2. Cascading Failure Check (Reject if 3 consecutive fail)
    for i in range(5):
        if status[i] == 0 and status[i+1] == 0 and status[i+2] == 0:
            log_rejection(session_name, "HARD_REJECT_CASCADING", f"3 consecutive failures at phase {i}, {i+1}, {i+2}")
            return False, []
            
    # 3. Count Threshold Check
    failed_count = status.count(0)
    if failed_count >= 4:
        log_rejection(session_name, "HARD_REJECT_COUNT", f"Total failed frames: {failed_count} (max 3 allowed)")
        return False, []
        
    interpolated_phases = []
    
    # 4. Edge Frame Extrapolation
    if status[0] == 0:
        if status[1] == 1 and status[2] == 1:
            raw_keypoints[0] = []
            for j in range(len(raw_keypoints[1])):
                kp1, kp2 = raw_keypoints[1][j], raw_keypoints[2][j]
                # Extrapolate backwards: p0 = p1 - (p2 - p1)
                extrap = [kp1[k] - (kp2[k] - kp1[k]) for k in range(3)] + [1.0, 1.0]
                raw_keypoints[0].append(extrap)
            status[0] = 1
            interpolated_phases.append("phase_1")
        else:
            log_rejection(session_name, "HARD_REJECT_EDGE", "Frame 1 failed and references are dirty")
            return False, []
            
    if status[6] == 0:
        if status[4] == 1 and status[5] == 1:
            raw_keypoints[6] = []
            for j in range(len(raw_keypoints[5])):
                kp4, kp5 = raw_keypoints[4][j], raw_keypoints[5][j]
                # Extrapolate forwards: p6 = p5 + (p5 - p4)
                extrap = [kp5[k] + (kp5[k] - kp4[k]) for k in range(3)] + [1.0, 1.0]
                raw_keypoints[6].append(extrap)
            status[6] = 1
            interpolated_phases.append("phase_7")
        else:
            log_rejection(session_name, "HARD_REJECT_EDGE", "Frame 7 failed and references are dirty")
            return False, []
            
    # 5. Standard Interpolation (Handles 1 or 2 consecutive missing frames)
    for i in range(1, 6):
        if status[i] == 0:
            # Check if it's a 2-frame gap
            if i < 5 and status[i+1] == 0:
                if status[i-1] == 1 and status[i+2] == 1:
                    raw_keypoints[i] = []
                    raw_keypoints[i+1] = []
                    for j in range(len(raw_keypoints[i-1])):
                        p0 = raw_keypoints[i-1][j]
                        p3 = raw_keypoints[i+2][j]
                        # Linear interpolation 1/3 and 2/3
                        p1 = [p0[k] + (p3[k] - p0[k]) / 3.0 for k in range(3)] + [1.0, 1.0]
                        p2 = [p0[k] + (p3[k] - p0[k]) * 2.0 / 3.0 for k in range(3)] + [1.0, 1.0]
                        raw_keypoints[i].append(p1)
                        raw_keypoints[i+1].append(p2)
                    status[i] = 1
                    status[i+1] = 1
                    interpolated_phases.extend([f"phase_{i+1}", f"phase_{i+2}"])
                else:
                    log_rejection(session_name, "HARD_REJECT_INTERP", f"Frames {i},{i+1} failed, but boundary frames missing")
                    return False, []
            elif status[i] == 0: # 1-frame gap
                if status[i-1] == 1 and status[i+1] == 1:
                    raw_keypoints[i] = []
                    for j in range(len(raw_keypoints[i-1])):
                        p0 = raw_keypoints[i-1][j]
                        p2 = raw_keypoints[i+1][j]
                        p1 = [(p0[k] + p2[k]) / 2.0 for k in range(3)] + [1.0, 1.0]
                        raw_keypoints[i].append(p1)
                    status[i] = 1
                    interpolated_phases.append(f"phase_{i+1}")
                else:
                    log_rejection(session_name, "HARD_REJECT_INTERP", f"Frame {i} failed, boundary missing")
                    return False, []

    return True, interpolated_phases


def extract_keypoints_in_memory(video_path, timestamps, batsman_name, angle, shot_index, macro_start):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    def _parse_time(t):
        try:
            if isinstance(t, list): return float(t[0])
            if isinstance(t, dict): return 0.0
            return float(t) if t is not None else 0.0
        except:
            return 0.0
        
    start_time = _parse_time(timestamps.get('start_time', 0))
    end_time = _parse_time(timestamps.get('end_time', 0))
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    window_frames = end_frame - start_frame
    
    if window_frames <= 0:
        return 0, []
        
    indices = [int(start_frame + (window_frames * i / (N_FRAMES - 1))) for i in range(N_FRAMES)]
    phase_labels = ["01_stance", "02_trigger", "03_backlift_start", "04_full_backlift", "05_downswing", "06_contact", "07_followthrough"]
    
    session_name = f"{batsman_name}_{angle}_{int(macro_start)}s_{shot_index:02d}"
    
    # Collect frames for shared processing
    frames_rgb = []
    for frame_num in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret: continue
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # --- Shared Feature Extraction ---
    success, raw_keypoints, interpolated_phases = extract_features_from_image_array(frames_rgb, session_name)
    if not success:
        return 0, []

    # Success! Write to CSV
    if interpolated_phases:
        print(f"[zero_storage_pipeline] {session_name}: interpolated phases {interpolated_phases}")
    interpolated_str = ";".join(interpolated_phases) if interpolated_phases else ""

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for idx in range(N_FRAMES):
            row = [session_name, phase_labels[idx]]
            for lm in raw_keypoints[idx]:
                row.extend(lm)
            row.append(interpolated_str)
            writer.writerow(row)

    # --- Labelling (in-memory, same frames — nothing touches disk) ---
    label_row = label_session_frames(session_name, frames_rgb)
    if label_row:
        save_label_row(label_row)
        print(f"   -> ✅ Label saved for {session_name}")
    else:
        print(f"   -> ⚠ No label saved for {session_name} (NVIDIA call failed) — keypoints are still valid, label it manually or re-run labelling later")

    return N_FRAMES, interpolated_phases

def process_single_row(url, batsman_name, angle, macro_start, macro_end):
    # Clean up broken files, INCLUDING yt-dlp's partial-download temp files
    # (e.g. temp_full_youtube.mp4.part, .ytdl). A stale .part file left over
    # from an interrupted previous run makes yt-dlp try to RESUME via an HTTP
    # Range request — if the remote content has since changed/expired, the
    # server rejects that with a 416 error that looks unrelated to its real
    # cause (this bit us once already).
    for base in [FULL_YT_VIDEO, AI_CHUNK_VIDEO]:
        for f in [base] + glob.glob(base + ".*"):
            if os.path.exists(f):
                os.remove(f)

    # STEP 1: Download full YouTube video
    print(f"\n📥 Downloading {url}...")

    # We DO NOT need audio for visual pose tracking!
    # By asking ONLY for 'bestvideo', we guarantee a download, save bandwidth, and bypass the need for ffmpeg merging.
    # continuedl=False: never resume a partial download — always start fresh,
    # so a leftover/corrupted .part file can't poison the next attempt.
    ydl_opts_base = {
        'format': 'bestvideo[height<=720][ext=mp4]/bestvideo[ext=mp4]/b',
        'outtmpl': FULL_YT_VIDEO, 'quiet': True, 'no_warnings': True,
        'continuedl': False,
    }
    
    # If the user exported a cookies.txt file, use it as the ultimate bypass
    if os.path.exists("cookies.txt"):
        ydl_opts_base['cookiefile'] = 'cookies.txt'
        print("   -> 🍪 Found cookies.txt! Using it to bypass YouTube bot detection...")
        
    success = False
    was_bot_detection = False

    # Try no cookies (or cookies.txt) first, then fallback to popular browsers
    for browser in [None, ('opera',), ('edge',), ('chrome',), ('firefox',), ('brave',)]:
        try:
            ydl_opts = ydl_opts_base.copy()
            if browser and not os.path.exists("cookies.txt"):
                ydl_opts['cookiesfrombrowser'] = browser
                print(f"   -> Retrying with {browser[0]} cookies to bypass bot detection...")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            success = True
            break
        except Exception as e:
            error_str = str(e)
            if browser is None:
                # If first try fails for a NON-bot reason, abort completely —
                # and remember that, so the message below tells the truth
                # instead of always blaming bot detection.
                if "Sign in to confirm" not in error_str and "bot" not in error_str.lower():
                    print(f"❌ Failed to download {url}: {e}")
                    break
                was_bot_detection = True
            else:
                # If a specific browser cookie fails (e.g. not installed, DPAPI locked), just skip to the next browser
                continue

    if not success:
        if was_bot_detection:
            print(f"🚨 YouTube Bot Detection blocked download for {url}.")
            print("   To fix this: Export a 'cookies.txt' file from your browser and save it to this folder.")
        else:
            print(f"🚨 Download failed for {url} (see the error above — not bot detection, cookies.txt won't help here).")
        return
        
    print("✅ Download complete!")

    # STEP 2: Pre-trim
    print(f"✂️ Slicing video from {macro_start}s to {macro_end}s...")
    full_video = None
    try:
        full_video = VideoFileClip(FULL_YT_VIDEO)
        actual_end = min(macro_end, full_video.duration)
        ai_chunk = full_video.subclipped(macro_start, actual_end)
        ai_chunk.write_videofile(AI_CHUNK_VIDEO, codec="libx264", audio=False, logger=None)
    except Exception as e:
        print(f"❌ Failed to cut video: {e}")
        if full_video:
            full_video.close()
        if os.path.exists(FULL_YT_VIDEO): os.remove(FULL_YT_VIDEO)
        return
        
    if full_video:
        full_video.close()
    # 🔥 IMMEDIATE DELETE OF MASSIVE YOUTUBE VIDEO 🔥
    if os.path.exists(FULL_YT_VIDEO): os.remove(FULL_YT_VIDEO)

    # STEP 3: AI shot detection — find the precise stance-to-follow-through
    # window within this macro-trimmed chunk. AI_CHUNK_VIDEO runs 0 to
    # (actual_end - macro_start) on ITS OWN timeline (it's already been sliced),
    # so the shot window returned here must be relative to that, not to the
    # original video's absolute timestamps.
    macro_length = actual_end - macro_start
    print(f"\n🧠 Finding batting shots within this {macro_length:.1f}s window...")
    ai_shots = find_shot_windows(AI_CHUNK_VIDEO, macro_length)
    if not ai_shots:
        print("🚨 No shots detected by NVIDIA vision model. Skipping this video.")
        if os.path.exists(AI_CHUNK_VIDEO): os.remove(AI_CHUNK_VIDEO)
        return
    print(f"   -> Found {len(ai_shots)} shot(s) in this window")
    for i, w in enumerate(ai_shots):
        print(f"      Shot {i+1}: {w['start_time']:.2f}s to {w['end_time']:.2f}s (within the chunk)")

    # STEP 4: In-Memory Coordinate Extraction (Zero Local Image Storage!)
    print(f"\n🎥 Extracting coordinates directly to CSV in-memory...")
    total_keypoints_saved = 0
    try:
        for index, shot in enumerate(ai_shots):
            frames_saved, _ = extract_keypoints_in_memory(AI_CHUNK_VIDEO, shot, batsman_name, angle, index+1, macro_start)
            total_keypoints_saved += frames_saved
            print(f"   -> Shot {index+1}: {frames_saved}/7 frame coordinates logged.")
    except Exception as e:
        print(f"❌ Error during memory extraction: {e}")
    finally:
        # 🔥 IMMEDIATE DELETE OF THE CHUNK VIDEO 🔥
        if os.path.exists(AI_CHUNK_VIDEO): os.remove(AI_CHUNK_VIDEO)
        
    print(f"🎉 Success! {total_keypoints_saved} rows of mathematical data safely stored in keypoints.csv! Laptop Storage Used: 0 MB.")

def run_zero_storage_pipeline():
    if not os.path.exists(CSV_FILE):
        print(f"🚨 Missing {CSV_FILE} file. Please create it first!")
        return

    print("🚀 Starting ZERO-STORAGE Batch Pipeline...")

    processed = 0
    with open(CSV_FILE, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            url = row['url'].strip()
            if not url: continue

            batsman_name = row['batsman_name'].strip()
            angle = row['angle'].strip()

            try:
                macro_start = float(row['macro_start_sec'])
                macro_end = float(row['macro_end_sec'])
            except ValueError:
                print(f"⚠ Skipping row due to invalid timestamps: {row}")
                continue

            print(f"\n{'='*50}")
            print(f"Processing Request: {batsman_name} ({angle}) | Window: {macro_start}s to {macro_end}s")
            try:
                process_single_row(url, batsman_name, angle, macro_start, macro_end)
                processed += 1
            except CreditsExhaustedError as e:
                print(f"\n🛑 STOPPING BATCH — {e}")
                print(f"   Processed {processed} video(s) successfully before running out.")
                print("   keypoints.csv / labels.csv have everything that succeeded so far — nothing lost.")
                return

            print("💤 Sleeping for 5 seconds before next video...")
            time.sleep(5)

    print(f"\n🏁 PIPELINE COMPLETE! Processed {processed} video(s). All pure coordinate data stored in keypoints.csv. Laptop Storage Used: 0 MB.")

if __name__ == "__main__":
    run_zero_storage_pipeline()
