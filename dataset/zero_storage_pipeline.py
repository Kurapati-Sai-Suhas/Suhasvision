import os
import glob
import threading
import time
import csv
import traceback
import cv2
import numpy as np
import urllib.request
import yt_dlp
from moviepy import VideoFileClip
from dotenv import load_dotenv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import zlib
from nvidia_client import find_shot_windows, find_shot_windows_auto, label_session_frames, CreditsExhaustedError


def _resolve_naming_anchor(url, macro_start):
    """
    The value used in place of a human-provided macro_start_sec for
    session/log naming when no macro window was given (auto-scan mode).
    A stable (NOT Python's per-process-randomized hash()) tag derived from
    the URL, so two different auto-scanned videos of the same batsman+angle
    never collide on session_prefix -- macro_start=0.0 for every auto-scanned
    row would otherwise make every such video's shot #1 "..._0s_01", etc.
    Shared between run_zero_storage_pipeline (idempotency prefix check) and
    process_single_row (actual naming) so the two can never compute a
    different anchor for the same row.
    """
    if macro_start is not None:
        return macro_start
    return zlib.crc32(url.encode("utf-8")) % 100000
from schema import CANONICAL_FRAME_NAMES
from subject_selection import MAX_POSE_CANDIDATES, MIN_TRACK_COVERAGE, select_subject
import contact_detection
import extraction_diagnostics as diagnostics
from extraction_config import get_config

# Every data file this module reads or writes lives next to it in dataset/,
# NOT in whatever the current working directory happens to be. Before this
# anchor existed, importing the module from another directory created
# keypoints.csv there and sent rejection logs to the wrong place (audit H10).
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def log_rejection(session_name, category, details=""):
    log_file = os.path.join(_MODULE_DIR, "pipeline_rejections.log")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {session_name} | {category} | {details}\n")

# ==========================================
# 1. INITIALIZATION & CONFIG
# ==========================================
load_dotenv(override=True)

FULL_YT_VIDEO = os.path.join(_MODULE_DIR, "temp_full_youtube.mp4")
AI_CHUNK_VIDEO = os.path.join(_MODULE_DIR, "temp_ai_chunk.mp4")
CSV_FILE = os.path.join(_MODULE_DIR, "batch_urls.csv")
OUTPUT_CSV = os.path.join(_MODULE_DIR, "keypoints.csv")
COOKIES_TXT = os.path.join(_MODULE_DIR, "cookies.txt")
N_FRAMES = 7

model_path = os.path.join(_MODULE_DIR, 'pose_landmarker_heavy.task')

# ONE PoseLandmarker for the whole process (Milestone 5, audits H5/H10):
# construction used to happen once at IMPORT (every importer — including
# Django boot — paid ~0.7s for a detector most of them never used) and then
# AGAIN inside every extract_features_from_image_array call, which built and
# destroyed a fresh detector per session/request. The singleton below is
# created lazily on first use and lives until process exit (or an explicit
# reset_shared_detector()). _DETECTOR_LOCK guards BOTH creation and use:
# MediaPipe landmarkers are not documented thread-safe, so concurrent Django
# requests serialize their detection passes on it — safe by construction,
# and at this scale they would contend on CPU anyway.
_shared_detector = None
_DETECTOR_LOCK = threading.Lock()


def _create_pose_detector():
    if not os.path.exists(model_path):
        print("Downloading MediaPipe Pose model...")
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
        urllib.request.urlretrieve(url, model_path)
    base_options = python.BaseOptions(model_asset_path=model_path)
    # num_poses > 1 (Milestone 4, audit C1): the default num_poses=1 let
    # MediaPipe silently pick ONE person per frame with no guarantee it was
    # the batsman. All candidates are collected; WHO to trust is
    # subject_selection.py's explicit session-level decision.
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False,
                                           num_poses=MAX_POSE_CANDIDATES)
    return vision.PoseLandmarker.create_from_options(options)


def _get_shared_detector_locked():
    """Returns the process-wide detector, creating it on first use.
    Caller MUST hold _DETECTOR_LOCK (the same lock that serializes use)."""
    global _shared_detector
    if _shared_detector is None:
        _shared_detector = _create_pose_detector()
    return _shared_detector


def reset_shared_detector():
    """Closes and clears the shared detector. For tests (isolation between
    fake and real detectors) and embedders that need an explicit teardown —
    normal operation just lets the singleton live until process exit."""
    global _shared_detector
    with _DETECTOR_LOCK:
        if _shared_detector is not None:
            _shared_detector.close()
            _shared_detector = None

def _expected_keypoints_header():
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
    return header


def ensure_output_csv_ready(output_csv=None):
    """
    Creates keypoints.csv with the expected header if it doesn't exist; if it
    does, verifies its header matches what this version of the code writes. A
    schema change (e.g. adding the interpolated_frames column) landing on a
    pre-existing file silently produces rows longer than the header, which
    corrupts the file for anyone reading it with pandas later — fail loudly
    instead. Raises RuntimeError on mismatch.

    Called at batch start (run_zero_storage_pipeline) and before each append
    (extract_keypoints_in_memory), NOT at import time: this check used to run
    as a module-import side effect, which created/validated keypoints.csv in
    the importer's CURRENT WORKING DIRECTORY — planting stray files (or an
    unrelated-looking RuntimeError) wherever the module was imported from
    (audit H10).
    """
    path = output_csv or OUTPUT_CSV
    expected_header = _expected_keypoints_header()
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(expected_header)
        return
    with open(path, encoding="utf-8") as f:
        existing_header = next(csv.reader(f), [])
    if existing_header != expected_header:
        raise RuntimeError(
            f"{path}'s header ({len(existing_header)} cols) doesn't match what this "
            f"script writes ({len(expected_header)} cols). Fix the header (or the schema "
            "change) before running — appending now would silently corrupt the file."
        )


LABELS_CSV = os.path.join(_MODULE_DIR, "labels.csv")
# bowling_type ("fast", "spin", or "unknown") is per-video metadata from
# batch_urls.csv, not something the AI shot-labelling call infers -- a
# batting-side camera angle usually doesn't show the bowler clearly enough
# to classify pace vs. spin reliably, so this is set by whoever adds the
# video, the same way angle/batsman_name already are.
LABELS_HEADER = ["session_name", "shot_type", "batting_hand", "bowling_type", "skill_level", "strength",
                 "weakness", "change_drill", "score_balance", "score_power", "score_technique", "score_defence"]


def save_label_row(label_row):
    """Appends one row (from nvidia_client.label_session_frames) to labels.csv."""
    file_exists = os.path.exists(LABELS_CSV)
    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(LABELS_HEADER)
        writer.writerow([label_row.get(col, "") for col in LABELS_HEADER])


def _detect_candidates(detector, frames_rgb):
    """Pass 1 — collect every candidate person per frame. A None frame slot
    (undecodable frame, see collect_phase_frames) contributes an empty
    candidate list, same semantics as "nobody detected"."""
    candidates_per_frame = []
    for frame_rgb in frames_rgb:
        if frame_rgb is None:
            candidates_per_frame.append([])
            continue
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)
        candidates_per_frame.append(list(detection_result.pose_landmarks or []))
    return candidates_per_frame


def extract_features_from_image_array(frames_rgb, session_name="unknown", detector=None, diag=None):
    """
    Core shared extraction logic (dataset ingestion AND Django serving).
    Takes a list of 7 RGB numpy arrays (None slots = undecodable frames).
    Detects up to MAX_POSE_CANDIDATES people per frame, selects ONE subject
    across the whole session (subject_selection.py, Milestone 4 — or rejects
    if no subject clearly dominates), validates the selected poses' topology
    per phase, applies the pipeline interpolation rules.

    detector: optional externally managed PoseLandmarker (tests inject
    fakes). Default (None) uses the process-wide shared singleton, with the
    detection pass serialized under _DETECTOR_LOCK — before Milestone 5,
    every call built and destroyed its own detector (~0.7s per session).

    Returns: (success_bool, list_of_raw_keypoints_or_error_string, interpolated_frame_indices)
    """
    if detector is not None:
        candidates_per_frame = _detect_candidates(detector, frames_rgb)
    else:
        with _DETECTOR_LOCK:
            candidates_per_frame = _detect_candidates(_get_shared_detector_locked(), frames_rgb)

    # Pass 2 — ONE session-level subject decision (subject_selection.py):
    # temporal track association, persistence qualification, and a
    # dominance requirement. No confident subject -> reject the session,
    # never silently score whoever the detector happened to find.
    selected, report = select_subject(candidates_per_frame)
    if diag is not None:
        # The FINE pass, recorded separately from the coarse one. If the two
        # passes disagree about who the batsman is, that disagreement is the
        # identity-switch signal -- invisible unless both are stored.
        diagnostics.record_subject_selection(diag, report, pass_name="fine")
    if selected is None:
        log_rejection(session_name, "SUBJECT_SELECTION_REJECTED", report["reason"])
        return False, f"No single trackable subject across the shot ({report['reason']}).", None
    if report["multi_track"]:
        # Logged whenever more than one track existed, so real runs build a
        # spot-checkable record of every multi-person decision — absence of
        # exactly this kind of positive logging is how the original
        # wrong-person defect went unnoticed (architecture_ground_truth.md).
        log_rejection(session_name, "SUBJECT_SELECTED",
                      f"{report['n_tracks']} candidate track(s), "
                      f"{report.get('traversing', 0)} dropped as traversing (bowler filter), winner covers "
                      f"{report['winner_coverage']}/{len(frames_rgb)} frames, mean torso {report['winner_scale']}, "
                      f"net travel {report.get('winner_displacement')} torsos")

    # Pass 3 — unchanged per-phase topology validation + conversion, now on
    # the SELECTED subject only.
    raw_keypoints = []
    for i, landmarks in enumerate(selected):
        if landmarks is None:
            raw_keypoints.append(None)
            continue
        frame_name = f"frame_0{i+1}"
        if validate_pose(landmarks, i, session_name, frame_name):
            kp_list = [[lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.0), getattr(lm, 'presence', 0.0)] for lm in landmarks]
            raw_keypoints.append(kp_list)
        else:
            raw_keypoints.append(None)

    if diag is not None:
        n_ok = sum(1 for kp in raw_keypoints if kp is not None)
        diag.pose_success_count = n_ok
        diag.pose_success_rate = round(n_ok / max(len(raw_keypoints), 1), 3)

    success, interpolated = apply_pipeline_rules(raw_keypoints, session_name)
    if not success:
        return False, "Pose rejected by kinematic validation rules.", None

    if diag is not None:
        diag.interpolated_frames = list(interpolated or [])

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
    # These rules hardcode the 7-phase session shape (the status[i+2] scans,
    # edge indices 4/5/6). A shorter list used to fall through to IndexError
    # (audit C3) when a caller silently dropped a failed frame read.
    if len(raw_keypoints) != N_FRAMES:
        log_rejection(session_name, "HARD_REJECT_FRAME_COUNT",
                      f"expected {N_FRAMES} phase slots, got {len(raw_keypoints)}")
        return False, []

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


# Milestone 4 (Wrist-Speed-Guided Adaptive Sampling): coarse pre-scan
# budget for locating the wrist-speed peak within a shot window. A
# reasonable starting point (dense enough to localize contact within a few
# raw frames on a typical ~1-3s shot window), not a validated constant --
# tune alongside MAX_BONE_CV / MIN_TEMPORAL_CONFIDENCE once more real
# videos have been run through this.
WRIST_SPEED_COARSE_SAMPLES = 20

# Disabled by default (2026-07-18 review). MediaPipe's PoseLandmarker here
# runs with the default num_poses=1 -- it detects ONE person per frame with
# no guarantee it's the batsman. Confirmed with real footage + landmark
# overlay (not assumed): in a nets-practice clip, the coarse scan locked
# onto the FEEDER/coach walking toward the crease with the ball, not the
# batsman -- the "wrist speed peak" it found was the feeder's arm swinging
# while walking. A candidate mitigation (reject speed spikes where the hip
# barely moved, on the theory that a real swing is wrist-dominant while a
# full person-swap would move everything) was tested directly against that
# same failure case and DISALLOWED: a person casually swinging an arm while
# walking produces the identical wrist-fast/hip-stable signature a real
# swing does (ratio 34.43 in the confirmed failure case), so it can't tell
# the two apart. No validated fix exists yet. Rather than ship a mechanism
# proven to silently corrupt phase sampling in exactly the kind of footage
# this pipeline processes, it defaults OFF -- extract_keypoints_in_memory
# always uses the original, previously-proven uniform formula until this
# is either fixed (e.g. multi-person detection + a subject-selection
# heuristic) and re-validated, or a source of shot windows guaranteed
# single-person-in-frame is confirmed. The functions below are unchanged
# and still correct/tested -- only their use in production is gated.
#
# Milestone 4 note: extract_features_from_image_array now does multi-person
# subject selection (subject_selection.py), but THIS coarse scan still uses
# the module-level single-pose detector and takes pose_landmarks[0] -- one
# more reason this flag stays False until the scan itself is rebuilt on top
# of the selection machinery and re-validated against real footage.
WRIST_SPEED_SAMPLING_ENABLED = False

# MediaPipe Pose landmark indices (same convention documented in
# validate_pose() above and in kinematic_validator.py's LANDMARKS list).
_LEFT_WRIST_IDX = 15
_RIGHT_WRIST_IDX = 16


def motion_energy_phase_indices(frames_gray, start_frame, end_frame, n_phases=N_FRAMES):
    """
    Non-uniform phase sampling by CUMULATIVE MOTION ENERGY, following
    MGSampler (Zhi, Tong, Wang & Wu, "MGSampler: An Explainable Sampling
    Strategy for Video Action Recognition", ICCV 2021).

    WHY: uniform sampling assumes the swing progresses at constant speed. It
    does not -- stance/trigger is a near-static setup while
    backlift->contact is explosive -- so uniform sampling systematically
    lands "contact" on a transitional frame and over-samples the idle setup.

    HOW: build the motion-energy profile over the window (mean absolute
    difference between consecutive sampled frames), take its cumulative
    distribution, and place the 7 phases at EQUAL INTERVALS OF CUMULATIVE
    MOTION rather than equal intervals of time. Fast phases therefore receive
    proportionally more samples automatically.

    WHY THIS SUCCEEDS WHERE THE WRIST-SPEED SAMPLER FAILED (and stays
    disabled, see WRIST_SPEED_SAMPLING_ENABLED): that approach depended on
    argmax of a single landmark's speed, so ONE wrong detection -- a feeder
    walking through frame -- silently relocated "contact" to the wrong
    person's arm swing. A cumulative distribution over whole-frame energy has
    no single point of catastrophic failure: a small moving figure elsewhere
    in frame perturbs the distribution slightly instead of capturing it, and
    the phases stay monotonically ordered by construction.

    Pure arithmetic on an already-decoded grayscale frame list -- no MediaPipe
    call, no API call, no second decode pass. Falls back to the original
    uniform formula (returns None) whenever the profile is degenerate (no
    motion at all, or too few readable frames), so behaviour degrades to
    today's proven sampling rather than to something unvalidated.
    """
    valid = [(i, f) for i, f in enumerate(frames_gray) if f is not None]
    if len(valid) < 3:
        return None

    diffs = []
    for (i_prev, f_prev), (i_cur, f_cur) in zip(valid, valid[1:]):
        if f_prev.shape != f_cur.shape:
            return None
        diffs.append((i_cur, float(np.mean(np.abs(f_cur.astype(np.int16) - f_prev.astype(np.int16))))))

    total = sum(d for _, d in diffs)
    if total <= 1e-6:
        return None  # completely static window -- uniform is as good as anything

    # Cumulative motion at each sampled position, normalised to [0, 1].
    cum, running = {valid[0][0]: 0.0}, 0.0
    for idx, d in diffs:
        running += d
        cum[idx] = running / total
    positions = sorted(cum)

    window_frames = end_frame - start_frame
    targets = [i / (n_phases - 1) for i in range(n_phases)]
    indices = []
    for t in targets:
        # First sampled position whose cumulative motion reaches this target.
        chosen = next((p for p in positions if cum[p] >= t - 1e-9), positions[-1])
        frac = chosen / (len(frames_gray) - 1) if len(frames_gray) > 1 else 0.0
        indices.append(int(start_frame + window_frames * frac))

    # Monotonic and in-bounds by construction, but assert cheaply rather than
    # trust it -- a non-monotonic phase list would mislabel every phase.
    indices = sorted(min(max(i, start_frame), end_frame) for i in indices)
    return indices


def _peak_speed_frame(wrist_samples):
    """
    Pure helper, no video/MediaPipe dependency -- unit-testable in
    isolation. Given a temporally-ordered list of either None (missing
    detection) or (frame_num, left_wrist_xyz, right_wrist_xyz) tuples,
    returns the frame_num of the LATER frame in the fastest consecutive
    pair (the max of left/right wrist displacement between them) -- the
    frame the wrist arrives at fastest, a proxy for the moment of contact.
    Returns None if fewer than one valid consecutive pair exists (all
    detections missing, or only one sample succeeded).
    """
    best_speed = -1.0
    best_frame = None
    for a, b in zip(wrist_samples, wrist_samples[1:]):
        if a is None or b is None:
            continue
        _, a_left, a_right = a
        b_frame, b_left, b_right = b
        speed = max(float(np.linalg.norm(b_left - a_left)), float(np.linalg.norm(b_right - a_right)))
        if speed > best_speed:
            best_speed = speed
            best_frame = b_frame
    return best_frame


def find_wrist_speed_peak_frame(cap, start_frame, end_frame, fps):
    """
    Coarse, bounded pre-scan of [start_frame, end_frame] to locate the raw
    frame where wrist speed peaks -- a proxy for the moment of bat-ball
    contact, used to anchor non-uniform phase sampling (see
    redistribute_phase_indices). Uses the process-wide shared detector
    (Milestone 5) rather than opening its own PoseLandmarker. NOTE: still
    takes pose_landmarks[0] with no subject selection -- one of the reasons
    WRIST_SPEED_SAMPLING_ENABLED stays False (see that flag's comment).

    Returns the coarse-scanned frame NUMBER (absolute, same space as
    start_frame/end_frame) at peak wrist speed, or None if too few frames
    yielded a usable wrist detection to compute a meaningful peak (window
    too short, video segment unreadable, or wrists not visible) -- callers
    must fall back to uniform sampling in that case.
    """
    window_frames = end_frame - start_frame
    n_coarse = min(WRIST_SPEED_COARSE_SAMPLES, window_frames + 1)
    if n_coarse < 2:
        return None

    coarse_frames = [int(start_frame + (window_frames * i / (n_coarse - 1))) for i in range(n_coarse)]

    wrist_samples = []
    with _DETECTOR_LOCK:
        detector = _get_shared_detector_locked()
        for frame_num in coarse_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                wrist_samples.append(None)
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = detector.detect(mp_image)
            if not detection_result.pose_landmarks:
                wrist_samples.append(None)
                continue
            landmarks = detection_result.pose_landmarks[0]
            left_wrist, right_wrist = landmarks[_LEFT_WRIST_IDX], landmarks[_RIGHT_WRIST_IDX]
            wrist_samples.append((
                frame_num,
                np.array([left_wrist.x, left_wrist.y, left_wrist.z]),
                np.array([right_wrist.x, right_wrist.y, right_wrist.z]),
            ))

    return _peak_speed_frame(wrist_samples)


def redistribute_phase_indices(start_frame, end_frame, peak_frame, n_phases=N_FRAMES):
    """
    Non-uniform replacement for the plain linear-interpolation index
    formula, anchored at `peak_frame` (the detected wrist-speed peak -- a
    proxy for contact). Pure arithmetic, no video/MediaPipe dependency --
    unit-testable in isolation (the same class of indexing logic that
    lived untested in Milestone 3's ablation script until review found the
    gap; extracted proactively here instead).

    Distributes the phases BEFORE contact (stance, trigger,
    backlift_start, full_backlift, downswing -- n_phases - 2 of them)
    linearly across [start_frame, peak_frame], and puts the last 2 phases
    (contact, follow-through) at peak_frame and end_frame respectively.
    Same phase COUNT, ORDER, and LABELS as the uniform formula -- only
    WHERE each phase is sampled changes.

    When peak_frame sits exactly at the fraction uniform sampling would
    already have placed contact (5/6 of the window, for the current
    7-phase schema), this produces the IDENTICAL indices the uniform
    formula would -- verified in test_zero_storage_pipeline.py. peak_frame
    is clamped into [start_frame, end_frame] defensively, since a caller
    could in principle pass a value slightly outside it.
    """
    peak_frame = max(start_frame, min(peak_frame, end_frame))
    n_before = n_phases - 2
    before = [int(start_frame + (peak_frame - start_frame) * i / n_before) for i in range(n_before)]
    after = [peak_frame, end_frame]
    return before + after


def collect_coarse_scan(cap, start_frame, end_frame, n_samples, session_name, detector=None):
    """
    Phase 1, step 1 — ONE pass over the shot window that serves BOTH
    downstream needs: multi-person detection (for subject selection) and a
    grayscale motion profile (for motion-energy sampling). Decoding once for
    two purposes rather than twice is the difference between this being
    affordable and not.

    Returns (frame_nums, candidates_per_frame, grays, n_read_failures):
      frame_nums            absolute frame numbers, evenly spanning the window
      candidates_per_frame  list of ALL detected people per sampled frame
      grays                 grayscale frames (None where the read failed),
                            positionally aligned with frame_nums
      n_read_failures       count of frames that could not be decoded

    Position is preserved on a failed read (None slot, empty candidate list)
    for the same reason collect_phase_frames does it: a silently shortened
    list shifts every later frame onto the wrong phase (audit C3).
    """
    window_frames = end_frame - start_frame
    n = max(2, min(n_samples, window_frames + 1))
    frame_nums = [int(start_frame + (window_frames * i / (n - 1))) for i in range(n)]

    grays, frames_rgb, n_failures = [], [], 0
    for frame_num in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            n_failures += 1
            grays.append(None)
            frames_rgb.append(None)
            continue
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if n_failures:
        log_rejection(session_name, "COARSE_SCAN_READ_FAILURES",
                      f"{n_failures}/{len(frame_nums)} coarse frames could not be decoded")

    if detector is not None:
        candidates = _detect_candidates(detector, frames_rgb)
    else:
        with _DETECTOR_LOCK:
            candidates = _detect_candidates(_get_shared_detector_locked(), frames_rgb)

    return frame_nums, candidates, grays, n_failures


def select_phase_frames(cap, start_frame, end_frame, session_name, config,
                        diag=None, detector=None):
    """
    Phase 1 — decide WHICH 7 frames to extract, in this order:

        coarse scan  ->  SUBJECT SELECTION  ->  contact detection
                                             ->  frame selection

    The ordering is the whole point. Before Phase 1, contact detection ran
    first and consumed `pose_landmarks[0]` — whoever the detector happened
    to return — which on real nets footage was the ball feeder. Here contact
    detection can only ever see the landmarks of a track that subject
    selection already confirmed, so that failure mode is structurally
    impossible rather than merely unlikely.

    Falls back, in order: contact-anchored -> motion-energy -> uniform. Every
    fallback is logged and recorded, never silent. Returns (indices, method).
    """
    window_frames = end_frame - start_frame
    uniform = [int(start_frame + (window_frames * i / (N_FRAMES - 1))) for i in range(N_FRAMES)]

    # A0 keeps the pre-Phase-1 path exactly: no coarse scan, no contact
    # detection, uniform sampling. Measured as the baseline, not assumed.
    if not config.use_motion_energy and not config.subject_selection_before_contact:
        return uniform, "uniform"

    frame_nums, candidates, grays, n_failures = collect_coarse_scan(
        cap, start_frame, end_frame, config.coarse_samples, session_name, detector=detector)
    if diag is not None:
        diag.frame_read_failures = n_failures

    # ---- Contact-anchored sampling (only on a CONFIRMED batsman) ----
    if config.subject_selection_before_contact:
        min_cov = max(MIN_TRACK_COVERAGE, int(config.coarse_coverage_fraction * len(frame_nums)))
        selected, report = select_subject(candidates, min_coverage=min_cov, return_track=True)
        if diag is not None:
            diagnostics.record_subject_selection(diag, report, pass_name="coarse")

        if selected is not None:
            track = report["winner_track"]
            # Landmarks of the CONFIRMED subject only. Any frame where that
            # track has no detection becomes None, so wrist_speed_series
            # skips the pair instead of measuring displacement across a gap.
            samples = [
                (frame_nums[i], track["poses"][i]) if i in track["poses"] else None
                for i in range(len(frame_nums))
            ]
            event = contact_detection.detect_contact(samples, start_frame, end_frame)
            if diag is not None:
                diagnostics.record_contact(diag, event)

            accept = event["valid"] if config.require_contact_quality else (event["frame_index"] is not None)
            if accept:
                indices = redistribute_phase_indices(
                    start_frame, end_frame, event["frame_index"], N_FRAMES)
                # Anchoring near a window edge collapses the pre-contact
                # phases onto the same frame. redistribute_phase_indices
                # documents that clustering as non-fatal, and in isolation it
                # is -- but THIS caller needs N_FRAMES *distinct* frames: a
                # repeated index means two phases receive identical
                # landmarks, so every velocity feature between them is
                # exactly zero and the model is fed a stillness that never
                # happened. Found by the Phase-1 benchmark (duplicate frames
                # on 16/102 clips under A2); fall back rather than emit it.
                if len(set(indices)) < N_FRAMES:
                    log_rejection(session_name, "CONTACT_ANCHOR_DEGENERATE",
                                  f"contact_frame={event['frame_index']} at fraction "
                                  f"{event['peak_fraction']} collapses phases onto "
                                  f"{len(set(indices))} distinct frames; falling back")
                else:
                    log_rejection(session_name, "CONTACT_ANCHORED_SAMPLING",
                                  f"contact_frame={event['frame_index']}, "
                                  f"fraction={event['peak_fraction']}, "
                                  f"confidence={event['confidence']}, "
                                  f"prominence={event['peak_prominence']}, "
                                  f"bilateral={event['bilateral_agreement']}")
                    return indices, "contact_anchored"
            else:
                log_rejection(session_name, "CONTACT_REJECTED", event["reason"])
        else:
            log_rejection(session_name, "COARSE_SUBJECT_REJECTED", report.get("reason", ""))

    # ---- Motion-energy sampling (MGSampler-style cumulative motion) ----
    if config.use_motion_energy:
        indices = motion_energy_phase_indices(grays, start_frame, end_frame, N_FRAMES)
        if indices is not None and len(set(indices)) == N_FRAMES:
            return indices, "motion_energy"
        if indices is None:
            log_rejection(session_name, "MOTION_ENERGY_UNAVAILABLE",
                          "degenerate motion profile; falling back to uniform sampling")
        else:
            # Motion energy is monotonic by construction but not DISTINCT:
            # when nearly all motion falls in one interval, several
            # cumulative targets resolve to the same sampled position. Same
            # consequence as the contact-anchor collapse above -- repeated
            # phases mean fabricated zero-velocity features.
            log_rejection(session_name, "MOTION_ENERGY_DEGENERATE",
                          f"motion profile collapsed onto {len(set(indices))} distinct "
                          f"frames; falling back to uniform sampling")

    # Uniform is the floor: there is nothing better to fall back to. If even
    # it cannot supply N_FRAMES distinct frames the window is simply too
    # short, which is recorded rather than silently passed downstream.
    if diag is not None and len(set(uniform)) < N_FRAMES:
        diag.fallback_reason = (f"window of {window_frames} frames cannot supply "
                                f"{N_FRAMES} distinct phases")
    return uniform, "uniform"


def _parse_time(t):
    """
    Coerces a shot-window start/end value from the AI detection payload into
    a float second count. The payload shape isn't fully trusted (historically
    a hallucinating model returned lists/dicts here), so malformed values
    degrade to 0.0. Narrow exception types on purpose: this used to be a bare
    `except:` that swallowed everything, including KeyboardInterrupt
    (audit M6). Module-level (not nested) so it's unit-testable.
    """
    try:
        if isinstance(t, list):
            return float(t[0])
        if isinstance(t, dict):
            return 0.0
        return float(t) if t is not None else 0.0
    except (TypeError, ValueError, IndexError):
        return 0.0


def collect_phase_frames(cap, indices, session_name):
    """
    Reads the frame at each index in `indices`, preserving POSITION: a failed
    decode yields a None slot rather than silently shrinking the list. Before
    this existed, a dropped read shifted every later frame onto the wrong
    phase label -- validated against the wrong phase's rules and written to
    keypoints.csv under the wrong name -- and could IndexError inside
    apply_pipeline_rules (audit C3). A None slot instead flows through
    extract_features_from_image_array as a missing detection, which the
    existing interpolation rules already know how to handle or reject.

    Public (no underscore) since Milestone 3: the Django serving path
    (backend ml_service.run_advanced_inference) reuses this same collection
    logic so serving and dataset ingestion read frames identically (audit C2).
    """
    frames_rgb = []
    for frame_num in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            log_rejection(session_name, "FRAME_READ_FAILED",
                          f"frame {frame_num} could not be decoded; phase slot kept as missing")
            frames_rgb.append(None)
            continue
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames_rgb


def extract_keypoints_in_memory(video_path, timestamps, batsman_name, angle, shot_index, macro_start, bowling_type="unknown", config=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_time = _parse_time(timestamps.get('start_time', 0))
    end_time = _parse_time(timestamps.get('end_time', 0))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    window_frames = end_frame - start_frame

    if window_frames <= 0:
        cap.release()
        return 0, []

    session_name = f"{batsman_name}_{angle}_{int(macro_start)}s_{shot_index:02d}"

    config = config or get_config()
    diag = diagnostics.ExtractionDiagnostics(
        session_name=session_name, video_id=os.path.basename(video_path),
        config_name=config.name, shot_start_frame=start_frame,
        shot_end_frame=end_frame, shot_duration_frames=window_frames, fps=fps,
    )

    # PHASE 1 REORDERING (see select_phase_frames): subject selection now
    # runs BEFORE contact detection, so the contact detector can only ever
    # see a confirmed batsman's landmarks. The pre-Phase-1 path ran
    # find_wrist_speed_peak_frame first on pose_landmarks[0] -- which is how
    # it locked onto the ball feeder -- and was disabled for that reason.
    # That function is retained (still tested) but is no longer on this path.
    indices, method = select_phase_frames(
        cap, start_frame, end_frame, session_name, config, diag=diag)
    diagnostics.record_frames(diag, indices, method)
    diag.fallback_used = (method == "uniform" and config.name != "A0")
    if diag.fallback_used:
        diag.fallback_reason = "contact and motion-energy sampling both unavailable"

    # Canonical phase names come from schema.py (audit H6) — this list used to
    # be one of several independent copies across the repo.
    phase_labels = CANONICAL_FRAME_NAMES

    # Collect frames for shared processing (position-preserving: a failed
    # read becomes a None slot, never a silent shift -- audit C3)
    frames_rgb = collect_phase_frames(cap, indices, session_name)
    cap.release()

    # --- Shared Feature Extraction ---
    success, raw_keypoints, interpolated_phases = extract_features_from_image_array(
        frames_rgb, session_name, diag=diag)
    if not success:
        diag.accepted = False
        diag.rejection_reason = str(raw_keypoints)
        diagnostics.write(diag)
        return 0, []

    # Belt-and-braces before the CSV write below indexes raw_keypoints[0..6]:
    # every path above should guarantee 7 slots, but a mismatch here would
    # corrupt keypoints.csv, so reject loudly instead of trusting it.
    if len(raw_keypoints) != N_FRAMES:
        log_rejection(session_name, "FRAME_COUNT_MISMATCH",
                      f"expected {N_FRAMES} keypoint rows after pipeline rules, got {len(raw_keypoints)}")
        diag.accepted = False
        diag.rejection_reason = f"frame count mismatch: {len(raw_keypoints)} != {N_FRAMES}"
        diagnostics.write(diag)
        return 0, []

    # Accepted: the 7 frames exist, belong to one selected subject, and
    # passed validation. Written before labelling so a NVIDIA outage cannot
    # cost us the extraction record.
    diag.accepted = True
    diagnostics.write(diag)

    # Success! Write to CSV
    if interpolated_phases:
        print(f"[zero_storage_pipeline] {session_name}: interpolated phases {interpolated_phases}")
    interpolated_str = ";".join(interpolated_phases) if interpolated_phases else ""

    # The header used to be guaranteed by an import-time side effect; now that
    # that's gone (audit H10), guarantee it at the write itself so a direct
    # caller can never append rows to a header-less or schema-mismatched file.
    # Costs one first-line read per shot -- noise next to the MediaPipe work.
    ensure_output_csv_ready()

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for idx in range(N_FRAMES):
            row = [session_name, phase_labels[idx]]
            for lm in raw_keypoints[idx]:
                row.extend(lm)
            row.append(interpolated_str)
            writer.writerow(row)

    # --- Labelling (in-memory, same frames — nothing touches disk) ---
    # A None slot (undecodable frame, see _collect_phase_frames) has keypoints
    # reconstructed by the interpolation rules above, but no PIXELS -- there is
    # nothing to send the labelling model for that phase, and its prompt
    # hardcodes a 7-frame phase mapping, so sending fewer would mislabel every
    # later phase. Skip labelling entirely for such sessions -- the same
    # graceful degradation as the existing NVIDIA-failure path below.
    if any(f is None for f in frames_rgb):
        log_rejection(session_name, "LABELLING_SKIPPED_MISSING_FRAME",
                      "keypoints saved (interpolated), but a frame had no decodable pixels to label")
        print(f"   -> ⚠ No label saved for {session_name} (a frame could not be decoded) — keypoints are still valid, label it manually or re-run labelling later")
        return N_FRAMES, interpolated_phases

    label_row = label_session_frames(session_name, frames_rgb)
    if label_row:
        # bowling_type isn't something the AI labelling call infers (the
        # batting-side camera angle doesn't reliably show the bowler) --
        # it's per-video metadata from batch_urls.csv, attached here.
        label_row["bowling_type"] = bowling_type
        save_label_row(label_row)
        print(f"   -> ✅ Label saved for {session_name}")
    else:
        print(f"   -> ⚠ No label saved for {session_name} (NVIDIA call failed) — keypoints are still valid, label it manually or re-run labelling later")

    return N_FRAMES, interpolated_phases

def process_single_row(url, batsman_name, angle, macro_start=None, macro_end=None, bowling_type="unknown"):
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
    
    # If the user exported a cookies.txt file (into dataset/, next to this
    # script -- module-anchored like every other data path, audit H10), use
    # it as the ultimate bypass
    if os.path.exists(COOKIES_TXT):
        ydl_opts_base['cookiefile'] = COOKIES_TXT
        print("   -> 🍪 Found cookies.txt! Using it to bypass YouTube bot detection...")
        
    success = False
    was_bot_detection = False

    # Try no cookies (or cookies.txt) first, then fallback to popular browsers
    for browser in [None, ('opera',), ('edge',), ('chrome',), ('firefox',), ('brave',)]:
        try:
            ydl_opts = ydl_opts_base.copy()
            if browser and not os.path.exists(COOKIES_TXT):
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
            print(f"   To fix this: Export a 'cookies.txt' file from your browser and save it as {COOKIES_TXT}")
        else:
            print(f"🚨 Download failed for {url} (see the error above — not bot detection, cookies.txt won't help here).")
        return
        
    print("✅ Download complete!")

    # STEP 2: Pre-trim to the human-provided macro window -- OR, if none was
    # given, scan the whole downloaded video automatically (no manual
    # macro_start_sec/macro_end_sec curation step required). This is the
    # fix for the "a person has to watch every video first" bottleneck --
    # distinct from, and unrelated to, the wrong-person-tracking defect
    # (already fixed separately by subject_selection.py). See
    # nvidia_client.find_shot_windows_auto's docstring for the tiling design.
    auto_scan = macro_start is None or macro_end is None
    naming_anchor = _resolve_naming_anchor(url, macro_start)  # only used for naming, never for timing math
    full_video = None
    if auto_scan:
        print("🔎 No macro window provided -- scanning the FULL downloaded video for batting shots...")
        try:
            full_video = VideoFileClip(FULL_YT_VIDEO)
            macro_start, macro_end = 0.0, full_video.duration
        except Exception as e:
            print(f"❌ Failed to read downloaded video: {e}")
            if os.path.exists(FULL_YT_VIDEO): os.remove(FULL_YT_VIDEO)
            return
        finally:
            if full_video:
                full_video.close()
        chunk_video_path = FULL_YT_VIDEO
        # NOT deleted here -- STEP 4's finally block deletes chunk_video_path
        # once shot detection + extraction are both done with it.
    else:
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
        macro_end = actual_end
        chunk_video_path = AI_CHUNK_VIDEO
        # 🔥 IMMEDIATE DELETE OF MASSIVE YOUTUBE VIDEO 🔥
        if os.path.exists(FULL_YT_VIDEO): os.remove(FULL_YT_VIDEO)

    # STEP 3: AI shot detection — find every precise stance-to-follow-through
    # window within this (macro-trimmed, or, in auto mode, full) video.
    # chunk_video_path runs 0 to macro_length on ITS OWN timeline in both
    # modes (a fresh 0-based file in manual mode; find_shot_windows_auto's
    # own start_offset tiling keeps auto-mode results consistent with that
    # same convention), so the shot windows returned are always relative to
    # chunk_video_path, not the original video's absolute timestamps.
    macro_length = macro_end - macro_start
    print(f"\n🧠 Finding batting shots within this {macro_length:.1f}s {'video (auto-scanned)' if auto_scan else 'window'}...")
    if auto_scan:
        ai_shots = find_shot_windows_auto(chunk_video_path, macro_length)
    else:
        ai_shots = find_shot_windows(chunk_video_path, macro_length)
    if not ai_shots:
        print("🚨 No shots detected by NVIDIA vision model. Skipping this video.")
        if os.path.exists(chunk_video_path): os.remove(chunk_video_path)
        return
    print(f"   -> Found {len(ai_shots)} shot(s) in this {'video' if auto_scan else 'window'}")
    for i, w in enumerate(ai_shots):
        print(f"      Shot {i+1}: {w['start_time']:.2f}s to {w['end_time']:.2f}s")

    # STEP 4: In-Memory Coordinate Extraction (Zero Local Image Storage!)
    print(f"\n🎥 Extracting coordinates directly to CSV in-memory...")
    total_keypoints_saved = 0
    try:
        for index, shot in enumerate(ai_shots):
            frames_saved, _ = extract_keypoints_in_memory(chunk_video_path, shot, batsman_name, angle, index+1, naming_anchor, bowling_type)
            total_keypoints_saved += frames_saved
            print(f"   -> Shot {index+1}: {frames_saved}/7 frame coordinates logged.")
    except Exception as e:
        # Keep the batch alive for the next video, but never again reduce the
        # real failure to a one-line print (that's how C3's IndexError went
        # undiagnosed -- audit M6). The full traceback goes to the rejection
        # log, flattened to one line because analyze_rejections.py counts
        # entries line-by-line on " | " and multi-line entries would show up
        # as junk lines there.
        tb_one_line = " || ".join(traceback.format_exc().splitlines())
        log_rejection(f"{batsman_name}_{angle}_{int(naming_anchor)}s", "EXTRACTION_ERROR", tb_one_line)
        print(f"❌ Error during memory extraction: {e} (full traceback in pipeline_rejections.log)")
    finally:
        # 🔥 IMMEDIATE DELETE OF THE CHUNK VIDEO (or, in auto mode, the full
        # downloaded video, kept alive until now for shot detection) 🔥
        if os.path.exists(chunk_video_path): os.remove(chunk_video_path)

    print(f"🎉 Success! {total_keypoints_saved} rows of mathematical data safely stored in keypoints.csv! Laptop Storage Used: 0 MB.")

def load_ingested_sessions(output_csv=None):
    """
    Returns the set of session_name values already present in keypoints.csv.
    Loaded ONCE at batch start (audit H1) so a re-run can skip rows whose
    sessions were already ingested instead of appending duplicate copies —
    session names are deterministic, so re-processing a row always collides.
    """
    path = output_csv or OUTPUT_CSV
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        return {row[0] for row in reader if row}


def sessions_matching_prefix(session_prefix, existing_sessions):
    """
    Sorted list of already-ingested session names belonging to one batch row.
    A row's sessions are all named f"{batsman}_{angle}_{int(macro_start)}s_NN",
    so the row-level identity is everything before the shot index. The
    trailing "s_" in the prefix is what makes prefix matching collision-safe:
    "..._10s_" can never accidentally match "..._105s_01" because the
    character after "10" differs ("s" vs "5").
    """
    return sorted(s for s in existing_sessions if s.startswith(session_prefix))


def run_zero_storage_pipeline(force=False):
    if not os.path.exists(CSV_FILE):
        print(f"🚨 Missing {CSV_FILE} file. Please create it first!")
        return

    # Fail loudly BEFORE any download/API work if keypoints.csv has a
    # mismatched schema (this check used to run at import time — audit H10).
    ensure_output_csv_ready()

    # Audit H1 (idempotent ingestion): re-running a batch used to re-download,
    # re-extract, and APPEND every row's sessions again — the corruption class
    # behind the 42-duplicate-session incident (architecture_ground_truth.md).
    existing_sessions = load_ingested_sessions()
    if force:
        print("⚠ --force: re-ingesting rows even if their sessions already exist in keypoints.csv.")
        print("  This APPENDS duplicate rows — remove the old sessions afterwards or the file is corrupt.")

    print("🚀 Starting ZERO-STORAGE Batch Pipeline...")

    processed = 0
    skipped = 0
    seen_prefixes = set()
    with open(CSV_FILE, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            url = row['url'].strip()
            if not url: continue

            batsman_name = row['batsman_name'].strip()
            angle = row['angle'].strip()
            # .get(), not row['bowling_type']: batch_urls.csv rows written
            # before this column existed shouldn't hard-fail the batch.
            bowling_type = (row.get('bowling_type') or "unknown").strip() or "unknown"

            # Blank/missing macro_start_sec or macro_end_sec means "no human
            # curated a macro window for this row" -- rather than an error,
            # this now triggers automatic full-video shot scanning
            # (nvidia_client.find_shot_windows_auto) instead of skipping the
            # row. A non-blank value that fails to parse as a float is still
            # treated as a genuine data-entry mistake and skipped, same as
            # before.
            raw_start = (row.get('macro_start_sec') or '').strip()
            raw_end = (row.get('macro_end_sec') or '').strip()
            if not raw_start or not raw_end:
                macro_start, macro_end = None, None
            else:
                try:
                    macro_start = float(raw_start)
                    macro_end = float(raw_end)
                except ValueError:
                    print(f"⚠ Skipping row due to invalid timestamps: {row}")
                    continue

            naming_anchor = _resolve_naming_anchor(url, macro_start)
            session_prefix = f"{batsman_name}_{angle}_{int(naming_anchor)}s_"

            # The same row listed twice in one batch file is never processed
            # twice — even under --force, which means "re-ingest despite
            # history", not "ingest twice in one run".
            if session_prefix in seen_prefixes:
                print(f"⏭ Skipping duplicate batch row for {session_prefix}* (already handled earlier in this run).")
                log_rejection(session_prefix, "SKIPPED_DUPLICATE_BATCH_ROW",
                              "same row appears more than once in batch_urls.csv")
                skipped += 1
                continue
            seen_prefixes.add(session_prefix)

            window_desc = f"{macro_start}s-{macro_end}s" if macro_start is not None else "auto-scan whole video"
            if not force:
                already = sessions_matching_prefix(session_prefix, existing_sessions)
                if already:
                    print(f"⏭ Skipping {batsman_name} ({angle}, {window_desc}): "
                          f"already ingested as {len(already)} session(s), e.g. {already[0]}. "
                          "Re-run with --force to re-ingest.")
                    log_rejection(session_prefix, "SKIPPED_ALREADY_INGESTED",
                                  f"{len(already)} session(s) already in keypoints.csv")
                    skipped += 1
                    continue

            print(f"\n{'='*50}")
            print(f"Processing Request: {batsman_name} ({angle}, {bowling_type}) | Window: {window_desc}")
            try:
                process_single_row(url, batsman_name, angle, macro_start, macro_end, bowling_type)
                processed += 1
            except CreditsExhaustedError as e:
                print(f"\n🛑 STOPPING BATCH — {e}")
                print(f"   Processed {processed} video(s) successfully before running out.")
                print("   keypoints.csv / labels.csv have everything that succeeded so far — nothing lost.")
                return

            print("💤 Sleeping for 5 seconds before next video...")
            time.sleep(5)

    print(f"\n🏁 PIPELINE COMPLETE! Processed {processed} video(s), skipped {skipped} already-ingested/duplicate row(s). All pure coordinate data stored in keypoints.csv. Laptop Storage Used: 0 MB.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Zero-storage batch ingestion: batch_urls.csv -> keypoints.csv + labels.csv")
    ap.add_argument("--force", action="store_true",
                    help="Process rows even if their sessions already exist in keypoints.csv. "
                         "This APPENDS duplicate rows - only use it after removing the old ones "
                         "(run audit_duplicates.py to check the file afterwards).")
    args = ap.parse_args()
    run_zero_storage_pipeline(force=args.force)
