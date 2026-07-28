import os
import sys
import logging
import threading
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from django.conf import settings

logger = logging.getLogger(__name__)

# Computed once, here, rather than at each call site that needs it --
# resolve_model_path() and get_models() below both used to recompute this
# same expression independently.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_dataset_dir = os.path.join(_BASE_DIR, "dataset")
if _dataset_dir not in sys.path:
    sys.path.insert(0, _dataset_dir)
from rule_based_scorer import score_from_keypoint_df  # noqa: E402 (path must be set up first)

# Milestone 3 (train/serve consistency, audit C2/M3): the serving path now
# routes frame reading, pose validation, and gap interpolation through the
# SAME functions the dataset-ingestion pipeline uses to build training data,
# instead of a private landmark loop that skipped validation and imputed
# missing frames by copying a neighbor verbatim (fabricating zero-velocity
# features the training data never contains). Note this import is heavier
# than ml_service's previous ones — zero_storage_pipeline constructs its
# module-level MediaPipe detector at import time; Milestone 5 consolidates
# the repo's detectors into one shared instance.
from zero_storage_pipeline import collect_phase_frames, extract_features_from_image_array  # noqa: E402

# Milestone 1 (ML single-source-of-truth consolidation): TemporalAttention and
# the feature schema used to be defined independently in this file. They now
# come from dataset/model_layers.py and dataset/schema.py respectively -- the
# same modules train_advanced_model.py, inference_service.py, and
# active_learning_retrain.py use, so there is exactly one definition of each,
# not several that can silently drift apart.
from model_layers import TemporalAttention  # noqa: E402
from schema import FEATURE_BASE_NAMES, EXPECTED_FEATURES as FEATURE_COLS, SEQ_LEN  # noqa: E402

# Globals to hold models in memory. _MODELS_LOCK (audit H3): two concurrent
# first-requests used to both see None and both load the multi-hundred-MB
# model, with the loser silently discarded.
_extractor = None
_att_layer = None
_MODELS_LOCK = threading.Lock()

def resolve_model_path():
    """Single source of truth for where the pinned model file lives. Pulled
    out of get_models() so callers that only need to know the path (e.g. a
    test deciding whether to skip because the gitignored model artifact
    isn't present) don't have to reimplement this -- which would itself be
    exactly the kind of duplication this milestone exists to eliminate.
    MODEL_FILENAME is pinned explicitly rather than auto-selected (unlike
    dataset/inference_service.py's glob-highest-version approach) so a
    production deploy always knows exactly which trained model is serving
    requests."""
    model_filename = os.environ.get('CRICKET_MODEL_FILENAME', 'cricket_stance_advanced_v5.keras')
    return os.path.join(_BASE_DIR, "dataset", model_filename)


def get_models():
    """Returns (extractor, att_layer), loading them exactly once per process.

    Milestone 5 changes: (1) double-checked locking (audit H3) so concurrent
    first-requests can't both load the model; (2) no MediaPipe detector here
    anymore — since Milestone 3 pose detection happens inside the shared
    pipeline (zero_storage_pipeline), which since Milestone 5 owns the ONE
    process-wide detector, so Django no longer constructs a second, unused
    one at startup."""
    global _extractor, _att_layer
    if _extractor is not None:
        return _extractor, _att_layer

    with _MODELS_LOCK:
        if _extractor is not None:  # another thread finished while we waited
            return _extractor, _att_layer

        model_path = resolve_model_path()
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

        extractor = tf.keras.Model(inputs=model.inputs, outputs=[model.outputs[0], bilstm_output])
        # Assign only fully-built objects; the lock-free fast path above must
        # never observe partial state.
        _att_layer = att_layer
        _extractor = extractor

    return _extractor, _att_layer

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

def _keypoints_to_dataframe(raw_keypoints):
    """
    Converts extract_features_from_image_array's validated output (SEQ_LEN
    frames x 33 landmarks x [x, y, z, visibility, presence]) into the
    per-frame landmark DataFrame that _run_model_inference and the
    rule-based fallback consume. x/y/z only — neither consumer reads
    visibility. Mirrors the conversion inference_service.extract_video_tensor
    performs; consolidating the two (together with the landmark-name list
    itself) is Milestone 7's angle/landmark single-source work.
    """
    rows = []
    for kp in raw_keypoints:
        row = {}
        for j, name in enumerate(landmarks_names):
            row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"] = kp[j][0], kp[j][1], kp[j][2]
        rows.append(row)
    return pd.DataFrame(rows)

SCORE_INDEX = {"balance_score": 0, "power_score": 1, "technique_score": 2, "defence_score": 3}

FEATURE_JOINT_LABELS = {
    "angle_knee_L": "left knee flexion", "angle_knee_R": "right knee flexion",
    "angle_hip_L": "left hip rotation", "angle_hip_R": "right hip rotation",
    "angle_elbow_L": "left elbow extension", "angle_elbow_R": "right elbow extension",
    "angle_shoulder_L": "left shoulder position", "angle_shoulder_R": "right shoulder position",
    "angle_ankle_L": "left ankle stability", "angle_ankle_R": "right ankle stability",
    "angle_trunk_L": "trunk lean (left side)", "angle_trunk_R": "trunk lean (right side)",
    "angle_arm_L": "left arm alignment", "angle_arm_R": "right arm alignment",
    "angle_head_tilt": "head tilt / eye level",
}

JOINT_DRILL_MAP = {
    "angle_knee_L": "Wall-sit knee-bend holds to groove a stable front-knee flex",
    "angle_knee_R": "Split-squat reps to strengthen back-knee drive",
    "angle_hip_L": "Resistance-band hip-rotation drills for a cleaner turn",
    "angle_hip_R": "Medicine-ball rotational throws to build hip drive",
    "angle_elbow_L": "Shadow-bat top-hand extension drills through the line",
    "angle_elbow_R": "Bottom-hand throwdown drills for a straighter elbow path",
    "angle_shoulder_L": "Mirror drill on shoulder alignment at set-up",
    "angle_shoulder_R": "Resistance-band shoulder rotation drill for a fuller turn",
    "angle_ankle_L": "Single-leg balance holds on the front ankle",
    "angle_ankle_R": "Calf-raise plus balance-board work for the back ankle",
    "angle_trunk_L": "Side-plank holds to control trunk lean",
    "angle_trunk_R": "Trunk-rotation med-ball drills",
    "angle_arm_L": "Top-hand-only shadow shots to groove arm path",
    "angle_arm_R": "Bottom-hand-only shadow shots to groove arm path",
    "angle_head_tilt": "Head-still drill: track a ball on a string without moving the head",
}

_background_cache = None


def _load_background_sequences(n=32):
    """Real (7, 30) feature sequences sampled from the training data, used as
    baselines for Expected Gradients attribution. Falls back to a single
    zero-vector baseline (equivalent to plain saliency) if the training CSV
    isn't available at runtime -- attribution still runs, just without the
    baseline-averaging that gives Expected Gradients its stability."""
    global _background_cache
    if _background_cache is not None:
        return _background_cache

    csv_path = os.path.join(_dataset_dir, "dataset_angles.csv")
    if not os.path.exists(csv_path):
        _background_cache = np.zeros((1, SEQ_LEN, len(FEATURE_COLS)), dtype=np.float32)
        return _background_cache

    df = pd.read_csv(csv_path)
    sequences = []
    for _, group in df.groupby("session_name"):
        if len(group) != 7:
            continue
        group = group.sort_values("frame_name")
        try:
            sequences.append(group[FEATURE_COLS].values.astype(np.float32))
        except KeyError:
            continue
        if len(sequences) >= n * 4:
            break

    if not sequences:
        _background_cache = np.zeros((1, SEQ_LEN, len(FEATURE_COLS)), dtype=np.float32)
        return _background_cache

    pool = np.array(sequences)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    _background_cache = pool[idx]
    return _background_cache


def _expected_gradients(extractor, x_input, output_index, n_baselines=12, n_steps=6):
    """
    Expected Gradients attribution (Erion et al., 2021) -- the same
    statistical method shap.GradientExplainer implements for differentiable
    models. Computed directly via tf.GradientTape: shap's own numba
    dependency fails to import on this machine (Windows Application Control
    policy blocks numba's compiled DLL -- a system-level security policy,
    not something a pip reinstall or code change can fix), so this is the
    real attribution math without shap's wrapper package.

    For each of several real baseline sequences drawn from the training
    distribution, walk the straight-line path from baseline to the actual
    input, average the target output's gradient w.r.t. the input along that
    path, and scale by (input - baseline). Averaging that over many
    baselines approximates the same expected-value integral
    GradientExplainer computes.

    Milestone 5 (audit M8): all baseline x step interpolants run as ONE
    batched tape pass instead of n_baselines * n_steps sequential ones —
    measured ~40x faster. Same math: rows are independent through the
    network (inference-mode BatchNorm uses moving averages, no cross-row
    coupling), so the gradient of the SUMMED target w.r.t. the batched
    input is exactly the per-row gradients — equal to the sequential
    version up to float reduction order (asserted against a sequential
    reference implementation in the test suite).
    """
    backgrounds = _load_background_sequences(n_baselines)  # (B, 7, F)
    n_backgrounds = backgrounds.shape[0]
    x_input_t = tf.constant(x_input, dtype=tf.float32)      # (1, 7, F)
    baselines_t = tf.constant(backgrounds, dtype=tf.float32)
    diffs = x_input_t - baselines_t                         # (B, 7, F)

    alphas = tf.constant([step / n_steps for step in range(1, n_steps + 1)], dtype=tf.float32)
    # (B, S, 7, F): every point on every baseline's straight-line path.
    interpolated = baselines_t[:, None, :, :] + alphas[None, :, None, None] * diffs[:, None, :, :]
    flat = tf.reshape(interpolated, (n_backgrounds * n_steps,) + tuple(x_input.shape[1:]))

    with tf.GradientTape() as tape:
        tape.watch(flat)
        scores, _ = extractor(flat, training=False)
        target = tf.reduce_sum(scores[:, output_index])
    grads = tape.gradient(target, flat)                     # (B*S, 7, F)

    grads = tf.reshape(grads, (n_backgrounds, n_steps) + tuple(x_input.shape[1:]))
    avg_grads = tf.reduce_mean(grads, axis=1)               # (B, 7, F) — mean over path steps
    attributions = tf.reduce_mean(avg_grads * diffs, axis=0)  # (7, F) — mean over baselines
    return attributions.numpy()  # signed attribution per frame/feature


PHASE_LABELS = ["stance", "trigger", "backlift start", "full backlift",
                "downswing", "contact", "follow-through"]


def _top_feature_drivers(extractor, x_input, weakest_key, top_k=2, top_phases=3):
    """
    Returns (joint_labels, drill_texts, phase_label) for the features with the
    largest |attribution| toward the weakest score.

    TWO-STAGE attribution (Ismail et al., "Benchmarking Deep Learning
    Interpretability in Time Series Predictions", NeurIPS 2020). Their
    benchmark finds saliency degrades when a method must attribute across the
    time AND feature axes at once, and that the effective fix is to first find
    the salient TIMESTEPS, then rank features conditioned on those timesteps.

    This previously summed |attribution| across all 7 phases, discarding the
    temporal axis entirely -- so "left knee flexion" could win on diffuse
    noise spread over the whole stroke while a sharp, real signal confined to
    contact lost. Now the phase ranking comes first, and features are ranked
    only within the top-scoring phases. Pure post-processing of the same
    attribution tensor -- no extra forward/backward passes, no added latency.

    The returned phase_label also makes the coaching copy concrete: "your
    front knee, during the downswing" instead of just "your front knee".
    """
    attributions = _expected_gradients(extractor, x_input, SCORE_INDEX[weakest_key])  # (7, F)
    abs_attr = np.abs(attributions)

    # Stage 1 -- which phases carry the signal.
    phase_importance = np.sum(abs_attr, axis=1)  # (7,)
    n_phases = min(top_phases, abs_attr.shape[0])
    top_phase_idx = np.argsort(-phase_importance)[:n_phases]
    dominant_phase = int(np.argmax(phase_importance))

    # Stage 2 -- rank features using only those phases.
    feature_importance = np.sum(abs_attr[top_phase_idx, :], axis=0)  # (F,)

    n_base = len(FEATURE_BASE_NAMES)
    combined = feature_importance[:n_base] + feature_importance[n_base:]  # pair angle with its velocity
    ranked_idx = np.argsort(-combined)[:top_k]

    joints = [FEATURE_BASE_NAMES[i] for i in ranked_idx]
    labels = [FEATURE_JOINT_LABELS[j] for j in joints]
    drills = [JOINT_DRILL_MAP[j] for j in joints]
    phase_label = PHASE_LABELS[dominant_phase] if dominant_phase < len(PHASE_LABELS) else None
    return labels, drills, phase_label


def _attention_focus(extractor, att_layer, x_input):
    """
    The model's own temporal attention weights -- "which of the 7 phases did
    the network concentrate on for this stroke". One tensordot over an
    already-computed (1, 7, 128) BiLSTM output; sub-millisecond, no extra
    model call.

    DELIBERATE WORDING CAVEAT (Jain & Wallace, "Attention is not Explanation",
    NAACL 2019; Wiegreffe & Pinter, "Attention is not not Explanation", EMNLP
    2019; Bibal et al., ACL 2022): single-head additive attention pooling is
    exactly the architecture class shown to admit alternative attention
    distributions that preserve the prediction. So this is reported as WHERE
    THE MODEL LOOKED, never as WHY it scored what it did -- the causal claim
    stays with Expected Gradients (§_expected_gradients), which is
    axiomatically grounded. Returns None on any failure; this is a nice-to-
    have signal and must never break an inference.
    """
    if att_layer is None:
        return None
    try:
        _, bilstm_out = extractor(x_input, training=False)  # (1, 7, 128)
        e = tf.keras.activations.tanh(tf.tensordot(bilstm_out, att_layer.W, axes=1) + att_layer.b)
        alpha = tf.keras.activations.softmax(e, axis=1).numpy()[0, :, 0]  # (7,)
        focus_idx = int(np.argmax(alpha))
        return {
            "phase": PHASE_LABELS[focus_idx] if focus_idx < len(PHASE_LABELS) else str(focus_idx),
            "weight": round(float(alpha[focus_idx]), 3),
            "distribution": {PHASE_LABELS[i]: round(float(w), 3)
                             for i, w in enumerate(alpha) if i < len(PHASE_LABELS)},
        }
    except Exception:
        logger.exception("Attention focus extraction failed; omitting it from the response")
        return None


class InvalidUploadError(ValueError):
    """The upload violates the single-shot contract (unreadable, too short,
    too long, or no reliably trackable batsman). The message is user-facing;
    views.analyze_stance maps this to HTTP 400 instead of the generic 500."""


# Upload contract (Milestone 3, Option B): serving samples SEQ_LEN frames
# uniformly across the WHOLE clip, while training data is sampled across a
# tight AI-detected shot window (~1-3.5s on real footage — see
# architecture_ground_truth.md's round-4 measurements). What makes uniform
# whole-clip sampling match the training distribution is the clip ITSELF
# being the shot window: a short, pre-trimmed video of one swing. 20s allows
# generous handling slop around a real swing while rejecting nets-session
# recordings and screen captures, whose uniform samples would mostly show
# idle content the model would score with false confidence.
MAX_UPLOAD_SECONDS = 20.0

# Forward passes for the MC-Dropout uncertainty estimate (mean + std of the
# sigmoid outputs under dropout). 30 is the value the sequential loop always
# used; since Milestone 5 they run as one batched pass.
MC_DROPOUT_PASSES = 30


def run_advanced_inference(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise InvalidUploadError(
            "We couldn't read that video file. Please upload a standard MP4, MOV, or WEBM clip."
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # some containers omit fps metadata; assume a standard rate
    duration_seconds = total_frames / fps

    if total_frames < SEQ_LEN:
        cap.release()
        raise InvalidUploadError(
            "That clip is too short to contain a full shot. "
            "Please upload a clip of one complete swing, from stance to follow-through."
        )
    if duration_seconds > MAX_UPLOAD_SECONDS:
        cap.release()
        raise InvalidUploadError(
            f"That clip is about {duration_seconds:.0f} seconds long. Analysis expects one "
            f"pre-trimmed shot of at most {MAX_UPLOAD_SECONDS:.0f} seconds — please trim the "
            "video to a single swing (stance to follow-through) and upload again."
        )

    # Same frame reading, pose validation, and gap interpolation the dataset
    # pipeline uses to build training data (audit C2/M3) — including C3's
    # position-preserving read semantics. The old private path here skipped
    # validation entirely and imputed undetected frames by copying a
    # neighbor, which fabricated zero-velocity features at those steps.
    session_name = f"upload_{os.path.basename(video_path)}"
    frame_indices = np.linspace(0, total_frames - 1, SEQ_LEN, dtype=int)
    frames_rgb = collect_phase_frames(cap, frame_indices, session_name)
    cap.release()

    success, result, interpolated_phases = extract_features_from_image_array(frames_rgb, session_name)
    if not success:
        logger.info("Upload %s rejected by pose validation: %s", session_name, result)
        raise InvalidUploadError(
            "We couldn't reliably track the batsman's pose through a full shot in this clip. "
            "Please upload a clear, well-lit video of one complete swing with the batsman fully in frame."
        )
    if interpolated_phases:
        logger.info("Upload %s: interpolated phases %s", session_name, interpolated_phases)

    combined_df = _keypoints_to_dataframe(result)

    # Reliability requirement (SRS FR-ERR-001), unchanged: if the ML model or
    # inference path fails for any reason (including the model file being
    # absent — get_models is inside the try on purpose), fall back to the
    # rule-based scorer on the same extracted keypoints rather than 500.
    try:
        extractor, att_layer = get_models()
        return _run_model_inference(combined_df, extractor, att_layer)
    except Exception:
        logger.exception("ML inference failed; falling back to rule-based scorer")
        return score_from_keypoint_df(combined_df)


def _run_model_inference(combined_df, extractor, att_layer=None):
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

    # MC-Dropout as ONE batched pass over 30 tiled copies instead of 30
    # sequential calls (Milestone 5, audit M8) — measured ~19x faster.
    # Statistically identical to the loop: the rows are byte-identical, so
    # BatchNormalization's training-mode batch statistics are unchanged
    # (mean/var over 30 identical copies == over 1), and Dropout draws an
    # INDEPENDENT mask per batch row — exactly the iid samples the loop drew,
    # just from one RNG dispatch. Verified two ways in the test suite: the
    # dropout-disabled tiled pass is numerically identical per-row to the
    # single-row pass, and batched-vs-sequential MC means agree within noise.
    tiled = np.repeat(X_input, MC_DROPOUT_PASSES, axis=0)
    scores, _ = extractor(tiled, training=True)
    scores_array = scores.numpy()
    mean_scores = np.mean(scores_array, axis=0) * 100.0
    # Std across the 30 MC-Dropout passes = the model's own uncertainty in each
    # score. Previously computed nowhere, discarded everywhere.
    std_scores = np.std(scores_array, axis=0) * 100.0

    # round(), not int(): truncation silently biased every score down by up
    # to a full point (79.96 -> 79) and made overall_score a mean-of-
    # truncations rather than a true mean (audit M4).
    scores_by_metric = {
        "balance_score": round(float(mean_scores[0])),
        "power_score": round(float(mean_scores[1])),
        "technique_score": round(float(mean_scores[2])),
        "defence_score": round(float(mean_scores[3])),
    }
    weakness, strength, weakest_key = _describe_extremes(scores_by_metric)

    # Real per-feature attribution (Expected Gradients) toward the weakest
    # score -- identifies which joints actually drove that score down for
    # this specific session, rather than just naming the lowest of 4 numbers.
    try:
        driver_labels, drill_texts, phase_label = _top_feature_drivers(extractor, X_input, weakest_key)
        attribution_drivers = driver_labels
        recommended_drill = "; ".join(drill_texts)
        # Two-stage attribution also tells us WHEN, not just WHAT -- name the
        # phase when we have it, since "during your downswing" is far more
        # actionable to a batter than a joint name alone.
        if phase_label:
            weakness = f"{weakness} Biggest drivers: {', '.join(driver_labels)}, mainly during your {phase_label}."
        else:
            weakness = f"{weakness} Biggest drivers: {', '.join(driver_labels)}."
    except Exception:
        logger.exception("Gradient attribution failed; shipping score-level weakness only")
        attribution_drivers = []
        recommended_drill = None

    attention_focus = _attention_focus(extractor, att_layer, X_input)

    return {
        **scores_by_metric,
        "overall_score": round(float(np.mean(mean_scores))),
        "confidence_variance": {
            "balance": round(float(std_scores[0]), 2),
            "power": round(float(std_scores[1]), 2),
            "technique": round(float(std_scores[2]), 2),
            "defence": round(float(std_scores[3]), 2),
        },
        "primary_weakness": weakness,
        "primary_strength": strength,
        "attribution_drivers": attribution_drivers,
        "recommended_drill": recommended_drill,
        # WHERE the model concentrated (its own attention weights) -- a second,
        # architecturally independent signal from the Expected-Gradients
        # attribution above. Reported as focus, never as cause; see
        # _attention_focus's docstring for the Jain & Wallace caveat.
        "attention_focus": attention_focus,
        "is_fallback": False,
    }


METRIC_LABELS = {
    "balance_score": ("Balance", "base stability and head position through the shot"),
    "power_score": ("Power", "weight transfer and bat speed through contact"),
    "technique_score": ("Technique", "grip, backlift, and bat path correctness"),
    "defence_score": ("Defence", "stump coverage and head-over-ball at contact"),
}


def _describe_extremes(scores_by_metric):
    ranked = sorted(scores_by_metric.items(), key=lambda kv: kv[1])
    weakest_key, weakest_val = ranked[0]
    strongest_key, strongest_val = ranked[-1]
    w_label, w_desc = METRIC_LABELS[weakest_key]
    s_label, s_desc = METRIC_LABELS[strongest_key]
    weakness = f"{w_label} ({weakest_val}/100) is the area to work on — {w_desc}."
    strength = f"{s_label} ({strongest_val}/100) is the strongest part of this stance — {s_desc}."
    return weakness, strength, weakest_key
