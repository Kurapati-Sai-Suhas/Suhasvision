"""
contact_detection.py — contact-adjacent event localization on a CONFIRMED
batsman track (Phase 1).

WHY THIS MODULE EXISTS
`zero_storage_pipeline.find_wrist_speed_peak_frame` implements the same core
idea (peak wrist speed as a proxy for bat-ball contact) and was correct
arithmetic, but it ran BEFORE any subject selection and consumed
`detection_result.pose_landmarks[0]` — whoever MediaPipe happened to return
first. On real nets footage that was the ball feeder, so the "contact" it
found was a feeder's arm swing while walking. The feature was disabled
(WRIST_SPEED_SAMPLING_ENABLED = False) rather than shipped broken.

The defect was the PIPELINE ORDER, not the algorithm. This module takes the
wrist series of an already-confirmed batsman track, so that failure mode is
structurally impossible here: there is no code path by which this file can
see a non-selected person's landmarks.

WHAT THIS IS AND IS NOT
This is **contact-adjacent event localization**, not physical bat-ball
contact detection. At 25-30 fps a cricket ball travelling ~130 km/h covers
roughly 1.2-1.4 m per frame, so the true instant of contact is not
resolvable from this footage at all (cf. event-camera impact literature,
which needs ~1000 fps and still reports centimetre-scale ambiguity). The
honest claim is "the frame at or adjacent to peak bat-hand speed", accurate
to about +/-1 frame by construction. Callers must not upgrade that claim.

UPGRADES OVER THE ORIGINAL ARGMAX
 1. Scale normalization — speed in torso lengths, not raw normalized units,
    so a camera-near and a camera-far batsman are treated alike.
 2. Temporal smoothing — a centred 3-point moving average. Deliberately
    mild: with ~20 coarse samples a heavy filter would erase the very peak
    being looked for.
 3. Bilateral agreement — cricket is a two-handed shot, so both wrists
    should accelerate together. A one-handed spike is more typical of
    someone gesturing than of a stroke.
 4. Prominence — how far the peak stands above the track's own median
    speed. A flat velocity curve has an argmax too; prominence is what
    separates "a real event" from "the largest value in noise".
 5. Plausible temporal range — a peak in the first or last few percent of
    the window is more likely idle motion at a window edge than a stroke.
 6. A scored event, not a bare index — callers can gate on confidence
    instead of being forced to trust or discard.
"""

import numpy as np

# MediaPipe Pose landmark indices (same convention as
# zero_storage_pipeline.validate_pose and kinematic_validator.LANDMARKS).
LEFT_WRIST_IDX, RIGHT_WRIST_IDX = 15, 16
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_LEFT_HIP, _RIGHT_HIP = 23, 24

_EPS = 1e-6

# ---------------------------------------------------------------------------
# Configuration. Every constant below is a documented starting point in the
# MAX_BONE_CV tradition of this repo, NOT a validated optimum. Each is stated
# with the reasoning that produced it so a future tuner knows what evidence
# would justify moving it.
# ---------------------------------------------------------------------------

# Minimum usable speed samples. Each sample is a consecutive PAIR, so N
# detections give N-1 speeds; 4 pairs is the fewest from which a peak can be
# distinguished from an endpoint at all.
MIN_SPEED_SAMPLES = 4

# Centred moving-average width for the speed curve. 3 = one neighbour each
# side. Chosen as the mildest filter that suppresses single-sample detector
# jitter without displacing a genuine 1-2 sample peak.
SMOOTHING_WINDOW = 3

# Peak must stand at least this many times above the track's median speed.
# 1.5x is deliberately permissive: it is meant to reject a flat curve (where
# peak ~= median), not to enforce an explosive stroke.
MIN_PEAK_PROMINENCE_RATIO = 1.5

# At the peak, the slower wrist must reach at least this fraction of the
# faster one. Cricket strokes are two-handed; 0.25 rejects a clearly
# one-armed motion while tolerating the real asymmetry between top and
# bottom hand through the swing.
MIN_BILATERAL_AGREEMENT = 0.25

# The peak must fall within this fraction range of the window. A shot window
# from find_shot_windows is bounded roughly at stance and follow-through, so
# contact sits well inside it; peaks pinned to the very edges usually mean
# the window is misaligned or the motion is pre/post-shot idle.
MIN_PEAK_FRACTION, MAX_PEAK_FRACTION = 0.15, 0.95

# Below this, callers should not anchor sampling on the detected contact and
# should fall back. Set at the point where two of the three quality signals
# would have to be weak simultaneously.
MIN_CONTACT_CONFIDENCE = 0.35


def _torso_length(pose):
    """Mid-hip to mid-shoulder — the same scale unit subject_selection uses,
    so 'speed in torso lengths' means the same thing in both modules."""
    hx = (pose[_LEFT_HIP].x + pose[_RIGHT_HIP].x) / 2.0
    hy = (pose[_LEFT_HIP].y + pose[_RIGHT_HIP].y) / 2.0
    sx = (pose[_LEFT_SHOULDER].x + pose[_RIGHT_SHOULDER].x) / 2.0
    sy = (pose[_LEFT_SHOULDER].y + pose[_RIGHT_SHOULDER].y) / 2.0
    return max(float(np.hypot(sx - hx, sy - hy)), _EPS)


def _smooth(values, window=SMOOTHING_WINDOW):
    """Centred moving average with edge replication. Pure numpy, no scipy
    dependency, deterministic. Returns the input unchanged when it is too
    short to smooth (never silently shrinks the series)."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def wrist_speed_series(samples):
    """
    samples: time-ordered list of either None (this track not detected in
    that frame) or (frame_num, pose). Returns
    (frames, left_speeds, right_speeds) where each speed is the per-pair
    displacement in TORSO LENGTHS, and `frames` holds the frame number of
    the LATER frame of each pair (the frame the wrist arrives at).

    Pairs spanning a gap (a None between two detections) are skipped rather
    than bridged: a displacement measured across missing frames is not a
    speed, and treating it as one would manufacture a spurious peak exactly
    where detection failed.
    """
    frames, left, right = [], [], []
    for a, b in zip(samples, samples[1:]):
        if a is None or b is None:
            continue
        (_, pose_a), (frame_b, pose_b) = a, b
        scale = (_torso_length(pose_a) + _torso_length(pose_b)) / 2.0
        lx = pose_b[LEFT_WRIST_IDX].x - pose_a[LEFT_WRIST_IDX].x
        ly = pose_b[LEFT_WRIST_IDX].y - pose_a[LEFT_WRIST_IDX].y
        rx = pose_b[RIGHT_WRIST_IDX].x - pose_a[RIGHT_WRIST_IDX].x
        ry = pose_b[RIGHT_WRIST_IDX].y - pose_a[RIGHT_WRIST_IDX].y
        frames.append(frame_b)
        left.append(float(np.hypot(lx, ly)) / scale)
        right.append(float(np.hypot(rx, ry)) / scale)
    return frames, np.asarray(left), np.asarray(right)


def detect_contact(samples, start_frame, end_frame):
    """
    Locate the contact-adjacent frame for a CONFIRMED batsman track.

    samples: time-ordered [(frame_num, pose) | None], all belonging to the
             SAME already-selected subject. Passing raw detector output here
             re-creates the original defect — callers must select first.

    Returns a dict:
        {"frame_index": int|None, "confidence": float, "peak_prominence":
         float, "bilateral_agreement": float, "peak_fraction": float,
         "method": "wrist_speed", "valid": bool, "reason": str,
         "n_speed_samples": int}

    `valid` False means "do not anchor on this" — the caller falls back to
    its existing sampling. It is never an exception: a weak contact signal
    is an expected outcome on real footage, not an error.
    """
    result = {
        "frame_index": None, "confidence": 0.0, "peak_prominence": 0.0,
        "bilateral_agreement": 0.0, "peak_fraction": None,
        "method": "wrist_speed", "valid": False, "reason": "",
        "n_speed_samples": 0,
    }

    frames, left, right = wrist_speed_series(samples)
    result["n_speed_samples"] = len(frames)
    if len(frames) < MIN_SPEED_SAMPLES:
        result["reason"] = f"only {len(frames)} usable speed samples (need >= {MIN_SPEED_SAMPLES})"
        return result

    left_s, right_s = _smooth(left), _smooth(right)
    combined = np.maximum(left_s, right_s)

    peak_i = int(np.argmax(combined))
    peak_frame = frames[peak_i]
    peak_speed = float(combined[peak_i])
    median_speed = float(np.median(combined))

    # Prominence: how far the peak stands above this track's own typical
    # speed. Scale-free, so it is comparable across clips and subjects.
    prominence = peak_speed / (median_speed + _EPS)
    result["peak_prominence"] = round(prominence, 3)

    # Bilateral agreement at the peak — a two-handed stroke moves both.
    l_at, r_at = float(left_s[peak_i]), float(right_s[peak_i])
    agreement = min(l_at, r_at) / (max(l_at, r_at) + _EPS)
    result["bilateral_agreement"] = round(agreement, 3)

    span = max(end_frame - start_frame, 1)
    fraction = (peak_frame - start_frame) / span
    result["peak_fraction"] = round(fraction, 3)
    result["frame_index"] = int(peak_frame)

    failures = []
    if prominence < MIN_PEAK_PROMINENCE_RATIO:
        failures.append(f"flat velocity curve (prominence {prominence:.2f} < {MIN_PEAK_PROMINENCE_RATIO})")
    if agreement < MIN_BILATERAL_AGREEMENT:
        failures.append(f"one-handed motion (bilateral {agreement:.2f} < {MIN_BILATERAL_AGREEMENT})")
    if not (MIN_PEAK_FRACTION <= fraction <= MAX_PEAK_FRACTION):
        failures.append(f"peak at window edge (fraction {fraction:.2f})")

    # Confidence blends the three quality signals. Prominence is the primary
    # evidence that an event happened at all, so it carries the most weight;
    # the other two are corroborating checks.
    prominence_score = min(1.0, prominence / (2.0 * MIN_PEAK_PROMINENCE_RATIO))
    agreement_score = min(1.0, agreement / max(MIN_BILATERAL_AGREEMENT, _EPS))
    centrality = 1.0 - abs(fraction - 0.6) / 0.6  # 0.6 == uniform sampling's own contact prior
    centrality_score = float(np.clip(centrality, 0.0, 1.0))
    confidence = 0.5 * prominence_score + 0.25 * agreement_score + 0.25 * centrality_score
    result["confidence"] = round(float(confidence), 3)

    if failures:
        result["reason"] = "; ".join(failures)
        return result
    if confidence < MIN_CONTACT_CONFIDENCE:
        result["reason"] = f"confidence {confidence:.2f} < {MIN_CONTACT_CONFIDENCE}"
        return result

    result["valid"] = True
    result["reason"] = "contact-adjacent frame located"
    return result
