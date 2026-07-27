"""
subject_selection.py — deterministic multi-person subject selection for the
7-phase pose pipeline (Milestone 4, audit C1).

WHY THIS EXISTS
MediaPipe's PoseLandmarker ran with num_poses=1 everywhere: ONE person per
frame, chosen by the detector, with no guarantee it's the batsman. Verified
with real footage + landmark overlays (architecture_ground_truth.md, review
rounds 2-5): on a real nets clip the pipeline tracked the ball FEEDER in 3/3
AI-detected shots, and only 2 of those 3 were accidentally caught downstream
by the bone-length CV filter. The fix: detect up to MAX_POSE_CANDIDATES
people per frame and make an explicit, session-level, deterministic choice —
or refuse to choose when no candidate clearly wins. Correct rejection is
preferable to silently scoring the wrong person.

HOW SELECTION WORKS (deliberately simple; every constant is a documented
starting point, not a validated optimum — tune against real multi-person
footage once a ground-truth-labeled clip exists):

1. TRACKS — candidates are associated across the sampled frames by hip-center
   proximity, gated in units of torso length (scale-invariant), with the gate
   widening proportionally after unseen gaps so a subject briefly lost to
   motion blur re-associates instead of spawning a new track.
2. QUALIFICATION — a track must cover >= MIN_TRACK_COVERAGE frames.
   Deliberately equal to what apply_pipeline_rules can survive (max 3 of 7
   missing): selecting a less-persistent subject would be rejected two steps
   later anyway.
3. WINNER — a single qualified track wins outright. With several, the winner
   must be BOTH at least as persistent as every rival AND clearly larger
   (SIZE_DOMINANCE_RATIO x mean torso) than every rival. If no track
   dominates on both axes, selection REJECTS rather than guesses:
   persistence alone can favor the wrong person (the feeder was detected in
   EVERY frame of the documented real failure), and size alone can favor a
   foreground walk-through. At most one track can dominate (two tracks
   cannot each be 1.25x the other's size), so the outcome never depends on
   iteration order.

Poses are consumed via attribute access (.x/.y), matching MediaPipe's
landmark objects; tests use namedtuples. Pure Python + math only — no heavy
imports, unit-testable everywhere.
"""

import math

# Detector-level cap on how many people MediaPipe may return per frame.
# 4 covers the realistic worst case in batting footage (batsman, bowler/
# feeder, keeper, umpire) without inviting far-background noise.
MAX_POSE_CANDIDATES = 4

# A track must appear in at least this many of the sampled frames to be a
# subject candidate. 4-of-7 matches apply_pipeline_rules' existing survival
# rule (>= 4 frames present after interpolation limits).
MIN_TRACK_COVERAGE = 4

# Association gate: how far (in torso lengths) a subject's hip center may
# move between consecutive OBSERVED frames. The 7 frames span a whole shot,
# so consecutive samples can be ~0.5s apart — a real swing moves the hips
# well under 2 torso lengths between phases. The gate scales with the gap
# when frames were missed, capped so nothing "teleports" across the frame.
GATE_TORSOS_PER_FRAME = 2.0
GATE_TORSOS_MAX = 4.0

# With several qualified tracks, the winner's mean torso length must be at
# least this multiple of every rival's — the filmed subject in batting
# footage is the prominent, camera-near figure.
SIZE_DOMINANCE_RATIO = 1.25

# MediaPipe Pose landmark indices (same convention as validate_pose and
# kinematic_validator.py).
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_LEFT_HIP, _RIGHT_HIP = 23, 24

_MIN_SCALE = 1e-6


def pose_center_and_scale(pose):
    """(hip-center x, hip-center y, torso length) for one 33-landmark pose.
    Torso length (mid-hip to mid-shoulder) is the scale unit: it's stable
    across batting phases, unlike limb spans mid-swing."""
    hip_x = (pose[_LEFT_HIP].x + pose[_RIGHT_HIP].x) / 2.0
    hip_y = (pose[_LEFT_HIP].y + pose[_RIGHT_HIP].y) / 2.0
    shoulder_x = (pose[_LEFT_SHOULDER].x + pose[_RIGHT_SHOULDER].x) / 2.0
    shoulder_y = (pose[_LEFT_SHOULDER].y + pose[_RIGHT_SHOULDER].y) / 2.0
    torso = math.hypot(shoulder_x - hip_x, shoulder_y - hip_y)
    return hip_x, hip_y, max(torso, _MIN_SCALE)


def _mean_scale(track):
    return sum(track["scales"]) / len(track["scales"])


def build_tracks(candidates_per_frame):
    """
    Greedy, deterministic temporal association of per-frame pose candidates
    into subject tracks. Candidates are processed in detector order; each
    matches the nearest open track within the gate (ties broken by lower
    track index), at most one candidate per track per frame; unmatched
    candidates open new tracks. Returns a list of track dicts:
      {"poses": {frame_idx: pose}, "last_center": (x, y),
       "last_frame": int, "scales": [floats]}
    """
    tracks = []
    for frame_idx, candidates in enumerate(candidates_per_frame):
        matched_track_ids = set()
        for pose in candidates:
            cx, cy, scale = pose_center_and_scale(pose)
            best = None  # (distance, track_idx)
            for t_idx, track in enumerate(tracks):
                if t_idx in matched_track_ids:
                    continue
                gap = frame_idx - track["last_frame"]
                # max() of the two scales: detection-scale jitter must widen
                # the gate, never shrink it into spurious track splits.
                gate = min(GATE_TORSOS_PER_FRAME * gap, GATE_TORSOS_MAX) * max(_mean_scale(track), scale)
                dist = math.hypot(cx - track["last_center"][0], cy - track["last_center"][1])
                if dist <= gate and (best is None or dist < best[0]):
                    best = (dist, t_idx)
            if best is None:
                tracks.append({
                    "poses": {frame_idx: pose},
                    "last_center": (cx, cy),
                    "last_frame": frame_idx,
                    "scales": [scale],
                })
                # A newly opened track is spoken for this frame — a second
                # candidate in the SAME frame must never join it.
                matched_track_ids.add(len(tracks) - 1)
            else:
                track = tracks[best[1]]
                track["poses"][frame_idx] = pose
                track["last_center"] = (cx, cy)
                track["last_frame"] = frame_idx
                track["scales"].append(scale)
                matched_track_ids.add(best[1])
    return tracks


def select_subject(candidates_per_frame):
    """
    The one session-level subject decision (never per-frame — a one-frame
    choice is exactly how the wrong person got tracked before).

    Returns (selected, report):
      selected — per-frame list of the chosen subject's pose, None in frames
                 where the chosen track has no detection (the existing
                 interpolation rules handle those), or None overall when no
                 subject qualifies or dominates.
      report   — {"n_tracks", "multi_track", "qualified", "reason",
                  "winner_coverage", "winner_scale"} for logging.
    """
    n_frames = len(candidates_per_frame)
    tracks = build_tracks(candidates_per_frame)
    report = {"n_tracks": len(tracks), "multi_track": len(tracks) > 1}

    if not tracks:
        report["reason"] = "no pose candidates in any frame"
        return None, report

    qualified = [t for t in tracks if len(t["poses"]) >= MIN_TRACK_COVERAGE]
    report["qualified"] = len(qualified)
    if not qualified:
        best_coverage = max(len(t["poses"]) for t in tracks)
        report["reason"] = (
            f"no persistent subject: best track covers {best_coverage}/{n_frames} "
            f"frames (need >= {MIN_TRACK_COVERAGE})"
        )
        return None, report

    if len(qualified) == 1:
        winner = qualified[0]
    else:
        max_coverage = max(len(t["poses"]) for t in qualified)
        winner = None
        for track in qualified:
            rivals = [t for t in qualified if t is not track]
            if len(track["poses"]) == max_coverage and all(
                _mean_scale(track) >= SIZE_DOMINANCE_RATIO * _mean_scale(rival) for rival in rivals
            ):
                winner = track
                break
        if winner is None:
            report["reason"] = (
                f"{len(qualified)} persistent subjects and none dominates (winner needs "
                f"max coverage AND {SIZE_DOMINANCE_RATIO}x the size of every rival) — refusing to guess"
            )
            return None, report

    report["winner_coverage"] = len(winner["poses"])
    report["winner_scale"] = round(_mean_scale(winner), 4)
    report["reason"] = "selected"
    selected = [winner["poses"].get(i) for i in range(n_frames)]
    return selected, report
