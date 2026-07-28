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
2b. TRAVERSAL FILTER — candidates whose net hip-center travel exceeds
   MAX_SUBJECT_DISPLACEMENT_TORSOS are dropped. This is the BOWLER filter:
   a camera-near bowler visible throughout the window is both large and
   persistent, so steps 2 and 3 alone would happily select him. What
   actually distinguishes the two is that a batsman plays from a fixed
   crease while a bowler traverses the scene.
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

# Traversal limits: how far a candidate's hip center may travel and still be
# a batsman playing from a CREASE. A track is disqualified only when it
# exceeds BOTH limits below.
#
# WHY THIS FILTER EXISTS: size and persistence alone cannot separate a
# batsman from a BOWLER. A bowler running in is often camera-near (so
# comparably large) and visible in every frame of the window (so equally
# persistent) — the exact profile the size+coverage rules were built to
# select. What separates them is that a batsman plays from a fixed crease
# (a stride, a trigger movement) while a bowler traverses the scene.
#
# WHY TWO MEASURES, AND WHY "BOTH": each normalization alone has a blind
# spot, and they are blind in opposite directions —
#   * Torso units are scale-invariant, but DEFLATE for a camera-near
#     subject: a person with a 0.30 torso cannot physically travel more than
#     ~3.3 of their own torso lengths across a normalized frame, so a
#     torso-only threshold can barely fire for the large, near bowler this
#     filter most needs to catch.
#   * Frame fraction directly measures "crossed the scene", but INFLATES in
#     a tight close-up, where a real batsman's ordinary stride can span a
#     large fraction of the frame.
# Requiring both to be exceeded keeps the true positives (a bowler is far
# from the crease AND crosses the frame, on either measure) while protecting
# the two false-positive cases (close-up batsman, distant batsman).
#
# Both are documented starting points in the MAX_BONE_CV tradition, not
# validated optima — tune against real multi-person footage once
# frame-accurate ground truth exists. Real batting motion measures roughly
# 0.3–1.5 torsos / 0.05–0.15 frame; both limits sit clearly above that.
#
# NOTE ON CAMERA PANNING: landmarks are normalized to the frame, so a camera
# that tracks the batsman keeps the batsman's normalized position roughly
# constant (low displacement, correctly kept) while sweeping background
# figures across the frame (high displacement, correctly disqualified). A
# camera panning AWAY from the batsman drops their coverage instead, which
# MIN_TRACK_COVERAGE already handles.
MAX_SUBJECT_DISPLACEMENT_TORSOS = 2.0
MAX_SUBJECT_DISPLACEMENT_FRAME = 0.25

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


def _net_displacement_raw(track):
    """Net hip-center travel (first -> last observed frame) in normalized
    frame units. Deliberately NET, not total path length: a batsman's hips
    move forward and back across a stroke, ending near where they started,
    so path length would penalise real batting motion. Net displacement
    measures what actually matters here — did this person end up somewhere
    else, i.e. traverse the scene."""
    centers = track["centers"]
    if len(centers) < 2:
        return 0.0
    (x0, y0), (x1, y1) = centers[0], centers[-1]
    return math.hypot(x1 - x0, y1 - y0)


def net_displacement_torsos(track):
    """Net travel in the track's own mean torso lengths — scale-invariant,
    so a near/large and far/small person moving the same fraction of their
    own body length score identically."""
    return _net_displacement_raw(track) / _mean_scale(track)


def net_displacement_frame(track):
    """Net travel as a fraction of the frame — measures 'crossed the scene'
    directly, without the torso-unit deflation that affects camera-near
    subjects (see MAX_SUBJECT_DISPLACEMENT_* for why both are needed)."""
    return _net_displacement_raw(track)


def is_traversing(track):
    """True when a track travels too far to be batting from a crease. Requires
    BOTH measures to be exceeded — see MAX_SUBJECT_DISPLACEMENT_* for why
    either one alone has a blind spot in the opposite direction."""
    return (net_displacement_torsos(track) > MAX_SUBJECT_DISPLACEMENT_TORSOS
            and net_displacement_frame(track) > MAX_SUBJECT_DISPLACEMENT_FRAME)


def build_tracks(candidates_per_frame):
    """
    Greedy, deterministic temporal association of per-frame pose candidates
    into subject tracks. Candidates are processed in detector order; each
    matches the nearest open track within the gate (ties broken by lower
    track index), at most one candidate per track per frame; unmatched
    candidates open new tracks. Returns a list of track dicts:
      {"poses": {frame_idx: pose}, "last_center": (x, y),
       "last_frame": int, "scales": [floats], "centers": [(x, y), ...]}

    "centers" holds every observed hip center in frame order (parallel to
    "scales"), so net_displacement_torsos can tell a batsman playing from a
    fixed crease apart from a bowler traversing the scene.
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
                    "centers": [(cx, cy)],
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
                track["centers"].append((cx, cy))
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
                  "traversing", "winner_coverage", "winner_scale",
                  "winner_displacement"} for logging.
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

    # Bowler/traversal filter. Size and persistence alone cannot separate a
    # batsman from a camera-near bowler who is visible throughout the window
    # -- that candidate profile is exactly what the dominance rules below
    # would happily select. A batsman plays from a fixed crease; anyone who
    # nets more than MAX_SUBJECT_DISPLACEMENT_TORSOS of travel crossed the
    # scene and is not batting at one.
    stationary = [t for t in qualified if not is_traversing(t)]
    report["traversing"] = len(qualified) - len(stationary)
    if not stationary:
        nearest = min(net_displacement_torsos(t) for t in qualified)
        report["reason"] = (
            f"every persistent candidate traverses the scene (nearest is {nearest:.1f} torso lengths "
            f"of net travel, limits {MAX_SUBJECT_DISPLACEMENT_TORSOS} torsos AND "
            f"{MAX_SUBJECT_DISPLACEMENT_FRAME} frame) -- no batsman at a crease here"
        )
        return None, report
    qualified = stationary

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
    report["winner_displacement"] = round(net_displacement_torsos(winner), 2)
    report["reason"] = "selected"
    selected = [winner["poses"].get(i) for i in range(n_frames)]
    return selected, report
