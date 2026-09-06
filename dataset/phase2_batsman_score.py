"""
phase2_batsman_score.py — decide WHICH TRACK is the batsman.

WHY NOT THE OBVIOUS RULES
Phase 1 measured each of these failing:
  "largest person"      inverted for front-view footage: the batsman is the
                        small distant figure and the bowler is camera-near.
                        SIZE_DOMINANCE_RATIO fired on 1 clip in 102 and,
                        where it did fire, preferred the wrong person.
  "closest person"      same failure, stated differently.
  "most moving person"  the bowler runs in; the batsman stands at a crease.
  "highest wrist speed" a bowling action produced the corpus-maximum contact
                        confidence (0.98) on the WRONG person.

WHAT ACTUALLY SEPARATES A BATSMAN, GEOMETRICALLY
A batsman plays from a crease. Their feet stay on one spot of the ground
plane and their apparent size barely changes. A bowler or feeder approaches
the camera: their foot line sweeps down the image and their box height grows
substantially. That is a calibration-free signal — it needs no homography, no
known camera pose, and no stump detection, and it does not assume the batsman
is near or far, only that they are STATIONARY IN DEPTH.

Every feature below is normalised into [0,1] and expressed in units of the
track's own body size, so it transfers across the corpus's mixed resolutions
(640x360 landscape through 1080x1920 portrait).

WEIGHTS ARE FITTED, NOT ASSERTED
Hand-picked weights are how the size-dominance rule got its authority in the
first place. Here a logistic regression is fitted on the DEV split only and
reported on EVAL. The fitted coefficients are printed so the rule stays
inspectable rather than becoming an opaque scorer.
"""

import math
from typing import Dict, List, Optional

# Confidence below which a batsman decision is refused outright. A wrong
# batsman silently corrupts a training row; a rejected clip costs one clip.
CONFIDENT_THRESHOLD = 0.60
AMBIGUOUS_THRESHOLD = 0.40

CONFIDENT_BATSMAN = "CONFIDENT_BATSMAN"
AMBIGUOUS = "AMBIGUOUS"
NO_BATSMAN = "NO_BATSMAN"

FEATURE_NAMES = (
    "coverage",
    "foot_stability",
    "scale_stability",
    "centroid_stability",
    "relative_height",
    "centrality",
    "vertical_position",
    "mean_conf",
)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def track_features(observations: List, frame_w: int, frame_h: int,
                   frames_processed: int) -> Dict[str, float]:
    """Interpretable, scale-invariant features for one track.

    `observations` is [(frame_index, (x1,y1,x2,y2), conf), ...].
    """
    if not observations:
        return {k: 0.0 for k in FEATURE_NAMES}

    heights = [b[3] - b[1] for _, b, _ in observations]
    bottoms = [b[3] for _, b, _ in observations]
    cx = [(b[0] + b[2]) / 2.0 for _, b, _ in observations]
    cy = [(b[1] + b[3]) / 2.0 for _, b, _ in observations]
    confs = [c for _, _, c in observations]

    h_med = max(_median(heights), 1e-6)

    # Foot-line stability: the crease prior in calibration-free form. A
    # batsman's feet stay on one spot of the ground plane; an approaching
    # bowler's foot line sweeps down the image. Measured in the track's OWN
    # body heights so distance from camera does not bias it.
    foot_stability = 1.0 / (1.0 + _std(bottoms) / h_med)

    # Scale stability: apparent size is near-constant for someone stationary
    # in depth, and grows sharply for someone running at the camera. This is
    # the feature that directly targets the Phase-1 bowler failure.
    scale_stability = 1.0 / (1.0 + _std(heights) / h_med)

    # Net centroid travel in own-body units (NET, not path length: a stroke
    # moves the body out and back, ending where it started).
    disp = math.hypot(cx[-1] - cx[0], cy[-1] - cy[0]) / h_med
    centroid_stability = 1.0 / (1.0 + disp)

    return {
        "coverage": min(1.0, len(observations) / max(frames_processed, 1)),
        "foot_stability": foot_stability,
        "scale_stability": scale_stability,
        "centroid_stability": centroid_stability,
        # Kept as a FEATURE, not a rule. Phase 1's error was making size
        # decisive; letting the fit decide its sign and magnitude is the
        # point.
        "relative_height": min(1.0, h_med / max(frame_h, 1)),
        "centrality": 1.0 - min(1.0, abs(_median(cx) - frame_w / 2.0) / max(frame_w / 2.0, 1)),
        "vertical_position": min(1.0, _median(bottoms) / max(frame_h, 1)),
        "mean_conf": sum(confs) / len(confs),
    }


# Coefficients fitted on the DEV split by phase2_fit_batsman_model.py.
# Populated there and pasted here so scoring stays dependency-free and
# deterministic. Until fitted, `score_batsman_track` refuses rather than
# guessing with placeholder weights.
FITTED = None


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def score_batsman_track(track_obs: List, scene_context: dict,
                        track_id: Optional[int] = None,
                        model: Optional[dict] = None) -> dict:
    """Score one track. Returns components alongside the score so a decision
    can always be explained, never just asserted."""
    model = model or FITTED
    feats = track_features(track_obs, scene_context["frame_w"],
                           scene_context["frame_h"],
                           scene_context["frames_processed"])
    if model is None:
        return {"track_id": track_id, "score": None, "components": feats,
                "confidence": None,
                "reason": "no fitted model; refusing to score with placeholder weights"}

    # Only the features the model was FITTED on. A geometry-only model
    # declares a subset, and silently scoring it with the full feature list
    # would either KeyError or invent coefficients.
    z = model["intercept"] + sum(w * feats[k] for k, w in model["coef"].items())
    p = _sigmoid(z)
    return {"track_id": track_id, "score": round(p, 4), "components":
            {k: round(v, 4) for k, v in feats.items()},
            "confidence": round(p, 4)}


def select_batsman(tracks: Dict[int, List], scene_context: dict,
                   model: Optional[dict] = None) -> dict:
    """Choose the batsman track, or explicitly decline.

    Returns a verdict, never a forced pick: a wrong batsman silently corrupts
    a training row, while a rejected clip costs one clip.
    """
    model = model or FITTED
    if not tracks:
        return {"verdict": NO_BATSMAN, "track_id": None, "confidence": 0.0,
                "reason": "no tracks", "scores": []}

    scores = [score_batsman_track(obs, scene_context, tid, model)
              for tid, obs in sorted(tracks.items())]
    if any(s["score"] is None for s in scores):
        return {"verdict": AMBIGUOUS, "track_id": None, "confidence": None,
                "reason": "unfitted model", "scores": scores}

    scores.sort(key=lambda s: -s["score"])
    best = scores[0]
    runner_up = scores[1]["score"] if len(scores) > 1 else 0.0
    margin = best["score"] - runner_up

    if best["score"] >= CONFIDENT_THRESHOLD and (len(scores) == 1 or margin >= 0.15):
        verdict = CONFIDENT_BATSMAN
        reason = "clear winner"
    elif best["score"] >= AMBIGUOUS_THRESHOLD:
        verdict = AMBIGUOUS
        reason = (f"top score {best['score']:.2f} with margin {margin:.2f} "
                  f"— not separable enough to commit")
    else:
        verdict = NO_BATSMAN
        reason = f"no track scores above {AMBIGUOUS_THRESHOLD}"

    return {"verdict": verdict,
            "track_id": best["track_id"] if verdict == CONFIDENT_BATSMAN else None,
            "best_track_id": best["track_id"],
            "confidence": best["score"], "margin": round(margin, 4),
            "reason": reason, "scores": scores}


# ---------------------------------------------------------------------------
# Baselines — the rules Phase 1 used or that are commonly assumed, so the new
# scorer is measured against them rather than merely replacing them.
# ---------------------------------------------------------------------------

def baseline_largest(tracks: Dict[int, List], scene_context: dict) -> Optional[int]:
    """'The batsman is the biggest person.' Phase 1's implicit assumption."""
    if not tracks:
        return None
    return max(tracks.items(),
               key=lambda kv: _median([b[3] - b[1] for _, b, _ in kv[1]]))[0]


def baseline_size_dominance(tracks: Dict[int, List], scene_context: dict,
                            ratio: float = 1.25, min_coverage_frac: float = 0.5):
    """A faithful port of Phase 1's subject_selection rule: the winner must
    have maximal coverage AND be `ratio` times larger than every rival, else
    it refuses. Returns None when it refuses (which Phase 1 counted as a
    rejection, not a wrong answer)."""
    if not tracks:
        return None
    need = max(1, int(min_coverage_frac * scene_context["frames_processed"]))
    qualified = {t: o for t, o in tracks.items() if len(o) >= need}
    if not qualified:
        return None
    sizes = {t: _median([b[3] - b[1] for _, b, _ in o]) for t, o in qualified.items()}
    max_cov = max(len(o) for o in qualified.values())
    for t, o in qualified.items():
        if len(o) != max_cov:
            continue
        if all(sizes[t] >= ratio * sizes[r] for r in qualified if r != t):
            return t
    return None


def baseline_most_central(tracks: Dict[int, List], scene_context: dict) -> Optional[int]:
    if not tracks:
        return None
    w = scene_context["frame_w"]
    return min(tracks.items(),
               key=lambda kv: abs(_median([(b[0] + b[2]) / 2 for _, b, _ in kv[1]]) - w / 2))[0]


def baseline_most_persistent(tracks: Dict[int, List], scene_context: dict) -> Optional[int]:
    if not tracks:
        return None
    return max(tracks.items(), key=lambda kv: len(kv[1]))[0]
