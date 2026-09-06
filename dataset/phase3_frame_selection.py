"""
phase3_frame_selection.py — choose the seven most informative frames from a
dense candidate pool, as a constrained optimization.

THE PROBLEM
Phase 1 and 2 both sampled seven frames directly (uniform, or anchored on a
contact estimate). That conflates two separate questions: WHICH frames exist
that are worth using, and WHICH SEVEN to keep. This module separates them:
generate a dense pool, score every candidate, then solve for the best seven
subject to hard constraints.

WHY DYNAMIC PROGRAMMING AND NOT SUBMODULAR GREEDY
The video-summarization literature solves "pick k diverse frames" with
submodular objectives (facility location being the standard choice), which
greedy maximizes to within (1 - 1/e) of optimal under a cardinality
constraint. That machinery does not fit here, because our problem is not an
unordered subset problem:

  * the seven frames must be strictly chronological;
  * each must fill a DIFFERENT phase slot, in order;
  * phase k's frame must come after phase k-1's.

Those are sequence constraints, not cardinality constraints. Once the
selection is an ordered assignment of 7 slots to N time-ordered candidates,
the problem becomes a shortest-path/alignment problem, which DP addresses
directly.

EXACTNESS: WHAT IS AND IS NOT PROVEN
The DP below is exact for THE OBJECTIVE IT IMPLEMENTS, and that objective is
deliberately first-order decomposable. It is not a universal solution to
every conceivable best-7 objective, and the distinction matters.

  objective       maximise
                    sum_{k=0..K-1} gain(i_k, k)
                  - W_red * sum_{k=1..K-1} red(i_{k-1}, i_k)
                  subject to i_0 < i_1 < ... < i_{K-1}
                             and frame(i_k) - frame(i_{k-1}) >= min_gap

  state           dp[k][i] = best achievable total over phases 0..k with
                  phase k assigned to candidate i
  initialization  dp[0][i] = gain(i, 0) for every candidate i
  transition      dp[k][i] = max over j < i satisfying the gap constraint of
                             ( dp[k-1][j] - W_red * red(j, i) ) + gain(i, k)
  termination     max over i of dp[K-1][i], recovered by backpointers
  complexity      O(K * N^2) time, O(K * N) space

  WHY THIS IS EXACT: every term in the objective depends on at most one
  candidate (gain) or one CONSECUTIVE pair (red). The objective therefore has
  the optimal-substructure property -- an optimal assignment of phases 0..k
  ending at i must contain an optimal assignment of phases 0..k-1 ending at
  its predecessor j, because nothing earlier than j can influence any term
  involving i. That is precisely the Markov property DP requires, so the
  table search is exhaustive over the feasible set rather than greedy.

  WHAT THIS DOES NOT COVER: objectives with higher-order interactions --
  global diversity across all seven frames simultaneously, redundancy between
  NON-consecutive selections, or any term depending on the set as a whole
  (facility-location coverage among them). Those break optimal substructure,
  and this DP would no longer be exact for them; they would need a different
  method, and submodular greedy with its (1 - 1/e) bound becomes the sensible
  choice at that point.

  So: exact for the implemented decomposable objective. The modelling claim
  being made is that consecutive-pair redundancy is sufficient for a
  seven-frame chronological sequence -- which is an engineering judgement
  open to challenge, not a theorem.

WHAT THIS MODULE DOES NOT DO
It does not decide who the batsman is, where the shot is, or where contact
is. It consumes those as inputs. That separation is deliberate: Phase 2
showed that a component which silently re-decides an upstream question is
how wrong-person errors hide.
"""

import math
from typing import Dict, List, Optional, Sequence

N_PHASES = 7
PHASE_NAMES = ("01_stance", "02_trigger", "03_backlift_start", "04_full_backlift",
               "05_downswing", "06_contact", "07_followthrough")

# Minimum frames between consecutively selected phases. Prevents the
# degenerate "seven frames inside 200 ms" solution that carries barely more
# information than one frame while consuming all seven slots.
MIN_PHASE_GAP = 2

# Hard floor on pose quality. A frame nobody can extract a pose from is not
# a candidate regardless of how well it fits a phase.
MIN_POSE_QUALITY = 0.25

# Expected position of each phase on a normalised timeline where the action
# onset is 0.0, contact is 1.0, and the action end is 1.0 + FOLLOW_SPAN.
# Derived from batting biomechanics rather than fitted: the pre-contact
# phases are NOT evenly spaced in time, because stance and trigger are
# near-static while downswing is explosive. These are documented starting
# points, in the MAX_BONE_CV tradition -- not validated optima.
PHASE_ANCHORS = (0.00,   # stance          - at onset
                 0.30,   # trigger         - early, still slow
                 0.55,   # backlift start  - bat begins to rise
                 0.75,   # full backlift   - top of the coil
                 0.92,   # downswing       - just before impact, explosive
                 1.00,   # contact         - the anchor itself
                 1.45)   # follow-through  - past contact
FOLLOW_SPAN = 0.6        # how far past contact the action is taken to run

# Width of the Gaussian phase-affinity window, in normalised timeline units.
PHASE_SIGMA = 0.16


class Candidate:
    """One frame in the pool, with everything known about it."""

    __slots__ = ("frame", "t", "pose_quality", "sharpness", "identity_conf",
                 "motion", "visibility", "occlusion", "temporal_stability",
                 "bat_conf", "extra")

    def __init__(self, frame, t=0.0, pose_quality=0.0, sharpness=0.0,
                 identity_conf=0.0, motion=0.0, visibility=0.0, occlusion=0.0,
                 temporal_stability=1.0, bat_conf=0.0, extra=None):
        self.frame = frame
        self.t = t
        self.pose_quality = pose_quality
        self.sharpness = sharpness
        self.identity_conf = identity_conf
        self.motion = motion
        self.visibility = visibility
        self.occlusion = occlusion
        self.temporal_stability = temporal_stability
        self.bat_conf = bat_conf
        self.extra = extra or {}

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__ if k != "extra"}


# Frame-quality weights. Interpretable and normalised; every component is in
# [0,1] before weighting, so the weights are directly comparable. They are
# NOT fitted -- there is no per-frame quality ground truth to fit them
# against, and inventing one would be worse than stating them plainly.
QUALITY_WEIGHTS = {
    "pose_quality": 0.34,        # can we get landmarks at all -- dominant
    "identity_conf": 0.22,       # is this the right person
    "sharpness": 0.18,           # motion blur kills landmark precision
    "visibility": 0.14,          # are the joints we need actually visible
    "temporal_stability": 0.12,  # is the box consistent with its neighbours
}
OCCLUSION_PENALTY = 0.30


def frame_quality(c: Candidate) -> float:
    """Identity- and pose-centred quality in [0,1].

    Motion is deliberately ABSENT. A high-motion frame is often the blurriest
    and least extractable one, so motion belongs in phase affinity (where it
    indicates WHICH phase a frame is) rather than in quality (which asks
    whether the frame is usable at all). Conflating the two is how a
    motion-maximising sampler ends up selecting the least analysable frames.
    """
    q = sum(w * max(0.0, min(1.0, getattr(c, k))) for k, w in QUALITY_WEIGHTS.items())
    q -= OCCLUSION_PENALTY * max(0.0, min(1.0, c.occlusion))
    return max(0.0, min(1.0, q))


def normalised_time(frame, onset, contact, end):
    """Map a frame onto the timeline where onset=0, contact=1, end=1+FOLLOW_SPAN."""
    if contact > onset:
        if frame <= contact:
            return (frame - onset) / float(contact - onset)
        if end > contact:
            return 1.0 + FOLLOW_SPAN * (frame - contact) / float(end - contact)
        return 1.0
    return 0.0


def phase_affinity(u: float, phase_idx: int, motion_norm: float = 0.5) -> float:
    """How well a candidate at normalised position `u` fits phase `phase_idx`.

    Two terms. Temporal: a Gaussian around the phase's anchor. Kinematic: the
    expected motion signature -- stance and trigger should be quiet, downswing
    and contact should be energetic. The kinematic term is what stops a purely
    positional model from labelling a static frame "downswing" just because it
    happens to fall at the right time.
    """
    temporal = math.exp(-((u - PHASE_ANCHORS[phase_idx]) ** 2) / (2 * PHASE_SIGMA ** 2))

    # Expected normalised motion per phase; same provenance as PHASE_ANCHORS.
    expected_motion = (0.10, 0.20, 0.40, 0.55, 0.90, 1.00, 0.60)[phase_idx]
    kinematic = 1.0 - abs(motion_norm - expected_motion)
    return 0.75 * temporal + 0.25 * max(0.0, kinematic)


# Objective weights. Phase fit is weighted above raw quality on purpose: a
# beautiful frame in the wrong phase slot breaks the (7,30) tensor's meaning,
# whereas a merely adequate frame in the right slot does not.
W_QUALITY = 0.40
W_PHASE = 0.45
W_CONTACT = 0.15
REDUNDANCY_WEIGHT = 0.35


def contact_relevance(frame, contact, phase_idx, fps=30.0):
    """Extra credit for the contact phase landing near the contact estimate.

    Only phase 6 (contact) gets this. Rewarding every phase for proximity to
    contact would pull the whole sequence into a cluster around impact --
    the exact temporal collapse the optimizer exists to prevent.
    """
    if phase_idx != 5 or contact is None:
        return 0.0
    dt = abs(frame - contact) / max(fps, 1.0)
    return math.exp(-(dt ** 2) / (2 * 0.08 ** 2))     # ~80 ms tolerance


def redundancy(a: Candidate, b: Candidate, fps=30.0) -> float:
    """Penalty for two consecutively selected frames being near-identical.
    Time-based rather than pixel-based: two frames 30 ms apart in a 3-second
    action are redundant whatever their pixels say."""
    dt = abs(b.frame - a.frame) / max(fps, 1.0)
    return math.exp(-(dt ** 2) / (2 * 0.10 ** 2))


def select_best_seven(candidates: Sequence[Candidate], onset, contact, end,
                      fps=30.0, min_gap=MIN_PHASE_GAP,
                      min_pose_quality=MIN_POSE_QUALITY) -> Dict:
    """Exact DP over ordered phase assignments.

    dp[k][i] = best total score with phase k placed on candidate i.
    Transition: dp[k][i] = max_{j < i, gap ok} dp[k-1][j] + gain(i,k)
                                              - REDUNDANCY_WEIGHT * red(j,i)

    Returns the selection plus enough detail to explain WHY each frame was
    chosen -- an unexplainable selection is not auditable, and Phase 1
    established that unauditable extraction is how wrong-person errors
    survive.
    """
    pool = [c for c in candidates if c.pose_quality >= min_pose_quality]
    pool = sorted(pool, key=lambda c: c.frame)
    n = len(pool)
    if n < N_PHASES:
        return {"ok": False,
                "reason": f"only {n} candidates clear the pose-quality floor "
                          f"({min_pose_quality}); need {N_PHASES}",
                "frames": None, "n_pool": n}

    motions = [c.motion for c in pool]
    lo, hi = min(motions), max(motions)
    span = (hi - lo) or 1.0
    mnorm = [(m - lo) / span for m in motions]
    us = [normalised_time(c.frame, onset, contact, end) for c in pool]
    quals = [frame_quality(c) for c in pool]

    def gain(i, k):
        return (W_QUALITY * quals[i]
                + W_PHASE * phase_affinity(us[i], k, mnorm[i])
                + W_CONTACT * contact_relevance(pool[i].frame, contact, k, fps))

    NEG = float("-inf")
    dp = [[NEG] * n for _ in range(N_PHASES)]
    back = [[-1] * n for _ in range(N_PHASES)]
    for i in range(n):
        dp[0][i] = gain(i, 0)

    for k in range(1, N_PHASES):
        for i in range(n):
            g = gain(i, k)
            best, bj = NEG, -1
            for j in range(i):
                if dp[k - 1][j] == NEG:
                    continue
                if pool[i].frame - pool[j].frame < min_gap:
                    continue
                v = dp[k - 1][j] - REDUNDANCY_WEIGHT * redundancy(pool[j], pool[i], fps)
                if v > best:
                    best, bj = v, j
            if bj >= 0:
                dp[k][i] = best + g
                back[k][i] = bj

    last = max(range(n), key=lambda i: dp[N_PHASES - 1][i])
    if dp[N_PHASES - 1][last] == NEG:
        return {"ok": False,
                "reason": f"no chronological assignment satisfies min_gap={min_gap} "
                          f"across {n} candidates",
                "frames": None, "n_pool": n}

    idx = [0] * N_PHASES
    i = last
    for k in range(N_PHASES - 1, -1, -1):
        idx[k] = i
        i = back[k][i]

    chosen = [pool[i] for i in idx]
    return {
        "ok": True,
        "frames": [c.frame for c in chosen],
        "total_score": round(dp[N_PHASES - 1][last], 4),
        "n_pool": n,
        "per_frame": [{
            "phase": PHASE_NAMES[k],
            "frame": chosen[k].frame,
            "t": round(chosen[k].frame / max(fps, 1.0), 3),
            "quality": round(quals[idx[k]], 4),
            "phase_affinity": round(phase_affinity(us[idx[k]], k, mnorm[idx[k]]), 4),
            "normalised_position": round(us[idx[k]], 3),
            "pose_quality": round(chosen[k].pose_quality, 4),
            "sharpness": round(chosen[k].sharpness, 4),
            "identity_conf": round(chosen[k].identity_conf, 4),
        } for k in range(N_PHASES)],
        "temporal_span_s": round((chosen[-1].frame - chosen[0].frame) / max(fps, 1.0), 3),
        "min_gap_frames": min(b.frame - a.frame for a, b in zip(chosen, chosen[1:])),
    }


# ---------------------------------------------------------------------------
# Baseline strategies, for the F0-F4 comparison
# ---------------------------------------------------------------------------

def select_uniform(candidates, onset, contact, end, fps=30.0):
    """F0 - what production does today."""
    pool = sorted(candidates, key=lambda c: c.frame)
    if len(pool) < N_PHASES:
        return {"ok": False, "reason": "pool too small", "frames": None}
    lo, hi = pool[0].frame, pool[-1].frame
    want = [lo + (hi - lo) * i / (N_PHASES - 1) for i in range(N_PHASES)]
    return {"ok": True, "frames": [min(pool, key=lambda c: abs(c.frame - w)).frame
                                   for w in want]}


def select_contact_anchored(candidates, onset, contact, end, fps=30.0):
    """F1 - Phase-1 behaviour: spread pre-contact phases, anchor phase 6."""
    pool = sorted(candidates, key=lambda c: c.frame)
    if len(pool) < N_PHASES or contact is None:
        return {"ok": False, "reason": "pool too small or no contact", "frames": None}
    start, endf = pool[0].frame, pool[-1].frame
    before = [start + (contact - start) * i / (N_PHASES - 2) for i in range(N_PHASES - 2)]
    want = before + [contact, endf]
    return {"ok": True, "frames": [min(pool, key=lambda c: abs(c.frame - w)).frame
                                   for w in want]}


def select_motion_top(candidates, onset, contact, end, fps=30.0):
    """F2 - the naive 'most motion' strategy, kept as a baseline precisely
    because it is the intuitive choice this module argues against."""
    pool = sorted(candidates, key=lambda c: -c.motion)[:N_PHASES]
    if len(pool) < N_PHASES:
        return {"ok": False, "reason": "pool too small", "frames": None}
    return {"ok": True, "frames": sorted(c.frame for c in pool)}


def select_phase_aware(candidates, onset, contact, end, fps=30.0):
    """F3 - phase affinity only, greedily, with no quality term and no
    redundancy penalty. Isolates what the phase model contributes on its own."""
    pool = sorted(candidates, key=lambda c: c.frame)
    if len(pool) < N_PHASES:
        return {"ok": False, "reason": "pool too small", "frames": None}
    motions = [c.motion for c in pool]
    lo, hi = min(motions), max(motions)
    span = (hi - lo) or 1.0
    us = [normalised_time(c.frame, onset, contact, end) for c in pool]

    chosen, used_after = [], -1
    for k in range(N_PHASES):
        best, bi = None, -1
        for i, c in enumerate(pool):
            if c.frame <= used_after:
                continue
            a = phase_affinity(us[i], k, (c.motion - lo) / span)
            if best is None or a > best:
                best, bi = a, i
        if bi < 0:
            return {"ok": False, "reason": "ran out of chronological candidates",
                    "frames": None}
        chosen.append(pool[bi].frame)
        used_after = pool[bi].frame
    return {"ok": True, "frames": chosen}


STRATEGIES = {
    "F0_uniform": select_uniform,
    "F1_contact_anchored": select_contact_anchored,
    "F2_motion_top": select_motion_top,
    "F3_phase_aware": select_phase_aware,
    "F4_optimized": lambda c, o, ct, e, fps=30.0: select_best_seven(c, o, ct, e, fps),
}
