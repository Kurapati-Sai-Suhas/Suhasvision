"""
phase3_failure_taxonomy.py — separate "the track wasn't there" from
"the pose model failed".

THE BUG THIS REPLACES
The old diagnostic used a single statistic, `k_pose_rate`, defined as

    k_pose_rate = (sampled frames with a pose) / (sampled frames)

and called a clip a pose failure when it fell below 0.4. But
phase3_semantic_features.extract() appends `None` to the sequence both when
the pose model ran and failed AND when the batsman track simply has no box on
that frame:

    box = by_frame.get(f)
    if box is None or fr is None:
        seq.append(None)

The pose model is never invoked in the second case. So `k_pose_rate` is a
product of two independent things, and only one of them is about pose:

    effective_pose_rate = coverage x P(pose | box exists)

A frame with no batsman track box is NOT evidence against the pose model, and
must not be counted as such.

THE SECOND BUG: AN UNREACHABLE BRANCH
The old categorise_failure() tested, in this order:

    if k_pose_rate < 0.4:  return "pose failure on the batsman crop"
    if coverage    < 0.4:  return "tracking failure - track too fragmented"

Since a pose requires a box, k_pose_rate <= coverage ALWAYS. So coverage < 0.4
implies k_pose_rate < 0.4, the pose branch returns first, and the tracking
branch is unreachable for exactly the clips it was written to catch. Every
fragmented-track failure was reported as a pose failure.

The fix is to test the factors in causal order — coverage first, because
coverage is upstream of pose and caps it — and to name the joint case rather
than forcing it into one bucket.
"""

# Both thresholds are the 0.4 the pipeline already used for k_pose_rate, kept
# so the corrected taxonomy stays comparable to the numbers it replaces.
COVERAGE_THRESHOLD = 0.40
POSE_THRESHOLD = 0.40

TRACKING_FAILURE = "TRACKING_FAILURE"
POSE_FAILURE = "POSE_FAILURE"
JOINT_FAILURE = "JOINT_TRACKING+POSE_FAILURE"
NO_FAILURE = "NO_FAILURE"

FAILURE_TYPES = (TRACKING_FAILURE, POSE_FAILURE, JOINT_FAILURE, NO_FAILURE)

SEMANTICS = {
    TRACKING_FAILURE: ("The batsman track is absent from most sampled frames. "
                       "Where a box does exist the pose model performs "
                       "acceptably. No pose model can fix this."),
    POSE_FAILURE: ("The batsman track is present, but the pose model fails on "
                   "the crops it is given. This is the only category a better "
                   "pose model can fix."),
    JOINT_FAILURE: ("Both factors are below threshold: the track is fragmented "
                    "AND the pose model fails on the frames that do exist. "
                    "Fixing pose alone raises the ceiling only to `coverage`."),
    NO_FAILURE: "Both factors are acceptable.",
}


def effective_pose_rate(coverage, pose_given_box):
    """The quantity the old `k_pose_rate` was measuring, stated explicitly.

    Kept as a function rather than inlined so the decomposition is testable
    and so the identity effective = coverage x P(pose|box) is asserted in one
    place instead of assumed in several.
    """
    if coverage is None or pose_given_box is None:
        return None
    return coverage * pose_given_box


def classify(coverage, pose_given_box,
             coverage_threshold=COVERAGE_THRESHOLD,
             pose_threshold=POSE_THRESHOLD):
    """Failure type from the two INDEPENDENT factors.

    Order matters and is causal, not arbitrary: coverage is upstream of pose
    and caps it, so it is tested first. Testing pose first is what made the
    tracking branch unreachable.

    `pose_given_box` may be None when the track has no boxes at all on the
    sampled frames — there is then no evidence about the pose model, and the
    verdict is TRACKING_FAILURE rather than a guess.
    """
    if coverage is None:
        return NO_FAILURE
    cov_bad = coverage < coverage_threshold
    if pose_given_box is None:
        # No box was ever supplied, so the pose model was never asked.
        return TRACKING_FAILURE if cov_bad else NO_FAILURE
    pose_bad = pose_given_box < pose_threshold
    if cov_bad and pose_bad:
        return JOINT_FAILURE
    if cov_bad:
        return TRACKING_FAILURE
    if pose_bad:
        return POSE_FAILURE
    return NO_FAILURE


def decompose(n_sampled, n_box, n_pose):
    """Full record from the three counts that extraction can actually observe.

    n_sampled : frames sampled for this clip
    n_box     : of those, frames where THIS track has a box
    n_pose     : of those boxed frames, frames where the pose model succeeded
    """
    if n_pose > n_box:
        raise ValueError(f"n_pose ({n_pose}) > n_box ({n_box}): a pose "
                         f"cannot exist without a box")
    if n_box > n_sampled:
        raise ValueError(f"n_box ({n_box}) > n_sampled ({n_sampled})")
    coverage = (n_box / n_sampled) if n_sampled else 0.0
    pose_given_box = (n_pose / n_box) if n_box else None
    eff = (n_pose / n_sampled) if n_sampled else 0.0
    return {
        "n_sampled": n_sampled,
        "n_box": n_box,
        "n_pose": n_pose,
        "coverage": round(coverage, 4),
        "pose_success_given_box": (round(pose_given_box, 4)
                                   if pose_given_box is not None else None),
        "effective_pose_rate": round(eff, 4),
        "failure_type": classify(coverage, pose_given_box),
    }
