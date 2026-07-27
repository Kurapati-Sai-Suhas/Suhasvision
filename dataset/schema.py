# schema.py — Single source of truth for the feature schema.
# Import from here in ANY script that needs EXPECTED_FEATURES, SEQ_LEN, or
# FEATURE_BASE_NAMES. This avoids triggering TensorFlow/MediaPipe on import
# just to read a list of strings and a sequence length -- keep it that way;
# do not add TensorFlow/Keras imports to this file (see model_layers.py for
# the shared Keras layer definitions instead).

# The 15 base angle names, before the _vel duplication below. Exposed
# separately because some consumers (e.g. gradient-attribution joint
# grouping) need to pair an angle with its own velocity counterpart rather
# than iterate the flat 30-item list.
#
# A tuple, deliberately -- this is imported by reference into every consumer
# module. A list here would let any one of them accidentally mutate the
# "single source of truth" in place (e.g. a stray .append()) and silently
# corrupt it for every other consumer in the same process. Only ever
# accessed via len()/integer indexing across the codebase, so tuple-vs-list
# changes nothing for any current caller.
FEATURE_BASE_NAMES = (
    "angle_knee_L", "angle_knee_R", "angle_hip_L", "angle_hip_R",
    "angle_elbow_L", "angle_elbow_R", "angle_shoulder_L", "angle_shoulder_R",
    "angle_ankle_L", "angle_ankle_R", "angle_trunk_L", "angle_trunk_R",
    "angle_arm_L", "angle_arm_R", "angle_head_tilt",
)

# Number of sampled frames (phases) per session: stance, trigger, backlift
# start, full backlift, downswing, contact, follow-through.
SEQ_LEN = 7

# The 7 canonical phase names, in temporal order, in the BARE convention
# ("01_stance") -- the format the current zero_storage_pipeline writes and
# the fixed point of feature_engineering.normalize_frame_name (older data
# used "frame_01_stance.jpg"; normalize_frame_name maps both to this form).
# Single source of truth (audit H6): independent copies of this list are how
# the two conventions diverged in the first place -- one stale copy in the
# inference tensor log kept WRITING the old format long after ingestion
# switched, re-seeding the exact mixed-convention state that once NaN'd 42%
# of the dataset. A tuple for the same immutability reason as
# FEATURE_BASE_NAMES above; call list() at pandas reindex sites.
CANONICAL_FRAME_NAMES = (
    "01_stance", "02_trigger", "03_backlift_start", "04_full_backlift",
    "05_downswing", "06_contact", "07_followthrough",
)

# NOT a tuple, unlike FEATURE_BASE_NAMES above: this is passed directly to
# pandas column selection (e.g. `df[EXPECTED_FEATURES]`) in several
# consumers, and pandas treats a tuple indexer differently from a list
# indexer in some contexts -- converting this one risks a silent behavior
# change. Treat it as read-only by convention instead: no consumer may
# mutate it in place.
EXPECTED_FEATURES = list(FEATURE_BASE_NAMES) + [c + "_vel" for c in FEATURE_BASE_NAMES]
