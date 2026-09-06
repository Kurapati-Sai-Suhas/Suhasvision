"""
phase3_annotation_schema.py — the Phase-3 temporal ground-truth schema.

WHY A SCHEMA MODULE RATHER THAN AN AD-HOC DICT
Phase 2's ground truth used bare booleans and it worked, but Phase 3 asks
questions where "unknown" and "no" are different answers and conflating them
would corrupt a metric. A clip with no visible contact is not a clip with
contact at frame 0. Every field below therefore has an explicit categorical
vocabulary, and `validate_record` rejects anything outside it rather than
letting a typo become a silent data point.

DESIGN RULE: NO FALSE PRECISION
Where the video cannot support a distinction, the schema does not offer one.
The clearest case is phase structure. The pipeline uses seven phases, but
seven phases are NOT reliably distinguishable by eye on this corpus -- at
30fps a trigger movement and a settled stance are frequently the same frame,
and backlift_start vs full_backlift is a judgement call that would not
survive a second annotator. The annotation therefore uses FIVE phases
(see PHASE_LABELS), and any phase-coverage metric must be computed against
those five. Claiming seven-phase ground truth we cannot actually produce
would make every downstream phase metric unfalsifiable.

SEPARATION OF PROPOSAL AND LABEL
`proposed_contact_frame` (machine) and `contact_frame` (human) are stored as
separate fields and never merged. That is what makes proposal quality
measurable after the fact, and it is what stops the benchmark from silently
grading the proposal against itself -- the exact circularity Phase 2 had to
mitigate in its detector-anchored boxes.
"""

from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "phase3-annotation-1.0"

# --- shot coherence -------------------------------------------------------
VALID_SINGLE_SHOT = "VALID_SINGLE_SHOT"
NO_SHOT = "NO_SHOT"
MULTIPLE_SHOTS = "MULTIPLE_SHOTS"
SCENE_CUT = "SCENE_CUT"
EXCESSIVE_DURATION = "EXCESSIVE_DURATION"
INSUFFICIENT_ACTION = "INSUFFICIENT_ACTION"
AMBIGUOUS = "AMBIGUOUS"

COHERENCE_LABELS = (
    VALID_SINGLE_SHOT, NO_SHOT, MULTIPLE_SHOTS, SCENE_CUT,
    EXCESSIVE_DURATION, INSUFFICIENT_ACTION, AMBIGUOUS,
)

# Only this verdict licenses shot_start/shot_end and a contact frame. Every
# other verdict is a reason the clip cannot supply temporal ground truth, and
# is kept distinct so the validation report can say WHICH kind of unusable.
COHERENCE_USABLE = (VALID_SINGLE_SHOT,)

# --- contact --------------------------------------------------------------
CONTACT_EXACT = "EXACT"
CONTACT_ADJACENT = "CONTACT_ADJACENT"
CONTACT_AMBIGUOUS = "AMBIGUOUS"
CONTACT_NOT_VISIBLE = "NOT_VISIBLE"

CONTACT_TYPES = (CONTACT_EXACT, CONTACT_ADJACENT, CONTACT_AMBIGUOUS, CONTACT_NOT_VISIBLE)

# Types that carry a usable frame index. AMBIGUOUS and NOT_VISIBLE do not:
# an ambiguous contact has no defensible frame, and forcing one would inject
# noise into contact-error metrics while looking like data.
CONTACT_WITH_FRAME = (CONTACT_EXACT, CONTACT_ADJACENT)

# Per-type tolerance, in frames, for scoring a predicted contact as correct.
# EXACT is held to +/-1 because the ball is visible and the instant is
# resolvable. CONTACT_ADJACENT is held to +/-3 because the label itself
# localises the bat through an impact zone rather than a visible impact --
# scoring it at +/-1 would be measuring the annotator's precision, not the
# detector's.
CONTACT_TOLERANCE = {CONTACT_EXACT: 1, CONTACT_ADJACENT: 3}

# --- phases (FIVE, deliberately) -----------------------------------------
PH_STANCE = "STANCE_PREPARATION"
PH_BACKLIFT = "BACKLIFT"
PH_DOWNSWING = "DOWNSWING"
PH_CONTACT = "CONTACT_ADJACENT"
PH_FOLLOW = "FOLLOW_THROUGH"

PHASE_LABELS = (PH_STANCE, PH_BACKLIFT, PH_DOWNSWING, PH_CONTACT, PH_FOLLOW)

# How the pipeline's seven phases map onto the five annotatable ones. Used to
# score seven-slot selections against five-phase ground truth without
# pretending the extra resolution was verified.
PIPELINE_PHASE_TO_ANNOTATED = {
    "01_stance": PH_STANCE,
    "02_trigger": PH_STANCE,          # not separable by eye at 30fps
    "03_backlift_start": PH_BACKLIFT,
    "04_full_backlift": PH_BACKLIFT,  # boundary is a judgement call
    "05_downswing": PH_DOWNSWING,
    "06_contact": PH_CONTACT,
    "07_followthrough": PH_FOLLOW,
}

# --- confidence -----------------------------------------------------------
CONF_HIGH, CONF_MEDIUM, CONF_LOW = "HIGH", "MEDIUM", "LOW"
CONFIDENCE_LEVELS = (CONF_HIGH, CONF_MEDIUM, CONF_LOW)

REQUIRED_FIELDS = ("video_id", "annotator", "coherence", "contact_type")


def blank_record(video_id: str, annotator: str = "A") -> Dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "video_id": video_id,
        "annotator": annotator,
        "batsman_track_id": None,
        # shot
        "coherence": None,
        "shot_valid": None,
        "shot_start_frame": None,
        "shot_end_frame": None,
        "shot_confidence": None,
        "scene_cut": False,
        "scene_cut_frames": [],
        "multiple_shots": False,
        "no_shot": False,
        # contact
        "proposed_contact_frame": None,     # machine, never overwritten
        "contact_frame": None,              # human
        "contact_type": None,
        "contact_confidence": None,
        # phases: label -> [start_frame, end_frame]
        "phases": {},
        "notes": "",
    }


def validate_record(r: Dict[str, Any]) -> List[str]:
    """Return a list of problems. Empty list means the record is consistent.

    This enforces the internal logic of the schema, not just field presence:
    a VALID_SINGLE_SHOT without boundaries, or an AMBIGUOUS contact that
    nevertheless carries a frame, are both contradictions that would quietly
    distort a metric.
    """
    errs = []
    for f in REQUIRED_FIELDS:
        if r.get(f) in (None, ""):
            errs.append(f"missing required field {f!r}")

    coh = r.get("coherence")
    if coh is not None and coh not in COHERENCE_LABELS:
        errs.append(f"coherence {coh!r} not in {COHERENCE_LABELS}")

    ct = r.get("contact_type")
    if ct is not None and ct not in CONTACT_TYPES:
        errs.append(f"contact_type {ct!r} not in {CONTACT_TYPES}")

    for f in ("shot_confidence", "contact_confidence"):
        v = r.get(f)
        if v is not None and v not in CONFIDENCE_LEVELS:
            errs.append(f"{f} {v!r} not in {CONFIDENCE_LEVELS}")

    usable = coh in COHERENCE_USABLE
    s, e = r.get("shot_start_frame"), r.get("shot_end_frame")

    if usable:
        if s is None or e is None:
            errs.append("VALID_SINGLE_SHOT requires shot_start_frame and shot_end_frame")
        elif e <= s:
            errs.append(f"shot_end_frame ({e}) must exceed shot_start_frame ({s})")
    else:
        if s is not None or e is not None:
            errs.append(f"coherence={coh} must not carry shot boundaries "
                        "(a non-single-shot window has no defensible ones)")

    if ct in CONTACT_WITH_FRAME:
        if r.get("contact_frame") is None:
            errs.append(f"contact_type={ct} requires a contact_frame")
        elif usable and s is not None and e is not None:
            if not (s <= r["contact_frame"] <= e):
                errs.append(f"contact_frame {r['contact_frame']} outside shot [{s},{e}]")
    elif r.get("contact_frame") is not None:
        errs.append(f"contact_type={ct} must not carry a contact_frame")

    for name, span in (r.get("phases") or {}).items():
        if name not in PHASE_LABELS:
            errs.append(f"phase {name!r} not in {PHASE_LABELS}")
            continue
        if (not isinstance(span, (list, tuple))) or len(span) != 2:
            errs.append(f"phase {name} span must be [start, end]")
            continue
        if span[1] < span[0]:
            errs.append(f"phase {name} span reversed: {span}")
        if usable and s is not None and e is not None:
            if span[0] < s or span[1] > e:
                errs.append(f"phase {name} span {span} outside shot [{s},{e}]")

    if r.get("scene_cut") and not r.get("scene_cut_frames"):
        errs.append("scene_cut=True requires a non-empty scene_cut_frames list")
    return errs


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Coverage summary — which clips can support which metric."""
    import collections
    coh = collections.Counter(r.get("coherence") for r in records)
    ctt = collections.Counter(r.get("contact_type") for r in records)

    usable_shot = [r for r in records if r.get("coherence") in COHERENCE_USABLE]
    usable_contact = [r for r in records if r.get("contact_type") in CONTACT_WITH_FRAME]
    with_phases = [r for r in records if r.get("phases")]

    return {
        "schema": SCHEMA_VERSION,
        "total_clips": len(records),
        "coherence_distribution": dict(coh),
        "contact_type_distribution": dict(ctt),
        "valid_single_shot": len(usable_shot),
        "no_shot": coh.get(NO_SHOT, 0),
        "multiple_shots": coh.get(MULTIPLE_SHOTS, 0),
        "scene_cut": coh.get(SCENE_CUT, 0),
        "excessive_duration": coh.get(EXCESSIVE_DURATION, 0),
        "insufficient_action": coh.get(INSUFFICIENT_ACTION, 0),
        "ambiguous": coh.get(AMBIGUOUS, 0),
        "contact_visible_exact": ctt.get(CONTACT_EXACT, 0),
        "contact_adjacent": ctt.get(CONTACT_ADJACENT, 0),
        "contact_ambiguous": ctt.get(CONTACT_AMBIGUOUS, 0),
        "contact_not_visible": ctt.get(CONTACT_NOT_VISIBLE, 0),
        "clips_with_phase_labels": len(with_phases),
        "usable_for_shot_localization": len(usable_shot),
        "usable_for_contact_metrics": len(usable_contact),
        "usable_for_phase_metrics": len(with_phases),
    }
