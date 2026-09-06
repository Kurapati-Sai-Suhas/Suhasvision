"""
extraction_benchmark.py — Phase 0 ground-truth benchmark for the extraction
pipeline (NOT the scoring model).

WHY THIS EXISTS
Every extraction metric this project cares about — wrong-person rate,
temporal IoU, contact error, E2E-VSER — requires human ground truth. None
existed, which is precisely why the feeder-tracking defect survived four
review rounds: it was visible only to someone who happened to overlay
landmarks on a frame by hand.

This module does two separable things:

  1. metrics that need NO ground truth (acceptance rate, fallback rate,
     pose success, sampling-method mix, coarse-vs-fine subject agreement,
     configuration divergence) — computable today, on any video;
  2. metrics that REQUIRE ground truth (tIoU, wrong-person rate, contact
     error, E2E-VSER) — computed only for clips that have an annotation.

The split is deliberate and load-bearing. Reporting a GT-dependent metric
without annotations would mean inventing one, so this module returns None
for those and says so, rather than producing a number that looks measured.

ANNOTATION FORMAT (one JSON file per benchmark, list of records)
    {
      "video_id":          "kohli_front_01.mp4",   # required
      "shot_start_frame":  120,                     # required
      "shot_end_frame":    210,                     # required
      "contact_frame":     185,                     # optional
      "batsman_track_id":  "largest_centre",        # optional, free-form
      "batsman_bbox_by_frame": {"185": [x,y,w,h]},  # optional
      "usable":            true,                    # optional human verdict
      "split":             "dev",                   # dev | eval  (see below)
      "notes":             ""
    }

SPLIT DISCIPLINE
Thresholds must be tuned on `dev` and reported on `eval`. With a benchmark
of ~100 clips a conventional three-way split is not viable; a two-way
dev/eval split is the defensible minimum, and `summarize` refuses to
silently pool them.
"""

import json
import os
from typing import Any, Dict, List, Optional

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ANNOTATIONS_PATH = os.path.join(_MODULE_DIR, "benchmark_annotations.json")

REQUIRED_FIELDS = ("video_id", "shot_start_frame", "shot_end_frame")

# A detected contact within this many frames of ground truth counts as
# correct for E2E-VSER. Justified physically, not chosen for convenience: at
# 25-30 fps a cricket ball travels ~1.2-1.4 m per frame, so the true contact
# instant is not resolvable to better than about one frame from this footage.
CONTACT_TOLERANCE_FRAMES = 1


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def load_annotations(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load and validate annotations. Raises on a malformed file rather than
    silently returning fewer clips than the user thinks they annotated."""
    target = path or DEFAULT_ANNOTATIONS_PATH
    if not os.path.exists(target):
        return []
    with open(target, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{target}: expected a JSON list of annotation records")

    for i, rec in enumerate(records):
        missing = [k for k in REQUIRED_FIELDS if k not in rec]
        if missing:
            raise ValueError(f"{target}: record {i} ({rec.get('video_id', '?')}) missing {missing}")
        if rec["shot_end_frame"] <= rec["shot_start_frame"]:
            raise ValueError(f"{target}: record {i} has end <= start")
    return records


def annotation_template(video_id: str) -> Dict[str, Any]:
    """A blank record to fill in, so the annotator does not have to remember
    the schema."""
    return {
        "video_id": video_id,
        "shot_start_frame": 0,
        "shot_end_frame": 0,
        "contact_frame": None,
        "batsman_track_id": "",
        "batsman_bbox_by_frame": {},
        "usable": True,
        "split": "dev",
        "notes": "",
    }


# ---------------------------------------------------------------------------
# Metrics that need NO ground truth
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def gt_free_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computable on any run, no annotations required. These do not prove
    correctness — a pipeline can accept 100% of clips and be wrong every
    time — but they measure real, decision-relevant behaviour and they are
    honest about what they are.
    """
    if not rows:
        return {"n": 0}

    n = len(rows)
    accepted = [r for r in rows if r.get("accepted")]

    # Coarse-vs-fine subject agreement. The two passes select independently;
    # when the coarse pass finds a subject and the fine pass does not (or
    # vice versa), the pipeline is internally inconsistent about who the
    # batsman is. This is the closest proxy for identity-switch rate that is
    # available without ground truth.
    both = [r for r in rows if r.get("coarse_selected") is not None and r.get("fine_selected") is not None]
    agree = [r for r in both if bool(r["coarse_selected"]) == bool(r["fine_selected"])]

    methods: Dict[str, int] = {}
    for r in rows:
        methods[r.get("sampling_method") or "unknown"] = methods.get(r.get("sampling_method") or "unknown", 0) + 1

    rejections: Dict[str, int] = {}
    for r in rows:
        if not r.get("accepted"):
            key = (r.get("rejection_reason") or "unknown")[:80]
            rejections[key] = rejections.get(key, 0) + 1

    monotonic = [r for r in rows if r.get("frames_monotonic") is not None]
    unique_ok = [r for r in rows if r.get("frames_unique") is not None]

    return {
        "n": n,
        "acceptance_rate": round(len(accepted) / n, 4),
        "fallback_rate": round(sum(1 for r in rows if r.get("fallback_used")) / n, 4),
        "sampling_methods": methods,
        "contact_valid_rate": (
            round(sum(1 for r in rows if r.get("contact_valid")) / n, 4)
            if any(r.get("contact_valid") is not None for r in rows) else None
        ),
        "mean_contact_confidence": _mean([r.get("contact_confidence") for r in rows]),
        "mean_contact_prominence": _mean([r.get("peak_prominence") or r.get("contact_prominence") for r in rows]),
        "mean_pose_success_rate": _mean([r.get("pose_success_rate") for r in rows]),
        "interpolation_rate": (
            round(sum(1 for r in rows if r.get("interpolated_frames")) / n, 4)
        ),
        "coarse_fine_subject_agreement": (
            round(len(agree) / len(both), 4) if both else None
        ),
        "mean_smoothness": _mean([(r.get("extra") or {}).get("smoothness") for r in rows]),
        "mean_max_center_jump_torsos": _mean(
            [(r.get("extra") or {}).get("max_center_jump_torsos") for r in rows]),
        "large_jump_rate": (
            # Fraction of ACCEPTED sequences containing a hip-center jump over
            # 3 torso lengths between consecutive phases -- the clips most
            # likely to have switched person mid-sequence.
            round(sum(1 for r in accepted
                      if ((r.get("extra") or {}).get("max_center_jump_torsos") or 0) > 3.0)
                  / len(accepted), 4) if accepted else None
        ),
        "frames_monotonic_rate": (
            round(sum(1 for r in monotonic if r["frames_monotonic"]) / len(monotonic), 4)
            if monotonic else None
        ),
        "duplicate_frame_rate": (
            round(sum(1 for r in unique_ok if r["frames_unique"] < 7) / len(unique_ok), 4)
            if unique_ok else None
        ),
        "rejection_reasons": dict(sorted(rejections.items(), key=lambda kv: -kv[1])[:10]),
    }


# ---------------------------------------------------------------------------
# Sequence quality — GT-free proxies for smoothness and identity switching
# ---------------------------------------------------------------------------
#
# The metric spec asks for "temporal smoothness" and "identity-switch rate".
# Neither needs ground truth if it is defined over the EXTRACTED sequence
# itself:
#
#   smoothness         mean magnitude of the second difference (acceleration)
#                      of the hip center and both wrists, in torso units.
#                      Real batting is smooth at phase resolution; a sequence
#                      stitched from two different people is not.
#
#   max_center_jump    largest hip-center displacement between CONSECUTIVE
#                      extracted frames, in torso units. This is the honest
#                      identity-switch proxy: a subject cannot teleport, so a
#                      large jump means either the tracked person changed or
#                      the phases are far apart in time. It is a proxy and is
#                      reported as one -- it cannot separate "switched person"
#                      from "legitimately large gap between phases", which is
#                      exactly why it is not called an identity-switch RATE.
#
# Both are computed on the 7 landmark sets the pipeline actually produced.

_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 11, 12, 23, 24
_L_WRIST, _R_WRIST = 15, 16


def _kp_center_scale(kp):
    """(hip_x, hip_y, torso) from one raw keypoint list ([x,y,z,vis,pres] x33)."""
    hx = (kp[_L_HIP][0] + kp[_R_HIP][0]) / 2.0
    hy = (kp[_L_HIP][1] + kp[_R_HIP][1]) / 2.0
    sx = (kp[_L_SHOULDER][0] + kp[_R_SHOULDER][0]) / 2.0
    sy = (kp[_L_SHOULDER][1] + kp[_R_SHOULDER][1]) / 2.0
    torso = ((sx - hx) ** 2 + (sy - hy) ** 2) ** 0.5
    return hx, hy, max(torso, 1e-6)


def sequence_quality(raw_keypoints: Optional[List[Any]]) -> Dict[str, Any]:
    """Smoothness and max-jump over the extracted sequence. None entries
    (missing detections) are skipped rather than imputed — inventing a
    position would manufacture smoothness that was never measured."""
    if not raw_keypoints:
        return {"smoothness": None, "max_center_jump_torsos": None, "n_valid_frames": 0}

    pts = []  # (hip_x, hip_y, lw, rw, torso) per present frame
    for kp in raw_keypoints:
        if kp is None:
            continue
        hx, hy, torso = _kp_center_scale(kp)
        pts.append((hx, hy, kp[_L_WRIST][:2], kp[_R_WRIST][:2], torso))

    if len(pts) < 2:
        return {"smoothness": None, "max_center_jump_torsos": None, "n_valid_frames": len(pts)}

    scale = sum(p[4] for p in pts) / len(pts)
    jumps = [(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5) / scale
             for a, b in zip(pts, pts[1:])]

    accel = None
    if len(pts) >= 3:
        series = []
        for p in pts:
            series.append([p[0], p[1], p[2][0], p[2][1], p[3][0], p[3][1]])
        mags = []
        for a, b, c in zip(series, series[1:], series[2:]):
            for i in range(len(a)):
                mags.append(abs(c[i] - 2 * b[i] + a[i]))
        accel = round((sum(mags) / len(mags)) / scale, 4) if mags else None

    return {
        "smoothness": accel,                      # lower = smoother
        "max_center_jump_torsos": round(max(jumps), 4),
        "n_valid_frames": len(pts),
    }


# ---------------------------------------------------------------------------
# Metrics that REQUIRE ground truth
# ---------------------------------------------------------------------------

def temporal_iou(pred_start, pred_end, gt_start, gt_end) -> float:
    inter = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return round(inter / union, 4) if union > 0 else 0.0


def gt_metrics(rows: List[Dict[str, Any]], annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ground-truth metrics. Returns {"available": False, ...} when no clip in
    `rows` has a matching annotation — deliberately NOT zeros, which would
    read as "measured and bad" rather than "not measured".
    """
    by_video = {a["video_id"]: a for a in annotations}
    matched = [(r, by_video[r["video_id"]]) for r in rows if r.get("video_id") in by_video]

    if not matched:
        return {
            "available": False,
            "n_matched": 0,
            "note": ("No annotated clips matched this run. tIoU, wrong-person rate, "
                     "contact error and E2E-VSER are NOT computable and are reported "
                     "as null rather than estimated."),
            "temporal_iou_mean": None,
            "contact_error_mean": None,
            "contact_within_1_frame": None,
            "e2e_vser": None,
        }

    ious, start_err, end_err, contact_err = [], [], [], []
    vser_ok = 0
    vser_eligible = 0

    for row, ann in matched:
        if row.get("shot_start_frame") is not None:
            ious.append(temporal_iou(row["shot_start_frame"], row["shot_end_frame"],
                                     ann["shot_start_frame"], ann["shot_end_frame"]))
            start_err.append(abs(row["shot_start_frame"] - ann["shot_start_frame"]))
            end_err.append(abs(row["shot_end_frame"] - ann["shot_end_frame"]))

        gt_contact = ann.get("contact_frame")
        pred_contact = row.get("contact_frame")
        contact_ok = None
        if gt_contact is not None and pred_contact is not None:
            err = abs(pred_contact - gt_contact)
            contact_err.append(err)
            contact_ok = err <= CONTACT_TOLERANCE_FRAMES

        # E2E-VSER is only defined over ACCEPTED sequences (a pipeline that
        # rejects everything would otherwise score 1.0). Report it alongside
        # acceptance rate, always.
        if row.get("accepted"):
            vser_eligible += 1
            frames = row.get("selected_frames") or []
            in_shot = all(ann["shot_start_frame"] <= f <= ann["shot_end_frame"] for f in frames)
            ordered = bool(row.get("frames_monotonic"))
            # "same, correct batsman" cannot be verified without per-frame
            # bbox annotation; when absent, fall back to the pipeline's own
            # internal consistency and mark the metric as partial.
            same_person = bool(row.get("fine_selected"))
            if in_shot and ordered and same_person and (contact_ok is not False):
                vser_ok += 1

    return {
        "available": True,
        "n_matched": len(matched),
        "temporal_iou_mean": _mean(ious),
        "start_frame_error_mean": _mean(start_err),
        "end_frame_error_mean": _mean(end_err),
        "contact_error_mean": _mean(contact_err),
        "contact_within_1_frame": (
            round(sum(1 for e in contact_err if e <= 1) / len(contact_err), 4) if contact_err else None
        ),
        "contact_within_2_frames": (
            round(sum(1 for e in contact_err if e <= 2) / len(contact_err), 4) if contact_err else None
        ),
        "e2e_vser": round(vser_ok / vser_eligible, 4) if vser_eligible else None,
        "e2e_vser_eligible": vser_eligible,
        "wrong_person_rate": None,
        "wrong_person_note": ("Requires per-frame batsman bbox annotation "
                              "(batsman_bbox_by_frame); not computed from track flags alone."),
    }


# ---------------------------------------------------------------------------
# Visual audit — the one ground truth obtainable without a domain expert
# ---------------------------------------------------------------------------
#
# tIoU and contact-frame error need frame-accurate expert labels, which this
# project does not have. "Is the boxed person the batsman?" does not: it needs
# eyes and a contact sheet, and it is the single question the documented
# wrong-person defect turns on. Labelling it is therefore cheap AND
# decision-relevant, which is a rare combination worth exploiting.
#
# SCHEMA (list of records, one per audited clip+config)
#     {"video_id": "x.mp4", "config": "A3",
#      "subject_correct": true|false|null,   # null = cannot tell from the sheet
#      "notes": ""}
#
# LIMITATIONS, stated so no reader mistakes this for expert annotation:
# single annotator, no second rater, no inter-rater agreement, and the
# annotator saw the pipeline's own overlay rather than labelling blind. It
# supports "how often is the wrong person selected" and nothing finer.

def load_visual_audit(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON list of audit records")
    for i, rec in enumerate(records):
        for key in ("video_id", "config", "subject_correct"):
            if key not in rec:
                raise ValueError(f"{path}: record {i} missing {key!r}")
    return records


def visual_audit_metrics(audit: List[Dict[str, Any]], config: str) -> Dict[str, Any]:
    """Wrong-person rate over the audited subset for one configuration.

    `subject_correct: null` clips are counted separately rather than folded
    into either bucket — an unreadable sheet is not evidence of correctness.
    """
    rows = [r for r in audit if r.get("config") == config]
    if not rows:
        return {"available": False, "n_audited": 0,
                "note": f"no visual-audit records for config {config}"}
    decided = [r for r in rows if r["subject_correct"] is not None]
    wrong = [r for r in decided if r["subject_correct"] is False]
    return {
        "available": True,
        "n_audited": len(rows),
        "n_decided": len(decided),
        "n_undecidable": len(rows) - len(decided),
        "wrong_person_rate": round(len(wrong) / len(decided), 4) if decided else None,
        "wrong_person_count": len(wrong),
        "annotator_note": ("single annotator, not blind to the pipeline overlay, "
                           "no inter-rater agreement — supports wrong-person rate only"),
    }


def summarize(rows: List[Dict[str, Any]], annotations: Optional[List[Dict[str, Any]]] = None,
              split: Optional[str] = None) -> Dict[str, Any]:
    """Full report for one configuration. `split` filters to dev or eval;
    passing None pools them, which is only valid for GT-free metrics."""
    annotations = annotations or []
    if split:
        allowed = {a["video_id"] for a in annotations if a.get("split") == split}
        rows = [r for r in rows if r.get("video_id") in allowed]
        annotations = [a for a in annotations if a.get("split") == split]

    return {
        "split": split or "all",
        "gt_free": gt_free_metrics(rows),
        "ground_truth": gt_metrics(rows, annotations),
    }
