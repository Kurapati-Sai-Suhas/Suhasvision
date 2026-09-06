"""
extraction_diagnostics.py — structured, machine-readable record of what the
extraction pipeline actually did for one shot (Phase 0).

WHY THIS EXISTS
Every silent-corruption bug in this project's history (wrong-person
tracking, the frame-name NaN wipe, the default-50 labels) survived because
the pipeline emitted prose to a log file and nothing emitted queryable
structure. `pipeline_rejections.log` is good for greppable post-mortems and
bad for "what is my wrong-person rate?" — you cannot compute a rate from
sentences.

This module records ONE row per extraction attempt, including the attempts
that fail, so acceptance rate and failure-mode breakdown are computable
rather than estimated. Diagnostics are written whenever a sink is
configured; they never alter extraction behaviour and never raise into the
pipeline.
"""

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIAGNOSTICS_PATH = os.path.join(_MODULE_DIR, "extraction_diagnostics.jsonl")

# Appends come from one process today, but the Django serving path is
# threaded and shares this module — guard the write rather than discover
# interleaved JSON lines later.
_WRITE_LOCK = threading.Lock()


@dataclass
class ExtractionDiagnostics:
    """One extraction attempt, successful or not."""

    session_name: str
    video_id: str = ""
    config_name: str = "A0"          # which A/B configuration produced this

    # --- shot localization ---
    shot_start_frame: Optional[int] = None
    shot_end_frame: Optional[int] = None
    shot_duration_frames: Optional[int] = None
    fps: Optional[float] = None

    # --- subject selection (coarse pass, before contact detection) ---
    coarse_n_tracks: Optional[int] = None
    coarse_qualified: Optional[int] = None
    coarse_traversing: Optional[int] = None
    coarse_selected: Optional[bool] = None
    coarse_winner_coverage: Optional[int] = None
    coarse_winner_scale: Optional[float] = None
    coarse_winner_displacement: Optional[float] = None
    coarse_margin: Optional[float] = None
    coarse_reason: str = ""

    # --- subject selection (fine pass, on the 7 chosen frames) ---
    fine_n_tracks: Optional[int] = None
    fine_selected: Optional[bool] = None
    fine_winner_coverage: Optional[int] = None
    fine_margin: Optional[float] = None
    fine_reason: str = ""

    # --- contact detection ---
    contact_frame: Optional[int] = None
    contact_confidence: Optional[float] = None
    contact_prominence: Optional[float] = None
    contact_bilateral: Optional[float] = None
    contact_fraction: Optional[float] = None
    contact_valid: Optional[bool] = None
    contact_reason: str = ""

    # --- frame selection ---
    sampling_method: str = ""        # uniform | motion_energy | contact_anchored
    selected_frames: List[int] = field(default_factory=list)
    frames_monotonic: Optional[bool] = None
    frames_unique: Optional[int] = None
    fallback_used: bool = False
    fallback_reason: str = ""

    # --- pose quality ---
    pose_success_count: Optional[int] = None
    pose_success_rate: Optional[float] = None
    interpolated_frames: List[str] = field(default_factory=list)
    frame_read_failures: Optional[int] = None

    # --- outcome ---
    accepted: bool = False
    rejection_reason: str = ""
    elapsed_seconds: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sanitize(report: Optional[dict]) -> dict:
    """subject_selection reports may carry `winner_track`, which holds raw
    MediaPipe landmark objects — not JSON-serializable and not wanted in a
    diagnostics row. Drop it rather than letting json.dumps explode inside
    the pipeline."""
    if not report:
        return {}
    return {k: v for k, v in report.items() if k != "winner_track"}


def record_subject_selection(diag: ExtractionDiagnostics, report: Optional[dict], *, pass_name: str) -> None:
    """Copy a subject_selection report into the diagnostics row. pass_name is
    'coarse' (dense scan, drives contact detection) or 'fine' (the 7 chosen
    frames). Recording BOTH is deliberate: if they disagree about who the
    batsman is, that disagreement is itself the identity-switch signal the
    benchmark needs, and it is invisible if only one is stored."""
    r = _sanitize(report)
    selected = bool(r.get("reason") == "selected")
    if pass_name == "coarse":
        diag.coarse_n_tracks = r.get("n_tracks")
        diag.coarse_qualified = r.get("qualified")
        diag.coarse_traversing = r.get("traversing")
        diag.coarse_selected = selected
        diag.coarse_winner_coverage = r.get("winner_coverage")
        diag.coarse_winner_scale = r.get("winner_scale")
        diag.coarse_winner_displacement = r.get("winner_displacement")
        diag.coarse_margin = r.get("margin")
        diag.coarse_reason = str(r.get("reason", ""))
    else:
        diag.fine_n_tracks = r.get("n_tracks")
        diag.fine_selected = selected
        diag.fine_winner_coverage = r.get("winner_coverage")
        diag.fine_margin = r.get("margin")
        diag.fine_reason = str(r.get("reason", ""))


def record_contact(diag: ExtractionDiagnostics, event: Optional[dict]) -> None:
    if not event:
        return
    diag.contact_frame = event.get("frame_index")
    diag.contact_confidence = event.get("confidence")
    diag.contact_prominence = event.get("peak_prominence")
    diag.contact_bilateral = event.get("bilateral_agreement")
    diag.contact_fraction = event.get("peak_fraction")
    diag.contact_valid = event.get("valid")
    diag.contact_reason = str(event.get("reason", ""))


def record_frames(diag: ExtractionDiagnostics, indices: List[int], method: str) -> None:
    diag.selected_frames = [int(i) for i in indices]
    diag.sampling_method = method
    diag.frames_monotonic = all(a <= b for a, b in zip(indices, indices[1:]))
    diag.frames_unique = len(set(indices))


def write(diag: ExtractionDiagnostics, path: Optional[str] = None) -> None:
    """Append one JSONL row. Never raises into the caller — a diagnostics
    failure must not be able to take down an extraction run."""
    target = path or os.environ.get("CRICKET_DIAGNOSTICS_PATH") or DEFAULT_DIAGNOSTICS_PATH
    try:
        line = json.dumps(diag.to_dict(), default=str)
        with _WRITE_LOCK:
            with open(target, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:  # noqa: BLE001 - deliberately swallowed, see docstring
        pass


def load(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read diagnostics back. Skips malformed lines rather than failing the
    whole load — a truncated final line from an interrupted run should not
    cost you the rest of the run's data."""
    target = path or os.environ.get("CRICKET_DIAGNOSTICS_PATH") or DEFAULT_DIAGNOSTICS_PATH
    if not os.path.exists(target):
        return []
    rows = []
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
