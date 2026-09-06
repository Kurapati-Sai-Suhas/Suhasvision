"""
extraction_config.py — the A/B switch for Phase-1 extraction experiments.

Every configuration below is a real, runnable pipeline variant. A0 reproduces
the exact behaviour that shipped before Phase 1, so "did this help?" is a
measurement rather than an argument. Configurations are additive: each one
turns on exactly one more thing than the previous, which is what makes the
ablation attributable.

Nothing here changes behaviour by itself — `zero_storage_pipeline` reads
ACTIVE_CONFIG, which defaults to A0 (unchanged production behaviour). The
benchmark runner swaps it explicitly.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionConfig:
    name: str

    # Wire in motion_energy_phase_indices (MGSampler-style cumulative-motion
    # sampling). Implemented and unit-tested since an earlier commit but
    # never called by the production path.
    use_motion_energy: bool = False

    # THE reordering. When True, subject selection runs on the coarse scan
    # FIRST, and contact detection sees only the confirmed batsman's
    # landmarks. When False, contact detection is skipped entirely — because
    # the pre-Phase-1 alternative (running it on pose_landmarks[0]) is the
    # defect being fixed and must not be reintroduced as an experimental arm.
    subject_selection_before_contact: bool = False

    # Require detect_contact's quality gates (prominence, bilateral
    # agreement, plausible range) to pass. When False, any located peak is
    # accepted — isolating "correct person" from "quality gating" so the
    # ablation can attribute the improvement to one or the other.
    require_contact_quality: bool = False

    # How many frames the coarse scan samples across the shot window. Reuses
    # the existing WRIST_SPEED_COARSE_SAMPLES value (20) so Phase 1 does not
    # silently change scan density at the same time as everything else.
    coarse_samples: int = 20

    # Fraction of coarse frames a track must appear in to qualify as the
    # subject. MIN_TRACK_COVERAGE (4) is calibrated for a 7-frame sequence;
    # 4-of-20 would be far too lenient. 0.5 requires the batsman to be
    # present through at least half the shot — a documented starting point,
    # not a validated optimum.
    coarse_coverage_fraction: float = 0.5


# A0 — exactly what shipped before Phase 1: uniform sampling, no contact
# detection (the wrist-speed sampler was disabled), subject selection only
# on the 7 already-chosen frames.
A0_BASELINE = ExtractionConfig(name="A0")

# A1 — + motion-guided sampling. Isolates the value of wiring in code that
# already existed.
A1_MOTION_ENERGY = ExtractionConfig(name="A1", use_motion_energy=True)

# A2 — + the reordering. Contact detection runs, but only on a confirmed
# batsman track. Quality gates off, so this arm measures the reordering
# alone.
A2_SUBJECT_FIRST = ExtractionConfig(
    name="A2", use_motion_energy=True, subject_selection_before_contact=True,
)

# A3 — + contact quality gating (smoothing, bilateral agreement, prominence,
# plausible temporal range).
A3_SCORED_CONTACT = ExtractionConfig(
    name="A3", use_motion_energy=True, subject_selection_before_contact=True,
    require_contact_quality=True,
)

CONFIGS = {c.name: c for c in (A0_BASELINE, A1_MOTION_ENERGY, A2_SUBJECT_FIRST, A3_SCORED_CONTACT)}

# Production default stays A0 until the benchmark says otherwise. Overridable
# by env var so the runner does not have to monkey-patch a module global.
ACTIVE_CONFIG = CONFIGS.get(os.environ.get("CRICKET_EXTRACTION_CONFIG", "A0"), A0_BASELINE)


def get_config(name=None):
    """Resolve a config by name, falling back to the active one."""
    if name is None:
        return ACTIVE_CONFIG
    if name not in CONFIGS:
        raise ValueError(f"unknown extraction config {name!r}; known: {sorted(CONFIGS)}")
    return CONFIGS[name]
