# Phase 0 + Phase 1 — Extraction Reliability Results

**Scope.** Ground-truth benchmark scaffolding, pipeline instrumentation, motion-guided
sampling, the subject-selection/contact-detection reordering, and a controlled A0–A3
ablation over the full 102-clip corpus. The Conv1D+BiLSTM+Attention scoring model was
not trained, modified, or evaluated. No new detection/tracking/pose models were added.

**Headline.** Phase 1 did **not** materially reduce wrong-person or wrong-frame errors.
Measured on the same 21 clips, the wrong-person rate is 29.4% under A0 and 26.3% under
A3 — five wrong-person selections in both. What Phase 1 did deliver is the ability to
*see* that, plus one genuine defect caught and fixed. The measurement then localises the
problem precisely: on 70 of 102 clips the detector surfaces exactly one person, so the
selection logic has nothing to choose between. The bottleneck is detection recall, not
selection policy — which changes what Phase 2 should be.

---

## 1. Files changed

**New**

| File | Purpose |
|---|---|
| `dataset/contact_detection.py` | Scored contact-event detection on a *confirmed* batsman track |
| `dataset/extraction_diagnostics.py` | Per-attempt diagnostics record + JSONL sink |
| `dataset/extraction_config.py` | The A0/A1/A2/A3 switch |
| `dataset/extraction_benchmark.py` | Metrics: GT-free, GT-dependent, sequence quality, visual audit |
| `dataset/extraction_visualizer.py` | Annotated contact sheets (single clip + corpus batch) |
| `dataset/run_extraction_experiment.py` | Ablation runner over identical clips/windows |
| `dataset/test_contact_detection.py` | 15 tests |
| `dataset/test_extraction_benchmark.py` | 14 tests |

**Modified**

| File | Change |
|---|---|
| `dataset/subject_selection.py` | `select_subject()` gained optional `min_coverage` and `return_track`; report gained `runner_up_scale`, `margin`, `min_coverage` |
| `dataset/zero_storage_pipeline.py` | Added `collect_coarse_scan()`, `select_phase_frames()`, and the distinct-phase-frame guard; `extract_features_from_image_array()` / `extract_keypoints_in_memory()` gained optional `diag`/`config` |
| `dataset/test_zero_storage_pipeline.py` | +4 tests pinning the distinct-frame guard |

**Artifacts**

`dataset/experiment_results/` — `final/diagnostics_{A0..A3}.jsonl` + `final/final_summary.json`
(authoritative, post-fix, uncontended), `clean/` (pre-fix, retained for the before/after in
§11), `sheets_A3/` (102 annotated contact sheets), `sheets_A0/` (21 paired sheets),
`visual_audit_A3.json`, `visual_audit_A0.json`.

---

## 2. What changed

**`subject_selection.py`** — two backward-compatible parameters. `min_coverage` exists
because `MIN_TRACK_COVERAGE = 4` is calibrated for a 7-frame sequence; applying 4-of-20 to
the new 20-frame coarse scan would have been far too lenient, so the coarse caller passes a
proportional value (`coarse_coverage_fraction = 0.5` → 10-of-20). `return_track` exposes the
winning track dict so contact detection consumes *exactly* the landmarks selection chose,
rather than re-deriving them and risking a different person. Existing callers pass neither
and are unaffected. `margin` and `runner_up_scale` are diagnostics only: a winner that
barely cleared the dominance bar is a different situation from one that was the only
candidate, and the benchmark needs to tell them apart.

**`zero_storage_pipeline.py`** — `collect_coarse_scan()` does one decode pass serving both
downstream needs (multi-person detection *and* the grayscale motion profile). Failed reads
keep their position as `None` slots rather than shortening the list, because a shortened
list shifts every later frame onto the wrong phase.

`select_phase_frames()` implements the reordering and the fallback chain
(`contact_anchored → motion_energy → uniform`), every fallback logged and recorded.

**The distinct-phase-frame guard** (found by this benchmark, see §11). Both index producers
guarantee *monotonic* indices; neither guaranteed *distinct* ones.
`redistribute_phase_indices` clusters all pre-contact phases onto one frame when contact
lands near a window edge, and `motion_energy_phase_indices` collapses when nearly all motion
falls in one interval. An existing test had blessed the clustering as "a tight cluster, not
a crash" — true of that pure function in isolation, but this caller needs 7 *distinct*
frames: a repeated index means two phases receive identical landmarks, so every velocity
feature between them is exactly zero — a stillness that never happened, fed straight into
the `(7, 30)` tensor. The guard rejects a degenerate index set and falls through the chain
rather than emitting it. It was added at the caller, where the 7-distinct-frames requirement
actually lives, so the pure function's documented and tested contract is untouched.

**`contact_detection.py`** — returns a scored event (`frame_index, confidence,
peak_prominence, bilateral_agreement, peak_fraction, valid, reason, n_speed_samples`), not a
bare index. Wrist-speed pairs spanning a detection gap are skipped rather than bridged,
because a displacement measured across a missing detection is not a speed — bridging it
manufactures a peak exactly where tracking failed.

**`extraction_benchmark.py`** — separates metrics needing no ground truth from those that do,
returning `None` plus an explanatory note for the latter rather than a number that looks
measured. Adds `sequence_quality()` (smoothness as a second difference; max hip-centre jump
in torso units) and the visual-audit loader.

---

## 3. Baseline architecture (what was actually running before)

Verified by reading the execution path, not assumed from the research report:

```
shot window (NVIDIA VLM)
  └─> uniform 7-frame sampling
        indices = [start + (window * i / 6) for i in range(7)]
  └─> collect_phase_frames  (7 RGB frames)
  └─> extract_features_from_image_array
        ├─ detect up to 4 candidates per frame
        ├─ select_subject  ← the ONLY subject decision, on the 7 frames
        ├─ validate_pose per phase
        └─ apply_pipeline_rules (interpolation limits)
  └─> (7, 30) feature tensor
```

Two facts confirmed in the repository before any changes:

- `motion_energy_phase_indices()` was defined and unit-tested but **never called** by the
  production path. It was dead code.
- `find_wrist_speed_peak_frame()` consumed `detection_result.pose_landmarks[0]` — whoever
  the detector happened to return first, with **no subject selection at all**. This is the
  documented feeder-tracking defect. It was disabled rather than fixed, so the shipped
  baseline had no contact detection.

## 4. Phase-1 architecture (what runs now)

```
shot window
  └─> collect_coarse_scan (20 frames, ONE decode pass)
        ├─ candidates per frame  ─┐
        └─ grayscale motion       │
  └─> SUBJECT SELECTION ──────────┘   (coarse pass, min_coverage = 10/20)
  └─> contact detection            ← sees ONLY the confirmed batsman's landmarks
  └─> frame selection: contact_anchored → motion_energy → uniform
        └─ distinct-phase-frame guard on each candidate
  └─> collect_phase_frames (7 RGB frames)
  └─> extract_features_from_image_array
        └─ select_subject (fine pass, recorded separately)
  └─> (7, 30) feature tensor        ← unchanged
```

The ordering is the substantive change. Contact detection can no longer see a person that
subject selection has not already confirmed, so the specific `pose_landmarks[0]` failure is
structurally impossible. Whether that *changed outcomes* is a separate question — §9 and
Question C.

---

## 5. Experiments

| Config | Motion energy | Subject-before-contact | Contact quality gating |
|---|---|---|---|
| **A0** baseline | ✗ | ✗ | ✗ |
| **A1** | ✓ | ✗ | ✗ |
| **A2** | ✓ | ✓ | ✗ |
| **A3** | ✓ | ✓ | ✓ |

A0 reproduces pre-Phase-1 behaviour exactly. Note that "contact detection on
`pose_landmarks[0]`" is deliberately **not** an experimental arm: it is the defect under
repair, and reintroducing it to measure against would have been the only way to make Phase 1
look better than it is.

**Protocol.** All 102 clips ≥100 KB from `dataset/raw_videos/`; identical clip list and
identical shot windows for every configuration; one diagnostics row per clip per config;
scoring model never loaded. Runs are reproducible: the pre-fix and post-fix runs each
produced **102/102 identical frame choices and identical accept/reject decisions** for every
configuration, confirming the pipeline is deterministic.

**Shot windows.** `--window-mode full` treats each whole clip as the shot. This removes VLM
shot-localisation error (and its per-call cost and nondeterminism) so the *downstream* stages
can be compared in isolation. It is also a real limitation — see §7 and §13.

---

## 6. Metrics

### 6.1 What is and is not computable

The corpus has **102 clips and zero expert annotations**. Metrics requiring frame-accurate
ground truth are reported as `null`, not estimated. `gt_metrics()` returns
`{"available": false, …}` with an explicit note rather than zeros, because a zero reads as
"measured and bad" instead of "not measured".

| Requested metric | Status |
|---|---|
| temporal IoU, start/end-frame error | **Not computable** — no annotated shot boundaries |
| missed-shot rate, false-shot rate | **Not computable** — same reason; also bypassed by `window-mode full` |
| batsman selection accuracy, wrong-person rate | **Measured by visual audit** (§6.4), not by expert GT |
| identity-switch rate | **Not computable as a rate**; a proxy is reported (§6.3) and it under-detects (§12) |
| ambiguous-selection rate | Computable — reported |
| contact-frame error, % within ±1 / ±2 frames | **Not computable** — no annotated contact frames |
| correct phase-order rate | Computable — reported |
| % of sequences with all 7 frames in the correct shot / same batsman | **Not computable** |
| pose success, interpolation rate, kinematic rejection rate | Computable — reported |
| temporal smoothness | Computable — reported (§6.3) |
| E2E-VSER | **Not computable** — requires the above |

### 6.2 Primary results (102 clips, post-fix, uncontended)

| Metric | A0 | A1 | A2 | A3 |
|---|---|---|---|---|
| Acceptance rate | 74.5% | 75.5% | 74.5% | 74.5% |
| Pose success rate | 0.8788 | 0.8822 | **0.8944** | 0.8915 |
| Interpolation rate | 23.5% | 26.5% | **22.5%** | 23.5% |
| Correct phase-order rate | 100% | 100% | 100% | 100% |
| Duplicate-frame clips | 0 | 0 | 0 | 0 |
| Coarse/fine subject agreement | n/a | n/a | 0.833 | 0.824 |
| Contact located (of 102) | — | — | 91 | 91 |
| Contact passing quality gates (of located) | — | — | 57.1% | 57.1% |
| Kinematic rejection rate | 15.7% | 13.7% | 11.8% | **10.8%** |
| No-subject rejection rate | 9.8% | 10.8% | 13.7% | 14.7% |
| Latency mean / median / p90 (s) | **0.60 / 0.55 / 0.78** | 2.16 / 1.96 / 2.81 | 2.16 / 2.01 / 2.81 | 2.19 / 2.05 / 2.74 |

Sampling-method mix: A0 `uniform` 102. A1 `motion_energy` 96, `uniform` 6.
A2 `contact_anchored` 75, `motion_energy` 27. A3 `contact_anchored` 52, `motion_energy` 48,
`uniform` 2.

### 6.3 Sequence quality (accepted clips)

| Config | n | Smoothness (lower=better) | 95% CI | Mean max hip jump (torsos) | Clips with jump >3 torsos |
|---|---|---|---|---|---|
| A0 | 76 | 0.4556 | (0.399, 0.512) | 0.713 | 0 |
| A1 | 77 | 0.4832 | (0.420, 0.546) | 0.699 | 1 |
| A2 | 76 | **0.4339** | (0.372, 0.496) | 0.674 | 0 |
| A3 | 76 | 0.4458 | (0.382, 0.510) | **0.669** | 1 |

### 6.4 Visual audit — wrong-person rate

The one form of ground truth obtainable without a domain expert: *"is the boxed person the
batsman?"* needs eyes and a contact sheet, and it is the exact question the documented defect
turns on. A systematic every-5th sample of 21 clips spanning all source groups was labelled
from the annotated sheets.

| | A0 | A3 |
|---|---|---|
| Audited | 21 | 21 |
| Decided | 17 | 19 |
| **Wrong-person selections** | **5** | **5** |
| **Wrong-person rate** | **29.4%** | **26.3%** |

Stratified by camera angle (A3, decided): **front 40% (4/10), side 20% (1/5), back 0% (0/4)**.

Among clips the pipeline **accepted** (i.e. what a rebuilt dataset would actually inherit),
the A3 wrong-person rate is **2/15 = 13.3%**. Kinematic validation caught 3 of the 5
wrong-person cases — the same accidental mechanism that caught 2 of 3 in the original
incident, not a designed defence.

**Limitations, stated plainly:** single annotator, not blind to the pipeline's own overlay,
no second rater, no inter-rater agreement, n=21. This supports a wrong-person rate and
nothing finer. For A0, six clips were labelled from A0 sheets directly and eleven were
inferred where both configs saw exactly one track and both selected — in that case the same
person was necessarily chosen. Four A0 clips are marked undecidable rather than guessed.

### 6.5 Why subject selection almost never chooses

| Observation | Count |
|---|---|
| Clips where the coarse scan found exactly **1** track | **70 / 102** |
| Clips where exactly **1** track *qualified* on coverage | **93 / 102** |
| Clips where 2 qualified tracks forced a dominance decision | **1 / 102** |
| Clips where the traversal ("bowler") filter disqualified anything | **2 / 102** |
| Clips where A0 and A3 fine passes both saw exactly one track | **73 / 102 (72%)** |

`SIZE_DOMINANCE_RATIO` was exercised on **one clip out of 102**. Among clips where a subject
was selected, there were **no contested winners** — every winner was the only qualified
candidate.

---

## 7. Failure examples

1. **`pro_player_front_09` — the documented feeder failure, reproduced under A3.** The
   selected subject is *bowling* at `06_contact` (ball visible in hand), with contact
   confidence **0.98**, the highest in the corpus. The batsman is the small teal figure at
   the far end of the net. Identical selection under A0 (single track in both).
2. **`pro_player_front_14` — wrong person plus identity switch.** A light-topped person in
   frames 1–2, a dark-clothed person from frame 4 onward, all one "track". Contact
   confidence 0.85 on a throwing action.
3. **`beginnerunorthodox1_front_01` — bowler selected.** The winning track has **no
   detection at all** during `01_stance` and `02_trigger`, the two phases the model scores;
   coverage 4/7. Net displacement 0.77 torsos, far under the 2.0 threshold, because the
   bowler runs *toward* the camera and depth motion barely moves the hip centre in the frame
   plane. Caught downstream by kinematic validation, not by selection.
4. **`kohli_side_16` — Phase 1 regression.** A0 correctly *rejected* (best track covers 2/7).
   A3 selected a person traversing mid-frame while the batsman stands unattended at the left
   edge.
5. **`kohli_side_06` — false rejection introduced by Phase 1.** A0 selects the correct
   batsman across all 7 frames; A3 rejects the clip. The coarse pass had selected — a
   coarse/fine disagreement.
6. **`sanjay_front _1` — the window is not a shot.** A 200-second net session; the 7
   "phases" are 30+ seconds apart and span multiple deliveries and camera positions.
7. **`pro_player_front_19` — scene cut inside the window.** Frames 1–6 are one net;
   `07_followthrough` is an entirely different net, lighting, and player.
8. **`roa_front_2` — degenerate poses.** Correct person, but collapsed squiggle skeletons at
   `01/05/07` and no pose at `03/04/06`, on a small subject in portrait 9:16 footage.

---

## 8. Visual inspection findings

The contact sheets were decisive, and they found things no aggregate metric flagged.

- **Wrong-person selection is invisible to every automated check currently in the pipeline.**
  In the worst case the pipeline produced a confident contact detection (0.98) on a bowling
  action and accepted the sequence. Acceptance rate, pose success, monotonicity and
  smoothness were all normal on that clip.
- **Contact confidence is anti-correlated with correctness.** Wrong-person clips average
  contact confidence **0.812**; correct-person clips average **0.724**. A bowling action
  produces a cleaner, more prominent, more bilaterally-agreeing wrist-speed peak than a real
  bat swing. Confidence cannot be used as a proxy for correctness — it points the wrong way.
- **Winner size is also mildly inverted.** Wrong-person winners average a 0.215 mean torso
  versus 0.190 for correct ones, consistent with the near-camera person winning. In
  front-view footage shot from the bowler's end, the batsman is the *smaller*, more distant
  figure — which is precisely the opposite of the assumption `SIZE_DOMINANCE_RATIO` encodes
  ("the filmed subject is the prominent, camera-near figure").
- **Contact-anchored redistribution places `07_followthrough` far too late.** On
  `instagram1_front_04`, contact is at t=1.07s and follow-through at t=2.97s, by which time
  the batsman has reset to stance. The last phase is pinned to the window end regardless of
  how early contact occurred.
- **Some windows contain no shot at all** (`kohli_front_01`: all 7 phases show the same
  address position across 6 seconds), and some contain several (`sanjay_front _1`).

---

## 9. Statistical comparison

**Acceptance.** McNemar exact test on the 102 paired clips, each config against A0:

| Comparison | Discordant (A0-only / other-only) | p |
|---|---|---|
| A1 vs A0 | 4 / 5 | 1.000 |
| A2 vs A0 | 6 / 6 | 1.000 |
| A3 vs A0 | 5 / 5 | 1.000 |

No configuration differs significantly from baseline in acceptance. The discordance is
churn, not improvement: roughly five clips swap in each direction.

**Smoothness.** Paired differences on clips accepted by both configs:

| Comparison | n | Mean Δ | 95% CI | Smoother on |
|---|---|---|---|---|
| A1 − A0 | 72 | +0.0152 | (−0.023, +0.053) | 31/72 (43%) |
| A2 − A0 | 70 | −0.0337 | (−0.087, +0.020) | 39/70 (56%) |
| A3 − A0 | 71 | −0.0224 | (−0.073, +0.028) | 37/71 (52%) |

Every interval crosses zero. A2 and A3 trend smoother, but the effect is not distinguishable
from noise at n≈70, and the "smoother on" proportions are barely above a coin flip.

**Wrong-person.** 5 wrong under A0 and 5 under A3 on the same 21 clips. With counts this
small no significance test is meaningful, and none is claimed. The honest reading is that
the point estimates are indistinguishable.

**Conclusion.** No metric shows a statistically defensible improvement from A1, A2, or A3
over A0. The one unambiguous, non-statistical improvement is the elimination of
duplicate-frame sequences (§11), which is a correctness fix, not an effect size.

---

## 10. Regression check

- **Feature interface unchanged.** `SEQ_LEN = 7`, `len(EXPECTED_FEATURES) = 30` →
  the tensor remains `(7, 30)`. `schema.py` was not modified.
- **A0 is bit-identical to the legacy formula.** `select_phase_frames(None, 100, 400, …, A0)`
  returns `[100, 150, 200, 250, 300, 350, 400]`, exactly matching
  `[int(start + (window * i / 6)) for i in range(7)]`.
- **Production behaviour is unchanged by default.** `ACTIVE_CONFIG` resolves to **A0** unless
  `CRICKET_EXTRACTION_CONFIG` is set, and `extract_keypoints_in_memory` uses
  `config or get_config()`.
- **All new parameters are optional.** `extract_features_from_image_array` gained only
  `diag=None`; its three existing callers (`extract_keypoints.py:68`,
  `inference_service.py:138`, `backend/backend/api/ml_service.py:411`) pass positionally and
  are unaffected. `collect_phase_frames` is unchanged.
- **The Django serving path is untouched.** `ml_service.py` imports only
  `collect_phase_frames` and `extract_features_from_image_array`, and still samples with
  `np.linspace(0, total_frames - 1, SEQ_LEN)`. Phase 1 has **not** been wired into serving —
  deliberately, since nothing here has yet earned promotion.
- **Tests: 165 pass**, up from 127 (+15 contact detection, +14 benchmark, +4 distinct-frame
  guard, minus reorganisation). No pre-existing test was weakened or deleted.

---

## 11. What worked

1. **Instrumentation and visualisation — the main deliverable.** The wrong-person defect was
   previously discoverable only by one person overlaying landmarks by hand. It is now visible
   in one glance at a contact sheet, reproducibly, across the whole corpus. Every finding in
   §7 and §8 came from that tooling.
2. **The benchmark caught a real, silent defect.** Duplicate phase frames — two phases
   receiving identical landmarks and therefore fabricated zero velocities:

   | Config | Duplicate-frame clips before | after |
   |---|---|---|
   | A1 | 6 | **0** |
   | A2 | 16 | **0** |
   | A3 | 2 | **0** |

   Acceptance was unaffected (±1 clip), and **no audited clip changed**, so the visual audit
   remains valid for the fixed build. This defect existed in code that had unit tests and had
   passed review; only corpus-level measurement surfaced it.
3. **Determinism.** Two independent full runs produced 102/102 identical frame choices and
   accept/reject decisions for all four configurations.
4. **The reordering is architecturally correct.** Contact detection now cannot see an
   unconfirmed person. The specific `pose_landmarks[0]` failure mode is gone by construction.
5. **Contact quality gating does what it claims.** 39 of the 91 located contacts fail the
   gates, for concrete and inspectable reasons — 27 "peak at window edge", 10 "flat velocity
   curve", 2 "one-handed motion". Applying the gates moves 23 clips off contact anchoring
   relative to A2; the remainder had already fallen back through the distinct-frame guard.
6. **Modest pose-quality gains.** A2/A3 improve pose success (0.879 → 0.894) and cut the
   kinematic rejection rate (15.7% → 10.8%). Small, consistent, but not significance-tested
   and not sufficient to justify promotion on its own.

## 12. What did not work

1. **Phase 1 did not reduce wrong-person error.** 29.4% (A0) vs 26.3% (A3), five wrong in
   both. On all four clips where A3 chose the wrong person *and* A0 also selected, A0 chose
   **the same** wrong person.
2. **The reordering could not help where it matters most.** Its premise is that subject
   selection has candidates to choose between. On 70/102 clips there is exactly one track, so
   "select the batsman first" degenerates to "take whoever was detected".
3. **Motion-guided sampling (A1) showed no measurable benefit.** Acceptance +1 clip
   (p=1.000), smoothness slightly *worse* (+0.0152, CI crosses zero), and it introduced the
   only two >3-torso jumps. It costs +1.56 s/clip.
4. **The bowler/traversal filter is nearly inert** — 2/102 clips. Its net-displacement
   measure is blind to motion in depth, which is exactly how a bowler approaches a
   camera-end viewpoint (`beginnerunorthodox1_front_01`: 0.77 torsos against a 2.0 threshold).
5. **The size-dominance rule encodes a false assumption** for front-view footage and was
   exercised on only 1/102 clips regardless.
6. **My own identity-switch proxy under-detects.** `max_center_jump_torsos` flagged 0–1 clips
   per config, while the visual audit found clear identity switches in
   `pro_player_front_14` and `pro_player_front_03`. A switch between two people who are
   spatially close relative to torso size produces no large jump. The proxy is reported for
   completeness but should not be trusted as an identity-switch rate — hence it is not
   labelled as one.
7. **Phase 1 introduced regressions on individual clips**: one correct rejection became a
   wrong-person selection (`kohli_side_16`), and one correct selection became a false
   rejection (`kohli_side_06`).
8. **Latency cost is real and unearned.** The coarse scan adds **+1.56 s/clip** (0.60 → 2.16
   s mean), a ~3.6× increase, for no measured reliability gain.

## 13. Remaining failure modes

| # | Failure mode | Evidence | Severity |
|---|---|---|---|
| 1 | **Detector misses the distant batsman entirely** | 70/102 clips yield 1 track; `num_poses=4` returns ~1.3 people | **Critical** — root cause |
| 2 | **Near-camera bowler/feeder selected as subject** | 5/19 audited; 40% of front-view clips | **Critical** |
| 3 | **Contact confidence rewards bowling actions** | wrong 0.812 vs correct 0.724; max 0.98 on a delivery | **High** |
| 4 | **Identity switches within one 7-frame sequence** | `pro_player_front_14`, `_03`; undetected by the jump proxy | **High** |
| 5 | **Window ≠ shot** | 5/102 clips >15 s, longest 200 s; scene cut in `pro_player_front_19` | **High** |
| 6 | **`07_followthrough` pinned to window end** | contact t=1.07 s → follow-through t=2.97 s | Medium |
| 7 | **Degenerate poses on small/portrait subjects** | `roa_front_2`, `pro_player_front_44` | Medium |
| 8 | **Coarse/fine subject disagreement** | 17–18 clips per config (agreement 0.82–0.83) | Medium |
| 9 | **Kinematic validation is the de-facto last line of defence** | caught 3/5 wrong-person cases, accidentally | Medium |

---

## 14. Recommendation for Phase 2

**Recommended: replace the person-detection stage — YOLO/RTMDet + ByteTrack — and replace
the size-dominance rule with a geometric batsman prior. Do not adopt BoT-SORT+ReID.**

The measured failure modes dictate this, in this order:

1. **Detection recall is the binding constraint.** With one candidate on 70/102 clips, no
   improvement to selection, association, or re-identification can help: there is nothing to
   select, associate, or re-identify. MediaPipe BlazePose's detector is built to find one
   prominent person and does not reliably surface the small, distant batsman even at
   `num_poses=4`. A general object detector run at adequate input resolution is the single
   change that turns "1.3 people" into a real candidate set. **This is the highest-priority
   change.**
2. **ByteTrack, not BoT-SORT+ReID.** Identity switches are real (failure mode 4) but they
   occur between few, visually distinct people over ~3-second windows. ByteTrack's
   low-confidence-association step directly addresses the observed pattern — a track dropping
   out as the subject shrinks or is occluded by netting — at a fraction of the cost. Appearance
   ReID solves crowded re-entry after long occlusion, which the corpus does not exhibit. The
   data does not justify it.
3. **RTMPose is second, not first, but is justified.** A top-down pose model on detector
   crops directly addresses failure mode 7 (degenerate poses on small subjects), which
   whole-frame BlazePose handles badly. It becomes near-free once (1) is in place, since the
   detector already supplies the boxes. Sequence it after detection+tracking is validated.
4. **Replace the size-dominance rule.** This is a policy fix, not a model, and it is cheap.
   Size dominance is both nearly-unexercised (1/102) and *inverted* for the camera-end
   footage where errors concentrate. A geometric prior — the batsman is the person at the
   crease, near the stumps, at the far end of the pitch axis — matches the actual failure
   distribution (front 40% vs back 0%). It must be validated against annotations, not tuned
   on the same clips it is reported on.

**Prerequisite, and it is not optional:** annotate a real benchmark. Every conclusion above
about *ranking* rests on a 21-clip single-annotator audit, and tIoU, contact error and
E2E-VSER remain uncomputable. Before Phase 2 is declared to have worked, ~50 clips need
frame-level labels (shot boundaries, contact frame, batsman box) split dev/eval, so
thresholds are tuned on dev and reported on eval. The schema and loader already exist in
`extraction_benchmark.py`.

**Also fix, cheaply, regardless of Phase 2:** shot segmentation must reject or split windows
that are not single shots (failure mode 5), and follow-through should be placed a bounded
interval after contact rather than pinned to the window end (failure mode 6).

**Do not promote A1, A2, or A3 to production yet.** No configuration beats A0 on any metric
at significance, and all cost ~3.6× the latency. A0 remains the default. The value delivered
by Phase 1 is the measurement apparatus and one correctness fix, not a better pipeline.

---

## Final decision rule

**Question A — Did Phase 1 materially reduce wrong-person and wrong-frame errors?**

**No.** Wrong-person: 5/17 decided (29.4%) under A0 versus 5/19 (26.3%) under A3 on the same
21 clips — indistinguishable, and on all four clips where A3 erred and A0 also selected, A0
selected *the same wrong person*. Phase 1 turned one A0 wrong-person case into a rejection
(`pro_player_front_19`), but also turned one correct A0 selection into a false rejection
(`kohli_side_06`) and one correct A0 rejection into a wrong-person selection
(`kohli_side_16`). Wrong-frame: phase order stayed 100% correct in all configs and
acceptance was unchanged (McNemar p=1.000 for every comparison); the one genuine wrong-frame
improvement is the elimination of duplicate phase frames (24 clips across A1/A2/A3 → 0),
which came from the fix the benchmark enabled rather than from the Phase-1 design.

**Question B — Did motion-guided sampling outperform uniform sampling?**

**No.** A1 versus A0: acceptance 75.5% vs 74.5% (McNemar p=1.000); paired smoothness
*worse* by +0.0152 (95% CI −0.023 to +0.053, smoother on only 31/72 clips); pose success
0.8822 vs 0.8788; interpolation rate worse (26.5% vs 23.5%). A1 also produced the only
>3-torso hip jump among accepted clips, and before the guard was added it produced duplicate
frames on 6 clips. It costs +1.56 s/clip. No measured benefit.

**Question C — Did moving subject selection before contact detection fix the demonstrated
feeder/contact failure?**

**Architecturally yes; empirically no.** Contact detection can no longer read
`pose_landmarks[0]`; it only ever sees a track subject selection confirmed. But the
demonstrated failure reproduces end-to-end: on `pro_player_front_09`, A3 confirms a subject,
and that subject is the feeder — who is then measured *bowling* and assigned contact
confidence 0.98, the corpus maximum. The reordering removed one mechanism for selecting the
wrong person while leaving the outcome unchanged, because the confirmation step itself is
unreliable when only one candidate exists. Worse, contact confidence is anti-correlated with
correctness (0.812 wrong vs 0.724 correct), so the new scoring cannot be used to catch it.

**Question D — Is the current subject-selection mechanism sufficient?**

**No, and the benchmark shows it is barely operative.** It found exactly one track on 70/102
clips and exactly one *qualified* track on 93/102. `SIZE_DOMINANCE_RATIO` was exercised on
**1 clip out of 102**, and among all selected clips there were no contested winners. The
traversal/bowler filter fired on 2/102 and is structurally blind to a bowler approaching in
depth (0.77 torsos measured against a 2.0 threshold on a clip where the bowler was
selected). Where the mechanism does engage, its core assumption is inverted: it prefers the
large near-camera figure, but in front-view footage the batsman is the small distant one —
matching the observed error distribution of front 40% / side 20% / back 0%. It is not
choosing the batsman; it is ratifying whoever was detected.

**Question E — What is the single highest-priority Phase-2 change?**

**Replace the person-detection stage with a real multi-person detector (YOLO/RTMDet) run at
sufficient resolution to find the distant batsman, paired with ByteTrack.** This is the
binding constraint: with a single candidate on 70/102 clips, every downstream improvement —
better selection policy, ReID, better pose — operates on an empty choice. Detection recall is
the precondition for all of them. The second change, cheap and to be done alongside, is
replacing size dominance with a crease/geometry batsman prior, since the current rule is both
nearly unexercised and pointed the wrong way. Neither can be *validated* until an annotated
benchmark exists, which is why annotation is the stated prerequisite rather than an
afterthought.
