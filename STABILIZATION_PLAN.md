# STABILIZATION_PLAN.md — Roadmap to Production/Interview Readiness

**Date:** 2026-07-20
**Inputs:** [PROJECT_UNDERSTANDING.md](PROJECT_UNDERSTANDING.md), [PROJECT_AUDIT.md](PROJECT_AUDIT.md) (issue IDs referenced below: C=Critical, H=High, M=Medium, L=Low).
**Constraints honored:** no new features, no redesigns, incremental changes, preserve existing behavior unless the behavior *is* the bug.

**Ordering principle:** crash-class bugs → silent-data-corruption bugs → serving correctness → robustness/observability → performance → hardening → consolidation/hygiene → tests & docs. Each milestone is independently shippable and leaves the repo strictly better; nothing later depends on skipping verification of something earlier.

---

## Milestone 0 — Safety Net (do first, ~half a day)

**Goal:** make every later change verifiable and reversible before touching logic.
**Work:**
- Snapshot current outputs as regression fixtures: run the existing unit tests, record `cross_validate.py` current numbers, save one known-good `keypoints.csv` session + its `dataset_angles.csv` rows as small test fixtures (sanitized).
- Add a `run_tests.bat`/one-liner that runs *all* existing unittest files from the correct directories (they currently require per-directory invocation knowledge).
- Write `DATA_LINEAGE.md` (M9): one table of every CSV/keras artifact — producer script, inputs, "current production?" flag. Documentation only; no file moves yet.
**Files:** new files only.
**Difficulty:** Low. **Impact:** High (everything downstream gets a verification baseline). **Risks:** none — no behavior changes.

## Milestone 1 — Crash-Class Bug Fixes (C4, C3-guard, H7)

**Goal:** no designed-for failure path crashes.
**Work:**
1. `predict_scores` failure return → 4-tuple, + regression test (C4).
2. `extract_keypoints_in_memory`: carry phase index with each collected frame; failed `cap.read()` → `None` slot handled by the *existing* interpolation rules (reuse, not new logic), with a `FRAME_READ_FAILED` log line; assert length-7 before the CSV write. Unit test with a mocked short-read (C3).
3. `analyze_stance` player lookup: `(id, academy)` get-or-404 instead of `get_or_create` on a client PK; test for the cross-academy id case (H7).
**Files:** `dataset/inference_service.py`, `dataset/zero_storage_pipeline.py`, `backend/backend/api/views.py`, test files.
**Difficulty:** Low. **Impact:** High. **Risks:** the C3 fix changes rejection accounting for shots that previously "succeeded" misaligned — that is the point; verify with the M0 fixture that a normal session's output is byte-identical.

## Milestone 2 — Ingestion Integrity: Idempotency + Failure Visibility (H1, M6, H10-partial)

**Goal:** re-running the batch pipeline is always safe; failures are diagnosable from one log.
**Work:**
1. At batch start, load existing session names from `keypoints.csv`; skip already-ingested rows with a clear log line; `--force` to override (H1).
2. One-off `audit_duplicates.py` report on the current `keypoints.csv` (counts, affected sessions) — report only, no deletion without your sign-off.
3. Replace the bare `except:` in `_parse_time` with `(TypeError, ValueError)`; STEP 4 catch-all logs traceback via `log_rejection` (M6).
4. Move the `keypoints.csv` bootstrap/header-check and `log_rejection` paths to module-directory-anchored paths (`__file__`-relative), executed from `run_zero_storage_pipeline()` rather than at import (the file-creation half of H10; detector half comes in M5).
**Files:** `dataset/zero_storage_pipeline.py`, new audit script, tests.
**Difficulty:** Low-Medium. **Impact:** High (kills the proven duplicate-ingestion corruption class). **Risks:** path anchoring must keep working for people who *do* run from `dataset/` — covered by keeping paths identical when cwd == module dir; test both.

## Milestone 3 — Serving-Path Correctness: Close the Train/Serve Skew (C2, M3, H6)

**Goal:** the Django product scores inputs from the same distribution the model trained on. This is the highest-value correctness milestone for the product.
**Work (incremental, reusing existing pipeline code — no new capability):**
1. Route `run_advanced_inference` frame processing through `extract_features_from_image_array` (topology validation + interpolation triage) exactly as `inference_service.extract_video_tensor` already does — replacing the copy-a-neighbor imputation (C2 validation half, M3).
2. Add an upload-duration guard consistent with the model's assumption (reject/warn on clips implausibly long for a single shot; threshold from the real shot-window distribution in the rejection logs). This is a constraint, not a feature: it makes the existing contract explicit.
3. Decide (your call, flagged for approval): (a) reuse `find_shot_windows` at serve time (adds NVIDIA dependency + latency to the product path), or (b) ship 1+2 only, documenting that uploads must be pre-trimmed single shots (the frontend upload UI already frames it this way). Recommendation: (b) now, (a) later if product needs it.
4. Fix the tensor-log frame naming to canonical bare names, shared constant with `feature_engineering` (H6).
**Files:** `backend/backend/api/ml_service.py`, `backend/backend/api/views.py`, `dataset/inference_service.py`, shared constant in `dataset/schema.py` (name list only — stays TF-free).
**Difficulty:** Medium. **Impact:** Critical-level (product scores become defensible; direct interview material). **Risks:** validation will now *reject* some uploads the old path scored garbage for — user-visible change that is the correct behavior; the existing fallback scorer + clear error message already handle the UX. Verify with M0 fixtures + a real clip before/after.

## Milestone 4 — The Wrong-Person Defect: Contain, Measure, then Fix (C1)

**Goal:** stop unmeasured data poisoning; land a *validated* subject-selection fix (per the standing project principle: attempt the real fix, don't scope-cut — but also don't ship unvalidated, which is how Milestone 4-the-feature ended up disabled).
**Work, staged so each step lands value alone:**
1. **Measure:** dataset audit script scoring every existing session for wrong-person suspicion (bone-CV outliers, landmark-position stability vs. session median, cross-session skeleton-size consistency for the same batsman name). Output: ranked suspect list + % of dataset affected. No deletions without review.
2. **Contain:** document the single-subject-framing requirement in `batch_urls.csv` conventions; add the audit to the standard post-ingestion sequence in `DATA_LINEAGE.md`.
3. **Fix:** `num_poses=N` + selection heuristic (candidates: largest bbox, most-central, most-persistent, most-wrist-motion-in-window computed per candidate). Validate exactly as the ground-truth doc's methodology demands: landmark overlays on the known-bad `temp_ai_chunk.mp4` shots (where the correct answer is documented) *before* enabling by default, behind a flag mirroring `WRIST_SPEED_SAMPLING_ENABLED`'s pattern with the same style of regression test.
4. **Re-evaluate:** re-run `cross_validate.py` on the post-cleanup dataset; update `EVALUATION_RESULTS.md` with superseding numbers (the doc's established convention).
**Files:** `dataset/zero_storage_pipeline.py`, new audit script, `backend/backend/api/ml_service.py` (same detector options), tests, docs.
**Difficulty:** High (the validation is the work). **Impact:** Highest of any milestone for dataset trustworthiness. **Risks:** heuristic may be wrong in new ways — mitigated by the flag-gated rollout and overlay validation; audit may reveal a large poisoned fraction (better to know before the interview than during it).

## Milestone 5 — Stability Under Concurrency + Hot-Path Performance (H3, H5, M8, M7)

**Goal:** the product survives concurrent use; inference latency drops enough that synchronous serving is tolerable at demo scale.
**Work:**
1. Thread-safe model init: lock-guarded double-checked lazy load, or eager load in `AppConfig.ready()` (H3); same pattern in `inference_service.get_model` .
2. Shared `PoseLandmarker` singleton passed into `extract_features_from_image_array` (parameter with a default — zero call-site churn) (H5; completes H10).
3. Batch MC-Dropout: tile input ×30, single forward pass; assert statistical equivalence (mean/std tolerance test vs. loop implementation) before swapping. Same batching for Expected Gradients' step loop (M8).
4. Sequential-read frame collection replacing per-index `cap.set` seeks in `extract_keypoints_in_memory` + `extract_video_tensor` (M7), verified frame-identical on a fixture video.
**Files:** `backend/backend/api/ml_service.py`, `backend/backend/api/apps.py`, `dataset/inference_service.py`, `dataset/zero_storage_pipeline.py`, `dataset/academic_scripts/mc_dropout_inference.py`, tests.
**Difficulty:** Medium. **Impact:** High (order-of-magnitude request latency cut; removes a whole failure class). **Risks:** batched dropout equivalence must be *tested*, not assumed (the equivalence-test-first pattern this repo already uses); eager loading slows server start — acceptable, note in runbook. Defer true async/queue serving (H4) — with latency cut ~10×, synchronous is acceptable for the interview demo; the `status` field and a worker split become a documented "next step" rather than pre-interview scope.

## Milestone 6 — API Hardening (H8, H9, H11, M12, M4)

**Goal:** close the auth/authz gaps that a technically-minded interviewer would probe.
**Work:**
1. `validate_password` + email validation in registration; `AnonRateThrottle` on auth endpoints (H8).
2. Explicit override serializer allow-listing the four metrics + `title`/`bonus_insight`; recompute `overall_score` after override; journal unchanged (H9).
3. Frontend: refresh-then-retry on 401 before logout (H11).
4. DRF pagination on session lists (M12). 5. `round()` not `int()` for scores (M4 — one-line, note it shifts persisted values by ≤1).
**Files:** `backend/backend/api/auth_views.py`, `serializers.py`, `views.py`, `settings.py`, `Frontend/src/lib/api.js`, tests.
**Difficulty:** Low-Medium. **Impact:** Medium-High. **Risks:** pagination changes response shape — the frontend consumes these lists, so update both sides together and verify dashboards render.

## Milestone 7 — Consolidation & Legacy Retirement (M1, M2, M10, M11, L1–L5)

**Goal:** finish the single-source-of-truth program; one maintained serving stack; a repo a stranger can read.
**Work:**
1. Shared angle-computation module (pandas/numpy-only) imported by `ml_service`, `inference_service`, `feature_engineering`; identity/equivalence tests in the established `assertIs`/numeric-equality style (M2). Shared `landmarks_names` and canonical-frame-names constants likewise.
2. Point `live_demo.py` + `visualize_attention.py` at `model_layers.TemporalAttention` — or archive `live_demo.py` if `dataset/app.py` supersedes it (your call) (M1).
3. Gemini path decision (recommend: port `app.py`'s window call to `nvidia_client`, move `ingest_video.py` to `_archive/`); prune Gemini/google deps from requirements after confirming nothing else imports them (M10).
4. Django `LOGGING` config; `logging` over `print` in pipeline modules, keeping `pipeline_rejections.log` semantics (M11).
5. Hygiene sweep: root scratch files → `_archive/`; fix `start_app.bat` text; delete the unused `.jsx`-or-`.tsx` duplicate shadcn set (verify imports first); mark SRS v1 extract as superseded (L1–L5, L7).
**Files:** many, all mechanical; each sub-item is an independent small commit.
**Difficulty:** Medium (volume, not complexity). **Impact:** Medium direct, High for maintainability and interview-navigability. **Risks:** consolidation must be relocation-not-rewrite (the Milestone-1-established discipline: numeric-equivalence check before switching); requirement pruning verified by a fresh-venv install + test run.

## Milestone 8 — Active-Learning Loop Repair (H2) *(optional pre-interview; do if time permits)*

**Goal:** the coach-override → retrain story is true again on the maintained stack.
**Work:** management command exporting `CoachOverrideLog` (+ per-session tensors, which requires `ml_service` to log tensors the way `inference_service` already does — reuse of an existing mechanism, not a new feature) into `active_learning_retrain.py`'s existing contract. Dry-run end-to-end once.
**Files:** new management command, `backend/backend/api/ml_service.py`, `dataset/active_learning_retrain.py`.
**Difficulty:** Medium. **Impact:** Medium (story-completeness; the DB journal preserves the data meanwhile, so nothing is being lost while this waits). **Risks:** tensor logging adds a write per inference — keep the schema-check pattern `inference_service` already has.

## Milestone 9 — Test & Documentation Closure (L6, M14, M15)

**Goal:** the invariants that matter are enforced by tests someone actually runs; the docs an interviewer sees are current.
**Work:**
1. API tests: `analyze_stance` (mocked inference — success, oversized, wrong-type, ML-failure fallback), auth register/login/refresh, override journaling, dashboard scoping.
2. `extract_batsman_name` tests per known naming convention + start carrying identity explicitly in new ingestion rows (M14 — additive column, backward compatible).
3. Single test-runner entry point + (if a remote exists) a minimal CI workflow running it.
4. Runbook section: environment quirks (Application Control/cv2 retry, numba block, OneDrive caveat — recommend moving live data dir out of sync) (M15); refresh `PROJECT_UNDERSTANDING.md` diagrams if Milestones 3–5 changed flows; one-page interview architecture cheat-sheet distilled from these documents.
**Difficulty:** Medium. **Impact:** High for interview-readiness specifically. **Risks:** none material.

---

## Sequencing Summary

| Order | Milestone | Theme | Difficulty | Impact | Fixes |
|---|---|---|---|---|---|
| 0 | Safety net | Verifiability | Low | High | (baseline) |
| 1 | Crash fixes | Stability | Low | High | C4, C3, H7 |
| 2 | Ingestion integrity | Data safety | Low-Med | High | H1, M6, H10½ |
| 3 | Train/serve skew | Correctness | Med | Critical | C2, M3, H6 |
| 4 | Wrong-person defect | Data trust | High | Highest | C1 |
| 5 | Concurrency + perf | Stability/perf | Med | High | H3, H5, M7, M8, H10½ |
| 6 | API hardening | Security | Low-Med | Med-High | H8, H9, H11, M12, M4 |
| 7 | Consolidation | Maintainability | Med | Med-High | M1, M2, M10, M11, L1–5 |
| 8 | Active-learning repair | Completeness | Med | Med | H2 |
| 9 | Tests + docs | Readiness | Med | High | L6, M14, M15 |

Deliberately deferred (documented, not scheduled): async/queued inference (H4 — mitigated by M5's latency cut), Postgres/object-storage migration (M13), NVIDIA two-pass window refinement (M5-audit), httpOnly-cookie auth storage (L3-adjacent). Each is a known, stated boundary rather than an unknown.

---

**Stopping here per instructions — no code has been modified. Awaiting your approval on: (1) this milestone ordering, (2) the Milestone 3 decision (serve-time shot detection vs. documented single-shot upload contract), and (3) whether Milestone 8 makes the pre-interview cut.**
