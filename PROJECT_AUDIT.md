# PROJECT_AUDIT.md — Senior Engineering Audit

**Date:** 2026-07-20
**Method:** full read-through of every first-party source file; claims about runtime behavior verified against the code paths cited (and, where noted, against findings already proven with real data in `docs/architecture_ground_truth.md`). **No code was modified.**
**Companion documents:** [PROJECT_UNDERSTANDING.md](PROJECT_UNDERSTANDING.md), [STABILIZATION_PLAN.md](STABILIZATION_PLAN.md).

Severity scale:
- **Critical** — corrupts data/results or breaks the product's core promise; fix before anything else.
- **High** — real bug or reliability/security hole with a concrete failure scenario.
- **Medium** — latent defect, drift risk, or meaningful performance/maintainability cost.
- **Low** — hygiene, polish, clarity.

---

## CRITICAL

### C1. Wrong-person pose tracking in the shot extraction pipeline (`num_poses=1`)
- **Files:** [dataset/zero_storage_pipeline.py](dataset/zero_storage_pipeline.py) (`extract_features_from_image_array`, module-level `detector`), [backend/backend/api/ml_service.py](backend/backend/api/ml_service.py) (`extract_landmarks`), [dataset/inference_service.py](dataset/inference_service.py)
- **Description:** Every MediaPipe `PoseLandmarker` in the repo runs with the default `num_poses=1`. The detector returns *one* person per frame with no guarantee it is the batsman. This is not theoretical: four review rounds documented in `architecture_ground_truth.md` proved with landmark overlays that on the real production input (`temp_ai_chunk.mp4`, real `find_shot_windows` output), **all 3 AI-detected shots tracked the ball feeder, not the batsman, with 7/7 "successful" pose detections each**.
- **Why it's a problem:** Sessions that pass every validation layer can contain the wrong person's kinematics. `validate_pose()` cannot catch it (a standing feeder passes the same Y-ordering a batsman does). The bone-CV filter caught it *accidentally* in 2 of the 3 real shots (a moving person's apparent proportions vary), and explicitly missed the third.
- **Impact:** Silent training-data poisoning of unknown extent (the existing `keypoints.csv` has never been audited for this); the same defect applies to product uploads with a bowler/coach/second player in frame. It also poisoned Milestone 4's validation and is why that feature is disabled.
- **Recommended solution:** (1) Raise `num_poses` and add a subject-selection heuristic — e.g. largest bounding box, most-central, most-persistent across the window, or "person whose wrists move most within the shot window" *computed across candidates*, validated against footage with known ground truth before enabling. (2) Write a one-off audit script that re-checks existing sessions (e.g. bone-length consistency + landmark-position stability vs. the session's own median) and quantifies how much of `keypoints.csv` is suspect. (3) Short-term operational mitigation: constrain `batch_urls.csv` to single-subject framing and note it in the file header.

### C2. Train/serve skew: the production serving path skips shot extraction and validation entirely
- **Files:** [ml_service.py:254](backend/backend/api/ml_service.py) (`run_advanced_inference`), vs. [dataset/zero_storage_pipeline.py](dataset/zero_storage_pipeline.py) and [dataset/inference_service.py:78](dataset/inference_service.py) (`extract_video_tensor`)
- **Description:** Training data comes from tight NVIDIA-detected shot windows, with topology validation, hard-reject triage, and physically-motivated interpolation. The Django endpoint instead `linspace`s 7 frames across the **whole uploaded file** (whatever its length or content), imputes undetected frames by *copying the nearest detected frame verbatim*, and runs the model on the result. No `validate_pose`, no `apply_pipeline_rules`, no shot windowing. The legacy Streamlit path (`inference_service.extract_video_tensor`) *does* reuse the real pipeline — the two serving stacks disagree, and the production one is the weaker.
- **Why it's a problem:** The model was trained exclusively on stance→follow-through windows of validated poses. A 30-second upload where the batting happens in seconds 4–7 yields 7 frames mostly of walking/setup — an input distribution the model has never seen — scored with false confidence. Copied-frame imputation also fabricates zero velocities (half the feature vector) for those frames.
- **Impact:** Product scores can be systematically meaningless for realistic uploads while looking authoritative; MC-Dropout variance won't reliably flag it (uncertainty is measured uncalibrated). This undermines the core product promise.
- **Recommended solution:** Route the Django path through the same building blocks the pipeline already exports: reuse `extract_features_from_image_array` (validation + interpolation) as `inference_service.extract_video_tensor` already does, and either (a) reuse `find_shot_windows` (accepting an NVIDIA dependency at serve time), or (b) require/prompt short single-shot clips and enforce a duration cap at upload, or (c) port the cheap local heuristic already in the codebase (motion-energy-based windowing) — decision for the roadmap; the skew itself is the bug.

### C3. Frame-read failure silently shifts phase alignment (and can crash the triage rules)
- **Files:** [zero_storage_pipeline.py:483](dataset/zero_storage_pipeline.py) (`extract_keypoints_in_memory` frame-collection loop), [zero_storage_pipeline.py:197](dataset/zero_storage_pipeline.py) (`apply_pipeline_rules`)
- **Description:** When `cap.read()` fails for a sampled index the loop does `continue`, so `frames_rgb` silently has fewer than 7 entries. Two consequences: (1) **phase misalignment** — `extract_features_from_image_array` treats `frames_rgb[i]` as phase *i*, so after one dropped frame every later frame is validated against the *wrong phase's* rules and written to `keypoints.csv` under the wrong `phase_label`; (2) **IndexError** — `apply_pipeline_rules` indexes `status[i+2]` for `i` in `range(5)` and the CSV writer loops `range(N_FRAMES)`, both assuming length 7. The crash is swallowed by the broad `except` in `process_single_row` ("Error during memory extraction"), so the batch continues but the shot is lost with a misleading generic message — or worse, the misaligned variant *succeeds* and writes mislabeled rows.
- **Why it's a problem:** By contrast, `inference_service.extract_video_tensor` explicitly checks `len(frames) != 7` — the pipeline's own author knew this mattered and the dataset path never got the guard. Mislabeled phases corrupt exactly the phase-conditional logic (stance-phase symmetry prior, `validate_pose` phase rules) recent milestones were built on.
- **Impact:** Low-frequency (requires a mid-window decode failure — most likely near end-of-file with rounded indices) but silent-data-corruption class when it hits.
- **Recommended solution:** Track which phase index each collected frame belongs to; on any failed read either mark that slot `None` (letting the existing interpolation rules handle it — the machinery already exists) or reject the shot with a specific `FRAME_READ_FAILED` rejection log. Add a length guard + unit test.

### C4. `predict_scores` returns a 3-tuple on failure where callers unpack 4 — the error path itself crashes
- **Files:** [inference_service.py:282](dataset/inference_service.py) (`return False, str(e), None`), [dataset/app.py:183](dataset/app.py) (`p_success, scores_or_err, variance, explanations = predict_scores(...)`)
- **Description:** The success path returns `(True, scores_dict, variance, explanations)`; the exception path returns `(False, str(e), None)`. Any inference failure raises `ValueError: not enough values to unpack` at the call site — inside Streamlit — replacing the real error message with an unpacking traceback.
- **Why it's a problem:** Error-handling code that throws masks the diagnostic it exists to deliver; this is the exact scenario (model missing/corrupt, bad tensor) where you most need the message.
- **Impact:** Every failure of the Streamlit inference path is a crash instead of the designed friendly error. One-line fix; classified Critical for the pipeline-stability goal because it is a guaranteed crash on a designed-for path.
- **Recommended solution:** `return False, str(e), None, None` (plus a regression test that calls `predict_scores` with a model-load failure forced).

---

## HIGH

### H1. Ingestion is not idempotent — re-runs append duplicate sessions
- **Files:** [zero_storage_pipeline.py](dataset/zero_storage_pipeline.py) (`run_zero_storage_pipeline`, `extract_keypoints_in_memory`)
- **Description:** `keypoints.csv`/`labels.csv` are append-only, and session names are deterministic (`batsman_angle_start_shotidx`). Re-running a batch row — after a crash, credit exhaustion, or simply re-running the CSV — appends a second full copy of each session. There is no "already processed" check.
- **Why it's a problem:** This is not hypothetical: 42 real sessions with duplicated `frame_name` rows were found during Milestone 3, and they *silently corrupted the symmetry prior and jerk computation* until a review caught it. Downstream `drop_duplicates` in `pad_group` papers over it by keeping arbitrary first rows.
- **Impact:** Data corruption risk on every operational hiccup; wasted NVIDIA credits re-labelling.
- **Recommended solution:** Before processing a row, check whether its would-be session prefix already exists in `keypoints.csv` (cheap set load at batch start) and skip with a log line; optionally a `--force` flag. Also dedupe-audit the current file once.

### H2. The active-learning loop is severed for the Django product
- **Files:** [active_learning_retrain.py:45](dataset/active_learning_retrain.py) (reads `coach_overrides.csv`), [backend/backend/api/views.py:55](backend/backend/api/views.py) (writes `CoachOverrideLog` DB rows), [dataset/app.py:250](dataset/app.py) (the only `coach_overrides.csv` writer)
- **Description:** The retraining loop's file contract (`coach_overrides.csv` + `anonymized_inference_tensors.csv`) is produced only by the legacy Streamlit path. The Django product journals overrides to the database and logs no inference tensors at all — so coach corrections made in the real product can never reach retraining.
- **Why it's a problem:** The active-learning story (a headline feature of the architecture, and an interview talking point) is functionally dead on the maintained stack; the DB journal exists but nothing consumes it.
- **Impact:** Silent feature regression; growing gap as product usage accumulates in the DB.
- **Recommended solution:** Either add an export management command (`CoachOverrideLog` + stored per-session tensors → the CSV contract) or point `active_learning_retrain.py` at the Django DB. Requires also persisting/serving-path tensor logging (currently `ml_service` discards the tensor).

### H3. Lazy model initialization is not thread-safe
- **Files:** [ml_service.py:54](backend/backend/api/ml_service.py) (`get_models`), [inference_service.py:47](dataset/inference_service.py) (`get_model`)
- **Description:** `if _extractor is not None ... return` / load / assign, with no lock. Django's runserver (and any threaded WSGI deployment) can execute two first-requests concurrently: both see `None`, both load TensorFlow model + MediaPipe detector.
- **Why it's a problem:** Double multi-hundred-MB loads (memory spike, possible GPU/DLL contention — this machine already has cv2 Application Control flakiness), and the later assignment silently discards one copy. Worst case, two requests hold different model objects mid-flight.
- **Impact:** Nondeterministic first-request failures/slowness under any concurrency.
- **Recommended solution:** A module-level `threading.Lock` with double-checked locking, or eager initialization in `AppConfig.ready()` (also removes cold-start latency from the first user request).

### H4. Synchronous, unbounded ML work inside the HTTP request
- **Files:** [views.py:67](backend/backend/api/views.py) (`analyze_stance`), [ml_service.py:301](backend/backend/api/ml_service.py) (`_run_model_inference`)
- **Description:** Each upload runs MediaPipe ×7, then 30 sequential MC-Dropout passes, then Expected Gradients (12 baselines × 6 steps = 72 tape passes) — order of 100+ model executions — inside the request/response cycle. The `AnalysisSession.status` field ("PENDING/ANALYZING/…") exists but is hardcoded to `COMPLETED`; nothing is queued.
- **Why it's a problem:** Multi-second-to-minute request times; a handful of concurrent uploads starves all workers; browser/proxy timeouts surface as failures after the work has already run.
- **Impact:** The product cannot serve more than a demo load; failure modes waste full inference runs.
- **Recommended solution:** Roadmap item, staged: first cut latency (H5/M8), then move inference to a background worker (even a simple thread + status polling honors the existing `status` field and the frontend's shape) — without redesigning the API.

### H5. A fresh `PoseLandmarker` is constructed per session inside the extraction hot path
- **Files:** [zero_storage_pipeline.py:120](dataset/zero_storage_pipeline.py) (`extract_features_from_image_array` builds `vision.PoseLandmarker.create_from_options` per call, with its own in-function imports and model-file re-check)
- **Description:** The module already builds a `detector` at import time (used only by the disabled wrist-speed scan), yet the core extraction function creates and destroys a new heavy-model detector for every 7-frame session — including on the Streamlit serving path, per user request.
- **Why it's a problem:** Model construction dominates runtime for a 7-frame job; the duplication also means two places to keep options in sync (and the function shadows the module import pattern with local imports).
- **Impact:** Significant avoidable latency per session/request; a batch of N shots pays N model loads.
- **Recommended solution:** Accept an optional detector argument (default to a shared singleton); keep the context-manager form for one-off callers if isolation is ever needed.

### H6. `anonymized_inference_tensors.csv` is written with the stale `frame_01_stance.jpg` naming convention
- **Files:** [inference_service.py:210](dataset/inference_service.py) (`canonical_frames` list), vs. [feature_engineering.py:7](dataset/academic_scripts/feature_engineering.py) (`normalize_frame_name`)
- **Description:** The exact dual-convention drift that silently NaN'd 42% of the dataset (fixed 2026-07-18 by normalizing at *read* time) is still being produced at *write* time by the active-learning tensor log. `active_learning_retrain.py` then merges this file with `dataset_angles.csv` rows using the bare convention.
- **Why it's a problem:** Re-introduces the proven-dangerous mixed-convention state into a file that feeds retraining; any future consumer that groups or reindexes by `frame_name` without normalizing repeats the original bug.
- **Impact:** Latent retraining-data corruption of the same class as the worst historical bug in this repo.
- **Recommended solution:** Write the bare canonical names (`01_stance`, …) — importable from one shared constant next to `normalize_frame_name` — and normalize once when reading the existing file.

### H7. Coach player creation uses a client-supplied primary key
- **Files:** [views.py:93](backend/backend/api/views.py) (`PlayerProfile.objects.get_or_create(id=player_id, academy=user.academy, ...)`)
- **Description:** A coach upload can name any integer `player_id`. `get_or_create` filters by `(id, academy)`: if that id exists in *another* academy, the get finds nothing and the create attempts to insert a row with a **duplicate primary key** → `IntegrityError` → unhandled 500. It also lets clients dictate PK values (breaking the sequence assumption) and probe for id existence via the error difference.
- **Impact:** Unhandled 500s on realistic coach flows; minor enumeration surface.
- **Recommended solution:** Look up players by `(id, academy)` and *404 if absent*; create new players through an explicit creation flow (or `get_or_create` on a non-PK natural key). Never write client-supplied PKs.

### H8. Registration bypasses password validation; auth endpoints are unthrottled
- **Files:** [auth_views.py:25](backend/backend/api/auth_views.py) (`create_user` without `validate_password`), [settings.py](backend/backend/backend/settings.py) (`AUTH_PASSWORD_VALIDATORS` configured but unused by this path; no DRF throttling)
- **Description:** `AUTH_PASSWORD_VALIDATORS` only applies where `validate_password()` is called — the register endpoint never calls it, so `"a"` is an accepted password. Login/register have no rate limiting. Email format is not validated (it becomes the username as-is).
- **Impact:** Trivially weak accounts; credential-stuffing/brute-force friendly login.
- **Recommended solution:** Call `django.contrib.auth.password_validation.validate_password` in `RegisterView` (returning field errors), add DRF `AnonRateThrottle` scoped to auth routes, validate email with Django's validator.

### H9. `fields='__all__'` on update serializers lets a coach PATCH arbitrary session fields
- **Files:** [serializers.py](backend/backend/api/serializers.py), [views.py:43](backend/backend/api/views.py) (`perform_update`)
- **Description:** The override flow is scoped (coach-only, own-academy queryset) but the serializer accepts every model field — including `player` (reassign a session to a different player id), `date_analyzed`-adjacent fields, `is_fallback`, `overall_score`, `confidence_variance`, `attribution_drivers`. Only the four sub-scores are journaled to `CoachOverrideLog`; everything else changes without a trace. `PlayerProfileSerializer` similarly exposes `user`/`academy`/`total_points` wherever it's ever written.
- **Impact:** Silent, unjournaled tampering with AI outputs and session ownership within an academy; undermines the traceability the override log exists to provide.
- **Recommended solution:** An explicit update serializer allow-listing the overridable metrics (+ `bonus_insight`/`title`); recompute `overall_score` server-side after overrides (currently a coach can override the four scores and leave `overall_score` inconsistent).

### H10. Import-time side effects in `zero_storage_pipeline.py` contaminate every importer
- **Files:** [zero_storage_pipeline.py:25](dataset/zero_storage_pipeline.py) (module top: `load_dotenv`, model download, detector construction, `keypoints.csv` creation/header check with possible `RuntimeError`)
- **Description:** Merely importing the module (as `inference_service`, the tests, and anything transitively touching them do) downloads a 30MB model if absent, instantiates a MediaPipe detector, and **creates or validates `keypoints.csv` in the current working directory** — raising `RuntimeError` if a mismatched-header file happens to exist there. `log_rejection` likewise writes `pipeline_rejections.log` relative to cwd.
- **Why it's a problem:** Importers inherit heavy startup cost, cwd-dependence, and stray file creation (a `keypoints.csv` appearing wherever tests run). The header `RuntimeError` at import time can brick unrelated tools.
- **Impact:** Fragile tests, slow imports, scattered artifacts; blocks any future packaging.
- **Recommended solution:** Move CSV bootstrap + detector creation into `run_zero_storage_pipeline()`/lazy accessors; anchor all data paths to the module's directory (`os.path.dirname(__file__)`), not cwd.

### H11. JWT refresh tokens are stored but never used; 401 mid-flow discards work
- **Files:** [Frontend/src/lib/api.js](Frontend/src/lib/api.js)
- **Description:** Login stores `refreshToken`, but `fetchWithAuth` responds to any 401 by wiping storage and hard-redirecting to `/`. The refresh endpoint (`/auth/login/refresh/`) is wired on the backend and never called. With SimpleJWT's default 5-minute access lifetime, users are logged out mid-session — including after waiting out a long `analyze_stance` upload.
- **Impact:** Users lose uploads/edits on token expiry; refresh infrastructure is dead weight.
- **Recommended solution:** On 401, attempt one refresh and retry the request; only then log out. (Storage location — localStorage vs. cookie — is noted in M-tier below.)

---

## MEDIUM

### M1. `TemporalAttention` consolidation has two stragglers
- **Files:** [live_demo.py:22](live_demo.py), [dataset/academic_scripts/visualize_attention.py:8](dataset/academic_scripts/visualize_attention.py)
- Milestone 1 unified four copies into `model_layers.py`; these two independent definitions remain (root demo + attention-visualizer). Same drift risk the milestone existed to kill — and `live_demo.py` also re-implements the whole angle block. Import from `model_layers` instead (or archive `live_demo.py` if superseded by `dataset/app.py`).

### M2. Angle math and landmark lists are still triplicated
- **Files:** [ml_service.py:99](backend/backend/api/ml_service.py), [inference_service.py:148](dataset/inference_service.py), [feature_engineering.py:21](dataset/academic_scripts/feature_engineering.py); `landmarks_names` in 4+ files
- The 15-angle block (the *actual feature definition of the model*) exists three times, kept in sync only by comments ("epsilon must match ml_service exactly"). A one-line divergence is a silent train/serve skew. Extract a shared `compute_angles(df) -> angles_df` next to `schema.py` (schema stays TF-free; this helper is pandas/numpy-only, so it can live in a light module all three import).

### M3. Serving-path frame imputation fabricates zero-velocity features
- **Files:** [ml_service.py:277](backend/backend/api/ml_service.py)
- Undetected frames are byte-copies of a neighbor, so `diff()` velocities become exactly 0 for half the features at those steps — a pattern the training data (which interpolates, preserving motion) never contains. Subsumed by C2's fix; listed separately so the interpolation-vs-copy distinction isn't lost.

### M4. `int()` truncation of scores instead of rounding
- **Files:** [ml_service.py:342](backend/backend/api/ml_service.py) (`int(mean_scores[i])`), same for `overall_score`
- 79.96 becomes 79. A systematic −0.5 bias, and the overall score is a mean-of-truncations. Cosmetic-adjacent but it's the headline number users see; use `round()`.

### M5. Shot-window temporal resolution degrades linearly with macro-window length — **PARTIALLY FIXED (2026-07-27)**
- **Files:** [nvidia_client.py](dataset/nvidia_client.py) (`find_shot_windows`, `MAX_IMAGES_PER_PROMPT=10`, now also `build_scan_tiles`, `merge_overlapping_windows`, `find_shot_windows_auto`), [zero_storage_pipeline.py](dataset/zero_storage_pipeline.py) (`process_single_row`, `_resolve_naming_anchor`)
- Original finding stands for the **manual** macro-window path: a human-entered `macro_start_sec`/`macro_end_sec` longer than ~25s still gets coarser resolution, unchanged.
- **Fixed for the new automatic path**, which also closes the separate, larger bottleneck this same finding's cause is downstream of: `batch_urls.csv`'s `macro_start_sec`/`macro_end_sec` no longer require a human to watch each video first. Leaving them blank now triggers `find_shot_windows_auto()`, which tiles the ENTIRE downloaded video into ≤25s-wide, 5s-overlapping windows and runs the existing, unchanged `find_shot_windows()` on each tile (via a new `start_offset` parameter — zero behavior change when unused), merging detections that span a tile boundary (`merge_overlapping_windows`). Every tile is always tight regardless of total video length, so resolution no longer degrades with length on this path. Verified against real `temp_ai_chunk.mp4` (10s, 2 real NVIDIA tile calls): found 2 real, non-overlapping shots, one of which (3.32-7.32s) genuinely spanned the tile boundary at 4.0s and was correctly merged rather than fragmented or duplicated. Session naming across different auto-scanned videos is disambiguated via a stable per-URL CRC32 tag (`_resolve_naming_anchor`) rather than Python's per-process-randomized `hash()`. 15 new unit tests in `dataset/test_nvidia_client.py` cover the tiling/merge/naming arithmetic in isolation. Cost trade-off: one real NVIDIA call per ~20s of video scanned, versus one call for a human-pre-trimmed window — worth stating explicitly when budgeting free-tier credits for a large batch.

### M6. Broad exception swallowing in the batch pipeline
- **Files:** [zero_storage_pipeline.py:432](dataset/zero_storage_pipeline.py) (`_parse_time` bare `except:`), [zero_storage_pipeline.py:636](dataset/zero_storage_pipeline.py) (STEP 4 `except Exception` prints and moves on)
- The bare `except` can hide even `KeyboardInterrupt`-adjacent issues in timestamp parsing; the STEP 4 catch-all reduced C3 to a print. Narrow the exception types, and route through `log_rejection` with the traceback so failures are greppable in the same place as rejections.

### M7. OpenCV frame seeking (`CAP_PROP_POS_FRAMES`) is repeated and codec-sensitive
- **Files:** `extract_keypoints_in_memory`, `find_wrist_speed_peak_frame`, `find_shot_windows`, `extract_video_tensor`
- Per-frame random seeks re-decode from the nearest keyframe (slow) and land imprecisely on some codecs (frame-offset errors — relevant when 7 indices are only a few frames apart in a ~1 s window). A single sequential read collecting the target indices is both faster and exact. Worth doing when touching these files; not urgent alone.

### M8. MC-Dropout and Expected Gradients are sequential where they could be batched
- **Files:** [ml_service.py:331](backend/backend/api/ml_service.py), [mc_dropout_inference.py:17](dataset/academic_scripts/mc_dropout_inference.py), [ml_service.py:200](backend/backend/api/ml_service.py)
- 30 separate `model(x, training=True)` calls can be one call on the input tiled ×30 (dropout masks are independent per batch row) — ~an order of magnitude latency cut. EG's 12×6 loop can batch the interpolation steps similarly. Straight optimization, no behavior change (means/stds statistically identical).

### M9. Data lineage is CLI-argument folklore
- **Files:** `dataset/` CSVs; `merge.py`/`augment.py`/`feature_engineering.py`/trainer default args
- ~10 generated CSV variants (`dataset.csv`, `_augmented`, `_angles`, `_angles_production`, `_frontview`, `.bak_*`) with no manifest of what produced what from what. Whether augmentation is in the production model's diet is answerable only from shell history (`cv_results_t1a_step7.json` shows `_production_augmented.csv` was used at least once). One `DATA_LINEAGE.md` (or a tiny `make_dataset.py` orchestrator that runs validator → confidence → merge → features with logged inputs/outputs) removes a whole class of "which file is real?" errors — and is an interview-credibility issue besides.

### M10. Legacy Gemini path is half-alive and crashes without its key
- **Files:** [dataset/ingest_video.py:29](dataset/ingest_video.py) (`raise RuntimeError` at import if no `GEMINI_API_KEY`), [dataset/app.py:13](dataset/app.py) (imports it)
- The batch pipeline moved to NVIDIA, but the Streamlit app still imports the Gemini module, whose *import* hard-fails without a key — so the demo app crashes at startup on a machine configured only for the current stack. Also keeps 6 Google/Gemini packages pinned in requirements. Decide: port `app.py` to `nvidia_client` or archive the Gemini path with `_archive/`.

### M11. No logging configuration; the pipeline logs via `print`
- **Files:** [settings.py](backend/backend/backend/settings.py) (no `LOGGING`), `dataset/*.py` throughout
- `logger.exception` in views/ml_service goes wherever the default handler points (dev console; nowhere useful under a real server). The dataset pipeline mixes `print` and the bespoke `pipeline_rejections.log`. Minimum: a `LOGGING` dict (console + rotating file, request id), and `logging` instead of `print` in pipeline modules (keep the rejection log — it's a good structured artifact — but write it via a logger).

### M12. Dashboard queries are unpaginated full-table serializations
- **Files:** [views.py:152-184](backend/backend/api/views.py)
- Learner and coach dashboards serialize *every* session ever, every request; the coach view is per-academy-wide. Fine at demo scale, quadratic pain later. DRF pagination (or `[:N]` recent + a count) on the list endpoints.

### M13. SQLite + local media as the production posture
- **Files:** [settings.py:89](backend/backend/backend/settings.py)
- Single-writer DB under a threaded server that also does long requests (H4 amplifies lock contention). Acceptable for the interview demo; flag as a known deployment boundary (Postgres + object storage) rather than something to change now.

### M14. `extract_batsman_name` identity parsing is convention-fragile
- **Files:** [evaluation_protocol.py:32](dataset/academic_scripts/evaluation_protocol.py)
- `parts[0]` breaks for any batsman name containing an underscore, and the `youtube_dataset` special-case shows the convention has already needed patching. Jitter suffixes (`_jitterA/B`) survive only by luck of token position. Every CV/leakage guarantee rests on this function. Recommendation: carry identity as an explicit column from ingestion (`batch_urls.csv` already has `batsman_name`) instead of re-parsing it from the session string; keep the parser as fallback for historical rows, with a unit test per known convention.

### M15. Windows/OneDrive environment couplings
- **Files:** everywhere (repo lives under `OneDrive\Documents\Notes\...`); memory notes: cv2 Application Control flakiness, numba DLL block
- CSV append + `.bak` churn + model files inside an actively-synced OneDrive folder risks file-lock races and sync conflicts on the *training data itself*; Application Control intermittently blocks cv2 (server start flakiness, retry-not-bug) and permanently blocks numba (worked around well). Document these in the README/runbook; consider moving the working data directory out of OneDrive sync (Low-effort, real stability win). Not code bugs, but they will bite demos.

---

## LOW

### L1. Repo hygiene
Root scratch files (`test_eval.py`, `test_tensor.py`, `evaluate_variance.py`, `claude_roadmap_extracted.txt`, `srs_extracted.txt`, `docs_output.txt`/`srs_output.txt` in dataset), stray data (`keypoints1.csv`, `ky.csv`, `dataset/cookies.txt` — gitignored but present), `.bak_*` snapshots mixed with live data, `projectinfo.pdf` (3.5MB) and `pose_landmarker_heavy.task` (30MB) committed to git history (the `.gitignore` now excludes `dataset/*.task` but the root copy is tracked), `.claude/worktrees/` residue. Move scraps to `_archive/` or delete; consider Git LFS note already in `.gitignore` comments.

### L2. `start_app.bat` says "Ollama Vision" and tells users to run `ollama serve`
Stale provider reference from two stacks ago; actively misleading to a new user (or interviewer) launching the app.

### L3. Frontend duplicate shadcn primitives (`.jsx` **and** `.tsx` for ~45 components)
Only one set is imported; the duplicate set is dead weight and a divergence trap. Also commented-out `useAuth` imports in routes, and `localStorage` JWT storage (XSS-exposed; acceptable for demo — pair with H11 if touched).

### L4. `create_annotated_video` writes `mp4v`, which many browsers can't play
Known and commented in-code ([inference_service.py:339](dataset/inference_service.py)); fine for local demo, would surprise in the web product.

### L5. Misleading names and docstrings
`log_rejection` also logs successes (`WRIST_SPEED_PEAK_FOUND`) — it's really `log_pipeline_event`; `mc_dropout_predict`'s `scale_to_100` default differs from the Gap B comment's expectation history; `views.create` returns 405 via a bare `Response(status=405)` (fine, but DRF's `MethodNotAllowed` is the idiomatic signal); `_expected_keypoints_header` vs. `EXPECTED_HEADER` naming near-collision across files.

### L6. Tests don't cover the API layer, and nothing runs any tests automatically
`backend/api/tests.py` covers consolidation identity + one forward pass; there are zero tests for `analyze_stance`, auth, override journaling, or dashboards, and no CI config anywhere. The excellent pipeline unit tests run only when someone remembers `python -m unittest`. (Fixing this is a roadmap milestone, not a one-liner.)

### L7. The SRS v1 text describes a different product
`srs_extracted.txt` (Claude Vision, FastAPI, ffmpeg.wasm, 6 score dimensions) predates the entire implemented system. Anyone reading it as documentation — including an interviewer — will be misled. Mark it superseded explicitly; `architecture_ground_truth.md` + the V2 docx are the truth.

---

## Cross-Cutting Observations

- **The pipeline's worst historical bugs were all silent-data-corruption, not crashes** (feeder tracking, frame-name NaN wipe, default-50 labels, duplicate sessions). The system's defenses have improved at *detecting* value-level implausibility but remain weak on *identity/provenance* checks (is this the right person? has this session been ingested before? which file fed this model?). C1/H1/M9 are the same theme.
- **Where a single source of truth was established, it stuck** (schema, attention layer, eval protocol — all with identity tests). Where consolidation stopped short (angle math, landmark lists, canonical frame names, serving stacks), each seam has either already caused an incident or mirrors one that did. Finishing that program is the highest-leverage maintainability work available.
- **The serving stack is the weakest link relative to the pipeline's rigor.** The dataset side has four validation layers and paired significance tests; the product side will happily score 7 uniformly-spaced frames of a 30-second clip of anything. Interview questions will probe exactly this seam.

**Issue counts:** 4 Critical · 11 High · 15 Medium · 7 Low.

*Next: [STABILIZATION_PLAN.md](STABILIZATION_PLAN.md) sequences the fixes into milestones. No code has been modified.*
