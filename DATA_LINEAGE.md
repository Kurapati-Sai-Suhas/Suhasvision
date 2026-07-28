# DATA_LINEAGE.md — What Produced What

**Date:** 2026-07-20 (Milestone 0 of [STABILIZATION_PLAN.md](STABILIZATION_PLAN.md)).
Documentation only — nothing was moved or deleted. All files below live in `dataset/` unless noted, and all generated data is gitignored (`dataset/*.csv`, `dataset/*.keras`); only the *scripts* and the fixtures in `dataset/test_fixtures/` are under version control.

## Current production chain (the one that matters)

```
batch_urls.csv  ──▶ zero_storage_pipeline.py ──▶ keypoints.csv + labels.csv
keypoints.csv   ──▶ kinematic_validator.py ──▶ keypoints_validated.csv   (+ kinematic_valid col, bone_length_report.csv)
                ──▶ academic_scripts/stance_symmetry_confidence.py       (+ temporal_confidence col, symmetry_confidence_report.csv)
keypoints_validated.csv + labels.csv ──▶ merge.py ──▶ dataset.csv        (3 quality gates + join)
dataset.csv     ──▶ academic_scripts/feature_engineering.py ──▶ dataset_angles*.csv
dataset_angles_production.csv ──▶ academic_scripts/train_advanced_model.py ──▶ cricket_stance_advanced_v*.keras
```

**Production flags, as of 2026-07-20:**

- **Serving model:** `cricket_stance_advanced_v5.keras` — **promoted 2026-07-20**, pinned by `backend/backend/api/ml_service.py` (`CRICKET_MODEL_FILENAME`, default now `v5`; also updated in `eval_uncertainty_correlation.py`'s matching default). This supersedes the earlier "deliberately not promoted" note below: v5 was subsequently direct-measured against v4 with matched methodology (18.61 vs. 11.85 MAE on the same 136 real sessions), passed a controlled, seeded 5-fold CV comparison, and was validated through the live `ml_service.run_advanced_inference` path against real video (see `docs/SuhasVision_SRS_v2.0.docx` §9) before promotion. The Streamlit path (`inference_service.py`) globs the highest `v*.keras` on disk, which is also v5 — the two stacks now agree.
- **Production training table:** `dataset_angles_production.csv` — 130 complete sessions / 42 identities, built 2026-07-18 *after* the frame-name-normalization and default-50-label fixes. Anything trained/evaluated on earlier tables predates those fixes and is superseded.
- **Evaluation baseline to cite:** ⚠️ SUPERSEDED 2026-07-28 — see IMPROVEMENT_ROADMAP.md. The corrected, non-test-fold-selected number is **MAE 15.08 ± 3.03 / Spearman 0.145**. Previously cited: 5-fold identity-grouped CV, overall MAE **11.56 ± 1.76** vs. constant-mean baseline 13.59 (`EVALUATION_RESULTS.md`, 2026-07-18; fold-level detail in `cv_results_t1a_step7.json`). This is the number any post-change re-evaluation must be compared against.
- **Test baseline:** 86/86 unit tests passing (2026-07-20, pre-Milestone-1): 37 in `dataset/`, 46 in `dataset/academic_scripts/`, 3 Django (`backend`). Run via `run_tests.bat`.

## Full artifact table

| Artifact | Produced by | Inputs | Status |
|---|---|---|---|
| `batch_urls.csv` | hand-maintained | — | **Production input** (URL, batsman_name, angle, macro window, bowling_type). **Since 2026-07-27, `macro_start_sec`/`macro_end_sec` are optional** — leave both blank to have `zero_storage_pipeline.process_single_row` auto-scan the entire downloaded video via `nvidia_client.find_shot_windows_auto` instead of requiring a human-curated macro window first. Existing rows with real values are unaffected (identical behavior to before). See PROJECT_AUDIT.md's updated M5 entry for verification detail. |
| `keypoints.csv` | `zero_storage_pipeline.py` | batch_urls.csv, NVIDIA NIM, MediaPipe | **Production** (append-only; since Milestone 2, re-runs skip already-ingested rows unless `--force`; integrity checked read-only by `audit_duplicates.py` — 2026-07-20 audit: 42 dup-frame sessions, 62 exact-dup rows, 183 incomplete sessions, all historical). Since Milestone 4, new rows go through multi-person subject selection (`subject_selection.py`, num_poses=4, session-level track choice, reject-when-ambiguous); rows ingested BEFORE Milestone 4 have never been audited for wrong-person tracking (audit C1 history) — treat pre-2026-07-20 sessions accordingly |
| `labels.csv` | `zero_storage_pipeline.py` → `nvidia_client.label_session_frames` | same run | **Production** (historical all-50 failure rows kept in file, excluded at merge) |
| `keypoints_validated.csv` | `kinematic_validator.py` then `stance_symmetry_confidence.py` (two passes, same file) | keypoints.csv | **Production** |
| `bone_length_report.csv` | `kinematic_validator.py` | keypoints.csv | Report |
| `symmetry_confidence_report.csv` | `stance_symmetry_confidence.py` | keypoints_validated.csv | Report |
| `dataset.csv` | `merge.py` | keypoints_validated.csv + labels.csv | **Production** |
| `dataset_augmented.csv` | `augment.py` | dataset.csv | Optional branch — only enters training when explicitly passed via `--input` |
| `dataset_angles.csv` | `feature_engineering.py` (default args) | dataset.csv | Working table (197 sessions); also read by `ml_service._load_background_sequences` for Expected-Gradients baselines |
| `dataset_angles_production.csv` | `feature_engineering.py` (via `--output`) | dataset.csv (post-fix rebuild, 2026-07-18) | **Production training table** |
| `dataset_angles_production_augmented.csv` | `augment.py` + `feature_engineering.py` | production chain | Used in at least one CV run (`cv_results_t1a_step7.json`) |
| `dataset_frontview.csv`, `dataset_angles_frontview.csv` | `filter_frontview.py` (+ feature_engineering) | dataset.csv | Research subset (2026-07-09 study) |
| `anonymized_inference_tensors.csv` | `inference_service.extract_video_tensor` (Streamlit serving) | user uploads | Active-learning input (Streamlit path only). Since Milestone 3 (audit H6) new rows use the canonical bare frame names from `schema.CANONICAL_FRAME_NAMES`; pre-existing `frame_*.jpg`-style rows remain and are normalized at read time by `normalize_frame_name` |
| `coach_overrides.csv` | `dataset/app.py` (Streamlit) — also appended by the manual script `test_phase2.py` | coach UI | Active-learning input (Django writes DB `CoachOverrideLog` instead — audit H2) |
| `rule_based_scores.csv` | `rule_based_scorer.process_keypoints` | keypoints.csv | Research/baseline |
| `attention_heatmap.csv`, `lstm_predictions.csv` | `visualize_attention.py`, `train_lstm_model.py` | models | Research (LSTM path superseded) |
| `cv_results_*.txt/.json`, `ablation_results_*.txt`, `diversity_report/` | evaluation scripts | dataset_angles*.csv | Evidence artifacts — keep |
| `keypoints1.csv`, `ky.csv` | unknown (predate current pipeline) | — | **Strays** — audit L1, archive in Milestone 7; do not use |
| `*.bak_YYYYMMDD_*.csv` | manual snapshots during the 2026-07-18 data-bug rebuild | — | Frozen snapshots — keep until Milestone 7 review |
| `pipeline_rejections.log` | `zero_storage_pipeline.log_rejection` | pipeline runs | Diagnostic log (also records successes like `WRIST_SPEED_PEAK_FOUND`) |
| `cricket_stance_lstm_v1.keras` | `train_lstm_model.py` | old table | Superseded |
| `cricket_stance_advanced_v2.keras` | `train_advanced_model.py` | pre-fix table | Superseded (was serving before v4) |
| `cricket_stance_advanced_v4.keras` | `train_advanced_model.py` | pre-fix table | Superseded 2026-07-20 (was serving; direct-measured MAE 18.61 vs. v5's 11.85 on the same 136 sessions) |
| `cricket_stance_advanced_v5.keras` | `train_advanced_model.py` | dataset_angles_production.csv | **Serving (Django + Streamlit, pinned/globbed) — promoted 2026-07-20** after direct comparison, controlled 5-fold CV, and live-path validation (see `docs/SuhasVision_SRS_v2.0.docx` §9). Ignore the single-split MAE 19.31 figure in EVALUATION_RESULTS.md — noisier than the 5-fold CV number (10.85±1.95) and superseded by it |
| `dataset/test_fixtures/*.csv` | Milestone 0 snapshot | keypoints.csv + dataset_angles_production.csv | **Committed** regression fixtures (session `abhishek_frontview_10s_01`) |
| `../pose_landmarker_heavy.task` (repo root) | downloaded from Google storage (auto-download in 3 modules) | — | **Production** MediaPipe model, pinned by content at that URL |

## Rules of thumb

1. If you regenerate anything in the chain, regenerate everything *downstream* of it in the same sitting, and note it here if the "production" designation moves.
2. Never hand-edit a generated CSV — fix the producer and re-run (the choke-point pattern `merge.py` established).
3. `.bak_*` files are historical evidence, not inputs. Nothing may read them.
