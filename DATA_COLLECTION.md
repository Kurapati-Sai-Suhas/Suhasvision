# DATA_COLLECTION.md — How to Collect More Data, and How Much

**Date:** 2026-07-27. Grounded in the current real production dataset (`dataset/dataset_angles_production.csv`) and the pipeline as it exists after the automatic shot-detection fix (see `PROJECT_AUDIT.md`'s updated M5 entry and `DATA_LINEAGE.md`).

## 1. Current composition (measured, not estimated)

| Dimension | Current | Documented V1 target | Gap |
|---|---|---|---|
| Sessions | 130 | — | — |
| Unique identities | 42 | — | the real lever on model generalization (see §3) |
| Batting hand | 129 right / **1 left** (0.8%) | not explicitly targeted, but real cricket populations run ~10-20% left-handed | severely under-represented |
| Skill level | 94 Amateur/Club (72%) / 19 Youth/Academy (15%) / 17 Professional (13%) | ~40% / 30% / 30% (this project's own stated collection target) | Professional and Youth/Academy are both well under target |
| Sessions per identity | mean 3.1, median 1.5, max 20 (one identity has 20 sessions; a quarter of identities have exactly 1) | — | a few identities are heavily over-represented while most are single-sample |

## 2. How to collect (the procedure, using the automated pipeline)

1. **Find candidate footage.** YouTube/Instagram clips of front-view batting against fast bowling — matching the current V1 scope (`docs/SuhasVision_SRS_v2.0.docx` §2.7). Prioritize: left-handed batters (any skill level), Professional-level footage, and Youth/Academy footage — in that order, per the gaps in §1.
2. **Add one row per video to `dataset/batch_urls.csv`:** `url, batsman_name, angle, macro_start_sec, macro_end_sec, bowling_type`.
   - **`macro_start_sec`/`macro_end_sec` can now be left blank.** As of the shot-extraction fix, a blank macro window triggers `find_shot_windows_auto()`, which scans the entire downloaded video automatically — no need to watch it first and hand-pick a rough window. Only fill these in if you already know a tight window and want to save NVIDIA API calls (auto-scan costs roughly one call per ~20s of video; a human-provided tight window costs one call total).
   - `angle` must be `frontview` (side/back view is out of V1 scope and will be filtered out at `filter_frontview.py`).
   - `bowling_type` should be `fast` to match current scope; only add `spin`/`swing` rows if you're deliberately starting the V2 dataset (they will not enter the current production table, per the scope filters in `merge.py`/`filter_frontview.py`).
3. **Run the pipeline:** `python zero_storage_pipeline.py` from `dataset/` (add `--force` only if deliberately re-ingesting a row). It's idempotent — safe to re-run after adding new rows; already-ingested sessions are skipped automatically.
4. **Re-run the quality-gate chain** in order: `kinematic_validator.py` → `academic_scripts/stance_symmetry_confidence.py` → `merge.py` → `filter_frontview.py` → `academic_scripts/feature_engineering.py --input dataset_frontview.csv --output dataset_angles_production.csv` (the exact chain in `DATA_LINEAGE.md`). Expect real attrition at each stage — see §4.
5. **Re-run cross-validation** (`cross_validate.py --input dataset_angles_production.csv --n_splits 5`, or more folds once identity count supports it — see §3) and compare the new MAE against the current baseline in `MODEL_EVALUATION.md` before deciding whether to retrain/promote a new model. Do not skip this: the project's established discipline is to promote only a model that measurably beats the current one, not the newest one (see `MODEL_EVALUATION.md` §4 for a real example of a change that was tested and rejected).

## 3. How much more is needed — concrete targets

The dataset's own evaluation history has already identified the actual bottleneck: **fold-to-fold variance in cross-validation is dominated by which one or two identities land in a held-out split, not by architecture** (documented as far back as the 12-identity/2026-07-09 snapshot, and still true at 42 identities — the current 5-fold CV's 95% confidence interval on MAE, (8.14, 13.56), is wide precisely because 5 folds means only ~8-9 identities per held-out fold). More raw sessions of the *same* 42 people would not fix this; more *identities* would.

| Target | Identities | Rationale |
|---|---|---|
| **Minimum, to properly power a stricter significance test** | ~80 (roughly 2×current) | Enables a real 8-10 fold CV with ~8-10 identities per fold. A 10-pair Wilcoxon test can reach p as low as ~0.002 if the model wins every fold, versus 5-fold's floor of 0.0625 — this is the difference between "directionally promising" and "statistically demonstrated" for an interview claim. |
| **Better, for a genuinely stronger interview story** | 150-200 | Meaningfully tightens the confidence interval on MAE and gives rank correlation (currently weak, ρ≈0.14-0.28) real room to improve — still two orders of magnitude below CricketVision's 8,540 clips, so frame this as "a credible, deliberately-scoped MVP," not "matching published work." |
| **Left-handed identities specifically** | at least 8-12 (not 1) | A tested mitigation for this exact gap (flip augmentation) was implemented and evaluated — and it measurably hurt accuracy (`MODEL_EVALUATION.md` §4). Synthetic correction doesn't work here; real left-handed footage is the only lever that has been shown not to backfire. |
| **Skill-level rebalancing** | roughly 25-30 more Professional, 20-25 more Youth/Academy sessions | To reach the project's own stated 40/30/30 target composition from the current 72/15/13 split, without needing to shed any existing Amateur/Club sessions. |

**Sessions per new identity:** 2-4 real sessions per new person is a reasonable target — enough to reduce single-recording noise for that identity without over-concentrating the dataset the way the current max-20 identity does. Prioritize breadth (more people) over depth (more clips of the same person) until the ~80-identity minimum is reached.

## 4. Expected attrition — collect more raw video than the target implies

Real, measured yield loss at each pipeline stage (not a guess):

- **Kinematic validity** (`kinematic_validator.py`, bone-length CV filter): historically flagged roughly 30-38% of front-view frames as invalid (task-tracked as needing re-tuning for front-view specifically — the threshold was originally tuned on a mixed-view sample). Budget for this being a real, non-trivial cut, not a rounding error.
- **Temporal confidence** (`stance_symmetry_confidence.py`): an additional ~2.8-15% of frames that already passed kinematic validity, depending on how strictly duplicate-`frame_name` sessions are counted (see `docs/architecture_ground_truth.md`'s Milestone 3 section for the exact historical numbers).
- **Wrong-person contamination**: `audit_duplicates.py`'s current real run found 269 affected session-entries (duplicate/incomplete) in the raw `keypoints.csv` — historical, from before the idempotent-ingestion and multi-person subject-selection fixes, so newly-collected sessions should not reproduce this class of loss, but it's a reminder that "sessions in `keypoints.csv`" and "sessions that reach the production table" are reliably different numbers.
- **Label failures**: `nvidia_client.label_session_frames` now skips (rather than fabricates) a label when the AI labeller's response is incomplete — a real, if usually small, additional attrition point.

**Practical rule of thumb:** expect to need on the order of 1.5-2× as many raw source videos as the net identity/session target in §3, given the combined effect of these filters. This is a reasoned estimate from the individual stage rates above, not an end-to-end measurement — track your own actual yield rate as you collect (the pipeline prints before/after counts at every filter stage) and refine this ratio with real numbers as you go.

## 5. What NOT to do

- **Don't use synthetic augmentation to fix the left-handed or skill-level gaps.** Already tried, already measured to hurt accuracy (`MODEL_EVALUATION.md` §4). Real data is the only lever with evidence behind it.
- **Don't skip the quality-gate chain to inflate session count.** A session that fails `kinematic_valid` or `temporal_confidence` is excluded for a real, measured reason (see `docs/architecture_ground_truth.md`); bypassing the gates reintroduces exactly the silent-corruption bugs this project spent significant effort finding and fixing.
- **Don't retrain and promote without re-running cross-validation first.** See `MODEL_EVALUATION.md`'s promotion discipline — a new dataset snapshot needs a new, controlled evaluation before any claim changes.
