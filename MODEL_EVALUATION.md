# MODEL_EVALUATION.md — Current Deployed Model, Consolidated

> ## ⚠️ SUPERSEDED, 2026-07-28 — READ THIS FIRST
>
> Every cross-validation figure in this document was produced by a protocol that
> passed the **test fold** into `model.fit(validation_data=...)` alongside
> `EarlyStopping(restore_best_weights=True)`, then scored on that same fold — so
> the reported weights were *selected* on the data being measured.
>
> Measured optimism bias on identical folds/seed: **MAE 10.85 → 15.08 (+39%)**,
> **Spearman 0.279 → 0.145**.
>
> The protocol is fixed (`evaluation_protocol.split_inner_validation`), but the
> numbers below have **not** been regenerated. Cite
> [`IMPROVEMENT_ROADMAP.md`](IMPROVEMENT_ROADMAP.md) instead until this file is
> re-run. The v4-vs-v5 *relative* comparison still holds (both measured the same
> way); the *absolute* values do not.

**Date:** 2026-07-27. **Scope:** the model actually serving both stacks right now — `cricket_stance_advanced_v5.keras` — evaluated against the currently-deployed environment default (verified directly: `ml_service.resolve_model_path()` resolves to v5 with no `CRICKET_MODEL_FILENAME` override set). Every number below cites the script/command that produced it; nothing here is estimated or reconstructed from memory.

---

## 1. What's deployed and why

| | |
|---|---|
| Serving model | `cricket_stance_advanced_v5.keras` (promoted 2026-07-20 after the comparison in §3 below) |
| Architecture | Conv1D(64) → BatchNorm → Dropout(0.2) → BiLSTM(64) → Dropout(0.3) → TemporalAttention → Dense(32) → Dense(4, sigmoid). 76,395 trainable params. |
| Training data | `dataset/dataset_angles_production.csv` — 130 sessions, 42 unique identities, 100% front-view, 100% fast bowling, zero placeholder (fallback-default-50) labels |
| Training command | `train_advanced_model.py --input dataset_angles_production.csv --model_out cricket_stance_advanced_v5.keras` |

## 2. Direct accuracy (methodology-matched, both models measured the same way)

MC-Dropout mean prediction vs. real label, no re-scaling, same 136 real sessions for both models:

| Model | Direct MAE |
|---|---|
| v4 (previous) | 18.61 |
| **v5 (deployed)** | **11.85** |

## 3. Controlled 5-fold cross-validation (the number to cite)

`cross_validate.py`-derived harness, `tf.keras.utils.set_random_seed(42)` applied identically before every fold in both runs (neither script previously fixed a training seed — added specifically so this comparison is fair), `--input dataset_angles_production.csv --n_splits 5`. Full fold-by-fold detail in `dataset/cv_results_t1a_step7.json`.

| Metric | Value |
|---|---|
| **Overall MAE** | **10.850 ± 1.952** |
| RMSE | 14.038 ± 1.872 |
| R² | 0.086 ± 0.097 |
| 95% CI (MAE, t-dist, n=5 folds) | (8.14, 13.56) |

Per-score MAE (mean across 5 folds): **Balance 10.25 · Power 9.58 · Technique 10.87 · Defence 12.70.** Defence is consistently the hardest of the four scores — true across every evaluation run in this project's history, not yet explained by a specific feature limitation.

**Baseline comparison:** beats a constant-mean predictor (13.59 MAE) in every one of the 5 folds. Paired Wilcoxon signed-rank p=0.0625 — the smallest p-value a 5-pair test can produce when every pair agrees in direction. This does not clear the conventional p<0.05 bar, and that's reported honestly rather than rounded up; a 5-for-5 directional win at this sample size is real signal, not proof at a stricter standard.

*Note on the two CV numbers in this repo's history:* `EVALUATION_RESULTS.md` separately cites 11.56 ± 1.76 for the same dataset — that number came from an earlier, **unseeded** run. 10.85 ± 1.95 above supersedes it; the difference is measurement rigor (a fixed seed makes the number reproducible and fairly comparable across models), not a real change in the data or model.

## 4. What was tried and rejected (a real negative result, kept)

Flip + Gaussian-jitter augmentation (4× the training set, same 42 identities) was implemented, leak-checked, and evaluated under the identical controlled protocol above:

| Metric | Production (baseline) | Augmented | Change |
|---|---|---|---|
| MAE | 10.850 ± 1.952 | 11.545 ± 1.854 | **+6.41% (worse)** |
| RMSE | 14.038 ± 1.872 | 14.489 ± 1.503 | +3.21% (worse) |
| R² | 0.086 ± 0.097 | -0.019 ± 0.230 | worse — turns negative |

**Not adopted.** Likely mechanism: flip/jitter clones multiply existing sessions rather than adding new identities, so with only 42 real identities the model fits those 42 more tightly (training converged in ~1/3 the epochs) without learning anything new about people it hasn't seen — worse, not better, held-out generalization. Full detail in `docs/SuhasVision_SRS_v2.0.docx` §7.5.

## 5. Live-path validation (not just an offline script)

v5 was exercised through the actual `ml_service.run_advanced_inference()` function — the same one the Django endpoint calls — against 5 real, previously-unused videos, before promotion:

- 5/5 successful inferences, 0 fallback triggers, 0 exceptions.
- Attribution output (Expected Gradients) genuinely varied by session — different sessions named different weakest-driving joints, not a fixed default.
- MC-Dropout uncertainty (`confidence_variance`) stayed in a sane 1.7–3.1 range, no collapse or explosion.
- A first-pass latency comparison made v5 look ~6× faster than v4 (5–20s vs. 32–58s); this was caught as a suspected sequential-run warm-up confound (same class of artifact caught once earlier this session) and re-tested with interleaved calls in one warm process — **v4 and v5 have no measurable latency difference** (16.38s vs. 16.48s, 0.6% apart, within noise). Both now benefit equally from the Milestone-5 batching described in §7.

## 6. Uncertainty calibration — explicitly not achieved

MC-Dropout's per-score standard deviation is real (30 stochastic forward passes, not decoration), but **not calibrated against real error**: `eval_uncertainty_correlation.py`, run against the live deployed model, found Spearman r between 0.02 and 0.08 (p > 0.28 on all four scores) between predicted uncertainty and real absolute error. It is surfaced to coaches as a qualitative High/Moderate/Low label — a rough signal of the network's own internal disagreement across stochastic passes, not a trustworthy confidence interval. Fixing this would need either a held-out calibration set with a fitted mapping (isotonic regression from raw std to empirical error) or a different uncertainty method (e.g. deep ensembles) — neither is built.

## 7. Serving-path correctness (closed since the last full audit)

As of this evaluation, the Django serving path (`ml_service.run_advanced_inference`) reuses the **same** validated frame-collection and pose-extraction functions the offline training pipeline uses (`zero_storage_pipeline.collect_phase_frames` / `extract_features_from_image_array`), including the same multi-person subject-selection logic — this closes what was, as of the 2026-07-20 audit, the single largest correctness gap between what the model was trained on and what it actually scores in production (previously: uniform whole-clip sampling with no validation, no interpolation, imputation by copying a neighbor frame verbatim). MC-Dropout and Expected Gradients now run as single batched tensor passes instead of sequential loops (~19× and ~40× faster respectively, verified numerically equivalent to the sequential versions in `tests.py`), and model loading is thread-safe (double-checked locking).

## 8. Known limitations (stated plainly)

- **42 unique identities**, only **1 left-handed** — the real ceiling on generalization claims, not architecture. See `DATA_COLLECTION.md` for concrete targets to move past this.
- **Historical raw-data contamination not yet cleaned**: `dataset/audit_duplicates.py`, re-run during this evaluation cycle, found 42 duplicate-`frame_name` sessions, 62 exact-duplicate rows, and 183 incomplete sessions still in `keypoints.csv` — all predating the idempotent-ingestion fix. These are measured, not deleted (no cleanup without explicit review, per this project's established discipline); merge.py's quality gates exclude most but not provably all of them from the actual training table.
- **Rank correlation is weak** (Spearman ρ ≈ 0.14–0.28 across scores in prior breakdowns) — the model beats a trivial baseline but the learned signal is modest, consistent with a ~130-session dataset regressing onto one AI labeller's opinions.
- **R² is small (0.086)** — the model explains only a small fraction of score variance beyond the mean; be precise about this in an interview rather than leading with MAE alone.

## 9. Bottom line

Real, honest, defensible numbers: v5 measurably and substantially outperforms the previously-deployed v4 (11.85 vs. 18.61 direct MAE), beats a trivial baseline in every fold, and passed live-path validation with no reliability or latency regression. It is a legitimate, if modest, applied-ML result appropriate for a ~130-session, small-dataset regime — not a finished product. The single highest-leverage next step is more real identities, not more architecture tuning or synthetic augmentation (already tried and rejected, §4) — see `DATA_COLLECTION.md`.
