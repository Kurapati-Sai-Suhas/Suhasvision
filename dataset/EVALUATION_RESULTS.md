**Update, 2026-07-18 (Milestone 2 — Unified Evaluation Protocol):** the numbers below were produced by `cross_validate.py` and `ablation_study.py` before this milestone unified their fold-generation logic, added Spearman/Kendall rank correlation, added a paired significance test against a trivial constant-mean baseline, and — for `ablation_study.py` specifically — fixed a real leakage gap (it grouped by de-flipped session name, not by batsman identity, so two different recordings of the same person could land in different folds). The MAE figures below are not necessarily wrong, but they were not measured under the current protocol and are not directly comparable to any future run of these scripts. Re-run `cross_validate.py` and `ablation_study.py` to produce current, protocol-consistent numbers (including rank correlation and significance testing) before citing this document as the model's current accuracy. Separately, `eval_uncertainty_correlation.py` was found to be silently evaluating the stale `cricket_stance_advanced_v2.keras` rather than the deployed `v4` model; run against the actual deployed model, MC-Dropout uncertainty shows a weak, not statistically significant correlation with real error on all four scores (r ranged 0.02-0.08, p > 0.28 on every metric) — the uncertainty estimate should not currently be treated as calibrated.

**Update, 2026-07-18 (Milestone 3 — Stance-Symmetry-Aware Temporal Confidence):** a second, complementary mandatory filter was added ahead of `merge.py` (see `docs/architecture_ground_truth.md`'s Milestone 3 notes for the full mechanism and review-fix history). Its own required ablation — `ablation_stance_symmetry_filter.py`, run against the current, much smaller labeled subset (120 sessions / 10 identities without the new filter, vs. this document's 360 sessions / 12 identities — label coverage has not caught back up to the 2026-07-09 numbers below) — found the filter costs 6 sessions of yield (120 → 114) and changes mean MAE from 8.43 ± 4.14 pts to 7.38 ± 2.38 pts (Wilcoxon p=0.4375, n=5 genuinely-paired folds — not yet statistically significant at this sample size). Not included in the tables below since it's a different experimental question (a data filter, not an architecture) on a different, smaller dataset slice — recorded here so a reader of this file knows the filter exists and where its numbers live.

**Update, 2026-07-18 (interview-deadline reprioritization) — CURRENT, definitive numbers, supersedes everything above:** two real, severe data bugs were found and fixed while rebuilding the dataset (full detail in `docs/architecture_ground_truth.md`): (1) `feature_engineering.py`'s frame-padding logic only matched one of two coexisting real `frame_name` conventions, silently wiping ALL metadata to NaN for 42% of sessions using the other; (2) `nvidia_client.py` silently defaulted any missing AI-labelling score to 50, producing 90 real label rows with all four scores exactly 50 — indistinguishable from a genuine assessment unless checked directly. Both are now fixed at the source (new labelling calls can't reproduce either), and the already-corrupted historical rows are excluded (not deleted from the raw files — filtered at `merge.py`'s existing choke point).

**Final production dataset** (`dataset_angles_production.csv`): 130 complete sessions / 42 unique identities, 100% frontview, 100% fast bowling (see `bowling_type` column, new this pass), zero placeholder scores. This is roughly 4x the identity count anything in this project has been evaluated against before (previous best was ~10-12 identities).

**5-Fold Grouped Cross-Validation** (`cross_validate.py --input dataset_angles_production.csv --n_splits 5`, identity-grouped, leak-checked every fold):

| Metric | MAE (mean ± std across 5 folds) | Mean Spearman rho |
|---|---|---|
| **Overall** | **11.56 ± 1.76** | — |
| Balance | 11.39 ± 1.40 | 0.248 |
| Power | 10.27 ± 2.07 | 0.278 |
| Technique | 10.80 ± 2.33 | 0.140 |
| Defence | 13.78 ± 2.51 | 0.216 |

Constant-mean baseline: 13.59 ± 1.44. The model beat the baseline in **every one of the 5 folds** (Wilcoxon statistic=0.000) — p=0.0625, which is the smallest p-value a 5-pair Wilcoxon signed-rank test can produce when every pair agrees in direction; it does not clear the conventional p<0.05 bar, and that is reported honestly rather than rounded up. A consistent 5-for-5 directional win is a real, meaningful signal at this sample size, not proof at a stricter standard — more identities would sharpen this. Skill-level breakdown: the model's error is lowest on Youth/Academy sessions (7.43) vs. Amateur/Club (12.23) and Professional (12.45) — plausible (less technical nuance to miss at the beginner level) but not yet explained further.

**Production model retrain** (`train_advanced_model.py --input dataset_angles_production.csv --model_out cricket_stance_advanced_v5.keras`, same architecture as deployed `v4`, single 80/20 identity-grouped held-out split, NOT cross-validated): Held-Out Validation MAE 19.31 pts. This single-split number is noisier than the 5-fold CV above by construction (which specific ~8 identities land in one 20% split matters a lot at 42 total identities) — **cite the 5-fold CV number (11.56 ± 1.76) as the model's real accuracy estimate, not this one.** `v5.keras` was saved as a new file, NOT deployed over the live `cricket_stance_advanced_v4.keras` (`CRICKET_MODEL_FILENAME` env var) — promoting it to production is a separate, explicit decision, not made here.

# Real Evaluation Results — 2026-07-09

These are the first evaluation numbers this project has ever actually run and saved.
Everything previously cited (SRS's "Overall MAE 15.45 ± 12.56", per-metric MAEs, ICC 0.87)
was either training-set error mislabeled as a generalization metric, or a hardcoded
assumption with no measurement behind it. The numbers below are real, leak-checked,
held-out results, produced by scripts that already existed in this repo
(`cross_validate.py`, `ablation_study.py`) but had never been executed —
`scikit-learn` wasn't even installed, so they couldn't have been.

Before running either script, `dataset_angles.csv` was regenerated from the current
`dataset.csv` (360 sessions) via `feature_engineering.py`. The file on disk previously
had 861 session names, 500 of which were single-frame stubs left over from an earlier,
larger `dataset.csv` that has since been filtered down — those orphaned rows are gone now.

## 3-Fold Grouped Cross-Validation (`cross_validate.py`)

360 sessions, grouped into 12 unique batsman/source identities (flip-augmented clones
kept in the same fold as their original, so there's no leakage).

| Metric | MAE (mean ± std across 3 folds) |
|---|---|
| **Overall** | **9.15 ± 0.78** |
| Balance | 8.92 ± 1.10 |
| Power | 8.82 ± 0.43 |
| Technique | 10.76 ± 1.08 |
| Defence | 8.10 ± 0.96 |

Full fold-by-fold output: `dataset/cv_results_2026-07-09.txt`.

## 5-Fold Architecture Ablation (`ablation_study.py`)

Same 360 sessions, 5-fold instead of 3, no class weighting (deliberately, to compare
architectures on raw generalization), four candidate architectures:

| Architecture | MAE (mean ± std across 5 folds) |
|---|---|
| Base LSTM | 19.67 ± 12.28 |
| LSTM + Attention | 19.32 ± 12.83 |
| Conv1D + LSTM | 15.57 ± 13.49 |
| **Conv1D + LSTM + Attention (current production model)** | **15.10 ± 12.55** |

This is the first real evidence that the current architecture choice is actually the
best of the four — previously that claim had no backing. Per-metric breakdown for the
production architecture:

| Metric | MAE |
|---|---|
| Balance | 13.99 ± 12.34 |
| Power | 17.13 ± 16.06 |
| Technique | 17.59 ± 15.33 |
| Defence | 11.70 ± 6.55 |

Full fold-by-fold output: `dataset/ablation_results_2026-07-09.txt`.

## Front-view-only subset (`dataset_frontview.csv`, `filter_frontview.py`)

Of the 360 sessions, 292 (81%) are already tagged front-view in their session name
(the rest: 42 side-view, 26 back-view). Filtering to just those 292 and re-running the
same 3-fold grouped CV:

| Metric | Mixed-view MAE | Front-view-only MAE |
|---|---|---|
| **Overall** | 9.15 ± 0.78 | **8.39 ± 0.56** |
| Balance | 8.92 ± 1.10 | 7.84 ± 0.38 |
| Power | 8.82 ± 0.43 | 8.45 ± 0.87 |
| Technique | 10.76 ± 1.08 | 9.81 ± 1.46 |
| Defence | 8.10 ± 0.96 | 7.45 ± 0.45 |

The front-view-only subset does slightly *better* across every metric, with lower fold
variance too. That's a real, positive early signal for committing to front-view-only —
a single consistent camera geometry is easier for this feature set to learn, exactly as
the front-view feature-robustness concern predicted. Full output:
`dataset/cv_results_frontview_2026-07-09.txt`.

## Why the two runs disagree (9.15 vs 15.10) — and what it means

Both are correct; they're measuring different things. The 5-fold run's Fold 5 MAE
blows out to 40-44 points **across all four architectures simultaneously** — including
the best one. That's not a model-capacity problem (every architecture struggles equally
on it); it's almost certainly one specific held-out batsman/source group whose labels or
extracted kinematics don't fit the pattern the rest of the dataset agrees on. With only
3 folds instead of 5, that same problem group gets diluted across a larger test split in
`cross_validate.py`, so it drags the average up less.

**Practical implication:** with only 12 unique batsman/source identities behind 360
sessions, fold-to-fold variance is dominated by which one or two identities land in the
test set, not by genuine architecture differences. This is exactly why the roadmap's
dataset-collection step (more identities, front-view-only, less shot-type/skill
concentration) matters more than further architecture tuning right now.

**Recommended next step before trusting either number as "the" model accuracy:**
identify which batsman/source group is Fold 5's outlier (check `ablation_results_2026-07-09.txt`
fold logs for the group ids) and inspect whether it's a labeling error, a corrupted
extraction, or a genuinely different playing style the model hasn't seen enough of.
