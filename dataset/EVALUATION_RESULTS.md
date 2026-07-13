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
