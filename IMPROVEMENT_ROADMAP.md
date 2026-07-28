# IMPROVEMENT_ROADMAP.md — Research-Grounded Findings & Fixes

**Date:** 2026-07-28. Every paper below was verified to exist; every number was measured on this repo, not estimated. Where something is unverified or was not run, it says so.

---

## ⚠️ THE HEADLINE FINDING: your reported accuracy was never a generalization estimate

You asked why real-world accuracy is poor. I found the cause in the evaluation code, and **measured it**.

`cross_validate.py` passed the **test fold** into `model.fit(validation_data=...)` alongside `EarlyStopping(restore_best_weights=True)`, then scored the model on that **same fold**. The returned weights were therefore explicitly *selected* to minimise loss on the data being reported. `train_advanced_model.py` had the same defect (one split doing double duty as early-stopping monitor and headline number).

I ran both protocols on identical folds, identical seed, identical architecture — the only difference being what early stopping is allowed to see:

| Protocol | MAE | Spearman ρ |
|---|---|---|
| **A — as previously shipped** (test fold used for early stopping) | **10.85 ± 1.95** | +0.279 |
| **B — corrected** (inner validation split carved from training fold) | **15.08 ± 3.03** | +0.145 |
| **Optimism bias** | **+4.23 points (+39%)** | −0.134 |

Fold-level detail: the bias was +1.31, +1.64, +8.09, +7.39, +2.74 — never negative, and on two folds catastrophic.

**This fully explains your symptom.** The model was never as good as 10.85; real-world performance around ~15 MAE is the model behaving exactly as its true generalization predicted. Nothing regressed — the measurement was wrong, and now it isn't.

**Fixed** in `evaluation_protocol.split_inner_validation()` + both call sites. All future numbers from these scripts are honest. **Every previously-published figure (including in `MODEL_EVALUATION.md`, `README.md`, and the SRS) is superseded.**

### 🔴 The consequence you need to hear plainly

Re-running the **actual production script** end-to-end under the corrected protocol:

```
python academic_scripts/cross_validate.py --input dataset_angles_production.csv --n_splits 5

Overall Held-Out MAE:        15.75 +/- 3.35
Constant-Mean Baseline MAE:  13.59 +/- 1.44
Wilcoxon vs. baseline:       statistic=5.000, p=0.6250 (n=5 folds)
```

**Under honest evaluation, the model does not currently beat predicting the mean.** It is *worse* than the trivial baseline (15.75 vs 13.59), and the difference is not significant in either direction.

(The 15.75 here vs 15.08 in my controlled comparison above is expected: `cross_validate.py` does not fix a training seed, so it carries run-to-run variance; the comparison script seeded every fold identically to isolate the protocol effect. Both land in the same place — ~15–16 MAE.)

This is the real state of the model. It does not mean the project is worthless — the pipeline, the data-quality engineering, the explainability work and the honesty of this correction are all genuinely strong. But **any claim that the model has learned to score batting technique is not currently supported by evidence**, and you should not make one in an interview until it is. The path to making it true is §1 below, and it starts with validating the labels.

---

## 1. Model accuracy on unseen data

### The metric itself is misleading you

Action Quality Assessment (AQA) does not report MAE as a headline. It reports **Spearman rank correlation**, introduced by Pirsiavash et al. and near-universal since — confirmed by *"A Decade of Action Quality Assessment"* (IJCV 2026, [arXiv:2502.02817](https://arxiv.org/abs/2502.02817)).

Why MAE flatters you: your labels are narrowly distributed, so a constant predictor already scores 13.59. Your R²=0.086 means you explain ~9% of variance. **Your true Spearman of 0.145 is the number that matters, and it is weak.**

Your closest possible baseline: **Pirsiavash, Vondrick & Torralba, "Assessing the Quality of Actions" (ECCV 2014)** — pose-only features + SVR on 150–159 samples, reporting **SRC 0.41–0.45**. That is a 2014 classical method, at your sample size, on your modality, beating your current system by ~3×. Large-dataset AQA sits at 0.88–0.95.

**Calibration:** <0.3 = noise · 0.4–0.6 = genuine result on small/hard data · >0.7 = large-dataset territory.

**Implemented:** `train_advanced_model.py` now prints per-score Spearman on a genuinely held-out test split alongside MAE.

### The label ceiling — likely your dominant constraint

Your labels come from **a single vision-LLM persona**, never validated against a human coach. *"The Disagreement Deconvolution"* (Gordon et al., CHI 2021, [doi:10.1145/3411764.3445423](https://dl.acm.org/doi/abs/10.1145/3411764.3445423)) shows collapsing subjective annotation into one "ground truth" overstates real-world performance. You have no aggregation at all, so **you cannot currently distinguish "model is bad" from "labels don't encode real batting quality."**

⚠️ **This is the single highest-information experiment available and I cannot do it for you** — see "What you must do", item 1.

### Recommended modelling changes (research-backed, not yet implemented)

| Change | Paper | Why it fits your 130-sample regime |
|---|---|---|
| **Pairwise/contrastive regression** | CoRe — Yu et al., ICCV 2021 ([arXiv:2108.07797](https://arxiv.org/abs/2108.07797)); PCLN — Li et al., ECCV 2022 | Regress score *differences* against exemplars instead of absolute scores → 130 samples become ~16,000 pairs. Input-agnostic: your (7,30) encoder drops in unchanged. Directly optimises rank ordering, which *is* Spearman. **Start with PCLN's auxiliary loss — cheapest version.** |
| **Score-distribution learning** | USDL/MUSDL — Tang et al., CVPR 2020 | Replace the scalar label with a Gaussian + KL loss. Explicitly motivated by label ambiguity from subjective judging — your exact problem. One-line loss swap. |
| **C-Mixup instead of flip/jitter** | Yao et al., NeurIPS 2022 ([arXiv:2210.05775](https://arxiv.org/abs/2210.05775)) | Mixes samples weighted by *label similarity*. Stays inside the real distribution. |
| **Shrink the model** | — | 76,395 params ÷ 130 samples ≈ 590 params/sample. Try 5–10k params, and run ridge regression on the flattened 210-dim input as an honest baseline. **If the deep model can't beat ridge, it isn't earning its keep.** |

### Why your flip+jitter augmentation made things *worse* — now explained

Two concrete mechanisms, consistent with *"Data Augmentation for Time-Series Classification"* (Gao et al., [arXiv:2310.10060](https://arxiv.org/abs/2310.10060)), which found some augmentations actively degrade performance via label-invariance violation:

1. **Horizontal flip is not label-preserving here.** Your set is 129 right-handed / 1 left-handed and the LLM rubric was implicitly written for right-handers. Flipping synthesises a left-handed distribution that never appears at test time, carrying labels never validated for it.
2. **Jitter disproportionately destroys your velocity channel.** Half your features are frame-to-frame differences; i.i.d. noise σ on angles gives differences variance 2σ² — velocities get ~2× the noise, with only 7 frames and no temporal smoothing to recover.

### How much more data?

Reaching Spearman ~0.4 (the 2014 pose-only benchmark) is primarily an **identity-count** problem, not a session-count one:

- **Minimum to power a real significance test: ~80 identities** (2× current). Your Wilcoxon p=0.0625 is *the floor* a 5-fold sign test can produce — mathematically incapable of showing significance. 8–10 folds fixes that.
- **Target for a genuinely strong result: 150–200 identities.**
- **Left-handed: at least 8–12 identities** (currently 1). Augmentation is proven not to substitute.
- **Sessions per new identity: 2–4.** Prioritise breadth over depth until ~80 identities.

Full procedure and attrition rates in [`DATA_COLLECTION.md`](DATA_COLLECTION.md).

---

## 2. Shot-extraction pipeline

### What I implemented: motion-energy phase sampling ✅

Your Stage-2 sampling was **purely uniform** across the detected window — assuming constant swing speed, which is false (stance/trigger is near-static; backlift→contact is explosive). So "contact" routinely landed on a transitional frame.

**MGSampler** (Zhi, Tong, Wang & Wu, ICCV 2021, [arXiv:2104.09952](https://arxiv.org/abs/2104.09952)) is the published fix: build the cumulative motion distribution over the window and sample at equal intervals of *cumulative motion* rather than time. Fast phases automatically receive more samples. Training-free, model-agnostic.

Implemented as `motion_energy_phase_indices()` in `zero_storage_pipeline.py`, with 7 unit tests.

**Critically, this succeeds where your disabled wrist-speed sampler failed.** That approach used `argmax` of one landmark's speed — a single point of catastrophic failure, which is exactly how it locked onto the ball feeder. A cumulative distribution over whole-frame energy degrades gracefully: a small moving figure perturbs the distribution slightly instead of capturing it, and phases stay monotonically ordered by construction. **Latency: zero added** (operates on already-decoded frames; no MediaPipe or API call).

### Recommended, not implemented

- **Iterative boundary refinement** — Wake et al., IEEE Access 2025 ([arXiv:2408.17422](https://arxiv.org/abs/2408.17422)). Training-free VLM method that re-samples around selected frames to converge on boundaries. Independently validates your existing "LLM picks indices, host does arithmetic" design. Cost: +1 API call per *detected shot* (not per 20s).
- **Motion-energy prefilter before the LLM** (SCSampler pattern — Korbar et al., ICCV 2019). Skip API calls on idle tiles. **Latency/cost: strongly negative — removes calls.**
- **SwingNet-style phase classifier** — GolfDB (McNally et al., CVPRW 2019, [arXiv:1903.06528](https://arxiv.org/abs/1903.06528)). 1,400 golf swings labelled with 8 canonical event frames — structurally identical to your 7 cricket phases, and strong validation that your phase framing is standard. Needs ~910 hand-labelled frame indices.

### Explicitly ruled out (do not pursue)

**ActionFormer / TriDet / BMN / BSN** — all require frame-accurate temporal boundary supervision on thousands of instances (THUMOS14, ActivityNet). You have 130 sessions and zero boundary annotations: **two to three orders of magnitude short.** **Ball tracking** (Abbas et al., [arXiv:2211.12009](https://arxiv.org/abs/2211.12009)) works on broadcast footage with a high-contrast ball; amateur net video won't support it.

---

## 3. Explainable AI

### First, a correction: **you are not using SHAP.**

The code implements **Expected Gradients** directly via `tf.GradientTape` (`ml_service.py`). The `shap` package is not installed or imported — its `numba` dependency is blocked by your machine's Application Control policy. EG *is* the estimator `shap.GradientExplainer` uses for differentiable models, so "we use SHAP-family attribution" is defensible; "we use SHAP" is not. Don't say the latter in an interview.

### Your current choice is already the right one — and better than most published work

**Sturmfels, Lundberg & Lee, "Visualizing the Impact of Feature Attribution Baselines"** (Distill 2020) is the published evidence on baseline sensitivity: a zero baseline silently asserts "zero = absence of this feature," which for a **joint angle is false** — 0° knee flexion is a specific extreme posture, not missingness. Sampling 12 real training-distribution baselines (per **Erion et al., Nature Machine Intelligence 2021**) is exactly the prescribed fix. Most applied projects get this wrong; you didn't.

Kranzinger et al.'s *scoping review of XAI in sports science* (2025, [doi:10.1007/s44163-025-00709-8](https://link.springer.com/article/10.1007/s44163-025-00709-8)) found SHAP overwhelmingly dominant with **minimal domain-expert involvement** — most work never asks a coach whether the explanation is usable. Your named-joint → plain-English → drill mapping is *ahead* of the reviewed field on explanation form.

**I found no published XAI paper on cricket batting technique specifically.** That's a genuine, modest novelty claim.

### What I implemented ✅

**1. Two-stage time-then-feature attribution.** Ismail et al., *"Benchmarking Deep Learning Interpretability in Time Series Predictions"* (NeurIPS 2020, [arXiv:2010.13924](https://arxiv.org/abs/2010.13924)) found saliency degrades when attributing across time *and* feature axes simultaneously, and that the fix is ranking timesteps first, then features conditioned on them. Your code summed |attribution| over all 7 phases, discarding the temporal axis — so a diffuse noisy feature could outrank a sharp real signal confined to contact. Now the phase ranking comes first. **Cost: 0.77 ms** (pure post-processing of the same tensor). Bonus: coaching copy is now *"your front knee, mainly during your downswing."*

**2. Attention weights surfaced** as `attention_focus` in the API response — the model's own answer to "which phase did I concentrate on." **Cost: 25.6 ms.** Shipped with a deliberate wording caveat: reported as *where the model looked*, never *why it scored that*. This follows the **Jain & Wallace (NAACL 2019) / Wiegreffe & Pinter (EMNLP 2019) / Bibal et al. (ACL 2022)** debate — single-head additive attention is precisely the class shown to admit alternative distributions preserving the prediction, so the causal claim stays with Expected Gradients.

**Total added latency: 26.3 ms against a ~16,000 ms request — 0.16%.**

### Recommended, not implemented

- **Faithfulness test** (ERASER comprehensiveness/sufficiency + Schlegel et al., ICCVW 2019): perturb the top-2 attributed features vs. a random-2 control across all 130 sessions, paired test. ~260 forward passes, runs offline in seconds. Converts "we use a principled method" into a measured number.
- **Sanity check** (Adebayo et al., *"Sanity Checks for Saliency Maps"*, NeurIPS 2018, [arXiv:1810.03292](https://arxiv.org/abs/1810.03292)): re-run EG against a randomly re-initialised model; near-zero correlation is the pass condition. A method that survives randomisation is an edge detector, not an explanation.
- **Do not install `shap`, TimeSHAP, Dynamask, or LIME.** All are strictly more expensive than what already runs, and the benchmark literature favours gradient methods on sequence data.

---

## 4. Data splits — what to use and what to collect

### Split protocol (fixed)

- **Cross-validation:** identity-grouped `GroupKFold`, with an **inner validation split carved from each training fold** for early stopping. The test fold is touched exactly once, for scoring. ✅ implemented
- **Single training run:** three-way 60/20/20 identity-grouped — train fits weights, val drives early stopping + LR, **test is touched once at the end**. ✅ implemented
- **Never** let early stopping, LR scheduling, or hyperparameter choice see the reported split. That was the bug.

One point in your favour: *"A Standardized Benchmark for Skeleton-Based Rehabilitation Assessment"* (Ismail-Fawaz et al., IEEE FG 2026, [arXiv:2507.21018](https://arxiv.org/abs/2507.21018)) notes many published numbers use **subject-leaky splits**. Your identity-grouped protocol is *more* honest than much of the literature — part of why your numbers look worse than published ones.

### What KIND of data prevents over/underfitting

Your current failure is **overfitting to 42 identities** (590 params/sample), and *simultaneously* underfitting the real task (R²=0.086). More of the same data fixes neither. What each axis buys you:

| Axis | Current | Why it matters |
|---|---|---|
| **Identities** | 42 | The binding constraint. Directly reduces fold variance and enables more folds. |
| **Left-handed** | 1 | The model has effectively never seen one. Cannot be synthesised (proven). |
| **Skill spread** | 72/15/13 vs 40/30/30 target | Narrow label range is why MAE flatters; wider genuine range gives the model something to rank. |
| **Camera/lighting variety** | front-view only, curated | Real-world uploads won't match your training distribution. |
| **Bowling type** | 100% fast | Spin/swing is V2, but the field is already tracked. |
| **Sessions per identity** | mean 3.1, max 20 | One identity has 20 sessions — over-concentration. Cap at ~4. |

---

## 5. Full data→model pipeline

Fixed this session: **evaluation protocol** (§the headline), **phase sampling** (§2), **attribution quality** (§3). Previously fixed: wrong-person tracking, train/serve skew, two data-corruption bugs, idempotent ingestion.

**Remaining gaps, ranked:**

1. **Label validity is unmeasured** (§1) — blocks interpreting everything downstream.
2. **Identity count** (42) — blocks statistical power.
3. **No faithfulness/sanity validation of explanations** (§3).
4. **Uncertainty is uncalibrated** — measured honestly (r 0.02–0.08, p>0.28). Would need isotonic regression on a held-out calibration set, or deep ensembles.
5. **Model capacity likely too high** for the data (§1).

---

## 6. Latency — verified, no regression

| Stage | Cost | Status |
|---|---|---|
| MC-Dropout (30 passes, batched) | 30.4 ms | pre-existing |
| Expected Gradients | 43.2 ms | pre-existing |
| **+ two-stage ranking** | **+0.77 ms** | **NEW** |
| **+ attention focus** | **+25.6 ms** | **NEW** |
| **Total added** | **26.3 ms** | **0.16% of a ~16 s request** |

Motion-energy sampling adds **zero** (operates on already-decoded frames). Corrected CV costs slightly *more offline training time* (smaller inner-train set → sometimes more epochs) but **nothing at inference**.

The real 16–19 s is video decode + MediaPipe, not the model. If you want a genuine latency win, that's where to look — not here.

---

## ⚠️ What YOU must do (I cannot)

1. **Validate your labels — highest priority.** Have a real coach (or 3 independent LLM personas) independently score 20–30 strokes you already have. Compute agreement with your existing labels. If two coaches only agree at ρ≈0.4, that's your ceiling and the model is nearer it than it looks. **One afternoon; changes how you interpret everything.**
2. **Collect more identities** (§1, `DATA_COLLECTION.md`). ~80 minimum, 8–12 left-handed. Nothing substitutes.
3. **Decide on the modelling changes** in §1 — I did not implement PCLN/USDL/C-Mixup because each changes training semantics and needs a controlled before/after against the *corrected* baseline. Now that the baseline is honest, those comparisons will finally be meaningful.
4. **Re-run the corrected evaluation and update every published number.** `python academic_scripts/cross_validate.py --input dataset_angles_production.csv --n_splits 5`. Cite ~15.08 MAE / ρ≈0.145, not 10.85.
5. **Run the faithfulness + sanity tests** (§3) before claiming the explanations are validated.

---

## Superseded documents

`MODEL_EVALUATION.md`, `README.md` §8, `DATA_LINEAGE.md`, and `docs/SuhasVision_SRS_v2.0.docx` all cite the pre-fix 10.85 ± 1.95 figure. **All are superseded by this document** until re-run under the corrected protocol.
