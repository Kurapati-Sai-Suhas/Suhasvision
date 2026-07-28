# REBUILD_ROADMAP.md — Dataset Rebuild → Validated Model

**Date:** 2026-07-28. The sequence to follow from "archive the old dataset" to "a model whose accuracy claim is defensible." Each phase has an **exit gate** — do not proceed past a gate you haven't cleared, because every later phase is uninterpretable without it.

---

## ⚠️ Read before you collect a single video

**Deliberately include identifiable professionals in the new dataset.** Your old identity list mixed named internationals (`viratKohli2`, `josbuttler2`, `benstokes`, `joeroot`, …) with anonymous amateur footage (`youtubedataset29`, …). That mix is your **only free source of external ground truth about batting quality**, and Phase 3 depends entirely on it.

If you rebuild with all-anonymous clips, you permanently lose the ability to run the single most decisive label-validation test available to you. Target roughly **15–20 recognisable professionals** among your identities, named clearly in `batch_urls.csv`'s `batsman_name`.

---

## Phase 0 — Archive, don't delete (10 minutes)

```bash
mkdir dataset/_archive_2026-07-28
git mv dataset/keypoints.csv dataset/labels.csv dataset/dataset*.csv dataset/_archive_2026-07-28/
```

Keeps the option to compare old vs. new distributions, and to prove the new data is genuinely cleaner. Deleting buys nothing that archiving doesn't.

**Gate:** old data preserved, `keypoints.csv` absent so the pipeline starts fresh.

---

## Phase 1 — Pilot collection (10 videos, ~1 hour)

Do **not** collect 150 videos and then discover a systematic problem. Collect 10, then stop and check:

```bash
cd dataset
python zero_storage_pipeline.py          # macro windows can be left blank now
python audit_duplicates.py               # must be completely clean on fresh data
grep SUBJECT_SELECTION_REJECTED pipeline_rejections.log | wc -l
grep SUBJECT_SELECTED pipeline_rejections.log        # shows the bowler-filter decisions
```

Then **spot-check 5 random sessions with landmark overlays** — confirm it's tracking the batsman, not a bowler/keeper/umpire. This is the check that caught the original feeder bug; nothing substitutes for looking at the actual frames.

**Gates:**
- `audit_duplicates.py` reports zero duplicates/incomplete sessions
- Subject-selection rejection rate **< 30%** — if higher, your footage framing is wrong, fix it before collecting 140 more
- 5/5 spot-checked sessions track the batsman
- Record your **actual per-video yield** and recompute the collection target below with your real number

---

## Phase 2 — Full collection (the bulk of the work)

| Target | Net sessions | Identities |
|---|---|---|
| **Minimum viable** | ~240 | 80 |
| **Good** | ~450 | 150 |

Composition constraints:
- **2–4 sessions per identity**, hard cap 4 (breadth beats depth)
- **8–12 left-handed identities** — augmentation is proven not to substitute
- **15–20 identifiable professionals** (see the warning above)
- **≥40 examples per shot type** if you want the shot classifier
- Skill spread toward 40/30/30 amateur/pro/youth

Budget **1.5–2× raw collection** for pipeline attrition → ~450–550 raw shot attempts → **~120–150 source videos**.

Then run the full chain: `kinematic_validator.py` → `stance_symmetry_confidence.py` → `merge.py` → `filter_frontview.py` → `feature_engineering.py`.

**Gate:** ≥80 identities surviving to `dataset_angles_production.csv`, with ≥8 left-handed and ≥15 named professionals.

---

## Phase 3 — Label validation (the steps you asked about, in order)

**This is the phase that determines whether anything downstream is meaningful.** Right now you cannot distinguish *"the model is bad"* from *"the labels don't encode batting quality"* — and no amount of modelling work resolves that ambiguity.

### 3.1 — Pro vs. amateur natural experiment *(free, do first, ~30 min)*

Mann-Whitney U on the four scores, named professionals vs. anonymous amateur footage.

- **Pros don't outscore amateurs → labels are broken.** Decisive; stop and fix labelling before any modelling.
- **Pros do outscore amateurs → labels carry real signal**, and the problem is modelling/sample size.

Honest confound to name in your writeup: broadcast footage is higher quality than amateur phone video, so a positive result may partly reflect video quality rather than technique. A **null result is unambiguous either way** — which is what makes this test worth running first.

### 3.2 — Multi-model agreement *(free, ~1 hour)*

Three **different model families** (not three prompts of one model): your NVIDIA NIM endpoint, Gemini free tier, and an OpenAI model — you already have all three client packages installed. Score the same 25–30 strokes with each, compute inter-rater Spearman.

Treat this as a **falsification test**: low agreement definitively proves the labels are noise; high agreement only proves they're *consistent*, since three LLMs can share the same bias.

### 3.3 — A human rater *(lower bar than you think)*

You do **not** need a professional coach — you need any independent rater:
- Your university/college cricket club — a club captain is a 20-minute favour, not a formal engagement
- r/Cricket or a cricket Discord — post 10 stance **pairs**, ask "which has better technique?" Pairwise judgments are far easier for non-experts than 0–100 scoring
- 2–3 knowledgeable friends doing pairwise comparisons gives a usable agreement floor

### 3.4 — If none of that lands, change the claim

Scope the claim to what you can support: **"predicts what a vision-language model assesses as batting quality"** rather than *"predicts batting quality."* Still a real supervised-learning problem, completely defensible, and it removes the unvalidated-ground-truth problem entirely.

**Gate:** you can state, with evidence, what your label ceiling is. If inter-rater ρ < 0.3, fix labelling before Phase 4 — a model cannot beat its labels.

---

## Phase 4 — Modelling (only now)

Run in this order, measuring against the **corrected** protocol (`cross_validate.py` now carves an inner validation split; the test fold is scored once):

1. **Ridge regression on the flattened 210-dim input** — an honest baseline. If the 76k-param network can't beat ridge, it isn't earning its capacity.
2. **Label-shuffle test** — shuffle labels, retrain. If performance is unchanged, the model is learning nothing from the features.
3. **Shrink the model** to 5–10k params. At 130 samples you were at ~590 params/sample; this is the single most likely fix.
4. **Rank-aware loss** — USDL (Tang et al., CVPR 2020) replaces the scalar label with a Gaussian + KL loss, explicitly designed for subjective-judging ambiguity. Roughly a loss swap.
5. **Consider an easier target** — 3 buckets (*needs work / okay / good*) or pairwise "is A better than B" are far more learnable at this scale *and* arguably more useful to a coach.

**Gate:** Spearman > 0.25 **and** MAE < the constant-mean baseline. Until you clear this, the model does not work, and more data won't fix it.

---

## Phase 5 — Shot classification (recommended parallel track)

Worth doing for a strategic reason: **it sidesteps the label-validity problem entirely.** A cover drive objectively *is* a cover drive — verifiable by eye, no coaching expertise required. Discrete classes are also far more learnable at small sample size than continuous 0–100 scores.

If the scoring model can't be made to work, **a working shot classifier plus honestly-reported scoring uncertainty is a stronger project** than a scoring model nobody can validate.

**Gate:** >70% accuracy on identity-grouped held-out folds across your 5 most common shot types.

---

## Phase 6 — Real-world robustness

- **Build a "wild" test set** — 10–15 phone-filmed clips, deliberately varied lighting/framing/distance, nothing curated. This is the only honest measure of real-world performance you'll have.
- **Calibrate uncertainty** — isotonic regression from MC-Dropout std → empirical error on a held-out calibration set (currently r=0.02–0.08, i.e. uninformative).
- **OOD rejection** — flag "this upload doesn't resemble my training data" and refuse to score, extending the reject-over-guess principle already in `subject_selection.py` to the model itself.

---

## What just changed in the pipeline (ready for Phase 1)

`subject_selection.py` now runs a **bowler/traversal filter** in addition to the existing coverage and size rules:

- A batsman plays from a **fixed crease**; a bowler **traverses the scene**. Net hip-center displacement separates them where size and persistence cannot — a camera-near bowler visible in every frame is both large and persistent, exactly the profile the old rules would have selected.
- Measured **two ways**, disqualifying only when **both** are exceeded (>2.0 torso lengths **and** >0.25 of frame width). Each measure alone has a blind spot in the opposite direction: torso-units deflate for camera-near subjects (a 0.30-torso body physically cannot travel more than ~3.3 of its own torso lengths across a frame), while frame-fraction inflates in tight close-ups where a real batsman's stride spans a lot of frame. Both blind spots are covered by regression tests.
- Uses **net** displacement, not path length — a batsman's hips move forward and back across a stroke and end near where they started; path length would penalise real batting motion.
- Every decision is logged (`SUBJECT_SELECTED` now reports how many candidates were dropped as traversing), so Phase 1's spot-check has a record to work from.

**Zero added latency** — operates on hip centers already computed for track association. 10 new tests; 157 tests passing repo-wide.
