# INTERVIEW_PREP.md — HyperVerge ML Intern, Cricket Stance Analyzer

**Every claim below was verified against the actual code today.** Where the CV and the repository disagree, the repository wins and I say so plainly.

---

# 🚨 PART 0 — READ THIS FIRST (4 CV claims that will not survive scrutiny)

Your CV bullet for this project contains **four statements the code does not support.** An automated interviewer probably won't catch them, but a human reviewing the transcript afterwards — or anyone who opens the GitHub repo linked on your CV — will. Fix the CV if you can; if you can't before the interview, **know exactly what you'll say when asked.**

### ❌ 1. "Vision LLM (LLaVA)" — the project does not use LLaVA

**Reality:** `dataset/nvidia_client.py:34` →
```python
NVIDIA_VISION_MODEL = os.environ.get("NVIDIA_VISION_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
```
The word "LLaVA" appears in exactly **one place** in the entire repo — a stale comment in `inference_service.py` ("Prevents reversed timestamps (LLaVA hallucination)"), left over from an earlier iteration. There is no LLaVA client, no LLaVA weights, no LLaVA API call.

**If asked "tell me about your LLaVA integration":**
> "I should correct that on my CV — it's a leftover from an earlier iteration. The shipped system calls NVIDIA NIM's `nemotron-3-nano-omni-30b`, a vision-language model, through an OpenAI-compatible endpoint. The architectural point is the same and it's the part I'd defend: I don't ask the model for timestamps, because vision-LLMs hallucinate them. I sample ≤10 frames myself, ask it only to return *frame indices*, and do all seconds↔frames arithmetic locally. Swapping providers is one environment variable."

That answer converts an error into evidence of good design judgement. **Do not bluff about LLaVA** — you'd be describing a system you haven't built, and follow-ups (What visual encoder? CLIP ViT-L/14? How did you host it?) will expose it instantly.

### ❌ 2. "Dense network (MAE: 11.7)" — this number is not what you think it is

**Reality:** `11.70` appears in exactly one place — `dataset/ablation_results_2026-07-09.txt`:
```
TABLE 3: PER-METRIC MAE FOR TCN-ATTENTION MODEL
Defence : 11.70 ± 6.55 pts
```
It is the **Defence per-metric MAE**, for the **Conv1D + Bi-LSTM + Attention** model — not an overall MAE, and not a Dense network. The overall MAE in that same run was **15.10 ± 12.55**.

### ❌ 3. "generalized better than an over-parameterized Bi-LSTM" — three problems

1. In `ablation_study.py` (the only saved architecture comparison), **all four candidates contain a Bi-LSTM**, and the Bi-LSTM variants *won*: Base LSTM 19.67 → LSTM+Attn 19.32 → Conv1D+LSTM 15.57 → **Conv1D+LSTM+Attn 15.10 (winner)**. There is no Dense model in it.
2. `evaluate_baselines.py` *does* compare an MLP against a Bi-LSTM — but that Bi-LSTM is `Bidirectional(LSTM(8))`, commented **"Massively reduced hidden state to prevent memorization."** That is the *opposite* of over-parameterized.
3. **No saved results file for `evaluate_baselines.py` exists anywhere in the repo.** The script is real; its output was never recorded.

**The killer follow-up you must be ready for:**
> *"If your Dense network generalized better, why does your deployed model contain a Bi-LSTM?"*

Your production model is `Conv1D(64) → BatchNorm → Dropout(0.2) → BiLSTM(64) → Dropout(0.3) → TemporalAttention → Dense(32) → Dense(4)`, 76,395 params. You ship the Bi-LSTM. That contradiction is indefensible unless you get ahead of it.

**Honest answer:**
> "That bullet overstates what I actually measured. I built `evaluate_baselines.py` to compare an MLP, a 1D-CNN, a shrunk Bi-LSTM, a rule-based model and a constant-mean baseline under identity-grouped leave-one-out CV — but I never saved its output, so I can't stand behind a specific number. What I *can* stand behind is the architecture ablation I did save: four variants, 5-fold identity-grouped, where Conv1D+BiLSTM+Attention won at 15.10 MAE. That's the model I deployed, which is why my production network is the Bi-LSTM one."

### ❌ 4. "capture feedback loops for future model retraining" — the loop is severed

`active_learning_retrain.py:45` reads `coach_overrides.csv`. The only writers of that file are `dataset/app.py` (the **legacy Streamlit demo**) and a manual test script. The Django backend writes `CoachOverrideLog` **database rows** — which nothing consumes.

**So coach overrides made in the actual product can never reach retraining.** Say "the override is captured and audit-logged today; wiring that journal into the retraining input is the next step and isn't done yet."

### ⚠️ And the one that outranks all four

**Until 2026-07-28 your evaluation had test-set leakage.** `cross_validate.py` passed the test fold into `model.fit(validation_data=...)` with `EarlyStopping(restore_best_weights=True)`, then scored on that same fold — weights were *selected* on the data being reported.

| Protocol | MAE | Spearman ρ |
|---|---|---|
| As previously shipped (leaked) | 10.85 ± 1.95 | 0.279 |
| **Corrected** | **15.08 ± 3.03** | **0.145** |
| Constant-mean baseline | **13.59** | — |

**Under honest evaluation the model does not beat predicting the mean.** Never quote 10.85 or 11.56. If accuracy comes up, lead with the leak — it is the single strongest thing you did on this project (see Q1 in the Top 50).

---

# PART 1 — THE PIPELINE, END TO END

Know this cold. Every project question routes back to some stage of it.

| # | Stage | In → Out | How it works | Failure mode |
|---|---|---|---|---|
| 1 | **Acquisition** | URL → MP4 | `yt-dlp`, video-only ≤720p, cookie fallback chain for bot detection | YouTube bot-block; stale `.part` files caused HTTP 416 |
| 2 | **Temporal segmentation** | Video → shot windows | Sample ≤10 evenly-spaced frames → NVIDIA NIM returns **frame indices** → local arithmetic converts to seconds. `find_shot_windows_auto` tiles a full video into 25s windows w/ 5s overlap, merges detections spanning boundaries | Coarse resolution on long windows; LLM returns degenerate single-frame windows (clamped) |
| 3 | **Phase sampling** | Window → 7 frame indices | Uniform `linspace` across the window. (`motion_energy_phase_indices`, an MGSampler-style cumulative-motion sampler, exists but the wrist-speed variant is disabled) | Swing isn't uniform in time — "contact" can land on a transitional frame |
| 4 | **Pose extraction** | 7 frames → 33 landmarks × 5 | MediaPipe BlazePose *heavy*, `num_poses=4` | Motion blur at contact; low-light footage ~45% detection |
| 5 | **Subject selection** | 4 candidates/frame → 1 person | `subject_selection.py`: hip-center track association → ≥4/7 coverage → size dominance (1.25×) → **traversal filter** (batsman at crease vs. bowler crossing scene) → **reject if ambiguous** | Was the project's worst bug: tracked the ball *feeder* in 3/3 real shots |
| 6 | **Validation & interpolation** | Poses → validated poses | `validate_pose` (per-phase Y-ordering topology), `apply_pipeline_rules` (reject ≥3 consecutive or ≥4 total missing; else linear interp interior, extrapolate edges) | Interpolation fabricates motion if overused |
| 7 | **Quality gates** | keypoints → dataset.csv | `kinematic_validator` (bone-length CV > 0.25 ⇒ invalid) → `stance_symmetry_confidence` (jerk + bilateral symmetry z-score) → `merge.py` (3 mandatory filters) | Consistently-wrong-but-self-consistent data passes |
| 8 | **Feature engineering** | Landmarks → (7, 30) | 15 joint angles via `arccos(v1·v2 / (‖v1‖‖v2‖ + 1e-6))`, ÷180 to [0,1]; + 15 frame-to-frame velocity deltas (first frame = 0) | Angles discard absolute position/limb length |
| 9 | **Model** | (7,30) → 4 scores | Conv1D(64,k=3) → BN → Drop(0.2) → BiLSTM(64) → Drop(0.3) → TemporalAttention → Dense(32) → Dense(4, sigmoid). 76,395 params | 590 params/sample — heavily over-parameterized for n=130 |
| 10 | **Uncertainty** | 1 input → 30 predictions | MC Dropout: input tiled ×30, one batched pass with `training=True`; mean = score, std = `confidence_variance` | **Not calibrated** (ρ=0.02–0.08 vs. real error) |
| 11 | **Explainability** | Scores → joint drivers | Expected Gradients (12 real baselines × 6 path steps), two-stage: rank phases first, then features within top phases | Local explanation, not causal |
| 12 | **Insight engine** | Weakest joints → drill | `JOINT_DRILL_MAP`: 15 joints → hardcoded drill text | Static lookup, not learned |
| 13 | **API** | Video → JSON | Django REST, JWT, ≤200MB, ≤20s clip; **zero-storage** (temp file deleted in `finally`); rule-based fallback on any ML failure | 16–19s synchronous, blocks a worker |

**Complexity:** dominated by MediaPipe (7 × ~170ms) and video decode, **not** the model. Model forward pass is ~30ms for all 30 MC passes batched; EG ~43ms. The 16–19s is decode + pose.

---

# PART 2 — CV CLAIM → IMPLEMENTATION → ATTACK

| CV Claim | Verdict | Where it lives | The attack you must survive |
|---|---|---|---|
| Vision LLM **LLaVA** | ❌ **False** | Nowhere; uses `nemotron-3-nano-omni-30b` | "Which visual encoder does LLaVA use?" → you can't answer about a model you didn't use |
| **Zero-shot temporal segmentation** | ✅ True | `nvidia_client.find_shot_windows` | "Why is it zero-shot?" → no training on your data; prompt-only |
| **MediaPipe BlazePose** | ✅ True | `pose_landmarker_heavy.task` | "Two-stage detector-tracker — what are the stages?" |
| **3D kinematic time-series** | ⚠️ Weak | x,y,z used in angle math | "How reliable is MediaPipe's z on monocular video?" → **your own `kinematic_validator.py` docstring says z is far less reliable.** Concede it |
| **15 joint angles** | ✅ True | `schema.FEATURE_BASE_NAMES` (15) + 15 `_vel` = 30 | "Why angles not raw coordinates?" → translation/scale invariance |
| **Synthetic Gaussian augmentation** | ⚠️ Mixed | `evaluate_baselines.augment_training_data` (σ=0.02, train-fold only ✓) **and** `augment.py` (flip+jitter σ=0.005) | **`augment.py`'s version was tested and REJECTED — it made MAE 6.4% worse and R² go negative.** "Prevented overfitting" is not demonstrated |
| **LOO-CV** | ✅ True, but | `evaluate_baselines.py: make_folds(groups, n_splits=None)` = `LeaveOneGroupOut`, **identity-grouped** | "Was your headline model evaluated with LOO?" → **No. 5-fold GroupKFold.** LOO was only for the baseline comparison |
| **Dense MAE 11.7** | ❌ **Misattributed** | 11.70 = *Defence* per-metric MAE of the **Bi-LSTM** model | "Overall or per-metric? Which model?" |
| **beat over-parameterized Bi-LSTM** | ❌ **Unsupported** | Bi-LSTM there is `LSTM(8)`, "massively reduced"; no saved results | "Then why is your deployed model a Bi-LSTM?" |
| **Regularization** | ✅ True | Dropout 0.5 (MLP), 0.2/0.3 + BatchNorm (prod) | "Why dropout before BiLSTM not after?" |
| **MC Dropout, 30 passes, predictive variance** | ✅ **Fully true** | `ml_service.py:371-479`, batched via `np.repeat` | "Why 30?" → have a real answer (see Q12) |
| **Flag uncertain for HITL review** | ⚠️ Partial | Variance computed + stored + displayed | **No threshold, no routing.** The coach *sees* a number; nothing is auto-flagged |
| **Django REST backend** | ✅ True | `views.py`, JWT, role-scoped | — |
| **Heuristic insight engine → drills** | ✅ True | `JOINT_DRILL_MAP` | "Is the drill mapping learned?" → no, hardcoded |
| **Expert-override endpoints** | ✅ True | `PATCH /sessions/{id}/` + `CoachOverrideLog` | — |
| **Feedback loop for retraining** | ❌ **Severed** | DB rows written; retrain script reads a CSV only Streamlit writes | "Walk me through how a coach correction reaches the next model" |

---

# PART 3 — TOP 50 QUESTIONS

## 🔥 MUST KNOW (1–15)

**Q1. Walk me through how you evaluated your model.**
*Concept: experimental rigor.* **This is your best answer in the whole interview — lead with it.**
> "I'll start with a bug I found in my own evaluation. My CV pipeline passed the test fold into `model.fit(validation_data=...)` with `EarlyStopping(restore_best_weights=True)`, then scored on that same fold — so the weights being reported had been *selected* on the data I was measuring. I measured the bias on identical folds and seed: MAE went 10.85 → 15.08, a 39% optimism bias, and Spearman fell 0.279 → 0.145. Fold-level bias was +1.31, +1.64, +8.09, +7.39, +2.74 — never negative. I fixed it by carving an inner validation split out of each training fold, so the test fold is touched exactly once. The honest number is ~15.1 MAE against a constant-mean baseline of 13.59 — meaning **my model does not currently beat predicting the mean.**"
*Follow-up:* "Why report that?" → "Because the alternative is a number I know is wrong. And it explains the real-world underperformance I couldn't otherwise account for."
*Common mistake:* quoting 10.85 as if it stands.

**Q2. What is data leakage? Give an example from your own project.**
Three real ones: (a) the early-stopping leak above; (b) sessions of the *same batsman* split across train/test — fixed with identity-grouped folds via `extract_batsman_name`; (c) augmented clones landing in a different fold from their source — prevented because the identity extractor strips `_flipped`/`_jitter` suffixes.

**Q3. Why identity-grouped CV instead of random splitting?**
> "42 identities, 130 sessions — one batsman has 20. Random splitting puts the same person's other sessions in training, so the model recognises *the person*, not the technique. That inflates the score and tells you nothing about a new player. `GroupKFold` on identity, plus an `assert_no_leakage` in every fold."

**Q4. What's the difference between LOO-CV and k-fold? Which did you use?**
LOO = n folds, one sample held out each time. Nearly unbiased, high variance, expensive (n model fits). k-fold trades a little bias for much less variance and cost.
> "I used **identity-grouped LeaveOneGroupOut** in `evaluate_baselines.py` — one fold per *batsman*, not per session, which is stricter than naive per-session LOO because it refuses to leak a person's other sessions. But my headline model was evaluated with **5-fold GroupKFold**, not LOO. I should be precise about that."

**Q5. Your dataset is 130 samples and your model has 76,395 parameters. Defend that.**
> "I can't fully. That's ~590 parameters per sample — the model has far more capacity than the data can constrain, and it's my leading hypothesis for why honest CV shows it failing to beat the mean. The 5-fold ablation did pick this architecture over three simpler ones, but that comparison ran under the leaked protocol, so I'd re-run it. My planned next step is shrinking to 5–10k params and running a ridge-regression baseline on the flattened 210-dim input — if the deep model can't beat ridge, it isn't earning its capacity."

**Q6. Why joint angles instead of raw keypoint coordinates?**
Invariance. A joint angle means the same thing wherever the batter stands and at any camera distance; raw (x,y) conflates body configuration with position in frame. With 42 identities you cannot afford to spend capacity learning that invariance from data. Cost: you discard absolute position, limb length, and body scale.

**Q7. Explain MC Dropout. What uncertainty does it capture?**
Dropout normally regularizes at *training* time only. MC Dropout keeps it active at *inference*, runs N stochastic passes, and treats the spread as uncertainty — Gal & Ghahramani (2016) show this approximates Bayesian inference over the weights. It captures **epistemic** (model) uncertainty — reducible with more data. It does *not* capture **aleatoric** (inherent label noise), which is arguably your dominant error source given single-LLM labels.
*Implementation:* input tiled ×30, one batched forward pass with `training=True`; dropout masks are drawn independently per batch row, so it's statistically identical to a 30-iteration loop but ~19× faster.

**Q8. Is your uncertainty calibrated?**
> "No, and I measured it rather than assuming. `eval_uncertainty_correlation.py` against the deployed model found Spearman 0.02–0.08 between predicted uncertainty and real absolute error, p>0.28 on all four scores. So it's a rough signal of the network disagreeing with itself, not a confidence interval. Calibrating it would need isotonic regression from MC std to empirical error on a held-out calibration set, or deep ensembles."

**Q9. Why MAE and not MSE? Is MAE 15 good?**
MAE is in the output unit (points on 0–100) and is robust to outliers; MSE penalises large errors quadratically and is what you *train* on (differentiable, well-behaved gradients). "Good" is only meaningful versus a baseline: **constant-mean is 13.59**, so 15.1 is worse than trivial.
> "Also, MAE is the wrong headline metric for this task. Action Quality Assessment reports **Spearman rank correlation** — because with narrowly-distributed labels, MAE flatters you. My honest Spearman is 0.145, which is weak. Pirsiavash et al. (ECCV 2014) got 0.41–0.45 with pose features + SVR on ~150 samples — a 2014 classical method at my sample size, roughly 3× better."

**Q10. How do you know your labels are any good?**
> "I don't, and that's the project's biggest open problem. Labels come from a single vision-LLM persona, never validated against a human coach. So I currently **cannot distinguish 'the model is bad' from 'the labels don't encode batting quality.'** The cheapest test available is a natural experiment already in my data: my identity list mixes named internationals — Kohli, Root, Buttler, Stokes — with anonymous amateur footage. If pros don't score systematically higher, the labels are broken. Mann-Whitney U, 30 minutes."

**Q11. Explain the temporal attention layer.**
Additive (Bahdanau-style), not Transformer self-attention:
```
e     = tanh(x·W + b)        # (batch, 7, 1) learned importance per phase
alpha = softmax(e, axis=1)   # normalized across the 7 phases
context = Σ(x * alpha)       # (batch, 128) weighted sum, not last timestep
```
135 parameters total.
*Why not Transformer self-attention?* At 7 timesteps and 42 identities, Q/K/V projections add capacity the data can't constrain. **This is a case of deliberately not using the more famous technique** — good interview material.

**Q12. Why exactly 30 MC Dropout passes?**
Be honest: > "Empirical convention, not derived. The standard error of the mean shrinks as 1/√N, so 30 gives roughly a 5.5× reduction over a single pass, and beyond ~30 you're paying linearly for diminishing returns. Since it's batched into one forward pass it costs ~30ms, so I didn't tune it. If uncertainty were calibrated and load-bearing, I'd choose N by measuring when the variance estimate stabilises."

**Q13. Your subject-selection filter — how do you separate a batsman from a bowler?**
> "Size and persistence can't: a camera-near bowler visible in every frame is both large and persistent, exactly what my dominance rules would pick. What separates them is that a **batsman plays from a fixed crease while a bowler traverses the scene.** So I measure net hip-center displacement — net, not path length, because a batsman's hips move forward and back and end near where they started."
*The subtle bit worth volunteering:* > "I measure it two ways and only disqualify when both agree. Torso-normalized units *deflate* for a camera-near subject — a 0.30-torso body physically can't travel more than ~3.3 of its own torso lengths across a normalized frame, so a torso-only threshold could barely ever fire for the exact bowler I'm trying to catch. Frame-fraction *inflates* in tight close-ups where a real batsman's stride spans a lot of frame. Requiring both covers each other's blind spot."

**Q14. What was the worst bug you found in this project?**
> "MediaPipe ran with `num_poses=1` — one person per frame, no guarantee it's the batsman. I overlaid the detected landmarks onto real frames and the nose landmark was on the **ball feeder's face**, not the batsman. It reproduced in 3/3 AI-detected shots on the real production path. Worse, the bone-length filter only caught 2 of those 3 by accident, because a walking person's apparent proportions vary. Fix: detect up to 4 people, associate into tracks, and make one session-level choice — **or reject the session.** A confident wrong answer silently poisons training data; a rejected session just costs yield, visibly."

**Q15. Explain your train/serve consistency.**
Both paths now call the *same* `collect_phase_frames` and `extract_features_from_image_array` from `zero_storage_pipeline.py`. Previously the Django path uniformly sampled the whole clip with no validation and imputed missing frames by *copying a neighbour verbatim* — which fabricates exactly-zero velocities for half the feature vector, a pattern training data never contains.

## 🟠 HIGH PRIORITY (16–32)

**Q16. What is BlazePose and how does it work?** Two-stage: a lightweight detector locates the person/ROI, then a tracker regresses 33 landmarks within it; on video the tracker reuses the previous frame's ROI. Designed for real-time CPU/mobile.

**Q17. 2D vs 3D pose — what does MediaPipe's z actually mean?** Depth relative to the hip midpoint, in units roughly proportional to x. From a single camera it's *inferred from learned priors*, not measured. Concede it's the weakest channel — that's exactly why you wrote a bone-length consistency filter.

**Q18. What's the bone-length filter and why is it novel?** A person's bone lengths don't change between frames of one session, so high coefficient-of-variation in an estimated bone length is itself evidence of a bad pose estimate — no ground truth needed. No cricket pose paper in the current literature applies an automated anatomical-consistency filter to monocular 3D extraction. **Its blind spot:** it catches *inconsistency*, not *consistent wrongness* — which is exactly how the feeder slipped through in one shot.

**Q19. How do you handle occlusion and missing landmarks?** Low visibility is *logged but not rejected* — in a profile sport the far-side limb is normally occluded and MediaPipe still infers position usefully. Structural failure is handled by `apply_pipeline_rules`: 1–2 interior missing frames get linear interpolation, edges get linear extrapolation, ≥3 consecutive or ≥4 total ⇒ hard reject.

**Q20. Why interpolate at all — isn't that fabricating data?** It is, which is why there's a hard ceiling. Recovering 1 frame of 7 from its two real neighbours preserves genuine motion; recovering 4 would be inventing more than it recovers.

**Q21. Explain LSTM: cell state, hidden state, gates.** Cell state = long-term memory carried along the sequence with only linear interactions (why gradients survive). Hidden state = the output/short-term state at each step. Forget gate decides what to drop from cell state, input gate what to add, output gate what to expose as hidden state. Bidirectional = two LSTMs (forward + backward), outputs concatenated → 64 units becomes 128-dim.

**Q22. Why is bidirectionality legitimate here?** The task scores an *already-complete* 7-phase sequence offline. Nothing is streaming, so withholding future-phase context would be an artificial handicap. It would be illegitimate for live during-the-swing prediction.

**Q23. Why Conv1D before the LSTM?** A kernel-3 convolution extracts local 3-adjacent-phase patterns in one operation instead of making the LSTM's gates learn them recurrently. It's a TCN-style stem (Bai et al. 2018), *not* a full TCN — no dilation, because 7 timesteps needs none. Don't call it a TCN without that caveat.

**Q24. What does "over-parameterized" mean, and how would you demonstrate it?** More free parameters than the data can determine — the model can fit training data (even noise) without learning generalizable structure. Demonstrate with: train/val gap, a learning curve that plateaus while train loss keeps falling, a **label-shuffle test** (if shuffled-label performance ≈ real, it's memorizing), or beating it with a much smaller model.

**Q25. What is the label-shuffle test and why does it matter?** Randomly permute labels, retrain. If performance barely changes, your features carry no signal about the target. Given a 0.145 Spearman, this is the single most informative experiment you haven't run.

**Q26. Explain L1 vs L2 vs dropout.** L1 adds λΣ|w| — drives weights to exactly zero, gives sparsity/feature selection. L2 adds λΣw² — shrinks weights smoothly, never exactly zero. Dropout randomly zeroes activations at train time, forcing redundant representations; approximates an ensemble. Your project uses dropout (0.2/0.3 production, 0.5 in baselines) and BatchNorm — no explicit L1/L2.

**Q27. Bias–variance for your specific project.** Simultaneously **high variance** (590 params/sample, wide fold spread ±3.03) *and* **high bias** in the sense that R²=0.086 means you explain ~9% of variance. More of the same data fixes neither — more *identities* addresses variance; better labels or features address the systematic part.

**Q28. Why does Gaussian jitter hurt velocity features disproportionately?** Half your features are frame-to-frame differences. If you add i.i.d. noise with variance σ² to each angle, the difference of two independent noisy values has variance **2σ²** — velocities get double the noise, with only 7 frames and no temporal smoothing to recover from it.

**Q29. When should augmentation be applied, and to what?** Inside the CV loop, to the **training fold only** — never validation or test, which must reflect the real input distribution. `evaluate_baselines.augment_training_data` does this correctly (called after the fold split). Augmenting before splitting would leak: a jittered clone of a test sample in training is near-duplicate leakage.

**Q30. Why did your flip augmentation make things worse?** Two mechanisms: (a) **flip isn't label-preserving here** — 129 right-handed / 1 left-handed, and the LLM rubric was implicitly written for right-handers, so flipping synthesises a left-handed distribution carrying labels never validated for it; (b) clones multiply *existing* identities without adding new ones, so the model fits its 42 people more tightly (converged in ~⅓ the epochs) without learning anything about new people.

**Q31. Explain Expected Gradients and why not plain SHAP.** EG integrates the output gradient along paths from *multiple real training-distribution baselines* to the input, averaging. Integrated Gradients uses one fixed baseline; EG removes that arbitrary choice — which matters because **a joint angle has no meaningful "zero"** (0° knee flexion is a specific extreme posture, not absence). It's the same estimator `shap.GradientExplainer` uses; implemented directly via `tf.GradientTape` because `shap`'s `numba` dependency is blocked by a Windows Application Control policy on the dev machine. **Say "SHAP-family attribution", never "we use SHAP."**

**Q32. Why is attention reported as "where the model looked" and not "why"?** Jain & Wallace (NAACL 2019) showed single-head additive attention admits *alternative* attention distributions that preserve the prediction — so attention weights aren't a faithful explanation. Your causal claim stays with Expected Gradients, which is axiomatically grounded. Surfacing attention as `attention_focus` with that caveat is the defensible framing.

## 🟡 MEDIUM (33–43)

**Q33.** Why sigmoid output not linear? — Scores are bounded [0,100] by definition; sigmoid enforces the range architecturally. Not softmax: the four scores are independent, not a distribution over one choice.

**Q34.** Why is Defence consistently the hardest score? — True across every run; not yet explained by a specific feature limitation. Say that rather than inventing a reason.

**Q35.** What's your constant-mean baseline and why does it matter? — Predict the training-fold mean for every test sample. It's the floor any real model must clear. Yours currently doesn't.

**Q36.** Why Wilcoxon signed-rank rather than a t-test? — Non-parametric, paired per-fold, no normality assumption on 5 points. **Critical caveat:** with 5 folds the smallest achievable p is 0.0625, so 5-fold CV is *mathematically incapable* of showing p<0.05. That's an argument for ~80 identities and 8–10 folds.

**Q37.** How does your zero-storage guarantee work? — Uploaded video written to a UUID temp path, deleted in a `finally` block regardless of success or exception. Source videos in the offline pipeline are deleted immediately after frame extraction.

**Q38.** Batch vs real-time inference; where's your latency? — 16–19s per request, fully synchronous, blocking a worker. **The model is not the bottleneck** — MC Dropout is ~30ms and EG ~43ms batched; the time is video decode + 7 MediaPipe calls. Optimising the model would be optimising the wrong thing.

**Q39.** How would you productionize this? — Async task queue (Celery/Redis) so upload response decouples from inference; the `status` field already anticipates it. Postgres over SQLite. Model versioning is a pinned env var (`CRICKET_MODEL_FILENAME`) — deliberately pinned, not glob-latest, so a deploy always knows which artifact serves.

**Q40.** How do you detect model drift in production? — Not implemented. Would monitor input feature distributions vs. the training distribution, the fallback-trigger rate, and the subject-rejection rate.

**Q41.** What's your fallback behaviour? — Any exception in the ML path falls back to `rule_based_scorer` on the same extracted keypoints (knee-flexion + elbow-extension heuristics), flagged `is_fallback=True`. Deliberately conservative: Power and Defence report a neutral 50 rather than a number two rules can't justify.

**Q42.** How is your API secured? — JWT (SimpleJWT), role determined structurally (`hasattr(user,'academy')` vs `playerprofile`), every queryset scoped at the ORM level, direct session creation blocked (405), coach verification gate, file type + 200MB cap. **Known gaps:** no password validation on register, no rate limiting, `fields='__all__'` on serializers.

**Q43.** Complexity of your feature engineering? — O(F·T) per session: 15 angles × 7 frames, each a constant-time dot product. Negligible next to pose extraction.

## 🟢 LOW (44–50)

**Q44.** Why CSV instead of a database for the offline pipeline? — Research artifact, single developer, git-diffable, inspectable with pandas. A DB adds value at concurrent-write scale the offline pipeline never operates at.
**Q45.** Why 7 phases? — Canonical batting sequence: stance, trigger, backlift start, full backlift, downswing, contact, follow-through. GolfDB (CVPRW 2019) uses 8 canonical events for golf — structurally the same idea, which validates the framing.
**Q46.** How did you pick the 15 angles? — Standard biomechanical joints (knee/hip/elbow/shoulder/ankle bilaterally, trunk lean, arm elevation, head tilt), chosen for coaching interpretability so an explanation names something a coach can act on.
**Q47.** What's `SEQ_LEN` and why fixed? — 7. Fixed-shape input for a fixed-architecture network; sessions are padded/reindexed to exactly 7 canonical frames.
**Q48.** Idempotent ingestion? — Session names are deterministic, so re-running a batch used to append duplicates (caused a real 42-duplicate-session incident). Now the batch loads existing session names once and skips, with `--force` to override.
**Q49.** How many tests? — 157 passing: 132 dataset/pipeline + 25 Django. Includes regression tests that encode specific past bugs.
**Q50.** What would you do differently from scratch? — "Validate labels before building anything. I built a full pipeline on labels I still can't verify encode batting quality. I'd also start with a coarser target — 3 buckets, or pairwise 'is A better than B' — because 4 continuous 0–100 scores from 130 samples is brutally hard, and pairwise is *objectively verifiable*."

---

# PART 4 — 30 TRAP QUESTIONS

These are designed to expose shallow understanding. Short, sharp answers.

1. **"Why LLaVA instead of OpenCV?"** — Category error, and correct the premise: it's not LLaVA (it's NVIDIA nemotron), and OpenCV isn't an alternative — OpenCV decodes frames, the VLM does semantic shot detection. You use both.
2. **"Why do you call your data 3D?"** — MediaPipe emits x,y,z, but z is inferred from learned priors on monocular video, not measured. It's the weakest channel; that's why I filter on bone-length consistency.
3. **"Could a Dense network model temporal information?"** — Only implicitly, by flattening 7×30 into 210 features and learning position-specific weights. It can't share parameters across time or generalise a pattern to a different phase.
4. **"You have 130 samples and 76k parameters. Isn't that absurd?"** — Yes, ~590 params/sample. It's my leading hypothesis for the generalization failure. Concede fully.
5. **"What does MAE 15 mean in real cricket terms?"** — On average the predicted score is 15 points off on a 0–100 scale — e.g. calling a 70 a 55. Given a constant predictor scores 13.59, it means my model adds nothing yet.
6. **"How do you know your Gaussian augmentation is realistic?"** — I don't validate realism directly, and one variant measurably hurt. σ=0.005 was chosen to be ~0.5% of the normalized frame, below MediaPipe's own jitter, but that's a design argument, not evidence.
7. **"Why exactly 30 MC passes?"** — Empirical convention; 1/√N error decay; batched so it's ~30ms. Not tuned.
8. **"Does your uncertainty correlate with error?"** — Measured: ρ=0.02–0.08, p>0.28. Essentially no. I report it as a qualitative label, not a confidence interval.
9. **"How do you know LOO-CV isn't leaking?"** — Folds are grouped by *identity*, not session, with an `assert_no_leakage` per fold, and the identity extractor strips augmentation suffixes so clones stay with their source.
10. **"Would your model work at a different camera angle?"** — Almost certainly not. Training is 100% front-view by explicit scope. Joint angles help but don't make it view-invariant.
11. **"What if two people are the same size in frame?"** — Selection refuses to choose and rejects the session. Deliberate: a wrong answer poisons data invisibly, a rejection is visible.
12. **"Is your attention mechanism an explanation?"** — No. Jain & Wallace showed alternative attention distributions preserve predictions. I report it as where the model looked; causal claims go to Expected Gradients.
13. **"You said SHAP on your CV. Which SHAP variant?"** — Correct it: it's Expected Gradients, the same estimator `GradientExplainer` uses, implemented directly because `numba` is blocked on my machine. "SHAP-family", not "SHAP".
14. **"Why is a constant-mean baseline 13.59? What does that tell you?"** — The labels are narrowly distributed. It tells me MAE flatters this task and rank correlation is the honest metric.
15. **"Your R² is 0.086. Explain that."** — The model explains ~9% of score variance beyond the mean. It's weak, and I lead with it rather than leading with MAE.
16. **"If your Dense net was better, why ship a Bi-LSTM?"** — The CV bullet overstates it; the saved ablation picked Conv1D+BiLSTM+Attention. See Part 0.
17. **"What's the difference between epistemic and aleatoric uncertainty? Which do you have more of?"** — MC Dropout gives epistemic. Given single-LLM labels never validated against a coach, aleatoric is likely dominant — and I'm not measuring it.
18. **"How would you prove your labels are valid?"** — Pro-vs-amateur natural experiment (Kohli et al. should outscore anonymous amateur footage); multi-model agreement across model families; a human rater doing pairwise comparisons.
19. **"Why not just use more data?"** — More *sessions* of the same 42 people wouldn't help — that's what augmentation effectively did, and it hurt. More *identities* is the lever.
20. **"Your bone-length filter — what can't it catch?"** — Consistent wrongness. A steadily-tracked wrong person has perfectly self-consistent bone lengths. That's literally how the feeder passed one of three shots.
21. **"What's the time complexity of your inference?"** — Dominated by 7 MediaPipe calls (~170ms each) and video decode, not the model (~75ms total for MC Dropout + EG batched).
22. **"Why is your API synchronous if inference takes 16 seconds?"** — It shouldn't be. Known gap; the `status` field anticipates a queue that isn't built.
23. **"How do you version models?"** — `CRICKET_MODEL_FILENAME` env var, explicitly pinned rather than glob-latest, so a deploy always knows which artifact serves. **Gap:** no logged deployment history.
24. **"How does a coach's correction improve the model?"** — Today it doesn't. It's audit-logged to the DB; the retraining script reads a CSV only the legacy Streamlit path writes. The loop is severed.
25. **"Why do you interpolate missing frames instead of rejecting?"** — Bounded recovery: 1–2 interior frames from real neighbours. ≥3 consecutive or ≥4 total rejects, because past that you're inventing more than recovering.
26. **"Why front-view only?"** — Measured: front-view-only scored better with lower fold variance than mixed-view. A single consistent camera geometry is easier for hand-engineered angles to learn.
27. **"What happens on a 30-second video?"** — Rejected with `InvalidUploadError`. Serving samples 7 frames across the whole clip, so the clip *must be* a single pre-trimmed shot; 20s cap makes that contract explicit.
28. **"Is your model's improvement over the baseline statistically significant?"** — No. Under corrected evaluation it's *worse* than baseline (15.75 vs 13.59), p=0.6250.
29. **"What's the one thing you'd fix first?"** — Validate the labels. Everything downstream is uninterpretable until I know whether the ceiling is the model or the ground truth.
30. **"Why should we hire someone whose model doesn't beat a baseline?"** — Because I found that out myself, measured it fold by fold, fixed the protocol that hid it, and reported it instead of shipping the flattering number. Most people never catch that class of bug — plenty of published papers don't. I'd rather hand you a real 15 than a fake 10.85.

---

# PART 5 — 30-MINUTE REVISION SHEET

## Numbers you must not get wrong
| Thing | Number |
|---|---|
| Honest CV MAE (corrected) | **~15.1** (leaked: 10.85 — never quote) |
| Constant-mean baseline | **13.59** ← model is worse |
| Honest Spearman | **0.145** (leaked: 0.279) |
| R² | 0.086 |
| Optimism bias from the leak | **+39%** |
| Sessions / identities | 130 / 42 (1 left-handed) |
| Model params | 76,395 (~590 per sample) |
| Input tensor | (7, 30) = 7 phases × (15 angles + 15 velocities) |
| MC Dropout passes | 30 |
| Expected Gradients | 12 baselines × 6 steps |
| Uncertainty–error correlation | ρ = 0.02–0.08, p>0.28 (uncalibrated) |
| Inference latency | 16–19s (decode+pose, *not* the model) |
| Tests passing | 157 |

## The architecture, in one line
`Input(7,30) → Conv1D(64,k=3) → BatchNorm → Dropout(0.2) → BiLSTM(64) → Dropout(0.3) → TemporalAttention → Dense(32) → Dense(4, sigmoid)`

## The four things to correct if asked
1. **Not LLaVA** → NVIDIA `nemotron-3-nano-omni-30b`
2. **11.7 is Defence per-metric MAE of the Bi-LSTM model** → not a Dense net's overall MAE
3. **Augmentation was rejected** → it made MAE 6.4% worse, R² negative
4. **Feedback loop is severed** → overrides are logged, not consumed

## Three stories that make you look strong
1. **The leakage catch** — found my own metric was inflated 39%, measured it per-fold, fixed the protocol, reported the worse number.
2. **The feeder bug** — landmark overlays proved the pipeline tracked the wrong person in 3/3 real shots; fixed with multi-person tracking + reject-when-ambiguous; also discovered my *own earlier validation claim* was wrong and retracted it.
3. **The rejected augmentation** — built it, measured it fairly, it hurt, so I didn't ship it and kept the negative result.

## One-liners
- **Why angles?** Translation/scale invariance at 42 identities.
- **Why not Transformer attention?** 7 timesteps, 42 identities — capacity the data can't constrain.
- **Why bidirectional?** Offline scoring of a complete sequence; nothing is streaming.
- **Why reject over guess?** A confident wrong answer poisons data invisibly; a rejection costs visible yield.
- **Epistemic vs aleatoric?** MC Dropout gives epistemic; my labels likely carry more aleatoric.
- **What's the bottleneck?** Labels and identity count — not architecture.

## Framing for the whole project
> "It's an honest small-data applied-ML system. The engineering is genuinely solid — validated pipeline, no train/serve skew, real explainability, 157 tests. The model doesn't work yet, and I can tell you precisely why: 42 identities, ~590 parameters per sample, and labels from one LLM that I haven't validated against a human. I found and fixed a test-set leak that was hiding it. The next step isn't a better architecture — it's validating the labels and collecting more identities."
