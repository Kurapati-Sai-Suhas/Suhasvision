# Phase 3 — Corrected Diagnostics + RTMPose Fallback Validated End-to-End

**Question.** Does the MediaPipe → RTMPose-s fallback actually improve the S2
batsman-identification system, or does it only produce more poses?

**Answer.** It produces more poses and does not improve identity decisions. On
both splits H1 returns *fewer* correct answers than H0. **Decision: C — keep
the hybrid experimental, do not adopt.**

**Scoring model.** Untouched. `SEQ_LEN = 7`, 15 3D angles + 15 velocities =
`(7, 30)`. RTMPose 2D was never inserted into that path.

---

## 1. Corrected failure taxonomy

The old diagnostic used one statistic, `k_pose_rate`, and called a clip a pose
failure below 0.4. `phase3_semantic_features.extract()` appends `None` to the
pose sequence in *two* different situations:

```python
box = by_frame.get(f)
if box is None or fr is None:
    seq.append(None)          # counted exactly like a pose failure
```

The pose model is never invoked when there is no box. So the statistic was a
product of two independent things, only one of them about pose:

```
effective_pose_rate = coverage × P(pose | box exists)
```

`phase3_failure_taxonomy.py` now reports the factors separately and classifies:

| Verdict | Meaning |
|---|---|
| `TRACKING_FAILURE` | Track absent from most sampled frames; pose is fine where a box exists. **No pose model can fix this.** |
| `POSE_FAILURE` | Track present, pose fails on the crops it is given. **The only category a better pose model can fix.** |
| `JOINT_TRACKING+POSE_FAILURE` | Both below threshold. Fixing pose raises the ceiling only to `coverage`. |
| `NO_FAILURE` | Both acceptable. |

### The unreachable branch

`categorise_failure()` tested, in this order:

```python
if k_pose_rate < 0.4:  return "pose failure on the batsman crop"
if coverage    < 0.4:  return "tracking failure — track too fragmented"
```

A pose requires a box, so `k_pose_rate ≤ coverage` **always**. Therefore
`coverage < 0.4` guarantees `k_pose_rate < 0.4`, the pose branch returns
first, and the tracking branch was **unreachable for exactly the clips it was
written to catch**. Every fragmented-track failure was reported as a pose
failure.

Fixed by testing the factors in causal order — coverage caps pose, so coverage
first — and by naming the joint case instead of forcing it into one bucket.
Regression tests assert both directions:

- low coverage + perfect pose where a box exists → `TRACKING_FAILURE`
- high coverage + low pose success → `POSE_FAILURE`
- a sweep over every count combination with bad coverage asserts that
  `POSE_FAILURE` is never returned on coverage alone

---

## 2. The new failure table (51 clips, annotated batsman track)

| Clip | coverage | pose \| box | effective | Verdict | On old list? |
|---|---|---|---|---|---|
| `pro_player_front_18` | 0.45 | 0.11 | 0.05 | POSE_FAILURE | yes |
| `pro_player_front_06` | 0.75 | 0.27 | 0.20 | POSE_FAILURE | yes |
| `pro_player_front_10` | 0.55 | 0.36 | 0.20 | POSE_FAILURE | yes |
| `pro_player_front_12` | 1.00 | 0.20 | 0.20 | POSE_FAILURE | **no** |
| `pro_player_front_44` | 0.95 | 0.26 | 0.25 | POSE_FAILURE | **no** |
| `sanjay_front _1` | 0.05 | **1.00** | 0.05 | TRACKING_FAILURE | yes |
| `kohli_side_16` | 0.25 | 0.40 | 0.10 | TRACKING_FAILURE | yes |
| `rishi_front_2` | 0.25 | 0.80 | 0.20 | TRACKING_FAILURE | yes |

**Counts: 5 POSE_FAILURE, 3 TRACKING_FAILURE, 43 NO_FAILURE.**

The old seven-clip list was wrong in both directions:

- **Mislabelled (4 of 7):** `kohli_side_12` (not failing at all — now "margin
  too small"), `kohli_side_16`, `rishi_front_2`, `sanjay_front _1` (all
  tracking failures).
- **Missed (2):** `pro_player_front_12` and `pro_player_front_44` are genuine
  pose failures the old test never flagged.

`sanjay_front _1` is the clearest case: the batsman track exists on **1 of 20**
sampled frames, and pose succeeds on **100%** of the frames it does get. It was
labelled a pose failure purely because `0.05 < 0.4`.

> **Supersedes** `phase3_pose_diagnosis.json` from the previous checkpoint,
> which sampled over `frames_processed` while extraction samples over
> `CAP_PROP_FRAME_COUNT`. That is why `rishi_front_2` reads coverage 0.25 here
> versus 0.95 there. This table uses the pipeline's own sampling and is the
> one that matches what S2 actually consumes.

---

## 3. The hybrid, defined exactly

```
H0  S2 + MediaPipe only
H1  S2 + MediaPipe, RTMPose-s where MediaPipe returns NO pose
```

The trigger is MediaPipe's refusal and nothing else — no confidence threshold,
no crop-size heuristic, no jitter test, no semantic suspicion. A refusal is a
signal the primary model produces for free, so there is no threshold here to
tune and nothing to overfit.

**Held identical:** detections, ByteTrack tracks, sampled frames, bat
detections, feature code, fitting routine, decision thresholds, dev/eval split.
Both conditions were computed in **one extraction pass**, so "everything else
constant" is true by construction rather than by intention.

This experiment uses the **actual candidate tracks** and the real S2 selector.
Ground truth scores the decision; it never makes it. (The previous pose
experiment anchored on the annotated batsman — correct for isolating pose,
wrong for this question.)

**Regression check on the refactor:** all **7250** feature values in the new H0
extraction are bit-identical to the previously committed cache, so restructuring
the extraction loop changed nothing.

Fallback activity across the corpus: 5000 sampled track-frames, 1524 with a
box, 899 poses from MediaPipe, **625 fallback calls, 625 recoveries** (RTMPose
cannot decline, so it recovers every call by construction).

---

## 4. H0 vs H1 — identity results

### Eval (27 clips, reported once)

| Arm | correct | wrong | refused | precision | wrong-person | refusal | accuracy |
|---|---|---|---|---|---|---|---|
| **H0** | **20** | 1 | 6 | 0.9524 | 0.0476 | 0.2222 | **0.7407** |
| **H1** | 18 | **0** | 9 | **1.0000** | **0.0000** | 0.3333 | 0.6667 |
| H1 with H0's frozen model | 20 | 1 | 6 | 0.9524 | 0.0476 | 0.2222 | 0.7407 |

### Dev (24 clips)

| Arm | correct | wrong | refused | precision | refusal | accuracy |
|---|---|---|---|---|---|---|
| **H0** | **20** | 0 | 4 | 1.0000 | 0.1667 | **0.8333** |
| **H1** | 18 | 0 | 6 | 1.0000 | 0.2500 | 0.7500 |
| H1 with H0's frozen model | 18 | 0 | 6 | 1.0000 | 0.2500 | 0.7500 |

**Candidate recall: 0.9801 (302 frames), identical for both arms** — pose
cannot change which candidates exist.

### Reading this honestly

H1's only gain is removing the single wrong answer on eval, and it pays two
correct answers for it. **H1 is never ahead on correct answers on either
split** (20 → 18 both times).

The frozen arm attributes the loss differently on each split, and that
inconsistency is worth stating rather than smoothing over:

- **dev:** frozen matches H1 → the recovered poses *alone* caused the two
  correct→refused, independent of refitting.
- **eval:** frozen matches H0 → the recovered poses alone changed nothing;
  the refitted coefficients caused the change.

With 24 and 27 clips these are small-sample effects and the mechanism is not
cleanly separable. What is consistent across both splits and both model
treatments is the direction: never better.

---

## 5. Per-clip decision changes

Five clips changed decision in total; **two on dev and two on eval are
regressions**.

| Split | Clip | Transition | GT | H0 conf / margin | H1 conf / margin |
|---|---|---|---|---|---|
| dev | `instagram2_front_09` | correct→refused | t4 | 0.6186 / 0.2560 | 0.4995 / 0.4118 |
| dev | `pro_player_back_27` | correct→refused | t5 | 0.6062 / 0.2005 | 0.5604 / 0.2305 |
| eval | `kohli_side_08` | correct→refused | t1 | 0.6459 / 0.6459 | 0.5491 / 0.5491 |
| eval | `pro_player_front_10` | correct→refused | t2 | 0.7229 / 0.3957 | 0.5650 / 0.1671 |
| eval | `kohli_side_12` | wrong→refused | t1 | 0.6431 / 0.3819 | 0.5678 / 0.4254 |

### Transitions

| Transition | dev | eval | |
|---|---|---|---|
| `correct→refused` | 2 | 2 | **regression** |
| `wrong→refused` | 0 | 1 | improvement |
| `refused→correct` | 0 | 0 | |
| `wrong→correct` | 0 | 0 | |
| `correct→wrong` | 0 | 0 | |
| `refused→wrong` | 0 | 0 | |

**Not one clip improved from refused→correct or wrong→correct.** The fallback
never converted a failure into a right answer.

### `pro_player_front_10` — the case that should have worked

This is the clearest genuine pose failure in the corpus, and the fallback did
exactly its job:

- GT track t2: pose frames **4 → 11** (7 recovered), `k_pose_rate` 0.20 → 0.55
- verdict `POSE_FAILURE` → `NO_FAILURE`

…and its score **fell** 0.7229 → 0.5650, flipping correct → refused. Repairing
the pose evidence on the right track made the decision worse.

---

## 6. Why more pose did not help

RTMPose returns a skeleton for **any** box, so the fallback supplies skeleton
evidence to non-batsman candidates too, not just the batsman. Measured on eval:

| | H0 | H1 |
|---|---|---|
| Mean score, **non-batsman** tracks | 0.0299 | **0.0372** ↑ |
| Mean score, **batsman** track | 0.7413 | **0.7248** ↓ |
| **Separation** (batsman − other) | **0.7114** | 0.6876 ↓ |
| Median margin | 0.8096 | 0.7480 |
| Clips with top confidence < 0.60 | 5 | **8** |

Wrong candidates gain more than the batsman does, separation shrinks, and
three more clips fall below the confidence threshold — which is precisely the
extra refusals. This is the "cannot decline" property from the previous
checkpoint reappearing at the identity level, expressed as over-refusal rather
than as confident-wrong.

**A hypothesis that did not survive testing.** The obvious suspect was
`k_visibility`, which averages MediaPipe's `visibility` and RTMPose's SimCC
score — two different uncalibrated scales — into one feature, and which drops
visibly on the changed clips (0.749 → 0.603 on `pro_player_front_10`). Removing
`k_visibility`, `k_pose_rate`, or both does **not** rescue H1: it stays at
18/0/9 on eval in every variant. Scale-mixing is real but is not the
explanation, and the separation-compression above is.

---

## 7. The new failure mode was watched for and did not occur

The specific risk was that the fallback manufactures skeleton evidence for a
bad candidate and converts a correct refusal into a confident wrong answer.

**`correct→wrong`: 0. `refused→wrong`: 0.** On both splits, H1's wrong count is
0 (H0: 0 dev, 1 eval). The failure mode did not materialise; the cost showed up
as over-refusal instead.

---

## 8. Latency

Both arms timed **interleaved in one warm process** on identical crops from the
actual candidate tracks. CPU for both models.

| | Value |
|---|---|
| MediaPipe per call | 52.63 ms median |
| RTMPose-s per fallback call | **6.03 ms median** |
| Fallback fired | 24 / 315 calls (**7.6%**), 2.0 per clip |
| H0 per clip | 1411.5 ms |
| H1 per clip | 1423.6 ms |
| **Marginal cost per clip** | **+12.1 ms (+0.9%)** |

The marginal cost is the number that matters, and it is negligible. **Cost is
not the reason to decline the hybrid — efficacy is.**

---

## 9. Silent MediaPipe failures — unchanged, as expected

The fallback triggers only on refusal, so it does nothing when MediaPipe
returns a *plausible but wrong* skeleton. Confirmed on `rishi_front_2`, where
MediaPipe returns a confident skeleton whose arms span two different people
(frame 490, visible in `pose_audit_sheets/rishi_front_2.jpg`): its H0 and H1
decisions are identical, and the clip does not appear among the changed clips.

This is expected and was not in scope to fix. Detecting silent bad poses needs
a pose-quality validator, which is not trivial — every quality metric available
here is ground-truth-free self-consistency, and a wrong-but-smooth skeleton
scores *well* on all of them. **Documented as a remaining limitation.**

---

## 10. Decision

### **C — Keep the hybrid experimental. Do not adopt.**

H1 is not promoted. `phase3_semantic_features.json` remains MediaPipe-only;
the H1 features live beside it as a labelled experimental artifact.

**Why not B (adopt the fallback).** It never produces more correct answers, on
either split. It removes eval's single wrong answer at a cost of two correct
ones, and drives refusal from 0.222 to 0.333. Adopting it would trade accuracy
for precision without being asked to, on a 27-clip eval where that trade is
within noise.

**Why not A (close the question).** The mechanism is understood, not mysterious,
and it is specific rather than fundamental: the fallback hurts because RTMPose
supplies evidence to *every* candidate. The previous checkpoint established
that RTMPose genuinely produces better skeletons on person-verified crops. Both
things are true, and the gap between them is an engineering problem, not a dead
end.

**Why not D (blocked on a pose-quality validator).** A validator would help,
but D overstates how well-specified the fix is. The measured problem is
*differential*: non-batsman scores rose more than the batsman's. A validator
that rejects bad poses is one plausible remedy; restricting the fallback to
the already-leading candidate, or renormalising skeleton features by how many
candidates received recovered poses, are others. Naming the validator as *the*
prerequisite would prejudge an open question.

### What would have to be true to adopt it later

1. `refused→correct` or `wrong→correct` transitions actually appear.
2. Batsman-vs-other separation does not shrink (currently 0.7114 → 0.6876).
3. Any feature-set change is chosen on **dev** and confirmed once on eval.

### A trap avoided

Dropping `k_pose_rate` from S2 gives eval 21 correct / 0 wrong (accuracy 0.778)
— better than H0 and H1 both. **It is worse on dev** (18 correct vs H0's 20,
accuracy 0.750 vs 0.833). The eval-side gain does not reproduce on dev, so it
is an eval-selected artifact and is **not adopted**. It is recorded here only
so the observation is not silently rediscovered and mistaken for a result.

---

## 11. Limitations

- **27 eval clips.** Differences of two clips are ~7 percentage points. The
  direction is consistent across dev and eval, but the magnitudes are not
  precise.
- The frozen-model arm disagrees between splits about *what* caused the loss
  (features vs refit); with these sample sizes that is not separable.
- The fallback was tested at exactly one operating point — the clean refusal
  trigger. Nothing here rules out a better-targeted trigger.
- RTMPose remains 2D. The scoring model's 15 angles are 3D, and the previous
  checkpoint measured a median 30.4° difference between 3D and 2D angles, so
  none of this touches the `(7, 30)` path.
