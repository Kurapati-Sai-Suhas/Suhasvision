# Phase 3 — Pose Bottleneck / RTMPose Evaluation

**Question.** Does RTMPose materially improve batsman pose availability and
biomechanical-frame quality on the exact failure cases that currently limit
S2/S3 — and is pose actually the binding bottleneck at all?

**Status of the scoring model.** Untouched. `SEQ_LEN = 7`, 30 features per
frame, Conv1D → BiLSTM → Attention → 4 sigmoid outputs. Nothing in this
experiment retrained or modified it.

---

## 0. Headline

The experiment did not answer the question it was asked; it first had to
correct the question. Two findings came before any model comparison, and both
change what the comparison means.

1. **The "pose failure" diagnosis that motivated this work is partly an
   artifact.** `k_pose_rate`, the statistic that labels a clip a pose failure,
   counts frames where the batsman *track does not exist* identically to
   frames where the pose model was called and failed. It conflates a tracking
   problem with a pose problem.

2. **RTMPose's apparent 100% pose availability is an artifact.** RTMPose is a
   top-down regressor and cannot decline. Handed a flat grey image it still
   returns a complete 26-keypoint skeleton. Its raw availability is 1.0 on
   every clip by construction, and reading that as "it fixed every pose
   failure" would be wrong.

With both artifacts removed, the real result is narrower but genuine: on
crops that do contain the batsman, RTMPose recovers frames MediaPipe drops,
and does so correctly — confirmed visually, not just numerically.

**Is pose the binding bottleneck?** Partly. Of the 10 clips below the
pipeline's own `k_pose_rate < 0.4` failure test, **5 are genuinely
pose-limited**, 2 are tracking failures that no pose model can reach, and the
rest are jointly limited. Of the 7 clips previously *labelled* pose failures,
only 3 actually are — while 3 genuine pose failures were never labelled at
all.

**Does RTMPose help?** Yes, materially and on the cases that matter:

| | Result |
|---|---|
| Frames MediaPipe drops that RTMPose recovers | **27.9%** overall, **50.8%** on the hard clips |
| Sub-threshold clips lifted above the pipeline's own 0.4 test | **8 of 10** |
| Quality on MediaPipe's *own* successful frames | RTMPose wins **every** metric |
| Latency (`rtmpose-s`, CPU) | **7.2× faster** than MediaPipe |

**Decision: D — fallback hybrid**, MediaPipe primary with `rtmpose-s` on the
frames MediaPipe refuses, **scoped to the semantic batsman-verification path
only**. The scoring model's `(7, 30)` features stay on MediaPipe, because
production computes those 15 angles in **3D** and RTMPose 2D moves them by a
median of **30.4°** — the tensor would keep its shape while its meaning
shifted. Full reasoning in §12.

---

## 1. What was held constant

The only intended difference is the pose model. Everything upstream is the
identical cached artifact:

| Stage | Held at |
|---|---|
| Detector + tracker | `tracks_yolo11m@640_bytetrack.json` (Phase 2 cache) |
| Clip set | the same 51 annotated clips |
| Person selected | the **annotated** batsman track, not the model-selected one |
| Sampled frames | identical indices per clip, one sequential decode |
| Bounding box | the same box handed to both models on each frame |
| Feature definition | the same 15 angles, recomputed identically from both |

Anchoring on the annotated batsman is deliberate: Phase 3 already showed that
anchoring on the *predicted* batsman silently evaluates the wrong person on
the feeder clips. Fixing identity by construction is what makes this a test of
pose alone.

### Implementation actually used (verified, not assumed)

| Property | Value |
|---|---|
| Model | `rtmpose-{s,m,x}_simcc-body7_pt-body7-halpe26_700e` |
| Keypoints | **halpe26** — 26 keypoints including feet |
| Input size | 192×256 (s, m), 288×384 (x) — **static in the ONNX graph** |
| Backend | ONNX Runtime |
| Device | **CPU** — no CUDAExecutionProvider available in this environment |
| Coordinates | absolute frame pixels, top-down from the supplied box |

**Why halpe26 and not the default COCO-17.** `angle_ankle_L/R` is defined as
knee → ankle → **foot_index**. COCO-17 has no foot keypoints, so benchmarking
the default variant would have silently dropped 2 of the 15 angles while still
producing plausible-looking numbers. halpe26 carries `l/r_big_toe` and heels,
so the `(7, 30)` representation is reconstructed identically from both models.

---

## 2. A constraint found by reading the production feature code

`academic_scripts/feature_engineering.py` computes all 15 angles in **three
dimensions** — it reads `{joint}_x/_y/_z` and takes 3D dot products. MediaPipe
supplies that `z`. **RTMPose 2D does not.**

So RTMPose is not a drop-in replacement even though it preserves 7 phases ×
30 features: the tensor keeps its *shape* while the angles change *meaning*
from 3D to 2D-projected. The frozen scoring model was trained on 3D angles;
feeding it 2D angles is train/serve skew, not a swap.

This is measured rather than argued — see §7 for how much `z` actually moves
the angles.

For the comparison itself, angles are computed in **2D for both** models, so
the variable under test is landmark quality and not dimensionality.

---

## 3. Finding 1 — the pose-failure diagnosis is partly a tracking failure

`phase3_batsman_analysis.categorise_failure()` labels a clip
`"pose failure on the batsman crop"` when `k_pose_rate < 0.4`.
`k_pose_rate` is produced by `phase3_semantic_features.extract()`, which
samples 20 frames uniformly across the whole clip and then, per track:

```python
box = by_frame.get(f)
if box is None or fr is None:
    seq.append(None)          # counted exactly like a pose failure
```

A sampled frame where the batsman **track does not exist** is recorded
identically to one where the pose model ran and failed. The pose model is
never invoked on those frames. So

```
k_pose_rate = coverage_on_sampled_frames  ×  P(pose | box exists)
```

and only the second factor is a pose-model problem. No pose model can change
the first.

### A second, purely logical consequence

`categorise_failure()` tests, in this order:

```python
if gt.get("k_pose_rate", 0) < 0.4:  return "pose failure on the batsman crop"
if gt.get("coverage",    0) < 0.4:  return "tracking failure — track too fragmented"
```

A pose requires a box, so `k_pose_rate ≤ coverage` always. Therefore
`coverage < 0.4` **implies** `k_pose_rate < 0.4`, the pose branch returns
first, and the `"tracking failure"` branch is **unreachable for exactly the
clips it was written to catch**. Every fragmented-track failure has been
reported as a pose failure.

_Quantified decomposition: §4._

---

## 4. Decomposition of the seven labelled pose failures

Measured with the pipeline's own uniform 20-frame sampling on the annotated
batsman track (`phase3_pose_diagnosis.py`), so the numbers are directly
comparable to `k_pose_rate`:

| Clip | Coverage | `k_pose_rate` (MP) | P(pose \| box) | What is actually wrong |
|---|---|---|---|---|
| `sanjay_front _1` | **0.05** | 0.05 | **1.00** | pure tracking — track exists on 1 of 20 frames; pose never fails |
| `kohli_side_12` | 0.40 | 0.35 | 0.88 | mostly coverage |
| `kohli_side_16` | **0.25** | 0.10 | 0.40 | coverage-capped below threshold |
| `pro_player_front_10` | 0.55 | 0.20 | 0.36 | genuinely pose-limited |
| `pro_player_front_06` | 0.75 | 0.20 | **0.27** | genuinely pose-limited |
| `pro_player_front_18` | 0.45 | 0.05 | **0.11** | genuinely pose-limited |
| `rishi_front_2` | 0.95 | **0.85** | 0.89 | not failing at all by this measure |

Of the seven clips labelled "pose failure on the batsman crop":

- **3 are genuinely pose-limited** — `pro_player_front_06`, `_10`, `_18`.
- **2 are coverage-capped** — `sanjay_front _1` and `kohli_side_16` have
  coverage below 0.4, so `k_pose_rate` **cannot reach 0.4 under any pose
  model whatsoever**.
- **1 is not failing** — `rishi_front_2` sits at 0.85.
- 1 (`kohli_side_12`) is jointly limited, with pose working on 88% of frames
  where a box exists.

### The label is wrong in both directions

Widening to all 51 clips, **10** sit below `k_pose_rate` 0.4 under MediaPipe —
and four of them were never on the labelled list:

| Clip | Coverage | `k_pose_rate` | P(pose \| box) | Labelled? |
|---|---|---|---|---|
| `sanjay_front _1` | 0.05 | 0.05 | 1.00 | yes |
| `kohli_side_12` | 0.40 | 0.35 | 0.88 | yes |
| `kohli_side_06` | 0.55 | 0.25 | 0.45 | **no** |
| `kohli_side_22` | 0.45 | 0.20 | 0.44 | **no** |
| `kohli_side_16` | 0.25 | 0.10 | 0.40 | yes |
| `pro_player_front_10` | 0.55 | 0.20 | 0.36 | yes |
| `pro_player_front_06` | 0.75 | 0.20 | 0.27 | yes |
| `pro_player_front_44` | 0.95 | 0.25 | 0.26 | **no** |
| `pro_player_front_12` | 1.00 | 0.20 | 0.20 | **no** |
| `pro_player_front_18` | 0.45 | 0.05 | 0.11 | yes |

**5 of the 10 are genuinely pose-limited** (P(pose|box) < 0.4), and three of
those five were never labelled as pose failures. `pro_player_front_12` is the
cleanest genuine case in the corpus: coverage 1.00, so tracking is perfect and
MediaPipe still fails on 80% of frames.

So the labelled list both over-includes (tracking failures wearing a pose
label) and under-includes (real pose failures never flagged). This is the
direct consequence of the conflation in §3.

---

## 5. Finding 2 — RTMPose cannot decline, so "availability" is not comparable

The first benchmark run reported `pose_availability = 1.0000` for every
RTMPose variant on every clip, including all seven failure clips. Taken at
face value that reads as "RTMPose eliminates 100% of pose failures."

It does not. A direct test:

| Input | MediaPipe | RTMPose-m |
|---|---|---|
| Empty corner patch of a real frame | returns a pose | returns a pose, mean score 0.172 |
| **Flat grey image, no content at all** | **no pose** | **full 26-keypoint skeleton, mean score 0.141** |

RTMPose is top-down: given a box it always emits keypoints. It has no way to
answer "nobody is here". Availability is therefore not a property the two
models share, and comparing them on it credits one model for being unable to
decline.

Every ground-truth-free metric inherits this caveat. A regressor's
hallucinations are *temporally smooth*, which flatters jitter and smoothness
precisely where the model is least trustworthy.

### Can a confidence gate repair this?

A gate on confidence is the obvious fix, and whether it *works* is empirical:
it works only if the score separates real batsmen from person-free background.

The null distribution was built from **815 person-free boxes** across all 51
clips, verified person-free by a permissive YOLO sweep (`conf=0.05`, class
person) plus every tracked box on the frame, with overlap measured as the
fraction of the candidate box covered — not IoU, since a small box sitting
*inside* a large person has low IoU but is entirely person.

> A first version of this null excluded only *tracked* people on frame 0. It
> produced a null 95th percentile (0.738) **above** RTMPose's mean confidence
> on real batsmen (0.617) — an impossible ordering that gave the contamination
> away. Cricket frames are full of fielders, umpires and crowd that are never
> tracked, so "not a track" is nowhere near "not a person".

| Model | AUC person vs background | Mean on batsmen | Mean on null |
|---|---|---|---|
| `rtmpose-s@192x256` | 0.961 | 0.593 | 0.205 |
| `rtmpose-m@192x256` | 0.940 | 0.616 | 0.238 |
| `rtmpose-x@288x384` | 0.918 | 0.674 | 0.285 |

The score **does** carry real signal (AUC 0.92–0.96). But the operating point
is what matters, and MediaPipe sits at a strict one on its own: it returns a
pose on **72.1%** of real batsman frames while falsely posing on only **1.7%**
of person-free boxes. Gating every RTMPose variant to that same 1.7%:

| Model | Gate | Real batsman frames kept |
|---|---|---|
| `mediapipe_heavy` | built in | **0.721** |
| `rtmpose-s@192x256` | 0.663 | 0.255 |
| `rtmpose-m@192x256` | 0.773 | 0.003 |
| `rtmpose-x@288x384` | 0.925 | 0.000 |

Recall at fixed false-positive rates:

| Model | FPR 1% | 2% | 5% | 10% |
|---|---|---|---|---|
| `rtmpose-s` | 0.160 | 0.294 | 0.733 | 0.977 |
| `rtmpose-m` | 0.000 | 0.019 | 0.393 | 0.933 |
| `rtmpose-x` | 0.000 | 0.000 | 0.086 | 0.793 |

Two things follow.

1. **Bigger RTMPose models hallucinate more confidently.** Null scores rise
   monotonically with capacity (0.205 → 0.238 → 0.285), so the gate must rise
   too, and recall collapses. `rtmpose-x` is the *worst* self-verifier.
2. **RTMPose cannot be used as its own person-verifier at a strict operating
   point.** As a standalone "is a batsman here?" detector it is clearly worse
   than MediaPipe.

**But this is not the deployment.** In this pipeline the presence question is
already answered upstream: boxes come from YOLO11m + ByteTrack tracks that are
person detections by construction. RTMPose is never asked to decide whether
someone is there. The null result therefore invalidates the *availability
metric* — it does not disqualify RTMPose for this use. What it forbids is
using RTMPose's own confidence as a quality gate, which §10 tests directly.

---

## 6. Quality on identical frames

The headline quality table is not a fair comparison: MediaPipe's numbers are
computed only on the frames where it *managed* to return a pose (72% of them,
and by construction the easier ones), while RTMPose's cover every frame
including the hard ones it alone attempted. That biases the comparison **in
MediaPipe's favour**.

Restricting to the **724 frames where every model returned a pose** removes
the bias. Lower is better throughout.

| Metric | MediaPipe | rtmpose-s | rtmpose-m | rtmpose-x |
|---|---|---|---|---|
| Bone-length CV | 0.2741 | 0.2502 | 0.2395 | **0.2306** |
| Jitter (torso units) | 0.3800 | **0.1709** | 0.1752 | 0.1779 |
| Implausible angle rate | 0.0108 | **0.0083** | 0.0097 | 0.0102 |
| Velocity discontinuity | 0.0880 | 0.0529 | 0.0462 | **0.0420** |
| Angle smoothness | 20.58 | 13.74 | 12.51 | **12.24** |

On the labelled pose-failure clips only:

| Metric | MediaPipe | rtmpose-s | rtmpose-m | rtmpose-x |
|---|---|---|---|---|
| Bone-length CV | 0.3271 | 0.2250 | 0.2292 | **0.2084** |
| Jitter | 0.5482 | 0.3445 | 0.3388 | **0.3283** |
| Velocity discontinuity | 0.1154 | 0.0768 | 0.0654 | **0.0374** |
| Angle smoothness | 27.74 | 21.56 | 19.18 | **18.88** |

**Every RTMPose variant beats MediaPipe on every metric, on MediaPipe's own
easier frames.** Jitter is the largest gap — 2.2× lower — which matters
directly because half of the `(7, 30)` tensor is angular *velocity*, and
velocity amplifies landmark noise.

**Where to be sceptical.** Jitter, smoothness and velocity discontinuity all
reward temporal smoothness, and a regressor is smooth by construction — these
three are not independent evidence. Bone-length CV is the more trustworthy
number: it is a geometric self-consistency check (a rigid body's bones must
not change length), and RTMPose wins it too (0.274 → 0.240), by a smaller
margin. The implausible-angle rate, which counts anatomically impossible knees
and elbows, also favours RTMPose. §11 is what turns this from suggestive into
established.

---

## 7. How much does MediaPipe's `z` actually matter?

Measured, not assumed: for every frame where MediaPipe returned a pose, the
same 10 three-point angles were computed twice — once in 3D as production
does, once in 2D — and compared.

> **Median difference: 30.44°**, averaged over clips.

That is not cosmetic. Dropping to 2D changes the production angles by roughly
30 degrees at the median, so a 2D pose model does not feed the frozen scoring
model the distribution it was trained on. The tensor would keep its `(7, 30)`
shape while its contents shifted substantially.

**What this does and does not say.** It does *not* say the 3D angles are more
correct — MediaPipe's `z` is a learned relative-depth estimate of unverified
accuracy, and this experiment has no depth ground truth. It says only that the
two are far apart, so the swap is **not distribution-preserving**. Retraining
would be required, and retraining is out of scope by the stop condition.

---

## 8. Latency

CPU for both models — no CUDA provider is available here, so this is a fair
CPU-vs-CPU comparison; a GPU RTMPose would be faster than shown. Both models
were timed **interleaved inside one warm process on identical frames**, which
is what makes the numbers comparable. (An earlier Phase-2 latency result was
wrong precisely because arms were timed in separate sequential batches.)

| Model | Median ms/frame/track | Mean | n |
|---|---|---|---|
| `mediapipe_heavy` | 99.63 | 113.32 | 1004 |
| **`rtmpose-s@192x256`** | **13.83** | 27.65 | 1004 |
| `rtmpose-m@192x256` | 82.03 | 96.02 | 1004 |
| `rtmpose-x@288x384` | 579.06 | 588.59 | 1004 |

`rtmpose-s` is **7.2× faster than MediaPipe**; `rtmpose-x` is 5.8× slower.
Cost is per frame *per track*, so a clip with several candidate tracks pays
this repeatedly — the x variant is not viable on CPU.

---

## 9. Resolution and capacity

**A confound that could not be removed.** The ONNX graphs have static input
shapes, so input resolution cannot be re-parameterised on one checkpoint —
each resolution is a different published model. Only three halpe26 checkpoints
exist, and **there is no `rtmpose-m` at 384×288**. So resolution and capacity
cannot be varied independently with published weights: the 384×288 model is
also the larger `x` backbone. `rtmpose-x@288x384` is therefore reported as an
**upper bound on RTMPose**, not as a resolution result. `s` vs `m` at a fixed
192×256 isolates capacity.

**Why more input resolution cannot help here.** RTMPose resizes each crop to
its fixed input, so a person already smaller than the input is *upsampled* —
no detail is recovered that the source never captured.

| Subset | Median batsman height | Min | Below the 256 px input |
|---|---|---|---|
| All 51 clips | 288.8 px | 119.3 px | 18 / 51 |
| The 7 labelled failures | **181.4 px** | 137.2 px | **5 / 7** |

The failure clips have markedly smaller batsmen (181 px vs 289 px median), and
five of the seven are already below the model's own input height. For those,
moving to a 384 px input adds interpolation, not information — which is
consistent with `rtmpose-x` failing to deliver a proportional gain despite
5.8× the cost.

---

## 10. Hybrid fallback

Per-frame paired outcomes, because per-clip averages cannot answer this: two
models can post identical availability while failing on disjoint frames
(fallback helps enormously) or on exactly the same frames (fallback helps not
at all).

| Subset | Both OK | MP fails, RTM OK | MP OK, RTM fails | Both fail |
|---|---|---|---|---|
| All 51 clips (1004 frames) | 724 | **280 (27.9%)** | 0 | 0 |
| Labelled failure clips (124 frames) | 61 | **63 (50.8%)** | 0 | 0 |

The two right-hand columns are **zero by construction**, not by merit —
RTMPose cannot fail because it cannot decline (§5). So this table says exactly
one useful thing, and it is about MediaPipe: it drops **27.9%** of frames
overall and **50.8%** on the hard clips, and RTMPose has an answer for every
one of them. Whether those answers are correct is settled visually in §11,
where they are.

**The trigger matters more than the models.** A fallback needs to know when to
fire. RTMPose's own confidence cannot be that trigger — §5 shows it cannot
self-verify at MediaPipe's operating point. But it does not need to be:
**MediaPipe returning no pose is itself a clean, free, reliable trigger.** The
hybrid therefore needs no new threshold and no tuning, which also means
nothing to overfit.

### Counterfactual against the pipeline's own failure test

How many of the 10 sub-threshold clips clear `k_pose_rate ≥ 0.4`?

| Configuration | Fixed | Still failing |
|---|---|---|
| RTMPose ungated (the in-pipeline case) | **8 of 10** | 2 |
| RTMPose gated by its own confidence | **0 of 10** | 10 |

The ungated row is the honest one *for this pipeline*, because every box
already comes from a YOLO + ByteTrack person detection — nothing ever asks the
pose model whether a person is present. The gated row is what would happen if
it had to self-verify, and confirms it cannot.

The 2 that remain — `sanjay_front _1` (coverage 0.05) and `kohli_side_16`
(0.25) — are **tracking failures**, and no pose model can fix them. That is
the ceiling on what this entire line of work could ever deliver.

---

## 11. Visual audit

Every metric above is ground-truth-free: bone-length CV, jitter and angle
plausibility all measure **self-consistency**. A skeleton can be perfectly
self-consistent and perfectly wrong. Phase 1 shipped a wrong-person bug that
looked fine in every table, so the numbers do not decide this alone.

Sheets in `dataset/phase3_results/pose_audit_sheets/` render both models'
skeletons on the *same* frames of the same annotated batsman
(`phase3_pose_audit.py`; MediaPipe yellow on top, RTMPose green below, hollow
dots mark confidence < 0.3).

What the sheets show on the genuinely pose-limited clips:

| Clip | MediaPipe | RTMPose-m |
|---|---|---|
| `pro_player_front_10` | **no pose on 4 of 6** sampled frames | correct skeleton on all 6 |
| `kohli_side_16` | **no pose on 4 of 6** | tracks the stance throughout |
| `pro_player_front_18` | **no pose on 4 of 6** | correct on all 6 |
| `pro_player_front_06` | **no pose on 4 of 6** | plausible on all 6 |
| `rishi_front_2` | pose on all 6, but one **spans two people** | correct on all 6 |
| `kohli_side_12` | correct on all 6 | correct on all 6 |
| `sanjay_front _1` | correct on all frames | correct on all frames |

Two things are visible that no table reports:

- **RTMPose is not hallucinating on these crops.** Where MediaPipe returns
  nothing, RTMPose's recovered skeletons are anatomically placed on the real
  batsman — feet planted, arms following the bat. The null-image result (§5)
  shows RTMPose *can* fabricate a skeleton from nothing, but on boxes that
  genuinely contain the batsman it is right.
- **MediaPipe fails silently as well as loudly.** On `rishi_front_2` frame
  490 MediaPipe returns a confident skeleton whose arms span **two different
  people**, while RTMPose stays on the batsman. This is the failure mode
  self-consistency metrics cannot catch: the wrong skeleton is smooth and
  anatomically plausible.
- `sanjay_front _1` and `kohli_side_12` show **both** models succeeding on
  every frame — direct visual confirmation that their "pose failure" labels
  are misdiagnoses (§3). `kohli_side_12` additionally shows the batsman
  *walking* rather than playing a shot, which is a shot-localization concern
  for the deferred L-phase, not a pose concern.

---

## 12. Decision

### **D — Fallback hybrid: MediaPipe primary, RTMPose when MediaPipe returns nothing.**

Adopted with `rtmpose-s@192x256`, and **scoped to the semantic
batsman-verification path only**. The scoring-model feature path stays on
MediaPipe. The scope split is the whole substance of the decision, so it is
stated precisely below.

### Why D and not the others

**Not A (keep MediaPipe).** MediaPipe drops 27.9% of frames overall and 50.8%
on the hard clips, and loses on every quality metric even on its own easier
frames (§6). The audit sheets show it also fails *silently* — on
`rishi_front_2` it returns a confident skeleton spanning two people. There is
a real deficiency to fix.

**Not B (replace wholesale).** Blocked by a hard constraint found in the
production code, not by preference: the 15 angles are computed in **3D** from
MediaPipe's `z`, and RTMPose 2D has no `z`. The angles differ by a **median of
30.4°** (§7). The `(7, 30)` tensor would keep its shape while its contents
shifted substantially — textbook train/serve skew against a frozen model.
Retraining is excluded by the stop condition. B is also unnecessary: MediaPipe
is not wrong on the 72% of frames it handles.

**Not C (RTMPose only for difficult crops).** This needs a definition of
"difficult" evaluated *before* running pose. The obvious candidate, subject
size, is a weak predictor: `pro_player_front_12` has perfect coverage and a
normal-sized batsman yet MediaPipe fails on 80% of its frames. D achieves the
same targeting with a trigger that is exactly correct by construction —
MediaPipe's own refusal — and needs no threshold.

**Not E (neither sufficient).** Too pessimistic. RTMPose recovers 8 of 10
sub-threshold clips and is visually correct where MediaPipe returns nothing.

### Exact scope

| Path | Uses `z`? | Decision |
|---|---|---|
| S2/S3 semantic batsman verification (`_pose_metrics`) | **No** — reads only `lm.x`, `lm.y` | **Adopt the hybrid now** |
| Scoring-model `(7, 30)` angles (`feature_engineering.py`) | **Yes** — 3D dot products | **Stay on MediaPipe** |

`_pose_metrics()` is purely 2D, so RTMPose is a genuine drop-in for the
identification path with no distribution shift — this is what makes the split
possible rather than arbitrary.

### Why `rtmpose-s` and not `m` or `x`

| | s | m | x |
|---|---|---|---|
| Latency vs MediaPipe | **7.2× faster** | 1.2× faster | 5.8× *slower* |
| Paired bone-length CV | 0.2502 | 0.2395 | 0.2306 |
| Self-verification AUC | **0.961** | 0.940 | 0.918 |

`s` is the best self-verifier, by far the cheapest, and its quality is within
noise of `m`. Capacity *hurts* verification here: null-box confidence rises
monotonically with model size, so bigger models hallucinate more confidently.
`x` costs 629 ms/frame/track on CPU for no proportional gain — unsurprising,
since 5 of 7 hard crops are already smaller than its input and are being
upsampled (§9).

### What this does not fix, and what to do next

1. **Two clips are tracking failures, not pose failures** (`sanjay_front _1`
   at 0.05 coverage, `kohli_side_16` at 0.25). No pose model reaches them.
2. **`k_pose_rate` conflates tracking with pose** (§3) and should be split
   into `coverage × P(pose | box)`. Until it is, the pipeline will keep
   misattributing tracking failures to the pose model — which is what sent
   this investigation after RTMPose in the first place.
3. **The unreachable-branch bug** in `categorise_failure()`: because
   `k_pose_rate ≤ coverage`, the pose test always fires first and
   `"tracking failure — batsman track too fragmented"` can never be returned.
4. **Three genuine pose failures were never labelled** (`kohli_side_06`,
   `pro_player_front_44`, `pro_player_front_12`), so the corpus of known
   failures was itself incomplete.

Items 2–4 are cheap corrections to the diagnostic layer and are worth more
than any further pose-model work: they change which clips the next experiment
is aimed at.

### Honest limitations

- All quality metrics are **ground-truth-free self-consistency** measures.
  There is no annotated keypoint ground truth in this corpus, so "better"
  means more self-consistent plus visually verified on a sample — not
  measured against true joint positions.
- The visual audit covers 6 frames on each of 7 clips. It is enough to
  establish that RTMPose's recoveries are real rather than hallucinated, and
  not enough to quantify accuracy.
- **CPU only.** No CUDA provider is available here. MediaPipe is also CPU, so
  the comparison is fair, but absolute latencies are not deployment figures.
- Resolution and capacity are **confounded** in the `x` arm (§9); no halpe26
  `rtmpose-m@384x288` exists to separate them.
- The 30.4° 3D-vs-2D figure says the two are far apart. It does **not**
  establish that MediaPipe's 3D angles are more *correct* — its `z` is a
  learned relative-depth estimate this experiment cannot validate.
