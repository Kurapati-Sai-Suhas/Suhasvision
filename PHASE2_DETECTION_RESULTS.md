# Phase 2 — Detection, Tracking and Batsman Identification

**Scope.** Replace the person-detection layer, add multi-object tracking, build a
track-level batsman-identification model with a cricket-geometric prior, annotate a real
ground-truth benchmark, and run a B0–B5 ablation. The Conv1D+BiLSTM+Attention scoring
model was not touched: not retrained, not rearchitected, no label or feature changes, and
the `(7, 30)` interface is unchanged.

---

## 1. Executive result

Phase 1 concluded that selection could not pick a batsman the detector never produced.
Phase 2 measured that directly and it is confirmed, decisively:

| | MediaPipe (current) | YOLO11m@640 |
|---|---|---|
| Batsman candidate recall (eval) | **0.465** | **0.987** |
| Recall on small/distant batsmen | **0.050** | **1.000** |
| People detected per frame | 0.93 | 1.84 |
| Detection latency (median) | 69.8 ms | **14.3 ms** |

The replacement is roughly twice as accurate and about five times faster. More pointedly:
**every single wrong-person error the MediaPipe baseline made was a case where the batsman
was never detected at all** — 6 of 6 on dev, 5 of 5 on eval. Not one was a selection
mistake. After swapping the detector that count drops to zero.

End-to-end, wrong-person rate on held-out eval falls from **20.0% to 5.6%**, and the fix
comes from the cricket-geometric prior rather than from tracking. Latency rises 15%
(69.8 → 80.6 ms/frame), and the bottleneck is now pose, not detection.

---

## 2. Root cause confirmed

Yes — detection recall really did explain the Phase-1 failures, and the evidence is
stronger than a correlation.

The B0 arm reproduces the Phase-1 pipeline (MediaPipe candidates + the size-dominance
selection rule). Its outcomes decompose as:

| Split | Wrong-person | of which the batsman was **never detected** |
|---|---|---|
| dev | 6 | **6 (100%)** |
| eval | 5 | **5 (100%)** |

There is no residual population of "detector found the batsman, selection picked the
bowler anyway". The selection logic was not choosing badly; it was ratifying the only
candidate it had. B0's measured eval wrong-person rate of **20.0%** is also consistent
with the independent 26.3% from the Phase-1 visual audit on a different 21-clip sample,
which is a useful cross-check that the new harness measures the same phenomenon.

One methodological note, because it changed the answer. The first run of this ablation
reported B0 at **0% wrong-person**, which contradicted Phase 1. The cause was in my own
accounting: when no track matched the annotated batsman, I bucketed the clip as "no ground
truth" rather than as a wrong answer. That silently rewards a detector for never finding
the batsman — an arm that detects nobody would score a perfect 0% error rate. Fixed in
`phase2_ablation._outcome()` and pinned by tests: committing to a person when no
ground-truth track exists is **wrong**, and only an explicit refusal is not an error.

---

## 3. Literature evidence

The search was deliberately narrow — what detector/tracker combination recovers a small
distant player in this footage while running locally — not another broad review.

**Tracking-by-detection is the right family for sports.** SportsMOT characterises sports
tracking by two properties that match this corpus: fast, variable-speed, non-linear motion,
and similar-but-distinguishable appearance (uniforms). It reports that tracking-by-detection
methods such as ByteTrack and OC-SORT generally outperform alternatives on it, while noting
that stronger motion models (e.g. MambaMOT, +8.2% HOTA over ByteTrack under the same
detector and association) do better still. That supports starting from ByteTrack and
treating motion modelling, not appearance, as the upgrade path.

**ReID earns its cost under frequent occlusion, which this corpus does not have.**
BoT-SORT adds an appearance embedding to re-identify objects across brief disappearances,
at roughly 30%+ more compute; the reported benefit is concentrated in scenes with frequent
mutual occlusion. In these clips the people are few, visually distinct (batsman in pads
versus a feeder in casual kit), and windows are ~3 seconds. That predicted ReID would buy
little here — and the measured comparison in §5 bears it out.

**Cricket-specific work overwhelmingly assumes the batsman is already isolated.** Published
cricket pose pipelines typically run MediaPipe or a CNN directly on footage framed on a
single player and then classify strokes; the reported accuracies concern stroke
classification, not subject identification. I found no cricket-specific published method
for deciding *which* person in a multi-person net scene is the batsman. That is the gap
this phase had to fill itself, and it is why §6–7 build a prior rather than cite one.

Sources:
- [SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Cui_SportsMOT_A_Large_Multi-Object_Tracking_Dataset_in_Multiple_Sports_Scenes_ICCV_2023_paper.pdf)
- [MambaMOT: State-Space Model as Motion Predictor for Multi-Object Tracking](https://arxiv.org/pdf/2403.10826)
- [BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/pdf/2206.14651)
- [Object Tracking with ByteTrack and BoT-SORT (Ultralytics)](https://academy.ultralytics.com/courses/yolo-in-production/tracking-with-bytetrack-and-botsort)
- [Enhancing Cricket Performance Analysis with Human Pose Estimation and Machine Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10422414/)
- [Pose Estimation for Analyzing Cricket Players Using CNN](https://link.springer.com/chapter/10.1007/978-3-032-08504-7_23)

---

## 4. Detector comparison

Measured on 303 annotated batsman frames across 51 clips. Primary metric is **Batsman
Candidate Recall**: the fraction of annotated batsman frames where some detection reaches
IoU ≥ 0.5 with the annotated box.

### Eval split (27 clips)

| Detector | Resolution | Batsman recall | Small-subject recall | People/frame | Duplicate rate | Tiny-box rate | s/frame |
|---|---|---|---|---|---|---|---|
| MediaPipe (current) | native | **0.465** | **0.050** | 0.93 | 0.000 | 0.000 | 0.0688 |
| YOLO11n | 640 | 0.962 | 0.950 | 1.96 | 0.000 | 0.000 | **0.0210** |
| YOLO11n | 960 | 0.937 | 0.950 | 2.20 | 0.000 | 0.003 | 0.0356 |
| **YOLO11m** | **640** | **0.987** | **1.000** | 1.84 | 0.000 | 0.000 | 0.0230 |
| YOLO11m | 960 | 0.975 | 0.950 | 1.83 | 0.000 | 0.000 | 0.0363 |
| YOLO11m | 1280 | 0.956 | 0.950 | 1.87 | 0.000 | 0.010 | 0.0575 |
| YOLO11x | 960 | 0.981 | 0.975 | 1.77 | 0.000 | 0.000 | 0.0617 |
| YOLO11x † | 1280 | 0.994 | 1.000 | 1.95 | 0.000 | 0.000 | 0.0935 |
| YOLOX-m (MMPose) ‡ | 640 | 0.937 | 0.900 | 1.53 | 0.000 | 0.000 | 0.1422 |

† Proposal detector — its recall is near-1.0 **by construction** and is shown as the
annotation ceiling, not as a contender. See §8.
‡ CPU (ONNX Runtime has no CUDA provider here). Its **recall is comparable; its runtime is
not**, and must not be read against the CUDA rows.

Dev split, for the three headline configurations: MediaPipe 0.424 / small 0.094;
YOLO11m@640 0.993 / small 1.000; YOLOX-m@640 0.965 / small 0.875.

**Resolution: smaller is better here, which is not the naive expectation.** YOLO11m goes
0.987 → 0.975 → 0.956 as input grows 640 → 960 → 1280. Upscaling adds no information: 77
of 102 clips are natively 640×360, so 1280 is pure interpolation, and it costs 2.5× the
runtime while slightly *increasing* the tiny-box rate. **640 is the recommended
resolution** — the task asked for the smallest resolution that recovers the distant
batsman reliably, and that is it.

**On RTMDet.** The shortlist named RTMDet and it could not be benchmarked as specified.
rtmlib publishes no RTMDet *person* model — its only RTMDet ONNX is for hands, and
`rtmdet-nano-person.zip` returns 404 on both the openmmlab and HuggingFace mirrors. The
genuine mmdet route requires mmcv, which has no prebuilt wheel for torch 2.11 + cu128 on
Windows. I substituted **YOLOX-HumanArt**, which is the person detector MMPose's own
top-down RTMPose pipelines actually ship, so the "MMPose-ecosystem detector" slot is
filled by a genuinely different architecture rather than left empty or faked.

**False positives are not reported as a rate.** A true FP rate needs every person in every
frame annotated, which this benchmark does not have; inventing one would be dishonest. Two
measurable proxies are reported instead — duplicate boxes (IoU > 0.7 with another
detection in the same frame) and sub-24px detections — and both are ~0 for every YOLO
configuration. One concrete FP was observed by eye: on `pro_player_front_42` a **dog** is
detected as a person at confidence 0.61.

---

## 5. Tracker comparison

Detector fixed at YOLO11m@640 so the arms differ only in association.

| Tracker | Split | Batsman track recall | Best-track coverage | Observed ID switches | Tracks/clip | Fragmentation | s/frame |
|---|---|---|---|---|---|---|---|
| **ByteTrack** | dev | 0.979 | **0.896** | **7** | 3.33 | 0.53 | 0.0720 |
| **ByteTrack** | eval | **0.981** | 0.817 | 19 | 6.30 | 2.18 | **0.0595** |
| BoT-SORT | dev | **0.993** | 0.882 | 10 | 3.25 | **0.48** | 0.0814 |
| BoT-SORT | eval | 0.975 | **0.835** | **17** | 4.70 | 3.42 | 0.0707 |

The two are near-equivalent and neither dominates: ByteTrack wins recall and speed on
eval, BoT-SORT wins coverage and switch count on eval, and they swap on dev. BoT-SORT's
ReID buys about 2pp of coverage and two fewer switches for ~19% more runtime — consistent
with the literature position that appearance features pay off under frequent occlusion,
which this corpus does not exhibit. **ByteTrack is selected**, on speed and simplicity,
with the honest note that the measurement does not establish it as more accurate.

**MOTA / HOTA / IDF1 are not reported.** They require every identity labelled in every
frame. This benchmark has sparse, single-subject annotation, so those numbers are not
computable and are absent rather than approximated. "Observed ID switches" is a **lower
bound**: a switch that happens and reverts between two annotated frames is invisible here.

A real limitation surfaced: best-track coverage is only **0.817–0.835** even though
detection recall is 0.98. The batsman is detected but the single best track loses them
roughly one annotated frame in six. Track continuity, not detection, is now the weaker link.

---

## 6. Batsman-selection algorithm

`score_batsman_track(track, scene_context)` returns a score with its components exposed, so
every decision can be explained rather than asserted:

```python
{"track_id": 2, "score": 0.7551, "confidence": 0.7551,
 "components": {"coverage": ..., "foot_stability": 0.9681, "scale_stability": 0.9214,
                "centroid_stability": ..., "relative_height": 0.3345,
                "centrality": ..., "vertical_position": ..., "mean_conf": ...}}
```

Features are normalised to [0,1] and expressed in the track's **own body-height units**, so
they transfer across the corpus's mixed resolutions (640×360 landscape through 1080×1920
portrait) and do not become proxies for camera distance.

Weights are **fitted on the dev split only** by logistic regression (plain gradient
descent, L2-regularised; sklearn is not installed and eight features do not justify adding
it). Hand-picked weights are how the size-dominance rule acquired its unearned authority
in the first place.

**Fitted coefficients** (positive raises the batsman score):

| Feature | Full model | Geometry-only model |
|---|---|---|
| foot_stability | +6.372 | **+9.878** |
| scale_stability | −6.312 | +2.148 |
| coverage | +4.933 | +5.481 |
| mean_conf | +3.726 | — |
| centrality | +3.077 | — |
| relative_height | +2.381 | — |
| vertical_position | −0.718 | +1.320 |
| centroid_stability | +0.333 | −1.462 |
| (intercept) | −10.049 | −15.408 |

`foot_stability` dominates both fits. **`scale_stability` flips sign between them
(−6.31 → +2.15), which is collinearity with `foot_stability`, not a finding** — direct
evidence that the eight-feature model is over-parameterised for 24 positive examples. The
individual coefficients of the full model should not be interpreted.

### Selection accuracy (eval, 27 clips)

| Rule | Correct | Wrong | Refused | Precision when committed | Accuracy over all clips |
|---|---|---|---|---|---|
| **Geometry-only model** | 17 | **1** | 9 | **0.944** | 0.630 |
| Full model | 18 | 2 | 7 | 0.900 | 0.667 |
| Phase-1 size dominance | 16 | 4 | 7 | 0.800 | 0.593 |
| "largest person" | 15 | 12 | 0 | 0.556 | 0.556 |
| "most central" | 12 | 15 | 0 | 0.444 | 0.444 |
| "most persistent" | 22 | 5 | 0 | 0.815 | **0.815** |

Two honest observations. First, **"most persistent" — simply the track present in the most
frames — is a strong baseline**, beating both fitted models on clips-correct because it
never refuses. It is not better for the purpose: it commits on everything and contaminates
5 clips, versus 1 for the geometry model. Second, on dev the geometry model refuses 11 of
24 and the full model 5 of 24, so the refusal rate is high and not stable across splits.
Thresholds were hand-set (0.60 confidence, 0.15 margin) and deliberately **not** tuned —
tuning them on eval is exactly the anti-bias violation the brief forbids.

The pipeline can say "I don't know": verdicts are `CONFIDENT_BATSMAN` / `AMBIGUOUS` /
`NO_BATSMAN`, and only the first names a track. Across all 51 clips the geometry model
returns 31 confident, 10 ambiguous, 10 no-batsman.

Detector confidence, track confidence, batsman confidence, pose confidence and contact
confidence are kept as separate concepts throughout and are never collapsed — Phase 1
established that contact confidence is *anti*-correlated with subject correctness.

---

## 7. Cricket-specific geometric prior

The prior is **calibration-free by design**: no homography, no known camera pose, no stump
detection, and — critically — **no assumption about whether the batsman is near or far**.
That assumption is precisely what Phase 1 got wrong.

The insight is that a batsman is **stationary in depth**. They play from a crease, so:

- their **foot line** (box bottom edge) stays at a near-constant image height, because
  their feet stay on one spot of the ground plane;
- their **apparent size** barely changes.

A bowler or feeder approaching the camera violates both: their foot line sweeps down the
image and their box height grows substantially. Both quantities are measured in the
track's own body-height units, so they are invariant to how far away the person is.

This is what the ablation isolates in B2 → B3, and it is where the wrong-person improvement
actually comes from (0.200 → 0.056), not from tracking.

**Where the prior fails, precisely.** `pro_player_front_06` is the single remaining eval
error. The near-camera person there is a *stationary feeder* who throws from a fixed spot —
so his foot stability is 0.937, nearly a batsman's, and the geometry cannot separate him.
The prior distinguishes a **moving** bowler from a batsman; it does not distinguish a
**stationary non-batsman**. Separating those requires evidence about what the person is
*doing* (skeleton dynamics) or *holding* (a bat), which this phase did not implement.

**Not implemented, and not claimed:** stump/crease detection, bat detection, and skeleton
dynamics as scoring features. The spec listed them as candidate signals; they are absent
here rather than faked, and §15 recommends them.

---

## 8. Benchmark methodology

**Corpus.** 51 clips sampled at stride 2 from the 102-clip corpus, spanning every source
group and all camera angles. Split **dev 24 / eval 27** by a stable SHA-256 hash of the
clip name, so the split cannot drift as clips are added and cannot be re-rolled to flatter
a result.

**Annotation.** 6 evenly-spaced frames per clip; **303 annotated batsman frames**. Labelling
is *proposal-assisted*: a high-recall detector's candidates are rendered as separately
cropped, numbered thumbnails and the annotator records which crop is the batsman.
Verifying a proposal is far cheaper than drawing a box and yields a pixel-accurate box for
free.

An earlier version drew all boxes on one shared image; overlapping candidates made index
assignment guesswork, so it was rebuilt as per-candidate crops before any labelling was
kept.

**The bias this introduces, and how it is handled.** Ground truth anchored to a detector
cannot contain a batsman that detector missed, which would flatter it. Mitigations:
1. the proposal detector is the highest-recall configuration available (YOLO11x@1280,
   conf 0.06) and is **excluded from the recommendation** — it appears in §4 only as the
   annotation ceiling;
2. frames where the batsman is visible but *unboxed* are recorded as `proposal_miss`, stay
   in the recall **denominator**, and are unscorable by every detector including the
   proposal detector. One such frame exists (`rishi_front_2`, f=3724);
3. frames with no batsman present are marked `ABSENT` (3 frames) and excluded entirely —
   conflating "absent" with "missed" would either invent misses or hide them.

**Residual bias, stated plainly:** a batsman that neither the proposal detector nor the
annotator noticed is invisible to this benchmark. That cannot be corrected without
exhaustive manual annotation.

**Annotator limitations:** single annotator, not blind to the proposals, no second rater,
no inter-rater agreement. Supports subject-identity metrics and nothing finer.

**Not annotated, therefore not computed:** shot boundaries and contact frames. tIoU,
contact-frame error and E2E-VSER remain uncomputable, exactly as in Phase 1.

---

## 9. B0–B5 ablation

Each arm adds exactly one thing. All arms are evaluated on the same annotated frames.

### Eval (27 clips, held out)

| Arm | What it adds | Candidate recall | Correct | Wrong | Refused | **Wrong-person rate** | Correct / all clips |
|---|---|---|---|---|---|---|---|
| **B0** | MediaPipe + Phase-1 selection | 0.468 | 20 | 5 | 2 | **0.200** | 0.741 |
| **B1** | + YOLO11m@640 detector | **0.994** | 16 | 2 | 9 | 0.111 | 0.593 |
| **B2** | + ByteTrack | 0.981 | 16 | 4 | 7 | **0.200** | 0.593 |
| **B3** | + crease geometry | 0.981 | 17 | **1** | 9 | **0.056** | 0.630 |
| **B4** | + full feature set | 0.981 | 18 | 2 | 7 | 0.100 | 0.667 |
| **B5** | + MediaPipe pose on crop | 0.981 | 18 | 2 | 7 | 0.100 | 0.667 |

### Dev (24 clips)

| Arm | Candidate recall | Correct | Wrong | Refused | Wrong-person rate |
|---|---|---|---|---|---|
| B0 | 0.424 | 18 | 6 | 0 | 0.250 |
| B1 | 0.993 | 17 | 1 | 6 | 0.056 |
| B2 | 0.979 | 19 | 0 | 5 | 0.000 |
| B3 | 0.979 | 13 | 0 | 11 | 0.000 |
| B4 | 0.979 | 19 | 0 | 5 | 0.000 |

**What each step actually bought:**

- **B0 → B1 (detector): the decisive step.** Candidate recall 0.468 → 0.994. Wrong-person
  halves to 0.111 purely because the batsman is now *available* to be chosen.
- **B1 → B2 (tracking): no improvement in subject accuracy.** Wrong-person goes *back up*
  to 0.200 — identical to B0 — because the selection rule is still size dominance.
  **Tracking alone does not fix wrong-person selection.** It is a prerequisite for
  track-level scoring, not a fix in itself.
- **B2 → B3 (crease geometry): the second decisive step.** 0.200 → 0.056. This is where
  the wrong-person improvement comes from.
- **B3 → B4 (more features): worse.** 0.056 → 0.100. Adding size and centrality back
  re-imports the failure mode the geometry prior was built to remove, consistent with the
  coefficient instability in §6.
- **B4 → B5 (pose on crop): no change to selection**, by construction; it measures pose
  feasibility (§13, Question F).

**Statistical caution.** These are 27 eval clips. B3 versus B4 is 1 wrong versus 2 wrong —
a one-clip difference, well inside noise. The B0 → B3 improvement (5 → 1) is larger and
directionally consistent across both splits, but no significance test on 27 clips would be
meaningful and none is claimed.

---

## 10. Failure examples

**Fixed — `pro_player_front_14` (the canonical Phase-1 failure).** Phase 1 tracked the
feeder here and assigned contact confidence 0.85 to a throwing action. Phase 2 selects
track t2, the distant batsman in teal at the far stumps, coinciding with the annotated box,
and rejects t6 — the near feeder — *despite t6 being 2.6× larger* (`relative_height` 0.87
versus 0.33). The trajectory panel shows exactly why: t2 is a tight stationary blob at the
crease while t5/t6 sweep across the scene. Verdict `CONFIDENT_BATSMAN`, confidence 0.755,
margin 0.422.

**Still failing — `pro_player_front_06`.** The only eval clip B3 gets wrong, and it does so
confidently (0.617, margin 0.245 — just over both thresholds). The near-camera person is a
*stationary feeder* throwing from a fixed spot: foot stability 0.937, essentially a
batsman's. The geometric prior separates a moving bowler from a batsman but cannot separate
a stationary non-batsman. This is the clearest signpost for Phase 3.

**Regression — `kohli_side_12` and `pro_player_front_10`.** B0 gets both right; B2 gets both
wrong. B3 *refuses* both rather than committing, which is the intended behaviour — the
geometry model declines when it cannot separate, and refusing beats guessing for dataset
construction.

**Detector false positive — `pro_player_front_42`.** A dog is detected as a person at
confidence 0.61. Harmless here (it loses on every geometric feature) but a real reminder
that "person" recall is not "player" recall.

**Long-session clips — `sanjay_front_1`, `rishi_front_2`.** Both correctly return
`NO_BATSMAN` (confidence 0.025 and 0.041). These are 200s and 78s whole net sessions, not
single shots; the pipeline declining them is correct behaviour.

---

## 11. Regression analysis

Nothing broke.

- **198 tests pass**, up from 165 (+33 new Phase-2 tests). No pre-existing test was
  weakened, skipped or deleted.
- **No Phase-1 or production file was modified in this phase.** `git status` shows the only
  modified files are the three changed during Phase 1; every Phase-2 artefact is a new
  `phase2_*` file.
- **`(7, 30)` feature interface is unchanged** — `SEQ_LEN = 7`, `len(EXPECTED_FEATURES) = 30`,
  `schema.py` untouched.
- **Production behaviour is unchanged**: `ACTIVE_CONFIG` still resolves to **A0**. None of
  the Phase-2 stack is wired into `zero_storage_pipeline` or the Django serving path. It is
  measured, not deployed.
- **The scoring model was not touched** — not retrained, no architecture, label or feature
  changes.

New tests specifically pin the accounting bug from §2 (`_outcome`), the scale-invariance of
every geometric feature, that "largest person" picks the wrong track in the front-view
configuration, and that an `AMBIGUOUS` verdict never names a track.

---

## 12. Latency

Measured on 96 real frames with 10 warmup iterations excluded (warmup loads weights,
allocates the CUDA context and triggers autotuning; including it misattributes one-off
setup to per-frame cost).

| Stage | Mean ms | Median ms | p90 ms |
|---|---|---|---|
| MediaPipe detection+pose (B0 baseline) | 67.89 | **69.76** | 73.55 |
| YOLO11m@640 detection | 15.03 | **14.31** | 15.97 |
| YOLO11m@640 + ByteTrack | 14.70 | 14.60 | 15.81 |
| MediaPipe pose on crop | 62.38 | **65.96** | 71.53 |

- **ByteTrack association overhead: 0.29 ms/frame** — effectively free.
- **Detection is 4.9× faster** than the MediaPipe baseline (14.3 vs 69.8 ms).
- **Recommended stack total: 80.6 ms/frame** vs **69.8 ms** for B0 — **+15%**.

The comparison is fair but needs one clarification: MediaPipe's `PoseLandmarker` produces
detection *and* landmarks in one call, so B0's 69.8 ms covers both. The new stack pays
14.6 ms for detection+tracking and then 66.0 ms for pose on the selected crop.

**The bottleneck has moved.** Pose is now 66 of 81 ms — 82% of the budget — while detection
is 18%. Any further latency work belongs in the pose stage, not the detector. For contrast,
Phase 1 added +1.56 s/clip for no measurable reliability gain; Phase 2 adds ~11 ms/frame
for a 3.6× reduction in wrong-person rate.

---

## 13. Recommended production configuration

**B3: YOLO11m @ 640, conf 0.10 → ByteTrack → geometry-only batsman scorer, with refusal
enabled.**

| | Value (eval) |
|---|---|
| Batsman candidate recall | 0.981 |
| Wrong-person rate | **0.056** |
| Clips committed | 18 of 27 |
| Clips refused | 9 of 27 |
| Latency | 80.6 ms/frame (+15% over B0) |

Rationale. The objective is maximum *correct* extraction at an acceptable rejection rate,
not maximum acceptance. For dataset construction the cost asymmetry is stark: a wrong
batsman silently poisons a training row, while a refused clip costs one clip. B3 yields 17
clean clips with 1 contaminated; B0 yields 20 with 5; "most persistent" yields 22 with 5.
B3 gives by far the cleanest data.

B4 is the alternative if throughput matters more than purity (18 correct, 2 wrong, 7
refused). I do not recommend it: its extra features are the ones that re-import the
size-dominance failure, and its coefficients are demonstrably unstable.

**Do not deploy yet.** The refusal rate differs sharply between dev (11/24) and eval (9/27),
so the confidence and margin thresholds need calibrating on dev before this replaces A0 in
production. That calibration is deliberately not done here, because doing it against the
eval numbers in this report would invalidate them.

---

## 14. Remaining limitations

1. **Stationary non-batsmen defeat the geometric prior** (`pro_player_front_06`). A feeder
   throwing from one spot is geometrically indistinguishable from a batsman.
2. **Refusal rate is high and unstable** — 33% on eval, 46% on dev. Thresholds are
   uncalibrated by design.
3. **Track continuity, not detection, is now the weak link.** Best-track coverage is
   0.817–0.835 against 0.98 detection recall.
4. **Benchmark is small.** 51 clips, 303 frames, 24 positive training examples. The
   eight-feature model is over-parameterised for it, as the coefficient sign flip shows.
5. **Single, non-blind annotator.** No second rater, no inter-rater agreement.
6. **Proposal-anchored ground truth.** A batsman missed by both the proposal detector and
   the annotator is invisible to every number here.
7. **No shot-boundary or contact annotation**, so tIoU, contact error and E2E-VSER remain
   uncomputable — unchanged from Phase 1.
8. **Bat evidence and skeleton dynamics were not implemented**, so two of the spec's
   candidate signals are untested rather than rejected.
9. **RTMDet was not benchmarked** (unavailable weights, unbuildable dependency); YOLOX-m
   stands in for the MMPose family, on CPU, so its runtime is not comparable.
10. **Long-session clips are stride-subsampled** to a 700-frame budget, weakening temporal
    association on exactly the clips that are not single shots.
11. **Nothing is wired into production.** All results are offline measurements.

---

## 15. Recommendation for Phase 3

**Priority 1 — resolve the stationary non-batsman with action/appearance evidence.** This
is the only failure mode the geometric prior structurally cannot address, and it is now the
dominant residual error. The cheapest credible signal is a **bat detector** (a small
fine-tuned YOLO on bat/pads), because "holds a bat and wears pads" separates batsman from
feeder in exactly the case geometry cannot. Skeleton-dynamics features (wrist excursion,
shoulder rotation, lower-body stability) are the alternative and become nearly free once
pose runs on the selected crop.

**Priority 2 — calibrate thresholds on dev and re-report on eval.** A 33–46% refusal rate
is the largest practical cost of the recommended configuration and is almost certainly
reducible, since dev shows the model achieving 1.0 precision when it commits.

**Priority 3 — improve track continuity, not the tracker.** Best-track coverage of ~0.82
against 0.98 detection recall is the gap. The literature points at motion modelling rather
than appearance for sports, so this is a matter of association tuning and gap-bridging, not
of adopting ReID.

**Priority 4 — annotate shot boundaries and contact frames** on the existing 51 clips. This
is the last blocker on tIoU, contact error and E2E-VSER, and it also unblocks a real answer
to Question G.

**Explicitly not recommended:** BoT-SORT/ReID (measured as no better here), higher detection
resolution (measured as worse), and retraining the scoring model (upstream identity is not
yet trustworthy enough to justify it).

---

## Final questions

**A — Did a real multi-person detector materially increase batsman candidate recall?**

**Yes, overwhelmingly.** 0.465 → 0.987 on eval (0.424 → 0.993 on dev). On small/distant
batsmen — the Phase-1 failure mode — recall goes from **0.050 to 1.000**. People detected
per frame roughly doubles (0.93 → 1.84). This is the single largest effect measured in
either phase.

**B — Did tracking reduce identity instability?**

**Partially, and less than expected.** ByteTrack holds a track on the batsman for 98.1% of
annotated frames, but the *single best* track covers only 81.7% — so identity is still lost
about one frame in six. Observed ID switches are 19 on eval (a lower bound, given sparse
annotation). Critically, **tracking did not improve subject accuracy at all**: B1 → B2
leaves wrong-person unchanged at 0.200. Tracking is a prerequisite for track-level scoring,
not a fix on its own.

**C — Did crease-aware geometry outperform size dominance?**

**Yes, clearly.** On the same tracks, size dominance gives 0.800 precision and a 0.200
wrong-person rate; the geometry model gives **0.944 precision and 0.056**. `foot_stability`
is the dominant fitted coefficient (+9.878). The B2 → B3 step is where the wrong-person
improvement actually happens. The caveat is that geometry buys this partly by refusing
more (9 of 27 versus 7).

**D — Did track-level batsman scoring reduce wrong-person selections?**

**Yes.** End to end, 0.200 (B0) → 0.056 (B3), a 3.6× reduction on held-out data. But the
credit belongs to the geometric prior, not to the scoring machinery: the full-feature model
is *worse* (0.100) than the geometry-only model, and a trivial "most persistent track"
baseline reaches 0.815 precision. Sophistication is not what helped; the right prior is.

**E — Does the new detector + tracker architecture outperform A0 on held-out data?**

**On correctness yes, on yield no, and the trade is worth making.** Wrong-person 0.200 →
0.056. But B0 commits to 25 of 27 clips while B3 commits to 18, so B0 produces more usable
clips in absolute terms (20 correct versus 17) — at the cost of 5 contaminated rows versus
1. For building a training dataset, where a wrong subject silently corrupts a label, B3 is
clearly better. If raw throughput mattered more, B0 would look competitive.

**F — Should RTMPose be introduced now, or is MediaPipe sufficient once detection is fixed?**

**Not yet — but it is the natural next lever.** Following §19, MediaPipe pose was tested on
YOLO crops first: success rate **0.814 on eval / 0.768 on dev**. That is workable, so
replacing the pose model is not *required* to make the pipeline function. Two facts argue
for revisiting it soon: pose is now **82% of the latency budget** (66 of 81 ms), and ~19%
of crops still yield no pose. RTMPose is a strong candidate on both counts — it is a
top-down model designed for exactly this crop-and-estimate pattern — but it should be
adopted on a measured comparison, not on reputation.

**G — Should shot localization happen before or after batsman identification?**

**The data does not support moving it, and my hypothesis was wrong.** I predicted that
global-frame motion is bowler-contaminated, so peak disagreement between global and
batsman-restricted motion would be large on multi-person clips and small on single-person
ones. Measured across 31 single-shot clips: multi-person median disagreement **17 frames**,
single-person median **17 frames** — identical. The prediction failed. Worse for the
hypothesis, the global signal has a *sharper* peak (prominence 4.66 versus 3.03).

Two caveats keep this from being a final verdict. My batsman-motion signal resizes each
crop to a fixed 64×128, which normalises away whole-body translation and so may suppress
the very displacement a stroke produces — that is a flaw in my implementation, not
necessarily in the idea. And with no annotated shot boundaries, localization *accuracy* is
not measurable at all; only disagreement and peak sharpness are. **Recommendation: keep
shot localization where it is, and re-test properly once contact frames are annotated
(Phase-3 priority 4).**

**H — What is the exact Phase-3 priority?**

**Add bat/pad evidence to the batsman score.** The geometric prior has taken wrong-person
from 20% to 5.6% and the single remaining eval failure is a *stationary feeder* that
geometry structurally cannot separate from a batsman. "Holds a bat and wears pads" resolves
exactly that case, is cheap (a small fine-tuned detector, or pose-derived features on a
crop that is already being computed), and attacks the dominant residual error rather than a
speculative one. Threshold calibration is a close second because the 33% refusal rate is
the main practical cost of the recommendation.
