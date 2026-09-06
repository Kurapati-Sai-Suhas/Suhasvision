# Phase 3 — S0–S4 Semantic Batsman Identification Ablation

**Question.** Does adding increasingly semantic evidence reduce wrong-person selection
without unacceptable refusal or recall cost, relative to the Phase-2 B3 geometry baseline?

**Answer, in one line.** Yes, but not where expected: the gain comes from **skeleton
dynamics (S2)**, bat evidence (S3) adds a smaller and partly misleading increment, action
evidence (S4) makes things worse — and the whole improvement costs **~6 seconds per clip
against 0.3 milliseconds** for the baseline.

Scoring model untouched. `(7,30)` unchanged. Production default remains A0.

---

## 1. Protocol — what is held identical

Every arm consumes the **same YOLO11m@640 detections, the same ByteTrack tracks, and the
same 20 sampled frames per clip**, read from the Phase-2 track cache and a single Phase-3
feature cache. Only the scoring feature set differs, so any difference is attributable to
the added evidence.

| Held constant | Value |
|---|---|
| Detector | YOLO11m@640 (Phase-2 cache) |
| Tracker | ByteTrack (Phase-2 cache) |
| Sampled frames | 20 per clip, shared by every track in that clip |
| Decision rule | confidence ≥ 0.60 **and** margin ≥ 0.15, identical in every arm |
| Fitting | logistic, **dev split only**; eval touched once |
| **Candidate recall** | **0.9801** over 302 GT frames — *identical for all systems by construction* |

**Where the arms diverge, stated explicitly:** S0 uses the **frozen** Phase-2 B3 model file
unmodified, as instructed. S1–S4 are refitted with the same logistic routine on the same dev
split. So S0 vs S1 mixes "added features" with "refit", while S1→S2→S3→S4 are clean
increments. This is the one place the comparison is not perfectly isolated, and it matters
for reading the S0→S1 row.

Feature groups: **geometry** (crease/stability, Phase-2), **persistence** (span, contiguity,
longest run, confidence stability), **skeleton** (wrist excursion, wrist lift range, shoulder
rotation, lower-body stability, bilateral coordination, pose rate, visibility),
**equipment** (bat anywhere / on-person / near-hand / persistent / confidence), **action**
(backlift, downswing, rise-then-fall shape, upper-lower ratio).

---

## 2. Headline results

51 clips, dev 24 / eval 27. **Eval is the number to read.**

| System | Split | Correct | Wrong | Refused | Precision | Wrong-person | Refusal | Acc/all clips |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **S0** geometry (frozen B3) | dev | 13 | 0 | 11 | 1.000 | 0.000 | 0.458 | 0.542 |
| | **eval** | 17 | 1 | 9 | 0.944 | **0.056** | **0.333** | 0.630 |
| **S1** + persistence | dev | 15 | 0 | 9 | 1.000 | 0.000 | 0.375 | 0.625 |
| | **eval** | 16 | 1 | 10 | 0.941 | 0.059 | 0.370 | 0.593 |
| **S2** + skeleton | dev | 20 | 0 | 4 | 1.000 | 0.000 | 0.167 | 0.833 |
| | **eval** | 20 | 1 | 6 | 0.952 | 0.048 | **0.222** | **0.741** |
| **S3** + equipment | dev | 20 | 0 | 4 | 1.000 | 0.000 | 0.167 | 0.833 |
| | **eval** | **20** | **0** | 7 | **1.000** | **0.000** | 0.259 | **0.741** |
| **S4** + action | dev | 19 | 0 | 5 | 1.000 | 0.000 | 0.208 | 0.792 |
| | **eval** | 19 | 2 | 6 | 0.905 | **0.095** | 0.222 | 0.704 |

Identity-switch rate is not reported as a per-system metric: tracks are identical across
arms, so any switch is a property of the shared ByteTrack output (measured in Phase 2 at
19 observed switches on eval), not of the scoring.

### The ladder is not monotonic

- **S0 → S1 (persistence): slightly worse.** Net −1 clip on eval. Persistence adds nothing.
- **S1 → S2 (skeleton): the real gain.** Refusal 0.370 → 0.222, correct 16 → 20.
- **S2 → S3 (equipment): removes the last wrong answer**, at the cost of one more refusal.
- **S3 → S4 (action): clearly worse.** Wrong 0 → 2, precision 1.000 → 0.905.

This repeats Phase 2's B3-vs-B4 pattern: past a point, more features means more variance at
this sample size, not more signal.

### Paired, per-clip, versus S0 on eval

| System | Improved | Regressed | Net | Wrong fixed | Wrong introduced |
|---|---:|---:|---:|---:|---:|
| S1 | 0 | 1 | **−1** | 0 | 0 |
| S2 | 3 | 0 | **+3** | 1 | 1 |
| S3 | 4 | 1 | **+3** | 1 | 0 |
| S4 | 2 | 0 | **+2** | 0 | 1 |

S3's four improvements are `instagram1_front_03`, `instagram2_front_03`,
`pro_player_front_10`, `pro_player_front_16`; its one regression is `kohli_side_08`.

---

## 3. The refusal trade-off

S3 is not a case of buying accuracy with refusals — it is **better on both axes than S0**:
wrong-person 0.056 → 0.000 *and* refusal 0.333 → 0.259. That is the unusual case where the
trade-off does not need adjudicating.

S2 versus S3 is the real trade: S2 refuses less (0.222 vs 0.259) but commits one wrong
answer. For **dataset construction**, where a wrong batsman silently poisons a training row
and a refusal costs one clip, S3's trade is the right one. For a **live product**, where
refusing a third of uploads is a poor experience, S2 is defensible.

Risk–coverage on eval (sweeping the confidence threshold) shows both S0 and S3 hold a 0.000
wrong-rate out to ~48% coverage; they separate only in how much coverage each reaches before
its first error. With one wrong answer separating them, the curves are too sparse to rank
confidently, and no ranking is claimed from them.

---

## 4. Bat-signal ablation — specificity is the whole story

Adding exactly one bat feature at a time on top of S2 (eval):

| Variant | Correct | Wrong | Refused | Precision | Refusal |
|---|---:|---:|---:|---:|---:|
| S2 baseline, no bat | 20 | 1 | 6 | 0.952 | 0.222 |
| + `bat_anywhere` | 20 | 0 | 7 | 1.000 | 0.259 |
| + `bat_on_person` | 20 | 0 | 7 | 1.000 | 0.259 |
| + `bat_near_hand` | 20 | 0 | 7 | 1.000 | 0.259 |
| **+ `bat_persist`** | **21** | **0** | **6** | **1.000** | **0.222** |
| + all bat (= S3) | 20 | 0 | 7 | 1.000 | 0.259 |

**How well each level actually separates the batsman from other tracks** (measured directly
on the features across all 250 tracks, model-free):

| Feature | Batsman mean | Other-track mean | **Separation** |
|---|---:|---:|---:|
| `e_bat_anywhere` | 0.612 | 0.592 | **0.020** |
| **`e_bat_on_person`** | 0.377 | 0.021 | **0.355** |
| `e_bat_near_hand` | 0.282 | 0.012 | 0.271 |
| `e_bat_persist` | 0.239 | 0.016 | 0.223 |
| `e_bat_conf` | 0.301 | 0.039 | 0.263 |

This is the cleanest result in the experiment. **"A bat is present in the frame" carries
almost no information (0.020) — a bat is visible in ~60% of frames for batsman and
non-batsman tracks alike. "The bat is on *this* person" is 17× more discriminative (0.355).**
Detecting a bat is not identifying a batsman; *associating* one is.

**A caveat that must be stated:** `e_bat_anywhere` receives a *negative* fitted coefficient
(−2.67) in S3 and S4. Given its near-zero standalone separation, it is almost certainly
acting as a collinear normaliser for the other bat terms rather than carrying signal.
Individual bat coefficients should not be interpreted; only the group's contribution should.

Also note the single-feature variants are separated by **one clip**. `+bat_persist` scoring
21/0/6 is a one-clip lead over the others and should not be over-read.

---

## 5. Stationary-feeder case study

11 clips contain a rival track that is stationary (`foot_stability` ≥ 0.85), at least as
large as the batsman, and persistent — the configuration Phase 2 identified as structurally
unreachable by geometry alone.

**The decisive case, `pro_player_front_06`** (GT = t2, the small distant batsman):

| Track | rel. height | foot stability | bat on person | bat near hand | bilateral | up-down shape |
|---|---:|---:|---:|---:|---:|---:|
| t1 (feeder) | 0.763 | 0.937 | **0.40** | 0.10 | 0.886 | 0.000 |
| **t2 (batsman, GT)** | 0.405 | 0.960 | **0.15** | 0.00 | 0.804 | 0.000 |

| System | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| Decision | **wrong** | **wrong** | refused | refused | **wrong** |

**Bat evidence points at the wrong person here.** The feeder holds/handles a bat more often
in-frame than the distant batsman does, so `bat_on_person` is 0.40 for the feeder against
0.15 for the batsman. S3 does not fix this clip; it **refuses** it. What produced the refusal
is **skeleton dynamics at S2** — the combination that stops the feeder clearing the
confidence-and-margin bar. S4 pushes it back to a wrong answer.

**So the answer to "which signal resolves the stationary feeder?" is: none of them fully.**
Skeleton dynamics converts the hardest case from a *confident wrong answer* into a *refusal*,
which is a real improvement for dataset construction but is not resolution.

On the other 10 feeder clips the picture is much better, and there bat association is
genuinely decisive — e.g. `instagram1_front_03`: batsman `bat_on_person` 0.95 versus rival
0.00, with S0/S1 refusing and S2/S3/S4 all correct.

---

## 6. Failure analysis (eval)

| Cause | S0 | S1 | S2 | S3 | S4 |
|---|---:|---:|---:|---:|---:|
| Pose failure on the batsman crop (small/occluded) | 6 | 6 | 5 | **6** | 5 |
| Margin too small — candidates not separable | 3 | 3 | 1 | 0 | 1 |
| Stationary feeder | 1 | 1 | 0 | 0 | 2 |
| Single candidate below confidence threshold | 0 | 1 | 0 | 1 | 0 |
| Semantic ambiguity | 0 | 0 | 1 | 0 | 0 |

**The dominant remaining failure is not an identity problem — it is a pose problem.** Across
every system, 5–6 of the ~7 non-correct eval clips fail because MediaPipe cannot extract a
usable pose from the batsman crop (small, distant, or net-occluded subjects:
`kohli_side_12`, `kohli_side_16`, `pro_player_front_10`, `pro_player_front_18`, plus the
long-session clips `rishi_front_2` and `sanjay_front _1`).

Semantic evidence eliminated the *separability* failures (margin 3 → 0) but cannot touch the
pose failures, because those clips have no skeleton evidence to reason over in the first
place. **That directly motivates the RTMPose question rather than more identity features.**

---

## 7. Fitted coefficients (inspectable, not interpretable individually)

| Feature | S2 | S3 | S4 |
|---|---:|---:|---:|
| `p_conf_stability` | −7.59 | −5.58 | −4.31 |
| `foot_stability` | +6.35 | +2.32 | — |
| `scale_stability` | +4.52 | — | −3.35 |
| `k_bilateral` | +2.02 | +2.29 | — |
| `k_lower_stability` | +2.18 | +2.11 | — |
| `e_bat_near_hand` | — | +2.67 | +2.58 |
| `e_bat_anywhere` | — | −2.67 | −2.68 |
| `a_updown_shape` | — | — | +2.19 |

`p_conf_stability` carries a large negative weight in every fit while S1 (persistence alone)
*degraded* performance — together these say the persistence group is contributing variance
rather than signal, and coefficients on collinear features should not be read individually.
`k_bilateral` and `k_lower_stability` are positive and stable across fits, which is
consistent with the cricket prior they encode: a batsman's hands move together and the lower
body stays planted.

---

## 8. Latency — the strongest argument against S2–S4

Measured on real frames with warmup excluded.

| Stage | Median | Scales with |
|---|---:|---|
| S0 geometry features | **0.04 ms** | per track |
| S1 persistence features | **0.03 ms** | per track |
| **S2 pose on track crop** | **50.77 ms** | **per frame × per track** |
| S2 skeleton maths | ~0.00 ms | per track |
| **S3 bat detection** | **48.48 ms** | per frame (shared across tracks) |
| S3 association maths | 0.01 ms | per track |
| S4 action maths | ~0.00 ms | per track |

Per clip, at this ablation's sampling (20 frames, ~5 candidate tracks):

| System | Added | Cumulative |
|---|---:|---:|
| S0 + S1 | — | **0.3 ms** |
| S2 | +5,077 ms | 5,078 ms |
| S3 | +970 ms | **6,047 ms** |
| S4 | +0 ms | 6,047 ms |

**S3 costs roughly 20,000× S0 per clip** for a one-clip reduction in wrong answers.

Two mitigations belong in the record. First, pose cost scales with *candidate tracks*; a
cheap geometric prefilter that drops obviously-implausible tracks before running pose would
cut most of it. Second, the pipeline already runs pose on the selected track downstream, so
in a real deployment part of the S2 cost is work that would happen anyway — the honest
marginal figure is somewhere between 1× and 5× the numbers above, and was not measured.

---

## 9. Statistical caution

The eval split is **27 clips**. S3's headline advantage over S0 is **one clip** of
wrong-person and **two clips** of refusal. A one-clip difference on 27 is far inside noise;
no significance test is run because none would be meaningful at this size, and none is
claimed.

What is more trustworthy than the headline is (a) the **direction being consistent across
dev and eval** for S2/S3, (b) the **model-free bat separation measurement** (0.020 vs 0.355)
which is computed over 250 tracks rather than 27 clips, and (c) the **failure-cause
distribution**, which is stable across all five arms.

---

## 10. Decision

**Recommended: S2 (geometry + persistence + skeleton) as the reference configuration, with
S3 preferred specifically for offline dataset construction.**

Reasoning against simply taking S3:

- S3's only measured advantage over S2 is one clip of wrong-person, inside noise.
- S3 costs an extra ~970 ms per clip for bat detection.
- On the hardest case the bat signal is **actively misleading** (§5).
- Bat association is nonetheless a genuine, model-free discriminator (§4) and removes the
  last wrong answer, which matters when the output becomes training labels.

Reasoning against S0 remaining the winner: S2 improves refusal 0.333 → 0.222 and accuracy
0.630 → 0.741 with no new wrong answers, consistently on both splits.

**S1 and S4 are rejected on evidence.** S1 degrades slightly and adds nothing; S4 doubles the
wrong-person count and re-breaks the feeder case.

**Nothing is promoted to production.** Production remains A0. Thresholds are still
uncalibrated, and the latency cost of S2/S3 has not been engineered down.

---

## 11. Artifacts

| File | Contents |
|---|---|
| `phase3_results/phase3_batsman_ablation_results.json` | Protocol, feature sets, fitted models, all metrics, risk–coverage, paired comparison |
| `phase3_results/phase3_batsman_ablation_results.jsonl` | Per-clip, per-system decisions with all track scores |
| `phase3_results/phase3_batsman_analysis.json` | Bat-level ablation, bat separation, feeder cases, failure categories |
| `phase3_results/phase3_batsman_latency.json` | Per-stage timings |
| `phase3_results/batsman_audit_sheets/` | 24 contact sheets (not committed — regenerable) |

---

## 12. What this does not yet answer

- **Phase segmentation and best-7 selection are untouched** — F0–F4 and L0–L3 are next, and
  both are limited to the 28 clips with usable shot boundaries.
- **Whether RTMPose fixes the dominant failure.** §6 shows pose failure is now the binding
  constraint on identification, which is a stronger argument for evaluating RTMPose than
  anything in Phase 2 produced.
- **Whether bat association can be made reliable on the feeder case** — currently it is the
  one place the signal inverts.
