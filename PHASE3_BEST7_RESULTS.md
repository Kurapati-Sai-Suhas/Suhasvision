# Phase 3D — Event Search Region + Best-7 Selection (F0–F4)

**Question.** Does contact + event structure improve Best-7 selection enough to
justify keeping these components?

**Answer.** Yes for the thing Best-7 exists to do — **F4 produces materially
better biomechanical `(7,30)` sequences** — but with one qualification that
must travel with the result: **only 41.6% of F4's frames fall inside the
labelled shot**, and that is worse than two simpler methods.

**Scoring model untouched.** `SEQ_LEN = 7`, 30 features, `(7,30)`. No retraining.

Tags: **[BENCHMARK]** · **[INFERENCE]** · **[NOT PROVEN]**

---

## 1. Frozen interface

```
S2  batsman identification      (committed, MediaPipe-only features)
L3  soft motion prior           -> soft_start, soft_end, shot_prior_confidence
L0  whole-clip fallback
C0  contact anchor              (machine proposal; see PHASE3_CONTACT_RESULTS)
```

L3 is exposed as a **soft prior, never a crop**. Its recall@IoU0.5 was 0.36, so
hard-cropping would discard the true shot on most clips.

`shot_prior_confidence` is L3's peak/median prominence.

---

## 2. Event search region

```
region = pad( L3 interval  ∪  [contact − 16, contact + 16] , 8 )
```
falling back to the L3 prior alone when contact is unavailable, and to the whole
clip when neither exists.

| | Value |
|---|---|
| **Contains the GT shot** | **100.0%** of 28 valid clips |
| Mean region width | 62.6 frames |
| Mean GT shot width | 22.7 frames |

The region is deliberately **2.8× wider** than the true event. That is the
correct objective at this stage — recall before selection — and §6 shows it is
also the source of the main limitation. **[BENCHMARK]**

**Candidate pool:** mean 31.9 per clip (min 0, max 40), inside the 20–50 target.
Frames without a batsman box are excluded — a frame with no subject cannot
contribute a biomechanical pose.

Per candidate: `frame, t, batsman_confidence, pose_quality, sharpness,
occlusion, motion, contact distance, temporal stability, bat confidence`.

**Motion is not quality.** They are separate fields throughout. `motion` says a
frame is informative about *when*; `sharpness` (variance of Laplacian) and
`pose_quality` (landmark visibility) say whether it is usable for
*biomechanics*. F2 below is what happens when the two are conflated.

---

## 3. Structural results (valid clips)

| Method | sel | ref | dup | min gap | contact-adj | pre | post | **pose quality** | sharpness | latency |
|---|---|---|---|---|---|---|---|---|---|---|
| F0 uniform | 24 | 4 | 0.00 | 9.7 | 0.6 | 3.6 | 3.2 | 0.543 | 0.765 | 0.027 ms |
| F1 contact-anchored | 24 | 4 | 0.04 | 4.9 | 1.1 | 5.4 | 1.0 | 0.543 | 0.772 | 0.028 ms |
| F2 motion-top | 24 | 4 | 0.00 | **1.3** | 1.7 | 4.0 | 2.8 | **0.505** | 0.776 | 0.011 ms |
| F3 phase-aware | 23 | 5 | 0.00 | 3.0 | **1.9** | 5.0 | 1.6 | 0.567 | 0.765 | 0.096 ms |
| **F4 optimized** | 23 | 5 | 0.00 | 4.6 | 1.0 | 4.8 | 2.2 | **0.798** | **0.778** | 0.939 ms |

Every method is chronological with no duplicates except F1 (0.04). F2's min gap
of 1.3 frames is the clustering failure the quality/motion distinction predicts.
**[BENCHMARK]**

---

## 4. Biomechanical `(7,30)` quality — the actual question

| Method | frames yielding angles (of 7) | missing | implausible angle rate | mean ang. velocity | **velocity stability** | abrupt jumps | fabricated zero-velocity |
|---|---|---|---|---|---|---|---|
| F0 | 4.79 | 2.21 | 0.012 | 19.45 | 10.49 | **3.35** | 0 |
| F1 | 4.88 | 2.13 | 0.015 | 23.18 | 10.27 | 4.62 | 0 |
| F2 | 4.46 | 2.54 | **0.007** | 21.37 | 12.58 | 5.16 | 0 |
| F3 | 4.74 | 2.26 | 0.012 | 21.93 | 9.52 | 4.38 | 0 |
| **F4** | **5.78** | **1.22** | 0.010 | 19.80 | **7.92** | 4.00 | 0 |

**F4 wins decisively on the metrics that decide whether a `(7,30)` tensor is
usable:**

- **~1 extra usable frame per sequence** (5.78 vs 4.46–4.88 of 7)
- **Missing landmarks nearly halved** (1.22 vs 2.13–2.54)
- **Best velocity stability** (7.92 vs 9.52–12.58) — this is the half of the
  tensor that is angular velocity, so noise here is fabricated kinematics
- **Never selects a pose-less frame** (0.0) while **F0 selects 2.1 per clip**
- Minimum pose quality 0.655 vs F0's 0.237; mean occlusion 0.645 vs 0.786

No method produces fabricated zero-velocity steps, so the Phase-1
duplicate-frame defect has not reappeared. **[BENCHMARK]**

F0 has the *fewest* abrupt jumps (3.35) purely because uniform spacing over a
wide region yields far-apart, smoothly-varying frames — it buys smoothness by
ignoring the event. **[INFERENCE]**

---

## 5. Failure analysis (valid clips)

| Cause | F0 | F1 | F2 | F3 | **F4** |
|---|---|---|---|---|---|
| Pose failure (3+ frames without landmarks) | 10 | 8 | 11 | 11 | **4** |
| Identity failure (upstream) | 4 | 4 | 4 | 4 | 4 |
| Excessive clustering | 0 | 0 | 9 | 0 | 0 |
| Optimizer refusal | 0 | 0 | 0 | 1 | 1 |
| **Total affected** | 14 | 12 | 24 | 16 | **9** |

**F4 more than halves F0's pose failures (4 vs 10)** — the pose-quality floor
works. F4's single refusal is explicit and diagnosable ("no chronological
assignment satisfies min_gap=2 across 10 candidates"), not a silent
degradation. Identity failures are constant at 4 across all methods, correctly
attributed upstream rather than to the selector. **[BENCHMARK]**

---

## 6. The qualification: event faithfulness

Fraction of the seven selected frames that fall **inside the labelled shot**:

| Method | in-shot fraction | median distance to shot |
|---|---|---|
| F2 motion-top | **0.571** | 0.0 |
| F3 phase-aware | 0.497 | 1.0 |
| **F4 optimized** | 0.416 | 2.0 |
| F1 contact-anchored | 0.369 | 5.0 |
| F0 uniform | 0.345 | 6.0 |

**F4 is worse than F2 and F3 here.** It optimizes pose quality and diversity
across a region 2.8× wider than the true event, so it legitimately prefers a
sharp, unoccluded frame outside the shot to a blurred one inside it.

**The cause is the region's width, not the selector.** The region is wide
because L3 is weak (recall@0.5 = 0.36), and it must stay wide to keep 100% GT
containment. Narrowing it requires better shot localization, not a different
Best-7 objective. **[INFERENCE]**

Visually confirmed in `best7_timeline_F4_optimized.png`: on
`instagram1_front_01` and `kohli_side_08` the L3 prior sits far from the GT
shot while the contact anchor is correct, and F4's frames spread across the
whole region.

---

## 7. Human reference — a proxy, and a biased one

**No expert frame-selection reference exists for this corpus and one cannot be
manufactured.** Asking which seven frames best represent a batting action for
biomechanical assessment requires a coach. What exists is the human *temporal*
annotation, so the reference here is the seven production phase anchors placed
on the **human** shot interval and **human** contact frame.

| Method | median frame distance | mean | coverage within ±3 |
|---|---|---|---|
| **F3 phase-aware** | **7.4** | **9.0** | **0.85** |
| F2 motion-top | 8.2 | 13.0 | 0.64 |
| F4 optimized | 10.1 | 11.6 | 0.76 |
| F1 contact-anchored | 10.6 | 12.6 | 0.81 |
| F0 uniform | 14.2 | 16.4 | 0.67 |

**This comparison is biased toward F3 by construction:** the reference is built
from `PHASE_ANCHORS`, and F3 *is* phase-affinity selection against those same
anchors. F3 winning it is close to tautological, so this table is **not** a
valid tiebreaker. **[NOT PROVEN as an independent ranking]**

It does say something useful about F0: last on both distance and coverage, so
uniform sampling is furthest from where a human placed the event.

---

## 8. Latency

| Method | median |
|---|---|
| F2 | 0.011 ms |
| F0 | 0.027 ms |
| F1 | 0.028 ms |
| F3 | 0.096 ms |
| F4 | **0.939 ms** |

F4 is ~35× the cost of F0 and still **under one millisecond**. Selection is
negligible against the ~194 ms/clip signal build and the per-frame pose cost.
Latency does not constrain this choice. **[BENCHMARK]**

---

## 9. Decision

### **Outcome A, qualified — adopt F4 as the Best-7 selector; keep it EXPERIMENTAL, not production.**

The stated purpose of this stage is to produce the best possible `(7,30)`
input sequence. F4 does that decisively: one extra usable frame per sequence,
half the missing landmarks, the best velocity stability, no pose-less frames,
and the fewest total failures — for under a millisecond.

**Why not B (prefer the simpler method).** The gain is not marginal. F0 selects
2.1 pose-less frames per clip and suffers 10 pose failures against F4's 4. A
`(7,30)` tensor built on 4.79 usable frames is materially worse than one built
on 5.78.

**Why not C.** F4's rejection barely rises (5 vs 4), and its single refusal is
explicit rather than a silent degradation.

**Why not D.** F4 is a clear improvement on every biomechanical validity
measure.

### Adopted configuration

```
region  = pad(L3 ∪ [C0 ± 16], 8), fallback L3-only, then whole clip
pool    = ≤40 candidates with a batsman box
select  = F4 (DP optimizer, pose-quality floor, phase + contact + diversity)
refuse  = explicitly, when constraints cannot be satisfied
```

### Carried forward as a known limitation

F4's event faithfulness (0.416 in-shot) is **worse** than F2's (0.571). It
selects usable frames, not necessarily the right moments. Any claim that F4
improves the *biomechanics of the shot* — as opposed to the *validity of the
tensor* — is **not proven** by this benchmark.

---

## 10. Limitations

- 28 valid clips, 24 with correct identity. Differences of one clip are ~4 points.
- Contact anchoring rests on C0, whose accuracy is confounded by non-blind
  annotation (see `PHASE3_CONTACT_RESULTS.md` §3).
- The human reference is annotation-derived and biased toward F3.
- Region containment is 100%, but on 28 clips that is an estimate, not a
  guarantee.
- No expert judgement of frame *informativeness* was available at any point.
- Biomechanical quality is measured by self-consistency (plausibility,
  stability, missingness). There is no keypoint ground truth, so "better
  sequence" means more internally valid, not verified against truth.

---

## 11. Recommendation for Phase 4

1. **The binding constraint is now shot localization, not frame selection.**
   F4 already extracts near the ceiling of what a 62-frame region allows;
   halving the region width would do more for event faithfulness than any
   further selector work. **[INFERENCE]**
2. **Blind re-annotation of a subset is the highest-value cheap experiment.**
   It would resolve C0's circularity and tell us whether contact anchoring is
   real. Everything contact-related currently rests on it.
3. **Do not retrain the scoring model on F4 sequences yet.** The `(7,30)`
   tensors are more valid but not more event-faithful; retraining now would
   bake in the region-width limitation.
4. **If a narrower region becomes available**, re-run this exact ablation
   before adopting — F4's ranking may change when the quality/relevance
   trade-off it is resolving becomes less severe.
