# Phase 3 — L0–L3 Shot Localization

**Question.** Given the selected batsman track, can we identify the temporal
interval of ONE coherent batting shot?

**Answer.** L3 (hybrid global × batsman-local motion) is the best of the four
methods, **but only when batsman identity is correct**, and no method is good
in absolute terms. Best result anywhere is recall@IoU0.5 = 0.45 on 11 clips.

**Scoring model.** Untouched. F0–F4 not started.

Every claim below is tagged **[BENCHMARK]** (measured here), **[INFERENCE]**
(engineering reasoning from measurements), or **[NOT PROVEN]**.

---

## 1. Ground-truth coverage

From the Phase-3 human annotations (`phase3-annotation-1.0`), 51 clips:

| Category | Count |
|---|---|
| **VALID_SINGLE_SHOT** | **28** (dev 14 / eval 14) |
| MULTIPLE_SHOTS | 12 |
| SCENE_CUT | 3 |
| EXCESSIVE_DURATION | 3 |
| AMBIGUOUS | 3 |
| NO_SHOT | 1 |
| INSUFFICIENT_ACTION | 1 |
| **Clips with usable `shot_start` + `shot_end`** | **28** |

All 28 valid clips carry bounds; **no invalid clip carries bounds**, so no
fake single-shot interval is imposed on anything. Shot duration is a median of
22 frames (min 18, max 35) at 30 fps, inside clips of 90 or 180 frames.

Annotation quality bounds what any IoU number can mean: on the 16
double-annotated clips, `shot_start` agreed to within 1 frame in 7 of 10 and
within 3 in 10 of 10; `shot_end` within 1 frame in 10 of 10. **[BENCHMARK]**

---

## 2. What L0 actually is

**The production pipeline has no intra-clip shot localizer.** `select_phase_frames()`
is *handed* `start_frame`/`end_frame`; those originate outside the pipeline in
the ingestion step (`nvidia_client.py` proposing `start_time`/`end_time` over a
long session video). Within the pipeline the supplied window is used as-is.

So on a benchmark of already-trimmed clips, the current method's interval **is
the clip**. That is measured rather than replaced, per the instruction to run
the existing method exactly as it exists. It is not a straw man: the annotated
shot occupies a median 22% of its clip, so whole-clip scores IoU ≈ 0.22 and
never fails catastrophically. **[BENCHMARK]**

---

## 3. Method definitions

| | Signal |
|---|---|
| **L0** | whole clip, unchanged |
| **L1** | global frame-difference energy |
| **L2** | same measure inside the batsman box, normalised by box area |
| **L3** | normalised(L1) × normalised(L2) |

L1/L2/L3 share **one** windowing rule — smooth, take the peak, expand while
the signal exceeds `baseline + α(peak − baseline)`, clamp duration, reject if
peak/baseline prominence < `τ`. Only the *signal* differs, so any difference
is attributable to where motion is measured. The hybrid is a parameter-free
product rather than a weighted sum, deliberately, to avoid fitting a weight.

`α`, `τ` and the duration clamp are grid-searched **on dev only** and applied
unchanged to eval, with the dev objective penalising intervals emitted on dev
invalid clips so the search cannot buy IoU by never rejecting.

Fitted on dev: L1 `α=0.20, τ=2.2, max=30`; L2 `α=0.30, τ=1.8, max=40`;
L3 `α=0.20, τ=1.2, max=30`.

Missing track frames are **interpolated, not zero-filled** — a zero would
assert "no motion here", which an absent box does not support.

---

## 4. Results — all valid clips

### Eval (14 valid, 13 invalid)

| Method | mean IoU | median IoU | @0.3 | @0.5 | @0.7 | start err | end err | rejected valid | false-shot |
|---|---|---|---|---|---|---|---|---|---|
| L0 | 0.212 | **0.225** | 0.07 | 0.00 | 0.00 | 27.0 | 47.5 | 0 | 1.00 |
| L1 | 0.165 | 0.000 | 0.21 | 0.21 | 0.07 | **4.0** | **4.0** | 9 | 0.54 |
| L2 | 0.160 | 0.000 | 0.29 | 0.21 | 0.00 | 3.5 | 18.0 | 8 | **0.46** |
| **L3** | **0.289** | 0.065 | **0.43** | **0.36** | **0.14** | 10.0 | 8.0 | 3 | 0.77 |

### Dev (14 valid)

| Method | mean IoU | median IoU | @0.3 | @0.5 | @0.7 | rejected valid | false-shot |
|---|---|---|---|---|---|---|---|
| L0 | 0.211 | 0.236 | 0.07 | 0.00 | 0.00 | 0 | 1.00 |
| L1 | 0.048 | 0.000 | 0.07 | 0.07 | 0.00 | 8 | 0.20 |
| L2 | 0.140 | 0.000 | 0.21 | 0.14 | 0.14 | 5 | 0.50 |
| L3 | 0.171 | 0.000 | 0.21 | 0.21 | 0.07 | 1 | 0.70 |

**L3 scores better on eval (0.289) than on dev (0.171).** With 14 valid clips
per split, a two-clip swing is ~14 points of recall. This is variance, not a
real effect, and it means **0.289 should not be quoted as L3's expected
performance**. The consistent, direction-level finding is L3 > L2 ≈ L1 and
L3 > L0 on threshold recall. **[BENCHMARK for the ordering; the magnitudes are NOT PROVEN]**

---

## 5. Identity-conditioned results — the measurement that matters

S2 selects the correct batsman on 40 of 51 clips; 24 of the 28 valid clips
have correct identity. Restricting to clips where S2 picked the annotated
batsman removes identity errors from the localization score:

### Eval, identity correct (n = 11)

| Method | mean IoU | median IoU | @0.3 | @0.5 | @0.7 | rejected valid |
|---|---|---|---|---|---|---|
| L0 | 0.220 | 0.225 | 0.09 | 0.00 | 0.00 | 0 |
| L1 | 0.193 | 0.000 | 0.27 | 0.27 | 0.09 | 8 |
| L2 | 0.203 | 0.000 | 0.36 | 0.27 | 0.00 | 5 |
| **L3** | **0.368** | **0.385** | **0.55** | **0.45** | **0.18** | **0** |

**This is the key result.** With identity correct, L3 rejects nothing, lifts
median IoU from L0's 0.225 to 0.385, and reaches recall@0.3 = 0.55.

**L3's advantage is contingent on correct identity.** On all eval clips L3's
*median* IoU (0.065) is **worse** than L0's (0.225), because on
identity-failure clips L3 either rejects or localizes on the wrong person,
while L0's whole-clip answer is uniformly mediocre but never zero. Conditioned
on identity, L3's median more than doubles L0's. **[BENCHMARK]**

This is exactly why identity and localization were scored separately: reporting
only the "all clips" table would have understated L3 by charging it for S2's
errors. **[BENCHMARK]**

---

## 6. Invalid-window rejection

No IoU is computed on invalid clips. They are scored on whether the method
declines them.

| Method | false-shot rate (eval) | multi-shot proxy | scene-cut contamination |
|---|---|---|---|
| L0 | 1.00 | 1.00 | 1.00 |
| L1 | 0.54 | 0.60 | 0.00 |
| L2 | **0.46** | **0.40** | 0.00 |
| L3 | 0.77 | 1.00 | 0.33 |

**This is L3's clear weakness.** Its dev-fitted `τ = 1.2` is permissive, so it
emits an interval on 77% of invalid clips and on every MULTIPLE_SHOTS clip. L2
is the best rejector. L0 never rejects by construction. **[BENCHMARK]**

**Multi-shot caveat:** MULTIPLE_SHOTS clips carry no per-shot boundaries, so
"does the interval contain more than one batting event" is **not computable**.
The number above is the honest proxy — did the method emit anything at all.
A method could emit a correct single-shot interval inside a multi-shot clip
and be counted against here. **[NOT PROVEN that these are true contaminations]**

Scene-cut contamination is measured directly against annotated cut frames, but
only 3 clips carry them, so 0.33 is one clip. **[BENCHMARK, n=3]**

---

## 7. Failure mechanisms

Clips below IoU 0.5, classified by mechanism (identity errors attributed
upstream, never as localization errors):

| Mechanism | L0 | L1 | L2 | **L3** |
|---|---|---|---|---|
| Locked onto a **later** motion peak | 0 | 1 | 3 | **7** |
| Locked onto an **earlier** motion peak | 0 | 4 | 5 | **5** |
| Insufficient movement — rejected | 0 | 15 | 9 | 0 |
| Wrong batsman (upstream) | 4 | 4 | 4 | 4 |
| Interval over-extended | 22 | 0 | 0 | 0 |
| Boundary / partial-overlap error | 2 | 0 | 2 | 4 |
| **Total below IoU 0.5** | **28** | 24 | 23 | **20** |

**L3's dominant failure is motion ambiguity: 12 of 20 failures lock onto the
wrong motion peak** (7 later, 5 earlier). Frame-difference energy cannot tell a
batting stroke from a bowler's run-up, a batsman walking back, or camera
motion. Concrete cases: `kohli_side_08` GT [20,45] → predicted [98,128];
`pro_player_front_16` GT [15,35] → predicted [66,78]. **[BENCHMARK]**

There is a systematic directional bias: median signed start error **+3.0**
frames and end error **−2.5** frames, with 62% of predictions starting late.
L3's intervals sit slightly late and slightly narrow relative to the human
boundaries. **[BENCHMARK]**

L1's dominant failure is the opposite — it rejects 15 clips it should have
localized, because its dev-fitted `τ = 2.2` is strict. **[BENCHMARK]**

Timeline diagrams for representative successes and each failure mode:
`dataset/phase3_results/shot_timeline_L3_hybrid.png` and
`shot_timeline_L0_whole_clip.png`.

---

## 8. Window-length robustness

Each clip cropped to ±30, ±45 frames around the GT shot centre and to the full
clip, then the prediction mapped back to absolute frames. Self-IoU between the
full-clip prediction and the cropped ones:

| Method | median self-IoU | mean | reject flips |
|---|---|---|---|
| L1 | **1.000** | 0.950 | 10 |
| L2 | **1.000** | 0.711 | 15 |
| L3 | 0.756 | 0.622 | **0** |

A real trade-off, in both directions: **L1/L2 keep the same interval when they
emit one but flip between rejecting and emitting 10–15 times** as context
changes; **L3 never flips its accept/reject decision but moves its boundaries
more.** Neither is robust in both senses. **[BENCHMARK]**

The crop is centred using ground truth, so this measures *stability under added
irrelevant context*, not accuracy — a method can be stably wrong. **[BENCHMARK, with that caveat]**

---

## 9. Latency

| Stage | Cost |
|---|---|
| Signal build (one video decode, shared by L1/L2/L3) | **194.2 ms/clip** median |
| L0 windowing | 0.0005 ms/clip |
| L1 windowing | 0.028 ms/clip (0.23 µs/frame) |
| L2 windowing | 0.035 ms/clip (0.22 µs/frame) |
| L3 windowing | 0.046 ms/clip (0.28 µs/frame) |
| **Total per clip** | L0 ≈ 0 ms; **L1/L2/L3 ≈ 194 ms** |

Decode dominates completely; the windowing arithmetic is ~0.02% of cost. **L2
costs no extra decode over L1** — both energies are computed in the same pass —
so choosing L3 over L1 is essentially free. Cost is not a differentiator among
L1/L2/L3. **[BENCHMARK]**

---

## 10. Selected configuration

### **L3 hybrid, conditional on S2 committing to a batsman; L0 whole-clip as fallback.**

```
if S2 refuses to identify a batsman:      use L0 (whole clip)
elif L3 rejects (no prominent peak):      use L0 (whole clip)
else:                                     use L3's interval
```

Frozen parameters (fit on dev): `α = 0.20`, `τ = 1.2`, `min_dur = 12`,
`max_dur = 30`, smoothing width 5.

**Why L3 wins.** It is the only method that beats the whole-clip baseline on
both mean and median IoU under correct identity (0.368/0.385 vs 0.220/0.225),
it has the best threshold recall at every threshold, it rejects the fewest
valid clips, and it never flips its accept/reject decision under window
changes. The product form also has a principled reason to work: it requires an
interval to be active *both* globally and on the batsman, which suppresses the
bowler-only and camera-only peaks that defeat L1, and the batsman-fidget peaks
that defeat L2. **[BENCHMARK for the numbers; INFERENCE for the mechanism]**

**Why the L0 fallback.** L3's advantage is contingent on identity, and its
median IoU on all eval clips (0.065) is *worse* than L0's (0.225). Falling back
to the whole clip when identity or prominence is unavailable keeps the
catastrophic cases at L0's uniform mediocrity instead of near-zero. **[INFERENCE]**

**What is NOT claimed.** That the earlier hypothesis "batsman-local motion beats
global motion" holds — L2 alone does **not** beat L1 consistently (dev 0.140 vs
0.048, eval 0.160 vs 0.165). Only their *product* helps. **[BENCHMARK]**

---

## 11. Limitations

- **No method is good.** Best recall@IoU0.5 anywhere is 0.45 on 11 clips.
  Roughly half of all valid clips are still localized poorly.
- **14 valid clips per split.** L3 scoring higher on eval than dev is a variance
  signature. Direction is trustworthy; magnitudes are not.
- **Identity-conditioned n = 11 on eval.** One clip is ~9 points.
- **Rejection is L3's weakest property** (false-shot 0.77). It is not safe to
  use L3's accept/reject as a coherence classifier.
- Multi-shot contamination is a proxy, not a measurement (§6).
- Scene-cut contamination rests on 3 clips.
- Robustness probe uses GT to place the crop; it measures stability only.
- Contact detection was deliberately kept **out** of the shot-boundary
  benchmark, per instruction, so no result here depends on it.

---

## 12. Recommendation for the contact + Best-7 stage

1. **Freeze L3 + L0 fallback** as the shot-localization configuration and pass
   the interval to the next stage as a **soft prior, not a hard crop.** With
   recall@0.5 ≈ 0.36–0.45, hard-cropping to L3's window would discard the true
   shot on roughly half of clips. **[INFERENCE from §4–5]**
2. **Use contact as an internal anchor to correct the late bias.** Contact was
   held out of this benchmark deliberately, and the systematic +3.0 frame late
   start / −2.5 frame early end (§7) is exactly the kind of error an
   independent anchor could fix. The 28 clips with usable contact labels are
   available to test this. **[INFERENCE]**
3. **Do not use L3's rejection as a coherence gate** (false-shot 0.77). If
   Best-7 needs a validity signal, L2's rejector (0.46) or the S2 identity
   verdict are better starting points. **[BENCHMARK]**
4. **The largest remaining win is not a better motion signal.** 12 of L3's 20
   failures are wrong-peak selection, which a *scalar* motion measure cannot
   resolve — the bowler's run-up genuinely produces more frame-difference
   energy than a compact stroke. Progress needs a signal that knows a batting
   stroke's *structure* (the S2 skeleton features already do), not a better
   threshold. **[INFERENCE]**
