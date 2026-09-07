# Phase 3E — Blind Event Validation, L3 Correction, Best-7 Reassessment

Tags: **[BENCHMARK]** measured · **[CONFOUNDED]** measured but entangled ·
**[INFERENCE]** reasoned · **[NOT PROVEN]** · **[BLOCKED]** needs a human.

---

## 0. What was and was not done

**Blind annotation could not be performed.** §1–§4 require a human annotator
who has never seen C0. I am the system under validation; labels I generate
would be neither blind nor human, and manufacturing them would destroy the only
clean evidence available for this question. **[BLOCKED]**

What was delivered instead:

| | Status |
|---|---|
| Blind annotation pack (subset, marker-free sheets, answer template) | **ready — needs you** |
| Partial leakage test from the **existing** re-annotation | **done [CONFOUNDED]** |
| L3 smoothing fix + full L0–L3 re-run | **done [BENCHMARK]** |
| R0–R4 event-region comparison | **done [BENCHMARK]** |
| F0–F4 re-evaluated on the corrected region | **done [BENCHMARK]** |
| Three objectives separated | **done** |

---

## 1. Partial leakage test (existing double annotation)

`DOUBLE_ANNOTATION` holds 16 clips re-annotated "from the sheets alone without
consulting" pass 1. Both passes saw the sheets — which carry C0 — so this is
**not** a blind study. But it supports one falsifiable inference:

> If C0 were *driving* the labels, both passes would be pulled toward it, and
> each pass would sit closer to C0 than the two passes sit to each other.

Contact frame, 10 clips where all three values exist:

| Comparison | n | median | mean | max | ±1 | ±3 |
|---|---|---|---|---|---|---|
| **human vs human** | 10 | 1.0 | **0.60** | **1** | **1.00** | 1.00 |
| C0 vs pass 1 | 10 | 1.0 | 1.10 | 4 | 0.70 | 0.90 |
| C0 vs pass 2 | 10 | 1.0 | 1.50 | 4 | 0.60 | 0.90 |

**The two human passes agree with each other roughly twice as closely as
either agrees with C0**, and they hit ±1 frame on 100% of clips against C0's
60–70%. They share information C0 does not carry.

> **Verdict — §6 Case C.** C0 is a useful but limited signal: measurably worse
> than a human annotator, and *not* explained away as an echo of itself. This
> argues **against** strong leakage. It does **not** prove independence,
> because both passes saw the marker. **[CONFOUNDED — directional only]**

### Does the contact-offset convention survive? (§5)

| | contact − start | end − contact | n |
|---|---|---|---|
| pass 1 | 10.6 ± 2.84 | 12.8 ± 2.04 | 10 |
| pass 2 | 9.9 ± 2.47 | 12.0 ± 2.10 | 10 |

The convention reproduces closely across passes. **[BENCHMARK]**

But both passes come from the same annotator working to the same protocol, so
this cannot distinguish "contact genuinely sits mid-event" from "the protocol
constructs the window around contact". §5's questions **C and D remain open**
and are exactly what the blind pack is designed to answer. **[NOT PROVEN]**

---

## 2. L3 smoothing correction (§7)

`phase3_shot_localization._smooth` used `np.convolve(..., mode="same")`, which
zero-pads and so decays every smoothed signal toward both clip ends,
suppressing motion energy where a shot may legitimately sit. Fixed with edge
replication. **Original results preserved verbatim** in
`phase3_shot_localization_original.json`; the corrected run is reported
alongside, never in place of it.

| | dev mIoU o→c | eval mIoU o→c | eval median o→c | eval @0.3 o→c | @0.5 o→c |
|---|---|---|---|---|---|
| L0 | 0.211 → 0.211 | 0.212 → 0.212 | 0.225 → 0.225 | 0.07 → 0.07 | 0.00 → 0.00 |
| L1 | 0.048 → 0.048 | 0.165 → 0.165 | 0.000 → 0.000 | 0.21 → 0.21 | 0.21 → 0.21 |
| L2 | 0.140 → 0.140 | 0.160 → 0.160 | 0.000 → 0.000 | 0.29 → 0.29 | 0.21 → 0.21 |
| **L3** | 0.171 → 0.171 | **0.289 → 0.244** | **0.065 → 0.000** | **0.43 → 0.36** | **0.36 → 0.29** |

Identity-conditioned eval (n = 11): L3 **0.368 → 0.311** mean,
**0.385 → 0.129** median, @0.3 **0.55 → 0.45**.

**L0, L1 and L2 are bit-identical** — L0 does no smoothing, and a single
signal's peak *location* is unaffected by symmetric edge decay. Only L3
changes, because it multiplies two independently normalised signals and the
edge decay distorts each differently. Dev is unchanged for every method, and no
fitted parameter changed. **[BENCHMARK]**

### Does the conclusion change?

**The selection does not: L3 is still the best method** on eval (0.244 vs L0
0.212, L1 0.165, L2 0.160) and identity-conditioned (0.311 vs L0 0.220), with
much better threshold recall.

**But one Phase-3C claim does not survive and is withdrawn.** That report said
L3 "lifts median IoU from L0's 0.225 to 0.385" under correct identity.
Corrected, L3's median is **0.129 — below L0's 0.225**. L3 now wins on *mean*
and on *threshold recall* while **losing on median**: it gets some clips well
and others badly wrong, where L0 is uniformly mediocre. The zero-padding bug
was incidentally flattering L3. **[BENCHMARK]**

---

## 3. Event-region reassessment (§8)

Containment and width are reported together and never collapsed into one score.

| Region | GT containment | mean width | width ÷ GT | mean IoU |
|---|---|---|---|---|
| R0 L3 only | **0.18** | 31.6 | 1.39 | 0.216 |
| R1 L3 + padding | 0.43 | 50.6 | 2.23 | 0.206 |
| R2 L3 ∪ contact | 0.93 | 48.2 | 2.12 | 0.521 |
| **R3 contact-centred (±18)** | **1.00** | **35.6** | **1.57** | **0.638** |
| R4 hybrid (Phase 3D's) | 1.00 | 62.6 | 2.76 | 0.387 |

**R3 dominates R4 on both axes** — same 100% containment, 43% narrower, and
IoU 0.638 vs 0.387. Phase 3D's region was not optimal and has been replaced.

**R0 is the headline warning: L3 alone contains the whole annotated shot only
18% of the time.** That is decisive confirmation that L3 must stay a soft prior
and must never be used as a crop. **[BENCHMARK]**

R2–R4 use C0 and are scored against C0-exposed annotations, so their
containment is optimistic; §1 argues the effect is small but not zero.
**[CONFOUNDED]**

---

## 4. F0–F4 on the corrected region (§9)

Same optimizer, same pool construction, same corrected L3 — only the region
changed (R4 → R3).

| Method | **in-shot fraction** | dist. to shot | **pose quality** | min pose quality | min gap | contact-adj |
|---|---|---|---|---|---|---|
| F0 uniform | 0.571 | 0.5 | 0.550 | 0.234 | 5.8 | 1.0 |
| F1 contact-anchored | 0.559 | 0.5 | 0.518 | 0.209 | 2.9 | 1.1 |
| **F2 motion-top** | **0.792** | 0.2 | 0.481 | 0.223 | **1.0** | 2.5 |
| F3 phase-aware | 0.652 | 0.0 | 0.505 | 0.193 | 1.6 | 2.0 |
| **F4 optimized** | 0.621 | 0.1 | **0.804** | **0.680** | 2.5 | 1.7 |

### The Phase-3D prediction was correct

Phase 3D recorded F4's in-shot fraction as 0.416 and attributed it to region
width rather than to the optimizer, predicting that narrowing the region would
fix it without touching F4.

> **F4's in-shot fraction rose 0.416 → 0.621** from the region change alone.
> The optimizer was not modified. **[BENCHMARK]**

F4 still trails F2 (0.792) and F3 (0.652) on raw event faithfulness, but the
gap has closed substantially.

### Why F2's 0.792 is not the win it looks like

F2's **min gap is 1.0 frame** — it clusters all seven selections onto adjacent
frames at the motion peak. A high in-shot fraction obtained by collapsing the
sequence defeats the purpose of a seven-phase temporal tensor, and F2 also has
the worst pose quality (0.481). **In-shot fraction alone is not a sufficient
criterion.** **[INFERENCE from the min-gap and pose-quality columns]**

F4's **minimum** pose quality (0.680) is roughly **3× every alternative**
(0.193–0.234). That is the quality floor doing exactly its job.

---

## 5. Three objectives, kept separate (§10)

| Objective | Status | Evidence |
|---|---|---|
| **1. Event faithfulness** — are the frames from the batting event? | **partial** | F4 0.621 in-shot after the region fix, up from 0.416; still below F2/F3. Scored against C0-exposed annotations. **[CONFOUNDED]** |
| **2. Biomechanical validity** — are they good pose observations? | **strong** | F4 wins pose quality 0.804 vs 0.481–0.550 and min pose quality 0.680 vs 0.193–0.234; Phase 3D also showed 5.78/7 usable angle frames vs 4.46–4.88 **[BENCHMARK]** |
| **3. Downstream usefulness** — do the tensors score better? | **none** | Requires scoring-model retraining, which is out of scope. **No claim is made.** **[NOT PROVEN]** |

---

## 6. F4 reassessment (§12)

### **Outcome C, executed and resolved.**

§12 Outcome C says: if F4 selects too many outside-event frames, *narrow the
region — do not weaken the optimizer*. That is exactly what happened, and it
worked: 0.416 → 0.621 with the optimizer untouched.

**F4 remains the selector.** It is the only method that is simultaneously
competitive on event faithfulness (0.621, second-best behind a method that
achieves its score by clustering) and dominant on biomechanical validity
(3× the minimum pose quality of any alternative).

**Outcome D does not apply:** the leakage evidence argues against C0 being an
artifact, so C0 is retained — but as **experimental** evidence pending the
blind study, not as a validated signal.

---

## 7. The blind annotation pack (ready for you)

- **Subset:** 12 clips, `phase3e_blind_subset.json`, chosen by **adversarial
  stratification** — deliberately including the two strongest C0 predictions
  (error 0) *and* the two weakest (errors 5 and 4), 3 identity failures, front /
  side / back views, the most-occluded clip, the highest-motion clip, and
  background-heavy/invalid clips. It is not a random sample that would flatter C0.
- **Sheets:** `phase3_results/blind_sheets/` — every frame with its index and a
  batsman crop with generous horizontal padding (bat and ball sit outside the
  person box). **No C0 marker, no L3 interval, no F4 frames, no region, no
  boxes, no skeletons.** Verified by inspection.
- **Answer file:** `phase3e_blind_annotations.TEMPLATE.jsonl`.

Nothing in this repository will generate the answers automatically, by design.

Once returned, `phase3e_leakage_analysis.py` extends directly to the blind
labels and resolves §6 Cases A/B/C, plus §5 questions C and D.

---

## 8. Limitations

- **The central question is still open.** Every contact-based number remains
  scored against C0-exposed labels. The partial test is directional only.
- The double-annotation subset gives **10 usable contact clips**. Differences
  below ~0.5 frames are not meaningful.
- Both annotation passes share one annotator and one protocol; they cannot
  separate a physical regularity from a protocol artifact.
- R3's 100% containment is measured on 28 clips against possibly-confounded
  bounds.
- Objective 3 is entirely unmeasured.
- L3 remains weak in absolute terms: corrected identity-conditioned median IoU
  0.129, and containment of the full shot only 18%.

---

## 9. Recommendation

1. **Run the blind pack.** It is the only thing standing between "C0 is
   probably fine" and "C0 is validated", and it gates every contact-dependent
   decision downstream.
2. **Adopt R3 in place of R4 now** — it is strictly better on both containment
   and width, and its benefit does not depend on the leakage question.
3. **Keep F4, keep it experimental.** Objective 2 is well-evidenced; Objective
   1 is improved but still second-best; Objective 3 is unmeasured.
4. **Do not retrain the scoring model yet**, and do not add a VLM or a
   Transformer — the event ground truth is not yet trustworthy enough to freeze
   the extraction pipeline.
