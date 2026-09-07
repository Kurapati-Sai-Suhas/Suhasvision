# Phase 3D — Contact Localization (C0–C6)

**Question.** Can contact be localized reliably, and can it correct L3's shot
interval?

**Answer.** C0 (the existing proposal) localizes contact to a median of 1 frame
and every other signal is 17–33 frames off. Contact-anchoring appears to lift
shot IoU to 0.86 — but that result is **confounded twice over** and must not be
read as a solved shot localizer.

Tags: **[BENCHMARK]** measured · **[INFERENCE]** reasoned · **[CONFOUNDED]** measured but not trustworthy.

---

## 1. Ground-truth coverage

| Contact type | n | Carries a frame? | dev | eval |
|---|---|---|---|---|
| **EXACT** (ball visible at bat) | 4 | yes | 3 | 1 |
| **CONTACT_ADJACENT** | 24 | yes | 11 | 13 |
| AMBIGUOUS | 16 | **no** | – | – |
| NOT_VISIBLE | 7 | **no** | – | – |

**28 usable** (dev 14 / eval 14). AMBIGUOUS and NOT_VISIBLE carry no frame at
all, so they cannot be — and are not — treated as frame labels. Confidence on
the usable set: MEDIUM 22, HIGH 4, LOW 2. **[BENCHMARK]**

The schema's own tolerances differ by type: **±1 frame for EXACT, ±3 for
CONTACT_ADJACENT**. A "within ±1 frame" figure computed over CONTACT_ADJACENT
clips is therefore partly measuring annotation noise rather than model error.

---

## 2. Contact results (28 clips)

| Signal | median err | mean err | ±1 | ±2 | ±3 | not-found |
|---|---|---|---|---|---|---|
| **C0 current proposal** | **1.0** | **1.7** | **0.54** | **0.75** | **0.86** | 0.00 |
| C1 wrist velocity | 22.5 | 30.7 | 0.00 | 0.04 | 0.11 | 0.14 |
| C2 bat association | 33.5 | 36.7 | 0.00 | 0.00 | 0.04 | 0.00 |
| C3 skeleton dynamics | 31.0 | 32.8 | 0.04 | 0.04 | 0.07 | 0.14 |
| C4 global motion | 30.5 | 30.8 | 0.07 | 0.14 | 0.18 | 0.00 |
| C5 batsman-local motion | 26.0 | 28.3 | 0.14 | 0.18 | 0.21 | 0.14 |
| C6 combined event | 16.0 | 20.1 | 0.04 | 0.11 | 0.14 | 0.00 |

**The gap is enormous — C0 is ~16× more accurate than anything else.** No
single-signal peak-picker is remotely competitive: wrist velocity, bat
evidence, skeleton dynamics and motion energy all peak tens of frames away
from the labelled contact. C6, the equal-weight combination, is the best of the
alternatives at 16 frames and still unusable. **[BENCHMARK]**

> **A smoothing bug found and fixed while building this.** The moving average
> originally used `np.convolve(..., mode="same")`, which zero-pads and so makes
> every smoothed signal decay toward both ends of a clip. For a
> derivative-based signal like C3 that manufactures a spurious descent in the
> final frames of *every* clip — a systematic bias toward late contact
> estimates. Fixed by replicating edge values. The corrected numbers are the
> ones above (C3 29.5→31.0, C4 23.5→30.5, C6 17.0→16.0); C0 is unaffected
> because it does not use this smoothing, and no conclusion changes.
> **The same zero-padding pattern exists in `phase3_shot_localization._smooth`
> and was left alone rather than silently altering already-reported L0–L3
> results — it is flagged for Phase 4.**

C6 uses an equal-weight mean of normalised signals, deliberately: a fitted
weighted sum would add five free parameters on 14 dev clips. Its failure is
therefore a statement about the *signals*, not about a fit.

### Broken down by identity and type

| Subset | n | median err | ±1 | ±3 |
|---|---|---|---|---|
| Identity correct | 24 | **1.0** | 0.58 | 0.92 |
| Identity wrong | 4 | 3.0 | 0.25 | 0.50 |
| EXACT | 4 | 1.5 | 0.50 | 0.75 |
| CONTACT_ADJACENT | 24 | 1.0 | 0.54 | 0.88 |
| dev | 14 | 1.0 | 0.64 | 0.86 |
| eval | 14 | 2.0 | 0.43 | 0.86 |

Contact is clearly better when the right person was selected (1.0 vs 3.0
frames), so identity errors are reported separately rather than pooled into the
contact score. **[BENCHMARK]**

---

## 3. Two confounds that change the interpretation

### 3a. The annotator was not blind to C0

`phase3_annotations.py` states it plainly: *"single annotator, not blind to the
machine proposal (it is drawn on the sheet)"*. C0's proposal was rendered on
the contact sheet the human labelled from, so the human label may be anchored
toward C0. **C0 vs human contact is therefore not a fully independent
measurement.** **[CONFOUNDED]**

Partial check: on the 4 EXACT clips — where the ball is visibly at the bat and
the human had direct evidence — C0's median error is 1.5, slightly *worse* than
on CONTACT_ADJACENT (1.0). If anchoring dominated, the reverse would be
expected. That is mild evidence against heavy anchoring, but **n = 4 cannot
settle it**. Resolving this needs blind re-annotation. **[NOT PROVEN]**

### 3b. The shot bounds are a fixed offset from contact

Measured over the 28 valid clips:

```
contact − shot_start = 10.7 ± 2.4 frames
shot_end − contact   = 12.0 ± 2.3 frames
```

The annotation convention places contact almost exactly in the middle of the
shot, with very low variance. Consequence:

> **Oracle**: a fixed ±10 window around the **true** contact frame scores
> mean IoU **0.815**, median **0.822**, recall@0.5 = **1.00**, @0.7 = 0.96.

So *any* accurate contact detector reconstructs the annotated shot interval
almost perfectly, without localizing anything. **[BENCHMARK]**

---

## 4. Contact as an L3 corrector

Three corrector forms (`recenter`, `expand`, `repair`) were grid-searched on
**dev only** and applied unchanged to eval.

| Config | dev mIoU | eval mIoU | eval median | eval @0.3 | eval @0.5 |
|---|---|---|---|---|---|
| L0 whole clip | 0.211 | 0.212 | 0.225 | 0.07 | 0.00 |
| L3 alone | 0.140 | 0.235 | 0.000 | 0.36 | 0.29 |
| **L3 + C0 contact** (recenter, ±10) | **0.816** | **0.864** | **0.885** | **1.00** | **1.00** |

Taken alone this looks like a solved problem. It is not. Substituting a contact
anchor the annotator never saw:

| Anchor signal | dev mIoU | eval mIoU | eval @0.5 |
|---|---|---|---|
| C0 (annotator saw it) | 0.816 | **0.864** | 1.00 |
| C6 combined | 0.367 | **0.155** | 0.21 |
| C5 local motion | 0.219 | 0.268 | 0.29 |
| C4 global motion | 0.157 | 0.268 | 0.21 |

(measured before the smoothing fix; C4/C5/C6 anchors are all far worse than C0
either way, and the conclusion does not depend on the exact values)

**Every independent anchor performs at or below plain L3 (0.235).** The entire
gain is attributable to C0, and C0 is the one signal entangled with the
annotation process. Combined with the ±10 oracle above, the honest reading is:

> The corrector is reconstructing an annotation convention from a contact
> estimate, not discovering shot boundaries. Its 0.86 IoU is real **with
> respect to this annotation** and is **not** evidence of a general shot
> localizer. **[CONFOUNDED]**

---

## 5. Contact is not the shot boundary

Kept structurally distinct throughout, per the spec: the shot interval is the
complete coherent event; contact is an impact-adjacent anchor *inside* it. The
Best-7 stage consumes contact as an internal anchor and never as a crop. §3b is
precisely why conflating them would have been self-fulfilling here. **[INFERENCE]**

Contact is also not a hard dependency: when it is AMBIGUOUS, NOT_VISIBLE or
absent, the event region falls back to the L3 prior plus padding, and the
pipeline continues on motion, phase and pose evidence.

---

## 6. What can and cannot be concluded

**Supported:** C0 localizes contact to ~1–2 frames on this corpus; every
alternative signal tested is 17–33 frames off; contact quality degrades when
identity is wrong.

**Confounded:** C0's accuracy against a non-blind annotator; the 0.86 shot IoU
from contact anchoring.

**Not proven:** that contact anchoring generalizes to footage where the
annotation convention does not hold; that C0 would survive blind
re-annotation.

**Recommendation.** Use C0 as the event anchor for Best-7 — it is the best
available and cheap — but **do not report contact-anchored shot IoU as a
localization result**, and treat blind re-annotation of a subset as the
highest-value cheap experiment available for Phase 4.
