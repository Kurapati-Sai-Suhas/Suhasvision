# Phase 3 Annotation Protocol

Reference document for producing the Phase-3 temporal ground truth. Every definition
here is **operational**: it can be applied to a frame strip without knowing the
pipeline's opinion, and two annotators following it should reach the same answer on a
clear clip. Where a distinction cannot be made reliably, this document says so and the
schema does not offer the distinction.

Schema: `dataset/phase3_annotation_schema.py` (`phase3-annotation-1.0`)
Sheets: `dataset/phase3_results/event_sheets/<clip>.jpg`

---

## 0. The governing rule

> **Annotate what the video shows, not what the pipeline proposed.**

The sheet displays a machine-proposed contact frame. It is a navigation aid. It is
stored separately (`proposed_contact_frame`) from the human label (`contact_frame`) so
that proposal quality can be measured afterwards. If the proposal looks wrong, it *is*
wrong — override it. A benchmark that inherits its own proposals measures nothing.

---

## 1. Shot coherence

Decide this **first**. It determines which other fields are even permitted.

| Label | Definition |
|---|---|
| `VALID_SINGLE_SHOT` | The window contains exactly one batting stroke by one batsman, from a settled pre-delivery position through to a completed follow-through, with no scene cut. |
| `NO_SHOT` | No batting stroke occurs. The batsman may be standing, walking, resetting, or absent. Shadow-batting with no delivery counts as `NO_SHOT`. |
| `MULTIPLE_SHOTS` | Two or more distinct strokes occur — the bat descends through an impact zone more than once. |
| `SCENE_CUT` | The visual scene changes discontinuously inside the window (different camera position, location, or lighting) such that consecutive frames are not one continuous take. |
| `EXCESSIVE_DURATION` | The window is a whole practice session rather than a delivery. Operationally: **> 15 seconds**, or containing more than one bowler run-up. |
| `INSUFFICIENT_ACTION` | A stroke may be occurring but too little of it is visible to bound — e.g. the batsman is occluded, out of frame, or the clip starts mid-downswing and ends before follow-through. |
| `AMBIGUOUS` | None of the above applies cleanly. Use sparingly and always add a note. |

**Precedence when several apply:** `SCENE_CUT` > `MULTIPLE_SHOTS` > `EXCESSIVE_DURATION`
> `NO_SHOT` > `INSUFFICIENT_ACTION` > `AMBIGUOUS`. Rationale: a cut invalidates the
window regardless of what else is in it, so it is recorded as the primary reason.

Only `VALID_SINGLE_SHOT` may carry `shot_start_frame` / `shot_end_frame`. Every other
verdict must leave them null — a window that is not one shot has no defensible boundaries,
and inventing them would put fabricated targets into the localization metrics.

---

## 2. Shot start

> **`shot_start_frame` = the last frame in which the batsman is in a settled position
> before any preparatory movement toward the stroke begins.**

"Settled" means the feet are planted and the bat is at rest (grounded or held still).
The first sign of preparatory movement is whichever comes first: the bat begins to
rise, the front foot begins to move, or the body begins to rotate away from the address
position.

**Edge cases, resolved:**
- If the clip begins with movement already underway, `shot_start_frame` = the **first
  frame in which the batsman is visible**. Note it as `start_clipped`.
- If the batsman fidgets before settling, take the **last** settle before the stroke,
  not the first.
- A trigger movement is *part of the stroke*, so it falls **after** `shot_start`.

## 3. Shot end

> **`shot_end_frame` = the frame at which the follow-through reaches its furthest
> extent and the batsman begins to return toward a neutral position.**

Operationally: the last frame in which the bat is still travelling along the stroke
path. The frame after which the bat decelerates and the body starts to reset.

**Edge cases, resolved:**
- If the batsman walks out of frame during follow-through, use the **last frame in
  which the follow-through is visible**. Note as `end_clipped`.
- If the clip ends mid-follow-through, use the **last frame of the clip**. Note as
  `end_clipped`.
- Do **not** extend `shot_end` to include the batsman resetting into stance. That reset
  is a separate event, and including it is exactly the error that put Phase 1's
  "follow-through" frame 1.9 seconds after contact.

---

## 4. Contact

Four categories, in decreasing order of evidential strength.

| Type | Definition | Carries a frame? | Scoring tolerance |
|---|---|---|---|
| `EXACT` | The ball is **visible** at or immediately adjacent to the bat face. The impact instant is directly observable. | Yes | ±1 frame |
| `CONTACT_ADJACENT` | The ball is not resolvable, but the bat unambiguously passes through an impact zone. Label the frame where the bat is at, or nearest to, the expected ball line. | Yes | ±3 frames |
| `AMBIGUOUS` | A stroke occurs but the impact instant cannot be localized to within roughly 3 frames — heavy blur, occlusion, or an off-camera ball line. | **No** | — |
| `NOT_VISIBLE` | No stroke occurs, or the batsman is not visible during the relevant period. | **No** | — |

**Why two tolerances.** `EXACT` is held to ±1 because the ball is visible and the
instant is genuinely resolvable at this frame rate. `CONTACT_ADJACENT` is held to ±3
because the label itself localises *the bat through a zone*, not a visible impact —
scoring it at ±1 would measure the annotator's precision rather than the detector's.

**`AMBIGUOUS` and `NOT_VISIBLE` must not carry a frame.** Forcing a number for a contact
you cannot see injects noise that looks like data. The schema rejects it.

---

## 5. Phases — five, not seven

The pipeline uses seven phases. **The annotation uses five.** This is deliberate.

At 30 fps on this footage, `stance` and `trigger` are frequently the same frame, and the
boundary between `backlift_start` and `full_backlift` is a judgement that would not
survive a second annotator. Annotating seven phases would manufacture precision the
video cannot support and would make every phase metric unfalsifiable.

| Phase | Definition (boundaries are inclusive) |
|---|---|
| `STANCE_PREPARATION` | From `shot_start` until the bat clearly begins to rise. Includes any trigger movement. |
| `BACKLIFT` | From the bat clearly rising until it reaches its highest point and momentarily stops or reverses. |
| `DOWNSWING` | From the bat beginning to descend until the frame before contact. |
| `CONTACT_ADJACENT` | The contact frame ±1. Present only when `contact_type` is `EXACT` or `CONTACT_ADJACENT`. |
| `FOLLOW_THROUGH` | From the frame after contact until `shot_end`. |

Phases are annotated as `[start_frame, end_frame]` spans and must lie inside
`[shot_start, shot_end]`. Phases may be **omitted** where not identifiable — a partial
phase set is honest; a complete guessed one is not.

**Consequence for the pipeline:** a seven-slot selection is scored against five-phase
ground truth via `PIPELINE_PHASE_TO_ANNOTATED`. Phase-coverage metrics therefore measure
coverage of five annotatable phases, and the report must say so rather than implying
seven-phase validation.

---

## 6. Scene cuts

> **A scene cut is a frame index `f` such that frames `f-1` and `f` are not from the
> same continuous camera take.**

Signals: abrupt change of background, camera position, framing, lighting, or player
set. A fast pan is **not** a cut. Record every cut frame in `scene_cut_frames`.

The sheet's `cuts detected: N` header is a histogram-based *hint* at correlation < 0.55.
Verify each by eye; the detector both misses gradual cuts and fires on fast pans, and
its false-positive/negative rate against these labels is itself a Phase-3 measurement.

---

## 7. Multiple shots

> **More than one stroke = the bat descends through an impact zone more than once,
> with a return to a preparatory position in between.**

A checked swing followed by a real stroke is **one** shot. Two deliveries, each played,
is `MULTIPLE_SHOTS`. When in doubt, look for the batsman resetting to stance between
them.

---

## 8. Confidence

Record `shot_confidence` and `contact_confidence` independently:

- `HIGH` — the definition applies unambiguously; a second annotator would agree.
- `MEDIUM` — the definition applies but the exact frame is arguable within a few frames.
- `LOW` — the label is the best available reading but genuinely uncertain.

`LOW`-confidence records are retained but reported separately, so a sensitivity analysis
can show whether any conclusion depends on them.

---

## 9. Handling ambiguity

1. Re-read the relevant definition above.
2. If the definition decides it, apply it even if the result feels unintuitive —
   consistency beats per-clip judgement.
3. If the definition does not decide it, use the `AMBIGUOUS` category and write a note
   naming what was unclear.
4. If the same ambiguity recurs across clips, that is a **definition defect**, not an
   annotation defect. Amend this document and re-annotate the affected clips.

---

## 10. Worked examples from this corpus

| Clip | Reading | Labels |
|---|---|---|
| `instagram2_front_05` | Portrait, single batsman, full stroke visible. Bat descends through the line around f49–51. | `VALID_SINGLE_SHOT`; contact `CONTACT_ADJACENT` |
| `sanjay_front _1` | 200 s of a full net session, many deliveries, camera moves. | `EXCESSIVE_DURATION`; no boundaries; contact `NOT_VISIBLE` |
| `pro_player_front_20` | Front-view nets for ~0.6 s, then an entirely different net and player. | `SCENE_CUT`; cut frame recorded; no boundaries |
| `pro_player_front_42` | A player walks with a bat; no delivery is played. | `NO_SHOT`; contact `NOT_VISIBLE` |
| `kohli_front_01` | Batsman holds the same address position for ~6 s; no stroke. | `NO_SHOT` |

---

## 11. Quality control

A subset is annotated **twice, independently**, and agreement is reported as:
`shot_start` / `shot_end` frame deltas, `contact_frame` delta, and exact-match rates for
`coherence` and `contact_type`.

Every disagreement is classified as one of:

- **annotation error** — the definition decides it and one pass applied it wrongly;
- **genuine visual ambiguity** — the video does not support a unique answer;
- **definition ambiguity** — the protocol does not decide the case.

Only the third requires amending this document. The classification is reported, not just
the agreement rate — an agreement number without the reasons behind its misses cannot
tell you whether the protocol is sound.
