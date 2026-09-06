# EXTRACTION_PIPELINE_RESEARCH.md
## Making the Cricket Shot / Frame Extraction Pipeline Reliable — A Research Report

**Date:** 2026-08 · **Scope:** upstream video → shot → person → phase → pose → features. The scoring model is explicitly *out of scope* until the input is trustworthy.

**Evidence labelling used throughout:**
- **[DIRECT]** — the cited paper evaluates the same or a very similar problem.
- **[RELATED]** — the paper solves a closely related sports/CV problem.
- **[INFERENCE]** — my engineering judgement; the paper does not prove this for cricket.

---

# 1. EXECUTIVE CONCLUSION

Three findings, in order of importance. All are verified against either your source code or published literature.

### Finding 1 — You are using a vision-language model for the one task the literature says VLMs are worst at

Your shot localization asks a VLM "which frames contain batting?" The 2025 literature on video-LLM temporal grounding is blunt about this: models "have excelled at understanding *what* happens in a video, yet they largely fail when asked *when*," and are characterised as **"largely temporally blind," with limited sensitivity to event ordering and duration** ([TimeLens](https://arxiv.org/pdf/2512.14698), [Grounded-VideoLLM](https://openreview.net/forum?id=YCwN7wQA6W), [natural-language temporal grounding benchmark](https://arxiv.org/pdf/2606.12300)). **[DIRECT]**

Your architecture partially compensates — you ask for *frame indices*, not timestamps, and do the arithmetic locally. That is genuinely the right mitigation and better than most applied uses. But it does not fix the underlying weakness: you are asking a temporally-blind model to make your most temporally-precise decision, then sampling everything downstream inside whatever window it returns.

### Finding 2 — MediaPipe is architecturally the wrong tool for multi-person cricket footage

MediaPipe's documented limitation is **"its inability to reliably track multiple people within a single frame, as the system is primarily designed for single-person pose estimation"** ([MediaPipe for sports apps](https://www.it-jim.com/blog/mediapipe-for-sports-apps/)). **[DIRECT]**

You already discovered this empirically — the feeder-tracking incident, where 7/7 frames returned a confident pose of the wrong human. Raising `num_poses=4` and adding `subject_selection.py` was the correct emergency fix, but you are now doing multi-person tracking *on top of a single-person estimator*, using hand-derived heuristics, with no appearance model and no re-identification.

Meanwhile [SportsMOT (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Cui_SportsMOT_A_Large_Multi-Object_Tracking_Dataset_in_Multiple_Sports_Scenes_ICCV_2023_paper.pdf) establishes for exactly this setting that **"the main challenge lies in object association rather than object localization"**, and that appearance features matter most: BoT-SORT-ReID reaches **HOTA 73.7** vs ByteTrack's **64.1** on sports footage. **[DIRECT]** Your current association uses hip-center proximity and torso size only — no appearance at all.

### Finding 3 — Your motion-energy sampler is dead code

Verified directly:

```
$ grep -n "motion_energy_phase_indices" dataset/*.py
zero_storage_pipeline.py:451:def motion_energy_phase_indices(...)     ← defined
test_motion_energy_sampling.py:...                                    ← tested (7 tests)
                                                                       ← NEVER CALLED
```

`extract_keypoints_in_memory` still runs `indices = [int(start_frame + (window_frames * i / (N_FRAMES - 1))) ...]` — plain uniform `linspace`. The MGSampler-grounded fix was written, tested, committed, and never wired in.

**The single most important structural insight in this report:** your wrist-speed contact sampler was disabled because it locked onto the feeder — but it failed *because it ran before subject selection*, taking `pose_landmarks[0]` from a raw detector. The algorithm was not wrong; **the pipeline order was wrong.** Reordering so that subject selection happens *first*, and contact detection runs only on the confirmed batsman track, recovers already-written, already-tested code and eliminates its failure mode. This is the highest-value, lowest-cost change available to you.

---

# 2. CURRENT PIPELINE DIAGNOSIS

Traced against source, not documentation.

| Stage | Code | What actually happens | Failure mode |
|---|---|---|---|
| Macro window | `batch_urls.csv` → `process_single_row` | Optional since the auto-scan change; blank ⇒ `find_shot_windows_auto` tiles the whole video (25s tiles, 5s overlap) | Tiling is sound; per-tile resolution is still ≤10 frames |
| Shot localization | `nvidia_client.find_shot_windows` | ≤10 evenly-spaced frames → VLM returns frame *indices* → local arithmetic | **VLM temporal blindness.** Window edges snap to sample times: a 25s tile at 10 samples = **2.5s granularity**, but a real cricket shot is ~1–3.5s. The quantisation is the same order as the event. |
| Phase sampling | `extract_keypoints_in_memory` | Uniform `linspace` across the window | A swing is not temporally uniform. Stance/trigger is near-static; backlift→contact is explosive. "Contact" routinely lands on a transitional frame. |
| Pose | `extract_features_from_image_array` | MediaPipe heavy, `num_poses=4` | Single-person estimator used multi-person |
| Subject selection | `subject_selection.py` | Track by hip-center → ≥4/7 coverage → 1.25× size dominance → traversal filter → **reject if ambiguous** | No appearance/ReID. Geometry only. Rejects rather than guesses (correct) but recovers no session. |
| Validation | `validate_pose`, `apply_pipeline_rules` | Per-phase Y-ordering topology; interpolate ≤2 interior, reject ≥3 consecutive/≥4 total | **Purely geometric.** A standing feeder passes every check. |
| Quality gates | `kinematic_validator`, `stance_symmetry_confidence`, `merge.py` | Bone-length CV > 0.25 ⇒ invalid; jerk + bilateral symmetry | Catches *inconsistency*, not *consistent wrongness*. Caught 2 of 3 feeder shots — by accident. |
| Features | `feature_engineering.py` | 15 angles + 15 velocities, pad to 7 | Inherits every upstream error silently |

### The causal chain, and where the damage is worst

```
bad temporal window ──► bad frames ──► WRONG PERSON ──► valid-looking pose ──► plausible features ──► confident score
      (2.5s quantisation)   (uniform)      (no ReID)        (topology passes)    (angles are real)     (no signal it's wrong)
```

**Quantified ranking of current damage** (from your own audit + this analysis):

1. **Wrong person** — measured at 3/3 AI-detected shots on real nets footage; only 2/3 caught downstream, and *by side-effect*. Catastrophic because it is silent.
2. **Wrong temporal window** — 2.5s quantisation against a 1–3.5s event.
3. **Wrong frames within window** — uniform sampling on a non-uniform action; fix exists but is not wired in.
4. **No semantic validation** — nothing anywhere asks "is this a batsman?"
5. **No ground truth** — you cannot currently measure any of the above. This is why 1–4 persisted.

---

# 3. LITERATURE REVIEW

Cricket-specific first, as instructed, then expanding outward.

## 3.1 Cricket-specific

### Cricket stroke extraction: Towards creation of a large-scale cricket actions dataset
**Gupta & Balan, 2019** · [arXiv:1901.03107](https://arxiv.org/abs/1901.03107)
- **Problem:** Temporal action localization of cricket strokes in untrimmed telecast video — *your exact Stage 5 problem*.
- **Method:** Random forest on summed grayscale histogram differences for camera-transition detection; two linear SVMs on HOG features to identify first frames per camera angle; combine positively-predicted shots at predicted boundaries. Deliberately avoided tracking/motion features for generalizability.
- **Dataset:** ~73 million frames across 1,110 videos.
- **Result:** **weighted mean TIoU 0.5097** on a small test-set sample.
- **Strengths:** Genuinely cricket-specific; cheap classical features.
- **Weaknesses:** Authors acknowledge evaluation was on a limited sample; effectiveness on the full dataset unclear.
- **Relevance to SuhasVision:** This is the sobering benchmark. A cricket-specific published method with **8.5× your video count** achieves ~0.51 TIoU on shot localization. Calibrate your expectations accordingly — and note it targets *broadcast* footage with camera cuts, which your nets footage lacks.
- **Transferability: HIGH** (same sport, same task) but the *method* is broadcast-specific.
- **Feasibility: Moderate** — but the camera-transition signal doesn't exist in single-camera nets video.

### Deep-Learning-Based CV Approach for Segmentation of Ball Deliveries and Tracking in Cricket
**2022** · [arXiv:2211.12009](https://arxiv.org/abs/2211.12009)
- **Relevance:** Delivery segmentation is the temporal unit above a shot. **Transferability: MEDIUM** — broadcast-oriented; ball tracking needs a high-contrast ball your amateur footage won't reliably provide.

### Cricket shot classification body of work (2024–2026)
[Shot-ViT](https://www.ije.ir/article_195819.html) · [Modern DL baselines for cricket shot classification, arXiv:2510.09187](https://arxiv.org/pdf/2510.09187) · [Nature Sci Reports 2026](https://www.nature.com/articles/s41598-026-52617-1) · [IEEE 2024](https://ieeexplore.ieee.org/document/10726249/)
- **Observation that matters for you:** this literature is almost entirely *classification of pre-trimmed clips*, not extraction from raw video. Reported figures (e.g. ~90.77% with ResNet-34 for pose classification, ~85% precision integrating pose estimation) assume the shot is already correctly cropped and the right player is in frame.
- **This is the research gap** (see §5).

## 3.2 Swing-phase segmentation — the closest structural analogue

### GolfDB: A Video Database for Golf Swing Sequencing
**McNally, Vats, Pinto, Dulhanty, McPhee, Wong — CVPRW 2019** · [arXiv:1903.06528](https://arxiv.org/pdf/1903.06528) · [code](https://github.com/wmcnally/golfdb)
- **Problem:** Detect 8 canonical swing events in trimmed golf swing videos — *structurally identical to your 7 cricket phases*.
- **Method (SwingNet):** MobileNetV2 CNN → bidirectional LSTM, mapping an RGB sequence to per-frame event probabilities. Deliberately lightweight.
- **Dataset:** **1,400 videos**, each labelled with event frames, bounding box, player name/sex, club type, view type.
- **Results:** **76.1%** average detection rate for all 8 events; **91.8%** for 6 of 8.
- **Strengths:** Direct proof that per-frame event-probability regression beats fixed-fraction sampling for swing phases. Lightweight enough for your compute budget. Public code.
- **Weaknesses:** Needs ~1,400 event-labelled videos. Golf swings are more standardised than cricket shots (fixed stance, no bowler, no reactive timing).
- **Relevance:** **This is the single most transferable paper in this report.** It validates your 7-phase framing as standard practice, and it gives you the exact architecture to replace uniform sampling with — *if* you can label events.
- **Transferability: HIGH.** Golf and cricket swings share the address→backswing→downswing→impact→follow-through structure.
- **Feasibility: Moderate** — architecture is easy; **the labelling is the work** (see §9).

### BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition
**2025** · [arXiv:2502.21085](https://arxiv.org/html/2502.21085v2)
- **Key idea for you:** fuses **skeleton joints + trajectories + player positions** to classify strokes. The explicit use of *player position* as a discriminative signal is directly borrowable for batsman-vs-bowler.
- **Transferability: MEDIUM-HIGH** (racket sport, similar swing dynamics, multi-person court).

## 3.3 Person identification and tracking

### SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes
**Cui et al. — ICCV 2023** · [arXiv:2304.05170](https://arxiv.org/pdf/2304.05170) · [ICCV PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Cui_SportsMOT_A_Large_Multi-Object_Tracking_Dataset_in_Multiple_Sports_Scenes_ICCV_2023_paper.pdf)
- **Dataset:** 240 sequences, >150,000 frames; basketball, volleyball, football. Characterised by fast variable-speed motion and **similar object appearance**.
- **Results:** BoT-SORT **HOTA 73.7 / IDF1 74.0 / AssA 61.5**; ByteTrack **HOTA 64.1 / IDF1 71.4 / AssA 52.3** (Train+Val).
- **The finding that should drive your redesign:** *"The main challenge of SportsMOT lies in object association rather than object localization,"* and **"BoT-SORT-ReID's appearance features matter most."**
- **Relevance:** Your wrong-person problem is an **association** problem, and the literature says appearance features are the strongest lever. You currently use none.
- **Transferability: MEDIUM-HIGH.** Cricket is easier in one respect (fewer people, less mutual occlusion) and harder in another (batsman and bowler wear near-identical kit in nets — which *weakens* the appearance advantage; be honest about this).
- **Feasibility: Easy-Moderate.** ByteTrack/BoT-SORT have mature open implementations.

### Skeleton-based Approach for On-the-ball Player Identification in Broadcast Soccer Videos
**2025** · [ResearchGate](https://www.researchgate.net/publication/397520715_Skeleton-based_Approach_for_On-the-ball_Player_Identification_on_Broadcast_Soccer_Videos)
- **Problem:** Identify *which* of many players is the one engaged with the ball — the closest published analogue to "which person is the batsman."
- **Relevance:** Validates the core premise that **skeleton dynamics alone can identify the action-relevant player** among several. **Transferability: MEDIUM** (different sport/geometry, same problem shape). **[RELATED]**

## 3.4 Pose estimation

### RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose
**Jiang et al., 2023** · [arXiv:2303.07399](https://arxiv.org/pdf/2303.07399) · [code](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- **Method:** SimCC-style head treating keypoint localization as classification; designed for real-time deployment.
- **Results:** **RTMPose-m — 75.8% AP on COCO val, 90+ FPS on an Intel i7-11700 CPU**; 430+ FPS on GTX 1660 Ti; RTMPose-s 72.2% AP at 70+ FPS on a Snapdragon 865.
- **Relevance:** **This is your MediaPipe replacement candidate.** It is genuinely multi-person (top-down on detector boxes), CPU-real-time, matching your "CPU-friendly deployment preferred" constraint. **[DIRECT]** for the multi-person deficiency.
- **Transferability: HIGH.** **Feasibility: Moderate** — MMPose install is heavier than `pip install mediapipe`; ONNX export is supported.

### ViTPose / ViTPose++
- Plain ViT backbone; ViTPose-G (1B params) set SOTA on COCO test-dev; ViTPose++ adds multi-dataset training (COCO, AIC, MPII, CrowdPose).
- **Verdict for you: do not use.** Accuracy-first, GPU-budget-dependent. Your bottleneck is *which person*, not *keypoint precision*. **[INFERENCE]**

### SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos
**ECCV 2022** · [arXiv:2112.13715](https://arxiv.org/abs/2112.13715) · [code](https://github.com/cure-lab/SmoothNet)
- **Method:** Temporal-only refinement network; models long-range temporal relations *per joint*, without noisy cross-joint correlations. Motion-aware fully-connected.
- **Key property:** *"As a temporal-only model, a unique advantage of SmoothNet is its strong transferability across various types of estimators and datasets."*
- **Relevance:** Drop-in temporal stabilisation regardless of which pose estimator you keep. **Transferability: HIGH. Feasibility: Easy.**
- **Caveat [INFERENCE]:** it operates on dense sequences. With only 7 sampled frames you have too little temporal context — you would need to pose-track densely across the shot, then sample. That is a real architectural implication, not a free add-on.

## 3.5 Temporal localization and frame selection

### TriDet: Temporal Action Detection with Relative Boundary Modeling
**Shi et al. — CVPR 2023** · [PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Shi_TriDet_Temporal_Action_Detection_With_Relative_Boundary_Modeling_CVPR_2023_paper.pdf) · [code](https://github.com/dingfengshi/TriDet)
- **Results:** **69.3% avg mAP on THUMOS14** (vs ActionFormer's 66.8%); SOTA on THUMOS14, HACS, EPIC-KITCHENS-100 at lower computational cost. [TBT-Former](https://arxiv.org/html/2512.01298) later reports 68.0%.
- **Verdict: DO NOT PURSUE.** These require frame-accurate boundary supervision over thousands of instances. You have 130 sessions and **zero boundary annotations** — two to three orders of magnitude short. **[DIRECT evidence of infeasibility]**

### MGSampler: An Explainable Sampling Strategy for Video Action Recognition
**Zhi, Tong, Wang, Wu — ICCV 2021** · [arXiv:2104.09952](https://arxiv.org/abs/2104.09952) · [code](https://github.com/MCG-NJU/MGSampler)
- **Method:** Two principles — *motion sensitive* and *motion uniform*. Builds a **cumulative motion distribution** and samples at equal intervals of accumulated motion rather than time.
- **Results:** effective across five benchmarks; generalizes across backbones, video models and datasets.
- **Relevance:** You have already implemented this correctly (`motion_energy_phase_indices`) — **you just never called it.** **Transferability: HIGH. Feasibility: Already done.**

## 3.6 Contact detection — an important negative result

[Event-based batting impact estimation](https://arxiv.org/pdf/2605.25656) and [tennis racket impact localization](https://arxiv.org/html/2506.08327) both rely on **event cameras** (microsecond temporal resolution, motion-blur immune). The batting paper notes that in baseball, *"a ball travelling at 100 km/h covers approximately 2.8 cm per frame even at 1,000 fps."*

**[DIRECT] implication for you:** at 25–30 fps, a cricket ball at ~130 km/h travels **~1.2–1.4 m per frame**. **True bat-ball contact is not resolvable from your footage.** Any "contact frame" you detect is inherently ±1 frame. Design for *contact-adjacent frame identification*, and do not chase precision your frame rate cannot deliver. This also means: do not buy an event camera; reframe the goal.

## 3.7 Downstream context

[A Comprehensive Survey of Action Quality Assessment (arXiv:2412.11149, Dec 2024)](https://arxiv.org/abs/2412.11149) confirms **Spearman Rank Correlation (SRCC) is the primary AQA metric**, over MAE. Benchmarks: MTL-AQA, AQA-7, FineDiving, JIGSAWS, LOGO, FS1000.

> ⚠️ **Correction to a previous document.** `IMPROVEMENT_ROADMAP.md` cites Pirsiavash et al. (ECCV 2014) as achieving "SRC 0.41–0.45" on ~150 samples. I attempted to re-verify that figure during this research and **could not confirm it from an accessible source**. The paper exists ([Springer](https://link.springer.com/chapter/10.1007/978-3-319-10599-4_36)), but treat the specific number as unverified until you read the PDF directly. Do not quote it in an interview.

---

# 4. LITERATURE → IDEA EXTRACTION

| Paper | Problem | Key technique | Result | Idea for SuhasVision | Difficulty |
|---|---|---|---|---|---|
| GolfDB/SwingNet (CVPRW'19) | Swing event detection | CNN→BiLSTM per-frame event probabilities | 76.1% (8 events), 91.8% (6/8) | Replace uniform sampling with learned phase detection | Moderate |
| SportsMOT (ICCV'23) | Sports MOT | Detection + association benchmark | BoT-SORT HOTA 73.7 | **Association, not detection, is the hard part; appearance matters most** | Easy-Mod |
| RTMPose (2023) | Real-time multi-person pose | SimCC head | 75.8% AP, 90+ FPS CPU | Multi-person pose that is actually designed for it | Moderate |
| SmoothNet (ECCV'22) | Pose jitter | Temporal-only refinement | Transfers across estimators | Stabilise pose before angles | Easy |
| MGSampler (ICCV'21) | Frame sampling | Cumulative motion distribution | Generalizes across backbones | **Already coded — wire it in** | Trivial |
| Cricket stroke extraction (2019) | Cricket TAL | Camera-cut RF + HOG SVM | TIoU 0.5097 (1,110 videos) | Realistic expectation calibration | Moderate |
| TriDet (CVPR'23) | TAL | Relative boundary modelling | 69.3% mAP THUMOS14 | **Ruled out** — data requirements | Research-grade |
| On-ball player ID (2025) | Which player has the ball | Skeleton dynamics | — | Skeleton alone can identify the action-relevant player | Moderate |
| BST (2025) | Badminton strokes | Skeleton + trajectory + **position** | — | Court/crease position as a discriminative feature | Moderate |
| Video-LLM grounding (2025) | Temporal grounding | Multiple | "Temporally blind" | **Stop using VLM as primary localizer** | — |
| Event-camera impact (2025-26) | Contact timing | Event cameras | 2.8cm/frame @1000fps | Contact is frame-rate-bounded; reframe goal | N/A |

### Synthesis: what the evidence collectively supports

The literature does **not** support a single end-to-end model at your data scale. It supports a **staged pipeline where each stage uses the cheapest method that is actually designed for that stage's problem**:

1. Detection + **appearance-based** tracking for association (SportsMOT) — because association is the hard part.
2. **Track-level** batsman selection fusing position, persistence, and skeleton dynamics (on-ball player ID, BST) — never per-frame.
3. **Motion-guided** frame selection (MGSampler) rather than uniform — already written.
4. Learned **phase** detection only if you can afford the labels (GolfDB); otherwise contact-anchored heuristics.
5. Temporal pose refinement (SmoothNet) after dense tracking.
6. VLM as a **verifier**, not a localizer — inverting its current role to match its actual strengths.

---

# 5. THE RESEARCH GAP

Supported by §3, not invented:

> **The cricket CV literature almost entirely assumes the extraction problem is already solved.** Published cricket work is dominated by *shot classification on pre-trimmed, single-subject clips* (Shot-ViT, the 2024–2026 classification body). The one cricket paper that tackles extraction directly (Gupta & Balan) targets *broadcast* footage and leans on camera-cut detection — a signal that does not exist in single-camera amateur nets video, and it still reports only ~0.51 TIoU.
>
> **Nobody has published a reliable batsman-identification method for multi-person amateur nets footage.** The on-ball-player-ID work is soccer/broadcast; SportsMOT is basketball/volleyball/football; MediaPipe is single-person by design.

**Your genuine, defensible contribution is not the scoring model — it is a validated extraction pipeline plus a benchmark for amateur multi-person cricket footage.** That is a real gap, it is publishable at a workshop level, and it is what your project is uniquely positioned to fill given the failure data you have already collected.

---

# 6. RECOMMENDED ARCHITECTURE

## THE ONE ARCHITECTURE I WOULD BUILD

**Design principle: reorder before you replace.** Your biggest wins come from fixing *sequencing* and *wiring* in code you already have, not from new models. Only after that do model swaps pay.

```
STAGE 1  VIDEO PREPROCESSING
         Single sequential decode pass. Cache grayscale @ reduced res for the
         motion profile. (Replaces repeated cap.set() random seeks — codec-
         imprecise and slow.)
              │  frames[], gray[]
              ▼
STAGE 2  PERSON DETECTION  ──  YOLOv8/11-pose (or RTMDet)
         All persons per frame, boxes + coarse keypoints in ONE pass.
              │  boxes[frame][person]
              ▼
STAGE 3  TRACKING  ──  ByteTrack (motion) → BoT-SORT-ReID if kit differs
         [SportsMOT: association is the hard part; appearance matters most]
              │  tracks[] with per-frame boxes
              ▼
STAGE 4  BATSMAN IDENTIFICATION  ── track-level, multi-signal (§7)
         ★ MOVED EARLIER — this is the key reordering
              │  ONE confirmed batsman track  (or REJECT)
              ▼
STAGE 5  SHOT LOCALIZATION  ── motion energy of the BATSMAN TRACK only
         VLM demoted from localizer to verifier
              │  refined [t_start, t_end]
              ▼
STAGE 6  CONTACT DETECTION  ── wrist-speed peak ON THE BATSMAN TRACK (§8)
         ★ Your existing, tested code — now safe, because Stage 4 ran first
              │  t_contact (±1 frame, acknowledged)
              ▼
STAGE 7  PHASE SEGMENTATION  ── contact-anchored, asymmetric
              ▼
STAGE 8  FRAME SELECTION  ── 7 indices (motion-guided within phases)
              ▼
STAGE 9  POSE  ──  RTMPose-m on the tracked box  (75.8% AP, 90+ FPS CPU)
              ▼
STAGE 10 TEMPORAL CONSISTENCY  ── track-constrained; SmoothNet if dense
              ▼
STAGE 11 FEATURES  ── UNCHANGED: 15 angles + 15 velocities → (7,30)
              ▼
STAGE 12 QUALITY GATE  ── ShotQuality ∈ [0,1]  (§10)
              │
        ┌─────┴─────┬──────────────┐
     ACCEPT       REVIEW         REJECT
        │            │              │
        ▼            ▼              ▼
STAGE 13 SCORING (unchanged) │  human queue │  logged, not scored
STAGE 14 MC-Dropout uncertainty (unchanged)
STAGE 15 Rule-based fallback (unchanged)
```

### Why this and not the alternatives

| Pipeline | Accuracy | Robustness | Latency | Difficulty | Data need | Verdict |
|---|---|---|---|---|---|---|
| **A. Current** (VLM → uniform 7 → MediaPipe) | Low | Low | 16–19s | — | none | Broken at Stages 4–6 |
| **B. Detector+tracker+multi-pose+heuristic** | Med-High | High | ~+2–4s | Moderate | none | **Core of the recommendation** |
| **C. B + bat detector + role classifier + contact** | High | High | +3–6s | High | ~500–2k labelled | Phase 2 target |
| **D. TAL model (TriDet/ActionFormer) + tracking** | ? | ? | High | Research-grade | 1000s boundary labels | **Ruled out** |
| **E. CV pipeline + VLM verification** | Med-High | **Highest** | +1 API call | Low on top of B | none | **Adopt — cheap insurance** |
| **F. End-to-end video model** | ? | ? | High | Research-grade | 10k+ clips | Ruled out |

**Recommendation: B + E now, C later.** B fixes the catastrophic failures with no new labels. E adds a semantic safety net for the exact failure geometry cannot catch. C is worth it only once you have annotation budget.

---

# 7. BATSMAN-SELECTION ALGORITHM

Track-level, never per-frame. Signals chosen because each is cheap *and* independently motivated by §3.

```python
def batsman_score(track, all_tracks, frame_shape):
    # S1 PERSISTENCE — the batsman is present throughout the shot
    #    [current code, keep] coverage / n_frames
    s_persist = len(track.frames) / n_frames

    # S2 SCALE DOMINANCE — the filmed subject is camera-near
    #    [current code, keep]
    s_scale = mean_torso(track) / max(mean_torso(t) for t in all_tracks)

    # S3 CREASE STATIONARITY — batsman bats from a fixed crease;
    #    bowler/feeder traverses.  [current code, keep — dual-normalized]
    s_static = 1.0 - min(1.0, net_displacement_torsos(track) / 2.0)

    # S4 SWING SIGNATURE  ★NEW — wrist angular excursion relative to hip
    #    motion. A batting swing is wrist-dominant around a stable base;
    #    walking/running is whole-body translation.
    #    [RELATED: BST uses skeleton dynamics for stroke discrimination]
    s_swing = wrist_excursion(track) / (hip_translation(track) + eps)

    # S5 ORIENTATION  ★NEW — batsman faces the bowler; shoulder line is
    #    roughly perpendicular to the camera in front-view footage.
    s_orient = shoulder_line_consistency(track)

    # S6 POSITIONAL PRIOR  ★NEW — batsman occupies a consistent image
    #    region across the shot.  [RELATED: BST uses court position]
    s_position = 1.0 - positional_variance(track)

    return weighted_sum([s_persist, s_scale, s_static,
                         s_swing, s_orient, s_position])
```

**Decision rule — keep your reject-over-guess discipline:**

```
best, runner_up = top two tracks by batsman_score
if best < ACCEPT_FLOOR:              REJECT  ("no confident batsman")
elif best - runner_up < MARGIN:      REJECT  ("ambiguous")
else:                                ACCEPT best
```

**Threshold calibration — do not hand-pick.** Label ~200 tracks from ~50 clips as batsman/not. Fit weights by logistic regression; choose `ACCEPT_FLOOR` and `MARGIN` on a precision-recall curve at **≥98% precision** on "is batsman", accepting whatever recall that costs. Rationale: a wrong person silently corrupts training data; a rejected session only costs visible yield. **[INFERENCE, but consistent with your existing project principle]**

---

# 8. CONTACT DETECTION

**Reuse `find_wrist_speed_peak_frame` — but move it after Stage 4.**

```python
# BEFORE (why it failed):
#   raw detector → pose_landmarks[0] → argmax wrist speed
#   ⇒ locked onto the feeder; measured wrist:hip ratio 34.43 at the
#     false peak, indistinguishable from a real swing.
#
# AFTER:
#   Stage 4 confirms ONE batsman track
#   → wrist speed computed ONLY on that track's keypoints
#   → the failure mode is structurally impossible
```

Robustness improvements over plain argmax:
1. **Smooth first** (Savitzky-Golay over the track's wrist series) — argmax on noisy per-frame speed is fragile.
2. **Bilateral** — use max(left, right) wrist, then verify both accelerate together (a real swing is two-handed in cricket).
3. **Plausibility window** — contact should fall in roughly the middle-to-late portion of the shot; reject peaks at the extreme edges (a peak at fraction <0.2 or >0.95 is likely idle motion).
4. **Report a confidence**, not just an index: peak prominence relative to the median speed.
5. **Accept ±1 frame.** Per §3.6 this is a hard physical bound at 25–30 fps. Do not over-engineer past it.

---

# 9. TEMPORAL SAMPLING STRATEGY

**Contact-anchored, asymmetric, motion-guided.** Uniform sampling is not defensible for a swing.

```
                 t_contact
                     ↓
  ├────────────────────┼──────────┤
  stance ... backlift  ●  follow-through
  ←──── 70% of span ──→ ←── 30% ──→
```

- **Phases 1–5** (stance → downswing) distributed across `[t_start, t_contact]` — **using MGSampler cumulative-motion spacing, not linear** (so the near-static stance/trigger doesn't consume half your frames).
- **Phase 6** = `t_contact` exactly.
- **Phase 7** = follow-through, sampled between `t_contact` and `t_end`.

This is precisely what `redistribute_phase_indices` already does, combined with `motion_energy_phase_indices` which is already written. **Both exist; neither is wired in.**

### Is 7 frames right?

**Keep 7.** Reasons: (a) it matches GolfDB's 8-event framing, so it is not arbitrary; (b) your trained model, feature schema, and entire dataset assume it — changing it invalidates everything at once; (c) at 130 sessions you cannot afford to re-annotate. **[INFERENCE]** Revisit only after extraction is fixed and you have ≥500 clean sessions. If you do revisit, test 9 and 12 against 7 in a controlled ablation, not on intuition.

---

# 10. QUALITY GATE

```
ShotQuality = w1·subject_confidence      (Stage 4 margin)
            + w2·pose_completeness       (frames with valid pose / 7)
            + w3·temporal_consistency    (track continuity, no ID switch)
            + w4·kinematic_validity      (existing bone-length CV)
            + w5·symmetry_confidence     (existing stance symmetry)
            + w6·contact_confidence      (peak prominence, §8)
            + w7·semantic_verification   (VLM check, §11)
```

**Calibrate, don't invent.** Label ~200 extracted sequences as *usable / not usable* by eye (landmark overlays — the method that caught the feeder bug). Fit weights by logistic regression against that label. Then:

- **ACCEPT** — above the threshold where precision on "usable" ≥ 95%
- **REVIEW** — the band where the classifier is genuinely uncertain
- **REJECT** — below the floor

Report `ShotQuality` in the API response alongside `confidence_variance`. It is a *different* thing from model uncertainty: this measures whether the *input* was trustworthy; MC-Dropout measures whether the *model* is confident. Conflating them is a mistake worth avoiding explicitly.

---

# 11. SEMANTIC VALIDATION — INVERT THE VLM'S ROLE

Your VLM is currently the *localizer* (its weakest capability). Make it the *verifier* (its strongest — "what is happening" rather than "when").

Send the 7 **selected, cropped, batsman-only** frames and ask:

```
1. Do all 7 images show the SAME person?              (identity check)
2. Is that person batting (not bowling/fielding)?     (role check)
3. Are these in chronological batting order?          (order check)
4. Which frame shows bat-ball contact?                (anchor cross-check)
5. Is a bat visible in the person's hands?            (equipment check)
```

**Cost: 1 API call per session** (down from ~30 in `find_shot_windows_auto` tiling — this is *cheaper* than today). **Every one of these is a "what/whether" question, which is what the literature says VLMs are good at.** Q4 gives you a free cross-check on Stage 6: if the VLM's contact frame and your wrist-speed peak disagree by >1 frame, flag for REVIEW.

---

# 12. GROUND-TRUTH BENCHMARK — DO THIS FIRST

**You cannot improve what you cannot measure, and you currently measure none of this.** This is the highest-priority deliverable in the entire report.

### Dataset: 100 clips, hand-annotated
Composition: 60 nets/amateur (the hard case, multi-person), 25 broadcast/pro, 15 deliberately adversarial (bowler prominent, occlusion, camera motion, left-handed).

### Per clip, annotate:
| Field | Effort |
|---|---|
| Shot start/end frame | ~30s |
| Batsman bounding box in 7 frames | ~60s |
| Contact frame | ~20s |
| Usable / not usable (binary) | ~10s |

**≈2 min/clip → ~3.5 hours total.** This is the single highest-ROI use of your time in this project.

### Metrics

```
Shot localization:      temporal IoU; start/end error (frames); miss rate; false-shot rate
Subject ID:             batsman accuracy; identity-switch rate; WRONG-PERSON RATE  ← the one that matters
Temporal sampling:      contact-frame error (frames); phase-order accuracy
Pose quality:           valid-pose rate; landmark temporal smoothness
```

### The headline metric

```
                      # sequences where ALL of:
                        (a) all 7 frames = same, correct batsman
                        (b) all 7 within the correct shot
                        (c) frames chronologically ordered
                        (d) contact within ±1 frame of ground truth
E2E-VSER  =  ───────────────────────────────────────────────────────
                      # sequences the pipeline ACCEPTED
```

Report **alongside acceptance rate** — a pipeline that rejects 95% of clips can trivially score E2E-VSER = 1.0. Both numbers, always, together.

### Realistic targets ("bulletproof", operationalised)

| Metric | Current (est.) | Target | Justification |
|---|---|---|---|
| Wrong-person rate | ~33%* | **<2%** | *Measured 3/3 on one nets clip — not dataset-wide; this is exactly why you need the benchmark |
| Identity-switch rate | unmeasured | <3% | Below SportsMOT AssA-implied error for an easier scene |
| Contact error | unmeasured | ≤1 frame | Physical bound (§3.6) |
| Shot tIoU | unmeasured | >0.7 | Above cricket-specific published 0.51 |
| **E2E-VSER** | **unmeasured** | **>90%** at ≥70% acceptance | The number to put on a CV |

---

# 13. MIGRATION ROADMAP

### Phase 0 — Instrumentation *(1 day)* ★ START HERE
- **Objective:** be able to see failures.
- **Code:** dump landmark-overlay JPEGs for every accepted session behind a `--debug` flag; log the full `subject_selection` report per session; build the 100-clip benchmark (§12).
- **Success:** you can state your current wrong-person rate with a number.
- **Accept:** benchmark exists, baseline measured.

### Phase 1 — Wire in what you already wrote *(1 day)* ★ HIGHEST ROI
- **Objective:** stop shipping dead code.
- **Code:** (a) call `motion_energy_phase_indices` in `extract_keypoints_in_memory`, falling back to uniform on `None`; (b) **reorder** so subject selection precedes contact detection; (c) re-enable `WRIST_SPEED_SAMPLING_ENABLED` *only* on the confirmed batsman track.
- **Success:** contact-frame error and E2E-VSER both improve on the Phase-0 baseline.
- **Accept:** no regression in acceptance rate; contact error ≤2 frames.

### Phase 2 — Detection + tracking *(1 week)*
- **Code:** add YOLO-pose detector + ByteTrack; replace `subject_selection.build_tracks` with real tracks; extend `batsman_score` with S4–S6 (§7).
- **Success:** wrong-person rate <5%.
- **Accept:** ≥50% relative reduction vs Phase 0.

### Phase 3 — Pose upgrade *(3–4 days)*
- **Code:** RTMPose-m on tracked boxes behind a feature flag; keep MediaPipe as fallback. Verify angle-schema compatibility (landmark index mapping differs — this is a real migration cost, budget for it).
- **Success:** valid-pose rate up, no latency regression on CPU.
- **Accept:** ≥ MediaPipe on every benchmark metric.

### Phase 4 — Semantic gate *(2–3 days)*
- **Code:** VLM verifier (§11); `ShotQuality` (§10); ACCEPT/REVIEW/REJECT routing.
- **Success:** wrong-person rate <2%; API cost *decreases*.

### Phase 5 — Rebuild dataset, then retrain *(ongoing)*
- Re-ingest with the fixed pipeline. **Only now** revisit the scoring model.
- **Accept:** honest CV MAE beats the 13.59 constant-mean baseline. (It currently does not — see `IMPROVEMENT_ROADMAP.md`.)

---

# 14. ABLATION PLAN

Additive, on the fixed 100-clip benchmark, measuring E2E-VSER / wrong-person rate / contact error:

| # | Configuration | Isolates |
|---|---|---|
| A0 | Current pipeline | Baseline |
| A1 | + motion-energy sampling | MGSampler value alone |
| A2 | A1 + reordered contact detection | **The reordering hypothesis** |
| A3 | A2 + detector/tracker | Real tracking vs hip-proximity |
| A4 | A3 + extended batsman score (S4–S6) | Multi-signal selection |
| A5 | A4 + RTMPose | Pose estimator swap |
| A6 | A5 + VLM semantic gate | Semantic verification |

**Predicted (to be falsified, not assumed):** A2 delivers the largest single jump, because it converts a *disabled* capability into a *working* one at near-zero cost. A3–A4 deliver the largest wrong-person reduction. A5 delivers the least — pose precision is not your bottleneck. **If A5 beats A3, my diagnosis is wrong and you should tell me.**

---

# 15. RISK ANALYSIS

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Batsman and bowler wear identical kit in nets** → ReID appearance features add little | **High** | This is the honest weak point of the SportsMOT transfer. Lean on S3/S4/S6 (position, swing signature, stationarity), not appearance. |
| Benchmark too small (100 clips) to resolve <2% rates | High | 100 clips cannot statistically confirm a 2% rate. Treat targets as directional; widen the benchmark if you approach them. |
| RTMPose landmark schema ≠ MediaPipe's 33 | Certain | Explicit index mapping + numerical equivalence test before switching — the discipline you already used for `TemporalAttention`. |
| Extra stages push latency past acceptable | Medium | Detection+tracking is the cost; batch, and the async queue is already a known gap. |
| Rejection rate climbs so high yield collapses | Medium | Report acceptance rate *with* E2E-VSER, always. REVIEW tier exists for exactly this. |
| Fixing extraction still leaves the model below baseline | **High** | Extraction is necessary, not sufficient. Your label-validity problem (single LLM rater, unvalidated) is a *separate* ceiling — see `IMPROVEMENT_ROADMAP.md`. Do not expect this work alone to fix accuracy. |

---

# 16. TOP 10 PAPERS TO READ

1. **GolfDB / SwingNet** (CVPRW 2019) — closest structural analogue; the phase-detection architecture you'd grow into.
2. **SportsMOT** (ICCV 2023) — proves association is the hard part; sets your tracker expectations.
3. **RTMPose** (2023) — your MediaPipe replacement, with CPU numbers that fit your constraint.
4. **MGSampler** (ICCV 2021) — the sampling theory behind code you already have.
5. **Cricket stroke extraction** (Gupta & Balan 2019) — the only cricket-specific extraction paper; calibrates expectations.
6. **SmoothNet** (ECCV 2022) — cheap temporal pose stabilisation.
7. **AQA survey** (arXiv 2412.11149) — why SRCC, not MAE, is your headline metric.
8. **BST** (2025) — skeleton + trajectory + position fusion for stroke discrimination.
9. **On-ball player identification** (2025) — skeleton-based action-relevant player selection.
10. **TriDet** (CVPR 2023) — read to understand *why you are not using it*.

# TOP 5 METHODS TO ACTUALLY IMPLEMENT

1. **Pipeline reordering: subject selection → contact detection** — near-zero cost, reuses tested code, eliminates a known catastrophic failure. Do this first.
2. **Wire in `motion_energy_phase_indices`** — one line; MGSampler-grounded; already unit-tested.
3. **Detector + ByteTrack + track-level multi-signal batsman scoring** — the real fix for wrong-person.
4. **VLM as verifier, not localizer** — cheaper than today *and* catches what geometry cannot.
5. **The 100-clip benchmark** — technically last in this list, practically first in time. Without it, 1–4 are unfalsifiable.

---

# 17. FINAL RECOMMENDATION

**Build Pipeline B+E: detector → tracker → track-level batsman identification → motion-guided contact-anchored sampling → RTMPose → quality gate → VLM verification.**

**But the first two days of work are not model work at all.** Build the benchmark, then wire in the code you already wrote and reorder two stages. My honest expectation is that Phase 0 + Phase 1 — roughly two days, no new models, no new dependencies — will produce a larger measured improvement than any model swap in this report, because they convert existing, tested, correct code from *disabled* to *working*.

The deepest finding of this investigation is not a paper. It is that **your wrist-speed contact sampler was never actually broken — it was correctly implemented, correctly tested, and wired into the wrong position in the pipeline.** You disabled it for the right reason on the evidence you had. Reordering the pipeline makes its failure mode structurally impossible.

Fix the order. Wire in the sampler. Measure it. *Then* buy new models.

---

## Sources

- [GolfDB: A Video Database for Golf Swing Sequencing (CVPRW 2019)](https://arxiv.org/pdf/1903.06528) · [code](https://github.com/wmcnally/golfdb)
- [SportsMOT (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Cui_SportsMOT_A_Large_Multi-Object_Tracking_Dataset_in_Multiple_Sports_Scenes_ICCV_2023_paper.pdf) · [arXiv](https://arxiv.org/pdf/2304.05170)
- [RTMPose (arXiv 2303.07399)](https://arxiv.org/pdf/2303.07399) · [MMPose code](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- [SmoothNet (ECCV 2022)](https://arxiv.org/abs/2112.13715) · [code](https://github.com/cure-lab/SmoothNet)
- [MGSampler (ICCV 2021)](https://arxiv.org/abs/2104.09952) · [code](https://github.com/MCG-NJU/MGSampler)
- [Cricket stroke extraction (arXiv 1901.03107)](https://arxiv.org/abs/1901.03107)
- [Ball delivery segmentation in cricket (arXiv 2211.12009)](https://arxiv.org/abs/2211.12009)
- [TriDet (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Shi_TriDet_Temporal_Action_Detection_With_Relative_Boundary_Modeling_CVPR_2023_paper.pdf) · [code](https://github.com/dingfengshi/TriDet)
- [TBT-Former](https://arxiv.org/html/2512.01298)
- [A Comprehensive Survey of Action Quality Assessment (arXiv 2412.11149)](https://arxiv.org/abs/2412.11149)
- [Grounded-VideoLLM (EMNLP 2025 Findings)](https://openreview.net/forum?id=YCwN7wQA6W) · [TimeLens](https://arxiv.org/pdf/2512.14698) · [hour-long grounding benchmark](https://arxiv.org/pdf/2606.12300)
- [BST: Badminton Stroke-type Transformer (arXiv 2502.21085)](https://arxiv.org/html/2502.21085v2)
- [Skeleton-based on-ball player identification (soccer)](https://www.researchgate.net/publication/397520715_Skeleton-based_Approach_for_On-the-ball_Player_Identification_on_Broadcast_Soccer_Videos)
- [MediaPipe for sports apps — limitations](https://www.it-jim.com/blog/mediapipe-for-sports-apps/)
- [Event-based batting impact estimation](https://arxiv.org/pdf/2605.25656) · [tennis racket impact](https://arxiv.org/html/2506.08327)
- [Modern DL baselines for cricket shot classification (arXiv 2510.09187)](https://arxiv.org/pdf/2510.09187) · [Shot-ViT](https://www.ije.ir/article_195819.html)
- [Pirsiavash et al., Assessing the Quality of Actions (ECCV 2014)](https://link.springer.com/chapter/10.1007/978-3-319-10599-4_36) — *numbers unverified, see §3.7*
