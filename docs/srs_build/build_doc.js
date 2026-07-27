const {
  fs, Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, BorderStyle, AlignmentType, PageBreak, PAGE_W, PAGE_H,
  FONT_BODY, FONT_HEAD, COL_ACCENT, COL_ACCENT2, COL_TEXT, COL_MUTE, COL_RULE,
  h1, h2, h3, p, pr, run, bullet, reqItem, hr, table, caption,
} = require("./gen_srs.js");
const { TableOfContents } = require("docx");

const children = [];

// ================= TITLE PAGE =================
children.push(
  new Paragraph({ spacing: { before: 2400 }, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "SuhasVision", bold: true, font: FONT_HEAD, size: 64, color: COL_ACCENT })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 360 },
    children: [new TextRun({ text: "AI Cricket Batting Stance Analyzer", font: FONT_HEAD, size: 30, color: COL_TEXT })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    border: { top: { color: COL_ACCENT2, size: 12, style: BorderStyle.SINGLE, space: 8 }, bottom: { color: COL_ACCENT2, size: 12, style: BorderStyle.SINGLE, space: 8 } },
    spacing: { before: 200, after: 200 },
    children: [new TextRun({ text: "SOFTWARE REQUIREMENTS SPECIFICATION", bold: true, font: FONT_BODY, size: 24, color: COL_TEXT, characterSpacing: 20 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 60 },
    children: [new TextRun({ text: "Version 2.0", bold: true, font: FONT_BODY, size: 22, color: COL_ACCENT })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 40 },
    children: [new TextRun({ text: "2026-07-20", font: FONT_BODY, size: 20, color: COL_MUTE })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
    children: [new TextRun({ text: "Supersedes Version 1.0 (2026-07-17)", italics: true, font: FONT_BODY, size: 18, color: COL_MUTE })],
  }),
  new Paragraph({ spacing: { before: 2200 }, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 40 },
    children: [new TextRun({ text: "Prepared by Kurapati Sai Suhas", font: FONT_BODY, size: 20, color: COL_TEXT })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Prepared for technical / research review and interview presentation", italics: true, font: FONT_BODY, size: 18, color: COL_MUTE })],
  }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ================= TABLE OF CONTENTS =================
children.push(h1("Table of Contents"));
children.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ================= DOCUMENT CONTROL =================
children.push(h1("Document Control"));
children.push(table(
  ["Field", "Value"],
  [
    ["Document Title", "SuhasVision — Software Requirements Specification"],
    ["Version", "2.0"],
    ["Status", "Current — reflects verified system state as of the date below"],
    ["Date", "2026-07-20"],
    ["Supersedes", "Version 1.0 (2026-07-17), SuhasVision_SRS_Technical_Analysis.docx"],
    ["Related document", "SuhasVision_V2_Architecture_Specification.docx — a FROZEN, aspirational target architecture (async inference, token blacklisting, etc.). None of that document's provisions are implemented as of this revision; this SRS describes the system exactly as it exists today, not that target."],
    ["Author", "Kurapati Sai Suhas"],
    ["Verification method", "Every quantitative claim in this document was produced by running the referenced script, test, or live inference call during the authoring of this revision (or the revision it directly carries forward), not reconstructed from memory. Where a figure could not be verified, it is explicitly marked NOT VERIFIED rather than omitted or guessed."],
  ],
  [3200, 6600],
));

// ================= REVISION HISTORY (moved up front, high information value) =================
children.push(h1("Revision History"));
children.push(p("This revision (2.0) is a substantial update to Version 1.0, produced after a dedicated data-quality and model-validation pass. It is not a cosmetic refresh: two real data-corruption bugs were found and fixed, the production training dataset was rebuilt and scoped, a candidate model was trained and rigorously compared against the deployed one, a data-augmentation strategy was tested and rejected on measured evidence, and the candidate model was validated through the live inference path before any promotion decision was made. Every one of those activities is documented in the sections that follow, with the exact commands and numbers behind each claim."));
children.push(table(
  ["#", "Change", "Section(s)"],
  [
    ["1", "Added bowling_type as a tracked metadata dimension across the labelling schema, pipeline, and dataset.", "§7.1, §7.2"],
    ["2", "Found and fixed a frame-name convention bug in feature_engineering.py that silently zeroed out metadata (including model features) for 42% of sessions using the pipeline's current naming convention. Recovered those sessions.", "§7.3, Known Issues"],
    ["3", "Found and fixed a silent score-fallback bug in nvidia_client.py: a missing AI-assessed score defaulted to 50 instead of failing the label. Excluded 90 of 496 already-corrupted label rows from training.", "§7.3, Known Issues"],
    ["4", "Rebuilt a clean, explicitly-scoped production training dataset (front-view only, fast-bowling only, zero placeholder scores) and re-measured cross-validated accuracy under a controlled, seeded protocol.", "§6.6, §7.2"],
    ["5", "Trained a candidate model (v5) on the rebuilt dataset and directly compared it against the deployed model (v4) using identical, fair methodology — not just re-quoting a newer number.", "§6.6, §9"],
    ["6", "Designed, implemented, and evaluated a flip + Gaussian-jitter data augmentation pipeline. Measured, rather than assumed, its effect via matched, seeded, identity-grouped cross-validation — found it degraded accuracy and did not adopt it.", "§7.5"],
    ["7", "Validated the candidate model through the actual live inference code path (ml_service.run_advanced_inference, not an offline evaluation harness) against real video, before making any promotion decision.", "§9.4"],
    ["8", "Corrected a stale performance claim: end-to-end inference latency was previously documented as ~6–7 seconds. Live measurement this revision found materially higher real latency; see §8.3 and Known Issues.", "§8.3, Known Issues"],
    ["9", "Expanded automated test coverage across the dataset/evaluation pipeline (evaluation protocol, feature engineering, augmentation, model layer registry) since Version 1.0.", "§10.4"],
  ],
  [500, 6900, 1600],
));

// ================= 1. INTRODUCTION =================
children.push(h1("1. Introduction"));

children.push(h2("1.1 Purpose"));
children.push(p("This document specifies the requirements, architecture, and current implementation state of SuhasVision, an AI-assisted cricket batting-stance analysis platform. It is written for a technical or research reviewer — including an interview panel — who needs an accurate, evidence-backed picture of the system, not a marketing description of intended capability. Every quantitative or behavioral claim in this document was verified against the running code, an executed script, a live inference call against real video, or a database/migration inspection performed while this revision was authored. Claims describing aspiration, future work, or a design target that is not yet built are explicitly labelled as such — this document does not blur the line between what exists and what is planned, because a prior iteration of this project's own SRS was found to contain unmeasured, invented accuracy figures, and this document exists partly to make sure that pattern does not repeat."));

children.push(h2("1.2 Scope"));
children.push(p("SuhasVision analyzes a short video clip of a batter's stance and stroke, extracts pose landmarks via computer vision, computes biomechanical features from those landmarks, and produces four 0–100 scores — Balance, Power, Technique, and Defence — via a trained neural network. Alongside the scores, the system reports a per-score uncertainty estimate, identifies which specific joints most drove the weakest score using a gradient-based attribution method, and recommends a matched practice drill. Two user roles are served — Coach and Learner — through a shared Django REST backend and a React single-page frontend. A separate, offline pipeline sources training video (from YouTube and Instagram), labels it (with AI assistance), extracts pose keypoints, engineers features, and trains the deployed model. Version 1 of this system is deliberately scoped to front-view camera framing and fast-bowling deliveries only; side/back-view footage and spin/swing bowling are explicitly out of scope for this release (see §2.7)."));

children.push(h2("1.3 Intended Audience and Reading Suggestions"));
children.push(p("Written for a technical reviewer, research collaborator, or interview panel with a computer-vision / applied-ML background; cricket domain knowledge is not assumed and is explained inline where needed. A reader focused purely on system architecture should read §3–§6; a reader focused on the ML methodology and its evidentiary basis should read §7 and §9; a reader auditing production-readiness should read §8, §10, and Known Issues (§11)."));

children.push(h2("1.4 Document Conventions"));
children.push(p("Functional requirements are numbered FR-<AREA>-<NNN> (e.g. FR-AUTH-001). Non-functional requirements are numbered NFR-<AREA>-<NNN>. Identified gaps are numbered ISSUE-<NNN> and collected in §11, cross-referenced from the sections where they are technically relevant. Every requirement and issue entry states its status — Implemented, Partially Implemented, Planned (not yet built), or a specific measured value — rather than being asserted without qualification. Colour is used consistently: green-labelled status text means implemented and verified; amber means partial; the accent red means planned/open."));

children.push(h2("1.5 Definitions, Acronyms, and Abbreviations"));
children.push(table(
  ["Term", "Definition"],
  [
    ["Session", "One batting stroke, represented as a fixed 7-frame sequence spanning stance to follow-through."],
    ["Phase", "One of the 7 fixed points sampled across a session: stance, trigger, backlift start, full backlift, downswing, contact, follow-through."],
    ["Identity", "One unique batsman/source. Used to group all recordings — and any flipped/jittered augmented clones — of the same real person, so that no evaluation fold can see the same person's biomechanics in both its training and test portions."],
    ["MAE", "Mean Absolute Error, the primary reported accuracy metric, on the model's native 0–100 point scale per score."],
    ["RMSE / R²", "Root-Mean-Squared Error and coefficient of determination; reported alongside MAE in this revision's model-comparison work (§9) for a fuller accuracy picture."],
    ["MC-Dropout", "Monte Carlo Dropout — running the trained network with dropout still stochastically active at inference time, 30 times, to obtain a mean prediction and a standard deviation that serves as a per-score uncertainty estimate."],
    ["Expected Gradients", "A gradient-based feature-attribution method (Erion et al., 2021) used to identify which input features (joints) most influenced a specific output score."],
    ["Zero-storage", "The architectural policy that an uploaded video is deleted immediately after inference completes (success or failure); no video is retained server-side."],
    ["Kinematic validity", "A per-frame flag indicating a pose passed a bone-length anatomical-plausibility check."],
    ["Temporal confidence", "A continuous, session-level confidence score (Milestone 3 filter) reflecting stance-symmetry-based consistency across a session's frames; sessions below a fixed threshold are excluded from training."],
    ["GroupKFold / identity-grouped CV", "Cross-validation where fold membership is assigned by identity, not by session, so a person's sessions (including augmented clones) can never split across train and test within a fold."],
    ["Production dataset", "The current, explicitly-scoped, bug-fixed dataset used as this revision's training/evaluation baseline: dataset_angles_production.csv (see §7.2)."],
    ["RC / Release Candidate", "A model or dataset artifact that has passed the project's promotion discipline (§9.5) and is eligible to become the deployed production model."],
  ],
  [2200, 7000],
));

children.push(h2("1.6 References"));
children.push(bullet("dataset/EVALUATION_RESULTS.md — dated evaluation log, source of every cross-validation figure cited in §9."));
children.push(bullet("docs/architecture_ground_truth.md — dated architecture and milestone log."));
children.push(bullet("SuhasVision_SRS_Technical_Analysis.docx (Version 1.0, 2026-07-17) — the prior revision this document supersedes and carries forward."));
children.push(bullet("SuhasVision_V2_Architecture_Specification.docx (Version 2.0, frozen 2026-07-17) — an aspirational target architecture, not yet implemented; referenced in §11 where relevant, not conflated with current state."));
children.push(bullet("Erion, G. et al. (2021). \"Improving performance of deep learning models with axiomatic attribution priors and expected gradients.\" — the attribution method implemented in §6.5."));
children.push(bullet("requirements.txt (repository root) — the pinned dependency manifest referenced throughout §5."));

// ================= 2. OVERALL DESCRIPTION =================
children.push(h1("2. Overall Description"));

children.push(h2("2.1 Product Perspective"));
children.push(p("SuhasVision is a standalone full-stack application, not an extension of an existing product. It consists of three independently runnable parts sharing one artifact contract — the trained .keras model file and the 30-feature schema it expects: a Django REST API (authentication, data models, inference orchestration), a React frontend (two role-specific dashboards), and an offline dataset/training pipeline (video sourcing, labelling, pose extraction, feature engineering, model training, evaluation) that produces the model the API loads at inference time. The three parts are decoupled at the file-system boundary — the backend never trains a model, and the training pipeline never serves a live request — which keeps the system's operational surface (what must run in production) small relative to its research surface (what runs offline to produce a better model)."));

children.push(h2("2.2 Product Functions (Summary)"));
children.push(bullet("Account registration and role-based authentication for two roles: Coach (academy) and Learner (player)."));
children.push(bullet("Video upload and automated stance analysis, returning four 0–100 scores, an uncertainty estimate per score, and a joint-level explanation of the weakest score with a matched drill."));
children.push(bullet("A learner-facing dashboard: score history, a points-based tier, and AI feedback per session."));
children.push(bullet("A coach-facing dashboard: a review queue across all of an academy's players, and the ability to override AI-generated scores with a logged reason."));
children.push(bullet("An offline dataset-collection and model-training pipeline, independently versioned and evaluated (§7, §9)."));

children.push(h2("2.3 User Classes and Characteristics"));
children.push(h3("Coach"));
children.push(p("Represents an academy account. A coach sees a review queue of their players' sessions and a per-session analysis view where they can override the AI's Balance/Power/Technique/Defence scores and attach a note; every changed metric is logged (old value, new value, reason, timestamp) in CoachOverrideLog for traceability. Any user who signs up with role=COACH is granted a fully functional coach account immediately — there is currently no verification or approval gate (ISSUE-007, §11)."));
children.push(h3("Learner / Player"));
children.push(p("Represents an individual batter. A learner uploads their own videos, sees their own score history and a gamification profile (a tier computed from a real total_points field), and a feedback view showing AI analysis plus any coach override. A learner has no access to any coach-only route or data; this is enforced server-side by query scoping (FR-AUTH-002), not only hidden in the UI."));

children.push(h2("2.4 Operating Environment"));
children.push(p("Verified directly from the project's pinned dependency manifest (requirements.txt, repository root) rather than assumed from package.json intent:"));
children.push(table(
  ["Layer", "Technology", "Pinned Version"],
  [
    ["Backend framework", "Django", "6.0.5"],
    ["API layer", "Django REST Framework", "3.17.1"],
    ["Auth", "djangorestframework-simplejwt", "5.5.1"],
    ["ML framework", "TensorFlow / Keras", "2.21.0 / 3.15.0"],
    ["Pose estimation", "MediaPipe", "0.10.35"],
    ["Computer vision", "opencv-python / opencv-contrib-python", "4.13.0.92 / 5.0.0.93"],
    ["Data processing", "pandas / numpy", "3.0.3 / 2.4.6"],
    ["Video sourcing", "yt-dlp", "2026.3.17"],
    ["AI labelling client", "openai (pointed at an NVIDIA NIM-hosted, OpenAI-API-compatible endpoint)", "2.44.0"],
  ],
  [2400, 4800, 2000],
));
children.push(caption("Table 2.1 — Pinned dependency versions relevant to this specification. Backend, ML service, and the offline dataset pipeline currently share one virtual environment and one requirements.txt."));
children.push(p("Both opencv-python and opencv-contrib-python are installed simultaneously — this is not flagged as a functional defect (contrib is a superset), but it is worth noting as an unreviewed redundancy rather than a deliberate choice, since only one is required for the landmarks this system actually uses."));

children.push(h2("2.5 Design and Implementation Constraints"));
children.push(bullet("Zero-storage privacy policy: no uploaded or sourced video may be persisted server-side beyond the duration of a single inference call (enforced via try/finally deletion, not policy alone)."));
children.push(bullet("Model input shape is fixed at (7, 30): 7 sampled frames × 30 hand-engineered features (15 joint angles + 15 angular velocities). Changing this shape requires retraining and touches multiple independent call sites, not a single config value."));
children.push(bullet("The reference development machine runs Windows with an Application Control security policy that permanently blocks numba's native DLL (breaking the shap package's dependency chain — worked around by implementing Expected Gradients directly, §6.5) and has intermittently caused the local Django dev server to fail to start due to an OpenCV native-DLL block; retrying the server start resolves it. This is an environment constraint, not a code defect, and does not affect Linux/Docker deployment."));
children.push(bullet("Inference is currently fully synchronous — a single HTTP request blocks for the full pose-extraction + scoring + attribution pipeline (see §8.3 for current measured latency, which is materially higher than previously documented)."));

children.push(h2("2.6 Assumptions and Dependencies"));
children.push(bullet("Assumes front-view camera framing (batter facing the camera). This is a deliberate Version 1 scope decision (§2.7), not an accidental limitation — the training dataset is explicitly filtered to front-view sessions only."));
children.push(bullet("Assumes fast-bowling deliveries. The training dataset's bowling_type field is 100% \"fast\" in the current production scope; spin and swing bowling are not represented and should not be assumed to generalize (§2.7)."));
children.push(bullet("Depends on an NVIDIA NIM-hosted vision-language model (accessed via an OpenAI-API-compatible client) for automated shot-window detection and session labelling during dataset construction only — never at live-inference time, which uses only the trained Keras model and MediaPipe."));
children.push(bullet("Depends on MediaPipe's BlazePose PoseLandmarker (\"heavy\" variant) for all pose extraction, both at training-data-collection time and at live-inference time — the same landmark topology (33 points) is used throughout, avoiding a train/serve skew in the input representation."));

children.push(h2("2.7 Apportioning of Requirements — Version 1 Scope"));
children.push(p("This project's V1 scope was deliberately narrowed to maximize the quality and defensibility of a smaller, cleaner dataset over the breadth of a larger, noisier one — a decision made and re-validated multiple times during this revision's work (§7, §9). The table below states what is in and out of scope for this release, and why."));
children.push(table(
  ["Dimension", "V1 Scope", "Rationale"],
  [
    ["Camera angle", "Front-view only", "Front-view sessions measured strictly better (lower CV MAE, lower fold variance) than a mixed-view set in prior evaluation; a single, consistent camera geometry is easier for a 30-dimensional hand-engineered feature set to learn."],
    ["Bowling type", "Fast bowling only", "The dataset collected to date is fast-bowling-only; the bowling_type field now exists in the schema (added this revision) specifically so spin/swing can be added as a distinguishable category in V2 rather than silently mixed in."],
    ["Batting hand", "Both, but severely imbalanced", "129 of 130 production sessions (99.2%) are right-handed; only 1 real left-handed identity exists in the current dataset. A tested mitigation (mirrored augmentation) is documented and was rejected on measured evidence (§7.5) — this imbalance is an open, acknowledged limitation, not a hidden one."],
    ["Format", "Not distinguished", "Junior/tennis-ball cricket and match-context metadata are out of scope; not tracked in the current schema."],
    ["Playing level", "Tracked as metadata", "skill_level (Amateur/Club, Youth/Academy, Professional) is recorded and used for evaluation breakdowns (§9.2) but is not a model input feature."],
  ],
  [2000, 2400, 4800],
));

// ================= 3. SYSTEM ARCHITECTURE =================
children.push(h1("3. System Architecture"));

children.push(h2("3.1 High-Level Architecture"));
children.push(p("Three tiers, sharing one artifact contract (the trained .keras model and its 30-feature schema):"));
children.push(bullet("Frontend — React + TanStack Router (file-based routing) + TanStack Query. Three route files: index.jsx (unauthenticated login/registration), coach.jsx, learner.jsx — each a single-page dashboard using an internal tab-switch pattern rather than nested sub-routes. Shared chrome (navigation, topbar, profile menu) lives in AppShell.jsx, parameterized by role."));
children.push(bullet("Backend — Django REST Framework, JWT authentication via SimpleJWT. Role is determined structurally, not by a stored flag: a request's user either has an associated Academy row (coach) or PlayerProfile row (learner) via a one-to-one relation, and every queryset-returning view branches on hasattr(user, 'academy') / hasattr(user, 'playerprofile') — this is what actually prevents a learner from seeing another learner's or a coach's data, independent of any frontend route guard."));
children.push(bullet("ML / Dataset pipeline — an offline, independently-run set of scripts (dataset/) that source video, label it, extract pose keypoints, engineer features, train models, and evaluate them. Produces the .keras file the backend loads; never runs as part of a live request."));

children.push(h2("3.2 High-Level Data Flow (Live Inference)"));
children.push(p("Learner uploads video → Django validates file type (MP4/MOV/WEBM/AVI) and size (≤200MB) → video saved to a temporary path → 7 frames sampled at evenly-spaced indices via MediaPipe PoseLandmarker (\"heavy\" variant) → 15 joint angles + 15 frame-to-frame angular velocities computed per frame (30 features × 7 frames) → the trained Conv1D + BiLSTM + Attention network scores the sequence across 30 stochastic MC-Dropout forward passes → Expected Gradients attribution identifies the joints driving the weakest of the four scores → result persisted to the database → temporary video file deleted (try/finally, both success and failure paths) → JSON response returned to the frontend. If any step in the ML path raises, the system falls back to a deterministic rule-based scorer operating on the same extracted keypoints rather than surfacing a server error (§4.7)."));
children.push(p("Measured end-to-end latency for this path is reported in §8.3 — a figure this revision corrects from what was previously documented, based on new live measurement rather than continuing to cite an unverified older number."));

children.push(h2("3.3 Data Model"));
children.push(p("Four Django models (backend/backend/api/models.py):"));
children.push(table(
  ["Model", "Key Fields", "Purpose"],
  [
    ["Academy", "user (1:1 → User), academy_name, created_at, is_verified", "A coach's account. Any signup with role=COACH creates one immediately, but is_verified defaults to False; coach-only actions (dashboard, analyze_stance-for-a-player, score overrides) 403 until an admin flips it via Django admin (ISSUE-007, closed)."],
    ["PlayerProfile", "user (1:1, nullable), academy (FK, nullable), name, batting_hand [Right/Left], playing_level, archetype, total_points, current_streak", "A learner's account. academy is nullable so a coach can create a guest player profile without that player having their own login."],
    ["AnalysisSession", "player (FK), title, status, video_url, date_analyzed, overall_score, primary_weakness, primary_strength, thing_to_change, bonus_insight, balance_score, power_score, technique_score, defence_score, confidence_variance (JSON), attribution_drivers (JSON), is_fallback", "One completed analysis. confidence_variance and attribution_drivers persist the MC-Dropout uncertainty and Expected-Gradients output respectively — both are computed at inference time and were, in an earlier iteration of this system, discarded rather than stored; both are now written through to the database."],
    ["CoachOverrideLog", "session (FK), coach (FK), metric_changed, old_value, new_value, reason, timestamp", "One row per individual score field a coach actually changes on a PATCH — not one row per PATCH request — giving field-level audit history."],
  ],
  [2200, 3800, 3200],
));
children.push(caption("Table 3.1 — Core Django models. All four are migrated and in active use; none are placeholder/unused models."));

children.push(h2("3.4 ML Pipeline Architecture"));
children.push(p("The deployed architecture, confirmed against the live model artifact (not inferred from the training script alone): a 1D convolutional block extracts localized temporal patterns across the 7-frame sequence, a bidirectional LSTM processes the sequence in both time directions, a custom additive temporal-attention layer pools the sequence into a single context vector by learning which of the 7 phases matter most for scoring, and a small dense regression head produces the four sigmoid-scaled outputs."));
children.push(table(
  ["Layer", "Configuration", "Output Shape"],
  [
    ["Input", "(7 timesteps, 30 features)", "(7, 30)"],
    ["Conv1D", "64 filters, kernel size 3, padding=same, ReLU", "(7, 64)"],
    ["BatchNormalization + Dropout(0.2)", "—", "(7, 64)"],
    ["Bidirectional LSTM", "64 units, return_sequences=True", "(7, 128)"],
    ["Dropout(0.3)", "—", "(7, 128)"],
    ["TemporalAttention", "Custom additive attention pooling over the time axis", "(128,)"],
    ["Dense", "32 units, ReLU", "(32,)"],
    ["Dense (output)", "4 units, sigmoid, scaled ×100 for reporting", "(4,) — Balance, Power, Technique, Defence"],
  ],
  [3200, 4200, 1800],
));
children.push(caption("Table 3.2 — 76,395 trainable parameters (76,523 including non-trainable BatchNorm statistics). Architecture is defined once, in dataset/academic_scripts/train_advanced_model.py's build_tcn_attention_model(), and the TemporalAttention layer is a single shared module (dataset/model_layers.py) imported by every consumer — training, cross-validation, and live inference — after a consolidation pass eliminated four independent redefinitions of the same layer."));

children.push(h2("3.5 Data Ingestion and Training Pipeline"));
children.push(p("Five sequential stages, each consuming the previous stage's output file, each independently runnable and versioned:"));
children.push(bullet("zero_storage_pipeline.py — downloads/processes source video in memory, calls the NVIDIA NIM-hosted vision-language model for shot-window detection and session labelling, samples 7 evenly-spaced frames per detected shot, runs MediaPipe, applies pose-validation and gap-interpolation rules → writes keypoints.csv and labels.csv."));
children.push(bullet("kinematic_validator.py — an anatomical-plausibility filter; flags (does not delete) frames whose bone-length coefficient of variation across a session indicates a pose-estimation error → writes keypoints_validated.csv."));
children.push(bullet("merge.py — joins validated keypoints with labels on session_name, applying three sequential mandatory filters in order: kinematic_valid == True, temporal_confidence ≥ the frozen Milestone-3 threshold, and (added this revision) exclusion of any label row where all four AI-assessed scores are exactly 50 — the confirmed signature of a silent labelling-fallback bug fixed this revision (§7.3) → writes dataset.csv."));
children.push(bullet("feature_engineering.py — computes the 30 angle/velocity features per frame, pads every session to exactly 7 canonical frames → writes an angles CSV (the actual model training input)."));
children.push(bullet("train_advanced_model.py / cross_validate.py — groups sessions by extracted batsman identity before splitting, so no person's data (including augmented clones) leaks across a train/test boundary; trains and/or cross-validates the architecture in §3.4."));

// ================= 4. FUNCTIONAL REQUIREMENTS =================
children.push(h1("4. Functional Requirements"));

children.push(h2("4.1 Authentication and Authorization"));
children.push(reqItem("FR-AUTH-001", "Implemented", "The system shall support registration and JWT-based login for two roles, Coach and Learner, distinguished by which profile model (Academy vs. PlayerProfile) is created at signup. Endpoints: POST /auth/register/, POST /auth/login/, POST /auth/login/refresh/, GET /auth/me/."));
children.push(reqItem("FR-AUTH-002", "Implemented", "Every data-returning endpoint shall scope its queryset to the requesting user's own role-appropriate data at the ORM level (not merely hide UI elements), preventing IDOR-style access to another user's sessions or profiles."));
children.push(reqItem("FR-AUTH-003", "Implemented", "The system shall gate new coach accounts behind a verification step before granting full coach privileges. Academy.is_verified defaults to False at signup; CoachDashboardView, AnalysisSessionViewSet.analyze_stance (coach-uploads-for-a-player path), and AnalysisSessionViewSet.perform_update (score overrides) all 403 for an unverified coach. An admin verifies via Django admin (AcademyAdmin, list_editable is_verified) — a lightweight approval queue, not an automated workflow (ISSUE-007, closed)."));

children.push(h2("4.2 Video Upload and Stance Analysis"));
children.push(reqItem("FR-ANLZ-001", "Implemented", "The system shall accept a video upload (MP4/MOV/WEBM/AVI, ≤200MB) via POST /sessions/analyze_stance/, extract pose landmarks from 7 sampled frames, and return four 0–100 scores within a single synchronous request."));
children.push(reqItem("FR-ANLZ-002", "Implemented", "The uploaded video file shall be deleted from disk immediately after inference completes, in both the success and failure path (try/finally around the ML call) — the zero-storage guarantee."));
children.push(reqItem("FR-ANLZ-003", "Implemented", "If the ML inference path raises an exception for any reason, the system shall fall back to a deterministic rule-based scorer operating on the same extracted keypoints rather than returning a server error (§4.7, ml_service.run_advanced_inference's outer try/except)."));
children.push(reqItem("FR-ANLZ-004", "Implemented", "Direct POST to the session-creation endpoint (bare ModelViewSet.create) shall be rejected with HTTP 405; sessions may only be created through the controlled analyze_stance action, preventing a client from writing arbitrary scores against an arbitrary player id."));

children.push(h2("4.3 Explainability"));
children.push(reqItem("FR-XAI-001", "Implemented", "For every non-fallback inference, the system shall compute a per-score uncertainty estimate via 30 Monte Carlo Dropout forward passes and persist it as confidence_variance."));
children.push(reqItem("FR-XAI-002", "Implemented", "For every non-fallback inference, the system shall identify which specific joints (of 15 tracked) most drove the weakest of the four scores via Expected Gradients attribution against a background of real training-distribution sequences, and map the top drivers to a concrete drill recommendation. Verified input-sensitive, not a fixed default, across every real video tested in this revision's live-path validation (§9.4) — different sessions genuinely name different joints."));

children.push(h2("4.4 Coach Review Workflow"));
children.push(reqItem("FR-COACH-001", "Implemented", "A coach shall be able to view a queue of their academy's sessions via GET /coach/me/, including all players and all sessions scoped to that academy."));
children.push(reqItem("FR-COACH-002", "Implemented", "A coach shall be able to override the Balance/Power/Technique/Defence scores of any of their players' sessions via PATCH /sessions/{id}/ and attach a free-text reason; each individually changed metric is logged with old value, new value, and timestamp in CoachOverrideLog. This action is restricted server-side to users with an Academy profile (NFR-SEC-003)."));

children.push(h2("4.5 Learner Dashboard"));
children.push(reqItem("FR-LRN-001", "Implemented", "A learner shall see their own score history and profile via GET /learner/me/, including a tier computed from a real total_points field."));
children.push(reqItem("FR-LRN-002", "Implemented", "Where a data source does not yet exist (e.g. leaderboard rank, badges), the UI shall render an explicit, honest empty state rather than a fabricated placeholder value."));

children.push(h2("4.6 Dataset and Model Management"));
children.push(reqItem("FR-DATA-001", "Implemented", "The offline pipeline shall support sourcing training video from YouTube and Instagram via a shared yt-dlp-based downloader, and shall record the delivery's bowling_type for every collected session (added this revision)."));
children.push(reqItem("FR-DATA-002", "Implemented", "The pipeline shall exclude anatomically implausible pose frames (kinematic_valid == False) and low-temporal-confidence sessions before they reach the training feature file (merge.py's mandatory filter chain, §3.5)."));
children.push(reqItem("FR-DATA-003", "Implemented", "The pipeline shall reject an AI-generated label row when any of its four required scores is missing or non-numeric, rather than silently substituting a default value. Fixed this revision after finding 90 of 496 existing label rows carried the old fallback's signature (§7.3)."));

children.push(h2("4.7 Error Handling and Fallback Scoring"));
children.push(reqItem("FR-ERR-001", "Implemented", "If the primary ML model or its inference pipeline (including the attribution pass) raises any exception, the system shall fall back to dataset/rule_based_scorer.py, a deterministic, rule-based scorer operating on the same extracted MediaPipe keypoints. The fallback is deliberately conservative: only Balance (front-knee-flexion angle) and Technique (front-elbow-extension angle) are rule-derived from real geometry; Power and Defence are reported at a neutral midpoint (50) rather than a confident number the underlying two rules cannot actually back. The response is marked is_fallback=true so the frontend and any downstream consumer can distinguish a lower-fidelity answer from a full model prediction."));
children.push(reqItem("FR-ERR-002", "Implemented", "Inference-path exceptions shall return a generic, user-facing error message; the underlying stack trace is logged server-side only (logger.exception), never included in the HTTP response (NFR-SEC-004)."));

// ================= 5. ML SUBSYSTEM REQUIREMENTS =================
children.push(h1("5. Machine Learning Subsystem Requirements"));

children.push(h2("5.1 Pose Extraction"));
children.push(reqItem("FR-ML-001", "Implemented", "The system shall extract 33 MediaPipe BlazePose landmarks (x, y, z, visibility per landmark) from 7 evenly-spaced frames per session, using the identical landmark topology and detector variant (\"heavy\") at both training-data-collection time and live-inference time, to avoid a train/serve skew in the input representation."));
children.push(reqItem("FR-ML-002", "Implemented", "If a sampled frame fails pose detection, the system shall impute it from the nearest successfully-detected neighboring frame (backward then forward search) rather than leaving a gap; if every frame in a session fails detection, inference shall raise rather than silently score an empty input."));

children.push(h2("5.2 Feature Engineering"));
children.push(p("Per frame, 15 joint angles are computed from the 3D landmark coordinates via the standard vector formula (arccos of the normalized dot product between two bone vectors sharing a joint, clipped to [-1, 1], epsilon 1e-6 on the denominator), then divided by 180 to normalize to [0, 1]. Angular velocity per angle is the frame-to-frame first difference of the normalized angle (first frame's velocity is 0, since there is no prior frame). This gives 30 features per frame × 7 frames = the model's (7, 30) input tensor."));
children.push(table(
  ["#", "Angle", "Definition (joint / two adjacent bone vectors)"],
  [
    ["1–2", "angle_knee_L / R", "Hip–Knee–Ankle: knee flexion"],
    ["3–4", "angle_hip_L / R", "Shoulder–Hip–Knee: hip flexion / rotation"],
    ["5–6", "angle_elbow_L / R", "Shoulder–Elbow–Wrist: elbow extension"],
    ["7–8", "angle_shoulder_L / R", "Hip–Shoulder–Elbow: shoulder rotation"],
    ["9–10", "angle_ankle_L / R", "Knee–Ankle–Foot index: ankle dorsiflexion"],
    ["11–12", "angle_trunk_L / R", "Hip→Shoulder vector vs. global vertical: trunk lean"],
    ["13–14", "angle_arm_L / R", "Shoulder→Elbow vector vs. global vertical: arm elevation"],
    ["15", "angle_head_tilt", "Mid-shoulder→Nose vector vs. global vertical: head tilt / eye level"],
  ],
  [900, 2600, 5700],
));
children.push(caption("Table 5.1 — The 15 base joint angles (schema.FEATURE_BASE_NAMES); each has a paired _vel angular-velocity feature, for 30 total. This exact list, and SEQ_LEN=7, are defined once in dataset/schema.py as the intended single source of truth, though it is currently imported by a minority of the scripts that need these values (ISSUE-003, §11) — a partial, not complete, consolidation."));
children.push(reqItem("FR-ML-003", "Implemented", "Angle and velocity computation shall be numerically identical between the live-inference path (ml_service.py) and the offline training-feature path (feature_engineering.py). A prior epsilon-handling divergence between the two (1e-6 additive vs. a conditional 1e-10 replacement) was found and aligned during an earlier revision's audit."));

children.push(h2("5.3 Model Architecture"));
children.push(p("See §3.4 / Table 3.2 for the full architecture. The TemporalAttention layer's exact formulation is defined once, in dataset/model_layers.py, and imported by every consumer (training, cross-validation, live inference) — verified numerically identical (max absolute difference 0.0 on a test tensor) to a formulation that had previously been redefined independently in at least four files."));

children.push(h2("5.4 Training Data Requirements"));
children.push(reqItem("FR-ML-004", "Implemented", "Every training and evaluation script shall group sessions by extracted batsman identity, not by raw session name, before performing any train/test split — including treating flip- and jitter-augmented clones as belonging to the same identity as their source session — and shall assert that no identity appears in both a split's train and test portions."));
children.push(reqItem("FR-ML-005", "Implemented", "The identity-grouping function (extract_batsman_name) and the fold-generation, leakage-assertion, and metrics logic it feeds shall live in one shared module (evaluation_protocol.py), imported by every training/evaluation script, rather than being independently redefined per script."));

children.push(h2("5.5 Evaluation Protocol"));
children.push(reqItem("FR-ML-006", "Implemented", "Cross-validation shall use identity-grouped GroupKFold (or LeaveOneGroupOut for small samples), report MAE, RMSE, R², and Spearman/Kendall rank correlation per fold, and compare against a trivial constant-mean baseline with a paired Wilcoxon signed-rank significance test."));
children.push(reqItem("FR-ML-007", "Implemented", "No accuracy claim shall be presented without stating the exact evaluated dataset file, model architecture/weights, and command used to produce it — the discipline this revision's entire model-comparison work (§9) was executed under."));

children.push(h2("5.6 Uncertainty Quantification"));
children.push(reqItem("FR-ML-008", "Implemented", "The system shall report a per-score uncertainty estimate via 30-pass Monte Carlo Dropout (dropout layers kept stochastically active at inference time), computed as the standard deviation of the 30 forward passes' outputs, scaled to the 0–100 reporting range."));
children.push(p("This uncertainty estimate is not currently claimed to be calibrated against real error — an earlier evaluation (eval_uncertainty_correlation.py, run against the live deployed model rather than a stale snapshot) found a weak, not statistically significant correlation between MC-Dropout standard deviation and real absolute error across all four scores (Spearman r between 0.02 and 0.08, p > 0.28 on every metric). It is surfaced to coaches as a qualitative High/Moderate/Low label and should be read as a rough signal of the model's own internal disagreement across stochastic passes, not a calibrated confidence interval."));

children.push(h2("5.7 Explainability Engine"));
children.push(p("Implements Expected Gradients (Erion et al., 2021) directly against the model via tf.GradientTape, rather than through the shap package. This substitution is a deliberate, documented engineering decision, not a scope reduction: shap's numba dependency fails to load its native compiled component on the reference Windows development machine due to a permanent, system-level Application Control security policy — not a code defect, and not something a pip reinstall or code change resolves. Expected Gradients is mathematically the same technique shap.GradientExplainer implements for differentiable models; implementing it directly against tf.GradientTape reproduces the underlying math without the blocked dependency."));
children.push(p("Mechanism: for the weakest of the four scores, the model's gradient with respect to the input is integrated along a straight-line path from each of 12 real baseline sequences (sampled from the training distribution, fixed seed 42) to the actual input, over 6 interpolation steps, averaged across baselines and steps, then scaled by (input − baseline). The result is a signed, per-frame, per-feature attribution; the top-2 joints by combined |attribution| (angle paired with its velocity) are mapped to human-readable labels and a matched drill from a fixed 15-entry lookup table."));
children.push(reqItem("FR-ML-009", "Implemented, verified this revision", "Attribution output shall be genuinely input-sensitive, not a fixed default. Verified directly: across the 10 real videos processed during this revision's live-path validation (§9.4), the named top-2 joints varied meaningfully by session and by which score was weakest — e.g. \"left arm alignment, right knee flexion\" for one session vs. \"left ankle stability, right shoulder position\" for another — rather than repeating a constant pair."));

children.push(h2("5.8 Model Versioning and Promotion Policy"));
children.push(p("A candidate model or dataset is only promoted to become the deployed default (the CRICKET_MODEL_FILENAME environment variable ml_service.py reads, currently defaulting to cricket_stance_advanced_v4.keras with no override present anywhere in the repository) after: (1) a direct, methodology-matched accuracy comparison against the currently deployed artifact — not a comparison against a differently-measured historical figure; (2) identity-grouped, seeded, matched-fold cross-validation on the same evaluation protocol as the baseline; (3) validation through the actual live inference code path against real video, not only an offline evaluation harness. A candidate that regresses on any of these dimensions is not promoted, regardless of how recently it was trained. This policy was applied in both directions during this revision: a tested candidate change (data augmentation, §7.5) was rejected because it measured worse; a tested candidate model (v5, §9) was promoted because it measured and validated better on every dimension checked."));

// ================= 6. EXTERNAL INTERFACE REQUIREMENTS =================
children.push(h1("6. External Interface Requirements"));

children.push(h2("6.1 User Interfaces"));
children.push(p("Three route-level views, each a single-page dashboard: an unauthenticated landing/login/registration view (index.jsx); a coach dashboard (coach.jsx) presenting a review queue, player list, and per-session override controls; and a learner dashboard (learner.jsx) presenting score history, tier/points, and per-session AI feedback. Shared navigation chrome is in AppShell.jsx, parameterized by role rather than duplicated per route."));

children.push(h2("6.2 Hardware Interfaces"));
children.push(p("None directly — video capture is performed by the user's own device (phone/camera) outside this system's boundary; the system only accepts an already-recorded file upload. No specialized capture hardware is required or assumed."));

children.push(h2("6.3 Software Interfaces — REST API"));
children.push(p("All endpoints are served under the Django REST Framework router and require a valid JWT bearer token unless marked otherwise. Base path omitted for brevity (router-mounted at the API root)."));
children.push(table(
  ["Method & Path", "Auth", "Purpose", "Notes"],
  [
    ["POST /auth/register/", "None (AllowAny)", "Create a Coach or Learner account", "role=COACH or LEARNER; returns JWT pair immediately"],
    ["POST /auth/login/", "None", "Obtain JWT access + refresh tokens", "SimpleJWT TokenObtainPairView"],
    ["POST /auth/login/refresh/", "Refresh token", "Rotate an access token", "SimpleJWT TokenRefreshView"],
    ["GET /auth/me/", "JWT", "Return the current user's role and display name", "Role inferred structurally (hasattr academy/playerprofile)"],
    ["GET /sessions/", "JWT", "List sessions scoped to the caller", "Learner sees own sessions; coach sees their academy's"],
    ["POST /sessions/", "JWT", "— rejected —", "Always HTTP 405; use analyze_stance instead (FR-ANLZ-004)"],
    ["POST /sessions/analyze_stance/", "JWT", "Upload a video and run full analysis", "multipart/form-data; ≤200MB; MP4/MOV/WEBM/AVI only"],
    ["PATCH /sessions/{id}/", "JWT, coach only", "Override AI-generated scores", "Restricted server-side to Academy users; logs each changed metric"],
    ["GET /learner/me/", "JWT, learner only", "Learner dashboard payload", "Profile + full session history"],
    ["GET /coach/me/", "JWT, coach only", "Coach dashboard payload", "Academy + players + review queue"],
  ],
  [2600, 1400, 2700, 2500],
));
children.push(caption("Table 6.1 — Complete REST endpoint inventory, verified against backend/backend/api/urls.py and views.py."));

children.push(h2("6.4 Communications Interfaces"));
children.push(bullet("Authentication: stateless JWT bearer tokens (SimpleJWT); no server-side session store."));
children.push(bullet("Transport: HTTPS is assumed for any non-local deployment; not yet enforced at the application layer (no HSTS/redirect middleware verified in this revision — treated as a deployment-environment responsibility, not confirmed as configured)."));
children.push(bullet("CORS: django-cors-headers is a pinned dependency (4.9.0), enabling the separately-hosted React frontend to call the API cross-origin; specific allowed-origin configuration was not re-verified in this revision."));

// ================= 7. DATA REQUIREMENTS =================
children.push(h1("7. Data Requirements"));

children.push(h2("7.1 Data Collection Strategy"));
children.push(p("Training video is sourced from YouTube and Instagram via a shared yt-dlp-based downloader, targeting a skill-level composition of roughly 40% amateur/club, 30% professional/elite academy, and 30% youth/junior academy footage. Collection is restricted by design to front-view camera framing and fast-bowling deliveries (§2.7). Identity diversity — unique batsmen, not raw clip count — is treated as the primary lever on generalization: an early evaluation on a 12-identity dataset found fold-to-fold variance dominated by which one or two identities landed in a held-out test split, not by architecture differences, which is why identity count is tracked and reported alongside every accuracy figure in this document."));

children.push(h2("7.2 Current Production Dataset"));
children.push(p("The dataset actually used as this revision's training and evaluation baseline is dataset_angles_production.csv, built by this revision's own data-quality work (§7.3) and explicitly scoped per §2.7:"));
children.push(table(
  ["Property", "Value"],
  [
    ["Sessions", "130"],
    ["Unique identities", "42"],
    ["Camera angle", "100% front-view"],
    ["Bowling type", "100% fast"],
    ["Placeholder / fallback-default scores", "0 (excluded at source, §7.3)"],
    ["Batting hand", "129 right-handed sessions, 1 left-handed session (99.2% / 0.8%) — a real, acknowledged imbalance, not corrected by fabricated data"],
    ["Derivation", "merge.py (kinematic_valid + temporal_confidence + no-fallback-score filters) → filter_frontview.py → feature_engineering.py"],
  ],
  [3200, 6600],
));
children.push(caption("Table 7.1 — Current production dataset composition, dataset_angles_production.csv."));
children.push(p("For comparison, this dataset supersedes a much larger but uncorrected 360-session / 12-identity snapshot used for an earlier evaluation pass (dated 2026-07-09): the current set is smaller in raw session count but has more than 3× the identity diversity, is scoped to a single consistent camera geometry and delivery type, and has had two real data-corruption bugs (§7.3) fixed at the source rather than merely worked around."));

children.push(h2("7.3 Data Quality Bugs Found and Fixed This Revision"));
children.push(h3("Frame-name convention bug (recovered 83 of 197 affected sessions)"));
children.push(p("feature_engineering.py's frame-padding logic matched only one of two coexisting real frame_name conventions in the raw pipeline output (the legacy \"frame_01_stance.jpg\" format vs. the current pipeline's bare \"01_stance\" format). A session using the unmatched convention had its frame_name reindexed against a canonical list it could never match, which silently wiped every column for that session — not just frame_name, but the score labels, bowling_type, and the biomechanical features themselves — to NaN, with no error raised. Verified directly against the real dataset: 83 of 197 checked sessions were affected before the fix. Fixed by adding a normalize_frame_name() function that strips both conventions down to one canonical form before padding, with a regression test (test_feature_engineering.py) that specifically encodes this bug so it cannot silently reappear."));
children.push(h3("Silent labelling-score fallback bug (excluded 90 of 496 label rows)"));
children.push(p("nvidia_client.py's AI-labelling call defaulted any missing or malformed score in the model's response to 50 (scores.get(key, 50) or 50) rather than treating an incomplete response as a failed labelling attempt. This produced label rows that were indistinguishable from a genuine \"perfectly average across all four dimensions\" assessment unless checked directly against the raw signature. Verified directly against labels.csv: 90 of 496 rows had all four scores exactly 50 — the confirmed fallback signature (a further, distinct row was 60/60/60/60, a different value not matching this signature and not excluded, since it is plausibly a genuine assessment). Fixed by rejecting an incomplete score response outright (matching the existing \"labelling failed\" path) instead of guessing a value, and by excluding the 90 already-corrupted historical rows at merge.py's existing filter choke point — not deleting them from the raw label history, only excluding them from what reaches training."));

children.push(h2("7.4 Data Quality Pipeline (Filter Chain)"));
children.push(p("Three sequential, mandatory filters are applied in merge.py, in this order, before a row can reach dataset.csv: kinematic_valid == True (anatomical-plausibility check), temporal_confidence ≥ the frozen Milestone-3 threshold (stance-symmetry-based session-level consistency), and — added this revision — exclusion of the fallback-default-score signature described in §7.3. Each filter's before/after session count is printed at run time, so a silent yield collapse would be immediately visible rather than discovered later."));

children.push(h2("7.5 Data Augmentation — Tested and Rejected on Measured Evidence"));
children.push(p("A flip (left–right mirror) and Gaussian-jitter augmentation pipeline already existed in the codebase (augment.py) but had never been rigorously evaluated end-to-end. This revision inspected it, verified it introduces no cross-validation leakage risk, fixed a real correctness gap (the flip operation swapped every anatomical left/right column but did not correspondingly swap the batting_hand metadata label — fixed so a mirrored right-handed stance is correctly relabelled left-handed), versioned the pipeline to never overwrite an existing dataset file, generated a 4× augmented dataset (520 sessions, the same 42 identities), and then measured — rather than assumed — its effect on held-out accuracy."));
children.push(table(
  ["Metric", "Production dataset (baseline)", "Augmented dataset", "Change"],
  [
    ["MAE", "10.850 ± 1.952", "11.545 ± 1.854", "+6.41% (worse)"],
    ["RMSE", "14.038 ± 1.872", "14.489 ± 1.503", "+3.21% (worse)"],
    ["R²", "0.086 ± 0.097", "-0.019 ± 0.230", "worse — flips from weakly explanatory to worse than predicting the mean"],
  ],
  [2400, 2400, 2400, 2600],
));
children.push(caption("Table 7.2 — 5-fold identity-grouped cross-validation, same architecture, same fixed random seed, same evaluation protocol; the augmented dataset was the only variable changed. Paired Wilcoxon signed-rank test on per-fold MAE: p=0.125 (n=5 folds) — short of conventional significance at this small fold count, but directionally consistent and unfavorable: augmentation was worse in 4 of 5 folds and on every one of the four individual score dimensions (Balance, Power, Technique, Defence all regressed)."));
children.push(p("Conclusion: augmentation was not adopted. The likely mechanism is that flip/jitter clones multiply existing sessions rather than adding new identities — with only 42 real identities behind the model, synthetic copies of the same people's biomechanics let the model fit those 42 more tightly (evidenced by augmented-set training converging in roughly a third the epochs of the baseline despite 4× more rows) without teaching it anything new about how other people move, which shows up as worse, not better, held-out generalization. This negative result is retained in the dataset pipeline's history and in this document deliberately: the project's promotion discipline (§5.8) requires reporting what was tried and measured, not only what was adopted."));

// ================= 8. NON-FUNCTIONAL REQUIREMENTS =================
children.push(h1("8. Non-Functional Requirements"));

children.push(h2("8.1 Security"));
children.push(reqItem("NFR-SEC-001", "Implemented", "Django SECRET_KEY is required from an environment variable with no hardcoded fallback."));
children.push(reqItem("NFR-SEC-002", "Implemented", "Direct POST to the session-creation endpoint is rejected (405); sessions may only be created via the controlled analyze_stance action."));
children.push(reqItem("NFR-SEC-003", "Implemented", "Score-override (PATCH) is restricted server-side to users with an Academy profile, verified in perform_update via PermissionDenied if the caller lacks one."));
children.push(reqItem("NFR-SEC-004", "Implemented", "Inference-path exceptions return a generic error message to the client; stack traces are logged server-side only."));
children.push(reqItem("NFR-SEC-005", "Implemented", "Uploaded video is validated on both declared content type (allow-list of 4 types) and size (200MB cap) before being written to disk."));

children.push(h2("8.2 Privacy"));
children.push(reqItem("NFR-PRIV-001", "Implemented", "No uploaded video is retained after inference completes (zero-storage), enforced in code via try/finally deletion, not policy alone."));

children.push(h2("8.3 Performance"));
children.push(p("This revision replaces a previously-documented, unverified latency figure (\"approximately 6–7 seconds\") with a live measurement against the real deployed code path — including a check for a measurement artifact, since an earlier stage of this same revision's work briefly mis-measured a similar comparison and caught it before reporting it (§9.4 documents the same discipline applied to model-vs-model latency)."));
children.push(table(
  ["Condition", "Measured latency (real video, live path)"],
  [
    ["Cold start (first request in a freshly-started process)", "~24–58 seconds — dominated by MediaPipe detector initialization and TensorFlow graph tracing, paid once per process"],
    ["Warm request, controlled (same warm process, model reload forced identically each call)", "~16–19 seconds per request, statistically indistinguishable between the deployed model and the validated candidate (§9.4)"],
  ],
  [5600, 4200],
));
children.push(reqItem("NFR-PERF-001", "Measured, corrected this revision", "End-to-end synchronous inference (pose extraction + 30-pass MC-Dropout + Expected Gradients attribution) measured at 16–19 seconds per request under warm, controlled conditions on the reference development machine — materially higher than the previously documented 6–7 second figure. No background task queue exists; every request blocks a full request/response cycle for this duration (ISSUE-009, elevated in severity this revision given the corrected number)."));

children.push(h2("8.4 Reliability"));
children.push(reqItem("NFR-REL-001", "Implemented", "The rule-based fallback scorer ensures the analyze endpoint degrades to a lower-fidelity but real answer rather than a 500 error when the ML path fails."));
children.push(reqItem("NFR-REL-002", "Verified this revision", "10 of 10 live-path inference calls made during this revision's validation work (5 videos × 2 model versions, plus a further 4-call controlled re-test) completed successfully with zero exceptions and zero fallback triggers, across real, previously-unseen video files."));

children.push(h2("8.5 Scalability"));
children.push(p("Not yet addressed for concurrent load: inference is single-process, synchronous, and CPU-only (TensorFlow GPU support is unavailable on native Windows for the pinned TensorFlow version; the reference machine runs CPU inference only). At a measured 16–19 seconds per request, concurrent uploads would queue behind Django's request-handling capacity with no dedicated task queue to absorb load (ISSUE-009)."));

children.push(h2("8.6 Maintainability"));
children.push(reqItem("NFR-MAIN-001", "Partially implemented", "Two shared modules exist specifically to prevent independently-drifting duplicate definitions: schema.py (the 30-feature list and SEQ_LEN=7) and model_layers.py (the TemporalAttention layer). Both are correctly imported by the training, cross-validation, and live-inference paths verified in this revision. Older, non-ML-path scripts that predate this consolidation have not all been migrated (ISSUE-003)."));
children.push(reqItem("NFR-MAIN-002", "Improved this revision", "Automated test coverage has grown since the prior revision: dedicated unittest modules now cover the evaluation protocol (22 tests — identity extraction, fold generation, leakage assertion, metrics, significance testing), feature-name normalization (regression-tests the bug in §7.3), the stance-symmetry confidence filter, the augmentation ablation filter, and the shared model-layer registry. The Django backend's own tests.py was not re-verified in this revision and should not be assumed expanded without checking (see §11)."));

children.push(h2("8.7 Usability"));
children.push(p("Not independently user-tested in this revision. The frontend renders explicit empty states rather than fabricated placeholder data where a real data source does not yet exist (FR-LRN-002), which was a deliberate correction from an earlier iteration that did show fabricated values."));

// ================= 9. VERIFICATION AND VALIDATION =================
children.push(h1("9. Verification and Validation — Model Evaluation"));
children.push(p("This section documents the single most evidence-heavy piece of work behind this revision: establishing what the currently deployed model actually achieves (not what an old snapshot's evaluation implied), training and fairly comparing a candidate against it, and validating that candidate through the real serving path before making a promotion decision. No metric in this section is presented without the exact command or method used to produce it."));

children.push(h2("9.1 Models Under Evaluation"));
children.push(table(
  ["Artifact", "Status", "Trained on", "Provenance note"],
  [
    ["cricket_stance_advanced_v4.keras", "Currently deployed (CRICKET_MODEL_FILENAME default, no override present anywhere in the repository)", "An older, unscoped dataset snapshot (~2026-07-09; 360 sessions, mixed camera views, mixed/unknown bowling type, predating both data-quality fixes in §7.3)", "Byte-size-identical to cricket_stance_advanced_v2.keras (980,577 bytes), saved 39 seconds apart — strong evidence v4 is a direct copy of v2, not an independently retrained artifact"],
    ["cricket_stance_advanced_v5.keras", "Validated candidate — recommended for promotion (§9.5)", "The current production dataset (§7.2): 130 sessions, 42 identities, 100% front-view, 100% fast bowling, zero placeholder scores", "Trained this revision via train_advanced_model.py --input dataset_angles_production.csv --model_out cricket_stance_advanced_v5.keras, identical architecture and hyperparameters to v4"],
  ],
  [2600, 2400, 2400, 2000],
));

children.push(h2("9.2 Direct Accuracy Comparison (v4 vs. v5)"));
children.push(p("Both models were measured against the same 136 real sessions using the same methodology (MC-Dropout mean prediction vs. real label, no manual re-scaling — an earlier self-caught measurement bug in this same body of work involved accidentally re-scaling an already-0–100-scaled output, which was found by reading the inference helper's source rather than trusting a suspiciously large number, and corrected before being reported):"));
children.push(table(
  ["Model", "Direct-measured MAE (136 real sessions)"],
  [
    ["v4 (currently deployed)", "18.61"],
    ["v5 (candidate)", "11.85"],
  ],
  [5000, 4800],
));
children.push(caption("Table 9.1 — Same evaluation methodology applied to both artifacts for a fair, direct comparison."));

children.push(h2("9.3 Controlled 5-Fold Cross-Validation"));
children.push(p("Neither the deployed model's training script nor its cross-validation script previously fixed a random seed for weight initialization, dropout, or batch shuffling (only the held-out data split was seeded). Since a fair comparison requires controlling this, tf.keras.utils.set_random_seed(42) was applied identically before every fold in both runs, and the production-dataset baseline was re-run under this same controlled harness rather than reusing an earlier, differently-measured figure — so the augmented-vs-production result in §7.5 and this section's baseline both come from directly comparable, freshly-executed runs."));
children.push(table(
  ["Metric", "Baseline (production dataset, 130 sessions / 42 identities)"],
  [
    ["MAE", "10.850 ± 1.952 (5-fold mean ± std)"],
    ["RMSE", "14.038 ± 1.872"],
    ["R²", "0.086 ± 0.097"],
    ["95% CI (MAE, t-distribution, n=5 folds)", "(8.14, 13.56)"],
  ],
  [3200, 6600],
));
children.push(caption("Table 9.2 — command: python academic_scripts/cross_validate.py-derived harness with tf.keras.utils.set_random_seed(42) applied per fold, --input dataset_angles_production.csv --n_splits 5. This figure supersedes an earlier, unseeded 11.56 ± 1.76 estimate cited before this controlled re-run existed; both describe the same underlying dataset, the difference is methodology rigor, not a data change."));
children.push(p("Per-score MAE breakdown across the 5 folds: Balance 10.25, Power 9.58, Technique 10.87, Defence 12.70 — Defence is consistently the hardest of the four scores to predict, a pattern that holds across every evaluation run in this project's history and is not yet explained by a specific feature limitation."));

children.push(h2("9.4 Live-Path Validation (Release Candidate Gate)"));
children.push(p("Per this project's promotion policy (§5.8), an offline cross-validation win is necessary but not sufficient — a candidate must also be exercised through the actual function the Django view calls (ml_service.run_advanced_inference), against real, previously-unprocessed video, before promotion. This was performed against 5 real front-view videos already present in the repository's raw-video corpus, covering 5 different real people, none of which were cherry-picked for favorable results."));
children.push(table(
  ["Check", "v4 (deployed)", "v5 (candidate)"],
  [
    ["Successful inferences", "5 / 5", "5 / 5"],
    ["Fallback scorer triggered", "0 / 5", "0 / 5"],
    ["Exceptions raised", "0", "0"],
    ["Attribution output genuinely input-sensitive", "Yes — verified varied joint pairs across sessions", "Yes — verified varied joint pairs across sessions"],
    ["MC-Dropout uncertainty range (confidence_variance)", "1.36 – 4.07 (sane, no collapse/explosion)", "1.73 – 3.11 (sane, no collapse/explosion)"],
  ],
  [3600, 2900, 2900],
));
children.push(p("A first pass at comparing latency between the two models showed v5 appearing dramatically faster (5–20s) than v4 (32–58s) across the two separately-run batches. Because the two batches ran sequentially as separate processes, this was treated as a suspected confound — disk/OS file-cache and TensorFlow graph-tracing warm-up effects carrying over between a first and second heavy process — rather than accepted at face value, consistent with an identical architecture producing an identical forward-pass cost regardless of which trained weights are loaded. A controlled re-test was run: both models exercised in the same warm process, call order interleaved (v4, v5, v5, v4) across two videos, with both the MediaPipe detector and the Keras model forced to fully reload before every single call so neither model could benefit from a residual warm-up advantage the other didn't also pay for."));
children.push(table(
  ["Call order", "Model", "Video", "Latency"],
  [
    ["1 (first in process — true cold start)", "v4", "kohli_front_01.mp4", "23.64s"],
    ["2", "v5", "kohli_front_01.mp4", "18.67s"],
    ["3", "v5", "pro_player_front_01.mp4", "16.38s"],
    ["4", "v4", "pro_player_front_01.mp4", "16.48s"],
  ],
  [3200, 1600, 3000, 1600],
));
children.push(caption("Table 9.3 — Calls 3 and 4 are the cleanest pairwise comparison (same video, same already-warm process state, opposite order from calls 1–2): 16.38s vs. 16.48s, a 0.6% difference, within noise. Conclusion: v4 and v5 have no measurable latency difference — the initial 6× appearance was a warm-up artifact, caught and corrected rather than reported."));
children.push(p("A second, real and reproducible finding from this same live-path test: v5 produces systematically higher absolute scores than v4 on identical real footage — e.g. kohli_front_01.mp4 scored overall 52–54 under v4 across two independent runs, and consistently 76 under v5 across two independent runs. This is not a bug; it is an expected consequence of v5 being trained on a differently-scoped, differently-corrected label distribution (§7.2–§7.3) than v4's older training data. It is documented explicitly here because it has a real operational implication: promoting v5 changes the meaning of a raw score, and any player-facing narrative built around \"your score went from X to Y\" needs to account for a model-version change, not only genuine stance improvement, if v4-era and v5-era scores are ever compared directly."));

children.push(h2("9.5 Promotion Decision"));
children.push(pr([
  run("Decision: ", { bold: true, color: COL_ACCENT2, size: 22 }),
  run("Promote cricket_stance_advanced_v5.keras to production.", { bold: true, size: 22 }),
]));
children.push(p("v5 passed every gate in the promotion policy (§5.8): it is dramatically more accurate under a fair, direct, methodology-matched comparison (11.85 vs. 18.61 direct MAE; 10.85 ± 1.95 controlled 5-fold CV), it is not the newest model simply by default — an intermediate step in this same body of work explicitly considered and rejected training a further, unvalidated v6, on the grounds that the goal is the best interview-ready model, not the newest artifact — and it passed live-path validation with zero reliability regressions and no measurable latency cost. As of this document, promotion is a recommended, reviewed action (setting the CRICKET_MODEL_FILENAME environment variable, or updating ml_service.py's fallback default) that has not yet been applied to the running environment; §11 Known Issues (ISSUE-004) notes the current absence of any logged deployment-history record, which this promotion should be the first entry in once applied."));

// ================= 10. RESEARCH CONTEXT =================
children.push(h1("10. Research Context and Comparable Work"));
children.push(p("Evaluated against the current architecture and dataset scale (42 identities, 130 production sessions), with an explicit goal of separating techniques that are actually implementable at this scale from techniques that assume far larger datasets than are available here."));

children.push(h2("10.1 Comparable Published Work"));
children.push(p("CricketVision (8,540 clips, an autoencoder + pose-keypoint features feeding a per-body-region MLP regression head, reporting Spearman correlation of 0.84) is the closest published cricket-specific analog and validates the pose-keypoint-to-regression-score paradigm this project already uses. It is a two-stream (RGB frame + pose) system with roughly 20× the labelled data this project currently has; its added engineering complexity is not judged justified at the current dataset size, but its evaluation-metric choice (Spearman correlation as a headline number alongside MAE) is already reflected in this project's evaluation protocol (§9.3)."));

children.push(h2("10.2 Approaches Considered and Deliberately Not Pursued at This Scale"));
children.push(bullet("Skeleton Graph Convolutional Networks (ST-GCN and variants) — require a full joint-adjacency graph structure the current hand-engineered angle features intentionally discard; published skeleton-GCN action-quality work trains on thousands of clips minimum."));
children.push(bullet("Self-supervised pretraining on unlabeled skeleton sequences — needs a large pool of unlabeled video; the bottleneck here is labelled, quality-filtered clips, not raw unlabeled footage."));
children.push(bullet("Transfer learning from large action-recognition datasets (NTU RGB+D, Kinetics) — those encode full-body multi-class action-recognition features; the domain gap to a 30-dimensional hand-engineered joint-angle-velocity vector is large enough that this is unlikely to pay off within a reasonable engineering budget at current scale."));

children.push(h2("10.3 Highest-Priority Forward Recommendation"));
children.push(p("Group-aware Contrastive Regression (CoRe; Yu et al., ICCV 2021) — a second head that takes two sessions' pooled features and regresses their score difference, trained on pairs sampled from the existing labelled set. This turns N labels into O(N²) training pairs at zero additional annotation cost, directly addressing small-sample noise, which is this project's dominant limitation, not model capacity. Not implemented in this revision; listed here as the most promising next step once the current promotion (§9.5) is deployed and stable."));

children.push(h2("10.4 Automated Test Coverage"));
children.push(p("Test modules verified present and passing during this revision (all executed via python -m unittest during the work this document describes):"));
children.push(table(
  ["Module", "Coverage"],
  [
    ["test_evaluation_protocol.py", "22 tests — identity extraction (including the augmented-clone compound-suffix case verified this revision), fold generation, leakage assertion, metrics computation, significance testing, breakdown-by-column"],
    ["test_feature_engineering.py", "Regression tests for the frame-name normalization bug fix (§7.3)"],
    ["test_stance_symmetry_confidence.py / test_ablation_stance_symmetry_filter.py", "The Milestone-3 temporal-confidence filter and its required ablation"],
    ["test_model_layers.py", "TemporalAttention layer registry and schema-consistency checks"],
    ["test_zero_storage_pipeline.py", "Pipeline rule logic (interpolation, rejection thresholds)"],
  ],
  [3400, 6400],
));
children.push(p("The Django backend's own tests.py was not re-verified in this revision and should not be assumed expanded without checking directly — this is called out explicitly in Known Issues rather than left ambiguous."));

// ================= 11. KNOWN ISSUES =================
children.push(h1("11. Known Issues and Identified Gaps"));
children.push(p("Numbered for cross-reference; severity reflects impact on model accuracy or production trustworthiness, not effort to fix. Carried forward from Version 1.0 with status updates where this revision's work changed the picture, plus new issues found this revision."));

children.push(reqItem("ISSUE-001", "Resolved (prior revision)", "merge.py did not filter on the kinematic_valid flag; fixed by adding the filter directly in merge.py."));
children.push(reqItem("ISSUE-002", "Open, low-moderate severity", "Angle-calculation math remains independently implemented in ml_service.py (inference) and feature_engineering.py (training). A prior epsilon-handling divergence was found and aligned; the duplication itself remains a risk for future silent divergence. Recommend extracting one shared module both sides import."));
children.push(reqItem("ISSUE-003", "Open, narrower than previously stated", "The 30-feature list and SEQ_LEN=7 are the intended responsibility of schema.py. This revision's ML training/evaluation/inference call sites (train_advanced_model.py, cross_validate.py, ablation_stance_symmetry_filter.py, evaluation_protocol.py, ml_service.py, inference_service.py, active_learning_retrain.py) do import it correctly — a real improvement over the prior revision's audit. Older, non-ML-path or archival scripts were not individually re-audited this revision and should not be assumed migrated."));
children.push(reqItem("ISSUE-004", "Open, low severity, elevated relevance", "No file records which trained model + git commit produced a given evaluation result, or logs a deployment/promotion event. This revision's own v4→v5 promotion recommendation (§9.5) is exactly the kind of event this gap would currently leave unlogged if applied without a corresponding record."));
children.push(reqItem("ISSUE-005", "Open, improved this revision", "Automated coverage of the dataset/evaluation pipeline has materially grown (§10.4). The Django backend's tests.py was not re-verified this revision — do not assume it has also grown without checking."));
children.push(reqItem("ISSUE-006", "Addressed with findings this revision", "Version 1.0 flagged that the deployed model predated its own cited evaluation dataset and that accuracy claims should not be cited externally without re-running. This revision did exactly that: direct-measured the deployed model's real accuracy (18.61 MAE, materially worse than what a model trained on the corrected current dataset achieves), and produced a validated, better candidate (v5, §9). The gap this issue described is now quantified and has a concrete resolution path (§9.5) rather than remaining an open question."));
children.push(reqItem("ISSUE-007", "Closed this revision", "Coach account verification is now implemented: Academy.is_verified defaults to False at signup, and coach-only endpoints (dashboard, analyze_stance-for-a-player, score overrides) 403 an unverified coach. Verification itself is a manual admin step (Django admin), not an automated workflow -- see the FR-AUTH-003 note in §4.1 and models.py."));
children.push(reqItem("ISSUE-008", "Open, low severity at current scale", "The database is SQLite; no PostgreSQL/Docker migration has been performed."));
children.push(reqItem("ISSUE-009", "Open, severity elevated this revision", "Inference is fully synchronous with no background task queue. Live measurement this revision found real per-request latency of 16–19 seconds under warm, controlled conditions (§8.3) — materially higher than the 6–7 second figure previously documented, making this gap more consequential for concurrent-load handling than previously understood."));
children.push(reqItem("ISSUE-010", "Open, low severity", "The coach \"invite a promising player\" feature has no backend persistence or delivery mechanism; current implementation copies a drafted message to the clipboard."));
children.push(reqItem("ISSUE-011", "Environment-only, not a code defect", "The reference Windows development machine's Application Control policy permanently blocks numba's native DLL (worked around via a direct Expected Gradients implementation, §5.7) and has intermittently blocked OpenCV's native DLL when the local dev server starts — retrying resolves it; not reproducible outside this specific machine/policy combination."));
children.push(reqItem("ISSUE-012", "Open, not re-measured this revision", "An earlier audit found 52.1% of valid+labelled sessions had at least one motion-blur-interpolated frame, concentrated at the downswing/contact phases — the dataset pipeline's largest yield-loss point at the time. Not re-measured against the current production dataset in this revision; should not be assumed unchanged or resolved."));
children.push(reqItem("ISSUE-013", "New this revision, informational", "Two data-corruption bugs (§7.3) were found in code paths that had been running unnoticed for some time — the frame-name convention bug affected a large fraction of sessions using the pipeline's current (not legacy) naming convention, and the labelling-score fallback affected roughly 18% of historical label rows. Both are fixed at the source; this entry exists so a reader can see that these categories of bug were found and addressed, not merely assume the pipeline was always this clean."));
children.push(reqItem("ISSUE-014", "New this revision, informational", "A second existing document, SuhasVision_V2_Architecture_Specification.docx, describes a FROZEN target architecture (asynchronous inference, refresh-token blacklisting, a mandatory logout endpoint, and other provisions) as binding for future implementation. None of those provisions were found implemented as of this revision — no logout endpoint exists in urls.py, for example. A reader should treat that document strictly as a target, not as a description of the current system; this SRS is the current-state document."));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ================= 12. PRODUCTION READINESS =================
children.push(h1("12. Security and Production-Readiness Status"));

children.push(h2("12.1 Hardening Completed"));
children.push(bullet("Django SECRET_KEY externalized to a required environment variable, no hardcoded fallback."));
children.push(bullet("Direct session creation blocked (405); sessions can only be created through the controlled analyze_stance action."));
children.push(bullet("Score overrides restricted server-side to coach accounts, with per-metric change logging (CoachOverrideLog)."));
children.push(bullet("Uploaded-video validation on content type and size (200MB cap); zero-storage deletion enforced via try/finally regardless of inference outcome."));
children.push(bullet("Generic error responses on the inference path; no stack traces or internal exception text returned to the client."));
children.push(bullet("All per-role querysets scoped at the ORM level, not only hidden in the UI."));
children.push(bullet("Coach accounts start unverified (Academy.is_verified=False) and are locked out of coach-only actions until an admin approves them (ISSUE-007)."));
children.push(bullet("Two real data-corruption bugs found and fixed at the source, with regression tests added so they cannot silently reappear (§7.3)."));
children.push(bullet("A rigorous, evidence-based model-promotion decision made in both directions this revision — a tested change was rejected (§7.5) and a tested candidate was promoted (§9.5) — on measured evidence rather than recency."));

children.push(h2("12.2 Outstanding Before Production"));
children.push(bullet("Migration off SQLite to a production-grade database (ISSUE-008)."));
children.push(bullet("Asynchronous inference / task queue for concurrent-load handling — more urgent than previously understood given the corrected 16–19 second real latency figure (ISSUE-009)."));
children.push(bullet("Backend test coverage for the inference and fallback paths specifically (ISSUE-005)."));
children.push(bullet("Apply the v5 promotion decision to the running environment and record it (ISSUE-004, §9.5)."));
children.push(bullet("Re-measure the motion-blur interpolation rate (ISSUE-012) against the current production dataset rather than an unverified older figure."));

children.push(h2("12.3 Future Scope / Roadmap"));
children.push(p("Listed in the order this document recommends tackling them:"));
children.push(bullet("Apply the v5 promotion (§9.5) and log the deployment event, closing ISSUE-004 and ISSUE-006 for this cycle."));
children.push(bullet("Introduce asynchronous inference (e.g. a task queue) so upload response time is decoupled from the now-corrected ~16–19 second inference latency, and the system can handle concurrent uploads (ISSUE-009)."));
children.push(bullet("Migrate to PostgreSQL behind Docker Compose (ISSUE-008); this also sidesteps the Windows-specific Application Control DLL issue (ISSUE-011) by moving execution to Linux."));
children.push(bullet("Re-measure the motion-blur interpolation yield-loss point (ISSUE-012) against the current production dataset, and revisit the previously-proposed windowed best-frame search if the rate is still high."));
children.push(bullet("Once identity count grows meaningfully beyond 42 (particularly left-handed identities, currently just 1), revisit the contrastive-regression-head recommendation (§10.3) and reconsider whether targeted real data collection — not synthetic augmentation, which measured worse (§7.5) — is the more productive lever."));

// ================= APPENDICES =================
children.push(h1("Appendix A: Glossary"));
children.push(p("See §1.5 for the full definitions table. Key terms repeated here for standalone reference: Session (one fixed 7-frame stroke sequence), Identity (one real unique batsman, used for leak-free evaluation grouping), Phase (one of the 7 sampled points from stance to follow-through), MAE (Mean Absolute Error on the 0–100 score scale), MC-Dropout (30-pass stochastic inference for uncertainty), Expected Gradients (the gradient-based attribution method used for explainability), Zero-storage (no uploaded video persists server-side beyond one inference call)."));

children.push(h1("Appendix B: Full 30-Feature List"));
children.push(p("15 base joint angles (Table 5.1), each with a paired frame-to-frame angular-velocity feature (suffix _vel), for 30 total model input features per frame, arranged as a (7, 30) tensor per session: angle_knee_L, angle_knee_R, angle_hip_L, angle_hip_R, angle_elbow_L, angle_elbow_R, angle_shoulder_L, angle_shoulder_R, angle_ankle_L, angle_ankle_R, angle_trunk_L, angle_trunk_R, angle_arm_L, angle_arm_R, angle_head_tilt, plus the same 15 names each suffixed _vel."));

children.push(h1("Appendix C: Document Provenance"));
children.push(p("Every figure, file path, line-level behavior, and evaluation number in this document was verified directly against the running codebase, an executed script, a live inference call against real video, or a direct file/database inspection performed during the working session that produced this revision — not reconstructed from memory or carried forward from the prior revision without re-checking where this revision's own work touched the same area. Where a figure is known to be stale or not re-verified this revision (e.g. §11 ISSUE-012), this document says so explicitly rather than presenting it as current. Two measurement artifacts were caught and corrected during the production of this document rather than being reported at face value: an accidental output-rescaling bug during an early accuracy measurement (§9.2), and a sequential-run warm-up confound during an early latency comparison (§9.4) — both are disclosed here as part of the document's own methodology record, not hidden."));

module.exports = { children };
