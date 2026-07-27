const {
  fs, Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, BorderStyle, AlignmentType, PageBreak, PAGE_W, PAGE_H,
  FONT_BODY, FONT_HEAD, COL_ACCENT, COL_ACCENT2, COL_TEXT, COL_MUTE, COL_RULE,
  h1, h2, h3, p, pr, run, bullet, reqItem, hr, table, caption,
} = require("./gen_srs.js");
const { TableOfContents } = require("docx");

const children = [];

function mono(text) {
  return new Paragraph({
    spacing: { before: 80, after: 100, line: 230 },
    shading: { type: "clear", fill: "F2EFE7" },
    border: { top: { color: COL_RULE, size: 4, style: BorderStyle.SINGLE, space: 3 }, bottom: { color: COL_RULE, size: 4, style: BorderStyle.SINGLE, space: 3 }, left: { color: COL_RULE, size: 4, style: BorderStyle.SINGLE, space: 3 }, right: { color: COL_RULE, size: 4, style: BorderStyle.SINGLE, space: 3 } },
    children: text.split("\n").map((line, i) => new TextRun({ text: line, font: "Consolas", size: 16, color: COL_TEXT, break: i > 0 ? 1 : 0 })),
  });
}
function qa(question, answer) {
  return [
    new Paragraph({ spacing: { before: 100, after: 40 }, children: [new TextRun({ text: "Q: ", bold: true, color: COL_ACCENT2, font: FONT_BODY, size: 19 }), new TextRun({ text: question, bold: true, font: FONT_BODY, size: 19, color: COL_TEXT })] }),
    new Paragraph({ spacing: { after: 70, line: 246 }, alignment: AlignmentType.JUSTIFIED, children: [new TextRun({ text: "A: ", bold: true, color: COL_ACCENT, font: FONT_BODY, size: 19 }), new TextRun({ text: answer, font: FONT_BODY, size: 19, color: COL_TEXT })] }),
  ];
}
function sixQ(name, obj) {
  const rows = [
    ["What is it?", obj.what],
    ["Why is it needed?", obj.why],
    ["Why this implementation?", obj.chosen],
    ["Alternatives considered", obj.alt],
    ["Limitations", obj.limits],
    ["Redesign from scratch today?", obj.redesign],
  ];
  const out = [h3(name)];
  out.push(table(["Question", "Answer"], rows, [2600, 6600]));
  return out;
}

// ================= TITLE PAGE =================
children.push(
  new Paragraph({ spacing: { before: 2000 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "SuhasVision", bold: true, font: FONT_HEAD, size: 62, color: COL_ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 300 }, children: [new TextRun({ text: "Engineering Deep-Dive & Onboarding Manual", font: FONT_HEAD, size: 28, color: COL_TEXT })] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    border: { top: { color: COL_ACCENT2, size: 12, style: BorderStyle.SINGLE, space: 8 }, bottom: { color: COL_ACCENT2, size: 12, style: BorderStyle.SINGLE, space: 8 } },
    spacing: { before: 160, after: 160 },
    children: [new TextRun({ text: "HOW THE SYSTEM WORKS, WHY IT WAS BUILT THIS WAY, AND HOW TO EXTEND IT", bold: true, font: FONT_BODY, size: 20, color: COL_TEXT, characterSpacing: 10 })],
  }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 40 }, children: [new TextRun({ text: "2026-07-20", font: FONT_BODY, size: 20, color: COL_MUTE })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "Written to make you the primary engineer for this codebase, not a user of it.", italics: true, font: FONT_BODY, size: 18, color: COL_MUTE })] }),
  new Paragraph({ spacing: { before: 2000 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Prepared by Kurapati Sai Suhas", font: FONT_BODY, size: 20, color: COL_TEXT })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

children.push(h1("Table of Contents"));
children.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

children.push(p("A note on how to read this document, before the content starts: this repository asked for depth on 16 fronts — every function traced, every decision defended with six questions, every technique's paper cited — inside a 15-20 page budget. Those two asks are in real tension: literally tracing every function with a six-question analysis would run past 100 pages. The calibration made here is to go genuinely deep — equations, real numbers, real bugs, real trade-offs — on the decisions and components that carry the most engineering and interview weight, and to use dense tables (not padded prose) everywhere the content is naturally repetitive (the research-foundations list, the engineering-decision log, interview Q&A). Nothing here is invented to fill space: every number, file path, and function name was verified against the running code or the project's own dated engineering log (architecture_ground_truth.md) while this document was written.", { italic: true }));

// ================= 1. EXECUTIVE OVERVIEW =================
children.push(h1("1. Executive Overview"));

children.push(h2("1.1 The Problem"));
children.push(p("A batting coach's real job is mostly pattern recognition built from thousands of hours of watching players: they see a stance, and within seconds they know what's wrong with it. That expertise doesn't scale — a single coach can watch maybe a dozen players a day, closely. SuhasVision's premise is that a meaningful slice of that pattern recognition — not the whole coaching relationship, just \"what does this stance's geometry say\" — can be learned from data and delivered instantly, to anyone with a phone camera. The system takes a short video of a batter's stroke and returns four 0–100 scores (Balance, Power, Technique, Defence), an uncertainty estimate on each, and a specific, actionable explanation of the single biggest thing to fix."));

children.push(h2("1.2 Why This Project Exists"));
children.push(p("Three real constraints shaped every architectural choice in this system, and understanding them explains almost every design decision you'll read about later: (1) no large, labelled, cricket-specific stance dataset exists publicly — CricketVision (the closest published comparable) has 8,540 clips; this project has 130 in its clean production set — so the whole system had to be built assuming small-data conditions, not scaled up from a big-data default. (2) The person building it does not have access to a studio, a fixed multi-camera rig, or professional players on demand — the training data is YouTube and Instagram clips, filmed by other people for other reasons, in whatever lighting and framing they happened to use. (3) There is a hard external deadline (a technical interview) that forces every decision to be judged not just on research elegance but on whether it produces something real, honest, and demonstrable in the time available. You will see this third constraint explicitly in the milestone history (§9) — several proposed improvements were built, tested, found wanting, and explicitly rejected rather than shipped, because a real negative result presented honestly is worth more (both scientifically and in an interview) than a shaky positive one."));

children.push(h2("1.3 Overall Workflow"));
children.push(p("At the highest level, two independent pipelines share one contract — a trained model file and a fixed 30-feature schema — and nothing else:"));
children.push(bullet("The offline pipeline (everything under dataset/) turns raw video into a trained model: source video → pose extraction → quality filtering → feature engineering → training → evaluation. It runs on a developer's machine, on a schedule the developer controls, and never touches a live user request."));
children.push(bullet("The online pipeline (backend/ + Frontend/) turns one user's uploaded video into four scores in a single HTTP request: upload → validate → extract pose → engineer features → run the trained model → explain the result → store it → delete the video → respond to the browser."));
children.push(p("The reason this separation matters architecturally: it means the live-serving code path can be small, fast to reason about, and rarely touched, while the research/data code path can be messy, iterative, and experimental — exactly the properties you want in each. A bug in a training script cannot take down the live app; a change to serve traffic faster cannot corrupt the training data."));

children.push(h2("1.4 Main Technologies and Why Each Was Chosen"));
children.push(table(
  ["Technology", "Role", "Why this one"],
  [
    ["MediaPipe BlazePose (PoseLandmarker, \"heavy\")", "2D+depth human pose estimation from a single RGB frame", "Runs in real time on CPU with no GPU dependency, ships a pretrained model (no training data of our own needed for this stage), and is the same detector used at both data-collection time and serving time, avoiding a train/serve skew in the input representation itself."],
    ["TensorFlow / Keras", "The trained scoring model", "Native Python/Keras API surface makes the Conv1D→BiLSTM→Attention architecture straightforward to express and to share layer code (model_layers.py) between training and serving without a serialization boundary."],
    ["Django + Django REST Framework", "Backend API, auth, data models", "Batteries-included ORM and admin make the four-model schema (§12) fast to stand up correctly (migrations, relations, permissions) without hand-rolling auth or a query layer."],
    ["React + TanStack Router/Query", "Frontend SPA", "File-based routing keeps the three role-specific views (login, coach, learner) as separate, independently-loadable route modules; TanStack Query gives cache-aware data fetching on top of a plain fetch client without a heavier state-management framework."],
    ["OpenCV (cv2)", "Video frame decoding/seeking", "The only practical way to seek to and decode an arbitrary frame index from an MP4 in Python without re-encoding the whole file."],
    ["yt-dlp", "Training-video acquisition", "The de facto standard for resilient YouTube/Instagram downloading, including format selection (video-only, no audio needed) and cookie-based bot-detection bypass."],
  ],
  [2600, 3200, 3600],
));

children.push(h2("1.5 High-Level Architecture"));
children.push(mono(
`+-----------------------------------------------------------------------+
|                            OFFLINE  (dataset/)                        |
|                                                                         |
|  YouTube/Instagram URL                                                 |
|        |  yt-dlp download + pre-trim                                  |
|        v                                                               |
|  zero_storage_pipeline.py --(NVIDIA NIM)--> shot windows               |
|        |  MediaPipe (shared detector) + subject_selection.py           |
|        v                                                               |
|  keypoints.csv, labels.csv                                             |
|        |  kinematic_validator.py + stance_symmetry_confidence.py       |
|        v                                                               |
|  keypoints_validated.csv --(merge.py, 3 mandatory filters)-->          |
|  dataset.csv --(feature_engineering.py)--> dataset_angles*.csv         |
|        |  train_advanced_model.py / cross_validate.py                 |
|        v                                                               |
|  cricket_stance_advanced_v{N}.keras   <---- the ONLY shared artifact   |
+---------------------------|-------------------------------------------+
                             |
+----------------------------v-------------------------------------------+
|                          ONLINE  (backend/ + Frontend/)                |
|                                                                         |
|  Browser upload (video) --> Django (analyze_stance) --> temp file      |
|        |  ml_service.run_advanced_inference()                          |
|        v                                                               |
|  MediaPipe (shared detector) -> 30-feature tensor (7,30)               |
|        |  loaded .keras model, 30x MC-Dropout, Expected Gradients      |
|        v                                                               |
|  4 scores + uncertainty + explanation --> AnalysisSession row           |
|        |  video deleted (try/finally)                                 |
|        v                                                               |
|  JSON response --> React dashboard (coach.jsx / learner.jsx)           |
+-------------------------------------------------------------------------+`
));

// ================= 2. REPOSITORY STRUCTURE =================
children.push(h1("2. Repository Structure"));
children.push(p("Four top-level areas, deliberately isolated by what they're allowed to depend on. The dependency rule that matters most: dataset/ never imports from backend/, and backend/ imports from dataset/ in exactly one direction — pulling shared, dependency-light modules (schema.py, model_layers.py, rule_based_scorer.py) in, never the reverse."));

children.push(h2("2.1 dataset/ — the offline research and training pipeline"));
children.push(table(
  ["File", "Purpose"],
  [
    ["zero_storage_pipeline.py", "Batch entry point: URL → downloaded → shot-detected → pose-extracted → CSV rows, video always deleted after."],
    ["subject_selection.py", "Multi-person track association and the single session-level \"who is the batsman\" decision (Milestone 4)."],
    ["nvidia_client.py", "Thin client around an NVIDIA NIM-hosted vision-language model for shot-window detection and AI-assisted labelling."],
    ["kinematic_validator.py", "Bone-length coefficient-of-variation filter — flags (not deletes) anatomically implausible frames."],
    ["academic_scripts/stance_symmetry_confidence.py", "Second, complementary quality filter — trajectory smoothness + stance-phase bilateral symmetry (Milestone 3)."],
    ["merge.py", "The single choke point where kinematic_valid, temporal_confidence, and the labelling-fallback filter are all mandatorily applied before data becomes dataset.csv."],
    ["filter_frontview.py", "Scopes the merged dataset down to front-view-only sessions."],
    ["academic_scripts/feature_engineering.py", "Raw keypoints → 15 joint angles + 15 velocities per frame, padded to exactly 7 frames per session."],
    ["academic_scripts/train_advanced_model.py", "Defines the model architecture and trains it on an identity-grouped held-out split."],
    ["academic_scripts/cross_validate.py, evaluation_protocol.py", "The shared, leak-checked cross-validation methodology every evaluation script uses."],
    ["academic_scripts/augment.py", "Flip + Gaussian-jitter data augmentation (tested this project cycle and not adopted — §9)."],
    ["schema.py, model_layers.py", "The two single-source-of-truth modules: the 30-feature list/SEQ_LEN, and the TemporalAttention layer."],
    ["rule_based_scorer.py", "The deterministic fallback scorer used when ML inference fails."],
  ],
  [3400, 6200],
));

children.push(h2("2.2 backend/backend/api/ — the Django REST API"));
children.push(table(
  ["File", "Purpose"],
  [
    ["models.py", "Academy, PlayerProfile, AnalysisSession, CoachOverrideLog — the four-table data model (§12)."],
    ["views.py", "AnalysisSessionViewSet (including analyze_stance, the main upload endpoint), LearnerDashboardView, CoachDashboardView."],
    ["auth_views.py", "RegisterView, UserProfileView — JWT-based signup/identity."],
    ["serializers.py", "DRF serializers for the two profile models and AnalysisSession."],
    ["ml_service.py", "The live inference path: loads the pinned model, runs MediaPipe, computes features, runs MC-Dropout + Expected Gradients, falls back to the rule-based scorer on any failure."],
    ["urls.py", "The complete REST endpoint map (Table 6.1 in the SRS; reproduced in §12 here)."],
  ],
  [2600, 7000],
));

children.push(h2("2.3 Frontend/src/ — the React SPA"));
children.push(table(
  ["File", "Purpose"],
  [
    ["routes/index.jsx", "Unauthenticated landing/login/registration."],
    ["routes/learner.jsx, routes/coach.jsx", "The two role dashboards — each a single file using internal tab state rather than nested routes."],
    ["components/AppShell.jsx", "Shared navigation chrome, parameterized by role."],
    ["lib/api.js", "The entire HTTP layer: one fetchWithAuth() wrapper around the native fetch API — JWT header injection, FormData-aware Content-Type handling, 401 auto-logout."],
    ["routes/__root.tsx", "Mounts the TanStack Query QueryClientProvider and the router outlet."],
  ],
  [2600, 7000],
));

children.push(h2("2.4 Execution Flow and Dependency Graph"));
children.push(mono(
`Frontend  ---HTTP(JWT)--->  backend/backend/api/
                                    |
                                    | imports (one direction only)
                                    v
                              dataset/{schema.py, model_layers.py,
                                       rule_based_scorer.py}

dataset/ (all other files) -- run independently, produce .keras + CSVs --
                                    |
                                    | .keras file path (env var, no code import)
                                    v
                              backend/backend/api/ml_service.py`
));
children.push(p("Note the asymmetry deliberately: the backend imports three small, dependency-light Python modules directly from dataset/ (so the live serving path and the offline training path can never silently disagree on the feature schema or the attention layer's math). But it never imports the training or evaluation scripts themselves, and the connection to a specific trained model is a filename string (CRICKET_MODEL_FILENAME), not a Python import — which is precisely what makes \"promote a new model\" a one-line, no-redeploy-of-training-code operation."));

// ================= 3. END-TO-END WORKFLOW =================
children.push(h1("3. End-to-End Workflow — Upload to Score"));
children.push(p("This section walks the live path a real user request takes, stage by stage. Every stage names the exact file and function responsible, so you can jump straight to the code."));

children.push(h3("Stage 1 — Upload (Frontend → Django)"));
children.push(p("A learner (or a coach, on a player's behalf) selects a video file. The frontend POSTs it as multipart/form-data to /sessions/analyze_stance/ with a JWT bearer token (lib/api.js's fetchWithAuth deliberately skips setting Content-Type when the body is FormData, letting the browser set the correct multipart boundary itself). Django (views.py: AnalysisSessionViewSet.analyze_stance) validates the file's declared content type against an allow-list and its size against a 200MB cap before touching the ML path at all — rejecting a bad upload should be cheap and instant, not discovered 20 seconds into pose extraction."));

children.push(h3("Stage 2 — Temporary Storage (zero-storage boundary begins)"));
children.push(p("The video is written to a UUID-named temp path under MEDIA_ROOT. This is the only moment the video touches disk on the server; a try/finally block around everything from here forward guarantees os.remove() runs whether inference succeeds or raises."));

children.push(h3("Stage 3 — Frame Sampling"));
children.push(p("ml_service.run_advanced_inference() opens the video with cv2.VideoCapture, computes 7 evenly-spaced frame indices via np.linspace(0, total_frames-1, 7), and seeks/decodes each one. If a specific frame fails to decode, the nearest successfully-decoded neighbor is copied in its place — a session never has a hard gap, only a lower-fidelity substitute for one phase."));

children.push(h3("Stage 4 — Pose Estimation and Subject Selection"));
children.push(p("Each of the 7 frames is passed to the process-wide shared MediaPipe PoseLandmarker (num_poses=4). All candidate people per frame are collected, then subject_selection.select_subject() makes one session-level decision: associate candidates into tracks by hip-center proximity, require ≥4-of-7 coverage to qualify, and require the winning track to dominate every rival on both persistence and torso size. If no candidate dominates, the session is rejected outright rather than scored on a guess (§8 explains why this design exists and what it replaced)."));

children.push(h3("Stage 5 — Feature Engineering (identical math to training)"));
children.push(p("The selected pose's 33 landmarks feed the same 15 joint-angle formulas used offline in feature_engineering.py (§6, §10) — each angle computed via the arccos-of-normalized-dot-product formula, divided by 180 to land in [0,1] — followed by their 15 frame-to-frame velocity deltas. The result is a (7, 30) tensor: 7 timesteps, 30 features."));

children.push(h3("Stage 6 — Deep Learning Inference"));
children.push(p("The (1, 7, 30) tensor is run through the loaded Keras model 30 separate times with dropout layers kept stochastically active (training=True at inference time — Monte Carlo Dropout, §5.6/§10). The mean of the 30 passes becomes each of the four 0–100 scores; the standard deviation becomes that score's confidence_variance."));

children.push(h3("Stage 7 — Explainability"));
children.push(p("Expected Gradients (§5.7, §10) computes a signed attribution for every one of the 30 features toward the single weakest of the four scores, using 12 real baseline sequences sampled from the training distribution and 6 interpolation steps per baseline. The top-2 joints by combined |attribution| are mapped to plain-language labels and a matched drill."));

children.push(h3("Stage 8 — Persistence and Response"));
children.push(p("An AnalysisSession row is created with the four scores, the uncertainty dict, the attribution list, and is_fallback=False. The temp video file is deleted. A JSON payload (the full serialized session) is returned to the browser."));

children.push(h3("Stage 9 — Frontend Rendering"));
children.push(p("TanStack Query's cache is invalidated/refetched for the relevant dashboard query, and learner.jsx or coach.jsx re-renders the new session: four score displays, an uncertainty badge (High/Moderate/Low, thresholded on the average confidence_variance), and the weakness/strength/drill text."));

children.push(h3("The failure path — what happens when something breaks"));
children.push(p("Any exception anywhere from Stage 4 through Stage 7 is caught by a single outer try/except in run_advanced_inference. Rather than a 500 error, dataset/rule_based_scorer.score_from_keypoint_df runs on the same already-extracted keypoints, producing a lower-fidelity but real answer (only Balance and Technique are rule-derived; Power and Defence report a neutral 50) with is_fallback=True. This is the single most important reliability property of the whole system: a bug in the neural network path degrades the answer, it never removes it."));

// ================= 4. COMPLETE CODE FLOW =================
children.push(h1("4. Complete Code Flow"));
children.push(p("A literal function-by-function call graph for the live path, with the exact data structure at each arrow. Read top to bottom as one HTTP request's lifetime."));

children.push(mono(
`views.AnalysisSessionViewSet.analyze_stance(request)
  request.FILES['video']                          : Django UploadedFile
  |
  +--> validates content_type in {mp4,mov,webm,avi}, size <= 200MB
  +--> writes to MEDIA_ROOT/videos/<uuid>_<name>   : str (file path)
  |
  +--> ml_service.run_advanced_inference(video_path)
        |
        +--> get_models()                          : (extractor, att_layer, detector) -- cached globals
        |     tf.keras.models.load_model(...)       : Keras Model, loaded ONCE per process
        |
        +--> cv2.VideoCapture(video_path)
        +--> np.linspace(...) -> 7 frame indices    : list[int]
        +--> extract_landmarks(frame, detector) x7  : pd.DataFrame (1 row, 33 landmarks x 5 cols) or None
        +--> pd.concat(...)                          : pd.DataFrame (7 rows)
        |
        +--> _run_model_inference(combined_df, extractor)
              +--> calculate_angle(...) x14, calculate_angle_with_vertical(...) x1
              |     : pd.Series (7,) per angle, appended to angles_df
              +--> angles_df / 180.0                : normalize to [0,1]
              +--> .diff().fillna(0)                : 15 velocity columns
              +--> angles_df.values.astype(float32)  : np.ndarray (7, 30)
              +--> np.expand_dims(..., axis=0)       : np.ndarray (1, 7, 30)  <- MODEL INPUT
              |
              +--> extractor(X_input, training=True) x30   : 30x [np.ndarray (1,4), np.ndarray (1,7,128)]
              +--> np.vstack + mean/std, x 100.0            : 2x np.ndarray (4,)  <- SCORES, UNCERTAINTY
              |
              +--> _top_feature_drivers(extractor, X_input, weakest_key)
                    +--> _expected_gradients(...)      : np.ndarray (7, 30) signed attribution
                    +--> np.argsort(-combined)[:2]      : top-2 feature indices
              |
              +--> returns dict{4 scores, overall, confidence_variance, weakness,
                                  strength, attribution_drivers, recommended_drill, is_fallback=False}
        |
        (any exception above) --> rule_based_scorer.score_from_keypoint_df(combined_df)
                                    : dict{4 scores, overall, weakness, strength, is_fallback=True}
  |
  +--> AnalysisSession.objects.create(**scores_data)   : Django model instance -> SQLite row
  +--> os.remove(video_path)                           : zero-storage guarantee, try/finally
  +--> Response({"session": AnalysisSessionSerializer(new_session).data})   : JSON`
));

children.push(h3("Where each data structure changes shape or type"));
children.push(table(
  ["Object", "Shape / Type", "Created", "Consumed / Destroyed"],
  [
    ["Uploaded video", "Bytes on disk", "analyze_stance(), written to temp path", "Deleted in the try/finally around run_advanced_inference"],
    ["Per-frame landmarks", "pd.DataFrame, 1 row × 165 cols (33 landmarks × 5)", "extract_landmarks() per sampled frame", "Concatenated into combined_df, then discarded"],
    ["combined_df", "pd.DataFrame, 7 rows × 165 cols", "pd.concat(dfs)", "Consumed by _run_model_inference to build angles_df"],
    ["angles_df", "pd.DataFrame, 7 rows × 30 cols", "calculate_angle/_with_vertical calls", "Converted to a numpy tensor, then discarded"],
    ["X_input", "np.ndarray, shape (1, 7, 30), float32", "np.expand_dims", "Fed to the model 30 times (MC-Dropout) and once per attribution baseline step"],
    ["scores_array", "np.ndarray, shape (30, 4)", "np.vstack of 30 MC-Dropout passes", "Reduced to mean/std, then discarded"],
    ["attribution", "np.ndarray, shape (7, 30)", "_expected_gradients", "Reduced to top-2 feature indices, then discarded"],
    ["AnalysisSession row", "Django ORM instance → SQLite row", ".objects.create(...)", "Persists; read back by the dashboards"],
  ],
  [2000, 2600, 2200, 2200],
));

// ================= 5. DEEP LEARNING ARCHITECTURE =================
children.push(h1("5. Deep Learning Architecture — Taught Like a Course"));
children.push(p("The model takes a (7, 30) tensor — 7 timesteps (stance → follow-through), 30 hand-engineered features per timestep — and outputs 4 numbers in [0,1] (later scaled ×100 for reporting). Every layer exists to answer one specific question about that sequence."));

children.push(table(
  ["#", "Layer", "Output Shape", "Params", "The question this layer answers"],
  [
    ["0", "Input", "(7, 30)", "0", "—"],
    ["1", "Conv1D(64, k=3, relu)", "(7, 64)", "5,824", "\"What local, short-range pattern (2-3 adjacent phases) shows up in these joint angles?\" — a sliding window across time, not across features."],
    ["2", "BatchNorm + Dropout(0.2)", "(7, 64)", "256", "Keeps activations stable across a small (42-identity) dataset; randomly zeroes 20% of Conv1D's output at train time so no single filter becomes a single point of failure."],
    ["3", "Bidirectional LSTM(64)", "(7, 128)", "66,048", "\"Given the whole sequence so far AND the whole sequence still to come, what's this timestep's context?\" — bidirectional because \"was the backlift high\" is a fact you'd want a human coach watching the full clip to use when judging the stance phase too, not just hindsight."],
    ["4", "Dropout(0.3)", "(7, 128)", "0", "Heavier dropout here than after Conv1D — the BiLSTM's 128-dim output is the highest-capacity, highest-overfitting-risk representation in the network."],
    ["5", "TemporalAttention (custom)", "(128,)", "135", "\"Which of the 7 phases actually matters most for THIS score?\" — collapses the sequence to one vector by a learned, per-timestep weight, rather than always trusting the last timestep (a plain LSTM's default) or averaging blindly."],
    ["6", "Dense(32, relu)", "(32,)", "4,128", "A small non-linear projection — lets the four scores share a compressed representation before splitting, rather than each being a linear function of the 128-dim context directly."],
    ["7", "Dense(4, sigmoid)", "(4,)", "132", "Squashes each of the 4 outputs into [0,1] — reported ×100. Sigmoid, not softmax: the four scores are independent, not a probability distribution over one choice."],
  ],
  [400, 2400, 1600, 1200, 3600],
));
children.push(caption("76,395 trainable parameters (76,523 including BatchNorm's non-trainable statistics) — confirmed directly against model.summary() on the deployed artifact, not estimated from the script."));

children.push(h2("5.1 Why Conv1D Before the LSTM (Not LSTM Alone)"));
children.push(p("A plain LSTM reading raw angle values timestep-by-timestep has to learn short-range patterns (e.g. \"the knee angle is dropping quickly across 2 consecutive phases\") the hard way, through its recurrent gates. A Conv1D layer with kernel_size=3 is explicitly built to extract exactly that kind of 3-adjacent-timestep pattern in one operation, handing the LSTM an already-locally-summarized sequence to reason over long-range. This TCN-then-RNN pattern (a 1D convolution stem before recurrence) mirrors Temporal Convolutional Network practice (Bai et al., 2018) without adopting a full TCN (dilated causal convolutions stacked deep) — at 7 timesteps total, there is no long sequence to need dilation for."));

children.push(h2("5.2 Why Bidirectional"));
children.push(p("The task is not real-time streaming prediction (where only past frames would be available) — it's offline scoring of an already-complete 7-phase sequence. There is no reason to withhold future-phase information from a decision about an earlier phase, and every phase benefits from full-sequence context (e.g. whether a slightly-bent stance knee turns out to matter depends on what the swing does afterward). This is confirmed, not assumed: architecture_ground_truth.md records that an earlier SRS draft incorrectly claimed the BiLSTM had been \"abandoned\"; direct inspection of the deployed model.summary() showed it is very much in active use."));

children.push(h2("5.3 The Attention Layer, in Detail"));
children.push(p("TemporalAttention (dataset/model_layers.py) is a small, additive (Bahdanau-style, 2014) attention mechanism, not the scaled dot-product self-attention of a Transformer:"));
children.push(mono(
`e     = tanh(x . W + b)          # (batch, 7, 1) -- a learned "importance" score per timestep
alpha = softmax(e, axis=1)        # (batch, 7, 1) -- normalized to sum to 1 across the 7 phases
context = sum(x * alpha, axis=1)  # (batch, 128)  -- weighted sum, NOT the last timestep`
));
children.push(p("Why this over a full Transformer self-attention block: at 7 timesteps and 42 training identities, a multi-head self-attention block's extra parameters would have far more capacity than the data can constrain — the whole point of this smaller mechanism is to let the model learn \"which phase(s) matter\" with a single 128×1 weight matrix (135 parameters total) instead of the 128×128-scale query/key/value projections a Transformer block would need. This is a case where a research-literature-standard component (Transformer attention) was deliberately NOT used, because it doesn't fit the data scale — see §7 for the full comparison."));

children.push(h2("5.4 Loss Function and Training Objective"));
children.push(p("Mean Squared Error (MSE) between the sigmoid-scaled prediction and the label (also scaled to [0,1]), optimized with Adam (learning rate 0.001). MAE is tracked as a metric for human-readable reporting (it's directly interpretable as \"average points off\"), but MSE is what gradients actually optimize — its squared penalty punishes a few badly-wrong sessions harder than MAE would, which is appropriate given how much per-session labelling noise exists at this dataset scale (a single mislabelled session shouldn't be treated as equally important as a systematic bias)."));

children.push(h2("5.5 Monte Carlo Dropout — Uncertainty Estimation"));
children.push(p("Standard inference disables dropout (deterministic forward pass). MC-Dropout (Gal & Ghahramani, 2016) deliberately keeps dropout active at inference (training=True) and runs the forward pass N=30 times; the 30 predictions form an empirical distribution whose mean is the reported score and whose standard deviation is confidence_variance. Intuition: dropout at training time is normally read as regularization (forcing redundancy so no single unit is load-bearing); MC-Dropout reinterprets the SAME mechanism at test time as approximate Bayesian inference over the network's weights — each of the 30 passes is a sample from a different \"thinned\" sub-network, and how much they disagree is a proxy for how uncertain the full ensemble actually is about this specific input."));
children.push(p("Important, honestly-reported limitation (see §9's evaluation record): this uncertainty estimate is not currently calibrated. A direct check against real error (eval_uncertainty_correlation.py, run against the live deployed model) found a weak, not-statistically-significant correlation between MC-Dropout standard deviation and real absolute error (Spearman r between 0.02 and 0.08, p>0.28 on all four scores). It is surfaced to coaches as a qualitative High/Moderate/Low label — a rough signal of the network's own internal disagreement, not a trustworthy confidence interval."));

children.push(h2("5.6 Expected Gradients — Explainability"));
children.push(p("For the single weakest of the four scores, Expected Gradients (Erion et al., 2021) answers \"which of the 30 input features actually pushed this specific prediction down\" — not just \"which features matter on average across the whole dataset\" (which a global feature-importance method would give). The mechanics, in equation form, for a single baseline x' and the true input x:"));
children.push(mono(
`EG_i(x) = (x_i - x'_i) * (1/m) * sum_{k=1..m} [ d f(x' + (k/m)(x - x')) / d x_i ]`
));
children.push(p("— the gradient of the model's output f with respect to feature i, evaluated at m=6 evenly-spaced points along the straight line from baseline to input, averaged, then scaled by how far feature i actually moved from that baseline. Expected Gradients extends this (Erion et al.'s actual contribution over the earlier Integrated Gradients, Sundararajan et al., 2017) by averaging that whole computation over multiple REAL baselines (12 sequences sampled from the training distribution here) rather than one arbitrary reference point (often an all-zeros vector) — which matters a lot for this feature space specifically, since a joint angle of 0 is not a meaningful \"absence of signal\" the way a black pixel is in an image model; there's no natural zero baseline for a knee angle."));

children.push(h2("5.7 Why Not Just Use shap"));
children.push(p("This is worth stating precisely, because it's a common interview probe: the implementation here is deliberately NOT the shap Python package, even though the underlying math (shap.GradientExplainer) is the same algorithm. shap's GradientExplainer depends on numba for JIT compilation, and numba's native compiled component fails to load on the reference development machine due to a permanent Windows Application Control security policy — not a version conflict, not a missing pip install. Implementing the equation directly against tf.GradientTape reproduces the exact math without that blocked dependency. This is documented here in detail because \"why didn't you just use the standard library\" is exactly the kind of question worth having a precise, honest answer to rather than a vague one."));

// ================= 6. DATASET PIPELINE =================
children.push(h1("6. Dataset Pipeline — Teaching the Reasoning"));

children.push(h2("6.1 Why MediaPipe (and not, say, OpenPose or a custom detector)"));
children.push(p("Three properties mattered more than raw accuracy benchmarks: it runs on CPU in near-real-time (this project has no GPU budget, at either data-collection or serving time), it ships a pretrained model so no bespoke pose-estimation training data had to be collected first, and — critically — it is used identically at data-collection time and live-serving time. That last point is a train/serve consistency guarantee: if training data came from one detector and live inference used another, any systematic difference in their landmark conventions would silently bias every prediction, and you'd have no easy way to tell that apart from the model simply being wrong."));

children.push(h2("6.2 Why Joint Angles, Not Raw Coordinates"));
children.push(p("The model could, in principle, take the 33 landmarks' raw (x,y,z) coordinates directly (99 numbers) instead of 15 derived angles. Angles were chosen because they are invariant to where the batter stands in the frame and largely invariant to camera distance — a joint angle means the same thing whether the batter is centered, off to one side, near, or far, while raw coordinates conflate \"body position in the frame\" with \"body position relative to itself,\" forcing the model to learn that invariance from data it doesn't have much of (42 identities). This is the single biggest reason the model can plausibly generalize at this dataset size at all."));

children.push(h2("6.3 Why Velocity (the Second 15 Features)"));
children.push(p("A stance quality judgment is not just about static angle values — the biggest technique/power reads a coach makes are about how fast something moves (e.g. a whippy downswing vs. a lazy one). Frame-to-frame angular velocity (the first difference of each normalized angle across the 7 sampled phases) gives the model a first-order temporal-derivative signal directly, rather than forcing it to infer motion purely from the BiLSTM's own recurrence over the raw angles. The first frame has no prior frame to difference against, so its velocity is defined as 0 — a deliberate, documented convention, not an accidental NaN."));

children.push(h2("6.4 Why Interpolation, and Exactly How Much of It Is Allowed"));
children.push(p("Real footage has motion blur, especially at the explosive downswing/contact phases, so some fraction of sampled frames will fail pose detection or fail the topology check. apply_pipeline_rules (zero_storage_pipeline.py) allows recovering 1–2 consecutive missing interior frames via linear interpolation between the two real, successfully-detected frames bounding the gap (at 1/3 and 2/3 for a 2-frame gap), and recovers a missing FIRST or LAST frame via linear extrapolation from the two frames just inside it. Anything worse — 3+ consecutive missing frames, or 4+ missing total — hard-rejects the whole session rather than fabricating more than a third of a 7-phase sequence from guesswork. This threshold is a real, deliberate trade-off: it recovers genuinely-usable sessions that would otherwise be thrown away for one bad frame, while drawing a hard line before interpolation starts doing more inventing than recovering."));

children.push(h2("6.5 Why Kinematic and Temporal-Confidence Validation (Two Separate Filters)"));
children.push(p("These catch two different failure modes, which is why both exist rather than one: kinematic_validator.py checks that a session's bone lengths (shoulder-to-elbow, hip-to-knee, etc.) stay self-consistent across its 7 frames — a real person's skeleton doesn't change size, so high frame-to-frame variance in an estimated bone length is itself evidence the pose estimate is wrong in at least one frame. stance_symmetry_confidence.py (Milestone 3) checks something bone-length consistency cannot: whether a joint's frame-to-frame TRAJECTORY is physically plausible (a jerk/second-difference measure, normalized by that joint's own total path length so a genuinely fast swing isn't penalized) and whether, specifically at the static stance/trigger phases, the left/right bilateral angles are close to what real, already-valid stance frames look like (z-scored against an empirically-fit reference distribution, not an arbitrary threshold). A session can pass one filter and fail the other — they are complementary, not redundant, which is exactly why the frozen V2 spec made both mandatory rather than picking one."));

children.push(h2("6.6 Why CSV, Not a Database, for the Offline Pipeline"));
children.push(p("The offline pipeline is a research artifact, iterated on constantly, by one person, on one machine — a plain, append-only, git-diffable CSV lets every stage's output be inspected with a text editor or pandas.read_csv, versioned trivially (a .bak_<timestamp> copy before any destructive regeneration, the convention used throughout this project's later revisions), and diffed between pipeline runs without any schema-migration overhead. A database would add real value at multi-user, concurrent-write scale — which this offline pipeline, by design, never operates at."));

children.push(h2("6.7 How Bad Data Is Actively Prevented (Not Just Filtered After the Fact)"));
children.push(p("Two real historical bugs (found and fixed this project cycle, and worth understanding because they're the clearest teaching examples of why this pipeline is built so defensively) illustrate the failure mode this section is really about: a silent data-corruption bug that runs for a long time before anyone notices, because nothing was checking for it."));
children.push(bullet("The frame-name convention bug: feature_engineering.py's padding logic matched only ONE of two coexisting real frame_name conventions used across the codebase's history (\"frame_01_stance.jpg\" vs. the current pipeline's bare \"01_stance\"). A session using the unmatched convention silently NaN'd out — not just its frame name, but every column, including its label scores. Verified against the real dataset: 83 of 197 sessions (42%) were affected before the fix — nearly half the already-collected, already-labelled dataset was being silently discarded."));
children.push(bullet("The labelling-fallback bug: the AI-labelling client defaulted any missing/malformed score in its response to 50, producing label rows indistinguishable from a genuine \"perfectly average on all four dimensions\" assessment unless checked directly against the exact-50-on-all-four signature. 90 of 496 real label rows matched it."));
children.push(p("The lesson generalized from both: every one of this pipeline's mandatory filters (kinematic_valid, temporal_confidence, the fallback-score exclusion) exists because a specific, real, measured failure mode was found — not because a generic \"data quality checklist\" said to add validation. And both bugs were caught by someone actually querying the data directly (counting exact-50 rows, counting NaN scores) rather than trusting that the pipeline \"should\" be producing clean data. That habit — verify the actual file, don't trust the code's intent — is the single most transferable lesson from this whole codebase."));

// ================= 7. RESEARCH FOUNDATIONS =================
children.push(h1("7. Research Foundations"));
children.push(p("For every major technique: the paper it comes from, the core idea, why this project uses it, and — where it matters — how this implementation differs from the paper's original form."));

children.push(table(
  ["Technique / Paper", "Core idea & why used here", "Advantages & limitations"],
  [
    ["MediaPipe BlazePose (Bazarevsky et al., 2020)", "Two-stage lightweight CNN producing 33 3D landmarks per person, built for real-time mobile/CPU inference. Used here because it's CPU-only, pretrained (no bespoke training data needed), and identical at data-collection and serving time (§6.1).", "Fast, well-maintained, supports multi-person detection (Milestone 4 relies on num_poses). Z-depth is measurably less reliable than x/y on monocular footage — the direct motivation for kinematic_validator.py's bone-length check. This project runs it at num_poses=4 with an explicit subject-selection layer on top, not the library default."],
    ["Monte Carlo Dropout (Gal & Ghahramani, ICML 2016)", "Keeping dropout active at test time and sampling N forward passes approximates Bayesian posterior inference over the weights, giving a predictive distribution at zero extra training cost. Used here as the cheapest way to get a per-prediction uncertainty signal from an already-trained model.", "Free at training time, ~30x inference cost at serving time. Not calibrated out of the box — verified directly on this project (§5.5): weak, non-significant correlation with real error. N=30 is a practical compute/precision choice, not derived from the paper."],
    ["Expected Gradients (Erion et al., 2021), extending Integrated Gradients (Sundararajan et al., 2017)", "Integrated Gradients integrates the output gradient along a path from one baseline to the input; Expected Gradients averages that over MANY real baselines, removing the need to hand-pick one arbitrary reference. Used here because there's no natural \"zero\" baseline for a joint angle.", "Same completeness axioms as Integrated Gradients without baseline-choice sensitivity; mathematically identical to shap.GradientExplainer. Still a local, single-prediction explanation. Implemented directly via tf.GradientTape (not the shap package) because shap's numba dependency is blocked on the reference machine (§5.7)."],
    ["Additive/Bahdanau Attention (2015) vs. Transformer self-attention (Vaswani et al., 2017)", "Bahdanau: one learned importance scalar per timestep, softmax-normalized. Transformer: full query/key/value pairwise relevance. Chosen here because at 7 timesteps / 42 identities, self-attention's extra Q/K/V parameters add capacity the data can't constrain — the additive form needs only a 128x1 matrix (135 params).", "Directly interpretable (the softmax weights ARE \"how much each phase mattered\"), cheap. Cannot model pairwise phase interactions. The one place a more modern, more famous technique was deliberately not used — worth citing in an interview as matching technique to data scale."],
    ["Temporal Convolutional stem (Bai, Kolter & Koltun, 2018)", "1D convolutions can match/beat RNNs on sequence tasks; the paper's own architecture stacks many dilated causal layers. Used here as a single Conv1D(k=3) layer, not a full TCN — at 7 timesteps there's no long sequence needing dilation.", "Extracts short-range adjacent-phase patterns in one operation instead of asking the LSTM's gates to learn them recurrently. A lightweight nod to the TCN idea, not a full implementation — shouldn't be described as one without that caveat."],
    ["Bidirectional LSTM (Hochreiter & Schmidhuber, 1997; Schuster & Paliwal, 1997)", "Gated recurrence mitigates vanishing gradients; bidirectional runs forward+backward LSTMs so every position sees full-sequence context. Used here because the task scores an already-complete sequence offline — no reason to withhold future-phase context.", "Every timestep incorporates the full 7-phase context. Cannot be used in a true real-time/live-during-the-swing setting — irrelevant here, since the product is post-hoc analysis, not live coaching."],
    ["Anatomical bone-length consistency filtering — this project's own contribution", "No cricket pose-estimation paper in the current literature applies an automated anatomical-consistency filter to monocular 3D extraction. Core idea: a person's bone lengths don't change across a session's frames, so high frame-to-frame variance in an estimated bone length is itself evidence of a bad estimate — no ground truth needed.", "Requires no extra model. Catches inconsistency, not wrongness that happens to be self-consistent — exactly the gap that let a wrong-person-tracking defect (§8, §9) slip through undetected in one of three real test shots."],
  ],
  [2600, 3800, 3200],
));

// ================= 8. ENGINEERING DECISIONS =================
children.push(h1("8. Why Every Major Engineering Decision Was Made"));
children.push(p("Six of the highest-value decisions in this codebase, each walked through with the full what/why/alternatives/limitations/redesign framework. These were chosen because they come up naturally in a technical interview and because each one taught a real, generalizable lesson, not just a local fix."));

children.push(...sixQ("8.1 The Shared Detector Singleton", {
  what: "One MediaPipe PoseLandmarker instance, created lazily on first use and held for the life of the process (zero_storage_pipeline.py's _shared_detector, guarded by _DETECTOR_LOCK), reused by every session/request instead of constructed fresh each time.",
  why: "Constructing a PoseLandmarker costs real time (~0.7s measured). Before this existed, EVERY session in a batch run — and every importer of the module, even ones that never used the detector — paid that cost independently, and a live Django request would have paid it on every single upload.",
  chosen: "A module-level singleton with a threading.Lock guarding both creation and use. The lock matters because MediaPipe's landmarker is not documented thread-safe — concurrent Django requests must serialize their detection calls on it, which is also honest about a real limitation: at this architecture's current scale, concurrent requests contend for one shared resource, not scale independently.",
  alt: "A detector pool (one per worker thread) would remove the serialization bottleneck but multiply memory use and construction cost by the pool size — not justified before concurrent load is actually a measured problem (§ISSUE-009 in the SRS). A per-request fresh detector (the original behavior) is simplest but the slowest by a wide, measured margin.",
  limits: "Serializes all concurrent inference on one lock — a real throughput ceiling under concurrent load that this design accepts rather than hides.",
  redesign: "Yes, but only once concurrency is a measured problem, not before: move to a small worker pool behind a task queue (Celery/RQ) so a request doesn't block Django's own thread on the lock — over-engineering this before there's real concurrent traffic would have been solving a problem that doesn't exist yet at the cost of real complexity that does.",
}));

children.push(...sixQ("8.2 Subject Selection: Reject Rather Than Guess", {
  what: "subject_selection.py's select_subject() returns None (an explicit rejection) rather than a best-effort pose whenever no single tracked person clearly dominates the frame on both persistence and size.",
  why: "The wrong-person-tracking defect (documented in exhaustive, honest detail across multiple review rounds in architecture_ground_truth.md — see §9) was found because MediaPipe's default single-person detection silently locked onto a ball feeder walking through frame, not the batsman, in 100% of 3 real AI-detected shots checked. A confident-looking WRONG answer is strictly worse than a session that's flagged and skipped, because a wrong answer corrupts training data invisibly while a skipped session just reduces yield, visibly and measurably.",
  chosen: "Track candidates by hip-center proximity, require >=4-of-7 frame coverage to qualify, and require the winning track to beat every rival on BOTH coverage AND a 1.25x torso-size dominance ratio. Both conditions, not one, because persistence alone can favor the wrong person (the feeder was tracked in every frame of the real failure case) and size alone can favor an unrelated close-up foreground walk-through.",
  alt: "A learned person re-identification model would be more robust but needs its own training data this project doesn't have. A simpler \"pick the largest person in frame 1\" heuristic was implicitly what num_poses=1 already did, and it's exactly what failed.",
  limits: "Validated against synthetic multi-person composites and real single-person footage, NOT against a real, ground-truth-labelled multi-person video (none exists in the repo — the original failure clip was deleted by the zero-storage design itself). The dominance thresholds (4-of-7, 1.25x) are documented starting points, not tuned optima.",
  redesign: "The core reject-over-guess principle: no, keep it — it's the single most defensible design choice in this codebase. The specific thresholds: yes, revisit once real multi-person ground-truth footage exists to tune against, rather than leaving documented-but-unvalidated constants indefinitely.",
}));

children.push(...sixQ("8.3 Idempotent Batch Ingestion", {
  what: "run_zero_storage_pipeline() loads the set of already-ingested session names once at batch start and skips any batch_urls.csv row whose sessions already exist, unless --force is passed.",
  why: "Re-running a batch used to re-download, re-extract, and APPEND every row's sessions again on every re-run — the exact corruption class behind a documented real incident (42 duplicate sessions in the historical data).",
  chosen: "A session-name-prefix set check before any download/extraction work starts, so a re-run is cheap (one file read) and safe by default, with an explicit, loudly-labelled escape hatch (--force) for the rare case of deliberately re-ingesting.",
  alt: "A database with a unique constraint on session_name would enforce this at the storage layer instead of in application logic — more robust, but disproportionate to a CSV-based, single-developer research pipeline (§6.6).",
  limits: "The check is prefix-based on session naming convention, not a content hash — if the naming convention itself changed without updating this check, silent duplication could return.",
  redesign: "No — this is proportionate to the actual failure mode and low-cost to maintain. Would only add a real database constraint if the pipeline moved to genuinely concurrent, multi-writer ingestion.",
}));

children.push(...sixQ("8.4 Train/Serve Consistency via schema.py and model_layers.py", {
  what: "The 30-feature list, SEQ_LEN=7, and the TemporalAttention layer's exact math are each defined in exactly one file, imported by every consumer — training scripts, evaluation scripts, and the live Django serving path alike.",
  why: "Before this consolidation (Milestone 1), TemporalAttention was independently redefined in four separate files, and one of those four copies (inference_service.py) used a different Keras API surface (tf.keras.backend.dot/softmax/sum vs. tf.tensordot/tf.keras.activations.softmax/tf.reduce_sum) than the other three — a silent divergence risk that could have produced a model that trains correctly but scores differently when served, with no error message anywhere.",
  chosen: "Two small, single-purpose modules: schema.py (deliberately dependency-free — importable without triggering the TensorFlow/MediaPipe import chain just to read a list of strings) and model_layers.py (the one file allowed to import TensorFlow for the shared layer). Before consolidating, the four independent implementations were verified numerically identical (max absolute difference 0.0 on a random test tensor) — proving they hadn't already silently diverged before merging them.",
  alt: "A shared config file (e.g. YAML) for just the feature list, with the attention layer left duplicated, would only solve half the problem. A full shared ML library package would be over-engineering for a project this size.",
  limits: "This is a discipline, not a language-enforced guarantee — a new script could still hardcode its own copy of the feature list if a future contributor doesn't know these modules exist. The SRS's Known Issues (§11) explicitly flags this as only partially closed: older, non-ML-path scripts weren't individually re-audited.",
  redesign: "No — this is close to the right amount of structure for a project at this scale. A stricter enforcement (e.g. a CI check that greps for hardcoded feature lists) would be a reasonable next step, not a full redesign.",
}));

children.push(p("A related, smaller case of the same discipline: the 7 canonical phase names (\"01_stance\" ... \"07_followthrough\") are defined once (schema.CANONICAL_FRAME_NAMES). Real data had TWO coexisting naming conventions in the wild (a legacy \"frame_01_stance.jpg\" format and the current bare \"01_stance\" format) after the pipeline's convention changed over time without every downstream consumer being updated — feature_engineering.py's padding logic matched only one, silently NaN-ing out every column (not just the frame name) for 42% of sessions using the other. Fixed with a normalize_frame_name() function that strips the optional prefix/suffix at read-time rather than migrating historical data — more robust to a hypothetical third convention appearing later, which migration wouldn't protect against."));

children.push(...sixQ("8.5 Rejection Over Guessing as a Repo-Wide Philosophy", {
  what: "Across independently-written modules — kinematic_validator, apply_pipeline_rules' hard-reject thresholds, subject_selection's dominance requirement, nvidia_client's score-fallback fix, stance_symmetry_confidence's duplicate-frame-name NaN handling — the same pattern recurs: when the system cannot confidently produce a correct answer, it says so explicitly (a flag, a rejection, a NaN, a logged reason) rather than silently substituting a plausible-looking default.",
  why: "Every real, severe bug found and fixed across this project's history was some variant of the OPPOSITE pattern: a missing AI-assessed score silently defaulting to 50; a frame-name mismatch silently producing NaN that wasn't checked for; a coach-invite feature silently claiming an email was sent when none was. Each one was invisible until someone went looking specifically for its exact signature.",
  chosen: "Prefer a loud, checkable failure signal (a boolean column, a rejection log entry with a specific category, a NaN that a downstream filter catches) over a default value that looks like real data.",
  alt: "Aggressive input validation with hard exceptions everywhere would be more disruptive to a research pipeline's iteration speed; silent defaults are faster to write but are exactly the anti-pattern this whole philosophy exists to avoid.",
  limits: "Rejection reduces yield — every one of these mechanisms costs real sessions/frames that a guess-based system would have kept (often wrongly). This is a deliberate, stated trade-off (traceable in the yield numbers throughout the milestone log), not a free improvement.",
  redesign: "No — if anything, this is the principle to protect most carefully as the codebase grows. It is the single idea a new contributor should internalize before touching any data-quality code in this repository.",
}));

// ================= 9. MILESTONE EVOLUTION =================
children.push(h1("9. Milestone Evolution"));
children.push(p("This is a real chronological account, not a tidied-up retrospective — including the parts where an early claim turned out to be wrong and was corrected on review, because that correction process is itself part of what this codebase teaches."));

children.push(h2("Milestone 0 — Original State and Audit Findings"));
children.push(p("The original project SRS contained accuracy figures with no measurement behind them — an \"Overall MAE 15.45 ± 12.56\" and an \"ICC 0.87\" that were never actually computed by any script in the repository. The TemporalAttention layer was independently redefined in four places with no guarantee any two matched. No shared evaluation methodology existed — different scripts split data differently, some by raw session name rather than by real identity, risking train/test leakage across a person's multiple recordings. Test coverage was essentially one pipeline-logic test file; the Django backend's own tests.py was empty boilerplate. This is the baseline every later milestone measures its improvement against."));

children.push(h2("Milestone 1 — ML Single-Source-of-Truth Consolidation"));
children.push(p("Extracted schema.py (the 30-feature list, SEQ_LEN) and model_layers.py (TemporalAttention) as the one place each is defined, verified numerically identical to all four prior independent copies before switching over (§8.4). Files changed: ml_service.py, train_advanced_model.py, inference_service.py, eval_uncertainty_correlation.py, plus the two new shared modules. Impact: closed a real silent-divergence risk between training and serving; zero change to the model's actual behavior (a deterministic forward pass was confirmed byte-for-byte identical before and after)."));

children.push(h2("Milestone 2 — Unified Evaluation Protocol"));
children.push(p("Extracted evaluation_protocol.py as the one home for identity extraction, fold generation, leakage assertion, MAE/RMSE/rank-correlation computation, and significance testing — previously each redefined or simply absent per evaluation script. This is also where a real leakage bug was found and fixed: ablation_study.py had grouped by de-flipped SESSION name rather than real batsman IDENTITY, meaning two different recordings of the same real person could land in different folds — a genuine, not cosmetic, generalization-claim risk. Impact: every evaluation number cited from this point forward in the project's history comes from one audited methodology, not four independently-written ones."));

children.push(h2("Milestone 3 — Stance-Symmetry-Aware Temporal Confidence"));
children.push(p("Added the second, complementary data-quality filter (§6.5). The engineering story here is a genuine lesson in review discipline: the first implementation used post-sort ROW POSITION as a proxy for phase index, which silently broke on 42 real sessions with duplicated frame_name values (two \"01_stance\" rows before the first real \"02_trigger\" row) — fixed by detecting duplicates and returning NaN confidence for the whole session rather than a plausible-looking wrong number. A second review found the paired significance test was comparing MAEs from two INDEPENDENTLY-generated fold splits (different held-out identities in each), making the p-value meaningless despite the code running without error — fixed by deriving both splits' folds from the same identity partition. A third review caught a reporting bug (n_below_threshold summed to the wrong total because duplicate-frame-name sessions were counted as \"scored and found low\" rather than \"never scored\"). Each of these three fixes came from someone re-checking the ACTUAL NUMBERS the code produced against what they should mean, not from re-reading the code for style."));

children.push(h2("Milestone 4 — Wrist-Speed-Guided Adaptive Sampling → Wrong-Person-Tracking Discovery → Subject Selection"));
children.push(p("This is the single richest engineering story in the codebase and worth understanding in full, because it demonstrates honest self-correction under real pressure better than any other part of this project."));
children.push(bullet("Goal: anchor 7-phase sampling on the detected peak wrist speed (a proxy for bat-ball contact) instead of uniform timing, since a swing accelerates non-uniformly. The pure arithmetic (redistribute_phase_indices) was proven correct by unit tests and is unaffected by everything below."));
children.push(bullet("The FIRST validation claim — that peak-finding correctly located a real swing — came from a quick visual read of a few sampled frames and was WRONG: it had found a between-deliveries idle period, not a swing. Caught on review and retracted rather than left standing."));
children.push(bullet("A denser re-check found the REAL defect: MediaPipe's default single-person detection had been tracking the ball feeder, not the batsman — confirmed by overlaying landmarks onto the actual frame (the nose landmark lands on the feeder's face). A candidate fix (reject wrist-fast/hip-stable spikes) was tested directly against this case and REJECTED: the measured ratio (34.43) was indistinguishable from a real swing's own signature."));
children.push(bullet("With no validated fix, the new feature shipped DISABLED by default rather than reverted or left on with just a warning — the pipeline kept its proven uniform sampling active. Follow-up proof-gathering confirmed the defect reproduces on the real production path in all 3 real AI-detected shots checked, and the existing bone-length filter only accidentally caught 2 of those 3."));
children.push(bullet("The eventual real fix: subject_selection.py (§8.2) — multi-person detection plus a dominance-based, reject-capable subject choice, measured against real composited scenes before shipping."));
children.push(p("Lesson generalized: an impressive-sounding \"verified against real video\" claim is only as good as how rigorously it was actually checked, and the correct response to finding your own prior claim was wrong is to say so plainly and re-investigate — not to quietly patch over it."));

children.push(h2("Milestone 5 — Shared Detector Singleton"));
children.push(p("Described in full in §8.1. Closed a real, measured performance cost (~0.7s of redundant detector construction per session/request) with a thread-safe, lazily-initialized singleton."));

children.push(h2("Planned — Confidence-Gated Attention (Not Yet Implemented)"));
children.push(p("The continuous temporal_confidence score computed by Milestone 3 is deliberately kept in the dataset (not discarded after gating) specifically so a future architecture change can feed it into the model's attention mechanism directly — letting the network down-weight a specific low-confidence frame's contribution rather than only ever seeing the same 7 frames whether their input quality was uniformly high or not. This is real, stated future work, not implemented behavior — listed here as exactly that."));

children.push(h2("The Interview-Readiness Cycle (Most Recent)"));
children.push(p("A separate, later, deadline-driven pass: added bowling_type as a tracked field; found and fixed two further real data-corruption bugs (the frame-name-convention and labelling-fallback bugs, §6.7); rebuilt a clean, explicitly-scoped production dataset (front-view only, fast-bowling only); trained a candidate model (v5) and directly, fairly compared it against the deployed model (v4) — finding the deployed model measurably worse (18.61 vs. 11.85 direct MAE) than what the corrected data could produce; designed, implemented, and rigorously evaluated a flip+jitter data-augmentation strategy — and found it measurably HURT accuracy (MAE +6.4%, R² turning negative), so it was not adopted, a real negative result kept rather than hidden; and validated the candidate model through the live serving path itself (not just an offline evaluation script) before promoting it. This cycle is the clearest demonstration in the whole project of a promotion discipline applied in both directions — reject a change that measures worse, adopt one that measures better — rather than shipping whatever is newest."));

// ================= 10. MATHEMATICAL FOUNDATIONS =================
children.push(h1("10. Mathematical Foundations"));
children.push(p("Every equation actually implemented in this codebase, gathered in one place for reference."));

children.push(h3("10.1 Joint Angle (Three-Point)"));
children.push(p("For a joint b with adjacent points a and c (e.g. hip-knee-ankle for knee flexion):"));
children.push(mono(`v1 = a - b,  v2 = c - b
angle = arccos( clip( (v1 . v2) / (|v1| |v2| + eps), -1, 1 ) ) * (180 / pi)
normalized_angle = angle / 180        (eps = 1e-6)`));

children.push(h3("10.2 Joint Angle Against the Vertical (Two-Point)"));
children.push(p("For trunk lean, arm elevation, and head tilt — the angle between a bone vector and the global vertical axis [0,1,0] (MediaPipe's y increases downward):"));
children.push(mono(`v1 = b - a,  v_vertical = [0, 1, 0]
angle = arccos( clip( (v1 . v_vertical) / (|v1| + eps), -1, 1 ) ) * (180 / pi)`));

children.push(h3("10.3 Angular Velocity"));
children.push(mono(`velocity[t] = normalized_angle[t] - normalized_angle[t-1]     (velocity[0] := 0)`));

children.push(h3("10.4 Trajectory Smoothness (Jerk-Based, Path-Length-Normalized)"));
children.push(p("For a tracked joint's 3D position p over a session (interior frames only; edges inherit their nearest interior neighbor's score):"));
children.push(mono(`jerk[i]        = | p[i-1] - 2*p[i] + p[i+1] |            (discrete second difference)
path_length    = sum over the session of | p[i] - p[i-1] |
smoothness[i]  = exp( -jerk[i] / (path_length + eps) )`));
children.push(p("The exp(-x) mapping turns an unbounded jerk-per-path-length ratio into a (0,1] score — 1.0 for a perfectly smooth trajectory, decaying toward 0 as the jerk becomes large relative to the joint's own total motion."));

children.push(h3("10.5 Stance-Phase Bilateral Symmetry (Z-Scored)"));
children.push(mono(`diff = angle_left - angle_right    (at stance/trigger phases only)
z    = (diff - reference_mean) / (reference_std + eps)
symmetry_score = exp( -0.5 * z^2 )`));
children.push(p("This is the Gaussian kernel form — a z-score of exactly 0 (perfectly typical bilateral symmetry) scores 1.0; scores decay smoothly (not as a hard cutoff) as the observed asymmetry moves away from what real, already-valid stance/trigger frames look like."));

children.push(h3("10.6 Monte Carlo Dropout — Predictive Mean and Uncertainty"));
children.push(mono(`{y_1, ..., y_30} = { f_dropout(x) }  x 30 stochastic forward passes (dropout ACTIVE)
mean_score = (1/30) * sum(y_k)                     <- reported score
std_score  = sqrt( (1/30) * sum( (y_k - mean_score)^2 ) )   <- confidence_variance`));

children.push(h3("10.7 Expected Gradients"));
children.push(mono(`EG_i(x) = (x_i - x'_i) * (1/m) * sum_{k=1..m} d/dx_i [ f( x' + (k/m)(x - x') ) ]
Final attribution = average of EG_i(x) over 12 sampled real baselines x'  (m = 6 steps)`));

children.push(h3("10.8 Loss and Evaluation Metrics"));
children.push(mono(`Training loss (per batch):  MSE = mean( (y_pred - y_true)^2 )     over 4 scores, [0,1] scale
Reported MAE:               mean( | y_pred - y_true | )           x100 scale
RMSE:                        sqrt( mean( (y_pred - y_true)^2 ) )
R-squared:                   1 - SS_residual / SS_total`));
children.push(p("Rank correlation (Spearman's rho, Kendall's tau) is additionally reported alongside MAE in this project's evaluation protocol specifically because a subjective 0-100 score is noisier in absolute value than in relative ordering — two graders might disagree on whether a stance is a 72 or a 76, but agree on which of two stances is better. MAE alone cannot distinguish \"wrong by a constant offset but correctly ordered\" from \"randomly wrong\"; rank correlation can."));

children.push(h3("10.9 Interpolation (Missing-Frame Recovery)"));
children.push(mono(`1-frame gap (positions i-1, i+1 known):     p[i] = (p[i-1] + p[i+1]) / 2
2-frame gap (positions i-1, i+2 known):     p[i]   = p[i-1] + (p[i+2]-p[i-1]) / 3
                                             p[i+1] = p[i-1] + (p[i+2]-p[i-1]) * 2/3
Edge extrapolation (first frame missing):   p[0] = p[1] - (p[2] - p[1])
Edge extrapolation (last frame missing):    p[6] = p[5] + (p[5] - p[4])`));

children.push(h3("10.10 Calibration — Explicitly Not Yet Achieved"));
children.push(p("\"Calibration\" here means: does a reported confidence_variance of, say, 5.0 points actually correspond to a real error of roughly 5.0 points on average, across many real predictions? This project's own measurement (§5.5) found it does not, currently — MC-Dropout's spread and real absolute error are only weakly and non-significantly correlated. A calibrated system would need either a held-out calibration set with a fitted mapping (e.g. isotonic regression from raw MC-Dropout std to empirical error, akin to Guo et al., \"On Calibration of Modern Neural Networks,\" 2017) or a fundamentally different uncertainty-quantification method (e.g. deep ensembles) — neither has been built. This gap is stated here in the same place as the equations that would need to change, deliberately, rather than left only as a prose caveat elsewhere."));

// ================= 11. COMPLETE DATA FLOW =================
children.push(h1("11. Complete Data Flow — Every Important Object, End to End"));
children.push(p("Section 4 traced the live-serving path's objects in detail. This table covers the full system, including the offline pipeline, so you can see how a piece of data's shape and meaning change as it moves between the two worlds."));
children.push(table(
  ["Object", "Shape / Type", "Created", "Consumed / Destroyed"],
  [
    ["Source video (full download)", "MP4 file on disk", "yt_dlp download, zero_storage_pipeline.py", "Deleted immediately after pre-trimming to the macro window"],
    ["Macro-trimmed chunk", "MP4 file on disk", "moviepy subclip", "Deleted after all shots within it are processed"],
    ["Shot windows", "list[{start_time, end_time}]", "nvidia_client.find_shot_windows (NVIDIA NIM call)", "Consumed by extract_keypoints_in_memory, one per shot"],
    ["Per-frame pose candidates", "list[list[MediaPipe pose]], up to 4 per frame", "_detect_candidates", "Consumed by subject_selection.select_subject, then discarded"],
    ["Selected subject's poses", "list[pose or None], length 7", "select_subject", "Converted to raw keypoint rows, written to keypoints.csv"],
    ["keypoints.csv row", "CSV row: session_name, frame_name, 33x5 landmark columns, interpolated_frames", "extract_keypoints_in_memory", "Read by kinematic_validator.py"],
    ["labels.csv row", "CSV row: session_name + 4 scores + metadata", "nvidia_client.label_session_frames", "Read by merge.py"],
    ["keypoints_validated.csv", "keypoints.csv + kinematic_valid (bool) + temporal_confidence (float)", "kinematic_validator.py, stance_symmetry_confidence.py", "Read by merge.py"],
    ["dataset.csv", "Merged, filtered keypoints + labels", "merge.py (3 mandatory filters applied)", "Read by filter_frontview.py and feature_engineering.py"],
    ["dataset_angles*.csv", "session_name, frame_name, 30 feature columns, metadata, 4 scores", "feature_engineering.py", "Read by train_advanced_model.py / cross_validate.py"],
    ["(X, y, session_ids)", "np.ndarray (N,7,30), (N,4), list[str]", "build_sequences()", "Split by identity, fed to model.fit / model.predict"],
    [".keras model file", "Serialized Keras model on disk", "model.save() at the end of training", "Loaded once per process by ml_service.get_models() or an evaluation script"],
    ["AnalysisSession", "Django ORM row", "views.analyze_stance, on successful (or fallback) inference", "Read by the learner/coach dashboards; never deleted by the app itself"],
  ],
  [2200, 2600, 2200, 2000],
));

// ================= 12. BACKEND ARCHITECTURE =================
children.push(h1("12. Backend Architecture"));
children.push(h2("12.1 Data Model"));
children.push(table(
  ["Model", "Key Fields", "Notes"],
  [
    ["Academy", "user (1:1), academy_name, created_at", "A coach's account."],
    ["PlayerProfile", "user (1:1, nullable), academy (FK, nullable), name, batting_hand, playing_level, archetype, total_points, current_streak", "A learner's account; academy is nullable so a coach can create a guest player without that player having their own login."],
    ["AnalysisSession", "player (FK), status, 4 scores, overall_score, primary_weakness/strength, thing_to_change, confidence_variance (JSON), attribution_drivers (JSON), is_fallback", "One completed analysis; confidence_variance and attribution_drivers are computed at inference time and persisted, not discarded."],
    ["CoachOverrideLog", "session (FK), coach (FK), metric_changed, old_value, new_value, reason, timestamp", "One row per individual score field a coach changes — field-level audit history, not just \"an update happened.\""],
  ],
  [2000, 4200, 2800],
));
children.push(h2("12.2 API Endpoints"));
children.push(table(
  ["Method & Path", "Purpose"],
  [
    ["POST /auth/register/, /auth/login/, /auth/login/refresh/, GET /auth/me/", "JWT-based signup and identity."],
    ["POST /sessions/analyze_stance/", "The main upload + analysis endpoint."],
    ["GET /sessions/, PATCH /sessions/{id}/", "List (scoped) sessions; coach-only score override with audit logging."],
    ["POST /sessions/ (direct)", "Deliberately rejected (405) — sessions may only be created via analyze_stance, closing an IDOR-style write path."],
    ["GET /learner/me/, GET /coach/me/", "Role-specific dashboard payloads."],
  ],
  [3600, 5400],
));
children.push(h2("12.3 Concurrency and Error Handling"));
children.push(p("Role is determined structurally (hasattr(user, 'academy') / hasattr(user, 'playerprofile')), not by a stored flag, and every queryset-returning view branches on it — this is what actually prevents cross-role data leakage, independent of any frontend guard. Concurrency is currently limited by the shared MediaPipe detector's lock (§8.1) and by having no background task queue — every inference request blocks its full HTTP request/response cycle (measured at 16-19 seconds warm; §14 lists this as the top near-term architecture improvement). Errors in the inference path are caught by one broad try/except that triggers the rule-based fallback (§3, §5); the exception's stack trace is logged server-side only, never returned to the client."));

// ================= 13. FRONTEND ARCHITECTURE =================
children.push(h1("13. Frontend Architecture"));
children.push(p("React with TanStack Router's file-based routing (three route files: index.jsx, learner.jsx, coach.jsx) and TanStack Query for server-state caching (confirmed directly: useQuery in both dashboard routes, a QueryClientProvider mounted in __root.tsx). Each dashboard is a single, fairly large file using internal tab-switch state (React's own useState) for its sub-views, rather than nested router routes per tab — a pragmatic choice for a two-role, small-page-count app, though it means each file mixes layout, data-fetching, and presentation more than a more modular structure would."));
children.push(h2("13.1 The API Layer"));
children.push(p("lib/api.js is a single function, fetchWithAuth(), wrapping the native fetch API: it reads the JWT access token from localStorage, injects it as a Bearer header, deliberately skips setting Content-Type when the request body is FormData (so the browser sets the correct multipart boundary for a video upload itself), and on a 401 response clears stored tokens and hard-redirects to the login route. There is no axios, no generated API client, and no request-level retry logic — a deliberately minimal layer, with TanStack Query providing caching/refetch behavior on top of it rather than the fetch wrapper itself."));
children.push(h2("13.2 Upload Flow"));
children.push(p("A learner selects a video file in the browser; the component builds a FormData payload and calls fetchWithAuth('/sessions/analyze_stance/', {method: 'POST', body: formData}). Because the live request takes 16-19 seconds (§14), the UI needs a loading state spanning that whole window — this is a real UX constraint directly caused by the backend's synchronous architecture, not a frontend limitation, and is one of the clearest arguments for the asynchronous-inference improvement listed in §14."));
children.push(h2("13.3 Result Rendering and User Interactions"));
children.push(p("On a successful response, TanStack Query's cache is invalidated so the relevant dashboard query refetches and the new session appears: four score displays, an uncertainty badge, and the weakness/strength/drill text. Where a data source doesn't yet exist server-side (e.g. a leaderboard rank), the UI renders an explicit, honest empty state rather than a fabricated placeholder — a deliberate correction from an earlier iteration that did show fabricated values, and worth knowing about as an example of the same \"don't guess, say so\" philosophy from §8.6 showing up in the frontend, not just the ML pipeline."));

// ================= 14. FUTURE IMPROVEMENTS =================
children.push(h1("14. Future Improvements"));
children.push(h2("14.1 Current Limitations (Stated Plainly)"));
children.push(bullet("Only 42 unique identities behind the production model, and only 1 of them left-handed — the single biggest ceiling on generalization claims right now, not model architecture."));
children.push(bullet("Uncertainty estimates (MC-Dropout) are not calibrated against real error (§10.10)."));
children.push(bullet("End-to-end live inference measured at 16-19 seconds per request, fully synchronous, no background task queue."));
children.push(bullet("The wrong-person-tracking mitigation (subject_selection.py) is validated on synthetic/composited scenes, not a real ground-truth-labelled multi-person video."));
children.push(h2("14.2 Highest-Value Research Opportunity"));
children.push(p("Group-aware Contrastive Regression (Yu et al., ICCV 2021, \"CoRe\") — a second head regressing the score DIFFERENCE between two sessions, trained on pairs sampled from the existing labelled set. This turns N labels into O(N^2) training signal at zero new annotation cost, directly targeting this project's actual bottleneck (too few labelled identities), rather than a bigger or deeper network."));
children.push(h2("14.3 Performance and Scalability"));
children.push(bullet("Introduce a background task queue (Celery/RQ + Redis) so upload response time is decoupled from the ~16-19s inference latency, and concurrent uploads no longer serialize on the shared-detector lock at the HTTP layer."));
children.push(bullet("Migrate off SQLite to PostgreSQL behind Docker Compose — also sidesteps the reference machine's Windows-specific Application Control DLL issues by moving execution to Linux."));
children.push(h2("14.4 Model Improvements"));
children.push(bullet("Re-run the temporal-confidence-gated attention idea (§9, \"Planned\") once enough labelled data exists to power a properly-controlled ablation."));
children.push(bullet("Revisit contrastive regression (§14.2) once identity count grows meaningfully — not synthetic augmentation, which this project's own controlled experiment measured as making accuracy worse (§7.5 of the SRS, §9 here)."));
children.push(h2("14.5 MLOps and Cloud Deployment"));
children.push(bullet("Log every model promotion event (which .keras file, which git commit, which dataset version) — currently absent; this document's own v4→v5 promotion (§9) is exactly the kind of event this gap would leave unrecorded if applied without one."));
children.push(bullet("Containerize the backend + ML service for reproducible deployment, and consider a managed model-serving layer once concurrent load is a measured (not hypothetical) problem."));

// ================= 15. INTERVIEW PREP =================
children.push(h1("15. Interview Preparation"));
children.push(p("High-yield questions, organized by topic, with strong answers and the follow-up a sharp interviewer would ask next."));

children.push(h3("On the model architecture"));
children.push(...qa(
  "Why Conv1D + BiLSTM + attention instead of a Transformer, given Transformers are the modern default?",
  "At 7 timesteps and 42 training identities, a Transformer's self-attention parameters would add capacity the dataset can't constrain — this project deliberately chose a lighter additive-attention mechanism (135 parameters) over full self-attention for that reason, not out of unfamiliarity with Transformers. Follow-up to expect: \"what would change your mind?\" — meaningfully more identities (hundreds, not tens), where the capacity would actually be usable rather than just memorized."
));
children.push(...qa(
  "Why sigmoid output instead of a linear regression head?",
  "The four scores are bounded to [0,100] by definition; sigmoid (scaled) enforces that range architecturally rather than relying on the loss function alone to keep predictions in-bounds, and keeps gradients well-behaved near the bounds compared to an unbounded linear output that could wander outside [0,100] mid-training."
));
children.push(h3("On uncertainty and explainability"));
children.push(...qa(
  "Is your uncertainty estimate trustworthy?",
  "Measured directly, not assumed: MC-Dropout's standard deviation shows only a weak, non-significant correlation with real error (Spearman r 0.02-0.08, p>0.28) when checked against the live deployed model. It's surfaced as a qualitative label, and that limitation is documented rather than glossed over — this is exactly the kind of honest answer an interviewer is testing for."
));
children.push(...qa(
  "Why not just use the shap library for explainability?",
  "Because shap's GradientExplainer depends on numba, and numba's compiled component is blocked by a Windows Application Control policy on the development machine — not a version issue. Expected Gradients was implemented directly against tf.GradientTape, reproducing the same underlying algorithm without the blocked dependency. Knowing the EXACT reason (not \"it didn't work\") is what separates a strong answer here."
));

children.push(h3("On the dataset and data quality"));
children.push(...qa(
  "What's the biggest data quality bug you found, and how did you find it?",
  "A frame-name convention mismatch silently NaN'd out 42% of sessions' metadata and features, because feature_engineering.py's padding logic only recognized one of two coexisting real naming conventions. It was found by directly counting NaN scores in the output file, not by reading the code and reasoning it should work — that verify-the-actual-file habit is the generalizable lesson."
));
children.push(...qa(
  "You tested data augmentation and didn't ship it. Why would you present a negative result instead of just not mentioning it?",
  "Because the alternative — silently not trying it, or trying it and hiding a bad result — would be worse in both directions: either missing a real potential improvement, or (worse) shipping something that measurably hurts accuracy because no one checked. The augmented dataset scored 6.4% worse MAE and had R^2 turn negative under a controlled, matched-seed, identity-grouped comparison against the production baseline; reporting that is exactly what a rigorous promotion process should produce sometimes."
));

children.push(h3("On the wrong-person-tracking story"));
children.push(...qa(
  "Tell me about a time you found a bug in your own system after believing it was fixed.",
  "The wrist-speed sampling feature's first validation claim — that it correctly located real swings in test footage — was accepted on a quick visual read of a few sampled frames, and turned out to be wrong: it had located a between-deliveries idle period, not a swing. Denser, more rigorous re-checking then found the actual underlying defect (wrong-person tracking), which the first, wrong validation had accidentally been covering up rather than catching. The fix wasn't just patching the wrist-speed feature — it was building a proper multi-person subject-selection layer and shipping the wrist-speed feature disabled until it could be safely rebuilt on top of that."
));

children.push(h3("On engineering process"));
children.push(...qa(
  "How do you decide whether to promote a new model to production?",
  "A candidate must beat the current deployed model on a direct, methodology-matched comparison (not a differently-measured historical number), pass identity-grouped cross-validation with a fixed seed for fair comparison, and be validated through the actual live serving code path against real input — not just an offline evaluation script. A candidate that regresses on any of these isn't promoted, regardless of how recently it was trained; this was applied literally in both directions on this project (an augmentation experiment was rejected for measuring worse; a new model was promoted for measuring and validating better)."
));

// ================= 16. REFERENCES =================
children.push(h1("16. References"));
children.push(h2("Papers"));
children.push(bullet("Bazarevsky, V. et al. (2020). BlazePose: On-device Real-time Body Pose Tracking. arXiv:2006.10204."));
children.push(bullet("Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. ICML."));
children.push(bullet("Sundararajan, M., Taly, A. & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML."));
children.push(bullet("Erion, G., Janizek, J. D., Sturmfels, P., Lundberg, S. & Lee, S.-I. (2021). Improving Performance of Deep Learning Models with Axiomatic Attribution Priors and Expected Gradients. Nature Machine Intelligence."));
children.push(bullet("Bahdanau, D., Cho, K. & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. ICLR."));
children.push(bullet("Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS."));
children.push(bullet("Bai, S., Kolter, J. Z. & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271."));
children.push(bullet("Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation."));
children.push(bullet("Schuster, M. & Paliwal, K. K. (1997). Bidirectional Recurrent Neural Networks. IEEE Transactions on Signal Processing."));
children.push(bullet("Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML."));
children.push(bullet("Yu, X. et al. (2021). Group-aware Contrastive Regression for Action Quality Assessment. ICCV. (CoRe — cited as the highest-priority forward recommendation, §14.2, not yet implemented.)"));
children.push(h2("Official Documentation"));
children.push(bullet("MediaPipe Pose Landmarker — Google AI Edge documentation."));
children.push(bullet("TensorFlow / Keras API reference."));
children.push(bullet("Django and Django REST Framework documentation."));
children.push(bullet("TanStack Router and TanStack Query documentation."));
children.push(h2("Internal Project Documents"));
children.push(bullet("docs/architecture_ground_truth.md — the dated engineering log this document draws its milestone history from."));
children.push(bullet("docs/SuhasVision_SRS_v2.0.docx — the companion requirements specification (current-state, IEEE-830-inspired)."));
children.push(bullet("dataset/EVALUATION_RESULTS.md — the dated evaluation log behind every accuracy figure cited here."));

module.exports = { children, mono, qa, sixQ };
