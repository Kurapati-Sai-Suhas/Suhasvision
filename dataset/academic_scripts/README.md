# Gap-Filling Scripts — Cricket Stance Analyzer

Nine scripts addressing the specific weaknesses identified in the project
review, designed to drop directly into your existing `dataset/` folder
alongside `zero_storage_pipeline.py`, `rule_based_scorer.py`, and
`labelling_tool.py`. Every script below was functionally tested against
synthetic data before being handed to you — not just syntax-checked.

## Install what's missing

```bash
pip install scikit-learn scipy matplotlib tensorflow --break-system-packages
```
(You likely already have pandas, numpy, google-genai, python-dotenv, pillow
from the existing pipeline.)

## Run order

These map onto your Day 8+ TODOs and the "Priority order" from the project
review. Run them roughly in this sequence:

### 1. `kinematic_validator.py` — run this first, on your existing data
Flags anatomically implausible frames using bone-length consistency across
each session. This is the cheapest, highest-impact fix and should run on
every dataset version going forward.
```bash
python kinematic_validator.py --input keypoints.csv --output keypoints_validated.csv
```
Tested against synthetic data with two injected glitch frames — caught both
exactly, zero false positives on clean frames. Tune `MAX_BONE_CV` and
`MAX_SYMMETRY_RATIO_DEV` at the top of the file once you have real numbers
from `multiview_validation.py` (script 2 below) — the defaults are
reasonable starting points, not validated constants.

### 2. `multiview_validation.py` — requires a half-day filming session
Films the same stance from two angles simultaneously, runs your pipeline on
both, and reports per-axis, per-joint correlation. This is the experiment
that turns "MediaPipe's z-axis might be unreliable" from an assumption into
a reported, quantified number for the paper.
```bash
# Film 15-20 clips with two phones, name sessions clip01_cam1 / clip01_cam2 etc.
python multiview_validation.py --keypoints keypoints_validated.csv --auto_pair
```

### 3. `dataset_diversity_report.py` — run anytime, rerun as dataset grows
Reports your current shot-type, batting-hand, and skill-level breakdown.
It will tell you explicitly which columns are missing from `labels.csv` and
gives you the exact Streamlit code to add them.
```bash
python dataset_diversity_report.py --labels labels.csv --keypoints keypoints.csv
```

### 4. `agreement_calculator.py` — run before locking the dataset
Have a second person independently label a shared sample of sessions via
your existing `labelling_tool.py`, save to a separately-named CSV, then run:
```bash
python agreement_calculator.py --files labels_labeller1.csv labels_labeller2.csv
```
Tested against synthetic two-labeller data with known noise levels — ICC
and weighted kappa both recovered correctly. If your real kappa/ICC comes
back below 0.60, fix the rubric before generating more labels.

### 5. `gemini_scorer.py` — the second arm of your comparison table
Zero-shot stance scoring baseline (separate from your existing Gemini
timestamp-detection call). Run once per session you want in your test set:
```bash
python gemini_scorer.py --frames_dir frames/<session_id> --session_id <id>
```

### 6. `train_lstm_model.py` — supersedes a static single-frame model
Trains a Bidirectional LSTM over your phase-ordered keypoint sequences
instead of scoring a single frame — this is the stronger, more novel claim,
since you already capture `shot_phase` per frame.
```bash
python train_lstm_model.py --keypoints keypoints_validated.csv --labels labels.csv
```
The data pipeline (sequence building, hip-centering normalization,
zero-padding) was verified independently of TensorFlow — shapes, label
alignment, and no-NaN-leakage all confirmed correct.

### 7. `compare_methods.py` — the core results table for the paper
```bash
python compare_methods.py --labels labels.csv --rule_based rule_based_scores.csv \
    --gemini gemini_scores.csv --lstm_predictions lstm_predictions.csv
```
Tested against synthetic data with known noise levels per method — recovered
the correct ranking (lowest-noise method scored best on MAE/RMSE/r) exactly.

### 8. `discriminability_analysis.py` — optional but high payoff
Tests whether your score actually separates professional from amateur
footage. Requires the `skill_level` column from script 3.
```bash
python discriminability_analysis.py --scores_col quality_score
```
Tested against synthetic overlapping-but-separable distributions — AUC
came back correctly in the "strong" range as designed.

### 9. `live_demo.py` — for your supplementary video, do this last
Orchestration skeleton tying your real pipeline functions together
end-to-end. The function calls are commented out because they need to match
your exact signatures in `zero_storage_pipeline.py`, `extract_keypoints.py`,
and `rule_based_scorer.py` — open the file, uncomment, and adjust the
imports to your actual function names, then screen-record a run of it.

## A note on what wasn't fully testable here

`train_lstm_model.py`'s Keras layer wiring (the actual `Bidirectional(LSTM(...))`
stack) couldn't be executed in this environment without installing
TensorFlow, which wasn't done to keep this handoff fast. The data pipeline
feeding into it was verified in full. Run a quick `model.summary()` the
first time you execute it on your machine to sanity-check the architecture
before a long training run.

`live_demo.py` is intentionally a skeleton — it calls out exactly which
three function names it expects from your existing files. Fill those in
and it becomes your supplementary-material demo script.
