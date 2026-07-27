# test_fixtures/

Known-good regression fixtures, snapshotted 2026-07-20 from the live production
data files (Milestone 0 of STABILIZATION_PLAN.md). Unlike `dataset/*.csv`,
files in this subdirectory are **committed to git** — they are the stable
reference the gitignored, regenerated data files get compared against.

| File | Source | Contents |
|---|---|---|
| `fixture_keypoints_session.csv` | `keypoints.csv` | One complete 7-phase session (`abhishek_frontview_10s_01`): raw MediaPipe output, 33 landmarks x {x,y,z,visibility,presence} per frame, zero interpolated phases, bare `frame_name` convention (`01_stance` ... `07_followthrough`). |
| `fixture_dataset_angles_session.csv` | `dataset_angles_production.csv` | The same session's training-ready rows: label metadata + 15 normalized angles + 15 velocities, no NaNs. |

Intended uses:

1. **Pipeline-change verification** — after modifying feature engineering or
   validation code, recompute features from the keypoints fixture and diff
   against the angles fixture. Any unexplained difference is a regression.
2. **Unit-test input** — a realistic, complete session for tests that need
   real landmark geometry rather than synthetic zeros.

The session derives from public YouTube footage already in the training set.
If the feature schema ever changes intentionally, regenerate BOTH files in the
same commit that changes it, and say so in the commit message.
