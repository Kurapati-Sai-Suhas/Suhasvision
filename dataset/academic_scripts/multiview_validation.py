"""
multiview_validation.py
Compares MediaPipe 3D keypoint estimates from two camera angles of the SAME
stance, filmed simultaneously, to quantify how reliable each coordinate
axis actually is. This is the single experiment most likely to satisfy a
reviewer's skepticism about monocular 3D pose estimation, and it directly
turns your z-axis weakness into a reported, quantified result instead of
an unexamined assumption.

HOW TO COLLECT THE DATA
Film 15-20 stance clips with two phones simultaneously from two angles
(e.g. front-on and side-on, or two oblique angles roughly 45-90 degrees
apart). Run your normal pipeline on both videos for each clip. Name the
resulting sessions consistently with a camera suffix, e.g.
"clip01_cam1" and "clip01_cam2", so this script can pair them automatically.

USAGE
    python multiview_validation.py --keypoints keypoints_validated.csv --auto_pair
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

LANDMARKS_TO_CHECK = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def auto_pairs(session_ids, suffix1="_cam1", suffix2="_cam2"):
    pairs = []
    bases = {s[:-len(suffix1)] for s in session_ids if s.endswith(suffix1)}
    for base in bases:
        c1, c2 = base + suffix1, base + suffix2
        if c1 in session_ids and c2 in session_ids:
            pairs.append((base, c1, c2))
    return pairs


def compare_pair(df: pd.DataFrame, session_a: str, session_b: str) -> dict:
    a = df[df.session_id == session_a].sort_values("frame_index").reset_index(drop=True)
    b = df[df.session_id == session_b].sort_values("frame_index").reset_index(drop=True)
    n = min(len(a), len(b))
    a, b = a.iloc[:n], b.iloc[:n]

    result = {}
    for lm in LANDMARKS_TO_CHECK:
        for axis in ("x", "y", "z"):
            col = f"{lm}_{axis}"
            if col not in df.columns:
                continue
            va, vb = a[col].values, b[col].values
            if len(va) < 3:
                continue
            r, _ = pearsonr(va, vb)
            mae = np.mean(np.abs(va - vb))
            result[f"{lm}_{axis}_r"] = round(r, 3)
            result[f"{lm}_{axis}_mae"] = round(float(mae), 4)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoints", default="keypoints_validated.csv")
    ap.add_argument("--auto_pair", action="store_true")
    ap.add_argument("--suffix1", default="_cam1")
    ap.add_argument("--suffix2", default="_cam2")
    ap.add_argument("--output", default="multiview_validation_report.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.keypoints)
    session_ids = df.session_id.unique().tolist()

    if not args.auto_pair:
        raise SystemExit("Currently only --auto_pair is implemented. Name your "
                          "sessions like 'clip01_cam1' / 'clip01_cam2'.")

    pairs = auto_pairs(session_ids, args.suffix1, args.suffix2)
    print(f"Found {len(pairs)} matched camera-angle pairs.")
    if not pairs:
        raise SystemExit("No matched pairs found. Check your session naming convention.")

    all_results = []
    for base, s1, s2 in pairs:
        r = compare_pair(df, s1, s2)
        r["clip"] = base
        all_results.append(r)

    report = pd.DataFrame(all_results).set_index("clip")
    report.to_csv(args.output)

    print("\nMEAN CORRELATION BY AXIS (across all clips and joints):")
    for axis in ("x", "y", "z"):
        r_cols = [c for c in report.columns if c.endswith(f"_{axis}_r")]
        if r_cols:
            mean_r = report[r_cols].values.mean()
            print(f"  {axis}-axis: mean r = {mean_r:.3f}")

    print(f"\nFull per-joint, per-clip report saved to {args.output}")
    print("\nFor the paper: report the z-axis mean r explicitly as your "
          "evidence for (or honest limitation around) depth reliability. "
          "Use this number to justify your MAX_BONE_CV threshold in "
          "kinematic_validator.py rather than leaving it as a guessed constant.")


if __name__ == "__main__":
    main()
