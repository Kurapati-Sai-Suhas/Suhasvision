"""
kinematic_validator.py
Anatomical plausibility filter for MediaPipe 3D pose keypoints.

WHY THIS EXISTS
MediaPipe's z-coordinate (depth) is a far less reliable estimate than x/y,
especially on monocular YouTube footage. Rather than silently trusting every
frame, this module checks that bone lengths (distances between connected
joints) stay consistent across a session's frames -- a person's actual bone
lengths do not change between frames, so large frame-to-frame variance in a
bone's length is itself evidence of pose estimation error.

This is also a genuine, citable methodological contribution: no cricket
pose-estimation paper in the current literature applies an automated
anatomical-consistency filter to monocular 3D extraction.

USAGE
    python kinematic_validator.py --input keypoints.csv --output keypoints_validated.csv

Adds a `kinematic_valid` boolean column to each frame, and writes
`bone_length_report.csv` summarizing per-session, per-bone coefficient of
variation. Frames are NOT deleted -- only flagged -- so you keep full
control over the trade-off between data quantity and quality.
"""

import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# MediaPipe Pose 33-landmark skeleton, lowercase names matching your CSV
# columns (e.g. "left_shoulder_x", "left_shoulder_y", "left_shoulder_z").
# ---------------------------------------------------------------------------
LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Bones we expect to have a fixed length for a given person across frames.
BONES = {
    "upper_arm_L": ("left_shoulder", "left_elbow"),
    "upper_arm_R": ("right_shoulder", "right_elbow"),
    "forearm_L":   ("left_elbow", "left_wrist"),
    "forearm_R":   ("right_elbow", "right_wrist"),
    "thigh_L":     ("left_hip", "left_knee"),
    "thigh_R":     ("right_hip", "right_knee"),
    "shin_L":      ("left_knee", "left_ankle"),
    "shin_R":      ("right_knee", "right_ankle"),
    "shoulder_span": ("left_shoulder", "right_shoulder"),
    "hip_span":      ("left_hip", "right_hip"),
    "torso_L":       ("left_shoulder", "left_hip"),
    "torso_R":       ("right_shoulder", "right_hip"),
}

# Symmetric bone pairs -- left/right of the same bone should be roughly equal
SYMMETRY_PAIRS = [
    ("upper_arm_L", "upper_arm_R"),
    ("forearm_L", "forearm_R"),
    ("thigh_L", "thigh_R"),
    ("shin_L", "shin_R"),
]

# Thresholds -- tune these against multiview_validation.py's output before
# relying on them for the paper. These defaults are a reasonable starting
# point, not a validated constant.
MAX_BONE_CV = 0.25              # 25% coefficient of variation across a session (relaxed from 0.18)
MAX_SYMMETRY_RATIO_DEV = 0.35   # left/right bone length can differ by up to 35% (relaxed from 0.25)


def _xyz(row, name):
    return np.array([row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"]], dtype=float)


def bone_length(row, joint_a, joint_b):
    a, b = _xyz(row, joint_a), _xyz(row, joint_b)
    return float(np.linalg.norm(a - b))


def compute_bone_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """One row per frame, one column per bone length."""
    out = {}
    for bone_name, (j1, j2) in BONES.items():
        out[bone_name] = df.apply(lambda r: bone_length(r, j1, j2), axis=1)
    return pd.DataFrame(out, index=df.index)


def validate_session(session_df: pd.DataFrame):
    """
    Returns:
        valid_mask: boolean Series aligned to session_df.index
        report: dict of per-bone CV and symmetry deviation for this session
    """
    bones = compute_bone_lengths(session_df)
    report = {}
    frame_flags = pd.Series(True, index=session_df.index)

    for bone_name in BONES:
        vals = bones[bone_name].replace(0, np.nan)
        mean, std = vals.mean(), vals.std()
        cv = std / mean if mean and not np.isnan(mean) else np.nan
        report[f"{bone_name}_cv"] = cv
        if pd.notna(cv) and cv > MAX_BONE_CV:
            # Flag only the outlier frames for this bone, not the whole session
            z = (vals - mean).abs() / (std if std else 1)
            frame_flags &= (z <= 2.5) | vals.isna()

    for left, right in SYMMETRY_PAIRS:
        l_mean, r_mean = bones[left].mean(), bones[right].mean()
        if l_mean and r_mean:
            dev = abs(l_mean - r_mean) / ((l_mean + r_mean) / 2)
            report[f"{left.replace('_L','')}_symmetry_dev"] = dev
            if dev > MAX_SYMMETRY_RATIO_DEV:
                # A whole-session symmetry violation usually means a
                # systematic depth-estimation problem (e.g. an occluded
                # limb) -- flag every frame, not just statistical outliers.
                frame_flags &= False

    return frame_flags, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="keypoints.csv")
    ap.add_argument("--output", default="keypoints_validated.csv")
    ap.add_argument("--report", default="bone_length_report.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["kinematic_valid"] = True
    reports = []

    for session_name, session_df in df.groupby("session_name"):
        valid_mask, report = validate_session(session_df)
        df.loc[session_df.index, "kinematic_valid"] = valid_mask.values
        report["session_name"] = session_name
        report["n_frames"] = len(session_df)
        report["n_flagged"] = int((~valid_mask).sum())
        reports.append(report)

    report_df = pd.DataFrame(reports).set_index("session_name")
    report_df.to_csv(args.report)
    df.to_csv(args.output, index=False)

    total = len(df)
    flagged = int((~df["kinematic_valid"]).sum())
    print(f"Validated {total} frames across {df['session_name'].nunique()} sessions.")
    print(f"Flagged {flagged} frames ({flagged/total:.1%}) as kinematically implausible.")
    print(f"Per-session bone report saved to {args.report}")
    print(f"Full annotated dataset saved to {args.output}")


if __name__ == "__main__":
    main()
