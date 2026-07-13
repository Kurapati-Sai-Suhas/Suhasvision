"""
validate_dataset.py — Run this to audit your frames before labelling.
Checks: frame count, blur, MediaPipe detectability, score distribution.
"""

import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv(override=True)

FRAMES_DIR = "frames"
BLUR_THRESHOLD = 30.0

def check_blur(img_path: str) -> float:
    img  = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def check_mediapipe(img_path: str) -> bool:
    try:
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        img  = cv2.imread(img_path)
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res  = pose.process(rgb)
        pose.close()
        return res.pose_landmarks is not None
    except ImportError:
        return None   # mediapipe not installed yet — skip this check

def validate_local():
    """Validate frames already downloaded locally."""
    sessions = [s for s in Path(FRAMES_DIR).iterdir() if s.is_dir()]

    if not sessions:
        print(f"No sessions found in {FRAMES_DIR}/")
        print("Run extract_frames.py first.")
        return

    print(f"Validating {len(sessions)} sessions in {FRAMES_DIR}/\n")

    report = []
    blurry_sessions  = []
    low_frame_sessions = []
    mediapipe_failed = []

    for session in sorted(sessions):
        frames      = sorted(session.glob("*.jpg"))
        frame_count = len(frames)

        blur_scores    = []
        mp_detections  = []

        for fp in frames:
            b = check_blur(str(fp))
            blur_scores.append(b)

            mp_ok = check_mediapipe(str(fp))
            if mp_ok is not None:
                mp_detections.append(mp_ok)

        avg_blur   = np.mean(blur_scores)  if blur_scores  else 0
        mp_rate    = np.mean(mp_detections) if mp_detections else None

        status = "✓ OK"
        if frame_count < 7:
            status = f"⚠ LOW FRAMES ({frame_count}/7)"
            low_frame_sessions.append(session.name)
        if avg_blur < BLUR_THRESHOLD:
            status = f"⚠ BLURRY (score={avg_blur:.1f})"
            blurry_sessions.append(session.name)
        if mp_rate is not None and mp_rate < 0.5:
            status = f"❌ MEDIAPIPE FAIL ({mp_rate*100:.0f}% detected)"
            mediapipe_failed.append(session.name)

        report.append({
            "session"    : session.name,
            "frames"     : frame_count,
            "avg_blur"   : round(avg_blur, 1),
            "mp_rate"    : round(mp_rate * 100, 0) if mp_rate is not None else "N/A",
            "status"     : status,
        })

    df = pd.DataFrame(report)
    print(df.to_string(index=False))

    print(f"\n{'='*55}")
    print(f"SUMMARY")
    print(f"  Total sessions       : {len(sessions)}")
    print(f"  Total frames         : {sum(r['frames'] for r in report)}")
    print(f"  Blurry sessions      : {len(blurry_sessions)}")
    print(f"  Low frame sessions   : {len(low_frame_sessions)}")
    print(f"  MediaPipe failures   : {len(mediapipe_failed)}")
    print(f"  Usable sessions      : {len(sessions) - len(blurry_sessions) - len(low_frame_sessions)}")

    if low_frame_sessions:
        print(f"\n⚠ Low frame sessions (re-run extract_frames.py on these):")
        for s in low_frame_sessions:
            print(f"   - {s}")

    if blurry_sessions:
        print(f"\n⚠ Blurry sessions (consider re-recording):")
        for s in blurry_sessions[:5]:
            print(f"   - {s}")

    df.to_csv("validation_report.csv", index=False)
    print(f"\nFull report saved to validation_report.csv")


def validate_from_azure():
    """Count frames per session directly in Azure (no download needed)."""
    conn = os.environ.get("AZURE_CONNECTION_STRING")
    if not conn:
        print("No AZURE_CONNECTION_STRING — skipping Azure check")
        return

    blob_svc = BlobServiceClient.from_connection_string(conn)
    cc       = blob_svc.get_container_client("frames")

    sessions = {}
    for b in cc.list_blobs():
        parts = b.name.split("/")
        if len(parts) == 2:
            session_name, fname = parts
            sessions.setdefault(session_name, []).append(fname)

    print(f"\nAzure 'frames' container:")
    print(f"  Total sessions : {len(sessions)}")
    print(f"  Total frames   : {sum(len(v) for v in sessions.values())}")

    incomplete = {k: v for k, v in sessions.items() if len(v) < 7}
    if incomplete:
        print(f"\n  ⚠ Incomplete sessions (<7 frames):")
        for s, frames in incomplete.items():
            print(f"     {s}: {len(frames)} frames")
    else:
        print(f"  ✓ All sessions have 7 frames")

    if Path(FRAMES_DIR).exists():
        local_sessions  = {s.name for s in Path(FRAMES_DIR).iterdir() if s.is_dir()}
        azure_sessions  = set(sessions.keys())
        missing_locally = azure_sessions - local_sessions
        if missing_locally:
            print(f"\n  ⚠ {len(missing_locally)} sessions in Azure but not downloaded locally:")
            print(f"     (Run extract_frames.py to download them)")


if __name__ == "__main__":
    print("=== DATA VALIDATION REPORT ===\n")
    validate_from_azure()
    print()
    validate_local()