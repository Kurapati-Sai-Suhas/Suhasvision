"""
extract_frames.py — Cricket Stance Analyzer
Fixes: 429 quota handling, exponential backoff, crash on retry failure,
       rate limiting between videos, model fallback chain.
"""

import cv2
import os
import time
import json
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

# ── Config ────────────────────────────────────────────────────────────
N_FRAMES           = 7
LOCAL_FRAMES_DIR   = "frames"

# Model fallback chain — if 2.5-flash quota exhausted, falls back to 1.5-flash
# 1.5-flash has higher free tier RPM (15 req/min vs 10)
GEMINI_MODELS      = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
ACTIVE_MODEL_IDX   = 0          # starts with best model, falls back on quota

# Rate limiting — conservative defaults for free tier
DELAY_BETWEEN_VIDEOS    = 15    # seconds between each video (keeps RPM safe)
DELAY_ON_QUOTA_HIT      = 65    # seconds to wait on 429 (quota resets per minute)
MAX_RETRIES_PER_VIDEO   = 3

IN_CONTAINER   = os.environ.get("AZURE_CONTAINER_NAME", "raw-videos")
OUT_CONTAINER  = "frames"

# ── Clients ───────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
AZURE_CONN_STR = os.environ.get("AZURE_CONNECTION_STRING")

if not GEMINI_API_KEY:
    raise RuntimeError("🚨 Missing GEMINI_API_KEY in .env")
if not AZURE_CONN_STR:
    raise RuntimeError("🚨 Missing AZURE_CONNECTION_STRING in .env")

gemini_client    = genai.Client(api_key=GEMINI_API_KEY)
blob_svc         = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
in_cc            = blob_svc.get_container_client(IN_CONTAINER)
out_cc           = blob_svc.get_container_client(OUT_CONTAINER)

# ── Helpers ───────────────────────────────────────────────────────────

def get_active_model() -> str:
    return GEMINI_MODELS[min(ACTIVE_MODEL_IDX, len(GEMINI_MODELS) - 1)]


def fallback_model():
    """Move to next model in fallback chain."""
    global ACTIVE_MODEL_IDX
    if ACTIVE_MODEL_IDX < len(GEMINI_MODELS) - 1:
        ACTIVE_MODEL_IDX += 1
        print(f"  [MODEL] Falling back to {get_active_model()}")
    else:
        print(f"  [MODEL] Already at last fallback: {get_active_model()}")


def already_processed(session_name: str) -> bool:
    """Check Azure 'frames' container — skip if already has 7 frames."""
    prefix   = f"{session_name}/"
    existing = list(out_cc.list_blobs(name_starts_with=prefix))
    if len(existing) >= N_FRAMES:
        print(f"  [SKIP] {session_name} already done ({len(existing)} frames in Azure)")
        return True
    return False


def get_video_duration(path: str) -> float:
    cap    = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total / fps


def get_stance_timestamps(video_path: str, duration: float) -> dict | None:
    """
    Asks Gemini to locate the batting sequence.
    Handles 429 with exponential backoff + model fallback.
    Never crashes the outer loop.
    """
    gemini_file = None

    for attempt in range(1, MAX_RETRIES_PER_VIDEO + 1):
        try:
            model = get_active_model()
            print(f"  [AI] Attempt {attempt}/{MAX_RETRIES_PER_VIDEO} using {model}...")

            # Upload video to Gemini Files API
            gemini_file = gemini_client.files.upload(file=video_path)

            # Wait for processing (max 60s)
            for _ in range(30):
                if gemini_file.state.name != "PROCESSING":
                    break
                time.sleep(2)
                gemini_file = gemini_client.files.get(name=gemini_file.name)

            prompt = f"""
You are an expert cricket biomechanics coach.
Video duration: {duration:.1f} seconds.

Watch this cricket batting video and locate the COMPLETE batting sequence:
- START: The moment BEFORE the batsman begins trigger movement (initial weight shift / footwork as bowler approaches).
- END: The moment AFTER the batsman completes follow-through post ball contact.

Rules:
- start_time >= 0.0
- end_time <= {duration:.1f}
- Window (end - start) must be between 1.0 and 8.0 seconds
- If multiple shots: pick the most clearly visible, most side-on one
- If no batting shot visible: return {{"error": "no shot found"}}

Return ONLY raw JSON. No markdown. No explanation.
Format: {{"start_time": 1.2, "end_time": 4.8}}
"""
            response = gemini_client.models.generate_content(
                model=model,
                contents=[gemini_file, prompt]
            )

            raw  = response.text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)

            if "error" in data:
                print(f"  [AI] Gemini: {data['error']} — skipping video")
                return None

            start = float(data["start_time"])
            end   = float(data["end_time"])

            # Clamp and validate
            start = max(0.0, start)
            end   = min(duration - 0.05, end)

            if (end - start) < 1.0:
                print(f"  [AI] Window too short ({end-start:.2f}s) — skipping")
                return None
            if (end - start) > 8.0:
                end = start + 8.0

            return {"start_time": start, "end_time": end}

        except json.JSONDecodeError:
            print(f"  [AI] Bad JSON on attempt {attempt} — retrying")

        except Exception as e:
            err = str(e)

            # ── Quota / Rate limit ─────────────────────────────────
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                # Try falling back to cheaper model first
                if attempt == 1:
                    fallback_model()
                wait = DELAY_ON_QUOTA_HIT * attempt  # exponential: 65s, 130s, 195s
                print(f"  [429] Quota hit. Waiting {wait}s before retry {attempt+1}...")
                time.sleep(wait)

            # ── Daily quota completely exhausted ───────────────────
            elif "daily" in err.lower() or "limit: 0" in err.lower():
                print(f"  [QUOTA] Daily quota exhausted for {get_active_model()}.")
                fallback_model()
                if ACTIVE_MODEL_IDX >= len(GEMINI_MODELS) - 1:
                    print("  [QUOTA] All models exhausted. Stop and resume tomorrow.")
                    return None
                time.sleep(10)

            else:
                print(f"  [ERROR] Unexpected error: {e}")
                return None

        finally:
            # Always clean up Gemini file
            if gemini_file:
                try:
                    gemini_client.files.delete(name=gemini_file.name)
                    gemini_file = None
                except Exception:
                    pass

    print(f"  [FAIL] {video_path} failed after {MAX_RETRIES_PER_VIDEO} attempts — moving on")
    return None


def extract_and_upload_frames(
    video_path: str,
    session_name: str,
    timestamps: dict
) -> int:
    """Extract 7 frames, save locally, upload to Azure."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    start_f = int(timestamps["start_time"] * fps)
    end_f   = int(timestamps["end_time"]   * fps)
    window  = end_f - start_f

    out_dir = Path(LOCAL_FRAMES_DIR) / session_name
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = [
        int(start_f + (window * i / (N_FRAMES - 1)))
        for i in range(N_FRAMES)
    ]

    phase_labels = [
        "01_stance", "02_trigger", "03_backlift_start",
        "04_full_backlift", "05_downswing", "06_contact", "07_followthrough"
    ]

    saved = 0
    for idx, frame_num in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        img_name   = f"frame_{phase_labels[idx]}.jpg"
        local_path = str(out_dir / img_name)
        cv2.imwrite(local_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        azure_path = f"{session_name}/{img_name}"
        try:
            bc = out_cc.get_blob_client(azure_path)
            with open(local_path, "rb") as data:
                bc.upload_blob(data, overwrite=True)
            saved += 1
        except Exception as e:
            print(f"  [UPLOAD] Failed {img_name}: {e}")

    cap.release()
    return saved


# ── Main ──────────────────────────────────────────────────────────────
def run():
    blobs = [
        b for b in in_cc.list_blobs()
        if b.name.lower().endswith((".mp4", ".mov", ".avi"))
    ]

    if not blobs:
        print("No videos found in Azure raw-videos container.")
        return

    print(f"Found {len(blobs)} videos. Starting pipeline with model: {get_active_model()}")
    print(f"Delay between videos: {DELAY_BETWEEN_VIDEOS}s\n")

    total_frames  = 0
    total_skipped = 0
    total_failed  = 0

    for i, blob in enumerate(blobs, 1):
        session = Path(blob.name).stem
        print(f"\n[{i}/{len(blobs)}] {blob.name}")

        if already_processed(session):
            total_skipped += 1
            continue

        temp = f"_temp_{Path(blob.name).name}"
        try:
            # Download
            print(f"  [↓] Downloading from Azure...")
            with open(temp, "wb") as f:
                f.write(in_cc.download_blob(blob.name).readall())

            duration = get_video_duration(temp)
            print(f"  [INFO] Duration: {duration:.1f}s | Model: {get_active_model()}")

            if duration < 1.0:
                print(f"  [SKIP] Too short")
                total_failed += 1
                continue

            # Get timestamps
            ts = get_stance_timestamps(temp, duration)
            if not ts:
                total_failed += 1
                continue

            print(f"  [✂] Window: {ts['start_time']:.2f}s → {ts['end_time']:.2f}s")

            # Extract + upload
            n = extract_and_upload_frames(temp, session, ts)
            print(f"  [✓] {n}/{N_FRAMES} frames saved + uploaded")
            total_frames += n

        except Exception as e:
            print(f"  [ERROR] {e}")
            total_failed += 1

        finally:
            if os.path.exists(temp):
                os.remove(temp)

        # Rate limit delay between videos
        if i < len(blobs):
            print(f"  [⏱] Waiting {DELAY_BETWEEN_VIDEOS}s before next video...")
            time.sleep(DELAY_BETWEEN_VIDEOS)

    print(f"\n{'='*50}")
    print(f"Pipeline complete!")
    print(f"  Frames extracted : {total_frames}")
    print(f"  Sessions skipped : {total_skipped} (already done)")
    print(f"  Sessions failed  : {total_failed}")


if __name__ == "__main__":
    run()