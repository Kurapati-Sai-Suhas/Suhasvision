"""
ingest_video.py — Cricket Stance Analyzer
Unified video ingestion module utilizing Gemini SDK (google.genai).
Replaces the old fractured scripts that used local Ollama endpoints.
"""

import cv2
import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

# ── Config ────────────────────────────────────────────────────────────
N_FRAMES           = 7
LOCAL_FRAMES_DIR   = "frames"

# Model fallback chain
GEMINI_MODELS      = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
ACTIVE_MODEL_IDX   = 0

DELAY_ON_QUOTA_HIT = 65

# ── Clients ───────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("🚨 Missing GEMINI_API_KEY in .env")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def get_active_model() -> str:
    return GEMINI_MODELS[min(ACTIVE_MODEL_IDX, len(GEMINI_MODELS) - 1)]


def fallback_model():
    """Move to next model in fallback chain."""
    global ACTIVE_MODEL_IDX
    if ACTIVE_MODEL_IDX < len(GEMINI_MODELS) - 1:
        ACTIVE_MODEL_IDX += 1
        print(f"  [MODEL] Falling back to {get_active_model()}")


def get_video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 30.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total / fps


def get_ai_timestamps(video_path: str, duration: float = None, session_name: str = "unknown", retries: int = 3) -> list:
    """
    Returns a list of dictionaries with start_time and end_time.
    Uses Gemini to locate the sequence. Returns empty list on failure.
    """
    if duration is None:
        duration = get_video_duration(video_path)
        
    gemini_file = None

    for attempt in range(1, retries + 1):
        try:
            model = get_active_model()
            print(f"  [AI] Attempt {attempt}/{retries} using {model}...")

            gemini_file = gemini_client.files.upload(file=video_path)

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

            raw = response.text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)

            if "error" in data:
                print(f"  [AI] Gemini: {data['error']}")
                return []

            start = float(data["start_time"])
            end   = float(data["end_time"])

            # Clamp and validate
            start = max(0.0, start)
            end   = min(duration - 0.05, end)

            if (end - start) < 1.0:
                print(f"  [AI] Window too short ({end-start:.2f}s)")
                return []
            if (end - start) > 8.0:
                end = start + 8.0

            return [{"start_time": start, "end_time": end}]

        except json.JSONDecodeError:
            print(f"  [AI] Bad JSON on attempt {attempt}")

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                if attempt == 1:
                    fallback_model()
                wait = DELAY_ON_QUOTA_HIT * attempt
                print(f"  [429] Quota hit. Waiting {wait}s...")
                time.sleep(wait)
            elif "daily" in err.lower() or "limit: 0" in err.lower():
                print(f"  [QUOTA] Daily quota exhausted for {get_active_model()}.")
                fallback_model()
                time.sleep(5)
            else:
                print(f"  [ERROR] {e}")

        finally:
            if gemini_file:
                try:
                    gemini_client.files.delete(name=gemini_file.name)
                    gemini_file = None
                except Exception:
                    pass

    return []
