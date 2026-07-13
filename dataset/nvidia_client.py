"""
nvidia_client.py — NVIDIA NIM vision calls for the zero-storage ingestion pipeline.

Replaces Gemini for both AI touchpoints in zero_storage_pipeline.py:
  1. find_shot_window()   — locate the batting stroke within a macro-trimmed clip
  2. label_session_frames() — score/label a 7-frame session

Design note on shot-window detection: rather than depending on any provider's
native video-timestamp understanding (which varies a lot in reliability across
NIM's free-tier vision models), we sample frames ourselves at a fixed rate and
only ask the model to pick which *frame index* bounds the shot. That's plain
multi-image reasoning, which is a well-supported, low-risk capability on NIM's
OpenAI-compatible endpoint. We control the timestamp math ourselves.
"""
import os
import json
import time
import base64
from io import BytesIO

import cv2
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Every NIM model shares the same /v1/chat/completions shape, so if this model
# is ever retired/renamed, swapping this one string is the only change needed.
# nemotron-3-nano-omni is NVIDIA's own vision+video model, and its official docs
# use almost exactly this pattern (video keyframes -> image array -> one call).
NVIDIA_VISION_MODEL = os.environ.get("NVIDIA_VISION_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")

_client = None


class CreditsExhaustedError(Exception):
    """Raised when NVIDIA returns 402 Payment Required — free-tier credits are gone.
    Deliberately NOT caught inside _chat_with_images: this should stop the whole
    batch immediately rather than being logged as a per-call failure and moving
    on to burn through the rest of batch_urls.csv with the same guaranteed failure."""
    pass


def get_client():
    global _client
    if _client is None:
        if not NVIDIA_API_KEY:
            raise RuntimeError("Missing NVIDIA_API_KEY in dataset/.env — get one free at build.nvidia.com")
        _client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    return _client


def _bgr_frame_to_data_url(frame_bgr, quality=85):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return _rgb_array_to_data_url(frame_rgb, quality)


def _rgb_array_to_data_url(frame_rgb, quality=90):
    img = Image.fromarray(frame_rgb)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _chat_with_images(content, max_tokens, retries=3):
    client = get_client()
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=NVIDIA_VISION_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"  [NVIDIA] Bad JSON on attempt {attempt}/{retries}, retrying...")
            time.sleep(3)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status == 402:
                # Out of credits — stop the whole batch, don't burn through the
                # rest of batch_urls.csv failing the same way on every row.
                raise CreditsExhaustedError(
                    "NVIDIA free-tier credits are exhausted (HTTP 402). "
                    "Check your balance at build.nvidia.com."
                ) from e
            if status == 429 or "429" in str(e) or "rate" in str(e).lower():
                wait = 15 * attempt
                print(f"  [NVIDIA] Rate limited, waiting {wait}s (attempt {attempt}/{retries})...")
                time.sleep(wait)
            else:
                print(f"  [NVIDIA] Request failed: {e}")
                return None
    return None


MAX_IMAGES_PER_PROMPT = 10  # NVIDIA's endpoint hard-caps at 12; stay safely under it


def find_shot_windows(video_path, duration, max_samples=MAX_IMAGES_PER_PROMPT, retries=3, max_shots=20):
    """
    Scans `video_path` (already macro-trimmed) from 0 to `duration` seconds,
    sampling up to `max_samples` EVENLY-SPACED frames across the whole window
    (not a fixed fps — the NVIDIA endpoint hard-rejects prompts with more than
    12 images, so sample count must stay constant regardless of window length).
    Returns a LIST of {"start_time": float, "end_time": float} dicts — one per
    distinct batting shot found, relative to video_path's OWN 0-based timeline
    (matches what extract_keypoints_in_memory expects). A single macro window
    can contain several separate shots (your existing dataset has session names
    going up to _14 from one video) — this finds all of them, not just one.

    Trade-off: longer macro windows get coarser temporal resolution (10 samples
    across 90s is one frame every 9s). Keep macro windows tight (15-30s) in
    batch_urls.csv for better shot-boundary precision.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(duration * fps)
    if total_frames < 3:
        cap.release()
        return []

    n_samples = max(3, min(max_samples, total_frames))
    if n_samples == 1:
        frame_nums = [0]
    else:
        frame_nums = sorted(set(
            min(int(round(total_frames * i / (n_samples - 1))), total_frames - 1)
            for i in range(n_samples)
        ))

    sampled = []
    for frame_num in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        sampled.append((frame_num / fps, frame))
    cap.release()

    if len(sampled) < 3:
        return []

    content = [{
        "type": "text",
        "text": (
            f"These are {len(sampled)} frames sampled evenly across a {duration:.1f}s cricket "
            f"video clip, numbered 0 to {len(sampled) - 1} in chronological order. "
            "Identify EVERY distinct complete batting sequence visible — there may be "
            "one, several, or none. For each one, give the frame index range from the "
            "moment BEFORE the batsman begins trigger movement, to the moment AFTER "
            "follow-through completes. Return ONLY raw JSON, no markdown, as an array: "
            '[{"start_index": <int>, "end_index": <int>}, ...] '
            f"(up to {max_shots} shots) or [] if no batting action is visible."
        ),
    }]
    for _, frame in sampled:
        content.append({"type": "image_url", "image_url": {"url": _bgr_frame_to_data_url(frame)}})

    data = _chat_with_images(content, max_tokens=800, retries=retries)
    if not data or not isinstance(data, list):
        return []

    last_idx = len(sampled) - 1
    windows = []
    for item in data[:max_shots]:
        try:
            start_idx = max(0, min(int(item["start_index"]), last_idx))
            end_idx = max(0, min(int(item["end_index"]), last_idx))
        except (KeyError, ValueError, TypeError):
            continue

        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        if end_idx == start_idx:
            # Degenerate single-frame "window" from the model — nudge to a
            # real 2-frame span instead of dropping it, but never step outside
            # the sampled array (this is exactly what crashed before: blindly
            # doing start_idx + 1 without checking it stayed in bounds).
            if end_idx < last_idx:
                end_idx += 1
            elif start_idx > 0:
                start_idx -= 1
            else:
                continue  # only one sampled frame total, no window possible

        windows.append({"start_time": sampled[start_idx][0], "end_time": sampled[end_idx][0]})
    return windows


LABEL_PROMPT = """You are a professional cricket batting coach with 20+ years of experience coaching at club and national level.

I will show you 7 sequential frames of a batsman's shot capturing the full batting sequence:
Frame 1 = Initial stance
Frame 2 = Trigger movement
Frame 3 = Backlift start
Frame 4 = Full backlift (peak)
Frame 5 = Downswing
Frame 6 = Ball contact
Frame 7 = Follow-through

Analyze the technique across all 7 frames. Return ONLY valid JSON — no markdown, no explanation.

Return EXACTLY this structure:
{
  "shot_type": "One of: Cover Drive / Pull Shot / Square Cut / Defensive / Sweep / Straight Drive / Hook / Other",
  "batting_hand": "Right-handed" or "Left-handed",
  "skill_level": "One of: Professional, Amateur/Club, Youth/Academy",
  "strength": "1 sentence",
  "weakness": "1 sentence",
  "change_drill": "1 sentence — a specific drill to fix the weakness",
  "scores": {"balance": <0-100>, "power": <0-100>, "technique": <0-100>, "defence": <0-100>}
}

Be specific. Avoid generic advice like 'improve your stance'."""


def label_session_frames(session_name, frames_rgb):
    """
    frames_rgb: list of 7 RGB numpy arrays (the canonical stance->follow-through
    frames already extracted in-memory — nothing is saved to disk).
    Returns a dict matching labels.csv's schema, or None on failure.
    """
    content = [{"type": "text", "text": LABEL_PROMPT}]
    for frame in frames_rgb:
        content.append({"type": "image_url", "image_url": {"url": _rgb_array_to_data_url(frame)}})

    data = _chat_with_images(content, max_tokens=500)
    if not data:
        print(f"  [NVIDIA] Labelling failed for {session_name}, skipping label row")
        return None

    scores = data.get("scores", {})
    return {
        "session_name": session_name,
        "shot_type": data.get("shot_type", "Other"),
        "batting_hand": data.get("batting_hand", "Right-handed"),
        "skill_level": data.get("skill_level", "Amateur/Club"),
        "strength": data.get("strength", ""),
        "weakness": data.get("weakness", ""),
        "change_drill": data.get("change_drill", ""),
        "score_balance": int(scores.get("balance", 50) or 50),
        "score_power": int(scores.get("power", 50) or 50),
        "score_technique": int(scores.get("technique", 50) or 50),
        "score_defence": int(scores.get("defence", 50) or 50),
    }
