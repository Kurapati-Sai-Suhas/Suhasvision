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


def find_shot_windows(video_path, duration, max_samples=MAX_IMAGES_PER_PROMPT, retries=3, max_shots=20, start_offset=0.0):
    """
    Scans `video_path` from `start_offset` to `start_offset + duration` seconds,
    sampling up to `max_samples` EVENLY-SPACED frames across that window
    (not a fixed fps — the NVIDIA endpoint hard-rejects prompts with more than
    12 images, so sample count must stay constant regardless of window length).
    Returns a LIST of {"start_time": float, "end_time": float} dicts — one per
    distinct batting shot found. Times are relative to `video_path`'s own
    0-based timeline (i.e. absolute, including `start_offset`), matching what
    extract_keypoints_in_memory expects — a caller passing start_offset=0
    (the original, and still default, behavior) sees identical output to
    before this parameter existed.

    start_offset lets a single long, untrimmed video be scanned in tiled
    chunks (see find_shot_windows_auto) without needing a separately-cut
    macro-trimmed file per chunk — the seek/sample math below is offset by
    it, everything else (prompting, index math, degenerate-window handling)
    is unchanged from the macro-trimmed-clip case.

    Trade-off: longer windows get coarser temporal resolution (10 samples
    across 90s is one frame every 9s). Keep each scanned window tight
    (15-30s) for good shot-boundary precision — find_shot_windows_auto does
    this automatically for a full video; a human-provided macro window in
    batch_urls.csv should follow the same guidance.
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

    offset_frames = int(round(start_offset * fps))
    sampled = []
    for frame_num in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, offset_frames + frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        sampled.append((start_offset + frame_num / fps, frame))
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


# ---------------------------------------------------------------------------
# Automatic macro-window detection (removes the manual macro_start_sec /
# macro_end_sec curation step from batch_urls.csv).
#
# WHY THIS EXISTS: until now, a human had to watch each source video first
# and hand-write a tight (15-30s) macro window before find_shot_windows()
# could run at its documented precision. That's a real, unautomated
# bottleneck distinct from (and unrelated to) the wrong-person-tracking
# defect (Milestone 4, subject_selection.py) -- confirmed by direct code
# inspection: nothing in this file or zero_storage_pipeline.py computed
# macro_start_sec/macro_end_sec; they were read verbatim from the CSV.
#
# HOW: tile the FULL video into overlapping windows no longer than the same
# 15-30s this module's docstrings already recommend for good shot-boundary
# resolution, run the existing, already-validated find_shot_windows() on
# each tile (via start_offset, so no per-tile temp file is cut), and merge
# any windows detected in the overlap between adjacent tiles. The overlap
# exists specifically so a shot straddling a tile boundary is still fully
# visible to at least one tile's sample set, rather than being split or
# missed -- the merge step then collapses the resulting duplicate/adjacent
# detections back into one real shot.
#
# This reuses find_shot_windows() completely unchanged in its actual
# detection logic (prompting, index math, degenerate-window handling) --
# only the *span of video it's pointed at* is new, exactly the same
# discipline this project used for every prior extension (relocate/reuse,
# verify equivalence, don't reimplement).
# ---------------------------------------------------------------------------

DEFAULT_SCAN_TILE_SECONDS = 25.0   # within this module's own "keep macro windows 15-30s" guidance
DEFAULT_SCAN_OVERLAP_SECONDS = 5.0  # must exceed a real shot's expected duration (~1-3.5s) with margin


def build_scan_tiles(total_duration, tile_seconds=DEFAULT_SCAN_TILE_SECONDS, overlap_seconds=DEFAULT_SCAN_OVERLAP_SECONDS):
    """
    Pure arithmetic, no video/NVIDIA dependency -- unit-testable in isolation
    (this project's established pattern for anything that decides frame/time
    indices, after the paired-fold and phase-redistribution bugs earlier in
    this codebase's history were both found in untested index arithmetic).

    Returns a list of (tile_start, tile_end) float tuples covering
    [0, total_duration] with `overlap_seconds` of overlap between
    consecutive tiles, each tile at most `tile_seconds` long. The final tile
    is clamped to total_duration rather than padded past it.

    Raises ValueError if overlap_seconds >= tile_seconds (a non-advancing or
    negative-progress tiling would loop forever / silently skip video).
    """
    if tile_seconds <= 0 or overlap_seconds < 0:
        raise ValueError("tile_seconds must be > 0 and overlap_seconds must be >= 0")
    if overlap_seconds >= tile_seconds:
        raise ValueError(f"overlap_seconds ({overlap_seconds}) must be < tile_seconds ({tile_seconds})")
    if total_duration <= 0:
        return []

    step = tile_seconds - overlap_seconds
    tiles = []
    start = 0.0
    while start < total_duration:
        end = min(start + tile_seconds, total_duration)
        tiles.append((start, end))
        if end >= total_duration:
            break
        start += step
    return tiles


def merge_overlapping_windows(windows):
    """
    Pure arithmetic. Given a list of {"start_time", "end_time"} dicts
    (potentially containing the same real shot detected twice from two
    overlapping tiles), sorts by start_time and merges any pair whose time
    ranges overlap AT ALL into their union -- two independent detections of
    the same shot from adjacent tiles will always overlap substantially
    (the shot itself is only ~1-3.5s; the tiles it appears in share
    DEFAULT_SCAN_OVERLAP_SECONDS=5s), while two genuinely different shots in
    the same video are expected not to overlap at all. Returns a new,
    merged, start_time-sorted list.
    """
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: w["start_time"])
    merged = [dict(ordered[0])]
    for w in ordered[1:]:
        last = merged[-1]
        if w["start_time"] <= last["end_time"]:
            last["end_time"] = max(last["end_time"], w["end_time"])
            last["start_time"] = min(last["start_time"], w["start_time"])
        else:
            merged.append(dict(w))
    return merged


def find_shot_windows_auto(video_path, total_duration, tile_seconds=DEFAULT_SCAN_TILE_SECONDS,
                            overlap_seconds=DEFAULT_SCAN_OVERLAP_SECONDS, max_samples=MAX_IMAGES_PER_PROMPT,
                            retries=3, max_shots_per_tile=20):
    """
    The no-human-macro-window entry point: scans an ENTIRE, untrimmed video
    (any length) for batting shots, with no macro_start_sec/macro_end_sec
    required. Tiles the full duration (build_scan_tiles), runs the existing
    find_shot_windows() on each tile via start_offset (no per-tile temp file
    cut needed), and merges detections spanning tile overlaps
    (merge_overlapping_windows).

    One NVIDIA call per tile -- a 10-minute video at the default 25s tiles /
    5s overlap is (600-5)/(25-5) + 1 = ~30 calls, versus 1 call for a
    human-pre-trimmed 25s macro window. This is the real, accepted cost of
    removing the manual step; callers processing many/long videos should be
    aware it consumes proportionally more free-tier credits (see
    CreditsExhaustedError) and budget accordingly.

    Returns the same shape as find_shot_windows(): a list of
    {"start_time", "end_time"} dicts, absolute to video_path's own timeline,
    ready to pass straight into extract_keypoints_in_memory exactly like a
    human-provided macro window's shots would be.
    """
    tiles = build_scan_tiles(total_duration, tile_seconds, overlap_seconds)
    all_windows = []
    for tile_start, tile_end in tiles:
        windows = find_shot_windows(
            video_path, duration=tile_end - tile_start, max_samples=max_samples,
            retries=retries, max_shots=max_shots_per_tile, start_offset=tile_start,
        )
        all_windows.extend(windows)
    return merge_overlapping_windows(all_windows)


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
    # Confirmed real bug (2026-07-18): defaulting a missing/malformed score
    # to 50 silently fabricated data -- 90 of 496 real labels.csv rows
    # ended up with ALL FOUR scores exactly 50 this way, indistinguishable
    # from a genuine (if oddly uniform) assessment unless you go looking.
    # A missing or non-numeric score means the model's response didn't
    # actually contain a usable assessment -- treat that the same as the
    # existing "no `data` at all" failure path (skip the label row) rather
    # than inventing a number.
    parsed_scores = {}
    for key in ("balance", "power", "technique", "defence"):
        try:
            parsed_scores[key] = int(scores[key])
        except (KeyError, TypeError, ValueError):
            print(f"  [NVIDIA] Labelling for {session_name} was missing/invalid score '{key}' -- skipping label row rather than guessing 50")
            return None

    return {
        "session_name": session_name,
        "shot_type": data.get("shot_type", "Other"),
        "batting_hand": data.get("batting_hand", "Right-handed"),
        "skill_level": data.get("skill_level", "Amateur/Club"),
        "strength": data.get("strength", ""),
        "weakness": data.get("weakness", ""),
        "change_drill": data.get("change_drill", ""),
        "score_balance": parsed_scores["balance"],
        "score_power": parsed_scores["power"],
        "score_technique": parsed_scores["technique"],
        "score_defence": parsed_scores["defence"],
    }
