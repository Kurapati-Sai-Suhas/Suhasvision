"""
auto_label_gemini.py — Cricket Stance Analyzer
Fixes: updated to new google.genai SDK, optimized 4-factor prompt,
       reads frames from LOCAL folder (populated by extract_frames.py),
       proper rate limiting, resume support.
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai                     # ← NEW SDK (matches extract_frames.py)
from google.genai import types

load_dotenv(override=True)

# ── Config ────────────────────────────────────────────────────────────
FRAMES_DIR  = "frames"          # local folder populated by extract_frames.py
LABEL_FILE  = "labels.csv"
GEMINI_MODEL = "gemini-2.0-flash"   # use flash for labelling — cheaper quota
SLEEP_SEC   = 6                 # delay between sessions (free tier: ~10 req/min)
SLEEP_ON_ERROR = 65             # delay on 429

# ── Client ────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("🚨 Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

# ── Optimized Prompt ──────────────────────────────────────────────────
# Based on project spec: 4-factor output with delivery vulnerability,
# specific drills, frame references, and structured JSON.
PROMPT = """You are a professional cricket batting coach with 20+ years of experience coaching at club and national level.

I will show you 7 sequential frames of a batsman's shot capturing the full batting sequence:
Frame 1 = Initial stance
Frame 2 = Trigger movement
Frame 3 = Backlift start
Frame 4 = Full backlift (peak)
Frame 5 = Downswing
Frame 6 = Ball contact
Frame 7 = Follow-through

Analyze the technique across all 7 frames. Return ONLY valid JSON — no markdown, no explanation, just the JSON object.

Return EXACTLY this structure:
{
  "shot_type": "One of: Cover Drive / Pull Shot / Square Cut / Defensive / Sweep / Straight Drive / Hook / Other",
  "batting_hand": "Right-handed" or "Left-handed",
  "skill_level": "One of: Professional, Amateur/Club, Youth/Academy",
  "weakness": [
    "Frame X: [specific flaw observed] — this causes [exact consequence, e.g. 'vulnerable to inswing on off stump because bat face is open at contact']"
  ],
  "strength": [
    "Frame X: [specific positive element] — this generates [exact benefit, e.g. 'power through the line because weight transfers fully onto front foot at Frame 6']"
  ],
  "change": [
    "[Specific body part] + [exact target position] + [drill to fix it, e.g. 'Redirect backlift so bat face points to fine leg at Frame 4 — drill: place a stump at fine leg and tap it on every practice backlift']"
  ],
  "bonus": [
    "Any additional coaching observation not covered above — trigger movement direction, grip pressure signs, balance issues visible across frames"
  ],
  "scores": {
    "balance": 0,
    "power": 0,
    "technique": 0,
    "defence": 0
  },
  "delivery_vulnerability": [
    "Based on detected weaknesses: [specific delivery type that will exploit the flaw, e.g. 'Full inswing from over the wicket — will find the gap between bat and pad given the open stance in Frame 1']"
  ],
  "player_type": "One of: aggressive-opener / classical-technician / defensive-anchor / lower-order-hitter / unknown"
}

Scoring guide (0-100):
- balance: stability of base and head position across all 7 frames
- power: weight transfer, hip rotation, bat speed indicators
- technique: correctness of grip, backlift, bat path, contact position
- defence: ability to cover stumps, bat angle at contact, head over ball

skill_level guide: judge from technique polish and context visible in-frame (kit quality, ground setting, camera setup), not just execution quality.

Be specific. Reference exact frame numbers. Avoid generic advice like 'improve your stance'."""


# ── Helpers ───────────────────────────────────────────────────────────

def load_frames(session_dir: Path) -> list:
    """Load all JPEGs from session folder as PIL Images, sorted by phase."""
    frames = []
    for fp in sorted(session_dir.glob("*.jpg")):
        try:
            frames.append(Image.open(str(fp)).convert("RGB"))
        except Exception as e:
            print(f"  [WARN] Could not open {fp.name}: {e}")
    return frames


def call_gemini_with_retry(images: list, session_name: str) -> dict | None:
    """Call Gemini with images. Handles 429 with backoff. Returns parsed dict or None."""
    for attempt in range(1, 4):
        try:
            # Build content: prompt first, then all images
            contents = [PROMPT] + images

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
            )

            raw = response.text.strip()

            # Strip markdown if Gemini adds it despite instructions
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]

            return json.loads(raw.strip())

        except json.JSONDecodeError:
            print(f"  [WARN] Bad JSON from Gemini for {session_name} (attempt {attempt})")
            print(f"         Raw response: {response.text[:300]}")
            time.sleep(5)

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = SLEEP_ON_ERROR * attempt
                print(f"  [429] Quota hit for {session_name}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [ERROR] {session_name}: {e}")
                return None

    return None


def label_session(session: Path) -> dict | None:
    """Load frames and get Gemini labels for one session."""
    images = load_frames(session)

    if len(images) == 0:
        print(f"  [SKIP] No frames found in {session.name}")
        return None

    if len(images) < 4:
        print(f"  [WARN] Only {len(images)} frames in {session.name} — labelling anyway")

    result = call_gemini_with_retry(images, session.name)
    if not result:
        return None

    # Extract scores safely with defaults
    scores = result.get("scores", {})

    return {
        # Column names match the schema merge.py/feature_engineering.py already
        # expect (session_name, shot_type, batting_hand, skill_level,
        # score_balance/power/technique/defence) — the previous version of this
        # script used "session"/"balance"/etc., which is incompatible with the
        # existing labels.csv on disk and crashes run()'s resume check.
        "session_name"          : session.name,
        "shot_type"             : result.get("shot_type", "Other"),
        "batting_hand"          : result.get("batting_hand", "Right-handed"),
        "skill_level"           : result.get("skill_level", "Amateur/Club"),
        "weakness"              : json.dumps(result.get("weakness", [])),
        "strength"              : json.dumps(result.get("strength", [])),
        "change_drill"          : json.dumps(result.get("change", [])),
        "bonus"                 : json.dumps(result.get("bonus", [])),
        "delivery_vulnerability": json.dumps(result.get("delivery_vulnerability", [])),
        "player_type"           : result.get("player_type", "unknown"),
        "score_balance"         : scores.get("balance",   50),
        "score_power"           : scores.get("power",     50),
        "score_technique"       : scores.get("technique", 50),
        "score_defence"         : scores.get("defence",   50),
    }


# ── Main ──────────────────────────────────────────────────────────────
def run():
    # Resume support — load already-labelled sessions
    done, rows = set(), []
    if Path(LABEL_FILE).exists():
        existing = pd.read_csv(LABEL_FILE)
        done     = set(existing["session_name"])
        rows     = existing.to_dict("records")
        print(f"Resuming — {len(done)} already labelled, skipping those")

    sessions = [
        s for s in sorted(Path(FRAMES_DIR).iterdir())
        if s.is_dir() and s.name not in done
    ]

    if not sessions:
        print("All sessions already labelled! Check labels.csv")
        return

    print(f"Found {len(sessions)} sessions to label with {GEMINI_MODEL}")
    print(f"Estimated time: {len(sessions) * (SLEEP_SEC + 8) // 60} minutes\n")

    success = 0
    failed  = 0

    for session in tqdm(sessions, desc="Labelling"):
        result = label_session(session)

        if result:
            rows.append(result)
            pd.DataFrame(rows).to_csv(LABEL_FILE, index=False)   # save after every session
            success += 1
        else:
            failed += 1

        # Rate limit delay
        time.sleep(SLEEP_SEC)

    print(f"\n✓ Done!")
    print(f"  Labelled  : {success} sessions")
    print(f"  Failed    : {failed} sessions")
    print(f"  Saved to  : {LABEL_FILE}")
    print(f"\nNew columns vs old version:")
    print(f"  + delivery_vulnerability — what bowling attack exploits each flaw")
    print(f"  + player_type            — batsman archetype classification")


if __name__ == "__main__":
    run()