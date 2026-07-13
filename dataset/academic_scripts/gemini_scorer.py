"""
gemini_scorer.py
Zero-shot biomechanical stance scoring baseline using Gemini Vision.

This is the second arm of the 4-way comparison the paper needs (rule-based
vs Gemini zero-shot vs custom LSTM vs human ground truth). It reuses the
same google-genai client pattern as zero_storage_pipeline.py, but asks
Gemini to directly SCORE the stance from frames, rather than just locate
shot timestamps.

USAGE
    python gemini_scorer.py --frames_dir frames/<session_id> --session_id <id>

Appends one row to gemini_scores.csv per session, scored on the same six
dimensions as the rule-based scorer so the two are directly comparable.
"""

import argparse
import json
import os
import time
import pandas as pd
from PIL import Image
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = """You are an expert cricket biomechanics coach. You will be
shown a sequence of frames from a single batting stance. Score the stance on
exactly these six dimensions, each from 0-100, using the following criteria:

- head_position: head level, eyes parallel to ground, minimal lateral drift
- foot_placement: front/back foot spacing 80-120% of shoulder width, appropriate opening angle
- knee_bend: front knee flexion in the 150-165 degree range at address
- bat_angle: bat face near-vertical (85-95 degrees from ground) at address
- hip_rotation: appropriate hip-shoulder separation for the shot type
- weight_distribution: balance appropriate for a defensive vs attacking shot

Respond ONLY with valid JSON in this exact schema, no other text:
{"head_position": int, "foot_placement": int, "knee_bend": int,
 "bat_angle": int, "hip_rotation": int, "weight_distribution": int,
 "overall_score": int, "reasoning": "one concise sentence per dimension"}
"""


def score_session(frames_dir: str, session_id: str, max_retries: int = 2) -> dict:
    image_paths = sorted(
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    images = [Image.open(p) for p in image_paths]
    contents = [SYSTEM_PROMPT] + images

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config={"response_mime_type": "application/json"},
            )
            result = json.loads(response.text)
            result["session_id"] = session_id
            return result
        except (json.JSONDecodeError, KeyError) as e:
            if attempt < max_retries:
                print(f"  Retry {attempt+1}/{max_retries} for {session_id}: {e}")
                time.sleep(2)
            else:
                print(f"  FAILED after {max_retries} retries: {session_id}")
                return {"session_id": session_id, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--session_id", required=True)
    ap.add_argument("--output", default="gemini_scores.csv")
    args = ap.parse_args()

    result = score_session(args.frames_dir, args.session_id)
    df_new = pd.DataFrame([result])

    if os.path.exists(args.output):
        df_existing = pd.read_csv(args.output)
        df_existing = df_existing[df_existing.session_id != args.session_id]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(args.output, index=False)
    print(f"Saved Gemini score for {args.session_id} to {args.output}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
