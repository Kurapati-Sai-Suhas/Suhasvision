"""
phase2_shot_localization.py — should shot localization run BEFORE or AFTER
batsman identification? (Phase-2 spec §22)

THE HYPOTHESIS
Global-frame motion energy cannot distinguish the batsman's stroke from the
bowler's run-up and delivery. Once the batsman's track is known, motion
measured INSIDE that track's box isolates the stroke and should localize it
more sharply.

WHAT CAN AND CANNOT BE CONCLUDED HERE
There are no annotated shot boundaries or contact frames in this benchmark,
so localization ACCURACY is not computable and is not claimed. What IS
computable, and is what this script reports:

  peak_disagreement  |argmax(global motion) - argmax(batsman motion)| in
                     frames. Large disagreement means the two signals are
                     measuring different events -- and only one of them can
                     be the batsman's stroke.
  prominence         peak / median of each signal. A sharper peak is a more
                     usable localization signal regardless of which frame is
                     correct.
  contamination test disagreement should be LARGE on multi-person clips and
                     SMALL on single-person clips. That is a falsifiable
                     prediction: if global motion were not bowler-contaminated,
                     the two groups would look the same.

The contamination test is the real experiment. It does not need shot
boundaries, because it asks whether the global signal tracks the wrong person,
not whether it finds the right frame.
"""

import argparse
import json
import os

import cv2
import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_MODULE_DIR, "phase2_results")


def _prominence(sig):
    sig = np.asarray(sig, dtype=np.float64)
    if sig.size < 3:
        return None
    med = float(np.median(sig))
    return round(float(sig.max()) / med, 3) if med > 1e-9 else None


def clip_signals(video_path, batsman_obs, max_frames=400):
    """Global and batsman-restricted motion-energy profiles over one clip.

    Both are mean absolute frame difference; the only difference is the
    region they are computed over, so the comparison isolates WHERE motion is
    measured rather than HOW.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n > max_frames:
        cap.release()
        return None                     # long session clips are not single shots

    box_by_frame = {f: b for f, b, _ in batsman_obs}
    prev_gray = prev_crop = None
    frames, g_energy, b_energy = [], [], []

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        box = box_by_frame.get(i)
        crop = None
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
            if x2 > x1 and y2 > y1:
                # Resized to a fixed size so the energy is not a function of
                # how large the box happens to be in that frame.
                crop = cv2.resize(gray[y1:y2, x1:x2], (64, 128))

        if prev_gray is not None:
            g_energy.append(float(np.mean(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)))))
            if crop is not None and prev_crop is not None:
                b_energy.append(float(np.mean(np.abs(crop.astype(np.int16) - prev_crop.astype(np.int16)))))
            else:
                b_energy.append(np.nan)
            frames.append(i)

        prev_gray = gray
        if crop is not None:
            prev_crop = crop
    cap.release()

    if len(frames) < 5:
        return None
    return {"frames": frames, "global": g_energy, "batsman": b_energy}


def analyse(gt, tracks_cache, model, video_dir):
    from phase2_batsman_score import select_batsman

    rows = []
    for clip in gt["clips"]:
        cid = clip["clip_id"]
        c = tracks_cache["clips"].get(cid)
        if not c:
            continue
        ctx = {"frame_w": c["frame_w"], "frame_h": c["frame_h"],
               "frames_processed": c["frames_processed"]}
        sel = select_batsman(c["tracks"], ctx, model)
        tid = sel.get("track_id")
        if tid is None:
            continue

        sig = clip_signals(os.path.join(video_dir, cid), c["tracks"][tid])
        if sig is None:
            continue

        g = np.asarray(sig["global"])
        b = np.asarray(sig["batsman"], dtype=np.float64)
        valid = ~np.isnan(b)
        if valid.sum() < 5:
            continue

        gi = int(np.argmax(g))
        bi_local = int(np.nanargmax(np.where(valid, b, -np.inf)))
        peak_global = sig["frames"][gi]
        peak_bats = sig["frames"][bi_local]

        rows.append({
            "clip_id": cid,
            "split": clip["split"],
            "n_tracks": len(c["tracks"]),
            "multi_person": len(c["tracks"]) > 1,
            "peak_global": peak_global,
            "peak_batsman": peak_bats,
            "disagreement_frames": abs(peak_global - peak_bats),
            "prominence_global": _prominence(g),
            "prominence_batsman": _prominence(b[valid]),
            "batsman_coverage": round(float(valid.sum()) / len(b), 3),
        })
    return rows


def summarise(rows):
    def stats(sub, key):
        vals = [r[key] for r in sub if r[key] is not None]
        if not vals:
            return None
        vals = sorted(vals)
        return {"n": len(vals), "mean": round(sum(vals) / len(vals), 2),
                "median": vals[len(vals) // 2]}

    multi = [r for r in rows if r["multi_person"]]
    single = [r for r in rows if not r["multi_person"]]
    return {
        "n_clips": len(rows),
        "all": {"disagreement": stats(rows, "disagreement_frames"),
                "prominence_global": stats(rows, "prominence_global"),
                "prominence_batsman": stats(rows, "prominence_batsman")},
        "multi_person": {"n": len(multi), "disagreement": stats(multi, "disagreement_frames")},
        "single_person": {"n": len(single), "disagreement": stats(single, "disagreement_frames")},
        "note": ("Localization ACCURACY is not computed: this benchmark has no "
                 "annotated shot boundaries or contact frames. Disagreement and "
                 "peak prominence are computable; which peak is correct is not."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--tracks", default=os.path.join(RESULTS, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--model", default=os.path.join(RESULTS, "phase2_batsman_model.json"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "phase2_shot_localization.json"))
    args = ap.parse_args()

    from phase2_cache_tracks import load_tracks
    from phase2_detector_benchmark import load_gt

    gt = load_gt()
    tracks = load_tracks(args.tracks)
    with open(args.model, encoding="utf-8") as f:
        model = json.load(f)["model"]

    rows = analyse(gt, tracks, model, args.videos)
    summary = summarise(rows)

    print(json.dumps(summary, indent=1))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_clip": rows}, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
