"""
phase3e_blind_annotate.py — build a TRULY blind annotation pack.

WHY THIS EXISTS
Every contact result so far rests on labels made by someone who could see the
C0 proposal drawn on the sheet. `phase3e_leakage_analysis` argues against
strong leakage, but it cannot prove independence because both annotation
passes saw the same marker. Only labels made without ever seeing a machine
estimate can settle it.

WHAT THIS TOOL GUARANTEES
The sheets it renders contain NO machine information of any kind:

    no C0 contact proposal        no L3 interval
    no F4 selected frames         no event search region
    no track boxes, no skeletons, no phase markers

Only the video frames, a frame index per tile, and a batsman crop. The
subset is chosen by an ADVERSARIAL stratification (below) so the pack is not
quietly biased toward easy clips.

WHAT IT CANNOT DO
It cannot annotate. A human must fill in the answer file. Nothing in this
repository may generate `phase3e_blind_annotations.jsonl` automatically — a
model-authored "blind" label is neither blind nor human, and would silently
destroy the only clean evidence available for this question.
"""

import argparse
import json
import os

import cv2
import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

STRATA = ("strong_C0", "weak_C0", "identity_correct", "identity_failure",
          "front_view", "side_view", "occlusion", "high_motion",
          "background_heavy")


def choose_subset(cache, recs, contact_est, n_target=12):
    """Adversarial stratification: deliberately include the cases most likely
    to break C0, not a random sample that would flatter it."""
    import numpy as np

    cands = []
    for cid, c in cache["clips"].items():
        r = recs.get(cid, {})
        frames = c["frames"]
        occl = [(f["pose"] or {}).get("min_visibility") for f in frames
                if f.get("pose")]
        mot = [f["local_motion"] for f in frames if f.get("local_motion") is not None]
        gt = r.get("contact_frame")
        c0 = contact_est.get(cid)
        err = abs(c0 - gt) if (c0 is not None and gt is not None) else None
        cands.append({
            "clip_id": cid, "split": c["split"],
            "identity_correct": bool(c["identity_correct"]),
            "coherence": r.get("coherence"),
            "contact_type": r.get("contact_type"),
            "c0_error_vs_original": err,
            "mean_min_visibility": (round(float(np.mean(occl)), 3) if occl else None),
            "mean_local_motion": (round(float(np.mean(mot)), 3) if mot else None),
            "view": ("front" if "front" in cid else
                     "side" if "side" in cid else
                     "back" if "back" in cid else "unknown"),
        })

    picked, why = [], {}

    def take(pool, tag, k=1):
        for c in pool:
            if len(picked) >= n_target:
                return
            if c["clip_id"] in why:
                why[c["clip_id"]].append(tag)
                continue
            picked.append(c)
            why[c["clip_id"]] = [tag]
            k -= 1
            if k <= 0:
                return

    scored = [c for c in cands if c["c0_error_vs_original"] is not None]
    take(sorted(scored, key=lambda c: c["c0_error_vs_original"]), "strong_C0", 2)
    take(sorted(scored, key=lambda c: -c["c0_error_vs_original"]), "weak_C0", 2)
    take([c for c in cands if not c["identity_correct"]], "identity_failure", 2)
    take([c for c in cands if c["identity_correct"]], "identity_correct", 1)
    take([c for c in cands if c["view"] == "front"], "front_view", 1)
    take([c for c in cands if c["view"] == "side"], "side_view", 1)
    take(sorted([c for c in cands if c["mean_min_visibility"] is not None],
                key=lambda c: c["mean_min_visibility"]), "occlusion", 1)
    take(sorted([c for c in cands if c["mean_local_motion"] is not None],
                key=lambda c: -c["mean_local_motion"]), "high_motion", 1)
    take([c for c in cands if c["coherence"] != "VALID_SINGLE_SHOT"],
         "background_heavy_or_invalid", 1)
    for c in picked:
        c["strata"] = why[c["clip_id"]]
    return picked


def render_sheet(clip_id, video_dir, cache_clip, out_dir, cols=8, tile=190):
    """Frames with index labels and a batsman crop. NO machine markers."""
    path = os.path.join(video_dir, clip_id)
    if not os.path.exists(path):
        return None
    by_frame = {f["frame"]: f for f in cache_clip["frames"]}
    cap = cv2.VideoCapture(path)
    tiles, i = [], -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        rec = by_frame.get(i)
        crop = fr
        if rec and rec.get("box"):
            b = rec["box"]
            h, w = fr.shape[:2]
            bw, bh = b[2] - b[0], b[3] - b[1]
            # Generous horizontal padding: bat and ball live outside the person
            # box, and the annotator needs to see them to judge contact.
            x1 = max(0, int(b[0] - 0.9 * bw)); y1 = max(0, int(b[1] - 0.35 * bh))
            x2 = min(w, int(b[2] + 0.9 * bw)); y2 = min(h, int(b[3] + 0.35 * bh))
            if x2 > x1 and y2 > y1:
                crop = fr[y1:y2, x1:x2]
        ih, iw = crop.shape[:2]
        s = tile / max(ih, 1)
        t = cv2.resize(crop, (max(1, int(iw * s)), tile))
        t = t[:, :tile] if t.shape[1] > tile else np.hstack(
            [t, np.zeros((tile, tile - t.shape[1], 3), np.uint8)])
        bar = np.zeros((16, tile, 3), np.uint8)
        cv2.putText(bar, f"f{i}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, t]))
    cap.release()
    if not tiles:
        return None
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    hdr = np.zeros((30, sheet.shape[1], 3), np.uint8)
    cv2.putText(hdr, f"{clip_id}   BLIND SHEET - no machine markers",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{os.path.splitext(clip_id)[0]}.jpg")
    cv2.imwrite(p, np.vstack([hdr, sheet]))
    return p


TEMPLATE_HEADER = """# Blind annotation answer file — Phase 3E
#
# Fill ONE line per clip, JSON per line. Do not open any other phase3 result
# file, and do not look at the existing annotations, before finishing.
#
# blind_contact_type : EXACT | CONTACT_ADJACENT | AMBIGUOUS | NOT_VISIBLE
# blind_contact_frame: integer, or null for AMBIGUOUS / NOT_VISIBLE
# blind_confidence   : HIGH | MEDIUM | LOW
# blind_shot_valid   : true/false
# blind_shot_start/end: integers, or null when not a valid single shot
#
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--cache", default=os.path.join(P3, "phase3_event_cache.json"))
    ap.add_argument("--bench", default=os.path.join(P3, "phase3_shot_localization_corrected.json"))
    ap.add_argument("--out-dir", default=os.path.join(P3, "blind_sheets"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--render", action="store_true",
                    help="render the sheets (slow); omit to only pick the subset")
    args = ap.parse_args()

    import phase3_contact_benchmark as CB
    cache, recs, _ = CB.load_context(args.cache, args.bench)
    est = {cid: CB.c0_proposal(cid) for cid in cache["clips"]}

    subset = choose_subset(cache, recs, est, args.n)
    manifest = os.path.join(P3, "phase3e_blind_subset.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"n": len(subset),
                   "strata_targeted": list(STRATA),
                   "selection": ("adversarial stratification — deliberately "
                                 "includes the weakest C0 predictions and "
                                 "identity failures, not a random sample"),
                   "clips": subset}, f, indent=1)

    tmpl = os.path.join(P3, "phase3e_blind_annotations.TEMPLATE.jsonl")
    with open(tmpl, "w", encoding="utf-8") as f:
        f.write(TEMPLATE_HEADER)
        for c in subset:
            f.write(json.dumps({
                "clip_id": c["clip_id"], "annotator": "REPLACE_ME",
                "blind_contact_type": None, "blind_contact_frame": None,
                "blind_confidence": None, "blind_shot_valid": None,
                "blind_shot_start": None, "blind_shot_end": None,
                "notes": ""}) + "\n")

    print(f"selected {len(subset)} clips for blind annotation:")
    for c in subset:
        print(f"  {c['clip_id']:<30} {','.join(c['strata']):<28} "
              f"view={c['view']:<7} idOK={c['identity_correct']} "
              f"c0_err={c['c0_error_vs_original']}")
    print(f"\nwrote {manifest}")
    print(f"wrote {tmpl}")
    if args.render:
        n = 0
        for c in subset:
            p = render_sheet(c["clip_id"], args.videos,
                             cache["clips"][c["clip_id"]], args.out_dir)
            if p:
                n += 1
        print(f"rendered {n} blind sheets to {args.out_dir}")
    else:
        print("\n(re-run with --render to produce the sheets)")


if __name__ == "__main__":
    main()
