"""
phase3_batsman_audit.py — contact sheets that make the S0-S4 decisions
visually inspectable.

Phase 1's lesson was that a numerically plausible pipeline can be tracking
the wrong person and only LOOKING catches it. The same applies to a semantic
scorer: a system that refuses the right clips for the wrong reasons looks
identical in a table.

Renders, per selected clip: sampled frames with every candidate track boxed
and labelled with its S0 and S3 scores, the annotated batsman box, and a
panel giving each system's decision plus the evidence components that drove
it.
"""

import argparse
import json
import os

import cv2
import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

_GT = (255, 170, 0)
_SEL = (0, 235, 0)
_REJ = (150, 150, 150)
_BAD = (0, 0, 255)
_W = (255, 255, 255)


def _lab(img, t, org, c, s=0.42, th=1):
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, c, th, cv2.LINE_AA)


def _fit(img, w, h):
    if img is None or img.size == 0:
        return np.zeros((h, w, 3), np.uint8)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))))
    c = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    c[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return c


def render(cid, clip_tracks, gt_clip, feats, decisions, video_dir, out_dir,
           tile_w=300, tile_h=200):
    path = os.path.join(video_dir, cid)
    if not os.path.exists(path):
        return None
    gt_frames = [f for f in gt_clip["frames"] if f["batsman_visible"] and f["batsman_bbox"]]
    if not gt_frames:
        return None

    sel_s3 = decisions.get("S3_equipment", {}).get("picked_track")
    gt_tid = decisions.get("S0_geometry", {}).get("gt_track")

    cap = cv2.VideoCapture(path)
    want = {f["frame"] for f in gt_frames}
    grabbed, i = {}, -1
    while i < max(want):
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i in want:
            grabbed[i] = fr
    cap.release()

    tiles = []
    for frec in gt_frames:
        fr = grabbed.get(frec["frame"])
        if fr is None:
            continue
        vis = fr.copy()
        x1, y1, x2, y2 = [int(v) for v in frec["batsman_bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), _GT, 2)
        _lab(vis, "GT", (x1 + 2, max(12, y1 - 4)), _GT, 0.45)

        for tid, obs in sorted(clip_tracks.items()):
            box = next((b for f, b, _ in obs if f == frec["frame"]), None)
            if box is None:
                continue
            chosen = (tid == sel_s3)
            col = _SEL if chosen else _REJ
            bx = [int(v) for v in box]
            cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), col, 3 if chosen else 1)
            _lab(vis, f"t{tid}", (bx[0] + 2, min(vis.shape[0] - 4, bx[3] + 14)), col, 0.42)

        t = _fit(vis, tile_w, tile_h)
        bar = np.zeros((18, tile_w, 3), np.uint8)
        _lab(bar, f"f{frec['frame']}", (4, 13), _W, 0.4)
        tiles.append(np.vstack([bar, t]))

    if not tiles:
        return None
    cols = 3
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)

    # Decision + evidence panel.
    W = grid.shape[1]
    panel = np.full((max(190, 26 + 20 * (len(clip_tracks) + 6)), W, 3), 20, np.uint8)
    y = 20
    _lab(panel, f"{cid}   GT track = t{gt_tid}", (8, y), _W, 0.55, 2); y += 22
    for name, d in decisions.items():
        col = _SEL if d["outcome"] == "correct" else (_BAD if d["outcome"] == "wrong" else _REJ)
        _lab(panel, f"{name:<16} {d['outcome']:<8} pick=t{d['picked_track']}  "
                    f"conf={d['confidence']}  margin={d['margin']}", (8, y), col, 0.44)
        y += 19
    y += 6
    _lab(panel, f"{'track':<8}{'relh':>7}{'foot':>7}{'batOn':>7}{'batHand':>9}"
                f"{'bilat':>7}{'updown':>8}", (8, y), _W, 0.4); y += 17
    for tid in sorted(clip_tracks):
        f = feats.get(str(tid), {}).get("features", {})
        mark = " <-GT" if tid == gt_tid else (" <-S3" if tid == sel_s3 else "")
        col = _GT if tid == gt_tid else (_SEL if tid == sel_s3 else _REJ)
        _lab(panel, f"t{tid:<7}{f.get('relative_height',0):>7.3f}"
                    f"{f.get('foot_stability',0):>7.3f}{f.get('e_bat_on_person',0):>7.2f}"
                    f"{f.get('e_bat_near_hand',0):>9.2f}{f.get('k_bilateral',0):>7.3f}"
                    f"{f.get('a_updown_shape',0):>8.3f}{mark}", (8, y), col, 0.4)
        y += 17

    sheet = np.vstack([grid, panel])
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{os.path.splitext(cid)[0]}.jpg")
    cv2.imwrite(p, sheet)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--out", default=os.path.join(P3, "batsman_audit_sheets"))
    args = ap.parse_args()

    from phase2_cache_tracks import load_tracks
    from phase2_detector_benchmark import load_gt
    import phase3_batsman_ablation as AB

    tracks = load_tracks(os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    gt = {c["clip_id"]: c for c in load_gt()["clips"]}
    with open(os.path.join(P3, "phase3_semantic_features.json"), encoding="utf-8") as f:
        feats = json.load(f)["clips"]
    with open(os.path.join(P3, "phase3_batsman_ablation_results.json"), encoding="utf-8") as f:
        abl = json.load(f)
    with open(os.path.join(P3, "phase3_batsman_analysis.json"), encoding="utf-8") as f:
        ana = json.load(f)

    dec = {}
    for name in AB.SYSTEMS:
        for sp in ("dev", "eval"):
            for d in abl["results"][name][sp]["detail"]:
                dec.setdefault(d["clip_id"], {})[name] = d

    # Select what to audit: every eval failure, every feeder case, clips where
    # S3 changes the pick vs S0, plus a few clean successes for contrast.
    want, why = set(), {}
    for cid, ds in dec.items():
        if ds.get("S0_geometry", {}).get("split") == "eval" or \
           ds.get("S3_equipment", {}).get("split") == "eval":
            for n in ("S0_geometry", "S3_equipment"):
                if ds.get(n, {}).get("outcome") in ("wrong", "refused"):
                    want.add(cid); why.setdefault(cid, []).append(f"{n}:{ds[n]['outcome']}")
        if ds.get("S0_geometry", {}).get("picked_track") != ds.get("S3_equipment", {}).get("picked_track"):
            want.add(cid); why.setdefault(cid, []).append("S3 changed the pick")
    for h in ana["stationary_feeder_clips"]:
        want.add(h["clip_id"]); why.setdefault(h["clip_id"], []).append("stationary feeder")
    successes = [c for c, ds in dec.items()
                 if ds.get("S3_equipment", {}).get("outcome") == "correct"][:4]
    for c in successes:
        want.add(c); why.setdefault(c, []).append("representative success")

    n = 0
    for cid in sorted(want):
        if cid not in tracks["clips"] or cid not in gt or cid not in feats:
            continue
        p = render(cid, tracks["clips"][cid]["tracks"], gt[cid],
                   feats[cid]["tracks"], dec[cid], args.videos, args.out)
        if p:
            n += 1
            print(f"{cid}: {', '.join(why[cid])}")
    print(f"\nwrote {n} audit sheets to {args.out}")


if __name__ == "__main__":
    main()
