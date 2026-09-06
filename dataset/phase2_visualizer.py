"""
phase2_visualizer.py — annotated sheets for the detector+tracker+selection
stack.

Phase 1's lesson was that a numerically plausible pipeline can be tracking the
wrong person, and only LOOKING catches it. Phase 2 adds tracking and a scoring
model, both of which can fail in ways that aggregate metrics smooth over, so
the same discipline applies to the new stage.

Each sheet shows, per sampled frame:
  * every detected person, boxed and labelled with its track id
  * the SELECTED batsman track in green, rejected candidates in grey
  * the ground-truth batsman box in blue, when annotated
  * per-track batsman score and the final verdict
and, once per clip, each track's trajectory drawn on a single panel so
identity drift is visible at a glance.
"""

import argparse
import os

import cv2
import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_MODULE_DIR, "phase2_results")

_SELECTED = (0, 230, 0)
_REJECTED = (150, 150, 150)
_GT = (255, 160, 0)
_WARN = (0, 0, 255)
_WHITE = (255, 255, 255)


def _label(img, text, org, colour, scale=0.45, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _fit(img, w, h):
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return np.zeros((h, w, 3), np.uint8)
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas


def trajectory_panel(tracks, frame_w, frame_h, selected_tid, w=420, h=260):
    """Every track's centre path in one panel. An identity switch shows up as
    a path that teleports between two people."""
    panel = np.full((h, w, 3), 22, np.uint8)
    sx, sy = w / max(frame_w, 1), h / max(frame_h, 1)
    for tid, obs in sorted(tracks.items()):
        pts = [(int(((b[0] + b[2]) / 2) * sx), int(((b[1] + b[3]) / 2) * sy))
               for _, b, _ in sorted(obs)]
        if len(pts) < 2:
            continue
        colour = _SELECTED if tid == selected_tid else _REJECTED
        for a, b in zip(pts, pts[1:]):
            cv2.line(panel, a, b, colour, 2 if tid == selected_tid else 1, cv2.LINE_AA)
        cv2.circle(panel, pts[0], 4, colour, -1)
        _label(panel, f"t{tid}", (pts[0][0] + 5, pts[0][1] - 5), colour, 0.4)
    _label(panel, "track trajectories (green = selected)", (6, 16), _WHITE, 0.42)
    return panel


def render(video_path, clip_tracks, ctx, gt_frames, selection, out_dir,
           tile_w=330, tile_h=210):
    tracks = clip_tracks["tracks"]
    selected = selection.get("track_id")
    best = selection.get("best_track_id")
    score_by_tid = {s["track_id"]: s["score"] for s in selection.get("scores", [])}

    frames_to_show = [f["frame"] for f in gt_frames]
    cap = cv2.VideoCapture(video_path)
    tiles = []
    for frec in gt_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frec["frame"])
        ok, frame = cap.read()
        if not ok:
            continue
        vis = frame.copy()

        if frec.get("batsman_bbox"):
            x1, y1, x2, y2 = [int(v) for v in frec["batsman_bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), _GT, 2)
            _label(vis, "GT", (x1 + 2, max(12, y1 - 4)), _GT, 0.45, 1)

        for tid, obs in sorted(tracks.items()):
            box = next((b for f, b, _ in obs if f == frec["frame"]), None)
            if box is None:
                continue
            chosen = (tid == selected)
            colour = _SELECTED if chosen else _REJECTED
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 3 if chosen else 1)
            sc = score_by_tid.get(tid)
            txt = f"t{tid}" + (f" {sc:.2f}" if sc is not None else "")
            _label(vis, txt, (x1 + 2, min(vis.shape[0] - 4, y2 + 15)), colour, 0.42,
                   2 if chosen else 1)

        tile = _fit(vis, tile_w, tile_h)
        banner = np.zeros((20, tile_w, 3), np.uint8)
        _label(banner, f"f={frec['frame']}", (4, 14), _WHITE, 0.42)
        tiles.append(np.vstack([banner, tile]))
    cap.release()

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

    traj = trajectory_panel(tracks, ctx["frame_w"], ctx["frame_h"], selected,
                            w=grid.shape[1] // 2, h=200)
    info = np.full((200, grid.shape[1] - traj.shape[1], 3), 22, np.uint8)
    verdict = selection.get("verdict", "?")
    vcol = _SELECTED if verdict == "CONFIDENT_BATSMAN" else _WARN
    _label(info, f"verdict: {verdict}", (8, 24), vcol, 0.6, 2)
    _label(info, f"selected track: {selected}   best: {best}", (8, 48), _WHITE, 0.45)
    _label(info, f"confidence: {selection.get('confidence')}   "
                 f"margin: {selection.get('margin')}", (8, 70), _WHITE, 0.45)
    _label(info, f"reason: {str(selection.get('reason'))[:60]}", (8, 92), _WHITE, 0.4)
    _label(info, f"tracks: {len(tracks)}   frames processed: "
                 f"{ctx['frames_processed']}", (8, 114), _WHITE, 0.4)
    y = 140
    for s in selection.get("scores", [])[:4]:
        c = s.get("components", {})
        _label(info, f"t{s['track_id']}: score {s['score']}  foot {c.get('foot_stability')} "
                     f"scale {c.get('scale_stability')} relh {c.get('relative_height')}",
               (8, y), _WHITE, 0.36)
        y += 18

    bottom = np.hstack([traj, info])
    return np.vstack([grid, bottom])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=os.path.join(RESULTS, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--model", default=os.path.join(RESULTS, "phase2_batsman_model.json"))
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "phase2_sheets"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clips", nargs="*", default=None)
    args = ap.parse_args()

    import json
    from phase2_cache_tracks import load_tracks
    from phase2_detector_benchmark import load_gt
    from phase2_batsman_score import select_batsman

    data = load_tracks(args.tracks)
    gt = {c["clip_id"]: c for c in load_gt()["clips"]}
    with open(args.model, encoding="utf-8") as f:
        model = json.load(f)["model"]

    os.makedirs(args.out, exist_ok=True)
    names = args.clips or list(data["clips"])
    if args.limit:
        names = names[:args.limit]

    for cid in names:
        c = data["clips"].get(cid)
        if not c or cid not in gt:
            continue
        ctx = {"frame_w": c["frame_w"], "frame_h": c["frame_h"],
               "frames_processed": c["frames_processed"]}
        sel = select_batsman(c["tracks"], ctx, model)
        sheet = render(os.path.join(args.videos, cid), c, ctx,
                       gt[cid]["frames"], sel, args.out)
        if sheet is None:
            continue
        ok = "OK" if sel["verdict"] == "CONFIDENT_BATSMAN" else sel["verdict"]
        path = os.path.join(args.out, f"{os.path.splitext(cid)[0]}__{ok}.jpg")
        cv2.imwrite(path, sheet)
        print(f"{cid}: {sel['verdict']} track={sel.get('track_id')} "
              f"conf={sel.get('confidence')}")
    print(f"\nsheets in {args.out}")


if __name__ == "__main__":
    main()
