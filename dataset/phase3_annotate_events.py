"""
phase3_annotate_events.py — annotate the temporal ground truth Phase 3 needs
and Phase 2 did not have: shot boundaries, contact frame, scene cuts, and
whether a window is one coherent batting event at all.

WHY THIS EXISTS
Phase 2 annotated WHO (batsman boxes on 6 frames per clip). Phase 3 asks
WHEN, and none of it is annotated: shot_start, shot_end, contact_frame and
phase events are absent on all 51 clips. Without them the shot-localization
ablation (L0-L3), contact-error metrics, phase accuracy and the best-7
evaluation are all uncomputable — exactly the gap Phase 2's report flagged as
its own priority 4.

PROPOSAL-ASSISTED, LIKE PHASE 2
Labelling temporal events from scratch is expensive and imprecise. Instead
this renders, per clip:

  * a COARSE strip spanning the whole clip, to locate the action and expose
    scene cuts, multiple deliveries, or no batting at all;
  * a FINE strip at every frame around the proposed contact, dense enough to
    pick the contact frame by eye at the +/-1 frame the metric claims.

The proposal comes from motion energy computed INSIDE the batsman's tracked
box, not the whole frame — global motion is dominated by whoever is nearest
the camera, which on this corpus is usually the bowler. The proposal is a
starting point the annotator overrides freely; it is not ground truth and is
recorded separately from the label so any proposal bias stays visible.

The batsman box comes from the Phase-2 cached ByteTrack tracks and the
fitted geometry scorer, so the temporal annotation is anchored to the same
subject decision the rest of the pipeline makes.
"""

import argparse
import json
import os

import cv2
import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

# Coarse-strip density. Fixed at 12 this was every 8th frame on a 90-frame
# clip, which is too sparse to place a shot boundary (+/-4 frames of
# uncertainty) or to rule out a second delivery. Scaled so short clips get a
# genuinely dense overview while long session clips stay readable.
COARSE_MIN, COARSE_MAX = 12, 24


def coarse_count(n_frames):
    return max(COARSE_MIN, min(COARSE_MAX, n_frames // 4))
FINE_RADIUS = 9        # +/- frames around the proposed contact
COARSE_W, FINE_W = 200, 190
ROW_H = 150
FINE_H = 230        # taller: fine tiles are batsman crops, judged by eye

_WHITE = (255, 255, 255)
_GREEN = (0, 230, 0)
_AMBER = (0, 190, 255)
_RED = (0, 0, 255)


def _label(img, text, org, colour, scale=0.42, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _fit(img, w, h):
    if img is None or img.size == 0:
        return np.zeros((h, w, 3), np.uint8)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas


def box_at(track_obs, frame_no, max_gap=8):
    """Batsman box at `frame_no`, or the nearest observation within max_gap.
    Tracks are stride-sampled on long clips, so an exact hit is not
    guaranteed and refusing to interpolate keeps the box honest."""
    best, best_d = None, None
    for f, b, _ in track_obs:
        d = abs(f - frame_no)
        if best_d is None or d < best_d:
            best, best_d = b, d
    return best if (best_d is not None and best_d <= max_gap) else None


def scan_profiles(video_path, track_obs, max_frames=1200):
    """ONE sequential pass producing both the in-box motion profile and the
    whole-frame scene-similarity profile.

    Sequential decode, not `cap.set()` seeking: random seeking forces H.264
    to re-decode from the nearest keyframe every time, which measured
    minutes per clip. Reading forward and skipping unwanted frames is the
    same data at a fraction of the cost, and computing both profiles in the
    same pass halves it again.

    Returns (motion_frames, motion_energy, sim_frames, sims).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], [], []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, -(-n // max_frames))

    m_frames, energy, prev_crop = [], [], None
    s_frames, sims, prev_hist = [], [], None

    i = -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue

        # --- whole-frame scene similarity (cut detection) ---
        hsv = cv2.cvtColor(cv2.resize(fr, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
        if prev_hist is not None:
            s_frames.append(i)
            sims.append(float(cv2.compareHist(prev_hist, h, cv2.HISTCMP_CORREL)))
        prev_hist = h

        # --- in-box motion energy (contact proposal) ---
        box = box_at(track_obs, i, max_gap=max(8, step * 2))
        if box is None:
            prev_crop = None
            continue
        fh, fw = fr.shape[:2]
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(fw, int(box[2])), min(fh, int(box[3]))
        if x2 <= x1 or y2 <= y1:
            prev_crop = None
            continue
        g = cv2.resize(cv2.cvtColor(fr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (64, 128))
        if prev_crop is not None:
            m_frames.append(i)
            energy.append(float(np.mean(np.abs(g.astype(np.int16) - prev_crop.astype(np.int16)))))
        prev_crop = g

    cap.release()
    return m_frames, energy, s_frames, sims


def propose_contact(frames, energy):
    """Peak of the smoothed in-box motion profile. A proposal only."""
    if len(energy) < 5:
        return None, 0.0
    k = np.ones(3) / 3.0
    sm = np.convolve(np.asarray(energy, float), k, mode="same")
    i = int(np.argmax(sm))
    med = float(np.median(sm)) or 1e-6
    return frames[i], round(float(sm[i]) / med, 2)


def render_sheet(video_path, clip, track_obs, proposal, cuts, out_dir):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Collect every frame the sheet needs in ONE sequential pass, for the
    # same reason as scan_profiles: seeking dominates the cost otherwise.
    c0 = proposal if proposal is not None else n // 2
    cn = coarse_count(n)
    wanted = set(int(round(i * (n - 1) / (cn - 1))) for i in range(cn))
    wanted |= set(i for i in range(c0 - FINE_RADIUS, c0 + FINE_RADIUS + 1) if 0 <= i < n)
    grabbed, i = {}, -1
    last = max(wanted) if wanted else -1
    while i < last:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i in wanted:
            grabbed[i] = fr

    def read(i):
        return grabbed.get(int(i))

    def tile(i, w, mark=None, crop_to_batsman=False, tile_h=ROW_H):
        """One filmstrip cell.

        `crop_to_batsman` is what makes the fine strip usable: at full-frame
        scale a distant batsman renders around 50px tall, far too small to
        judge bat-ball contact by eye. Cropping to the tracked box with
        padding puts the batsman at usable size, which is the whole point of
        the fine strip.
        """
        fr = read(i)
        if fr is None:
            t = np.zeros((tile_h, w, 3), np.uint8)
            _label(t, f"f{i} X", (4, 16), _RED)
            return t
        box = box_at(track_obs, i, max_gap=10)
        if crop_to_batsman and box is not None:
            fh, fw = fr.shape[:2]
            bw, bh = box[2] - box[0], box[3] - box[1]
            # Generous horizontal padding: the bat and the ball live OUTSIDE
            # the person box, and they are exactly what contact is judged on.
            px, py = 1.1 * bw, 0.25 * bh
            x1 = max(0, int(box[0] - px)); y1 = max(0, int(box[1] - py))
            x2 = min(fw, int(box[2] + px)); y2 = min(fh, int(box[3] + py))
            if x2 > x1 and y2 > y1:
                fr = fr[y1:y2, x1:x2]
        elif box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(fr, (x1, y1), (x2, y2), _GREEN, 2)
        t = _fit(fr, w, tile_h)
        bar = np.zeros((18, w, 3), np.uint8)
        col = _AMBER if mark else _WHITE
        _label(bar, f"f{i}" + (f" {mark}" if mark else f"  {i/fps:.2f}s"), (3, 13), col, 0.38)
        return np.vstack([bar, t])

    # Coarse strip: whole clip
    coarse_idx = [int(round(i * (n - 1) / (cn - 1))) for i in range(cn)]
    coarse = [tile(i, COARSE_W, "CUT?" if any(abs(i - c) < n / cn for c in cuts) else None)
              for i in coarse_idx]

    # Fine strip: every frame around the proposal, CROPPED to the batsman.
    c = proposal if proposal is not None else n // 2
    fine_idx = [i for i in range(c - FINE_RADIUS, c + FINE_RADIUS + 1) if 0 <= i < n]
    fine = [tile(i, FINE_W, "PROPOSED" if i == c else None,
                 crop_to_batsman=True, tile_h=FINE_H) for i in fine_idx]
    cap.release()

    def strip(tiles, per_row):
        rows = []
        for r in range(0, len(tiles), per_row):
            row = tiles[r:r + per_row]
            while len(row) < per_row:
                row.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(row))
        return np.vstack(rows)

    coarse_img = strip(coarse, 6)
    fine_img = strip(fine, 10) if fine else np.zeros((10, coarse_img.shape[1], 3), np.uint8)
    width = max(coarse_img.shape[1], fine_img.shape[1])
    coarse_img = np.pad(coarse_img, ((0, 0), (0, width - coarse_img.shape[1]), (0, 0)))
    fine_img = np.pad(fine_img, ((0, 0), (0, width - fine_img.shape[1]), (0, 0)))

    def band(text, colour=_WHITE, h=26, scale=0.5):
        b = np.full((h, width, 3), 20, np.uint8)
        _label(b, text, (8, int(h * 0.68)), colour, scale, 1)
        return b

    sheet = np.vstack([
        band(f"{clip['clip_id']}  |  {clip['split'].upper()}  |  {n}f @ {fps:.0f}fps"
             f"  |  {n/fps:.1f}s  |  cuts detected: {len(cuts)}", _WHITE, 30, 0.55),
        band("COARSE - whole clip: is this ONE shot? scene cut? no batting?", _AMBER),
        coarse_img,
        band(f"FINE - every frame around proposed contact f{c}  (pick the true contact)", _AMBER),
        fine_img,
    ])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{os.path.splitext(clip['clip_id'])[0]}.jpg")
    cv2.imwrite(path, sheet)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--tracks", default=os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--model", default=os.path.join(P2, "phase2_batsman_model_geom.json"))
    ap.add_argument("--out", default=os.path.join(P3, "event_sheets"))
    ap.add_argument("--cut-threshold", type=float, default=0.55,
                    help="histogram correlation below this flags a candidate cut")
    args = ap.parse_args()

    from phase2_cache_tracks import load_tracks
    from phase2_detector_benchmark import load_gt
    from phase2_batsman_score import select_batsman
    from phase2_fit_batsman_model import gt_track_for_clip

    data = load_tracks(args.tracks)
    gt = load_gt()
    with open(args.model, encoding="utf-8") as f:
        model = json.load(f)["model"]

    os.makedirs(P3, exist_ok=True)
    proposals = []
    for clip in gt["clips"]:
        cid = clip["clip_id"]
        c = data["clips"].get(cid)
        path = os.path.join(args.videos, cid)
        if not c or not os.path.exists(path):
            continue
        ctx = {"frame_w": c["frame_w"], "frame_h": c["frame_h"],
               "frames_processed": c["frames_processed"]}
        sel = select_batsman(c["tracks"], ctx, model)

        # Anchor on the ANNOTATED batsman, not the model's prediction.
        #
        # Building temporal ground truth on top of the model's batsman guess
        # would inherit its errors: on instagram1_front_01 the geometry model
        # selects the near-camera feeder, so a contact proposal and a cropped
        # filmstrip built from it describe the wrong person entirely, and any
        # contact label taken from that sheet would be silently worthless.
        # Phase 2 already annotated the true batsman box on 6 frames per clip,
        # so the track matching those boxes is the defensible anchor. The
        # model's own verdict is still recorded, so proposal-vs-truth and
        # selection-vs-truth stay separately measurable.
        gt_frames = [f for f in clip["frames"] if f["batsman_visible"] and f["batsman_bbox"]]
        gt_tid, n_match = gt_track_for_clip(c["tracks"], gt_frames)

        if gt_tid is not None:
            tid, anchor = gt_tid, "annotated_batsman"
        else:
            # No track matches the annotated batsman -- detection missed them.
            # Fall back to the model's pick so the clip still gets a sheet,
            # but flag it: any label from this sheet is suspect.
            tid = sel.get("track_id") or sel.get("best_track_id")
            anchor = "model_fallback_NO_GT_TRACK"
        obs = c["tracks"].get(tid, []) if tid is not None else []

        frames, energy, cf, sims = scan_profiles(path, obs)
        contact, prom = propose_contact(frames, energy)
        cuts = [f for f, s in zip(cf, sims) if s < args.cut_threshold]

        sheet = render_sheet(path, clip, obs, contact, cuts, args.out)
        proposals.append({
            "clip_id": cid, "split": clip["split"],
            "n_frames": clip["n_frames"], "fps": clip["fps"],
            "batsman_track": tid,
            "track_anchor": anchor,
            "gt_matched_frames": n_match,
            "model_predicted_track": sel.get("track_id"),
            "model_verdict": sel["verdict"],
            "model_agrees_with_gt": (sel.get("track_id") == gt_tid
                                     and gt_tid is not None),
            "proposed_contact": contact, "proposal_prominence": prom,
            "candidate_cuts": cuts[:20], "n_candidate_cuts": len(cuts),
            "sheet": os.path.basename(sheet),
        })
        flag = "" if anchor == "annotated_batsman" else "  [NO GT TRACK]"
        print(f"{cid}: contact~{contact} prom={prom} cuts={len(cuts)} "
              f"track={tid} anchor={anchor} model={sel['verdict']}{flag}")

    out = os.path.join(P3, "phase3_event_proposals.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"coarse_min": COARSE_MIN, "coarse_max": COARSE_MAX,
                   "fine_radius": FINE_RADIUS,
                   "cut_threshold": args.cut_threshold, "clips": proposals}, f, indent=1)
    print(f"\nwrote {out}  ({len(proposals)} clips)")
    print(f"sheets in {args.out}")


if __name__ == "__main__":
    main()
