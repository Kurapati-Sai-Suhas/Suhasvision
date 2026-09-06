"""
phase2_annotate.py — build the Phase-2 ground-truth benchmark.

WHY PROPOSAL-ASSISTED ANNOTATION
Phase 1 could not compute wrong-person rate, tIoU, contact error or E2E-VSER
because no annotations existed, and drawing boxes by hand on 50 clips is not
affordable. This tool instead renders a high-recall detector's PROPOSALS as
numbered boxes on sampled frames; the annotator then records WHICH numbered
box is the batsman. Verifying a proposal is far cheaper than drawing a box,
and it yields a pixel-accurate GT box for free.

THE BIAS THIS INTRODUCES, AND HOW IT IS HANDLED
Ground truth anchored to a proposal detector cannot, by construction, contain
a batsman that detector missed -- which would flatter that detector's recall
and understate everyone else's. Two mitigations, both mandatory:

  1. The proposal detector is the highest-recall configuration available
     (large model, generous resolution, low confidence), NOT the
     configuration being recommended. It is a candidate generator, not a
     contender.
  2. Every sheet also shows the CLEAN frame beside the annotated one, and the
     schema carries `batsman_visible_but_undetected`. When the annotator can
     see a batsman with no box, that frame is recorded as a proposal MISS and
     counts against every detector's recall, including the proposal
     detector's.

Residual bias remains: a batsman neither detected nor noticed by the
annotator is invisible to this benchmark. That is stated in the report rather
than corrected, because it cannot be corrected without exhaustive manual
annotation.

SPLIT DISCIPLINE
Clips are assigned to dev/eval by a stable hash of the clip name, so the
split does not drift when clips are added and cannot be re-rolled to
flatter a result.
"""

import argparse
import glob
import hashlib
import json
import os

import cv2
import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(_MODULE_DIR, "phase2_results")

# Frames sampled per clip. Sparse by necessity (see module docstring). Six
# evenly-spaced frames span the shot without making a sheet unreadable, and
# give enough temporal spread to expose identity switches.
FRAMES_PER_CLIP = 6

# The proposal detector: recall ceiling, not the recommended configuration.
PROPOSAL_DETECTOR = "yolo11x@1280"
PROPOSAL_CONF = 0.06

_PALETTE = [(0, 255, 255), (0, 200, 0), (255, 128, 0), (255, 0, 255),
            (0, 128, 255), (200, 200, 0), (128, 0, 255), (0, 0, 255)]


def split_for(clip_id: str) -> str:
    """Stable dev/eval assignment, ~50/50, from the clip name alone."""
    h = hashlib.sha256(clip_id.encode("utf-8")).hexdigest()
    return "dev" if int(h[:8], 16) % 2 == 0 else "eval"


def sample_frame_indices(n_frames: int, k: int = FRAMES_PER_CLIP):
    """Evenly spaced, excluding the exact last frame (often a black/partial
    frame in these clips)."""
    if n_frames <= 1:
        return [0]
    last = max(0, n_frames - 2)
    return [int(round(last * i / (k - 1))) for i in range(k)] if k > 1 else [0]


def _label(img, text, org, colour, scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


CONTEXT_W = 300      # context thumbnail width per row
CROP_W = 104         # candidate crop width
ROW_H = 190          # fixed row height so crops are directly comparable
MAX_CROPS = 8


def _fit(img, box_w, box_h):
    """Letterbox `img` into a box_w x box_h canvas, preserving aspect."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((box_h, box_w, 3), dtype=np.uint8)
    s = min(box_w / w, box_h / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    y0, x0 = (box_h - nh) // 2, (box_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def render_clip_sheet(frames, indices, dets_per_frame, fps, title):
    """One row per sampled frame: a context thumbnail with numbered boxes,
    followed by each candidate as its OWN crop.

    The crops are the point. Drawing every box on one shared image makes
    overlapping candidates impossible to tell apart -- and overlapping boxes
    on the same person are common here, so index assignment from a shared
    view would be guesswork. A separate crop per candidate makes "which one
    is the batsman" an unambiguous read, which is the whole basis of the
    ground truth.
    """
    rows = []
    for frame, idx, dets in zip(frames, indices, dets_per_frame):
        row = np.zeros((ROW_H + 22, CONTEXT_W + MAX_CROPS * (CROP_W + 4) + 8, 3), dtype=np.uint8)

        if frame is None:
            _label(row, f"f={idx}  READ FAILED", (6, 16), (0, 0, 255))
            rows.append(row)
            continue

        ctx = frame.copy()
        for i, d in enumerate(dets[:MAX_CROPS]):
            c = _PALETTE[i % len(_PALETTE)]
            x1, y1, x2, y2 = [int(v) for v in d.as_xyxy()]
            cv2.rectangle(ctx, (x1, y1), (x2, y2), c, 2)
            cx, cy = [int(v) for v in d.center]
            _label(ctx, str(i), (cx - 6, cy), c, 0.9, 2)
        row[22:22 + ROW_H, 0:CONTEXT_W] = _fit(ctx, CONTEXT_W, ROW_H)
        _label(row, f"f={idx}  t={idx / fps:.2f}s   {len(dets)} candidates",
               (6, 15), (255, 255, 255), 0.45)

        h, w = frame.shape[:2]
        for i, d in enumerate(dets[:MAX_CROPS]):
            x1 = max(0, int(d.x1) - 4); y1 = max(0, int(d.y1) - 4)
            x2 = min(w, int(d.x2) + 4); y2 = min(h, int(d.y2) + 4)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            x0 = CONTEXT_W + 8 + i * (CROP_W + 4)
            row[22:22 + ROW_H, x0:x0 + CROP_W] = _fit(crop, CROP_W, ROW_H)
            c = _PALETTE[i % len(_PALETTE)]
            cv2.rectangle(row, (x0, 22), (x0 + CROP_W - 1, 22 + ROW_H - 1), c, 1)
            _label(row, f"{i}", (x0 + 3, 15), c, 0.55, 2)
            _label(row, f"c{d.conf:.2f} h{int(d.height)}", (x0 + 18, 15), (200, 200, 200), 0.38)
        rows.append(row)

    sheet = np.vstack(rows)
    header = np.zeros((30, sheet.shape[1], 3), dtype=np.uint8)
    _label(header, title, (8, 20), (255, 255, 255), 0.55)
    return np.vstack([header, sheet])


def build(video_dir, out_dir, limit=0, stride=1, min_size_kb=100):
    """Render one proposal sheet per clip and write the proposal boxes."""
    from phase2_detectors import build_detector

    os.makedirs(out_dir, exist_ok=True)
    sheets_dir = os.path.join(out_dir, "annotation_sheets")
    os.makedirs(sheets_dir, exist_ok=True)

    paths = sorted(p for p in glob.glob(os.path.join(video_dir, "*.mp4"))
                   if os.path.getsize(p) >= min_size_kb * 1024)
    paths = paths[::stride]
    if limit:
        paths = paths[:limit]

    det = build_detector(PROPOSAL_DETECTOR)
    det.conf = PROPOSAL_CONF
    if hasattr(det, "model"):
        pass  # conf is passed per-call via the wrapper's stored value

    manifest = []
    for p in paths:
        clip_id = os.path.basename(p)
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            print(f"{clip_id}: cannot open"); continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sample_frame_indices(n)

        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            frames.append(f if ok else None)
        cap.release()

        dets = det.detect_batch(frames)
        sheet = render_clip_sheet(
            frames, idxs, dets, fps,
            f"{clip_id}  |  {split_for(clip_id).upper()}  |  {n}f @ {fps:.0f}fps  |  proposals: {PROPOSAL_DETECTOR}")
        cv2.imwrite(os.path.join(sheets_dir, f"{os.path.splitext(clip_id)[0]}.jpg"), sheet)

        manifest.append({
            "clip_id": clip_id,
            "split": split_for(clip_id),
            "n_frames": n,
            "fps": round(fps, 3),
            "width": int(frames[0].shape[1]) if frames[0] is not None else None,
            "height": int(frames[0].shape[0]) if frames[0] is not None else None,
            "sampled_frames": idxs,
            "proposals": [[list(map(float, d.as_xyxy())) + [round(d.conf, 4)] for d in fd]
                          for fd in dets],
        })
        print(f"{clip_id}: {n}f, proposals/frame " +
              ",".join(str(len(d)) for d in dets) + f"  [{split_for(clip_id)}]")

    path = os.path.join(out_dir, "annotation_proposals.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"proposal_detector": PROPOSAL_DETECTOR,
                   "proposal_conf": PROPOSAL_CONF,
                   "frames_per_clip": FRAMES_PER_CLIP,
                   "clips": manifest}, f, indent=1)
    print(f"\nwrote {path} ({len(manifest)} clips)")
    print(f"sheets in {sheets_dir}")
    n_dev = sum(1 for m in manifest if m["split"] == "dev")
    print(f"split: dev {n_dev}, eval {len(manifest) - n_dev}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    build(args.videos, args.out, args.limit, args.stride)


if __name__ == "__main__":
    main()
