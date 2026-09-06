"""
extraction_visualizer.py — annotated contact sheets for extracted shots.

WHY THIS EXISTS
The wrong-person defect was found by one person overlaying landmarks on one
frame by hand. Every automated check passed it: MediaPipe returned a
confident pose, topology validation passed (a standing feeder has the same
Y-ordering as a standing batsman), and the bone-length filter caught only 2
of 3 cases, by accident. The one thing that caught it reliably was LOOKING.

This module makes that cheap and repeatable. A pipeline that produces a
plausible score while tracking the bowler should be obvious in one glance at
a contact sheet, not discoverable only by a careful reviewer.

USAGE
    python extraction_visualizer.py --video raw_videos/kohli_front_01.mp4 \
        --start-sec 0 --end-sec 4 --config A3 --out sheets/

Drawing is best-effort: a visualization failure must never affect extraction.
"""

import argparse
import os

import cv2
import numpy as np

from schema import CANONICAL_FRAME_NAMES

# MediaPipe Pose skeleton edges (subset that matters for batting technique —
# the face mesh adds clutter without informing "is this the right person").
SKELETON_EDGES = [
    (11, 12), (11, 23), (12, 24), (23, 24),          # torso
    (11, 13), (13, 15), (12, 14), (14, 16),          # arms
    (23, 25), (25, 27), (24, 26), (26, 28),          # legs
    (27, 31), (28, 32),                               # feet
]

_GREEN = (0, 220, 0)
_RED = (0, 0, 255)
_YELLOW = (0, 215, 255)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def _put_label(img, text, org, color=_WHITE, scale=0.45, thickness=1):
    """Text with a dark outline so it stays readable over any footage."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_pose(img, landmarks, color=_GREEN):
    """Draw the selected subject's skeleton and bounding box."""
    if landmarks is None:
        return img
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in SKELETON_EDGES:
        if a < len(pts) and b < len(pts):
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
        if i < len(pts):
            cv2.circle(img, pts[i], 3, color, -1, cv2.LINE_AA)

    body = [pts[i] for i in range(11, 33) if i < len(pts)]
    if body:
        xs, ys = [p[0] for p in body], [p[1] for p in body]
        pad = 12
        cv2.rectangle(img, (max(0, min(xs) - pad), max(0, min(ys) - pad)),
                      (min(w, max(xs) + pad), min(h, max(ys) + pad)), color, 2)
    return img


def annotate_frame(frame_bgr, *, frame_num, fps, phase, is_contact,
                   landmarks=None, track_label="", confidence=None, tile_w=320):
    """One annotated tile of the contact sheet."""
    img = frame_bgr.copy()
    colour = _YELLOW if is_contact else _GREEN
    draw_pose(img, landmarks, colour)

    h, w = img.shape[:2]
    scale = tile_w / max(w, 1)
    img = cv2.resize(img, (tile_w, max(1, int(h * scale))))

    banner_h = 62
    canvas = np.zeros((img.shape[0] + banner_h, tile_w, 3), dtype=np.uint8)
    canvas[banner_h:, :] = img

    ts = frame_num / fps if fps else 0.0
    _put_label(canvas, f"{phase}", (6, 16), colour, 0.5)
    _put_label(canvas, f"f={frame_num}  t={ts:.2f}s", (6, 33), _WHITE, 0.42)
    tail = track_label or ""
    if confidence is not None:
        tail = f"{tail}  conf={confidence:.2f}".strip()
    if tail:
        _put_label(canvas, tail, (6, 50), _WHITE, 0.40)
    if is_contact:
        _put_label(canvas, "CONTACT", (tile_w - 78, 16), _YELLOW, 0.45)
    if landmarks is None:
        _put_label(canvas, "NO POSE", (tile_w - 74, 33), _RED, 0.45)
    return canvas


def build_contact_sheet(frames_bgr, indices, fps, selected_poses=None,
                        contact_frame=None, contact_confidence=None,
                        track_label="", title="", cols=4, tile_w=320):
    """Compose the annotated tiles into a single inspectable image."""
    selected_poses = selected_poses or [None] * len(frames_bgr)
    tiles = []
    for i, (frame, idx) in enumerate(zip(frames_bgr, indices)):
        phase = CANONICAL_FRAME_NAMES[i] if i < len(CANONICAL_FRAME_NAMES) else f"frame_{i}"
        if frame is None:
            tile = np.zeros((int(tile_w * 0.75) + 62, tile_w, 3), dtype=np.uint8)
            _put_label(tile, phase, (6, 16), _RED, 0.5)
            _put_label(tile, f"f={idx}  READ FAILED", (6, 33), _RED, 0.42)
        else:
            tile = annotate_frame(
                frame, frame_num=idx, fps=fps, phase=phase,
                is_contact=(contact_frame is not None and idx == contact_frame),
                landmarks=selected_poses[i] if i < len(selected_poses) else None,
                track_label=track_label,
                confidence=contact_confidence if idx == contact_frame else None,
                tile_w=tile_w)
        tiles.append(tile)

    if not tiles:
        return None
    tile_h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, tile_h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]

    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)

    if title:
        header = np.zeros((30, sheet.shape[1], 3), dtype=np.uint8)
        _put_label(header, title, (8, 20), _WHITE, 0.55)
        sheet = np.vstack([header, sheet])
    return sheet


def visualize_extraction(video_path, start_sec, end_sec, config_name="A3", out_dir="."):
    """Run the real extraction path and render what it selected.

    Imported lazily: zero_storage_pipeline pulls in MediaPipe/TensorFlow, and
    the drawing helpers above should stay usable (and unit-testable) without
    paying that cost.
    """
    import zero_storage_pipeline as zsp
    from extraction_config import get_config

    config = get_config(config_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame, end_frame = int(start_sec * fps), int(end_sec * fps)

    diag = zsp.diagnostics.ExtractionDiagnostics(
        session_name=os.path.basename(video_path), video_id=os.path.basename(video_path),
        config_name=config.name, shot_start_frame=start_frame, shot_end_frame=end_frame,
        fps=fps)

    indices, method = zsp.select_phase_frames(
        cap, start_frame, end_frame, os.path.basename(video_path), config, diag=diag)
    zsp.diagnostics.record_frames(diag, indices, method)

    frames_bgr = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        frames_bgr.append(frame if ok else None)
    cap.release()

    # Re-run selection on exactly these 7 frames so the drawn skeleton is the
    # subject the pipeline actually used downstream, not a fresh guess.
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f is not None else None for f in frames_bgr]
    with zsp._DETECTOR_LOCK:
        candidates = zsp._detect_candidates(zsp._get_shared_detector_locked(), frames_rgb)
    selected, report = zsp.select_subject(candidates)

    os.makedirs(out_dir, exist_ok=True)
    title = (f"{os.path.basename(video_path)} | {config.name} | {method} | "
             f"subject={'OK' if selected else 'REJECTED'} | contact={diag.contact_frame}")
    sheet = build_contact_sheet(
        frames_bgr, indices, fps,
        selected_poses=selected if selected else [None] * len(indices),
        contact_frame=diag.contact_frame, contact_confidence=diag.contact_confidence,
        track_label=f"tracks={report.get('n_tracks')}", title=title)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{stem}__{config.name}__{method}.jpg")
    cv2.imwrite(out_path, sheet)
    return out_path, diag, report


def _clip_duration_sec(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (n - 1) / fps if n > 1 and fps > 0 else None


def batch_visualize(video_dir, config_name, out_dir, limit=0, min_size_kb=100):
    """Render a sheet per clip so wrong-person selection can be AUDITED at a
    glance across the corpus rather than one clip at a time.

    Uses the whole clip as the shot window, matching
    run_extraction_experiment's default, so the sheets depict the same
    decisions the reported metrics were computed from.
    """
    import glob
    paths = sorted(p for p in glob.glob(os.path.join(video_dir, "*.mp4"))
                   if os.path.getsize(p) >= min_size_kb * 1024)
    if limit:
        paths = paths[:limit]

    written = []
    for p in paths:
        dur = _clip_duration_sec(p)
        if dur is None:
            continue
        try:
            out, diag, report = visualize_extraction(p, 0.0, dur, config_name, out_dir)
            written.append((out, diag, report))
            print(f"{os.path.basename(p)}: {diag.sampling_method}, "
                  f"tracks={report.get('n_tracks')}, subject={report.get('reason')}")
        except Exception as exc:  # noqa: BLE001 - one bad clip must not end the audit
            print(f"{os.path.basename(p)}: FAILED {type(exc).__name__}: {exc}")
    return written


def main():
    ap = argparse.ArgumentParser(description="Render annotated contact sheets for extracted shots.")
    ap.add_argument("--video", help="single clip")
    ap.add_argument("--videos", help="directory of clips (batch mode)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, help="required with --video")
    ap.add_argument("--config", default="A3")
    ap.add_argument("--out", default="extraction_sheets")
    args = ap.parse_args()

    if args.videos:
        written = batch_visualize(args.videos, args.config, args.out, args.limit)
        print(f"\nwrote {len(written)} sheets to {args.out}")
        return

    if not args.video or args.end_sec is None:
        ap.error("provide --videos DIR, or --video FILE with --end-sec")

    path, diag, report = visualize_extraction(
        args.video, args.start_sec, args.end_sec, args.config, args.out)
    print(f"wrote {path}")
    print(f"  method={diag.sampling_method}  frames={diag.selected_frames}")
    print(f"  contact={diag.contact_frame} (conf={diag.contact_confidence}, valid={diag.contact_valid})")
    print(f"  subject: {report.get('reason')}  tracks={report.get('n_tracks')}")


if __name__ == "__main__":
    main()
