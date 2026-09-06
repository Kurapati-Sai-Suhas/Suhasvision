"""
phase2_tracking.py — multi-object tracking over detector candidates.

Produces, per clip, tracks of the form
    track_id -> [(frame_index, (x1,y1,x2,y2), conf), ...]

FRAME SELECTION
Tracking needs temporal continuity, so frames cannot simply be the 6
annotated ones. But the evaluation must land EXACTLY on the annotated frames,
otherwise the ground truth does not apply. The frame list is therefore the
union of a uniform stride and the annotated frames, sorted — continuity for
the tracker, exact alignment for the benchmark, no interpolation or
nearest-frame fudging.

Long clips (one is 5999 frames) are stride-subsampled to a frame budget.
That weakens temporal association on exactly the clips that are whole net
sessions rather than single shots, which is recorded rather than hidden.

ByteTrack and BoT-SORT come from ultralytics rather than being reimplemented
here: a hand-rolled tracker would put this phase's conclusions at the mercy
of my own association bugs, which is precisely the failure Phase 1 was about.
"""

import os
from typing import Dict, List, Tuple

import cv2

DEFAULT_FRAME_BUDGET = 700


def frame_plan(n_frames: int, must_include: List[int], budget: int = DEFAULT_FRAME_BUDGET):
    """Union of a uniform stride and the frames that must be evaluated."""
    stride = max(1, -(-n_frames // budget))          # ceil division
    frames = set(range(0, n_frames, stride))
    frames.update(i for i in must_include if 0 <= i < n_frames)
    return sorted(frames), stride


def track_clip(video_path: str, model, tracker_cfg: str, must_include: List[int],
               imgsz: int = 640, conf: float = 0.10,
               budget: int = DEFAULT_FRAME_BUDGET) -> Tuple[Dict[int, List], dict]:
    """Run detector+tracker over one clip. Returns (tracks, meta)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}, {"error": "cannot open"}
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    plan, stride = frame_plan(n_frames, must_include, budget)

    # A fresh tracker per clip: persist=True carries state across calls, so
    # without an explicit reset the previous clip's tracks would leak into
    # this one and manufacture identity switches that never happened.
    if hasattr(model, "predictor") and model.predictor is not None:
        try:
            model.predictor.trackers = None
        except Exception:  # noqa: BLE001 - best effort reset
            pass
        model.predictor = None

    tracks: Dict[int, List] = {}
    n_detections = n_untracked = 0
    for frame_no in plan:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue
        res = model.track(frame, persist=True, tracker=tracker_cfg, classes=[0],
                          imgsz=imgsz, conf=conf, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        n_detections += len(xyxy)
        if boxes.id is None:
            # Detections the tracker refused to assign an id to (usually
            # low-confidence first-frame boxes). Counted, not silently dropped.
            n_untracked += len(xyxy)
            continue
        ids = boxes.id.cpu().numpy().astype(int)
        for tid, box, c in zip(ids, xyxy, confs):
            tracks.setdefault(int(tid), []).append(
                (frame_no, tuple(float(v) for v in box), float(c)))

    cap.release()
    return tracks, {
        "n_frames": n_frames, "fps": fps, "stride": stride,
        "frames_processed": len(plan), "n_detections": n_detections,
        "n_untracked_detections": n_untracked,
        "subsampled": stride > 1,
    }


def track_stats(tracks: Dict[int, List], frames_processed: int) -> dict:
    """Structural properties of the track set, no ground truth required."""
    if not tracks:
        return {"n_tracks": 0, "mean_coverage": None, "max_coverage": None,
                "mean_fragmentation": None}

    coverages, frags = [], []
    for obs in tracks.values():
        frames = sorted(f for f, _, _ in obs)
        coverages.append(len(frames))
        # Fragmentation: how many times the track goes absent and comes back.
        # Measured over the PROCESSED frame list, so stride does not inflate it.
        gaps = sum(1 for a, b in zip(frames, frames[1:]) if b - a > 1)
        frags.append(gaps)

    return {
        "n_tracks": len(tracks),
        "mean_coverage": round(sum(coverages) / len(coverages), 2),
        "max_coverage": max(coverages),
        "max_coverage_fraction": round(max(coverages) / max(frames_processed, 1), 3),
        "mean_fragmentation": round(sum(frags) / len(frags), 2),
        "total_fragmentation": sum(frags),
    }


def build_tracking_model(weights="yolo11m.pt"):
    from ultralytics import YOLO
    return YOLO(weights)


def bbox_at(tracks: Dict[int, List], track_id: int, frame_no: int):
    for f, box, _ in tracks.get(track_id, []):
        if f == frame_no:
            return box
    return None


def tracks_at_frame(tracks: Dict[int, List], frame_no: int):
    """[(track_id, bbox, conf)] present in this exact frame."""
    out = []
    for tid, obs in tracks.items():
        for f, box, c in obs:
            if f == frame_no:
                out.append((tid, box, c))
                break
    return out
