"""
phase3_event_cache.py — one expensive pass over every clip, so that the
contact benchmark (C0-C6) and the Best-7 ablation (F0-F4) all read IDENTICAL
per-frame evidence.

WHY A CACHE
Every C and F method must see the same frames, the same batsman boxes, the
same poses and the same bat detections. Recomputing per experiment invites
drift, and the S0-S4 / H0-H1 / L0-L3 stages all showed how easily a
comparison stops measuring what it claims to. One pass, many cheap
experiments.

WHAT IS STORED PER FRAME (on the S2-selected batsman track)
    box, pose landmarks present, pose quality, blur, occlusion proxy,
    wrist kinematics, bat evidence, global motion, batsman-local motion

MOTION IS NOT QUALITY
Deliberately separate fields. `motion` says a frame is informative about WHEN
the event happens; `blur` / `pose_quality` say whether the frame is usable for
BIOMECHANICS. A frame can be high-motion and unusable — that is exactly the
case the Best-7 optimizer has to resolve, so the two must never be collapsed.

NO GROUND TRUTH IS READ HERE. The cache is inference-time information only.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

MAX_FRAMES = 200
COCO_BAT = 34
PAD = 0.12

# MediaPipe landmark indices (same constants the semantic features use).
L_SH, R_SH, L_WR, R_WR = 11, 12, 15, 16
L_HIP, R_HIP, L_ANK, R_ANK = 23, 24, 27, 28


def _blur(gray_crop):
    """Variance of Laplacian: the standard sharpness proxy. Higher = sharper.

    Normalised by crop area is wrong here (variance is already intensive), but
    it IS scale-sensitive, so it is stored raw and normalised per clip later —
    comparing sharpness across clips of different resolution is meaningless.
    """
    if gray_crop is None or gray_crop.size < 16:
        return None
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def _pose_fields(lm, crop_box):
    """Interpretable per-frame pose descriptors, in torso units."""
    if lm is None:
        return None
    x1, y1, x2, y2 = crop_box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    P = lambda i: (lm[i].x * w, lm[i].y * h)
    try:
        lsh, rsh, lhip, rhip = P(L_SH), P(R_SH), P(L_HIP), P(R_HIP)
        lwr, rwr, lank, rank = P(L_WR), P(R_WR), P(L_ANK), P(R_ANK)
    except (IndexError, AttributeError):
        return None
    msh = ((lsh[0] + rsh[0]) / 2, (lsh[1] + rsh[1]) / 2)
    mhip = ((lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2)
    torso = float(np.hypot(msh[0] - mhip[0], msh[1] - mhip[1]))
    if torso < 1e-6:
        return None
    vis = [float(getattr(lm[i], "visibility", 0.0))
           for i in (L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP, L_ANK, R_ANK)]
    return {
        "torso": round(torso, 3),
        # Absolute frame coords, so wrist speed survives a moving box.
        "lwr": [round(x1 + lwr[0], 2), round(y1 + lwr[1], 2)],
        "rwr": [round(x1 + rwr[0], 2), round(y1 + rwr[1], 2)],
        "msh": [round(x1 + msh[0], 2), round(y1 + msh[1], 2)],
        "mhip": [round(x1 + mhip[0], 2), round(y1 + mhip[1], 2)],
        "mank": [round(x1 + (lank[0] + rank[0]) / 2, 2),
                 round(y1 + (lank[1] + rank[1]) / 2, 2)],
        # Wrist height above the shoulder line, in torso units — the backlift
        # signature, and a phase cue that needs no 3D.
        "wrist_lift": round((msh[1] - min(lwr[1], rwr[1])) / torso, 4),
        "pose_visibility": round(float(np.mean(vis)), 4),
        "min_visibility": round(float(np.min(vis)), 4),
    }


def build(video_dir, tracks_path, features_path, out_path, max_frames=MAX_FRAMES):
    from phase2_cache_tracks import load_tracks
    from ultralytics import YOLO
    import zero_storage_pipeline as zsp
    import phase3_shot_benchmark as SB

    tracks = load_tracks(tracks_path)
    sel = SB.s2_selection(tracks_path, features_path)
    equip = YOLO("yolo11x.pt")

    with open(os.path.join(P2, "phase2_groundtruth.json"), encoding="utf-8") as f:
        split_of = {c["clip_id"]: c["split"] for c in json.load(f)["clips"]}

    out = {}
    t_start = time.perf_counter()
    for ci, cid in enumerate(sorted(tracks["clips"]), 1):
        path = os.path.join(video_dir, cid)
        if not os.path.exists(path):
            continue
        c = tracks["clips"][cid]
        s = sel.get(cid, {})
        tid = s.get("picked_track")
        by_frame = ({f: b for f, b, _ in c["tracks"][tid]}
                    if tid is not None and tid in c["tracks"] else {})

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        prev_gray = None
        i = -1
        t0 = time.perf_counter()
        while i < max_frames - 1:
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            rec = {"frame": i, "t": round(i / fps, 4),
                   "global_motion": None, "local_motion": None,
                   "has_box": False, "box": None, "track_conf": None,
                   "blur": None, "pose": None, "bat": None}

            if prev_gray is not None:
                d = cv2.absdiff(gray, prev_gray)
                rec["global_motion"] = round(float(d.mean()), 5)

            box = by_frame.get(i)
            if box is not None:
                rec["has_box"] = True
                h, w = gray.shape
                bw, bh = box[2] - box[0], box[3] - box[1]
                x1 = max(0, int(box[0] - PAD * bw)); y1 = max(0, int(box[1] - PAD * bh))
                x2 = min(w, int(box[2] + PAD * bw)); y2 = min(h, int(box[3] + PAD * bh))
                rec["box"] = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                if x2 > x1 and y2 > y1:
                    rec["blur"] = _blur(gray[y1:y2, x1:x2])
                    if prev_gray is not None:
                        d = cv2.absdiff(gray, prev_gray)
                        rec["local_motion"] = round(float(d[y1:y2, x1:x2].mean()), 5)
                    crop = fr[y1:y2, x1:x2]
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    with zsp._DETECTOR_LOCK:
                        cands = zsp._detect_candidates(
                            zsp._get_shared_detector_locked(), [rgb])
                    lm = cands[0][0] if cands[0] else None
                    rec["pose"] = _pose_fields(lm, (x1, y1, x2, y2))

                    # Bat evidence for THIS person on THIS frame.
                    r = equip.predict(fr, imgsz=960, classes=[COCO_BAT],
                                      conf=0.05, verbose=False, device=0)[0]
                    best = None
                    if r.boxes is not None and len(r.boxes):
                        for xy, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                          r.boxes.conf.cpu().numpy()):
                            bx = [float(v) for v in xy]
                            ix1, iy1 = max(bx[0], box[0]), max(bx[1], box[1])
                            ix2, iy2 = min(bx[2], box[2]), min(bx[3], box[3])
                            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                            area = (bx[2] - bx[0]) * (bx[3] - bx[1])
                            frac = inter / area if area > 0 else 0.0
                            if best is None or frac > best["on_person_frac"]:
                                best = {"conf": round(float(cf), 4),
                                        "on_person_frac": round(frac, 4),
                                        "cx": round((bx[0] + bx[2]) / 2, 1),
                                        "cy": round((bx[1] + bx[3]) / 2, 1)}
                    rec["bat"] = best
            frames.append(rec)
            prev_gray = gray
        cap.release()

        out[cid] = {
            "clip_id": cid, "split": split_of.get(cid), "fps": round(fps, 3),
            "n_frames": len(frames),
            "picked_track": tid,
            "identity_correct": s.get("identity_correct"),
            "identity_outcome": s.get("identity_outcome"),
            "build_s": round(time.perf_counter() - t0, 3),
            "frames": frames,
        }
        if ci % 5 == 0:
            print(f"  {ci} clips  ({time.perf_counter() - t_start:.0f}s elapsed)")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"max_frames": max_frames, "pad": PAD, "clips": out}, f)
    n = sum(len(v["frames"]) for v in out.values())
    print(f"wrote {out_path}  ({len(out)} clips, {n} frames)")


def load(path=None):
    with open(path or os.path.join(P3, "phase3_event_cache.json"),
              encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--tracks", default=os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--features", default=os.path.join(P3, "phase3_semantic_features.json"))
    ap.add_argument("--out", default=os.path.join(P3, "phase3_event_cache.json"))
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    args = ap.parse_args()
    os.makedirs(P3, exist_ok=True)
    build(args.videos, args.tracks, args.features, args.out, args.max_frames)


if __name__ == "__main__":
    main()
