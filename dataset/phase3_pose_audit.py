"""
phase3_pose_audit.py — side-by-side skeleton sheets, MediaPipe vs RTMPose.

Every metric in the pose benchmark is ground-truth-free: bone-length CV,
jitter and angle plausibility all measure SELF-CONSISTENCY. A model can be
perfectly self-consistent and perfectly wrong -- a skeleton locked onto the
wrong person, or onto a limb-shaped background edge, scores well on all of
them. Phase 1 shipped a wrong-person bug that looked fine in every table, so
the numbers do not get to decide this alone.

These sheets put both models' skeletons on the SAME frames so a human can see
which one is actually on the batsman and which joints are placed wrongly.
"""

import argparse
import json
import os

import cv2
import numpy as np

import phase3_pose_models as PM

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

EDGES = (("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"),
         ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"),
         ("r_elbow", "r_wrist"), ("l_shoulder", "l_hip"),
         ("r_shoulder", "r_hip"), ("l_hip", "r_hip"), ("l_hip", "l_knee"),
         ("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
         ("l_ankle", "l_foot_index"), ("r_ankle", "r_foot_index"))


def draw(img, pose, colour, box):
    if not (pose and pose.ok):
        cv2.putText(img, "NO POSE", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)
        return img
    ox, oy = box[0], box[1]
    for a, b in EDGES:
        if a in pose.pts and b in pose.pts:
            pa = (int(pose.pts[a][0] - ox), int(pose.pts[a][1] - oy))
            pb = (int(pose.pts[b][0] - ox), int(pose.pts[b][1] - oy))
            cv2.line(img, pa, pb, colour, 2, cv2.LINE_AA)
    for j, p in pose.pts.items():
        c = (int(p[0] - ox), int(p[1] - oy))
        # Low-confidence joints hollow, so a confidently-wrong placement is
        # visually distinct from an uncertain one.
        conf = pose.scores.get(j, 0.0)
        cv2.circle(img, c, 3, colour, -1 if conf >= 0.3 else 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--tracks", default=os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--out", default=os.path.join(P3, "pose_audit_sheets"))
    ap.add_argument("--variant", default="rtmpose-m@192x256")
    ap.add_argument("--clips", default="", help="comma-separated; default = the pose failures")
    ap.add_argument("--n", type=int, default=6, help="frames per sheet")
    args = ap.parse_args()

    from phase2_cache_tracks import load_tracks
    from phase2_detector_benchmark import load_gt
    from phase3_pose_benchmark import gt_track, clip_plan, POSE_FAILURE_CLIPS

    tracks = load_tracks(args.tracks)
    gt = {c["clip_id"]: c for c in load_gt()["clips"]}
    want = ([c.strip() for c in args.clips.split(",") if c.strip()]
            or list(POSE_FAILURE_CLIPS))

    mp = PM.MediaPipePose()
    rtm = PM.RTMPosePose(variant=args.variant)
    os.makedirs(args.out, exist_ok=True)

    for cid in want:
        if cid not in tracks["clips"] or cid not in gt:
            print(f"{cid}: not in tracks/gt, skipped")
            continue
        ct = tracks["clips"][cid]["tracks"]
        gframes = [f for f in gt[cid]["frames"]
                   if f.get("batsman_visible") and f.get("batsman_bbox")]
        tid = gt_track(ct, gframes)
        if tid is None or tid not in ct:
            print(f"{cid}: no track matches annotated batsman")
            continue
        plan = clip_plan(ct, tid, n_sample=args.n)
        path = os.path.join(args.videos, cid)
        if not os.path.exists(path):
            continue

        want_f = {f for f, _ in plan}
        boxes = dict(plan)
        cap = cv2.VideoCapture(path)
        cols, i, last = [], -1, max(want_f)
        while i < last:
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            if i not in want_f:
                continue
            box = boxes[i]
            h, w = fr.shape[:2]
            bw, bh = box[2] - box[0], box[3] - box[1]
            pad = 0.15
            x1 = max(0, int(box[0] - pad * bw)); y1 = max(0, int(box[1] - pad * bh))
            x2 = min(w, int(box[2] + pad * bw)); y2 = min(h, int(box[3] + pad * bh))
            crop = fr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            a = draw(crop.copy(), mp.infer(fr, box), (0, 220, 255), (x1, y1))
            b = draw(crop.copy(), rtm.infer(fr, box), (0, 255, 120), (x1, y1))
            th = 260
            sc = th / max(1, a.shape[0])
            a = cv2.resize(a, (max(1, int(a.shape[1] * sc)), th))
            b = cv2.resize(b, (max(1, int(b.shape[1] * sc)), th))
            bar = np.zeros((18, a.shape[1], 3), np.uint8)
            cv2.putText(bar, f"f{i}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cols.append(np.vstack([bar, a, b]))
        cap.release()
        if not cols:
            continue
        H = max(c.shape[0] for c in cols)
        cols = [np.vstack([c, np.zeros((H - c.shape[0], c.shape[1], 3), np.uint8)])
                for c in cols]
        grid = np.hstack(cols)
        hdr = np.zeros((30, grid.shape[1], 3), np.uint8)
        cv2.putText(hdr, f"{cid}  track t{tid}   TOP=MediaPipe (yellow)   "
                         f"BOTTOM={args.variant} (green)   hollow dot = conf<0.3",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        p = os.path.join(args.out, f"{os.path.splitext(cid)[0]}.jpg")
        cv2.imwrite(p, np.vstack([hdr, grid]))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
