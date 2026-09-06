"""
phase3_semantic_features.py — per-track semantic evidence for the S0-S4
batsman-identification ablation.

WHY A SEPARATE CACHING STAGE
Every S0-S4 system must consume the SAME YOLO detections and the SAME
ByteTrack tracks; only the scoring differs. Extracting features once into a
cache guarantees that by construction — no arm can accidentally re-detect,
re-track, or re-sample and contaminate the comparison. It also makes the
ablation cheap to re-run while thresholds are being explored on dev.

FEATURE GROUPS, matching the ablation ladder
  geometry   (S0) crease/stability features, already fitted in Phase 2
  persistence(S1) how continuously the track actually exists
  skeleton   (S2) interpretable joint dynamics from pose on the track crop
  equipment  (S3) bat detection AND bat-to-person association
  action     (S4) whether the temporal structure looks like a batting stroke

DELIBERATE DISTINCTION: "moves a lot" is not batting evidence.
A stationary feeder throwing balls has high wrist speed, and Phase 1 measured
contact confidence being ANTI-correlated with subject correctness because a
bowling action produces a cleaner wrist-speed peak than a bat swing. So the
skeleton and action groups below are shaped to capture STRUCTURE (does the
wrist rise above the shoulder and then descend? do both hands move together?
does the lower body stay planted?) rather than magnitude.
"""

import argparse
import json
import math
import os

import cv2
import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

# Frames sampled per clip. Shared by every track in the clip so tracks are
# compared on identical evidence.
N_SAMPLE = 20

# MediaPipe body landmark indices.
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28

COCO_BAT, COCO_BALL = 34, 32

# Bat association thresholds. Documented starting points, not fitted optima.
BAT_ON_PERSON_OVERLAP = 0.30   # fraction of the bat box inside the person box
BAT_NEAR_HAND_TORSOS = 1.20    # bat centre within this many torso lengths of a wrist


def _safe(a, b, default=0.0):
    return a / b if b else default


def _std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def sample_frames(n_frames, k=N_SAMPLE):
    if n_frames <= 1:
        return [0]
    return sorted({int(round(i * (n_frames - 1) / (k - 1))) for i in range(k)})


# ---------------------------------------------------------------------------
# Persistence  (S1)
# ---------------------------------------------------------------------------

def persistence_features(obs, frames_processed, n_frames):
    """How continuously does this track actually exist?

    Phase 2's `coverage` counts observations but says nothing about whether
    they are contiguous. A track that flickers in and out across the whole
    clip and one that exists solidly for half of it can have identical
    coverage and very different reliability.
    """
    if not obs:
        return {"p_span": 0.0, "p_contiguity": 0.0, "p_longest_run": 0.0,
                "p_conf_stability": 0.0, "p_conf_mean": 0.0}

    frames = sorted(f for f, _, _ in obs)
    confs = [c for _, _, c in obs]
    span = (frames[-1] - frames[0]) / max(n_frames - 1, 1)

    gaps = sum(1 for a, b in zip(frames, frames[1:]) if b - a > 1)
    contiguity = 1.0 / (1.0 + gaps / max(len(frames), 1) * 10.0)

    longest, cur = 1, 1
    for a, b in zip(frames, frames[1:]):
        cur = cur + 1 if b - a <= 1 else 1
        longest = max(longest, cur)

    return {
        "p_span": min(1.0, span),
        "p_contiguity": contiguity,
        "p_longest_run": min(1.0, longest / max(frames_processed, 1)),
        "p_conf_stability": 1.0 / (1.0 + _std(confs)),
        "p_conf_mean": sum(confs) / len(confs),
    }


# ---------------------------------------------------------------------------
# Skeleton dynamics  (S2)
# ---------------------------------------------------------------------------

def _pose_metrics(landmarks, box):
    """Normalised joint positions in the crop, in torso units."""
    if landmarks is None:
        return None
    w = box[2] - box[0]
    h = box[3] - box[1]
    if w <= 0 or h <= 0:
        return None

    def pt(i):
        lm = landmarks[i]
        return (lm.x * w, lm.y * h)

    lsh, rsh = pt(L_SH), pt(R_SH)
    lhip, rhip = pt(L_HIP), pt(R_HIP)
    msh = ((lsh[0] + rsh[0]) / 2, (lsh[1] + rsh[1]) / 2)
    mhip = ((lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2)
    torso = math.hypot(msh[0] - mhip[0], msh[1] - mhip[1])
    if torso < 1e-6:
        return None

    lwr, rwr = pt(L_WR), pt(R_WR)
    lank, rank = pt(L_ANK), pt(R_ANK)
    # Shoulder-line angle: a rotation proxy that needs no 3D.
    sh_angle = math.degrees(math.atan2(rsh[1] - lsh[1], rsh[0] - lsh[0]))
    # Wrist height ABOVE the shoulder line, in torso units. Positive means the
    # hands are raised — the backlift signature.
    lift = (msh[1] - min(lwr[1], rwr[1])) / torso

    return {
        "torso": torso,
        "wrist_mid": ((lwr[0] + rwr[0]) / 2 / torso, (lwr[1] + rwr[1]) / 2 / torso),
        "lwr": (lwr[0] / torso, lwr[1] / torso),
        "rwr": (rwr[0] / torso, rwr[1] / torso),
        "hip_mid": (mhip[0] / torso, mhip[1] / torso),
        "ankle_mid": ((lank[0] + rank[0]) / 2 / torso, (lank[1] + rank[1]) / 2 / torso),
        "sh_angle": sh_angle,
        "wrist_lift": lift,
        "vis": float(np.mean([getattr(landmarks[i], "visibility", 0.0)
                              for i in (L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP)])),
    }


def skeleton_features(seq):
    """Interpretable joint dynamics over the sampled frames of one track.

    `seq` is a list of _pose_metrics dicts (or None where pose failed).
    """
    m = [s for s in seq if s]
    if len(m) < 3:
        return {"k_pose_rate": _safe(len(m), len(seq)), "k_wrist_excursion": 0.0,
                "k_wrist_lift_range": 0.0, "k_shoulder_rotation": 0.0,
                "k_lower_stability": 0.0, "k_bilateral": 0.0, "k_visibility": 0.0}

    def disp(key):
        pts = [s[key] for s in m]
        return max(math.hypot(b[0] - a[0], b[1] - a[1]) for a in pts for b in pts)

    lifts = [s["wrist_lift"] for s in m]
    angles = [s["sh_angle"] for s in m]

    # Bilateral coordination: do the two wrists move together? A two-handed
    # bat swing couples them; a one-armed throw does not.
    lsp = [math.hypot(b["lwr"][0] - a["lwr"][0], b["lwr"][1] - a["lwr"][1])
           for a, b in zip(m, m[1:])]
    rsp = [math.hypot(b["rwr"][0] - a["rwr"][0], b["rwr"][1] - a["rwr"][1])
           for a, b in zip(m, m[1:])]
    if len(lsp) >= 2 and _std(lsp) > 1e-9 and _std(rsp) > 1e-9:
        ml, mr = sum(lsp) / len(lsp), sum(rsp) / len(rsp)
        cov = sum((a - ml) * (b - mr) for a, b in zip(lsp, rsp)) / len(lsp)
        bilateral = max(0.0, min(1.0, (cov / (_std(lsp) * _std(rsp)) + 1) / 2))
    else:
        bilateral = 0.5

    # Lower body: a batsman plays from a crease, so ankles stay planted even
    # while the arms swing. Inverted so higher = more stable.
    ankle_travel = disp("ankle_mid")

    return {
        "k_pose_rate": _safe(len(m), len(seq)),
        "k_wrist_excursion": min(1.0, disp("wrist_mid") / 4.0),
        "k_wrist_lift_range": min(1.0, (max(lifts) - min(lifts)) / 2.0),
        "k_shoulder_rotation": min(1.0, (max(angles) - min(angles)) / 90.0),
        "k_lower_stability": 1.0 / (1.0 + ankle_travel),
        "k_bilateral": bilateral,
        "k_visibility": float(np.mean([s["vis"] for s in m])),
    }


# ---------------------------------------------------------------------------
# Equipment  (S3)
# ---------------------------------------------------------------------------

def equipment_features(bat_events, seq, n_sampled):
    """Bat evidence at four increasing levels of specificity, so the ablation
    can show WHICH level (if any) actually discriminates:

        e_bat_anywhere   a bat exists in the frame at all
        e_bat_on_person  the bat overlaps THIS track's box
        e_bat_near_hand  the bat is near this track's wrists
        e_bat_persist    the association holds across time

    Detecting a bat is not identifying a batsman: a feeder standing beside a
    bat, or a bat lying on the ground, both fire the first level.
    """
    if not n_sampled:
        return {"e_bat_anywhere": 0.0, "e_bat_on_person": 0.0,
                "e_bat_near_hand": 0.0, "e_bat_persist": 0.0, "e_bat_conf": 0.0}

    any_n = sum(1 for b in bat_events if b["any"])
    on_n = sum(1 for b in bat_events if b["on_person"])
    hand_n = sum(1 for b in bat_events if b["near_hand"])
    confs = [b["conf"] for b in bat_events if b["on_person"]]

    # Persistence: longest run of consecutive sampled frames with the bat on
    # this person. A single frame's association is noise; a sustained one is
    # evidence.
    longest, cur = 0, 0
    for b in bat_events:
        cur = cur + 1 if b["on_person"] else 0
        longest = max(longest, cur)

    return {
        "e_bat_anywhere": _safe(any_n, n_sampled),
        "e_bat_on_person": _safe(on_n, n_sampled),
        "e_bat_near_hand": _safe(hand_n, n_sampled),
        "e_bat_persist": _safe(longest, n_sampled),
        "e_bat_conf": (sum(confs) / len(confs)) if confs else 0.0,
    }


# ---------------------------------------------------------------------------
# Batting action  (S4)
# ---------------------------------------------------------------------------

def action_features(seq):
    """Does the temporal structure look like a batting stroke?

    The signature sought is a RISE-THEN-FALL in wrist height relative to the
    shoulders (backlift then downswing), with the lower body comparatively
    quiet. Magnitude of motion is explicitly not the signal — that is what a
    bowling action maximises.
    """
    m = [s for s in seq if s]
    if len(m) < 5:
        return {"a_backlift": 0.0, "a_downswing": 0.0,
                "a_updown_shape": 0.0, "a_upper_lower_ratio": 0.0}

    lifts = [s["wrist_lift"] for s in m]
    peak = int(np.argmax(lifts))

    # Backlift: how far the hands rise from their lowest pre-peak point.
    pre = lifts[:peak + 1]
    backlift = (max(pre) - min(pre)) if len(pre) >= 2 else 0.0

    # Downswing: how far they fall after the peak.
    post = lifts[peak:]
    downswing = (max(post) - min(post)) if len(post) >= 2 else 0.0

    # Shape: the peak should sit INSIDE the window, with real rise and fall on
    # both sides. A monotonic profile (someone lifting a bat and holding it,
    # or a single throwing motion) scores low.
    interior = 1.0 - abs((peak / max(len(lifts) - 1, 1)) - 0.5) * 2.0
    shape = interior * min(1.0, backlift) * min(1.0, downswing)

    # Upper body should move more than the lower body during a stroke played
    # from a crease.
    wr = [math.hypot(b["wrist_mid"][0] - a["wrist_mid"][0],
                     b["wrist_mid"][1] - a["wrist_mid"][1]) for a, b in zip(m, m[1:])]
    an = [math.hypot(b["ankle_mid"][0] - a["ankle_mid"][0],
                     b["ankle_mid"][1] - a["ankle_mid"][1]) for a, b in zip(m, m[1:])]
    upper, lower = sum(wr) / len(wr), sum(an) / len(an)
    ratio = upper / (upper + lower) if (upper + lower) > 1e-9 else 0.5

    return {
        "a_backlift": min(1.0, backlift / 1.5),
        "a_downswing": min(1.0, downswing / 1.5),
        "a_updown_shape": max(0.0, min(1.0, shape)),
        "a_upper_lower_ratio": max(0.0, min(1.0, ratio)),
    }


# ---------------------------------------------------------------------------
# Extraction driver
# ---------------------------------------------------------------------------

def overlap_frac(small, big):
    ix1, iy1 = max(small[0], big[0]), max(small[1], big[1])
    ix2, iy2 = min(small[2], big[2]), min(small[3], big[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    a = (small[2] - small[0]) * (small[3] - small[1])
    return (iw * ih) / a if a > 0 else 0.0


def extract(video_dir, tracks_path, out_path, pad=0.12):
    from phase2_cache_tracks import load_tracks
    from phase2_batsman_score import track_features
    import zero_storage_pipeline as zsp
    from ultralytics import YOLO

    data = load_tracks(tracks_path)
    equip = YOLO("yolo11x.pt")
    out = {}

    for ci, (cid, c) in enumerate(data["clips"].items(), 1):
        path = os.path.join(video_dir, cid)
        if not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sample_frames(n_frames)

        # ONE sequential decode of the sampled frames, shared by all tracks.
        grabbed, i, last = {}, -1, max(idxs)
        while i < last:
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            if i in grabbed or i in idxs:
                grabbed[i] = fr
        cap.release()

        # Equipment detection once per sampled frame, shared by all tracks.
        present = [(f, grabbed[f]) for f in idxs if f in grabbed]
        bats_by_frame = {}
        if present:
            preds = equip.predict([fr for _, fr in present], imgsz=960,
                                  classes=[COCO_BAT], conf=0.05,
                                  verbose=False, device=0)
            for (f, _), r in zip(present, preds):
                b = r.boxes
                bats_by_frame[f] = ([] if b is None or len(b) == 0 else
                                    [(list(map(float, xy)), float(cf))
                                     for xy, cf in zip(b.xyxy.cpu().numpy(),
                                                       b.conf.cpu().numpy())])

        ctx = {"frame_w": c["frame_w"], "frame_h": c["frame_h"],
               "frames_processed": c["frames_processed"]}
        clip_out = {"split": c["split"], "n_frames": n_frames,
                    "sampled": idxs, "tracks": {}}

        for tid, obs in c["tracks"].items():
            by_frame = {f: b for f, b, _ in obs}
            seq, bat_events = [], []

            for f in idxs:
                box = by_frame.get(f)
                fr = grabbed.get(f)
                if box is None or fr is None:
                    seq.append(None)
                    bat_events.append({"any": bool(bats_by_frame.get(f)),
                                       "on_person": False, "near_hand": False,
                                       "conf": 0.0})
                    continue

                h, w = fr.shape[:2]
                bw, bh = box[2] - box[0], box[3] - box[1]
                x1 = max(0, int(box[0] - pad * bw)); y1 = max(0, int(box[1] - pad * bh))
                x2 = min(w, int(box[2] + pad * bw)); y2 = min(h, int(box[3] + pad * bh))
                crop = fr[y1:y2, x1:x2]
                if crop.size == 0:
                    seq.append(None)
                    bat_events.append({"any": False, "on_person": False,
                                       "near_hand": False, "conf": 0.0})
                    continue

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                with zsp._DETECTOR_LOCK:
                    cands = zsp._detect_candidates(zsp._get_shared_detector_locked(), [rgb])
                lm = cands[0][0] if cands[0] else None
                pm = _pose_metrics(lm, (x1, y1, x2, y2))
                seq.append(pm)

                # Bat association for THIS track at THIS frame.
                bats = bats_by_frame.get(f, [])
                on_person, near_hand, best_conf = False, False, 0.0
                for bbox, bconf in bats:
                    if overlap_frac(bbox, box) >= BAT_ON_PERSON_OVERLAP:
                        on_person = True
                        best_conf = max(best_conf, bconf)
                        if pm:
                            bcx = (bbox[0] + bbox[2]) / 2 - x1
                            bcy = (bbox[1] + bbox[3]) / 2 - y1
                            t = pm["torso"]
                            for key in ("lwr", "rwr"):
                                wx, wy = pm[key][0] * t, pm[key][1] * t
                                if math.hypot(bcx - wx, bcy - wy) <= BAT_NEAR_HAND_TORSOS * t:
                                    near_hand = True
                bat_events.append({"any": bool(bats), "on_person": on_person,
                                   "near_hand": near_hand, "conf": best_conf})

            feats = {}
            feats.update(track_features(obs, ctx["frame_w"], ctx["frame_h"],
                                        ctx["frames_processed"]))
            feats.update(persistence_features(obs, ctx["frames_processed"], n_frames))
            feats.update(skeleton_features(seq))
            feats.update(equipment_features(bat_events, seq, len(idxs)))
            feats.update(action_features(seq))
            clip_out["tracks"][str(tid)] = {
                "n_obs": len(obs),
                "features": {k: round(float(v), 5) for k, v in feats.items()},
            }

        clip_out["context"] = ctx
        out[cid] = clip_out
        if ci % 10 == 0:
            print(f"  {ci}/{len(data['clips'])} clips")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n_sample": N_SAMPLE,
                   "bat_on_person_overlap": BAT_ON_PERSON_OVERLAP,
                   "bat_near_hand_torsos": BAT_NEAR_HAND_TORSOS,
                   "clips": out}, f)
    n_tracks = sum(len(v["tracks"]) for v in out.values())
    print(f"wrote {out_path}  ({len(out)} clips, {n_tracks} tracks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--tracks", default=os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--out", default=os.path.join(P3, "phase3_semantic_features.json"))
    args = ap.parse_args()
    os.makedirs(P3, exist_ok=True)
    extract(args.videos, args.tracks, args.out)


if __name__ == "__main__":
    main()
