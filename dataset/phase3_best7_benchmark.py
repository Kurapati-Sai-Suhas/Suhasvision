"""
phase3_best7_benchmark.py — event search region, candidate pool, and the
F0-F4 Best-7 ablation.

INFERENCE-TIME INFORMATION ONLY
The event region, the contact anchor and the candidate pool are built from
L3's motion prior and the C0 contact proposal — both machine signals. Ground
truth is used to SCORE the result and never to choose frames. The GT-window
robustness probe from the L-stage is diagnostic and is not used here.

EVENT SEARCH REGION (deliberately wide)
L3 is a soft prior, not a crop: its recall@IoU0.5 was 0.36, so hard-cropping to
it would discard the true shot on most clips. The region is therefore the UNION
of L3's interval and a contact-centred window, padded, and clamped to the clip.
The objective at this stage is RECALL — does the region contain the true shot —
not narrowness.

MOTION IS NOT QUALITY
The candidate pool carries them as separate fields. `motion` says a frame is
informative about WHEN; `sharpness` (variance of Laplacian) and `pose_quality`
say whether it is usable for BIOMECHANICS. A fast frame can be motion-blurred
and useless for joint angles, which is exactly the tension F4 has to resolve.
"""

import argparse
import json
import os
import statistics
import time

import numpy as np

import phase3_contact_signals as CS
import phase3_frame_selection as FS
import phase3_shot_localization as SL

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

REGION_PAD = 8          # frames added either side of the union
CONTACT_HALF = 16       # half-width of the contact-centred window
POOL_TARGET = 40        # candidates per clip, within 20-50


def event_region(clip, l3p, contact, pad=REGION_PAD, half=CONTACT_HALF):
    """Union of the L3 prior and a contact-centred window, padded."""
    n = clip["n_frames"]
    lo, hi = None, None
    if l3p is not None and not l3p["rejected"]:
        lo, hi = l3p["start"], l3p["end"]
    if contact is not None:
        clo, chi = contact - half, contact + half
        lo = clo if lo is None else min(lo, clo)
        hi = chi if hi is None else max(hi, chi)
    if lo is None:
        return (0, max(1, n - 1), "whole_clip_fallback")
    lo = max(0, lo - pad)
    hi = min(n - 1, hi + pad)
    if hi - lo < 14:
        c = (lo + hi) // 2
        lo, hi = max(0, c - 10), min(n - 1, c + 10)
    return (lo, hi, "l3_union_contact")


def build_pool(clip, region, target=POOL_TARGET):
    """Candidates inside the region, with full metadata.

    Frames without a batsman box are excluded: a frame where the tracked
    person is absent cannot contribute a biomechanical pose, and including it
    would let a method "select" a frame that has no subject.
    """
    lo, hi = region[0], region[1]
    frames = [f for f in clip["frames"] if lo <= f["frame"] <= hi and f["has_box"]]
    if not frames:
        return []
    if len(frames) > target:
        idx = np.linspace(0, len(frames) - 1, target).round().astype(int)
        frames = [frames[i] for i in sorted(set(idx))]

    blurs = [f["blur"] for f in frames if f["blur"] is not None]
    bmax = max(blurs) if blurs else 1.0
    mots = [f["local_motion"] for f in frames if f["local_motion"] is not None]
    mmax = max(mots) if mots else 1.0

    pool = []
    for f in frames:
        p = f.get("pose") or {}
        vis = p.get("pose_visibility", 0.0)
        minvis = p.get("min_visibility", 0.0)
        pool.append(FS.Candidate(
            frame=f["frame"], t=f["t"],
            # Pose quality is landmark-based, NOT motion-based.
            pose_quality=(vis if p else 0.0),
            sharpness=((f["blur"] / bmax) if (f["blur"] is not None and bmax > 0) else 0.0),
            identity_conf=1.0 if f["has_box"] else 0.0,
            motion=((f["local_motion"] / mmax)
                    if (f["local_motion"] is not None and mmax > 0) else 0.0),
            visibility=vis,
            # Low minimum visibility means at least one tracked joint is
            # hidden — a better occlusion proxy than mean visibility.
            occlusion=float(max(0.0, 1.0 - minvis)),
            temporal_stability=1.0,
            bat_conf=((f["bat"] or {}).get("conf", 0.0)),
            extra={"has_pose": bool(p), "wrist_lift": p.get("wrist_lift")},
        ))
    return pool


def sequence_metrics(sel, contact, region, fps):
    """Structural properties of one selected 7-frame sequence."""
    if not sel:
        return {"n": 0, "valid": False}
    fr = [c.frame for c in sel]
    gaps = [b - a for a, b in zip(fr, fr[1:])]
    pre = sum(1 for f in fr if contact is not None and f < contact)
    post = sum(1 for f in fr if contact is not None and f > contact)
    near = sum(1 for f in fr if contact is not None and abs(f - contact) <= 3)
    return {
        "n": len(fr),
        "frames": fr,
        "duplicates": len(fr) - len(set(fr)),
        "chronological": all(b > a for a, b in zip(fr, fr[1:])),
        "min_gap": min(gaps) if gaps else 0,
        "span": fr[-1] - fr[0],
        "temporal_diversity": round(float(np.std(fr)), 2) if len(fr) > 1 else 0.0,
        "pre_contact": pre, "post_contact": post, "contact_adjacent": near,
        "mean_pose_quality": round(float(np.mean([c.pose_quality for c in sel])), 4),
        "min_pose_quality": round(float(min(c.pose_quality for c in sel)), 4),
        "mean_sharpness": round(float(np.mean([c.sharpness for c in sel])), 4),
        "mean_occlusion": round(float(np.mean([c.occlusion for c in sel])), 4),
        "n_without_pose": sum(1 for c in sel if not c.extra.get("has_pose")),
        "inside_region": all(region[0] <= f <= region[1] for f in fr),
    }


def biomech_quality(clip_id, frames_sel, video_dir, cache_clip, pose_cache):
    """The actual (7,30) question: is this a better biomechanical sequence?

    Re-runs pose on the selected frames to obtain full landmarks, then builds
    the production 15 angles and their frame-to-frame velocities.
    """
    import cv2
    import phase3_pose_models as PM
    import phase3_semantic_features as SF

    by_frame = {f["frame"]: f for f in cache_clip["frames"]}
    need = [f for f in frames_sel if (clip_id, f) not in pose_cache]
    if need:
        path = os.path.join(video_dir, clip_id)
        cap = cv2.VideoCapture(path)
        want = set(need)
        i = -1
        while i < max(want):
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            if i not in want:
                continue
            rec = by_frame.get(i)
            res = None
            if rec and rec["box"]:
                b = rec["box"]
                h, w = fr.shape[:2]
                bw, bh = b[2] - b[0], b[3] - b[1]
                x1 = max(0, int(b[0] - 0.12 * bw)); y1 = max(0, int(b[1] - 0.12 * bh))
                x2 = min(w, int(b[2] + 0.12 * bw)); y2 = min(h, int(b[3] + 0.12 * bh))
                if x2 > x1 and y2 > y1:
                    res = PM.MediaPipePose().infer(fr, (b[0], b[1], b[2], b[3]))
            pose_cache[(clip_id, i)] = res
        cap.release()

    seq = [pose_cache.get((clip_id, f)) for f in frames_sel]
    angs = [PM.fifteen_angles(p) if (p and p.ok) else None for p in seq]
    n_missing = sum(1 for a in angs if a is None)
    have = [a for a in angs if a]
    if len(have) < 2:
        return {"n_frames_with_angles": len(have), "missing_frames": n_missing,
                "usable": False}

    keys = sorted(have[0].keys())
    invalid = sum(1 for a in have for k in keys if a.get(k) is None)
    implausible = sum(1 for a in have for k in keys
                      if a.get(k) is not None and k.startswith(("angle_knee", "angle_elbow"))
                      and (a[k] < 15 or a[k] > 179))
    # Velocities between CONSECUTIVE SELECTED frames — the second half of the
    # (7,30) tensor. Duplicate or near-duplicate frames show up here as
    # fabricated zero velocity.
    vels, jumps, zero_vel = [], 0, 0
    for a, b in zip(angs, angs[1:]):
        if not (a and b):
            continue
        d = [abs(b[k] - a[k]) for k in keys
             if a.get(k) is not None and b.get(k) is not None]
        if not d:
            continue
        vels.append(float(np.mean(d)))
        jumps += sum(1 for x in d if x > 60)
        if max(d) < 1e-6:
            zero_vel += 1
    total = len(have) * len(keys)
    return {
        "usable": True,
        "n_frames_with_angles": len(have),
        "missing_frames": n_missing,
        "invalid_angle_rate": round(invalid / total, 4) if total else None,
        "implausible_angle_rate": round(implausible / total, 4) if total else None,
        "mean_angular_velocity": round(float(np.mean(vels)), 3) if vels else None,
        "velocity_stability": (round(float(np.std(vels)), 3) if len(vels) > 1 else None),
        "abrupt_jumps": jumps,
        "fabricated_zero_velocity_steps": zero_vel,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--cache", default=os.path.join(P3, "phase3_event_cache.json"))
    ap.add_argument("--bench", default=os.path.join(P3, "phase3_shot_localization.json"))
    ap.add_argument("--contact", default=os.path.join(P3, "phase3_contact_results.json"))
    ap.add_argument("--out", default=os.path.join(P3, "phase3_best7_summary.json"))
    ap.add_argument("--jsonl", default=os.path.join(P3, "phase3_best7_results.jsonl"))
    ap.add_argument("--no-biomech", action="store_true")
    args = ap.parse_args()

    import phase3_contact_benchmark as CB
    cache, recs, params = CB.load_context(args.cache, args.bench)
    est = CB.contact_estimates(cache)

    pose_cache = {}
    rows, timings = [], {k: [] for k in FS.STRATEGIES}
    region_hit = region_widths = 0
    region_stats = []

    for cid, clip in cache["clips"].items():
        rec = recs.get(cid, {})
        frames = clip["frames"]
        g, l = CS.global_motion(frames), CS.local_motion(frames)
        l3p = SL.predict("L3_hybrid", clip["n_frames"], g, l, **params) if g is not None else None
        contact = est[cid]["C0_current_proposal"]["frame"]
        region = event_region(clip, l3p, contact)
        pool = build_pool(clip, region)

        gt = None
        if rec.get("coherence") == "VALID_SINGLE_SHOT" and rec.get("shot_start_frame") is not None:
            gt = (rec["shot_start_frame"], rec["shot_end_frame"])
            region_widths += (region[1] - region[0])
            if region[0] <= gt[0] and region[1] >= gt[1]:
                region_hit += 1
            region_stats.append({
                "clip_id": cid, "split": clip["split"],
                "region": [region[0], region[1]], "gt": list(gt),
                "contains_gt": bool(region[0] <= gt[0] and region[1] >= gt[1]),
                "width": region[1] - region[0],
                "gt_width": gt[1] - gt[0],
                "iou": round(SL.temporal_iou((region[0], region[1]), gt), 4),
            })

        row = {"clip_id": cid, "split": clip["split"],
               "identity_correct": clip["identity_correct"],
               "coherence": rec.get("coherence"),
               "gt_shot": list(gt) if gt else None,
               "gt_contact": rec.get("contact_frame"),
               "contact_type": rec.get("contact_type"),
               "l3": (None if (l3p is None or l3p["rejected"])
                      else [l3p["start"], l3p["end"]]),
               "l3_confidence": (l3p or {}).get("prominence"),
               "contact_estimate": contact,
               "region": [region[0], region[1]], "region_source": region[2],
               "pool_size": len(pool), "methods": {}}

        for name, fn in FS.STRATEGIES.items():
            if not pool:
                row["methods"][name] = {"selected": None, "reason": "empty_pool"}
                continue
            onset, end = region[0], region[1]
            t0 = time.perf_counter()
            try:
                sel = fn(pool, onset, contact if contact is not None else (onset + end) // 2,
                         end, fps=clip["fps"])
            except Exception as e:                      # noqa: BLE001
                row["methods"][name] = {"selected": None, "reason": f"error:{e}"}
                continue
            timings[name].append(time.perf_counter() - t0)
            # Strategies return {"ok", "frames": [frame indices], "reason"};
            # map back to the pooled candidates to recover their metadata.
            if isinstance(sel, dict):
                idx = sel.get("frames") or []
                reason = sel.get("reason", "") if not sel.get("ok") else ""
            else:
                idx, reason = list(sel or []), ""
            by_f = {c.frame: c for c in pool}
            cands = [by_f[i] for i in (idx or []) if i in by_f]
            m = sequence_metrics(cands, contact, region, clip["fps"])
            entry = {"selected": m.get("frames"), "reason": reason, **m}
            if not args.no_biomech and m.get("frames"):
                entry["biomech"] = biomech_quality(cid, m["frames"], args.videos,
                                                   clip, pose_cache)
            row["methods"][name] = entry
        rows.append(row)

    with open(args.jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---- aggregate ----
    def agg(name, key, subset=None, of="methods"):
        vals = []
        for r in rows:
            if subset and not subset(r):
                continue
            m = r["methods"].get(name) or {}
            v = m.get(key) if of == "methods" else (m.get("biomech") or {}).get(key)
            if isinstance(v, bool):
                v = float(v)
            if isinstance(v, (int, float)):
                vals.append(v)
        return round(float(np.mean(vals)), 4) if vals else None

    valid = lambda r: r["coherence"] == "VALID_SINGLE_SHOT"
    summary = {
        "n_clips": len(rows),
        "pool": {"target": POOL_TARGET,
                 "mean_size": round(float(np.mean([r["pool_size"] for r in rows])), 1),
                 "min": min(r["pool_size"] for r in rows),
                 "max": max(r["pool_size"] for r in rows)},
        "event_region": {
            "n_valid_clips": len(region_stats),
            "contains_gt_shot_rate": round(region_hit / len(region_stats), 4) if region_stats else None,
            "mean_width": round(region_widths / len(region_stats), 1) if region_stats else None,
            "mean_gt_width": round(float(np.mean([s["gt_width"] for s in region_stats])), 1) if region_stats else None,
            "mean_iou_with_gt": round(float(np.mean([s["iou"] for s in region_stats])), 4) if region_stats else None,
            "note": ("Optimised for RECALL, not narrowness: a wide region that "
                     "contains the shot is the correct objective before Best-7."),
            "per_clip": region_stats,
        },
        "methods": {},
    }
    struct_keys = ("duplicates", "chronological", "min_gap", "span",
                   "temporal_diversity", "pre_contact", "post_contact",
                   "contact_adjacent", "mean_pose_quality", "min_pose_quality",
                   "mean_sharpness", "mean_occlusion", "n_without_pose",
                   "inside_region")
    bio_keys = ("n_frames_with_angles", "missing_frames", "invalid_angle_rate",
                "implausible_angle_rate", "mean_angular_velocity",
                "velocity_stability", "abrupt_jumps",
                "fabricated_zero_velocity_steps")
    for name in FS.STRATEGIES:
        e = {"structural": {k: agg(name, k, valid) for k in struct_keys},
             "biomechanical": {k: agg(name, k, valid, of="bio") for k in bio_keys},
             "n_selected": sum(1 for r in rows if valid(r) and (r["methods"].get(name) or {}).get("frames")),
             "n_refused": sum(1 for r in rows if valid(r) and not (r["methods"].get(name) or {}).get("frames")),
             "latency_ms_median": (round(1000 * statistics.median(timings[name]), 4)
                                   if timings[name] else None)}
        e["structural_identity_correct"] = {
            k: agg(name, k, lambda r: valid(r) and r["identity_correct"])
            for k in ("duplicates", "min_gap", "contact_adjacent", "mean_pose_quality")}
        summary["methods"][name] = e

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    print(f"clips {len(rows)}   pool mean {summary['pool']['mean_size']} "
          f"(min {summary['pool']['min']}, max {summary['pool']['max']})")
    er = summary["event_region"]
    print(f"event region: contains GT shot {er['contains_gt_shot_rate']:.2%} "
          f"of {er['n_valid_clips']} valid clips; mean width {er['mean_width']} "
          f"vs GT width {er['mean_gt_width']}")
    print(f"\n{'method':<22}{'sel':>5}{'ref':>5}{'dup':>6}{'minGap':>8}{'ctAdj':>7}"
          f"{'preC':>6}{'postC':>7}{'poseQ':>7}{'sharp':>7}{'jumps':>7}{'zeroV':>7}{'ms':>8}")
    for name, e in summary["methods"].items():
        s, b = e["structural"], e["biomechanical"]
        f_ = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
        print(f"{name:<22}{e['n_selected']:>5}{e['n_refused']:>5}"
              f"{f_(s['duplicates']):>6}{f_(s['min_gap'],1):>8}"
              f"{f_(s['contact_adjacent'],1):>7}{f_(s['pre_contact'],1):>6}"
              f"{f_(s['post_contact'],1):>7}{f_(s['mean_pose_quality'],3):>7}"
              f"{f_(s['mean_sharpness'],3):>7}{f_(b['abrupt_jumps'],1):>7}"
              f"{f_(b['fabricated_zero_velocity_steps'],2):>7}"
              f"{e['latency_ms_median']:>8.3f}")
    print(f"\nwrote {args.out}\nwrote {args.jsonl}")


if __name__ == "__main__":
    main()
