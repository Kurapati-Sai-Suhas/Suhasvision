"""
phase3e_region_best7.py — R0-R4 event-region constructions, and F0-F4
re-evaluated with the CORRECTED L3 smoothing.

REGION OBJECTIVE
Two competing quantities, reported together and never collapsed into one score:

    containment  does the region contain the whole annotated shot?
    width        how much irrelevant footage does it drag in?

Phase 3D used pad(L3 U [C0+/-16], 8) and got 100% containment at 2.8x the true
width. That was the right objective THEN (recall before selection) but it is
also what let F4 select high-quality frames from outside the event. R0-R4 make
the trade-off explicit so the choice is measured rather than inherited.

THREE OBJECTIVES, KEPT SEPARATE
  1 event faithfulness    are the seven frames from the batting event?
  2 biomechanical validity are they good pose observations?
  3 downstream usefulness  do the (7,30) tensors score better?

This script measures 1 and 2. Objective 3 needs scoring-model retraining, which
is explicitly out of scope, so no claim about it is made anywhere.

CONFOUND CARRIED FORWARD
Regions R2-R4 use the C0 contact proposal and are scored against annotations
made by someone who saw C0. phase3e_leakage_analysis finds the two annotation
passes agree with each other (mean 0.60 frames) more closely than either agrees
with C0 (1.10 / 1.50), which argues against strong leakage — but it is not a
blind study, so contact-based containment numbers remain optimistic.
"""

import argparse
import json
import os

import numpy as np

import phase3_best7_benchmark as B7
import phase3_contact_signals as CS
import phase3_frame_selection as FS
import phase3_shot_localization as SL

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

REGIONS = ("R0_l3_only", "R1_l3_padded", "R2_l3_union_contact",
           "R3_contact_centred", "R4_hybrid_current")


def build_region(kind, clip, l3p, contact):
    n = clip["n_frames"]
    have_l3 = l3p is not None and not l3p["rejected"]
    if kind == "R0_l3_only":
        return ((l3p["start"], l3p["end"]) if have_l3 else (0, max(1, n - 1)))
    if kind == "R1_l3_padded":
        if not have_l3:
            return (0, max(1, n - 1))
        return (max(0, l3p["start"] - 12), min(n - 1, l3p["end"] + 12))
    if kind == "R3_contact_centred":
        if contact is None:
            return ((l3p["start"], l3p["end"]) if have_l3 else (0, max(1, n - 1)))
        return (max(0, contact - 18), min(n - 1, contact + 18))
    if kind == "R2_l3_union_contact":
        lo, hi = B7.event_region(clip, l3p, contact, pad=0, half=16)[:2]
        return (lo, hi)
    lo, hi = B7.event_region(clip, l3p, contact)[:2]      # R4, the current one
    return (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--cache", default=os.path.join(P3, "phase3_event_cache.json"))
    ap.add_argument("--bench", default=os.path.join(P3, "phase3_shot_localization_corrected.json"))
    ap.add_argument("--out", default=os.path.join(P3, "phase3e_event_validation.json"))
    ap.add_argument("--best7-out", default=os.path.join(P3, "phase3e_best7_blind_eval.json"))
    args = ap.parse_args()

    import phase3_contact_benchmark as CB
    cache, recs, params = CB.load_context(args.cache, args.bench)
    est = CB.contact_estimates(cache)

    # ---- per-clip machine signals, computed once ----
    ctx = {}
    for cid, clip in cache["clips"].items():
        frames = clip["frames"]
        g, l = CS.global_motion(frames), CS.local_motion(frames)
        l3p = (SL.predict("L3_hybrid", clip["n_frames"], g, l, **params)
               if g is not None else None)
        ctx[cid] = {"clip": clip, "l3": l3p,
                    "contact": est[cid]["C0_current_proposal"]["frame"],
                    "rec": recs.get(cid, {})}

    # ---- R0-R4 ----
    region_out = {}
    for kind in REGIONS:
        cont, widths, ious, det = 0, [], [], []
        n = 0
        for cid, c in ctx.items():
            r = c["rec"]
            if r.get("coherence") != "VALID_SINGLE_SHOT" or r.get("shot_start_frame") is None:
                continue
            n += 1
            gt = (r["shot_start_frame"], r["shot_end_frame"])
            lo, hi = build_region(kind, c["clip"], c["l3"], c["contact"])
            ok = lo <= gt[0] and hi >= gt[1]
            cont += int(ok)
            widths.append(hi - lo)
            ious.append(SL.temporal_iou((lo, hi), gt))
            det.append({"clip_id": cid, "region": [lo, hi], "gt": list(gt),
                        "contains": bool(ok), "width": hi - lo})
        region_out[kind] = {
            "n_valid": n,
            "containment": round(cont / n, 4) if n else None,
            "mean_width": round(float(np.mean(widths)), 1) if widths else None,
            "median_width": round(float(np.median(widths)), 1) if widths else None,
            "mean_iou_with_gt": round(float(np.mean(ious)), 4) if ious else None,
            "width_ratio_vs_gt": (round(float(np.mean(widths)) / 22.7, 2)
                                  if widths else None),
            "detail": det,
        }

    # ---- F0-F4 on the best-containment region, corrected L3 ----
    best_region = max(REGIONS, key=lambda k: (region_out[k]["containment"] or 0,
                                              -(region_out[k]["mean_width"] or 1e9)))
    f_out = {}
    per_clip = []
    for cid, c in ctx.items():
        clip = c["clip"]
        lo, hi = build_region(best_region, clip, c["l3"], c["contact"])
        pool = B7.build_pool(clip, (lo, hi))
        r = c["rec"]
        gt = ((r["shot_start_frame"], r["shot_end_frame"])
              if r.get("coherence") == "VALID_SINGLE_SHOT"
              and r.get("shot_start_frame") is not None else None)
        row = {"clip_id": cid, "split": clip["split"],
               "identity_correct": clip["identity_correct"],
               "gt_shot": list(gt) if gt else None,
               "region": [lo, hi], "pool_size": len(pool), "methods": {}}
        for name, fn in FS.STRATEGIES.items():
            if not pool:
                row["methods"][name] = {"frames": None}
                continue
            try:
                sel = fn(pool, lo, c["contact"] if c["contact"] is not None else (lo + hi) // 2,
                         hi, fps=clip["fps"])
            except Exception:                              # noqa: BLE001
                row["methods"][name] = {"frames": None}
                continue
            idx = sel.get("frames") if isinstance(sel, dict) else list(sel or [])
            by_f = {x.frame: x for x in pool}
            cands = [by_f[i] for i in (idx or []) if i in by_f]
            m = B7.sequence_metrics(cands, c["contact"], (lo, hi), clip["fps"])
            entry = {"frames": m.get("frames"),
                     "mean_pose_quality": m.get("mean_pose_quality"),
                     "min_pose_quality": m.get("min_pose_quality"),
                     "min_gap": m.get("min_gap"),
                     "contact_adjacent": m.get("contact_adjacent")}
            if gt and m.get("frames"):
                fr = m["frames"]
                entry["in_shot_fraction"] = round(
                    sum(1 for f in fr if gt[0] <= f <= gt[1]) / len(fr), 4)
                entry["median_distance_to_shot"] = float(np.median(
                    [0 if gt[0] <= f <= gt[1] else min(abs(f - gt[0]), abs(f - gt[1]))
                     for f in fr]))
            row["methods"][name] = entry
        per_clip.append(row)

    def agg(name, key, subset=None):
        v = [(r["methods"].get(name) or {}).get(key) for r in per_clip
             if (subset is None or subset(r))]
        v = [x for x in v if isinstance(x, (int, float))]
        return round(float(np.mean(v)), 4) if v else None

    valid = lambda r: r["gt_shot"] is not None
    for name in FS.STRATEGIES:
        f_out[name] = {
            "in_shot_fraction": agg(name, "in_shot_fraction", valid),
            "median_distance_to_shot": agg(name, "median_distance_to_shot", valid),
            "mean_pose_quality": agg(name, "mean_pose_quality", valid),
            "min_pose_quality": agg(name, "min_pose_quality", valid),
            "min_gap": agg(name, "min_gap", valid),
            "contact_adjacent": agg(name, "contact_adjacent", valid),
            "in_shot_identity_correct": agg(
                name, "in_shot_fraction",
                lambda r: valid(r) and r["identity_correct"]),
        }

    out = {
        "regions": region_out,
        "selected_region": best_region,
        "selection_rule": "highest containment, then narrowest mean width",
        "confound": ("R2-R4 use the C0 contact proposal and are scored against "
                     "annotations made by someone who saw C0. See "
                     "phase3e_leakage_analysis.json — the evidence argues "
                     "against strong leakage but is not a blind study."),
        "l3_source": os.path.basename(args.bench),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    with open(args.best7_out, "w", encoding="utf-8") as f:
        json.dump({"region": best_region, "l3_source": os.path.basename(args.bench),
                   "objectives": {
                       "1_event_faithfulness": "in_shot_fraction, median_distance_to_shot",
                       "2_biomechanical_validity": "pose quality, min_gap",
                       "3_downstream_usefulness": "NOT MEASURED — needs retraining, out of scope"},
                   "methods": f_out, "per_clip": per_clip}, f, indent=1)

    print(f"{'region':<24}{'contain':>9}{'meanW':>8}{'medW':>7}{'W/GT':>7}{'IoU':>8}")
    for k, v in region_out.items():
        f_ = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
        print(f"{k:<24}{f_(v['containment'],2):>9}{v['mean_width']:>8.1f}"
              f"{v['median_width']:>7.1f}{v['width_ratio_vs_gt']:>7.2f}"
              f"{f_(v['mean_iou_with_gt']):>8}")
    print(f"\nselected region: {best_region}")
    print(f"\n{'method':<22}{'inShot':>8}{'distShot':>10}{'inShot(id)':>12}"
          f"{'poseQ':>8}{'minPoseQ':>10}{'minGap':>8}{'ctAdj':>7}")
    for k, v in f_out.items():
        f_ = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
        print(f"{k:<22}{f_(v['in_shot_fraction']):>8}"
              f"{f_(v['median_distance_to_shot'],1):>10}"
              f"{f_(v['in_shot_identity_correct']):>12}"
              f"{f_(v['mean_pose_quality']):>8}{f_(v['min_pose_quality']):>10}"
              f"{f_(v['min_gap'],1):>8}{f_(v['contact_adjacent'],1):>7}")
    print(f"\nwrote {args.out}\nwrote {args.best7_out}")


if __name__ == "__main__":
    main()
