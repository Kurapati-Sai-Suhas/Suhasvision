"""
phase3_shot_analysis.py — latency, window-length robustness, failure
mechanism classification and timeline diagrams for L0-L3.

The benchmark says WHICH method wins. This says whether the win is for a
reason worth trusting: is it stable when irrelevant frames are added, what
does it cost, and when it fails, WHY.
"""

import argparse
import json
import os
import statistics
import time

import numpy as np

import phase3_shot_localization as SL

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")


# ---------------------------------------------------------------------------
# Failure mechanism
# ---------------------------------------------------------------------------

def classify_failure(rec, clip, params, method):
    """One documented mechanism per non-successful valid clip.

    Ordered so that upstream causes are attributed upstream: an identity error
    is never reported as a localization error.
    """
    if not rec["identity_correct"]:
        return "wrong batsman — identity error upstream of localization"
    if rec["rejected"]:
        return "insufficient movement — peak prominence below rejection threshold"
    pred, gt = rec["pred"], rec["gt"]
    if clip["scene_cut_frames"] and SL.crosses(pred, clip["scene_cut_frames"]):
        return "scene cut — predicted interval spans an annotated cut"
    if pred[0] >= gt[1]:
        return "motion ambiguity — locked onto a later motion peak entirely after the shot"
    if pred[1] <= gt[0]:
        return "motion ambiguity — locked onto an earlier motion peak entirely before the shot"
    iou = SL.temporal_iou(tuple(pred), tuple(gt))
    if iou >= 0.3:
        return "partial overlap — boundary error only"
    if (pred[1] - pred[0]) < 0.5 * (gt[1] - gt[0]):
        return "proposal window — interval collapsed well inside the true shot"
    if (pred[1] - pred[0]) > 1.8 * (gt[1] - gt[0]):
        return "proposal window — interval over-extended beyond the true shot"
    return "boundary error — overlapping but poorly aligned"


# ---------------------------------------------------------------------------
# Window-length robustness
# ---------------------------------------------------------------------------

def robustness(clips, method, params, pads=(30, 45, None)):
    """Does the prediction move when irrelevant frames surround the action?

    The clip is cropped to +/- pad frames around the GT shot centre and the
    method is re-run on the cropped signal, then mapped back to absolute
    frames. `None` means the full clip.

    This uses ground truth to CHOOSE the crop, so it measures stability, not
    accuracy — a method that is right for the wrong reason can still be
    stable. Reported as such.
    """
    out = {}
    for c in clips:
        if c["coherence"] != "VALID_SINGLE_SHOT" or c["gt_start"] is None:
            continue
        centre = (c["gt_start"] + c["gt_end"]) // 2
        preds = {}
        for pad in pads:
            if pad is None:
                lo, hi = 0, len(c["g"])
            else:
                lo = max(0, centre - pad)
                hi = min(len(c["g"]), centre + pad)
            if hi - lo < 10:
                continue
            g = c["g"][lo:hi]
            l = c["l"][lo:hi] if c["l"] is not None else None
            p = SL.predict(method, hi - lo, g, l, **params)
            preds[str(pad)] = (None if p["rejected"]
                               else [p["start"] + lo, p["end"] + lo])
        if len(preds) >= 2:
            out[c["clip_id"]] = preds

    # Stability = IoU between the full-clip prediction and each cropped one.
    ious, flips = [], 0
    for cid, preds in out.items():
        full = preds.get("None")
        for k, v in preds.items():
            if k == "None":
                continue
            if full is None or v is None:
                if full != v:
                    flips += 1
                continue
            ious.append(SL.temporal_iou(tuple(full), tuple(v)))
    return {
        "n_clips": len(out),
        "median_self_iou_across_windows": (round(float(np.median(ious)), 4)
                                           if ious else None),
        "mean_self_iou_across_windows": (round(float(np.mean(ious)), 4)
                                         if ious else None),
        "reject_flips": flips,
        "note": ("Crop is centred on the GT shot, so this measures STABILITY "
                 "under added irrelevant context, not accuracy."),
        "per_clip": out,
    }


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def latency(clips, video_dir, params_by_method, n=12):
    """Cost per clip, split into the parts that actually dominate.

    Signal construction (one video decode) is shared by L1/L2/L3 and is by far
    the largest term; the windowing arithmetic is negligible. Reporting only a
    single number per method would hide that.
    """
    sample = [c for c in clips[:n]]
    decode = [c["decode_s"] for c in sample if c.get("decode_s")]
    out = {
        "n_clips_timed": len(sample),
        "signal_build_per_clip_ms": {
            "median": round(1000 * statistics.median(decode), 1) if decode else None,
            "mean": round(1000 * statistics.mean(decode), 1) if decode else None,
        },
        "signal_build_note": ("One decode per clip, shared by L1/L2/L3. Global "
                              "and batsman-local energy are computed in the "
                              "same pass, so L2 costs no extra decode."),
        "methods": {},
    }
    for m in SL.METHODS:
        p = params_by_method.get(m, {})
        ts = []
        for c in sample:
            t0 = time.perf_counter()
            SL.predict(m, c["n_frames"], c["g"], c["l"], **p)
            ts.append(time.perf_counter() - t0)
        per_frame = [t / max(c["n_frames"], 1) for t, c in zip(ts, sample)]
        out["methods"][m] = {
            "windowing_per_clip_ms": round(1000 * statistics.median(ts), 4),
            "windowing_per_frame_us": round(1e6 * statistics.median(per_frame), 3),
            "total_per_clip_ms": round(
                1000 * (statistics.median(ts)
                        + (statistics.median(decode) if decode and m != "L0_whole_clip" else 0.0)), 1),
        }
    return out


# ---------------------------------------------------------------------------
# Timeline diagram
# ---------------------------------------------------------------------------

def timeline_sheet(rows, out_path, width=1100, row_h=34):
    """ASCII-free timeline: GT bar above, prediction bar below, per clip."""
    import cv2
    if not rows:
        return None
    h = 60 + row_h * len(rows)
    img = np.full((h, width, 3), 22, np.uint8)
    x0, x1 = 250, width - 30

    def lab(t, org, col, s=0.42):
        # cv2's Hershey fonts are ASCII-only; an em-dash renders as "???".
        t = (t.replace("—", "-").replace("–", "-")
              .encode("ascii", "replace").decode("ascii"))
        cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, col, 1, cv2.LINE_AA)

    lab("clip", (12, 30), (235, 235, 235), 0.5)
    lab("timeline  (orange = ground truth, green = prediction)",
        (x0, 30), (235, 235, 235), 0.5)

    for i, r in enumerate(rows):
        y = 58 + i * row_h
        n = max(r["n_frames"], 1)
        sx = lambda f: int(x0 + (x1 - x0) * max(0, min(f, n)) / n)
        cv2.line(img, (x0, y + 12), (x1, y + 12), (70, 70, 70), 1)
        lab(f"{r['clip_id'][:26]}", (12, y + 10), (200, 200, 200), 0.38)
        lab(f"{r['tag']}", (12, y + 24), (140, 140, 190), 0.34)
        g = r["gt"]
        cv2.rectangle(img, (sx(g[0]), y + 2), (sx(g[1]), y + 10), (0, 170, 255), -1)
        if r["pred"]:
            p = r["pred"]
            col = (0, 220, 0) if r["iou"] >= 0.3 else (0, 90, 220)
            cv2.rectangle(img, (sx(p[0]), y + 14), (sx(p[1]), y + 22), col, -1)
            lab(f"IoU {r['iou']:.2f}", (x1 - 70, y + 22), (230, 230, 230), 0.36)
        else:
            lab("REJECTED", (x1 - 80, y + 22), (0, 90, 220), 0.38)
        for f in r.get("scene_cut_frames", []):
            cv2.line(img, (sx(f), y + 1), (sx(f), y + 24), (255, 80, 255), 1)
    cv2.imwrite(out_path, img)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--tracks", default=os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--features", default=os.path.join(P3, "phase3_semantic_features.json"))
    ap.add_argument("--bench", default=os.path.join(P3, "phase3_shot_localization.json"))
    ap.add_argument("--out", default=os.path.join(P3, "phase3_shot_analysis.json"))
    args = ap.parse_args()

    import phase3_shot_benchmark as SB
    with open(args.bench, encoding="utf-8") as f:
        bench = json.load(f)
    params_by_method = {m: v["params_fit_on_dev"]
                        for m, v in bench["methods"].items()}

    clips = SB.gather(args.videos, args.tracks, args.features)
    by_id = {c["clip_id"]: c for c in clips}
    print(f"gathered {len(clips)} clips")

    out = {"latency": latency(clips, args.videos, params_by_method),
           "robustness": {}, "failure_modes": {}}

    for m in SL.METHODS:
        if m != "L0_whole_clip":
            out["robustness"][m] = robustness(clips, m, params_by_method[m])

    # Failure mechanisms for the chosen method and the baseline.
    for m in SL.METHODS:
        det = bench["methods"][m]["all"]["detail"]
        cats = {}
        for r in det:
            c = by_id.get(r["clip_id"])
            if not c:
                continue
            iou = r["iou"]
            if iou >= 0.5:
                continue
            cause = classify_failure(r, c, params_by_method[m], m)
            cats.setdefault(cause, []).append(r["clip_id"])
        out["failure_modes"][m] = {"n_below_iou_0.5": sum(len(v) for v in cats.values()),
                                   "by_cause": {k: sorted(v) for k, v in
                                                sorted(cats.items(),
                                                       key=lambda kv: -len(kv[1]))}}

    # Timeline sheets: representative successes and each failure mode, for the
    # best method plus the baseline for contrast.
    sheets = {}
    for m in ("L3_hybrid", "L0_whole_clip"):
        det = {r["clip_id"]: r for r in bench["methods"][m]["all"]["detail"]}
        rows = []
        seen = set()
        ordered = sorted(det.values(), key=lambda r: -r["iou"])
        for r in ordered[:4]:
            c = by_id[r["clip_id"]]
            rows.append({"clip_id": r["clip_id"], "tag": f"success IoU {r['iou']:.2f}",
                         "gt": r["gt"], "pred": r["pred"], "iou": r["iou"],
                         "n_frames": c["n_frames"],
                         "scene_cut_frames": c["scene_cut_frames"]})
            seen.add(r["clip_id"])
        for cause, ids in out["failure_modes"][m]["by_cause"].items():
            for cid in ids:
                if cid in seen or cid not in det:
                    continue
                r, c = det[cid], by_id[cid]
                rows.append({"clip_id": cid, "tag": cause[:44],
                             "gt": r["gt"], "pred": r["pred"], "iou": r["iou"],
                             "n_frames": c["n_frames"],
                             "scene_cut_frames": c["scene_cut_frames"]})
                seen.add(cid)
                break
        p = os.path.join(P3, f"shot_timeline_{m}.png")
        sheets[m] = timeline_sheet(rows, p)
    out["timeline_sheets"] = sheets

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print("\nLATENCY")
    sb = out["latency"]["signal_build_per_clip_ms"]
    print(f"  signal build (shared, one decode): {sb['median']} ms/clip median")
    for m, v in out["latency"]["methods"].items():
        print(f"  {m:<20} windowing {v['windowing_per_clip_ms']:>8.4f} ms/clip"
              f"   {v['windowing_per_frame_us']:>8.3f} us/frame"
              f"   total {v['total_per_clip_ms']:>7.1f} ms")

    print("\nWINDOW-LENGTH ROBUSTNESS (self-IoU across short/normal/full)")
    for m, v in out["robustness"].items():
        print(f"  {m:<20} median {v['median_self_iou_across_windows']}  "
              f"mean {v['mean_self_iou_across_windows']}  "
              f"reject flips {v['reject_flips']}  (n={v['n_clips']})")

    print("\nFAILURE MECHANISMS (clips below IoU 0.5)")
    for m, v in out["failure_modes"].items():
        print(f"  {m}  (n={v['n_below_iou_0.5']})")
        for cause, ids in v["by_cause"].items():
            print(f"      {len(ids):>2}  {cause}")
    print(f"\nwrote {args.out}")
    for m, p in sheets.items():
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
