"""
phase3_batsman_ablation.py — S0-S4 semantic batsman identification.

PROTOCOL
Every system consumes identical YOLO detections and identical ByteTrack
tracks (from the Phase-2 cache) and identical sampled frames (from the
Phase-3 feature cache). ONLY the scoring feature set differs, so any
difference in outcome is attributable to the added evidence and nothing else.

  S0  geometry only            - the frozen Phase-2 B3 model, unmodified
  S1  + persistence            - how continuously the track exists
  S2  + skeleton dynamics      - interpretable joint structure
  S3  + equipment evidence     - bat detection AND bat-person association
  S4  + batting-action evidence- rise-then-fall stroke structure

S1-S4 are fitted on the DEV split only, with the same logistic routine and
the same decision thresholds as S0. EVAL is touched once, to report.

WHAT COUNTS AS WRONG
Committing to a track that is not the annotated batsman track — including
when no track matches the batsman at all. Only an explicit refusal is not an
error. (Phase 2 shipped a bug where "detector never found the batsman" was
bucketed as excluded rather than wrong, which rewards a system for finding
nobody; that accounting is not repeated here.)
"""

import argparse
import json
import math
import os
import statistics

import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

GEOM = ("coverage", "foot_stability", "scale_stability",
        "centroid_stability", "vertical_position")
PERSIST = ("p_span", "p_contiguity", "p_longest_run", "p_conf_stability", "p_conf_mean")
SKEL = ("k_pose_rate", "k_wrist_excursion", "k_wrist_lift_range",
        "k_shoulder_rotation", "k_lower_stability", "k_bilateral", "k_visibility")
EQUIP = ("e_bat_anywhere", "e_bat_on_person", "e_bat_near_hand",
         "e_bat_persist", "e_bat_conf")
ACTION = ("a_backlift", "a_downswing", "a_updown_shape", "a_upper_lower_ratio")

SYSTEMS = {
    "S0_geometry":      GEOM,
    "S1_persistence":   GEOM + PERSIST,
    "S2_skeleton":      GEOM + PERSIST + SKEL,
    "S3_equipment":     GEOM + PERSIST + SKEL + EQUIP,
    "S4_action":        GEOM + PERSIST + SKEL + EQUIP + ACTION,
}

# Same decision rule as Phase 2, held constant across every arm so the
# comparison measures evidence rather than threshold tuning.
CONFIDENT_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.15


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_logistic(X, y, l2=1.0, iters=4000, lr=0.5):
    """Same routine Phase 2 used, so S0's frozen weights and the refitted
    arms come from an identical procedure."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, d = X.shape
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    w, b = np.zeros(d), 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xs @ w + b, -30, 30)))
        w -= lr * (Xs.T @ (p - y) / n + (l2 / n) * w)
        b -= lr * float((p - y).mean())
    return w / sd, b - float((w * mu / sd).sum())


def decide(scores):
    """scores: {track_id: p}. Returns (verdict, track_id, conf, margin)."""
    if not scores:
        return "NO_BATSMAN", None, 0.0, 0.0
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    tid, best = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best - runner
    if best >= CONFIDENT_THRESHOLD and (len(ranked) == 1 or margin >= MARGIN_THRESHOLD):
        return "CONFIDENT_BATSMAN", tid, best, margin
    if best >= 0.40:
        return "AMBIGUOUS", None, best, margin
    return "NO_BATSMAN", None, best, margin


def load_everything(tracks_path, feats_path):
    from phase2_cache_tracks import load_tracks
    from phase2_detector_benchmark import load_gt
    from phase2_fit_batsman_model import gt_track_for_clip

    tracks = load_tracks(tracks_path)
    with open(feats_path, encoding="utf-8") as f:
        feats = json.load(f)
    gt = {c["clip_id"]: c for c in load_gt()["clips"]}

    rows = []
    for cid, fc in feats["clips"].items():
        clip_gt = gt.get(cid)
        tc = tracks["clips"].get(cid)
        if not clip_gt or not tc:
            continue
        gt_frames = [f for f in clip_gt["frames"] if f["batsman_visible"] and f["batsman_bbox"]]
        gt_tid, n_match = gt_track_for_clip(tc["tracks"], gt_frames)
        rows.append({
            "clip_id": cid, "split": fc["split"], "gt_track": gt_tid,
            "gt_match_frames": n_match, "n_gt_frames": len(gt_frames),
            "tracks": {int(t): v["features"] for t, v in fc["tracks"].items()},
            # Corrected coverage/pose decomposition, carried so failure
            # categorisation can tell tracking from pose. Never fed to any
            # system's feature list.
            "diagnostics": {int(t): v.get("diagnostic")
                            for t, v in fc["tracks"].items() if v.get("diagnostic")},
            "n_tracks": len(fc["tracks"]),
        })
    return rows


def candidate_recall(rows, tracks_path):
    """Fraction of annotated batsman frames covered by SOME track. Identical
    for every system by construction — the ablation shares detections and
    tracks — and reported to make that explicit rather than implied."""
    from phase2_cache_tracks import load_tracks
    from phase2_detector_benchmark import load_gt
    from phase2_detectors import Detection, iou

    tracks = load_tracks(tracks_path)
    gt = {c["clip_id"]: c for c in load_gt()["clips"]}
    hit = tot = 0
    for cid, tc in tracks["clips"].items():
        clip = gt.get(cid)
        if not clip:
            continue
        for fr in clip["frames"]:
            if not (fr["batsman_visible"] and fr["batsman_bbox"]):
                continue
            tot += 1
            best = 0.0
            for obs in tc["tracks"].values():
                for f, b, _ in obs:
                    if f == fr["frame"]:
                        best = max(best, iou(Detection(*b, conf=1.0), fr["batsman_bbox"]))
                        break
            hit += int(best >= 0.5)
    return round(hit / tot, 4) if tot else None, tot


def evaluate(rows, feature_names, model, split):
    sub = [r for r in rows if r["split"] == split]
    tally = {"correct": 0, "wrong": 0, "refused": 0}
    verdicts = {"CONFIDENT_BATSMAN": 0, "AMBIGUOUS": 0, "NO_BATSMAN": 0}
    detail, coverage_pts = [], []

    for r in sub:
        scores = {}
        for tid, f in r["tracks"].items():
            z = model["intercept"] + sum(model["coef"][k] * f.get(k, 0.0) for k in feature_names)
            scores[tid] = sigmoid(z)
        verdict, pick, conf, margin = decide(scores)
        verdicts[verdict] += 1

        if pick is None:
            outcome = "refused"
        elif r["gt_track"] is not None and pick == r["gt_track"]:
            outcome = "correct"
        else:
            outcome = "wrong"
        tally[outcome] += 1

        coverage_pts.append((conf, outcome))
        detail.append({
            "clip_id": r["clip_id"], "split": split, "outcome": outcome,
            "verdict": verdict, "picked_track": pick, "gt_track": r["gt_track"],
            "confidence": round(conf, 4), "margin": round(margin, 4),
            "n_tracks": r["n_tracks"],
            "scores": {str(t): round(s, 4) for t, s in sorted(scores.items())},
        })

    n = len(sub)
    committed = tally["correct"] + tally["wrong"]
    return {
        "split": split, "clips": n, **tally, "verdicts": verdicts,
        "batsman_precision": round(tally["correct"] / committed, 4) if committed else None,
        "wrong_person_rate": round(tally["wrong"] / committed, 4) if committed else None,
        "refusal_rate": round(tally["refused"] / n, 4) if n else None,
        "false_positive_rate": round(tally["wrong"] / n, 4) if n else None,
        "accuracy_all_clips": round(tally["correct"] / n, 4) if n else None,
        "_coverage": coverage_pts, "detail": detail,
    }


def risk_coverage(points):
    """Wrong-person rate as a function of how many clips we choose to commit
    on, by sweeping the confidence threshold. This is what makes the
    refusal/accuracy tradeoff comparable across systems rather than a single
    operating point."""
    pts = sorted(points, key=lambda p: -p[0])
    out, correct, wrong = [], 0, 0
    for i, (conf, outcome) in enumerate(pts, 1):
        if outcome == "correct":
            correct += 1
        elif outcome == "wrong":
            wrong += 1
        dec = correct + wrong
        if dec:
            out.append({"threshold": round(conf, 4),
                        "coverage": round(i / len(pts), 4),
                        "wrong_rate_among_committed": round(wrong / dec, 4)})
    return out[::max(1, len(out) // 12)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--features", default=os.path.join(P3, "phase3_semantic_features.json"))
    ap.add_argument("--s0-model", default=os.path.join(P2, "phase2_batsman_model_geom.json"))
    args = ap.parse_args()

    rows = load_everything(args.tracks, args.features)
    dev = [r for r in rows if r["split"] == "dev"]
    rec, n_gt_frames = candidate_recall(rows, args.tracks)
    print(f"clips {len(rows)} (dev {len(dev)}, eval {len(rows)-len(dev)}) | "
          f"tracks {sum(r['n_tracks'] for r in rows)}")
    print(f"candidate recall (shared by all systems): {rec} over {n_gt_frames} GT frames\n")

    with open(args.s0_model, encoding="utf-8") as f:
        frozen = json.load(f)["model"]

    results, models = {}, {}
    for name, feats in SYSTEMS.items():
        if name == "S0_geometry":
            model = frozen                      # frozen Phase-2 B3, unmodified
        else:
            X = [[r["tracks"][t].get(k, 0.0) for k in feats]
                 for r in dev for t in r["tracks"] if r["gt_track"] is not None]
            y = [1 if t == r["gt_track"] else 0
                 for r in dev for t in r["tracks"] if r["gt_track"] is not None]
            coef, inter = fit_logistic(X, y)
            model = {"coef": {k: float(v) for k, v in zip(feats, coef)},
                     "intercept": float(inter)}
        models[name] = model
        results[name] = {sp: evaluate(rows, feats, model, sp) for sp in ("dev", "eval")}

    print(f"{'system':<16}{'split':<6}{'corr':>5}{'wrong':>6}{'ref':>5}"
          f"{'prec':>7}{'wrong%':>8}{'refuse%':>9}{'acc/all':>9}")
    for name in SYSTEMS:
        for sp in ("dev", "eval"):
            r = results[name][sp]
            print(f"{name:<16}{sp:<6}{r['correct']:>5}{r['wrong']:>6}{r['refused']:>5}"
                  f"{str(r['batsman_precision']):>7}{str(r['wrong_person_rate']):>8}"
                  f"{str(r['refusal_rate']):>9}{str(r['accuracy_all_clips']):>9}")

    # Paired per-clip comparison against S0 on eval.
    print(f"\n{'='*70}\nPAIRED vs S0 (eval)\n{'='*70}")
    s0 = {d["clip_id"]: d["outcome"] for d in results["S0_geometry"]["eval"]["detail"]}
    paired = {}
    for name in list(SYSTEMS)[1:]:
        cur = {d["clip_id"]: d["outcome"] for d in results[name]["eval"]["detail"]}
        imp = [c for c in cur if s0[c] != "correct" and cur[c] == "correct"]
        reg = [c for c in cur if s0[c] == "correct" and cur[c] != "correct"]
        fixed_wrong = [c for c in cur if s0[c] == "wrong" and cur[c] != "wrong"]
        new_wrong = [c for c in cur if s0[c] != "wrong" and cur[c] == "wrong"]
        paired[name] = {"improved": imp, "regressed": reg,
                        "wrong_fixed": fixed_wrong, "wrong_introduced": new_wrong,
                        "net": len(imp) - len(reg)}
        print(f"  {name:<16} improved {len(imp):>2}  regressed {len(reg):>2}  "
              f"net {len(imp)-len(reg):+d}   wrong fixed {len(fixed_wrong)}, "
              f"introduced {len(new_wrong)}")

    payload = {
        "protocol": {
            "shared_detections": "yolo11m@640",
            "shared_tracker": "bytetrack",
            "shared_sampled_frames": True,
            "fitting": "S1-S4 logistic fitted on DEV only; S0 is the frozen Phase-2 B3 model",
            "decision_rule": {"confident_threshold": CONFIDENT_THRESHOLD,
                              "margin_threshold": MARGIN_THRESHOLD,
                              "held_constant_across_systems": True},
            "candidate_recall_shared": rec,
            "n_gt_frames": n_gt_frames,
        },
        "feature_sets": {k: list(v) for k, v in SYSTEMS.items()},
        "models": models,
        "results": {n: {sp: {k: v for k, v in r.items() if k != "_coverage"}
                        for sp, r in s.items()} for n, s in results.items()},
        "risk_coverage": {n: risk_coverage(results[n]["eval"]["_coverage"])
                          for n in SYSTEMS},
        "paired_vs_S0_eval": paired,
    }
    out = os.path.join(P3, "phase3_batsman_ablation_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)

    jl = os.path.join(P3, "phase3_batsman_ablation_results.jsonl")
    with open(jl, "w", encoding="utf-8") as f:
        for name in SYSTEMS:
            for sp in ("dev", "eval"):
                for d in results[name][sp]["detail"]:
                    f.write(json.dumps({"system": name, **d}) + "\n")

    print(f"\nwrote {out}")
    print(f"wrote {jl}")


if __name__ == "__main__":
    main()
