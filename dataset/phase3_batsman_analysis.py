"""
phase3_batsman_analysis.py — the analyses the headline table cannot answer.

  1. Bat-signal ablation: WHICH level of bat specificity helps, if any.
     "Can we detect a bat" is not the question; "does bat evidence improve
     batsman identification" is.
  2. Stationary-feeder case study: per-system decisions on the clips whose
     non-batsman candidate is itself stationary — the failure mode Phase 2
     identified as structurally unreachable by geometry.
  3. Failure categorisation for every wrong/refused eval clip.
  4. Fitted coefficients, so the scorer stays inspectable.
"""

import json
import os
import statistics

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

import phase3_batsman_ablation as AB
import phase3_failure_taxonomy as FT


def bat_level_ablation(rows, dev):
    """S2 plus exactly one bat feature at a time, then all of them."""
    base = AB.GEOM + AB.PERSIST + AB.SKEL
    variants = {
        "S2_baseline_no_bat": base,
        "+bat_anywhere": base + ("e_bat_anywhere",),
        "+bat_on_person": base + ("e_bat_on_person",),
        "+bat_near_hand": base + ("e_bat_near_hand",),
        "+bat_persist": base + ("e_bat_persist",),
        "+all_bat (=S3)": base + AB.EQUIP,
    }
    out = {}
    for name, feats in variants.items():
        X = [[r["tracks"][t].get(k, 0.0) for k in feats]
             for r in dev for t in r["tracks"] if r["gt_track"] is not None]
        y = [1 if t == r["gt_track"] else 0
             for r in dev for t in r["tracks"] if r["gt_track"] is not None]
        coef, inter = AB.fit_logistic(X, y)
        model = {"coef": dict(zip(feats, map(float, coef))), "intercept": float(inter)}
        ev = AB.evaluate(rows, feats, model, "eval")
        out[name] = {k: ev[k] for k in
                     ("correct", "wrong", "refused", "batsman_precision",
                      "wrong_person_rate", "refusal_rate", "accuracy_all_clips")}
    return out


def bat_discrimination(rows):
    """Does bat evidence separate the batsman track from the others at all?
    Measured directly on the features, independent of any fitted model."""
    out = {}
    for feat in AB.EQUIP:
        pos, neg = [], []
        for r in rows:
            if r["gt_track"] is None:
                continue
            for tid, f in r["tracks"].items():
                (pos if tid == r["gt_track"] else neg).append(f.get(feat, 0.0))
        if pos and neg:
            out[feat] = {
                "batsman_mean": round(statistics.mean(pos), 4),
                "other_mean": round(statistics.mean(neg), 4),
                "separation": round(statistics.mean(pos) - statistics.mean(neg), 4),
                "n_batsman": len(pos), "n_other": len(neg),
            }
    return out


def find_feeder_clips(rows):
    """Clips whose strongest NON-batsman candidate is itself stationary and
    at least as large as the batsman — the geometry-defeating configuration.
    """
    hits = []
    for r in rows:
        if r["gt_track"] is None:
            continue
        gt = r["tracks"].get(r["gt_track"])
        if not gt:
            continue
        for tid, f in r["tracks"].items():
            if tid == r["gt_track"]:
                continue
            stationary = f.get("foot_stability", 0) >= 0.85
            bigger = f.get("relative_height", 0) >= gt.get("relative_height", 0)
            persistent = f.get("coverage", 0) >= 0.4
            if stationary and bigger and persistent:
                hits.append({
                    "clip_id": r["clip_id"], "split": r["split"],
                    "gt_track": r["gt_track"], "rival_track": tid,
                    "rival_foot_stability": round(f.get("foot_stability", 0), 4),
                    "rival_rel_height": round(f.get("relative_height", 0), 4),
                    "gt_rel_height": round(gt.get("relative_height", 0), 4),
                    "rival_bat_on_person": round(f.get("e_bat_on_person", 0), 4),
                    "gt_bat_on_person": round(gt.get("e_bat_on_person", 0), 4),
                    "rival_bat_near_hand": round(f.get("e_bat_near_hand", 0), 4),
                    "gt_bat_near_hand": round(gt.get("e_bat_near_hand", 0), 4),
                    "rival_updown": round(f.get("a_updown_shape", 0), 4),
                    "gt_updown": round(gt.get("a_updown_shape", 0), 4),
                })
                break
    return hits


def categorise_failure(r, d, feats, diags=None):
    """One documented cause per non-correct eval clip.

    THE ORDERING BUG THIS FIXES
    The refusal branch used to test

        if k_pose_rate < 0.4:  -> "pose failure on the batsman crop"
        if coverage    < 0.4:  -> "tracking failure - track too fragmented"

    Because a pose requires a box, k_pose_rate <= coverage ALWAYS. So
    coverage < 0.4 guaranteed k_pose_rate < 0.4, the pose branch returned
    first, and the tracking branch was unreachable for exactly the clips it
    was written to catch. Every fragmented-track failure was reported as a
    pose failure -- which is what sent a whole investigation after the pose
    model when the batsman track was the real problem.

    The fix is to test the two INDEPENDENT factors in causal order (coverage
    caps pose, so coverage first) via phase3_failure_taxonomy, which also
    names the joint case instead of forcing it into one bucket.

    `diags` carries the per-track decomposition produced during extraction.
    It is required for the corrected verdict, because `k_pose_rate` alone
    cannot distinguish the two causes -- that is the whole point.
    """
    if r["gt_track"] is None:
        return "detector/tracking failure — no track matches the annotated batsman"
    gt = feats.get(r["gt_track"], {})
    if d["outcome"] == "wrong":
        picked = feats.get(d["picked_track"], {})
        if picked.get("foot_stability", 0) >= 0.85 and \
           picked.get("relative_height", 0) >= gt.get("relative_height", 0):
            return "stationary feeder — rival is stationary and larger"
        if picked.get("e_bat_on_person", 0) > gt.get("e_bat_on_person", 0):
            return "bat association failure — bat attributed to the wrong person"
        return "semantic ambiguity — rival scored higher on mixed evidence"
    # refused
    if r["n_tracks"] == 1:
        return "single candidate below confidence threshold"

    gd = (diags or {}).get(r["gt_track"])
    if gd:
        verdict = gd["failure_type"]
        cov = gd["coverage"]
        pgb = gd["pose_success_given_box"]
        if verdict == FT.TRACKING_FAILURE:
            return (f"tracking failure — batsman track covers only "
                    f"{cov:.0%} of sampled frames "
                    f"(pose succeeds on {pgb:.0%} of the frames it does get)"
                    if pgb is not None else
                    f"tracking failure — batsman track covers only {cov:.0%} "
                    f"of sampled frames (pose never invoked)")
        if verdict == FT.POSE_FAILURE:
            return (f"pose failure on the batsman crop — track covers "
                    f"{cov:.0%} of frames but pose succeeds on only "
                    f"{pgb:.0%} of them")
        if verdict == FT.JOINT_FAILURE:
            return (f"joint tracking+pose failure — coverage {cov:.0%}, "
                    f"pose {pgb:.0%} of boxed frames")
    else:
        # No decomposition available (features predate the fix). Say so
        # rather than silently falling back to the conflated test.
        if gt.get("k_pose_rate", 0) < 0.4:
            return ("low effective pose rate, cause UNRESOLVED — "
                    "re-extract features to separate tracking from pose")

    return "margin too small — candidates not separable"


def main():
    rows = AB.load_everything(
        os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"),
        os.path.join(P3, "phase3_semantic_features.json"))
    dev = [r for r in rows if r["split"] == "dev"]
    by_clip = {r["clip_id"]: r for r in rows}

    with open(os.path.join(P3, "phase3_batsman_ablation_results.json"), encoding="utf-8") as f:
        abl = json.load(f)

    print("=" * 74)
    print("1. BAT-SIGNAL ABLATION (eval) — which level of specificity helps?")
    print("=" * 74)
    lvl = bat_level_ablation(rows, dev)
    print(f"  {'variant':<22}{'corr':>5}{'wrong':>6}{'ref':>5}{'prec':>8}{'refuse%':>9}")
    for k, v in lvl.items():
        print(f"  {k:<22}{v['correct']:>5}{v['wrong']:>6}{v['refused']:>5}"
              f"{str(v['batsman_precision']):>8}{str(v['refusal_rate']):>9}")

    print(f"\n{'='*74}\n2. DOES BAT EVIDENCE SEPARATE BATSMAN FROM OTHERS? (all tracks)\n{'='*74}")
    disc = bat_discrimination(rows)
    print(f"  {'feature':<20}{'batsman':>10}{'other':>10}{'separation':>12}")
    for k, v in disc.items():
        print(f"  {k:<20}{v['batsman_mean']:>10}{v['other_mean']:>10}{v['separation']:>12}")

    print(f"\n{'='*74}\n3. STATIONARY-FEEDER CASE STUDY\n{'='*74}")
    feeders = find_feeder_clips(rows)
    print(f"  clips with a stationary, larger, persistent rival: {len(feeders)}")
    detail_by = {}
    for name in AB.SYSTEMS:
        for d in abl["results"][name]["eval"]["detail"] + abl["results"][name]["dev"]["detail"]:
            detail_by.setdefault(d["clip_id"], {})[name] = d
    for h in feeders:
        cid = h["clip_id"]
        decisions = detail_by.get(cid, {})
        line = "  ".join(f"{n.split('_')[0]}:{decisions[n]['outcome'][:4]}"
                         for n in AB.SYSTEMS if n in decisions)
        print(f"\n  {cid}  [{h['split']}]  gt=t{h['gt_track']} rival=t{h['rival_track']}")
        print(f"    rival foot_stab {h['rival_foot_stability']}  "
              f"rel_h rival {h['rival_rel_height']} vs gt {h['gt_rel_height']}")
        print(f"    bat_on_person   rival {h['rival_bat_on_person']} vs gt {h['gt_bat_on_person']}")
        print(f"    bat_near_hand   rival {h['rival_bat_near_hand']} vs gt {h['gt_bat_near_hand']}")
        print(f"    updown_shape    rival {h['rival_updown']} vs gt {h['gt_updown']}")
        print(f"    decisions: {line}")

    print(f"\n{'='*74}\n4. FAILURE ANALYSIS (eval, per system)\n{'='*74}")
    failures = {}
    for name in AB.SYSTEMS:
        cats = {}
        for d in abl["results"][name]["eval"]["detail"]:
            if d["outcome"] == "correct":
                continue
            r = by_clip[d["clip_id"]]
            c = categorise_failure(r, d, r["tracks"], r.get("diagnostics"))
            cats.setdefault(c, []).append(d["clip_id"])
        failures[name] = cats
        print(f"\n  {name}:")
        for c, clips in sorted(cats.items(), key=lambda kv: -len(kv[1])):
            print(f"    {len(clips):>2}  {c}")
            for cl in clips[:3]:
                print(f"          {cl}")

    print(f"\n{'='*74}\n5. FITTED COEFFICIENTS (top magnitude)\n{'='*74}")
    for name in ("S2_skeleton", "S3_equipment", "S4_action"):
        m = abl["models"][name]["coef"]
        top = sorted(m.items(), key=lambda kv: -abs(kv[1]))[:6]
        print(f"\n  {name}:")
        for k, v in top:
            print(f"    {k:<24}{v:+8.3f}")

    out = os.path.join(P3, "phase3_batsman_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"bat_level_ablation": lvl, "bat_discrimination": disc,
                   "stationary_feeder_clips": feeders,
                   "failure_categories": failures}, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
