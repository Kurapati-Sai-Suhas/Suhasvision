"""
phase3_contact_benchmark.py — C0-C6 contact localization, and whether contact
can correct L3's shot interval.

GROUND-TRUTH DISCIPLINE
Only EXACT and CONTACT_ADJACENT annotations carry a contact frame; AMBIGUOUS
and NOT_VISIBLE carry none and are never treated as frame labels. The two
usable types have DIFFERENT precision by construction — the schema's own
tolerances are +/-1 frame for EXACT and +/-3 for CONTACT_ADJACENT — so a
"within +/-1 frame" figure computed over CONTACT_ADJACENT clips is partly
measuring annotation noise, not model error. Results are broken out by type
for that reason.

IDENTITY IS SEPARATED
Contact is measured on the S2-selected batsman. A clip where S2 picked the
wrong person is an identity failure, not a contact failure, and is reported
separately rather than pooled.

THE CORRECTOR EXPERIMENT
L3 is a soft prior, not a crop. Three corrector forms are tried, all fitted on
DEV only and evaluated unchanged on eval:

  recenter  place a window of fitted width around the contact estimate
  expand    union of L3's interval with a contact-centred window
  repair    keep L3 unless contact falls outside it, then recenter

`repair` exists because the failure analysis showed L3's dominant error is
locking onto the WRONG motion peak; a corrector that only nudges boundaries
cannot fix that, but one that relocates the interval when an independent
signal disagrees might.
"""

import argparse
import json
import os

import numpy as np

import phase3_contact_signals as CS
import phase3_shot_localization as SL

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

USABLE_TYPES = ("EXACT", "CONTACT_ADJACENT")
CORRECTORS = ("recenter", "expand", "repair")


def load_context(cache_path, bench_path):
    from phase3_annotations import build_records
    import phase3_event_cache as EC

    cache = EC.load(cache_path)
    recs = {r["video_id"]: r for r in build_records()}
    with open(bench_path, encoding="utf-8") as f:
        bench = json.load(f)
    l3 = bench["methods"]["L3_hybrid"]
    l3_params = l3["params_fit_on_dev"]
    return cache, recs, l3_params


def c0_proposal(cid):
    """The existing contact proposal, read from the committed artifact rather
    than recomputed, so C0 is measured exactly as it exists."""
    p = os.path.join(P3, "phase3_event_proposals.json")
    if not hasattr(c0_proposal, "_cache"):
        with open(p, encoding="utf-8") as f:
            c0_proposal._cache = {c["clip_id"]: c for c in json.load(f)["clips"]}
    return c0_proposal._cache.get(cid, {}).get("proposed_contact")


def contact_estimates(cache):
    """Every signal's contact estimate for every clip. No ground truth used."""
    out = {}
    for cid, c in cache["clips"].items():
        frames = c["frames"]
        est = {"C0_current_proposal": {"frame": c0_proposal(cid),
                                       "confidence": None, "reason": ""}}
        for name in CS.SIGNALS:
            if name == "C0_current_proposal":
                continue
            est[name] = CS.estimate(name, frames)
        out[cid] = est
    return out


def evaluate_contact(cache, recs, est, subset=None):
    """Frame-error metrics over clips with a usable contact label."""
    rows = []
    for cid, c in cache["clips"].items():
        r = recs.get(cid)
        if not r or r["contact_type"] not in USABLE_TYPES:
            continue
        if r["contact_frame"] is None:
            continue
        if subset and not subset(c, r):
            continue
        rows.append((cid, c, r))

    out = {}
    for name in CS.SIGNALS:
        errs, missing = [], 0
        detail = []
        for cid, c, r in rows:
            f = est[cid][name]["frame"]
            if f is None:
                missing += 1
                detail.append({"clip_id": cid, "gt": r["contact_frame"],
                               "pred": None, "err": None,
                               "contact_type": r["contact_type"],
                               "identity_correct": c["identity_correct"]})
                continue
            e = abs(f - r["contact_frame"])
            errs.append(e)
            detail.append({"clip_id": cid, "gt": r["contact_frame"], "pred": f,
                           "err": e, "contact_type": r["contact_type"],
                           "identity_correct": c["identity_correct"],
                           "split": c["split"]})
        n = len(rows)
        within = lambda k: (round(sum(1 for e in errs if e <= k) / n, 4)
                            if n else None)
        out[name] = {
            "n_clips": n, "n_estimated": len(errs), "n_not_found": missing,
            "not_found_rate": round(missing / n, 4) if n else None,
            "median_abs_err": round(float(np.median(errs)), 2) if errs else None,
            "mean_abs_err": round(float(np.mean(errs)), 2) if errs else None,
            "within_1": within(1), "within_2": within(2), "within_3": within(3),
            "detail": detail,
        }
    return out


# ---------------------------------------------------------------------------
# Contact as an L3 corrector
# ---------------------------------------------------------------------------

def l3_interval(c, params):
    """L3's soft prior for one clip, from the cached motion signals."""
    frames = c["frames"]
    g = CS.global_motion(frames)
    l = CS.local_motion(frames)
    if g is None:
        return None
    p = SL.predict("L3_hybrid", c["n_frames"], g, l, **params)
    return p


def apply_corrector(kind, l3p, contact, n_frames, pre, post, margin):
    """Combine L3's prior with a contact anchor. Returns (start, end)."""
    have_l3 = l3p is not None and not l3p["rejected"]
    if contact is None:
        return (l3p["start"], l3p["end"]) if have_l3 else None
    lo = max(0, contact - pre)
    hi = min(n_frames - 1, contact + post)
    if kind == "recenter" or not have_l3:
        return (lo, hi)
    a, b = l3p["start"], l3p["end"]
    if kind == "expand":
        return (max(0, min(a, contact - margin)),
                min(n_frames - 1, max(b, contact + margin)))
    if kind == "repair":
        # Only relocate when the independent anchor lands outside L3's
        # interval, i.e. when the two disagree about WHICH event this is.
        if a <= contact <= b:
            return (a, b)
        return (lo, hi)
    return (a, b)


def evaluate_shot(cache, recs, est, params, kind, signal, pre, post, margin,
                  split=None):
    ious, se, ee, detail = [], [], [], []
    n_rej = 0
    for cid, c in cache["clips"].items():
        r = recs.get(cid)
        if not r or r["coherence"] != "VALID_SINGLE_SHOT":
            continue
        if r["shot_start_frame"] is None:
            continue
        if split and c["split"] != split:
            continue
        gt = (r["shot_start_frame"], r["shot_end_frame"])
        l3p = l3_interval(c, params)
        if kind == "L0":
            pred = (0, max(1, c["n_frames"] - 1))
        elif kind == "L3":
            pred = ((l3p["start"], l3p["end"])
                    if l3p and not l3p["rejected"] else None)
        else:
            contact = est[cid][signal]["frame"]
            pred = apply_corrector(kind, l3p, contact, c["n_frames"],
                                   pre, post, margin)
        if pred is None:
            n_rej += 1
            ious.append(0.0)
            detail.append({"clip_id": cid, "gt": list(gt), "pred": None,
                           "iou": 0.0, "split": c["split"],
                           "identity_correct": c["identity_correct"]})
            continue
        iou = SL.temporal_iou(pred, gt)
        ious.append(iou)
        se.append(abs(pred[0] - gt[0]))
        ee.append(abs(pred[1] - gt[1]))
        detail.append({"clip_id": cid, "gt": list(gt), "pred": list(pred),
                       "iou": round(iou, 4), "split": c["split"],
                       "identity_correct": c["identity_correct"]})
    rate = lambda t: (round(sum(1 for i in ious if i >= t) / len(ious), 4)
                      if ious else None)
    return {
        "n_valid": len(ious), "rejected": n_rej,
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "median_iou": round(float(np.median(ious)), 4) if ious else None,
        "recall@0.3": rate(0.3), "recall@0.5": rate(0.5), "recall@0.7": rate(0.7),
        "median_start_err": round(float(np.median(se)), 2) if se else None,
        "median_end_err": round(float(np.median(ee)), 2) if ee else None,
        "detail": detail,
    }


def fit_corrector_on_dev(cache, recs, est, params, signal):
    """Grid-search corrector shape on DEV only."""
    best = None
    for kind in CORRECTORS:
        for pre in (6, 10, 14, 18):
            for post in (6, 10, 14, 18):
                for margin in (4, 8, 12):
                    r = evaluate_shot(cache, recs, est, params, kind, signal,
                                      pre, post, margin, split="dev")
                    if r["mean_iou"] is None:
                        continue
                    if best is None or r["mean_iou"] > best["dev_mean_iou"]:
                        best = {"kind": kind, "pre": pre, "post": post,
                                "margin": margin,
                                "dev_mean_iou": r["mean_iou"]}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(P3, "phase3_event_cache.json"))
    ap.add_argument("--bench", default=os.path.join(P3, "phase3_shot_localization.json"))
    ap.add_argument("--out", default=os.path.join(P3, "phase3_contact_results.json"))
    ap.add_argument("--jsonl", default=os.path.join(P3, "phase3_contact_results.jsonl"))
    args = ap.parse_args()

    cache, recs, params = load_context(args.cache, args.bench)
    est = contact_estimates(cache)
    print(f"clips in cache: {len(cache['clips'])}")

    out = {"l3_params": params,
           "usable_contact_types": list(USABLE_TYPES),
           "tolerance_note": ("Schema tolerance is +/-1 frame for EXACT and "
                              "+/-3 for CONTACT_ADJACENT. within_1 over "
                              "CONTACT_ADJACENT clips partly measures "
                              "annotation noise."),
           "contact": {}}

    out["contact"]["all"] = evaluate_contact(cache, recs, est)
    out["contact"]["identity_correct"] = evaluate_contact(
        cache, recs, est, subset=lambda c, r: c["identity_correct"])
    out["contact"]["identity_wrong"] = evaluate_contact(
        cache, recs, est, subset=lambda c, r: not c["identity_correct"])
    for t in USABLE_TYPES:
        out["contact"][f"type_{t}"] = evaluate_contact(
            cache, recs, est, subset=lambda c, r, t=t: r["contact_type"] == t)
    for sp in ("dev", "eval"):
        out["contact"][f"split_{sp}"] = evaluate_contact(
            cache, recs, est, subset=lambda c, r, sp=sp: c["split"] == sp)

    # Best contact signal, chosen on DEV by median error.
    dev = out["contact"]["split_dev"]
    ranked = sorted((v["median_abs_err"], k) for k, v in dev.items()
                    if v["median_abs_err"] is not None)
    best_signal = ranked[0][1] if ranked else "C6_combined_event"
    out["best_contact_signal_chosen_on_dev"] = best_signal
    out["dev_ranking"] = [{"signal": k, "median_abs_err": e} for e, k in ranked]

    # Corrector experiment.
    fit = fit_corrector_on_dev(cache, recs, est, params, best_signal)
    out["corrector_fit_on_dev"] = fit
    out["shot"] = {}
    for label, kind in (("L0", "L0"), ("L3", "L3"),
                        ("L3_plus_contact", fit["kind"] if fit else "repair")):
        for sp in ("dev", "eval"):
            r = evaluate_shot(cache, recs, est, params, kind, best_signal,
                              (fit or {}).get("pre", 10), (fit or {}).get("post", 10),
                              (fit or {}).get("margin", 8), split=sp)
            out["shot"][f"{label}_{sp}"] = r

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    with open(args.jsonl, "w", encoding="utf-8") as f:
        for cid in sorted(est):
            r = recs.get(cid, {})
            f.write(json.dumps({
                "clip_id": cid,
                "split": cache["clips"][cid]["split"],
                "identity_correct": cache["clips"][cid]["identity_correct"],
                "contact_type": r.get("contact_type"),
                "gt_contact_frame": r.get("contact_frame"),
                "estimates": {k: v["frame"] for k, v in est[cid].items()},
                "confidences": {k: v["confidence"] for k, v in est[cid].items()},
            }) + "\n")

    print(f"\nCONTACT (all clips with a usable label, n="
          f"{out['contact']['all']['C6_combined_event']['n_clips']})")
    print(f"{'signal':<24}{'medErr':>8}{'meanErr':>9}{'±1':>7}{'±2':>7}{'±3':>7}{'notFound':>10}")
    for k, v in out["contact"]["all"].items():
        f_ = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"
        print(f"{k:<24}{f_(v['median_abs_err'],1):>8}{f_(v['mean_abs_err'],1):>9}"
              f"{f_(v['within_1']):>7}{f_(v['within_2']):>7}{f_(v['within_3']):>7}"
              f"{f_(v['not_found_rate']):>10}")

    print(f"\nbest contact signal (chosen on dev): {best_signal}")
    print(f"corrector fit on dev: {fit}")
    print(f"\nSHOT LOCALIZATION")
    print(f"{'config':<26}{'mIoU':>8}{'medIoU':>9}{'@0.3':>7}{'@0.5':>7}{'rej':>5}")
    for k, v in out["shot"].items():
        f_ = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
        print(f"{k:<26}{f_(v['mean_iou']):>8}{f_(v['median_iou']):>9}"
              f"{f_(v['recall@0.3'],2):>7}{f_(v['recall@0.5'],2):>7}{v['rejected']:>5}")
    print(f"\nwrote {args.out}\nwrote {args.jsonl}")


if __name__ == "__main__":
    main()
