"""
phase3e_leakage_analysis.py — what the EXISTING re-annotation can and cannot
tell us about C0 leakage.

THE QUESTION
Phase 3D found C0 localizes contact to a median of 1 frame, but the annotator
was not blind to C0 (it is drawn on the contact sheet). Is C0 genuinely
accurate, or were the labels pulled toward it?

WHAT THIS SCRIPT CAN ANSWER, AND WHAT IT CANNOT
A fully blind test needs an annotator who never saw C0. That does not exist
yet and cannot be manufactured here. What DOES exist is `DOUBLE_ANNOTATION`:
16 clips re-annotated "from the sheets alone without consulting" pass 1. So:

    pass 1  saw the sheets (C0 drawn on them), did not see pass 2
    pass 2  saw the sheets (C0 drawn on them), did not see pass 1

Both passes saw C0, so this is NOT a leakage-free test. But it supports one
informative inference:

    If C0 were DRIVING the labels, both passes would be pulled toward C0 and
    |pass1 - C0| and |pass2 - C0| would both be SMALLER than |pass1 - pass2|.

    If the two passes agree with each other MORE closely than either agrees
    with C0, they share a signal C0 does not have -- i.e. they are reading the
    video, not copying the marker.

That is a falsifiable, directional test on evidence already in the repo. It
cannot prove independence, and it is reported as weaker than a blind study.
"""

import argparse
import json
import os

import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.join(_MODULE_DIR, "phase3_results")


def load():
    from phase3_annotations import ANNOTATIONS, DOUBLE_ANNOTATION
    import phase3_contact_benchmark as CB

    rows = []
    for cid, d in DOUBLE_ANNOTATION.items():
        a = ANNOTATIONS.get(cid)
        if not a:
            continue
        p1 = {"coherence": a[0], "start": a[1], "end": a[2], "contact": a[3],
              "type": a[4]}
        p2 = {"coherence": d[0], "start": d[1], "end": d[2], "contact": d[3],
              "type": d[4]}
        rows.append({"clip_id": cid, "pass1": p1, "pass2": p2,
                     "c0": CB.c0_proposal(cid)})
    return rows


def summarise(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return {"n": len(v),
            "median": round(float(np.median(v)), 2),
            "mean": round(float(np.mean(v)), 2),
            "max": int(max(v)),
            "within_1": round(sum(1 for x in v if x <= 1) / len(v), 4),
            "within_2": round(sum(1 for x in v if x <= 2) / len(v), 4),
            "within_3": round(sum(1 for x in v if x <= 3) / len(v), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(P3, "phase3e_leakage_analysis.json"))
    args = ap.parse_args()

    rows = load()
    usable = [r for r in rows
              if r["pass1"]["contact"] is not None
              and r["pass2"]["contact"] is not None
              and r["c0"] is not None]

    hh = [abs(r["pass1"]["contact"] - r["pass2"]["contact"]) for r in usable]
    c1 = [abs(r["c0"] - r["pass1"]["contact"]) for r in usable]
    c2 = [abs(r["c0"] - r["pass2"]["contact"]) for r in usable]

    # Shot boundaries, on clips both passes called a valid single shot.
    both_valid = [r for r in rows
                  if r["pass1"]["start"] is not None and r["pass2"]["start"] is not None]
    hh_start = [abs(r["pass1"]["start"] - r["pass2"]["start"]) for r in both_valid]
    hh_end = [abs(r["pass1"]["end"] - r["pass2"]["end"]) for r in both_valid]

    # Does the contact-to-boundary offset convention hold in BOTH passes?
    off = {}
    for k, p in (("pass1", "pass1"), ("pass2", "pass2")):
        pre = [r[p]["contact"] - r[p]["start"] for r in both_valid
               if r[p]["contact"] is not None]
        post = [r[p]["end"] - r[p]["contact"] for r in both_valid
                if r[p]["contact"] is not None]
        off[k] = {
            "n": len(pre),
            "contact_minus_start": {"mean": round(float(np.mean(pre)), 2),
                                    "sd": round(float(np.std(pre)), 2)},
            "end_minus_contact": {"mean": round(float(np.mean(post)), 2),
                                  "sd": round(float(np.std(post)), 2)},
        }

    hh_s, c1_s, c2_s = summarise(hh), summarise(c1), summarise(c2)
    verdict = None
    if hh_s and c1_s and c2_s:
        human_closer = hh_s["mean"] < min(c1_s["mean"], c2_s["mean"])
        verdict = (
            "AGAINST strong leakage — the two annotation passes agree with each "
            "other more closely than either agrees with C0, so they share "
            "information C0 does not carry."
            if human_closer else
            "CONSISTENT WITH leakage — each pass sits closer to C0 than the two "
            "passes sit to each other, which is what copying the marker would "
            "produce.")

    out = {
        "what_this_is": ("Partial leakage test using the EXISTING re-annotation. "
                         "Both passes saw the contact sheets with C0 drawn on "
                         "them, so this is NOT a blind study and cannot prove "
                         "independence."),
        "n_double_annotated": len(rows),
        "n_usable_for_contact": len(usable),
        "contact": {
            "human_vs_human": hh_s,
            "C0_vs_pass1": c1_s,
            "C0_vs_pass2": c2_s,
            "directional_verdict": verdict,
        },
        "shot_boundaries": {
            "n": len(both_valid),
            "human_vs_human_start": summarise(hh_start),
            "human_vs_human_end": summarise(hh_end),
        },
        "contact_offset_convention": off,
        "per_clip": [
            {"clip_id": r["clip_id"],
             "pass1_contact": r["pass1"]["contact"],
             "pass2_contact": r["pass2"]["contact"],
             "c0": r["c0"],
             "human_human_delta": (abs(r["pass1"]["contact"] - r["pass2"]["contact"])
                                   if r["pass1"]["contact"] is not None
                                   and r["pass2"]["contact"] is not None else None),
             "c0_vs_pass1": (abs(r["c0"] - r["pass1"]["contact"])
                             if r["c0"] is not None and r["pass1"]["contact"] is not None else None),
             "c0_vs_pass2": (abs(r["c0"] - r["pass2"]["contact"])
                             if r["c0"] is not None and r["pass2"]["contact"] is not None else None)}
            for r in rows],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(f"double-annotated clips: {len(rows)}   usable for contact: {len(usable)}")
    print(f"\n{'comparison':<22}{'n':>4}{'median':>8}{'mean':>7}{'max':>5}{'±1':>7}{'±3':>7}")
    for lbl, s in (("human vs human", hh_s), ("C0 vs pass 1", c1_s),
                   ("C0 vs pass 2", c2_s)):
        if s:
            print(f"{lbl:<22}{s['n']:>4}{s['median']:>8.1f}{s['mean']:>7.2f}"
                  f"{s['max']:>5}{s['within_1']:>7.2f}{s['within_3']:>7.2f}")
    print(f"\nverdict: {verdict}")
    print(f"\ncontact-offset convention:")
    for k, v in off.items():
        print(f"  {k}: contact-start {v['contact_minus_start']['mean']} "
              f"+/- {v['contact_minus_start']['sd']}   "
              f"end-contact {v['end_minus_contact']['mean']} "
              f"+/- {v['end_minus_contact']['sd']}  (n={v['n']})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
