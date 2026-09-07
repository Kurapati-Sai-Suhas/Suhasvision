"""
phase3_best7_audit.py — human reference, failure breakdown and visual audit
for the F0-F4 Best-7 ablation.

ON THE "HUMAN REFERENCE"
No expert frame-selection reference exists for this corpus, and one cannot be
manufactured here: asking which seven frames a coach would choose requires a
coach. What DOES exist is the human temporal annotation — shot bounds and a
contact frame — labelled by a person.

So the reference used here is an ANNOTATION-DERIVED sequence: the seven phase
anchors placed on the HUMAN shot interval and HUMAN contact frame. It encodes
a human's judgement of when the event happened and where impact was, but NOT a
human's judgement of which frames are biomechanically informative. It is a
proxy, and it is labelled as one everywhere it appears.

It is used for EVALUATION ONLY. No F method sees it (§16).
"""

import argparse
import json
import os

import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.join(_MODULE_DIR, "phase3_results")

# The production phase anchors, as fractions of onset->contact->end.
ANCHORS = (0.00, 0.30, 0.55, 0.75, 0.92, 1.00, 1.45)


def reference_sequence(start, end, contact, n_frames):
    """Seven phase anchors on the HUMAN interval. u=1.0 is contact; u>1 is
    follow-through past contact, matching PHASE_ANCHORS' convention."""
    if start is None or contact is None:
        return None
    pre = max(1, contact - start)
    post = max(1, end - contact)
    out = []
    for u in ANCHORS:
        f = (start + u * pre) if u <= 1.0 else (contact + (u - 1.0) * post / 0.45)
        out.append(int(round(min(max(f, 0), n_frames - 1))))
    # Enforce strict chronology without inventing frames.
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = min(out[i - 1] + 1, n_frames - 1)
    return out


def seq_distance(a, b):
    """Mean absolute frame distance between two seven-frame sequences,
    position by position. Both are chronological, so position i of one is
    directly comparable to position i of the other."""
    if not a or not b or len(a) != len(b):
        return None
    return round(float(np.mean([abs(x - y) for x, y in zip(a, b)])), 2)


def coverage_overlap(a, b, tol=3):
    """Fraction of reference frames matched by some selected frame within
    `tol`. Less strict than positional distance and closer to 'did it look at
    the same moments'."""
    if not a or not b:
        return None
    return round(sum(1 for r in b if any(abs(r - s) <= tol for s in a)) / len(b), 4)


def classify(row, name):
    """One documented cause per clip that produced no usable sequence, or a
    degraded one. Upstream causes are attributed upstream."""
    m = row["methods"].get(name) or {}
    if not row["identity_correct"]:
        return "identity failure — wrong batsman selected upstream"
    if row["pool_size"] == 0:
        return "frame-quality failure — no frame in the region has a batsman box"
    if not m.get("frames"):
        return f"optimizer refusal — {m.get('reason') or 'no sequence returned'}"
    bio = m.get("biomech") or {}
    if bio.get("missing_frames", 0) >= 3:
        return "pose failure — 3+ selected frames yield no landmarks"
    if m.get("duplicates", 0) > 0:
        return "excessive clustering — duplicate frames selected"
    if m.get("min_gap", 99) <= 1:
        return "excessive clustering — selected frames adjacent"
    if row["gt_shot"] and row["contact_estimate"] is not None and row["gt_contact"] is not None:
        if abs(row["contact_estimate"] - row["gt_contact"]) > 5:
            return "contact failure — anchor more than 5 frames from labelled contact"
    if row["gt_shot"]:
        s, e = row["gt_shot"]
        fr = m["frames"]
        if fr[-1] < s or fr[0] > e:
            return "shot-localization failure — selected frames outside the labelled shot"
    return None


def timeline(rows, out_path, method="F4_optimized", width=1180, row_h=40):
    import cv2
    if not rows:
        return None
    h = 70 + row_h * len(rows)
    img = np.full((h, width, 3), 22, np.uint8)
    x0, x1 = 260, width - 24

    def lab(t, org, col, s=0.4):
        t = t.replace("—", "-").encode("ascii", "replace").decode("ascii")
        cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, col, 1, cv2.LINE_AA)

    lab(f"Best-7 audit  [{method}]", (12, 24), (240, 240, 240), 0.55)
    lab("grey=search region  orange=GT shot  blue=L3 prior  magenta=contact  green=selected 7",
        (12, 46), (190, 190, 190), 0.38)

    for i, r in enumerate(rows):
        y = 66 + i * row_h
        n = max(r["n_frames"], 1)
        sx = lambda f: int(x0 + (x1 - x0) * max(0, min(f, n)) / n)
        lab(r["clip_id"][:30], (10, y + 10), (205, 205, 205), 0.36)
        lab(r["tag"][:44], (10, y + 24), (150, 150, 200), 0.32)
        reg = r["region"]
        cv2.rectangle(img, (sx(reg[0]), y + 2), (sx(reg[1]), y + 28), (58, 58, 58), -1)
        if r["gt"]:
            cv2.rectangle(img, (sx(r["gt"][0]), y + 4), (sx(r["gt"][1]), y + 11),
                          (0, 170, 255), -1)
        if r["l3"]:
            cv2.rectangle(img, (sx(r["l3"][0]), y + 13), (sx(r["l3"][1]), y + 18),
                          (255, 160, 0), -1)
        if r["contact"] is not None:
            cv2.line(img, (sx(r["contact"]), y + 2), (sx(r["contact"]), y + 28),
                     (255, 0, 255), 1)
        for f in (r["selected"] or []):
            cv2.circle(img, (sx(f), y + 24), 3, (0, 230, 0), -1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(P3, "phase3_best7_results.jsonl"))
    ap.add_argument("--out", default=os.path.join(P3, "phase3_best7_audit.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    methods = list(rows[0]["methods"].keys())

    # ---- annotation-derived reference ----
    ref_rows = []
    for r in rows:
        if not r["gt_shot"] or r["gt_contact"] is None:
            continue
        n = max((r["region"][1] + 1), (r["gt_shot"][1] + 1))
        ref = reference_sequence(r["gt_shot"][0], r["gt_shot"][1],
                                 r["gt_contact"], n + 1)
        if not ref:
            continue
        entry = {"clip_id": r["clip_id"], "split": r["split"], "reference": ref}
        for m in methods:
            sel = (r["methods"].get(m) or {}).get("frames")
            entry[m] = {"distance": seq_distance(sel, ref),
                        "coverage_within_3": coverage_overlap(sel, ref)}
        ref_rows.append(entry)

    ref_summary = {}
    for m in methods:
        d = [e[m]["distance"] for e in ref_rows if e[m]["distance"] is not None]
        c = [e[m]["coverage_within_3"] for e in ref_rows
             if e[m]["coverage_within_3"] is not None]
        ref_summary[m] = {
            "n": len(d),
            "median_frame_distance": round(float(np.median(d)), 2) if d else None,
            "mean_frame_distance": round(float(np.mean(d)), 2) if d else None,
            "mean_coverage_within_3": round(float(np.mean(c)), 4) if c else None,
        }

    # ---- failure breakdown ----
    failures = {}
    for m in methods:
        cats = {}
        for r in rows:
            if r["coherence"] != "VALID_SINGLE_SHOT":
                continue
            c = classify(r, m)
            if c:
                cats.setdefault(c, []).append(r["clip_id"])
        failures[m] = {"n_affected": sum(len(v) for v in cats.values()),
                       "by_cause": {k: sorted(v) for k, v in
                                    sorted(cats.items(), key=lambda kv: -len(kv[1]))}}

    out = {
        "human_reference": {
            "kind": "ANNOTATION-DERIVED, not an expert frame-selection reference",
            "definition": ("Seven production phase anchors placed on the HUMAN "
                           "shot interval and HUMAN contact frame."),
            "limitation": ("Encodes a human's judgement of WHEN the event and "
                           "impact occurred, not which frames are "
                           "biomechanically informative. A true expert "
                           "reference requires a coach and does not exist for "
                           "this corpus."),
            "n_clips": len(ref_rows),
            "summary": ref_summary,
            "per_clip": ref_rows,
        },
        "failure_modes": failures,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    # ---- visual audit ----
    sheets = {}
    for m in ("F4_optimized", "F0_uniform"):
        picked, seen = [], set()
        scored = [r for r in rows if r["coherence"] == "VALID_SINGLE_SHOT"]
        good = sorted(
            (r for r in scored if (r["methods"].get(m) or {}).get("frames")),
            key=lambda r: -((r["methods"][m].get("biomech") or {}).get("n_frames_with_angles") or 0))
        for r in good[:3]:
            picked.append((r, "success - most usable angle frames"))
            seen.add(r["clip_id"])
        for cause, ids in failures[m]["by_cause"].items():
            for cid in ids:
                if cid in seen:
                    continue
                r = next((x for x in rows if x["clip_id"] == cid), None)
                if r:
                    picked.append((r, cause))
                    seen.add(cid)
                break
        disp = [{"clip_id": r["clip_id"], "tag": tag,
                 "n_frames": max(r["region"][1] + 10, (r["gt_shot"] or [0, 1])[1] + 10),
                 "region": r["region"], "gt": r["gt_shot"], "l3": r["l3"],
                 "contact": r["contact_estimate"],
                 "selected": (r["methods"].get(m) or {}).get("frames")}
                for r, tag in picked]
        sheets[m] = timeline(disp, os.path.join(P3, f"best7_timeline_{m}.png"), m)
    out["timeline_sheets"] = sheets
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print("ANNOTATION-DERIVED REFERENCE (proxy, not an expert selection)")
    print(f"{'method':<22}{'n':>4}{'medDist':>9}{'meanDist':>10}{'cover<=3':>10}")
    for m, v in ref_summary.items():
        f_ = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"
        print(f"{m:<22}{v['n']:>4}{f_(v['median_frame_distance'],1):>9}"
              f"{f_(v['mean_frame_distance'],1):>10}{f_(v['mean_coverage_within_3']):>10}")

    print("\nFAILURE MODES (valid clips)")
    for m, v in failures.items():
        print(f"  {m}  (n={v['n_affected']})")
        for cause, ids in v["by_cause"].items():
            print(f"      {len(ids):>2}  {cause}")
    print(f"\nwrote {args.out}")
    for m, p in sheets.items():
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
