"""
phase3_build_annotations.py — validate the Phase-3 labels, measure
annotation agreement, and emit the machine-readable artifacts.

Produces:
    phase3_results/phase3_annotations.jsonl
    phase3_results/phase3_annotation_summary.json
    phase3_results/phase3_ground_truth_validation.json

The validation report is the point of this script, not the JSONL. Before any
S0-S4 / L0-L3 / F0-F4 ablation runs, we need to know how many clips can
actually support each metric — different experiments legitimately use
different subsets, and forcing every metric onto every clip would fabricate
targets where the video supports none.
"""

import json
import os
import statistics

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")


def agreement(ann, dbl):
    """Compare the independent re-annotation subset against pass 1."""
    from phase3_annotation_schema import (VALID_SINGLE_SHOT, CONTACT_WITH_FRAME)

    rows, coh_match, ct_match = [], 0, 0
    start_d, end_d, contact_d = [], [], []

    for cid, second in dbl.items():
        first = ann[cid]
        f_coh, f_s, f_e, f_c, f_ct = first[0], first[1], first[2], first[3], first[4]
        s_coh, s_s, s_e, s_c, s_ct = second

        row = {"clip_id": cid,
               "coherence_1": f_coh, "coherence_2": s_coh,
               "coherence_agree": f_coh == s_coh,
               "contact_type_1": f_ct, "contact_type_2": s_ct,
               "contact_type_agree": f_ct == s_ct}
        coh_match += int(f_coh == s_coh)
        ct_match += int(f_ct == s_ct)

        if f_coh == s_coh == VALID_SINGLE_SHOT and None not in (f_s, s_s, f_e, s_e):
            row["shot_start_delta"] = abs(f_s - s_s)
            row["shot_end_delta"] = abs(f_e - s_e)
            start_d.append(abs(f_s - s_s))
            end_d.append(abs(f_e - s_e))
        if f_ct in CONTACT_WITH_FRAME and s_ct in CONTACT_WITH_FRAME \
                and None not in (f_c, s_c):
            row["contact_delta"] = abs(f_c - s_c)
            contact_d.append(abs(f_c - s_c))
        rows.append(row)

    def stats(xs):
        if not xs:
            return None
        return {"n": len(xs), "mean": round(statistics.mean(xs), 2),
                "median": statistics.median(xs), "max": max(xs),
                "within_1": sum(1 for x in xs if x <= 1),
                "within_3": sum(1 for x in xs if x <= 3)}

    n = len(dbl)
    return {
        "n_double_annotated": n,
        "coherence_exact_agreement": round(coh_match / n, 4) if n else None,
        "contact_type_exact_agreement": round(ct_match / n, 4) if n else None,
        "shot_start_delta": stats(start_d),
        "shot_end_delta": stats(end_d),
        "contact_delta": stats(contact_d),
        "per_clip": rows,
    }


def main():
    from phase3_annotation_schema import (
        validate_record, summarise, COHERENCE_USABLE, CONTACT_WITH_FRAME,
        CONTACT_EXACT, CONTACT_TOLERANCE, SCHEMA_VERSION,
    )
    from phase3_annotations import ANNOTATIONS, DOUBLE_ANNOTATION, build_records

    os.makedirs(P3, exist_ok=True)

    # --- 1. account for every benchmark clip ---------------------------
    with open(os.path.join(P2, "phase2_groundtruth.json"), encoding="utf-8") as f:
        gt = json.load(f)
    bench = [c["clip_id"] for c in gt["clips"]]
    split_of = {c["clip_id"]: c["split"] for c in gt["clips"]}

    missing = [c for c in bench if c not in ANNOTATIONS]
    extra = [c for c in ANNOTATIONS if c not in bench]
    if missing or extra:
        print(f"CLIP MISMATCH — missing {missing}, extra {extra}")
    else:
        print(f"clip accounting OK: {len(bench)}/{len(bench)} benchmark clips annotated")

    # --- 2. validate every record --------------------------------------
    records = build_records()
    with open(os.path.join(P3, "phase3_event_proposals.json"), encoding="utf-8") as f:
        props = {c["clip_id"]: c for c in json.load(f)["clips"]}

    problems = {}
    for r in records:
        r["split"] = split_of.get(r["video_id"])
        p = props.get(r["video_id"], {})
        r["proposed_contact_frame"] = p.get("proposed_contact")
        r["batsman_track_id"] = p.get("batsman_track")
        r["track_anchor"] = p.get("track_anchor")
        errs = validate_record(r)
        if errs:
            problems[r["video_id"]] = errs

    if problems:
        print(f"\nSCHEMA VALIDATION FAILURES ({len(problems)}):")
        for cid, errs in problems.items():
            print(f"  {cid}: {errs}")
    else:
        print(f"schema validation OK: {len(records)}/{len(records)} records consistent")

    # --- 3. proposal quality (machine vs human) ------------------------
    deltas, prop_hits = [], 0
    for r in records:
        if r["contact_type"] in CONTACT_WITH_FRAME and r["proposed_contact_frame"] is not None:
            d = abs(r["proposed_contact_frame"] - r["contact_frame"])
            deltas.append(d)
            tol = CONTACT_TOLERANCE.get(r["contact_type"], 3)
            prop_hits += int(d <= tol)
    proposal = {
        "n_comparable": len(deltas),
        "mean_abs_error_frames": round(statistics.mean(deltas), 2) if deltas else None,
        "median_abs_error_frames": statistics.median(deltas) if deltas else None,
        "within_per_type_tolerance": prop_hits,
        "within_tolerance_rate": round(prop_hits / len(deltas), 4) if deltas else None,
        "note": ("Machine proposal vs human label on clips where a contact frame "
                 "exists. This is why the two are stored separately: the proposal "
                 "must never become the ground truth it is graded against."),
    }

    # --- 4. summary + coverage ------------------------------------------
    summary = summarise(records)
    summary["by_split"] = {}
    for sp in ("dev", "eval"):
        rs = [r for r in records if r["split"] == sp]
        summary["by_split"][sp] = {
            "clips": len(rs),
            "valid_single_shot": sum(1 for r in rs if r["coherence"] in COHERENCE_USABLE),
            "usable_contact": sum(1 for r in rs if r["contact_type"] in CONTACT_WITH_FRAME),
        }
    summary["proposal_quality"] = proposal
    summary["agreement"] = agreement(ANNOTATIONS, DOUBLE_ANNOTATION)

    # --- 5. write artifacts ---------------------------------------------
    jsonl = os.path.join(P3, "phase3_annotations.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with open(os.path.join(P3, "phase3_annotation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    usable_shot = [r for r in records if r["coherence"] in COHERENCE_USABLE]
    usable_contact = [r for r in records if r["contact_type"] in CONTACT_WITH_FRAME]
    exact = [r for r in usable_contact if r["contact_type"] == CONTACT_EXACT]
    low_conf = [r for r in records if "LOW" in (r.get("shot_confidence"),
                                                r.get("contact_confidence"))]

    validation = {
        "schema": SCHEMA_VERSION,
        "total_clips": len(records),
        "schema_validation_failures": problems,
        "reliable_shot_boundaries": {
            "count": len(usable_shot),
            "dev": sum(1 for r in usable_shot if r["split"] == "dev"),
            "eval": sum(1 for r in usable_shot if r["split"] == "eval"),
            "excluding_low_confidence": sum(1 for r in usable_shot
                                            if r["shot_confidence"] != "LOW"),
        },
        "usable_contact_labels": {
            "count": len(usable_contact),
            "exact_ball_visible": len(exact),
            "contact_adjacent": len(usable_contact) - len(exact),
            "dev": sum(1 for r in usable_contact if r["split"] == "dev"),
            "eval": sum(1 for r in usable_contact if r["split"] == "eval"),
        },
        "usable_phase_labels": {
            "count": 0,
            "note": ("No per-phase spans were annotated. The five-phase vocabulary "
                     "exists in the schema, but on 30fps footage the STANCE/BACKLIFT "
                     "and DOWNSWING/CONTACT boundaries could not be placed "
                     "reproducibly from the sheets, and guessing them would make "
                     "phase-accuracy metrics unfalsifiable. Phase metrics are "
                     "therefore NOT computable and must not be reported."),
        },
        "too_ambiguous": {
            "multiple_shots": summary["multiple_shots"],
            "scene_cut": summary["scene_cut"],
            "excessive_duration": summary["excessive_duration"],
            "no_shot": summary["no_shot"],
            "insufficient_action": summary["insufficient_action"],
            "ambiguous": summary["ambiguous"],
        },
        "low_confidence_clips": [r["video_id"] for r in low_conf],
        "benchmark_subsets": {
            "L0-L3_shot_localization": {
                "n": len(usable_shot),
                "clips": [r["video_id"] for r in usable_shot],
            },
            "contact_evaluation": {
                "n": len(usable_contact),
                "clips": [r["video_id"] for r in usable_contact],
            },
            "S0-S4_batsman_identification": {
                "n": len(records),
                "note": "All 51 clips: Phase-2 batsman boxes already cover every clip.",
            },
            "F0-F4_frame_selection": {
                "n": len(usable_shot),
                "note": ("Requires a bounded shot to select frames within, so it "
                         "shares the shot-localization subset."),
            },
            "phase_evaluation": {"n": 0, "note": "Not computable — see usable_phase_labels."},
        },
    }
    with open(os.path.join(P3, "phase3_ground_truth_validation.json"), "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=1)

    # --- 6. print the checkpoint ----------------------------------------
    print(f"\n{'='*64}\nGROUND-TRUTH COVERAGE\n{'='*64}")
    print(f"  total clips                    : {len(records)}")
    for k, v in summary["coherence_distribution"].items():
        print(f"    {k:<24} {v}")
    print(f"\n  reliable shot boundaries       : {len(usable_shot)}"
          f"  (dev {validation['reliable_shot_boundaries']['dev']},"
          f" eval {validation['reliable_shot_boundaries']['eval']})")
    print(f"  usable contact labels          : {len(usable_contact)}"
          f"  (EXACT {len(exact)}, ADJACENT {len(usable_contact)-len(exact)})")
    print(f"  usable phase labels            : 0  (not annotatable — see report)")
    print(f"  LOW-confidence clips           : {len(low_conf)}")

    a = summary["agreement"]
    print(f"\n{'='*64}\nANNOTATION AGREEMENT ({a['n_double_annotated']} clips re-annotated)\n{'='*64}")
    print(f"  coherence exact agreement      : {a['coherence_exact_agreement']:.1%}")
    print(f"  contact-type exact agreement   : {a['contact_type_exact_agreement']:.1%}")
    for k in ("shot_start_delta", "shot_end_delta", "contact_delta"):
        s = a[k]
        if s:
            print(f"  {k:<30} : mean {s['mean']} median {s['median']} "
                  f"max {s['max']}  (<=1: {s['within_1']}/{s['n']})")

    print(f"\n{'='*64}\nPROPOSAL QUALITY (machine vs human)\n{'='*64}")
    print(f"  comparable clips               : {proposal['n_comparable']}")
    print(f"  mean |error|                   : {proposal['mean_abs_error_frames']} frames")
    print(f"  median |error|                 : {proposal['median_abs_error_frames']} frames")
    print(f"  within per-type tolerance      : {proposal['within_tolerance_rate']:.1%}")

    print(f"\nwrote {jsonl}")
    print(f"wrote {os.path.join(P3, 'phase3_annotation_summary.json')}")
    print(f"wrote {os.path.join(P3, 'phase3_ground_truth_validation.json')}")
    return 0 if not problems and not missing and not extra else 1


if __name__ == "__main__":
    raise SystemExit(main())
