"""
phase3_annotations.py — the Phase-3 temporal ground-truth labels.

Produced by visually inspecting one event sheet per clip
(`phase3_results/event_sheets/`) against the operational definitions in
`docs/PHASE3_ANNOTATION_PROTOCOL.md`.

Sheets are anchored on the ANNOTATED batsman track (Phase-2 boxes), not on
the model's predicted track — building temporal ground truth on the model's
guess would inherit its errors, and on several clips the geometry model
selects the near-camera feeder.

Format per clip:
    (coherence, shot_start, shot_end, contact_frame, contact_type,
     shot_conf, contact_conf, scene_cut_frames, note)

shot_start/shot_end/contact_frame are None wherever the coherence verdict or
contact type does not license them; `phase3_annotation_schema.validate_record`
enforces that rather than trusting this table.

ANNOTATOR LIMITATIONS, stated plainly: single annotator, not blind to the
machine proposal (it is drawn on the sheet), no second independent rater
beyond the re-annotation subset in `DOUBLE_ANNOTATION`. Confidence levels are
recorded per field so a sensitivity analysis can exclude LOW-confidence rows.
"""

from phase3_annotation_schema import (
    VALID_SINGLE_SHOT, NO_SHOT, MULTIPLE_SHOTS, SCENE_CUT,
    EXCESSIVE_DURATION, INSUFFICIENT_ACTION, AMBIGUOUS,
    CONTACT_EXACT, CONTACT_ADJACENT, CONTACT_AMBIGUOUS, CONTACT_NOT_VISIBLE,
    CONF_HIGH, CONF_MEDIUM, CONF_LOW,
)

V, NS, MS, SC, ED, IA, AM = (VALID_SINGLE_SHOT, NO_SHOT, MULTIPLE_SHOTS,
                             SCENE_CUT, EXCESSIVE_DURATION,
                             INSUFFICIENT_ACTION, AMBIGUOUS)
EX, ADJ, CAM, NV = (CONTACT_EXACT, CONTACT_ADJACENT,
                    CONTACT_AMBIGUOUS, CONTACT_NOT_VISIBLE)
H, M, L = CONF_HIGH, CONF_MEDIUM, CONF_LOW

ANNOTATIONS = {
    # ---- clips with a cleanly boundable single stroke -------------------
    "instagram1_front_01.mp4": (V, 5, 28, 16, EX, M, H, [],
        "Ball visible approaching f11-15, at bat near ground f16, gone f17. "
        "A second stroke begins ~f85 but does not complete in-window."),
    "instagram2_front_03.mp4": (V, 26, 45, 33, EX, H, H, [],
        "Pink ball clearly visible approaching f27-31, at the bat f33."),
    "pro_player_front_44.mp4": (V, 23, 45, 31, EX, H, H, [],
        "White ball clearly adjacent to bat face at f31."),
    "roa_front_1.mp4": (V, 110, 145, 127, EX, H, H, [],
        "60fps. Red ball clearly visible: approaching f121-126, at bat f127, "
        "bat through f128. Best contact evidence in the corpus."),

    "instagram2_front_05.mp4": (V, 38, 58, 50, ADJ, M, M, [],
        "Bat descends f48-50; ball not resolvable."),
    "instagram2_front_07.mp4": (V, 34, 52, 40, ADJ, M, M, [], ""),
    "beginnerunorthodox1_front_01.mp4": (V, 48, 70, 58, ADJ, L, L, [],
        "Batsman is distant; track crop drifts toward the near bowler on "
        "f44-53, so both boundary and contact are LOW confidence."),
    "kohli_side_08.mp4": (V, 20, 45, 35, ADJ, M, M, [], ""),
    "kohli_side_10.mp4": (V, 78, 102, 90, ADJ, M, M, [], ""),
    "kohli_side_14.mp4": (V, 43, 62, 51, ADJ, M, M, [],
        "6s clip of continuous practice; this is the one cleanly boundable stroke."),
    "kohli_side_16.mp4": (V, 0, 20, 9, ADJ, M, M, [],
        "Clip opens on the stroke; start_clipped. Batsman leaves frame after ~f62."),
    "kohli_side_20.mp4": (V, 35, 58, 46, ADJ, M, M, [], ""),
    "kohli_side_24.mp4": (V, 42, 62, 51, ADJ, M, M, [], ""),
    "kohli_side_28.mp4": (V, 52, 76, 63, ADJ, M, M, [], ""),
    "kohli_side_32.mp4": (V, 55, 78, 66, ADJ, M, M, [], ""),
    "pro_player_back_23.mp4": (V, 18, 45, 30, ADJ, M, M, [], ""),
    "pro_player_back_25.mp4": (V, 48, 70, 61, ADJ, L, L, [],
        "Back view, bat largely occluded by the body; contact is inferred "
        "from arm extension rather than seen."),
    "pro_player_back_27.mp4": (V, 50, 78, 63, ADJ, M, M, [], ""),
    "pro_player_back_31.mp4": (V, 42, 68, 56, ADJ, M, M, [], ""),
    "pro_player_back_33.mp4": (V, 20, 45, 31, ADJ, M, M, [], ""),
    "pro_player_back_38.mp4": (V, 13, 38, 23, ADJ, M, M, [], ""),
    "pro_player_front_01.mp4": (V, 28, 48, 36, ADJ, M, M, [], ""),
    "pro_player_front_03.mp4": (V, 55, 78, 64, ADJ, M, M, [], ""),
    "pro_player_front_06.mp4": (V, 38, 58, 51, ADJ, M, M, [],
        "The clip whose batsman the Phase-2 geometry model gets wrong "
        "(stationary feeder). GT anchoring fixes the sheet."),
    "pro_player_front_10.mp4": (V, 39, 58, 50, ADJ, M, M, [], ""),
    "pro_player_front_12.mp4": (V, 30, 50, 39, ADJ, M, M, [], ""),
    "pro_player_front_14.mp4": (V, 5, 28, 16, ADJ, M, M, [], ""),
    "pro_player_front_16.mp4": (V, 15, 35, 24, ADJ, M, M, [], ""),

    # ---- more than one stroke, no single boundable window ---------------
    "Sai_front_3.mp4": (MS, None, None, None, CAM, M, M, [],
        "11.6s of continuous practice. Proposal landed at f5 (clip start) - "
        "clearly wrong, a good proposal-failure example."),
    "instagram1_front_03.mp4": (MS, None, None, None, CAM, M, M, [],
        "Strokes ~f8-21 and ~f76-89. Proposal f24 landed on a held "
        "follow-through pose, so the fine strip never covered a real contact."),
    "instagram2_front_09.mp4": (MS, None, None, None, CAM, M, M, [],
        "Repeated held backlift positions."),
    "kohli_front_02.mp4": (MS, None, None, None, CAM, M, M, [],
        "6s of continuous net practice, several strokes."),
    "kohli_side_04.mp4": (MS, None, None, None, CAM, M, M, [], ""),
    "kohli_side_06.mp4": (MS, None, None, None, CAM, M, M, [],
        "Shadow-batting / repeated bat lifts, no isolable delivery."),
    "kohli_side_12.mp4": (MS, None, None, None, CAM, M, M, [],
        "Proposal f68 lands on the batsman walking, not a stroke."),
    "kohli_side_18.mp4": (MS, None, None, None, CAM, M, M, [], ""),
    "kohli_side_26.mp4": (MS, None, None, None, CAM, M, M, [], ""),
    "kohli_side_30.mp4": (MS, None, None, None, CAM, M, M, [], ""),
    "pro_player_back_29.mp4": (MS, None, None, None, CAM, M, M, [],
        "Repeated forward defensives; strokes are not separated by a reset."),
    "pro_player_front_08.mp4": (MS, None, None, None, CAM, M, M, [],
        "Clip opens mid-stroke and contains later action."),

    # ---- scene cuts ------------------------------------------------------
    "pro_player_front_20.mp4": (SC, None, None, None, NV, H, H, [2],
        "f0 is front-view nets; from ~f4 an entirely different setup."),
    "pro_player_front_18.mp4": (SC, None, None, None, NV, H, H, [10],
        "Nets scene through ~f4, stone-wall scene from ~f17."),
    "pro_player_back_40.mp4": (SC, None, None, None, NV, H, H, [78],
        "Nets through ~f76, blue-tarp scene from ~f81."),

    # ---- windows that are whole sessions ---------------------------------
    "Sai_front_1.mp4": (ED, None, None, None, NV, H, H, [],
        "20.8s net session. Fine strip shows the batsman occluded by a net pole."),
    "rishi_front_2.mp4": (ED, None, None, None, NV, H, H, [],
        "77.9s session. Track drifts onto a near-camera person."),
    "sanjay_front _1.mp4": (ED, None, None, None, NV, H, H, [],
        "200s session, many deliveries, camera repositioned. No contact proposal "
        "was produced at all."),

    # ---- no stroke -------------------------------------------------------
    "pro_player_front_42.mp4": (NS, None, None, None, NV, H, H, [],
        "A player walks past carrying a bat; no delivery is played."),

    # ---- genuinely ambiguous --------------------------------------------
    "instagram2_front_01.mp4": (AM, None, None, None, CAM, L, L, [],
        "Held backlift positions; cannot tell shadow practice from played "
        "deliveries at this framing."),
    "pro_player_back_21.mp4": (AM, None, None, None, CAM, L, L, [],
        "Batsman mostly static with bat raised; no clear delivery."),
    "pro_player_back_36.mp4": (AM, None, None, None, CAM, L, L, [],
        "Cut detector fired at f55/f59 but the visual evidence reads as a "
        "zoom/exposure change rather than a hard cut - recorded as a "
        "detector false positive rather than a scene cut."),
    "kohli_side_22.mp4": (IA, None, None, None, CAM, M, M, [],
        "Clip opens mid-downswing; contact is at or before f2, so the stroke "
        "cannot be bounded."),
}


# Independent re-annotation of a subset, performed from the sheets alone
# without consulting the table above. Same (coherence, start, end, contact,
# contact_type) tuple shape, for the agreement analysis required by the
# protocol's quality-control section.
DOUBLE_ANNOTATION = {
    "instagram1_front_01.mp4": (V, 6, 29, 16, EX),
    "instagram2_front_03.mp4": (V, 25, 44, 33, EX),
    "roa_front_1.mp4": (V, 112, 144, 127, EX),
    "pro_player_front_44.mp4": (V, 24, 44, 31, EX),
    "kohli_side_24.mp4": (V, 44, 63, 52, ADJ),
    "kohli_side_28.mp4": (V, 53, 75, 62, ADJ),
    "pro_player_back_33.mp4": (V, 22, 44, 32, ADJ),
    "pro_player_front_12.mp4": (V, 31, 49, 40, ADJ),
    "pro_player_front_16.mp4": (V, 16, 36, 25, ADJ),
    "pro_player_back_31.mp4": (V, 43, 67, 57, ADJ),
    "pro_player_front_20.mp4": (SC, None, None, None, NV),
    "pro_player_front_42.mp4": (NS, None, None, None, NV),
    "sanjay_front _1.mp4": (ED, None, None, None, NV),
    "kohli_side_06.mp4": (MS, None, None, None, CAM),
    "pro_player_back_36.mp4": (AM, None, None, None, CAM),
    # Disagreement, deliberately kept: pass 1 called this MULTIPLE_SHOTS,
    # pass 2 judged one stroke boundable at f8-f24. Classified in the report
    # as a DEFINITION ambiguity, not an annotation error.
    "instagram1_front_03.mp4": (V, 8, 24, 16, ADJ),
}


def build_records(annotator="A"):
    from phase3_annotation_schema import blank_record
    out = []
    for cid, (coh, s, e, c, ct, sconf, cconf, cuts, note) in ANNOTATIONS.items():
        r = blank_record(cid, annotator)
        r.update(coherence=coh, shot_valid=(coh == VALID_SINGLE_SHOT),
                 shot_start_frame=s, shot_end_frame=e,
                 shot_confidence=sconf, contact_frame=c, contact_type=ct,
                 contact_confidence=cconf,
                 scene_cut=bool(cuts), scene_cut_frames=list(cuts),
                 multiple_shots=(coh == MULTIPLE_SHOTS),
                 no_shot=(coh == NO_SHOT), notes=note)
        out.append(r)
    return out
