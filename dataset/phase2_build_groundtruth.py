"""
phase2_build_groundtruth.py — merge the human annotation labels with the
proposal boxes to produce the Phase-2 ground-truth benchmark.

The labels below were produced by visually inspecting one contact sheet per
clip (see phase2_annotate.py), where each sampled frame shows every proposal
as a separate numbered crop. The annotator recorded, per frame, WHICH crop is
the batsman.

Label encoding, per sampled frame:
    int   index of the proposal that is the batsman
    MISS  batsman is clearly visible but NO proposal covers them
          -> counts in the recall DENOMINATOR and against every detector,
             including the proposal detector (this is what keeps the
             proposal-anchored ground truth from flattering itself)
    ABSENT  no batsman in the frame at all (camera moved, clip ended,
            between deliveries) -> excluded from the denominator entirely

Keeping MISS and ABSENT distinct matters: conflating them would either
invent misses or hide them.
"""

import json
import os

MISS = "MISS"
ABSENT = "ABSENT"

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_MODULE_DIR, "phase2_results")

# clip_id -> per-sampled-frame batsman label (6 frames per clip)
LABELS = {
    "Sai_front_1.mp4":                  [0, 0, 0, 0, 0, 0],
    "Sai_front_3.mp4":                  [0, 0, 0, 0, 0, 0],
    "beginnerunorthodox1_front_01.mp4": [0, 1, 2, 2, 1, 1],
    "instagram1_front_01.mp4":          [0, 1, 1, 1, 1, 1],
    "instagram1_front_03.mp4":          [1, 1, 1, 1, 1, 1],
    "instagram2_front_01.mp4":          [0, 0, 0, 0, 0, 0],
    "instagram2_front_03.mp4":          [0, 0, 0, 0, 0, 0],
    "instagram2_front_05.mp4":          [0, 0, 0, 0, 0, 0],
    "instagram2_front_07.mp4":          [0, 0, 0, 0, 0, 0],
    "instagram2_front_09.mp4":          [0, 0, 0, 0, 0, 0],
    "kohli_front_02.mp4":               [0, 0, 0, 0, 0, 0],
    "kohli_side_04.mp4":                [0, 0, 0, 0, 0, 0],
    "kohli_side_06.mp4":                [0, 0, 0, 0, 0, 0],
    "kohli_side_08.mp4":                [0, 0, 0, 0, 0, ABSENT],
    "kohli_side_10.mp4":                [0, 0, 0, 0, 0, 0],
    "kohli_side_12.mp4":                [0, 0, 0, 0, 0, 0],
    "kohli_side_14.mp4":                [1, 0, 0, 0, 0, 1],
    "kohli_side_16.mp4":                [0, 0, ABSENT, 0, 0, 0],
    "kohli_side_18.mp4":                [0, 0, 0, 0, 0, 0],
    "kohli_side_20.mp4":                [0, 0, 0, 0, 1, 0],
    "kohli_side_22.mp4":                [0, 0, 0, 0, 1, 0],
    "kohli_side_24.mp4":                [0, 0, 0, 0, 0, 0],
    "kohli_side_26.mp4":                [0, 0, 0, 0, 0, 0],
    "kohli_side_28.mp4":                [0, 0, 0, 0, 0, 0],
    "kohli_side_30.mp4":                [0, 0, 0, 0, 0, 0],
    "kohli_side_32.mp4":                [0, 0, 0, 0, 0, 0],
    "pro_player_back_21.mp4":           [0, 0, 1, 0, 1, 1],
    "pro_player_back_23.mp4":           [0, 0, 0, 0, 0, 0],
    "pro_player_back_25.mp4":           [0, 0, 0, 0, 0, 0],
    "pro_player_back_27.mp4":           [2, 2, 0, 0, 0, 1],
    "pro_player_back_29.mp4":           [0, 0, 0, 0, 0, 0],
    "pro_player_back_31.mp4":           [1, 0, 0, 1, 1, 0],
    "pro_player_back_33.mp4":           [0, 0, 0, 0, 0, 0],
    "pro_player_back_36.mp4":           [2, 1, 0, 0, 0, 0],
    "pro_player_back_38.mp4":           [0, 0, 0, 0, 0, 0],
    "pro_player_back_40.mp4":           [0, 0, 0, 1, 0, 0],
    "pro_player_front_01.mp4":          [0, 0, 1, 1, 1, 0],
    "pro_player_front_03.mp4":          [1, 0, 0, 0, 1, 0],
    "pro_player_front_06.mp4":          [1, 1, 1, 1, 1, 1],
    "pro_player_front_08.mp4":          [1, 0, 1, 1, 1, 0],
    "pro_player_front_10.mp4":          [1, 1, 1, 0, 1, 1],
    "pro_player_front_12.mp4":          [1, 1, 1, 0, 1, 1],
    "pro_player_front_14.mp4":          [1, 0, 0, 1, 1, 1],
    "pro_player_front_16.mp4":          [1, 1, 1, 1, 1, 1],
    "pro_player_front_18.mp4":          [2, 0, 1, 1, 1, 1],
    "pro_player_front_20.mp4":          [1, 0, 0, 0, 0, 0],
    "pro_player_front_42.mp4":          [0, 0, 0, 0, 0, 0],
    "pro_player_front_44.mp4":          [0, 0, 0, 0, 0, 0],
    "rishi_front_2.mp4":                [0, 0, 0, 0, MISS, 1],
    "roa_front_1.mp4":                  [0, 0, 0, 0, 0, 0],
    "sanjay_front _1.mp4":              [0, 0, 0, 0, 1, ABSENT],
}

# Clip-level observations recorded during the same pass.
SCENE_CUT = {"pro_player_front_20.mp4", "rishi_front_2.mp4", "sanjay_front _1.mp4",
             "pro_player_back_40.mp4"}
# Clips where the sampled window is a long session rather than a single shot.
NOT_A_SINGLE_SHOT = {"sanjay_front _1.mp4", "rishi_front_2.mp4", "Sai_front_1.mp4",
                     "Sai_front_3.mp4"}
# Clips where no actual batting stroke is played in the sampled frames.
NO_BATTING = {"pro_player_front_42.mp4"}


def build():
    with open(os.path.join(RESULTS, "annotation_proposals.json"), encoding="utf-8") as f:
        proposals = json.load(f)

    out, n_frames, n_absent, n_miss = [], 0, 0, 0
    for clip in proposals["clips"]:
        cid = clip["clip_id"]
        if cid not in LABELS:
            print(f"WARNING: no labels for {cid}, skipping")
            continue
        labels = LABELS[cid]
        if len(labels) != len(clip["sampled_frames"]):
            raise ValueError(f"{cid}: {len(labels)} labels vs "
                             f"{len(clip['sampled_frames'])} sampled frames")

        frames = []
        for idx, (frame_no, label) in enumerate(zip(clip["sampled_frames"], labels)):
            rec = {"frame": frame_no}
            if label == ABSENT:
                rec.update(batsman_visible=False, batsman_bbox=None, source="absent")
                n_absent += 1
            elif label == MISS:
                rec.update(batsman_visible=True, batsman_bbox=None, source="proposal_miss")
                n_miss += 1
                n_frames += 1
            else:
                boxes = clip["proposals"][idx]
                if label >= len(boxes):
                    raise ValueError(f"{cid} frame {frame_no}: label {label} but "
                                     f"only {len(boxes)} proposals")
                rec.update(batsman_visible=True, batsman_bbox=boxes[label][:4],
                           source="proposal", proposal_index=label,
                           proposal_conf=boxes[label][4])
                n_frames += 1
            frames.append(rec)

        out.append({
            "clip_id": cid,
            "split": clip["split"],
            "n_frames": clip["n_frames"],
            "fps": clip["fps"],
            "width": clip["width"],
            "height": clip["height"],
            "scene_cut": cid in SCENE_CUT,
            "single_shot": cid not in NOT_A_SINGLE_SHOT,
            "contains_batting": cid not in NO_BATTING,
            "frames": frames,
        })

    path = os.path.join(RESULTS, "phase2_groundtruth.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Batsman ground truth from proposal-assisted visual annotation. "
                     "batsman_bbox is null when source=='proposal_miss' (batsman "
                     "visible but undetected by the proposal detector) -- those frames "
                     "count in the recall denominator and no detector can score them. "
                     "Frames with batsman_visible==false are excluded entirely."),
            "proposal_detector": proposals["proposal_detector"],
            "annotator": "single annotator, not blind to proposals, no second rater",
            "clips": out,
        }, f, indent=1)

    dev = sum(1 for c in out if c["split"] == "dev")
    print(f"clips            : {len(out)}  (dev {dev}, eval {len(out) - dev})")
    print(f"GT batsman frames: {n_frames}  (of which {n_miss} are proposal misses)")
    print(f"excluded (absent): {n_absent}")
    print(f"wrote {path}")


if __name__ == "__main__":
    build()
