"""
phase3_hybrid_latency.py — what does the fallback actually cost?

The standalone RTMPose figure is the wrong number. The fallback fires only
where MediaPipe returns no pose, so the cost that matters is the MARGINAL
cost per clip:

    marginal = (frames MediaPipe refused) x (one RTMPose call)

H0 and H1 are timed in the SAME warm process over the SAME candidate tracks
and the same crops. Timing them in separate runs is what produced a bogus
6x result earlier in this project; interleaving is the fix.

Both models run on CPU here (no CUDAExecutionProvider is available), and
MediaPipe is CPU too, so the comparison is like-for-like.
"""

import argparse
import json
import os
import statistics
import time

import cv2

import phase3_pose_models as PM

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(_MODULE_DIR, "phase2_results")
P3 = os.path.join(_MODULE_DIR, "phase3_results")

WARMUP = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(_MODULE_DIR, "raw_videos"))
    ap.add_argument("--tracks", default=os.path.join(P2, "tracks_yolo11m@640_bytetrack.json"))
    ap.add_argument("--variant", default="rtmpose-s@192x256")
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(P3, "phase3_hybrid_latency.json"))
    args = ap.parse_args()

    from phase2_cache_tracks import load_tracks
    from phase3_semantic_features import sample_frames

    tracks = load_tracks(args.tracks)
    mp = PM.MediaPipePose()
    rtm = PM.RTMPosePose(variant=args.variant)

    # Collect real (frame, box) work items from the ACTUAL candidate tracks,
    # which is what the pipeline pays for -- not just the batsman track.
    items = []
    per_clip = {}
    for cid in sorted(tracks["clips"])[:args.clips]:
        c = tracks["clips"][cid]
        path = os.path.join(args.videos, cid)
        if not os.path.exists(path):
            continue
        nf = int(c.get("frames_processed") or 0)
        idxs = set(sample_frames(nf))
        cap = cv2.VideoCapture(path)
        grabbed, i = {}, -1
        while i < (max(idxs) if idxs else -1):
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            if i in idxs:
                grabbed[i] = fr
        cap.release()
        n = 0
        for tid, obs in c["tracks"].items():
            by_frame = {f: b for f, b, _ in obs}
            for f in sorted(idxs):
                box, fr = by_frame.get(f), grabbed.get(f)
                if box is None or fr is None:
                    continue
                items.append((fr, box))
                n += 1
        per_clip[cid] = n
    print(f"{len(items)} (frame, box) work items from {len(per_clip)} clips")

    for fr, box in items[:WARMUP]:
        mp.infer(fr, box)
        rtm.infer(fr, box)

    mp_t, rtm_t, n_refused = [], [], 0
    for fr, box in items:
        t0 = time.perf_counter()
        r = mp.infer(fr, box)
        mp_t.append(time.perf_counter() - t0)
        if not r.ok:
            n_refused += 1
            t0 = time.perf_counter()
            rtm.infer(fr, box)
            rtm_t.append(time.perf_counter() - t0)

    def stat(ts):
        if not ts:
            return None
        return {"mean_ms": round(1000 * statistics.mean(ts), 3),
                "median_ms": round(1000 * statistics.median(ts), 3),
                "total_s": round(sum(ts), 3), "n": len(ts)}

    n_clips = len(per_clip)
    h0_total = sum(mp_t)
    h1_total = h0_total + sum(rtm_t)
    out = {
        "variant": args.variant,
        "device": "CPU (both models; no CUDAExecutionProvider available)",
        "method": "H0 and H1 timed interleaved in one warm process on identical crops",
        "n_work_items": len(items),
        "n_clips": n_clips,
        "mediapipe_per_call": stat(mp_t),
        "rtmpose_fallback_per_call": stat(rtm_t),
        "fallback_calls": n_refused,
        "fallback_rate": round(n_refused / len(items), 4) if items else None,
        "H0_total_s": round(h0_total, 3),
        "H1_total_s": round(h1_total, 3),
        "H0_per_clip_ms": round(1000 * h0_total / n_clips, 1) if n_clips else None,
        "H1_per_clip_ms": round(1000 * h1_total / n_clips, 1) if n_clips else None,
        "marginal_cost_per_clip_ms": round(1000 * sum(rtm_t) / n_clips, 1) if n_clips else None,
        "marginal_overhead_pct": round(100 * sum(rtm_t) / h0_total, 1) if h0_total else None,
        "fallback_calls_per_clip": round(n_refused / n_clips, 2) if n_clips else None,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(f"\nMediaPipe per call   : {out['mediapipe_per_call']['median_ms']:.2f} ms median")
    if out["rtmpose_fallback_per_call"]:
        print(f"RTMPose per call     : {out['rtmpose_fallback_per_call']['median_ms']:.2f} ms median")
    print(f"fallback fired       : {n_refused}/{len(items)} calls "
          f"({out['fallback_rate']:.1%}), {out['fallback_calls_per_clip']} per clip")
    print(f"H0 per clip          : {out['H0_per_clip_ms']:.1f} ms")
    print(f"H1 per clip          : {out['H1_per_clip_ms']:.1f} ms")
    print(f"marginal per clip    : {out['marginal_cost_per_clip_ms']:.1f} ms "
          f"(+{out['marginal_overhead_pct']:.1f}%)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
