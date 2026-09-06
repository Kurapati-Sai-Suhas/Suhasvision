"""
run_extraction_experiment.py — run the SAME clips through A0/A1/A2/A3 and
report per-configuration metrics.

Design constraints that make the comparison trustworthy:
  * every configuration sees the identical clip list and identical shot
    windows, so nothing can be attributed to a different subset;
  * each configuration writes to its own diagnostics file;
  * per-clip rows are kept, not just aggregates, so paired comparisons are
    possible and outliers stay visible;
  * the scoring model is never loaded — this measures extraction only.

Shot windows: `--window-mode full` treats the whole clip as the shot
(deterministic, free, no API calls) which is the right default for measuring
DOWNSTREAM stages in isolation. `--window-mode nvidia` calls the real shot
detector and costs credits. The default is `full` precisely so that shot
localization error does not silently contaminate a comparison of the stages
after it.

USAGE
    python run_extraction_experiment.py --videos raw_videos --limit 20
    python run_extraction_experiment.py --videos raw_videos --configs A0 A3
"""

import argparse
import glob
import json
import os
import time

import cv2


def _probe(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_frames < 10 or fps <= 0:
        return None
    return fps, n_frames


def run_config(video_paths, config_name, out_dir, window_mode="full"):
    """Run one configuration over every clip. Returns the diagnostics rows."""
    import zero_storage_pipeline as zsp
    from extraction_config import get_config
    import extraction_diagnostics as diagnostics

    config = get_config(config_name)
    diag_path = os.path.join(out_dir, f"diagnostics_{config_name}.jsonl")
    if os.path.exists(diag_path):
        os.remove(diag_path)

    for video_path in video_paths:
        probe = _probe(video_path)
        if probe is None:
            continue
        fps, n_frames = probe
        video_id = os.path.basename(video_path)

        if window_mode == "full":
            start_frame, end_frame = 0, n_frames - 1
        else:
            raise NotImplementedError("nvidia window mode: call find_shot_windows_auto explicitly")

        cap = cv2.VideoCapture(video_path)
        diag = diagnostics.ExtractionDiagnostics(
            session_name=os.path.splitext(video_id)[0], video_id=video_id,
            config_name=config.name, shot_start_frame=start_frame,
            shot_end_frame=end_frame, shot_duration_frames=end_frame - start_frame,
            fps=fps)

        t0 = time.time()
        try:
            indices, method = zsp.select_phase_frames(
                cap, start_frame, end_frame, diag.session_name, config, diag=diag)
            diagnostics.record_frames(diag, indices, method)
            diag.fallback_used = (method == "uniform" and config.name != "A0")

            frames_rgb = zsp.collect_phase_frames(cap, indices, diag.session_name)
            ok, result, interpolated = zsp.extract_features_from_image_array(
                frames_rgb, diag.session_name, diag=diag)
            diag.accepted = bool(ok)
            if ok:
                # Smoothness / max-jump over the sequence the pipeline really
                # produced. Stored in `extra` so the diagnostics dataclass
                # stays a record of the pipeline rather than of the benchmark.
                import extraction_benchmark as _bench
                diag.extra.update(_bench.sequence_quality(result))
            else:
                diag.rejection_reason = str(result)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not end the run
            diag.accepted = False
            diag.rejection_reason = f"exception: {type(exc).__name__}: {exc}"
        finally:
            cap.release()

        diag.elapsed_seconds = round(time.time() - t0, 3)
        diagnostics.write(diag, path=diag_path)

    return diagnostics.load(diag_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="raw_videos", help="directory of .mp4 clips")
    ap.add_argument("--configs", nargs="+", default=["A0", "A1", "A2", "A3"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    ap.add_argument("--out", default="experiment_results")
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--min-size-kb", type=int, default=100,
                    help="skip truncated/empty files")
    args = ap.parse_args()

    import extraction_benchmark as bench

    os.makedirs(args.out, exist_ok=True)
    paths = sorted(p for p in glob.glob(os.path.join(args.videos, "*.mp4"))
                   if os.path.getsize(p) >= args.min_size_kb * 1024)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"No usable clips found in {args.videos}")
        return

    annotations = bench.load_annotations(args.annotations) if args.annotations else []
    print(f"Clips: {len(paths)} | Configs: {args.configs} | Annotations: {len(annotations)}")

    summary = {}
    for name in args.configs:
        print(f"\n=== {name} ===")
        t0 = time.time()
        rows = run_config(paths, name, args.out)
        elapsed = time.time() - t0
        report = bench.summarize(rows, annotations)
        report["wall_seconds"] = round(elapsed, 1)
        report["mean_seconds_per_clip"] = round(elapsed / max(len(paths), 1), 2)
        summary[name] = report

        g = report["gt_free"]
        print(f"  accepted {g['acceptance_rate']:.1%} | methods {g['sampling_methods']}")
        print(f"  pose_success {g['mean_pose_success_rate']} | "
              f"coarse/fine subject agreement {g['coarse_fine_subject_agreement']}")
        print(f"  contact_valid_rate {g['contact_valid_rate']} | "
              f"mean_contact_conf {g['mean_contact_confidence']}")
        print(f"  {report['wall_seconds']}s total, {report['mean_seconds_per_clip']}s/clip")

    # Frame-selection divergence: how often does each config actually choose
    # different frames from A0? A config that changes nothing cannot help or
    # hurt, and that is worth knowing before interpreting any other metric.
    baseline = {r["video_id"]: r.get("selected_frames") for r in
                __import__("extraction_diagnostics").load(os.path.join(args.out, "diagnostics_A0.jsonl"))}
    if baseline:
        for name in args.configs:
            if name == "A0":
                continue
            rows = __import__("extraction_diagnostics").load(
                os.path.join(args.out, f"diagnostics_{name}.jsonl"))
            comparable = [r for r in rows if r["video_id"] in baseline]
            differ = [r for r in comparable if r.get("selected_frames") != baseline[r["video_id"]]]
            if comparable:
                summary[name]["frame_divergence_vs_A0"] = round(len(differ) / len(comparable), 4)

    out_path = os.path.join(args.out, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
