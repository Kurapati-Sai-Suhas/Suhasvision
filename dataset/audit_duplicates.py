"""
audit_duplicates.py — read-only duplicate/integrity report for keypoints.csv
(Milestone 2 of STABILIZATION_PLAN.md, follow-through on audit H1).

WHY THIS EXISTS
Ingestion was append-only with deterministic session names until Milestone 2
added the already-ingested skip, so any historical re-run silently appended a
second copy of every session it re-processed. 42 real sessions with duplicated
frame_name rows were found this way during Milestone 3 (they corrupted the
symmetry prior and jerk calculation until review caught it). This tool
measures how much of the CURRENT file is affected, in every duplicate shape
that matters downstream.

READ-ONLY BY DESIGN: it never writes or modifies anything. Cleaning the file
is a separate, human-approved step.

USAGE
    python audit_duplicates.py [--input path/to/keypoints.csv]

Exit code 0 if the file is clean, 1 if any issue was found (so it can gate a
script), 2 if the file couldn't be read.
"""

import argparse
import os
import sys

import pandas as pd

from schema import SEQ_LEN

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(_MODULE_DIR, "keypoints.csv")

# How many affected sessions to name per bucket before eliding — the point of
# the console report is a readable overview, not an exhaustive dump.
MAX_LISTED = 20


def analyze_keypoints(df):
    """
    Pure analysis, no I/O — returns a report dict. A session can appear in
    more than one bucket (e.g. a doubled-up session is both overlong AND has
    duplicated frame_names).

    Buckets:
      duplicated_frame_names: {session: {frame_name: count}} where the same
        frame_name appears more than once within a session — the exact
        corruption class behind the 42-session incident.
      exact_duplicate_row_count / exact_duplicate_row_sessions: full-row
        duplicates (every column identical), the signature of a straight
        re-ingestion append.
      overlong_sessions / incomplete_sessions: {session: row_count} for
        sessions with more / fewer than SEQ_LEN rows.
    """
    report = {
        "total_rows": len(df),
        "unique_sessions": int(df["session_name"].nunique()),
        "expected_rows_per_session": SEQ_LEN,
    }

    counts = df.groupby("session_name").size()
    report["overlong_sessions"] = {s: int(c) for s, c in counts[counts > SEQ_LEN].items()}
    report["incomplete_sessions"] = {s: int(c) for s, c in counts[counts < SEQ_LEN].items()}

    frame_counts = df.groupby(["session_name", "frame_name"]).size()
    dup_frames = frame_counts[frame_counts > 1]
    duplicated_frame_names = {}
    for (session, frame), c in dup_frames.items():
        duplicated_frame_names.setdefault(session, {})[frame] = int(c)
    report["duplicated_frame_names"] = duplicated_frame_names

    dup_row_mask = df.duplicated()
    report["exact_duplicate_row_count"] = int(dup_row_mask.sum())
    report["exact_duplicate_row_sessions"] = sorted(df.loc[dup_row_mask, "session_name"].unique())

    return report


def report_is_clean(report):
    return not (
        report["duplicated_frame_names"]
        or report["exact_duplicate_row_count"]
        or report["overlong_sessions"]
        or report["incomplete_sessions"]
    )


def _list_sessions(names):
    shown = list(names)[:MAX_LISTED]
    lines = [f"    - {name}" for name in shown]
    if len(names) > MAX_LISTED:
        lines.append(f"    ... and {len(names) - MAX_LISTED} more")
    return "\n".join(lines)


def print_report(report, source_path):
    print("=" * 60)
    print(" keypoints.csv duplicate/integrity audit (READ-ONLY)")
    print("=" * 60)
    print(f"File: {source_path}")
    print(f"Total rows: {report['total_rows']} | Unique sessions: {report['unique_sessions']} "
          f"| Expected rows per session: {report['expected_rows_per_session']}")
    print()

    dup_frames = report["duplicated_frame_names"]
    print(f"[1] Sessions with duplicated frame_name rows: {len(dup_frames)}")
    if dup_frames:
        shown = sorted(dup_frames)[:MAX_LISTED]
        for session in shown:
            frames = ", ".join(f"{fr} x{c}" for fr, c in sorted(dup_frames[session].items()))
            print(f"    - {session}: {frames}")
        if len(dup_frames) > MAX_LISTED:
            print(f"    ... and {len(dup_frames) - MAX_LISTED} more")
    print()

    n_dup_rows = report["exact_duplicate_row_count"]
    dup_row_sessions = report["exact_duplicate_row_sessions"]
    print(f"[2] Exact full-row duplicates: {n_dup_rows} row(s) across {len(dup_row_sessions)} session(s)")
    if dup_row_sessions:
        print(_list_sessions(dup_row_sessions))
    print()

    overlong = report["overlong_sessions"]
    print(f"[3] Overlong sessions (> {report['expected_rows_per_session']} rows): {len(overlong)}")
    if overlong:
        print(_list_sessions([f"{s} ({c} rows)" for s, c in sorted(overlong.items())]))
    print()

    incomplete = report["incomplete_sessions"]
    print(f"[4] Incomplete sessions (< {report['expected_rows_per_session']} rows): {len(incomplete)}")
    if incomplete:
        print(_list_sessions([f"{s} ({c} rows)" for s, c in sorted(incomplete.items())]))
    print()

    # ASCII-only result lines: this report is read on consoles whose codepage
    # may not render em-dashes/emoji (they show as mojibake).
    if report_is_clean(report):
        print("RESULT: CLEAN - no duplicate or malformed sessions found.")
    else:
        n_issues = (len(dup_frames) + len(overlong) + len(incomplete)
                    + (1 if n_dup_rows else 0))
        print(f"RESULT: {n_issues} affected session entries across the buckets above.")
        print("This tool modifies NOTHING. Cleaning the file is a separate, deliberate step:")
        print("fix the producer first (idempotent ingestion landed in Milestone 2), then")
        print("de-duplicate with an explicit, reviewed script.")


def main():
    ap = argparse.ArgumentParser(description="Read-only duplicate/integrity audit of keypoints.csv")
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help=f"Path to the keypoints CSV to audit (default: {DEFAULT_INPUT})")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"File not found: {args.input}")
        return 2
    except Exception as e:
        print(f"Could not read {args.input}: {e}")
        return 2

    if "session_name" not in df.columns or "frame_name" not in df.columns:
        print(f"{args.input} is missing session_name/frame_name columns — is this really a keypoints CSV?")
        return 2

    report = analyze_keypoints(df)
    print_report(report, args.input)
    return 0 if report_is_clean(report) else 1


if __name__ == "__main__":
    sys.exit(main())
