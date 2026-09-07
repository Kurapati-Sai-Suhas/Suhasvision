"""
phase3_shot_localization.py — find the temporal interval of ONE coherent
batting shot, given the selected batsman track.

FOUR METHODS

  L0  current baseline      the whole clip window, unchanged
  L1  global motion         frame-difference energy over the whole frame
  L2  batsman-local motion  the same measure inside the batsman box
  L3  hybrid                normalised product of the two

WHY L0 IS THE WHOLE CLIP, AND WHY THAT IS NOT A STRAW MAN
The production pipeline has no intra-clip shot localizer. `select_phase_frames`
is HANDED `start_frame`/`end_frame`; those come from an external ingestion step
(`nvidia_client.py` proposing `start_time`/`end_time` over a long session
video), and within the pipeline the supplied window is used as-is. So on a
benchmark of already-trimmed clips the current method's interval is the clip
itself. That is measured rather than replaced, per the instruction to run the
existing method exactly as it exists, and it is a real floor: the annotated
shot occupies a median 22% of its clip, so whole-clip scores IoU ~0.22.

L1/L2/L3 SHARE ONE WINDOWING RULE
Only the SIGNAL differs between them. The peak-finding, boundary expansion,
duration clamp and rejection test are identical code, so a difference in the
results is attributable to WHERE motion is measured and nothing else. This is
the same discipline the S0-S4 and H0/H1 experiments used.

NO HARD-CODED WINNER
Phase 2 tested the hypothesis "batsman-local motion beats global motion" and
did NOT confirm it. All three signals are therefore kept and scored; the
hybrid is not assumed to win either.

EVERY TUNABLE IS FIT ON DEV
`alpha` (boundary threshold), `tau` (rejection prominence) and the duration
clamp are chosen on the dev split only and applied unchanged to eval.
"""

import math

import cv2
import numpy as np

# Shared windowing defaults. These are STARTING points; the benchmark fits
# them on dev and records what it chose.
SMOOTH_WIDTH = 5
ALPHA = 0.35        # boundary at baseline + alpha*(peak - baseline)
TAU = 1.60          # reject if peak/baseline prominence is below this
MIN_DUR = 12
MAX_DUR = 45

METHODS = ("L0_whole_clip", "L1_global_motion", "L2_batsman_motion", "L3_hybrid")


def _smooth(x, w=SMOOTH_WIDTH):
    if w <= 1 or len(x) < w:
        return np.asarray(x, float)
    k = np.ones(w, float) / w
    return np.convolve(np.asarray(x, float), k, mode="same")


def motion_signals(video_path, track_by_frame=None, max_frames=400):
    """Global and batsman-local frame-difference energy for one clip.

    Both are mean absolute difference between consecutive grayscale frames;
    the ONLY difference is the region. Local energy is measured inside the
    batsman box and normalised by box area, so a large box does not score
    higher merely for containing more pixels.

    Returns (n_frames, global_signal, local_signal). `local` is None when no
    track boxes were supplied. Signals are indexed by the LATER frame of each
    pair, i.e. signal[t] describes the change from t-1 to t.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, None, None
    prev = None
    g, l = [], []
    n = 0
    while n < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            d = cv2.absdiff(gray, prev)
            g.append(float(d.mean()))
            box = (track_by_frame or {}).get(n)
            if box is None:
                l.append(np.nan)
            else:
                h, w = gray.shape
                x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
                x2 = min(w, int(box[2])); y2 = min(h, int(box[3]))
                if x2 > x1 and y2 > y1:
                    l.append(float(d[y1:y2, x1:x2].mean()))
                else:
                    l.append(np.nan)
        prev = gray
        n += 1
    cap.release()
    if not g:
        return n, None, None
    local = np.asarray(l, float) if track_by_frame else None
    return n, np.asarray(g, float), local


def _fill_gaps(sig):
    """Frames where the track is absent carry NaN. Interpolating rather than
    zero-filling matters: a zero would be read as 'no motion here', which is a
    claim the missing box does not support."""
    if sig is None:
        return None
    s = np.asarray(sig, float).copy()
    if np.all(np.isnan(s)):
        return None
    idx = np.arange(len(s))
    good = ~np.isnan(s)
    s[~good] = np.interp(idx[~good], idx[good], s[good])
    return s


def _normalise(s):
    if s is None:
        return None
    lo, hi = float(np.min(s)), float(np.max(s))
    return (s - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(s)


def build_signal(method, g, l):
    """The one place the three signals differ."""
    if method == "L1_global_motion":
        return g
    if method == "L2_batsman_motion":
        return _fill_gaps(l)
    if method == "L3_hybrid":
        lf = _fill_gaps(l)
        if g is None or lf is None:
            return None
        # Product of normalised signals: an interval must be active BOTH
        # globally and on the batsman. Interpretable and parameter-free --
        # deliberately not a weighted sum, which would add a weight to fit.
        return _normalise(g) * _normalise(lf)
    return None


def localize(signal, n_frames, alpha=ALPHA, tau=TAU,
             min_dur=MIN_DUR, max_dur=MAX_DUR, smooth=SMOOTH_WIDTH):
    """Peak-anchored interval from a motion signal.

    Returns {"start", "end", "prominence", "rejected", "reason"}. Rejection is
    a first-class outcome: a clip with no prominent peak should be declined
    rather than assigned an arbitrary window, which is what makes the
    false-shot rate on invalid clips measurable.
    """
    if signal is None or len(signal) < 5:
        return {"start": None, "end": None, "prominence": None,
                "rejected": True, "reason": "no_signal"}
    s = _smooth(signal, smooth)
    base = float(np.median(s))
    peak = float(np.max(s))
    p = int(np.argmax(s))
    prominence = (peak / base) if base > 1e-9 else float("inf")

    if not np.isfinite(prominence) or prominence < tau:
        return {"start": None, "end": None,
                "prominence": round(float(prominence), 4)
                if np.isfinite(prominence) else None,
                "rejected": True, "reason": "peak_not_prominent"}

    thr = base + alpha * (peak - base)
    a = p
    while a > 0 and s[a - 1] > thr:
        a -= 1
    b = p
    while b < len(s) - 1 and s[b + 1] > thr:
        b += 1

    # Signal index t describes motion INTO frame t+1 (it was built from pairs),
    # so shift by one to land on frame indices.
    start, end = a + 1, b + 1
    dur = end - start
    if dur < min_dur:
        grow = (min_dur - dur)
        start = max(0, start - grow // 2)
        end = min(n_frames - 1, start + min_dur)
    elif dur > max_dur:
        # Keep the peak, trim symmetrically around it.
        c = p + 1
        start = max(0, c - max_dur // 2)
        end = min(n_frames - 1, start + max_dur)
    start = max(0, min(start, n_frames - 1))
    end = max(start + 1, min(end, n_frames - 1))
    return {"start": int(start), "end": int(end),
            "prominence": round(float(prominence), 4),
            "rejected": False, "reason": ""}


def predict(method, n_frames, g, l, **kw):
    """One interval per clip, for any of the four methods."""
    if method == "L0_whole_clip":
        # The current pipeline's behaviour: the supplied window IS the shot.
        # It never rejects, which is itself a measurable property.
        return {"start": 0, "end": max(1, n_frames - 1), "prominence": None,
                "rejected": False, "reason": ""}
    return localize(build_signal(method, g, l), n_frames, **kw)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def temporal_iou(a, b):
    """IoU of two closed frame intervals."""
    if a is None or b is None:
        return 0.0
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return (inter / union) if union > 1e-9 else 0.0


def crosses(interval, frames):
    """Does the interval span any of these frame indices?"""
    if interval is None or not frames:
        return False
    return any(interval[0] <= f <= interval[1] for f in frames)
