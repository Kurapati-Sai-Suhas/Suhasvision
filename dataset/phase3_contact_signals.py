"""
phase3_contact_signals.py — candidate contact signals C0-C6.

CONTACT IS NOT THE SHOT BOUNDARY
A contact estimate is an ANCHOR INSIDE the event, not its start or end. The
shot interval is the complete coherent batting event; the contact region is
the impact-adjacent moment within it. They are produced and evaluated
separately here, and the Best-7 stage uses contact as an internal anchor
rather than as a crop.

SIGNALS
  C0  current contact proposal      the existing machinery, measured as-is
  C1  wrist / hand velocity         fastest wrist displacement, torso units
  C2  bat evidence + association    bat detected ON this person, near a wrist
  C3  skeleton dynamics             downswing: rate of fall of wrist lift
  C4  global motion                 whole-frame difference energy
  C5  batsman-local motion          the same inside the batsman box
  C6  combined event score          equal-weight mean of normalised C1-C5

WHY C6 IS AN EQUAL-WEIGHT MEAN AND NOT A FITTED SUM
A weighted sum would introduce five free parameters fitted on 14 dev clips,
which this benchmark cannot support without overfitting. An equal-weight mean
of per-clip min-max normalised signals has NO free parameters, so its score is
attributable to the signals agreeing rather than to a fit. If C6 wins, it wins
because independent evidence concurs.

Every signal is computed from inference-time information only. No ground truth
is read.
"""

import numpy as np

SIGNALS = ("C0_current_proposal", "C1_wrist_velocity", "C2_bat_association",
           "C3_skeleton_dynamics", "C4_global_motion", "C5_local_motion",
           "C6_combined_event")

SMOOTH = 3


def _smooth(x, w=SMOOTH):
    """Moving average with EDGE padding, not zero padding.

    np.convolve(..., mode="same") pads with zeros, so a smoothed signal decays
    towards both ends of every clip. For a derivative-based signal like
    skeleton_dynamics that manufactures a spurious descent in the final frames
    of every clip — a systematic bias toward late contact estimates. Replicating
    the edge values keeps the ends flat instead.
    """
    x = np.asarray(x, float)
    if w <= 1 or len(x) < w:
        return x
    pad = w // 2
    padded = np.pad(x, pad, mode="edge")
    return np.convolve(padded, np.ones(w) / w, mode="valid")[:len(x)]


def _nan_series(frames, fn):
    return np.array([fn(f) if f else np.nan for f in frames], float)


def _fill(s):
    """Interpolate gaps rather than zero-filling: a missing box is an absence
    of evidence, not evidence of stillness."""
    s = np.asarray(s, float).copy()
    if np.all(np.isnan(s)):
        return None
    idx = np.arange(len(s))
    good = ~np.isnan(s)
    s[~good] = np.interp(idx[~good], idx[good], s[good])
    return s


def _norm(s):
    if s is None:
        return None
    lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    return (s - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(s)


def wrist_velocity(frames):
    """Per-frame wrist displacement in torso units, taking the faster hand.

    Torso normalisation matters: without it a batsman nearer the camera scores
    higher for the same stroke.
    """
    out = np.full(len(frames), np.nan)
    prev = None
    for i, f in enumerate(frames):
        p = f.get("pose")
        if not p:
            prev = None
            continue
        if prev is not None:
            t = max(p["torso"], 1e-6)
            dl = np.hypot(p["lwr"][0] - prev["lwr"][0], p["lwr"][1] - prev["lwr"][1])
            dr = np.hypot(p["rwr"][0] - prev["rwr"][0], p["rwr"][1] - prev["rwr"][1])
            out[i] = max(dl, dr) / t
        prev = p
    return _fill(out)


def bat_association(frames):
    """Bat detected ON this person and close to a wrist.

    Phase 3 established that "a bat is in frame" separates batsman from
    non-batsman by 0.020 while "the bat is on THIS person" separates by 0.355.
    The same logic applies temporally: an unassociated bat detection says
    nothing about when contact happened.
    """
    out = np.zeros(len(frames))
    for i, f in enumerate(frames):
        b, p = f.get("bat"), f.get("pose")
        if not b:
            continue
        score = b["conf"] * b["on_person_frac"]
        if p:
            t = max(p["torso"], 1e-6)
            d = min(np.hypot(b["cx"] - p["lwr"][0], b["cy"] - p["lwr"][1]),
                    np.hypot(b["cx"] - p["rwr"][0], b["cy"] - p["rwr"][1])) / t
            # Closer to a hand counts for more, decaying over one torso length.
            score *= float(np.exp(-max(0.0, d - 0.5)))
        out[i] = score
    return out


def skeleton_dynamics(frames):
    """Downswing rate: how fast the hands are DROPPING relative to the
    shoulder line. Contact in a cricket stroke is impact-adjacent to the
    steepest descent of the hands, not merely to fast hands."""
    lift = _fill(_nan_series(frames, lambda f: (f["pose"] or {}).get("wrist_lift")))
    if lift is None:
        return None
    d = np.gradient(_smooth(lift))
    return np.maximum(0.0, -d)          # descent only


def global_motion(frames):
    return _fill(_nan_series(frames, lambda f: f.get("global_motion")))


def local_motion(frames):
    return _fill(_nan_series(frames, lambda f: f.get("local_motion")))


def combined_event(frames):
    """Equal-weight mean of the normalised evidence signals. No free
    parameters, so a win here means the signals agree."""
    parts = [wrist_velocity(frames), bat_association(frames),
             skeleton_dynamics(frames), global_motion(frames),
             local_motion(frames)]
    parts = [_norm(p) for p in parts if p is not None]
    if not parts:
        return None
    return np.mean(np.vstack(parts), axis=0)


def signal_for(name, frames):
    if name == "C1_wrist_velocity":
        return wrist_velocity(frames)
    if name == "C2_bat_association":
        return bat_association(frames)
    if name == "C3_skeleton_dynamics":
        return skeleton_dynamics(frames)
    if name == "C4_global_motion":
        return global_motion(frames)
    if name == "C5_local_motion":
        return local_motion(frames)
    if name == "C6_combined_event":
        return combined_event(frames)
    return None


def estimate(name, frames, restrict=None):
    """Contact frame estimate: the argmax of the signal.

    `restrict` optionally limits the search to a (start, end) soft prior. It is
    an option, not the default, because the true event may fall outside a
    predicted interval — hard-cropping to L3 is exactly what this stage was
    told not to do.
    """
    s = signal_for(name, frames)
    if s is None or len(s) == 0:
        return {"frame": None, "confidence": None, "reason": "no_signal"}
    s = _smooth(s)
    lo, hi = 0, len(s)
    if restrict:
        lo = max(0, int(restrict[0]))
        hi = min(len(s), int(restrict[1]) + 1)
        if hi - lo < 3:
            lo, hi = 0, len(s)
    seg = s[lo:hi]
    if not np.any(np.isfinite(seg)):
        return {"frame": None, "confidence": None, "reason": "no_finite_values"}
    k = int(np.nanargmax(seg)) + lo
    base = float(np.nanmedian(s))
    peak = float(s[k])
    prom = (peak / base) if base > 1e-9 else None
    return {"frame": k,
            "confidence": round(float(prom), 4) if prom is not None else None,
            "reason": ""}
