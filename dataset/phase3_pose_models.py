"""
phase3_pose_models.py — MediaPipe and RTMPose behind one interface, mapped to
the SAME canonical joints the 15-angle feature set needs.

WHY A CANONICAL MAPPING MATTERS MORE THAN IT LOOKS
RTMPose's default COCO-17 output has no foot landmarks, and `angle_ankle_L/R`
is defined as knee -> ankle -> foot_index. Benchmarking the COCO-17 variant
would silently drop 2 of the 15 angles and make the feature-quality
comparison meaningless while still producing plausible-looking numbers. The
halpe26 variant is used instead precisely because it carries feet, so the
(7,30) representation is reconstructed identically from both models.

Both models receive the SAME track bounding box from the same cached
ByteTrack output. Each is then used the way it is designed to be used:

  MediaPipe  on the cropped region, which is what the production pipeline
             already does
  RTMPose    top-down on the full frame with the bbox, which is its native
             mode; it crops and resizes internally

That is a fair deployment comparison rather than an artificial one, but it
does mean RTMPose reads from original pixels while MediaPipe reads a crop we
made. The difference is one resampling step and is recorded here rather than
hidden.

CONFIGURATION UNDER TEST (recorded so the experiment is reproducible)
  model        rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192
  input size   192x256 (default), varied in the resolution experiment
  backend      onnxruntime
  device       CPU  -- no CUDAExecutionProvider is available in this
               environment, so RTMPose runs on CPU. MediaPipe also runs on
               CPU, so the latency comparison is CPU-vs-CPU and fair; a
               GPU RTMPose would be faster than reported here.
  keypoints    26 (halpe26)

A CONSTRAINT FOUND BY READING THE PRODUCTION FEATURE CODE, NOT ASSUMED
academic_scripts/feature_engineering.py computes all 15 angles in THREE
dimensions -- it reads `{joint}_x/_y/_z` and takes 3D dot products.
MediaPipe supplies that z (a learned depth estimate relative to the hips).
RTMPose 2D does not: it returns (x, y) only. So RTMPose is NOT a drop-in
replacement for MediaPipe even though it preserves 7 phases x 30 features:
the tensor keeps its SHAPE while the angles change MEANING from 3D to 2D
projected. The frozen scoring model was trained on 3D angles, so feeding it
2D angles would be train/serve skew, not a swap.

This is recorded because it constrains the decision regardless of how the
quality numbers come out, and it is measured rather than argued: the
`z_contribution` metric below quantifies how much MediaPipe's z actually
changes the angles, so we know whether the 3D-vs-2D difference is material
or cosmetic.

For the comparison itself, angles are computed in 2D for BOTH models. That
isolates the variable under test (landmark quality) instead of confounding it
with a dimensionality difference only one model has.
"""

import math
import os

import cv2
import numpy as np

# The joints the 15-angle feature set consumes.
CANONICAL = ("nose", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
             "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee",
             "l_ankle", "r_ankle", "l_foot_index", "r_foot_index")

MEDIAPIPE_IDX = {"nose": 0, "l_shoulder": 11, "r_shoulder": 12,
                 "l_elbow": 13, "r_elbow": 14, "l_wrist": 15, "r_wrist": 16,
                 "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26,
                 "l_ankle": 27, "r_ankle": 28, "l_foot_index": 31,
                 "r_foot_index": 32}

HALPE26_IDX = {"nose": 0, "l_shoulder": 5, "r_shoulder": 6,
               "l_elbow": 7, "r_elbow": 8, "l_wrist": 9, "r_wrist": 10,
               "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14,
               "l_ankle": 15, "r_ankle": 16, "l_foot_index": 20,
               "r_foot_index": 21}

_BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"

# The ONNX graphs have STATIC input shapes, so input resolution cannot be
# re-parameterised on a single checkpoint -- each resolution is a different
# published model. Only three halpe26 checkpoints exist, and crucially there
# is NO rtmpose-m at 384x288. So resolution and capacity CANNOT be varied
# independently with the published weights: the 384x288 model is also the
# larger 'x' backbone. That confound is named at every point it appears
# rather than being quietly presented as a resolution result.
RTMPOSE_VARIANTS = {
    "rtmpose-s@192x256": (_BASE + "rtmpose-s_simcc-body7_pt-body7-halpe26_700e-"
                          "256x192-7f134165_20230605.zip", (192, 256)),
    "rtmpose-m@192x256": (_BASE + "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-"
                          "256x192-4d3e73dd_20230605.zip", (192, 256)),
    "rtmpose-x@288x384": (_BASE + "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-"
                          "384x288-7fb6e239_20230606.zip", (288, 384)),
}
RTMPOSE_HALPE26 = RTMPOSE_VARIANTS["rtmpose-m@192x256"][0]

# Bones used for the anatomical-consistency check. A rigid body's bone
# lengths should not change frame to frame; variation is a direct measure of
# landmark noise that needs no ground truth.
BONES = (("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
         ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
         ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
         ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
         ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"))


class PoseResult:
    """Canonical joints in FRAME pixel coordinates, plus per-joint scores.

    `pts3` carries (x, y, z) when the model estimates depth (MediaPipe) and is
    empty when it does not (RTMPose 2D). It exists only to measure how much
    the z channel actually changes the production angles.
    """
    __slots__ = ("pts", "scores", "ok", "pts3")

    def __init__(self, pts=None, scores=None, pts3=None):
        self.pts = pts or {}
        self.scores = scores or {}
        self.pts3 = pts3 or {}
        self.ok = bool(pts)

    def visible(self, thresh):
        return sum(1 for j, s in self.scores.items() if s >= thresh)

    def torso(self):
        try:
            msh = ((self.pts["l_shoulder"][0] + self.pts["r_shoulder"][0]) / 2,
                   (self.pts["l_shoulder"][1] + self.pts["r_shoulder"][1]) / 2)
            mhip = ((self.pts["l_hip"][0] + self.pts["r_hip"][0]) / 2,
                    (self.pts["l_hip"][1] + self.pts["r_hip"][1]) / 2)
            return math.hypot(msh[0] - mhip[0], msh[1] - mhip[1])
        except KeyError:
            return 0.0


class MediaPipePose:
    """The current production pose model, run on the track crop."""
    name = "mediapipe_heavy"

    def __init__(self, pad=0.12):
        self.pad = pad

    def infer(self, frame, box):
        import zero_storage_pipeline as zsp
        h, w = frame.shape[:2]
        bw, bh = box[2] - box[0], box[3] - box[1]
        x1 = max(0, int(box[0] - self.pad * bw)); y1 = max(0, int(box[1] - self.pad * bh))
        x2 = min(w, int(box[2] + self.pad * bw)); y2 = min(h, int(box[3] + self.pad * bh))
        if x2 <= x1 or y2 <= y1:
            return PoseResult()
        crop = frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with zsp._DETECTOR_LOCK:
            cands = zsp._detect_candidates(zsp._get_shared_detector_locked(), [rgb])
        if not cands[0]:
            return PoseResult()
        lm = cands[0][0]
        ch, cw = crop.shape[:2]
        pts, sc, p3 = {}, {}, {}
        for name, i in MEDIAPIPE_IDX.items():
            if i >= len(lm):
                continue
            pts[name] = (x1 + lm[i].x * cw, y1 + lm[i].y * ch)
            sc[name] = float(getattr(lm[i], "visibility", 0.0))
            # z is in normalised units on roughly the same scale as x; keep it
            # in the crop's own normalised frame the way production does.
            p3[name] = (lm[i].x * cw, lm[i].y * ch, float(lm[i].z) * cw)
        return PoseResult(pts, sc, p3)


class RTMPosePose:
    """RTMPose halpe26, top-down on the full frame with the track bbox."""

    def __init__(self, variant="rtmpose-m@192x256", device="cpu"):
        from rtmlib import RTMPose
        url, input_size = RTMPOSE_VARIANTS[variant]
        self.input_size = input_size
        self.name = variant
        self.model = RTMPose(onnx_model=url, model_input_size=input_size,
                             backend="onnxruntime", device=device)

    def infer(self, frame, box):
        bbox = np.array([[box[0], box[1], box[2], box[3]]], dtype=np.float32)
        kpts, scores = self.model(frame, bboxes=bbox)
        k = np.asarray(kpts)
        s = np.asarray(scores)
        if k.size == 0:
            return PoseResult()
        k, s = k[0], s[0]
        pts, sc = {}, {}
        for name, i in HALPE26_IDX.items():
            if i >= len(k):
                continue
            pts[name] = (float(k[i][0]), float(k[i][1]))
            sc[name] = float(s[i])
        return PoseResult(pts, sc)


class _Landmark:
    """Duck-type of a MediaPipe NormalizedLandmark: .x/.y in [0,1] of the
    crop, plus .visibility."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, visibility):
        self.x = x
        self.y = y
        self.z = 0.0          # RTMPose 2D has no depth; see module docstring
        self.visibility = visibility


def to_mediapipe_landmarks(pose, box):
    """Adapt a PoseResult (absolute frame pixels) to the landmark list that
    phase3_semantic_features._pose_metrics() expects.

    _pose_metrics() indexes MediaPipe's 33-point layout and reads `.x`/`.y` as
    fractions of the CROP. Rather than fork that function for a second
    keypoint layout — which would risk the two paths drifting apart — the
    RTMPose skeleton is placed at the MediaPipe indices so the downstream
    semantic feature code is byte-for-byte the same for both backends. That is
    what makes H0 and H1 differ only in the pose model.

    Returns None when the pose is unusable, matching the `lm = ... if ... else
    None` convention at the call site.
    """
    if not (pose and pose.ok):
        return None
    x1, y1, x2, y2 = box
    w, h = (x2 - x1), (y2 - y1)
    if w <= 0 or h <= 0:
        return None
    lms = [_Landmark(0.0, 0.0, 0.0) for _ in range(33)]
    for name, idx in MEDIAPIPE_IDX.items():
        p = pose.pts.get(name)
        if p is None:
            continue
        lms[idx] = _Landmark((p[0] - x1) / w, (p[1] - y1) / h,
                             float(pose.scores.get(name, 0.0)))
    return lms


# ---------------------------------------------------------------------------
# Quality measures — none of these need ground truth
# ---------------------------------------------------------------------------

def bone_length_cv(seq):
    """Coefficient of variation of each bone length across the sequence,
    averaged. A rigid body gives ~0; landmark jitter inflates it. This is the
    same idea as the pipeline's existing kinematic_validator."""
    cvs = []
    for a, b in BONES:
        L = []
        for p in seq:
            if p and a in p.pts and b in p.pts:
                t = p.torso()
                if t > 1e-6:
                    L.append(math.hypot(p.pts[a][0] - p.pts[b][0],
                                        p.pts[a][1] - p.pts[b][1]) / t)
        if len(L) >= 3:
            m = sum(L) / len(L)
            if m > 1e-9:
                sd = math.sqrt(sum((x - m) ** 2 for x in L) / (len(L) - 1))
                cvs.append(sd / m)
    return (sum(cvs) / len(cvs)) if cvs else None


def jitter(seq):
    """Median per-joint frame-to-frame displacement in torso units. A model
    that finds more skeletons but shakes them is worse for the velocity half
    of the (7,30) tensor, so this is measured explicitly rather than assumed."""
    d = []
    for p, q in zip(seq, seq[1:]):
        if not (p and q):
            continue
        t = (p.torso() + q.torso()) / 2
        if t <= 1e-6:
            continue
        for j in CANONICAL:
            if j in p.pts and j in q.pts:
                d.append(math.hypot(q.pts[j][0] - p.pts[j][0],
                                    q.pts[j][1] - p.pts[j][1]) / t)
    if not d:
        return None
    d.sort()
    return d[len(d) // 2]


def _angle(a, b, c):
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1); n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2 + 1e-6)))
    return math.degrees(math.acos(cos))


def _angle_vertical(a, b):
    v = (b[0] - a[0], b[1] - a[1])
    n = math.hypot(*v)
    if n < 1e-9:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, v[1] / (n + 1e-6)))))


def fifteen_angles(p):
    """The production 15-angle vector, computed identically from either model.

    Mirrors academic_scripts/feature_engineering.py: 12 three-point joint
    angles plus 3 angles against the vertical.
    """
    if not p or not p.ok:
        return None
    P = p.pts
    need = lambda *ks: all(k in P for k in ks)
    out = {}
    tri = [("angle_knee_L", "l_hip", "l_knee", "l_ankle"),
           ("angle_knee_R", "r_hip", "r_knee", "r_ankle"),
           ("angle_hip_L", "l_shoulder", "l_hip", "l_knee"),
           ("angle_hip_R", "r_shoulder", "r_hip", "r_knee"),
           ("angle_elbow_L", "l_shoulder", "l_elbow", "l_wrist"),
           ("angle_elbow_R", "r_shoulder", "r_elbow", "r_wrist"),
           ("angle_shoulder_L", "l_hip", "l_shoulder", "l_elbow"),
           ("angle_shoulder_R", "r_hip", "r_shoulder", "r_elbow"),
           ("angle_ankle_L", "l_knee", "l_ankle", "l_foot_index"),
           ("angle_ankle_R", "r_knee", "r_ankle", "r_foot_index")]
    for nm, a, b, c in tri:
        out[nm] = _angle(P[a], P[b], P[c]) if need(a, b, c) else None
    ver = [("angle_trunk_L", "l_hip", "l_shoulder"),
           ("angle_trunk_R", "r_hip", "r_shoulder"),
           ("angle_arm_L", "l_shoulder", "l_elbow"),
           ("angle_arm_R", "r_shoulder", "r_elbow")]
    for nm, a, b in ver:
        out[nm] = _angle_vertical(P[a], P[b]) if need(a, b) else None
    if need("l_shoulder", "r_shoulder", "nose"):
        msh = ((P["l_shoulder"][0] + P["r_shoulder"][0]) / 2,
               (P["l_shoulder"][1] + P["r_shoulder"][1]) / 2)
        out["angle_head_tilt"] = _angle_vertical(msh, P["nose"])
    else:
        out["angle_head_tilt"] = None
    return out


def _angle3(a, b, c):
    v1 = tuple(a[i] - b[i] for i in range(3))
    v2 = tuple(c[i] - b[i] for i in range(3))
    n1 = math.sqrt(sum(x * x for x in v1)); n2 = math.sqrt(sum(x * x for x in v2))
    dot = sum(v1[i] * v2[i] for i in range(3))
    cos = max(-1.0, min(1.0, dot / (n1 * n2 + 1e-6)))
    return math.degrees(math.acos(cos))


def z_contribution(seq):
    """How many degrees MediaPipe's z channel actually moves the production
    angles, versus computing the same angles in 2D.

    This decides whether losing z to RTMPose is material or cosmetic, and it
    is only computable for a model that estimates depth.
    """
    tri = [("l_hip", "l_knee", "l_ankle"), ("r_hip", "r_knee", "r_ankle"),
           ("l_shoulder", "l_hip", "l_knee"), ("r_shoulder", "r_hip", "r_knee"),
           ("l_shoulder", "l_elbow", "l_wrist"), ("r_shoulder", "r_elbow", "r_wrist"),
           ("l_hip", "l_shoulder", "l_elbow"), ("r_hip", "r_shoulder", "r_elbow"),
           ("l_knee", "l_ankle", "l_foot_index"), ("r_knee", "r_ankle", "r_foot_index")]
    diffs = []
    for p in seq:
        if not (p and p.pts3):
            continue
        for a, b, c in tri:
            if a in p.pts3 and b in p.pts3 and c in p.pts3:
                d3 = _angle3(p.pts3[a], p.pts3[b], p.pts3[c])
                d2 = _angle(p.pts3[a][:2], p.pts3[b][:2], p.pts3[c][:2])
                if d2 is not None:
                    diffs.append(abs(d3 - d2))
    if not diffs:
        return None
    diffs.sort()
    return {"median_deg": round(diffs[len(diffs) // 2], 2),
            "p90_deg": round(diffs[int(0.9 * len(diffs))], 2),
            "max_deg": round(diffs[-1], 2), "n": len(diffs)}


def feature_quality(seq):
    """Downstream feature validity and smoothness over a sequence.

    `invalid_rate` counts angles that could not be computed at all.
    `implausible_rate` counts anatomically impossible values (a human knee or
    elbow does not read 0 or 180 degrees in a batting stroke).
    `velocity_discontinuity` counts frame-to-frame angle jumps above 60
    degrees, which the velocity half of the tensor would encode as fiction.
    """
    angs = [fifteen_angles(p) for p in seq]
    have = [a for a in angs if a]
    if not have:
        return {"n_frames_with_angles": 0, "invalid_rate": None,
                "implausible_rate": None, "velocity_discontinuity": None,
                "angle_smoothness": None}

    total = miss = implausible = 0
    for a in have:
        for k, v in a.items():
            total += 1
            if v is None:
                miss += 1
            elif k.startswith(("angle_knee", "angle_elbow")) and (v < 15 or v > 179):
                implausible += 1

    jumps = big = 0
    accel = []
    for a, b in zip(angs, angs[1:]):
        if not (a and b):
            continue
        for k in a:
            if a[k] is not None and b[k] is not None:
                d = abs(b[k] - a[k])
                jumps += 1
                if d > 60:
                    big += 1
    series = {}
    for a in angs:
        if a:
            for k, v in a.items():
                if v is not None:
                    series.setdefault(k, []).append(v)
    for k, s in series.items():
        if len(s) >= 3:
            accel += [abs(c - 2 * b + a) for a, b, c in zip(s, s[1:], s[2:])]

    return {
        "n_frames_with_angles": len(have),
        "invalid_rate": round(miss / total, 4) if total else None,
        "implausible_rate": round(implausible / total, 4) if total else None,
        "velocity_discontinuity": round(big / jumps, 4) if jumps else None,
        "angle_smoothness": round(float(np.median(accel)), 3) if accel else None,
    }
