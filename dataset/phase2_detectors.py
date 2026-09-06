"""
phase2_detectors.py — a uniform person-detector interface so MediaPipe, YOLO
and RTMDet can be benchmarked against each other on identical frames.

WHY THIS EXISTS
Phase 1 established that the pipeline's candidate generation is the binding
constraint: on 70/102 clips MediaPipe's PoseLandmarker surfaced exactly one
person, so subject selection had nothing to choose between and simply
ratified whoever was detected. On the documented failure clip
(pro_player_front_09) MediaPipe returns ONE person -- the near-camera bowler
-- while the batsman stands 108px tall at the far stumps.

Everything here returns the same thing: a list of Detection boxes in PIXEL
coordinates for one BGR frame. No tracking, no selection, no pose. Keeping
detection separable is what makes "did the detector even see the batsman?"
answerable independently of every downstream stage.

MediaPipe is wrapped as a detector (pose landmarks -> enclosing box) so the
CURRENT pipeline is a first-class benchmark arm rather than an assumed
baseline. Its boxes are derived from body landmarks only (indices 11-32);
face landmarks are excluded because BlazePose extrapolates them well past
the silhouette on side-on subjects, which would inflate the box.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Body landmarks only (shoulders down). See module docstring.
_BODY_LANDMARKS = list(range(11, 33))


@dataclass
class Detection:
    """One detected person in pixel coordinates. `extra` carries
    detector-specific payload (e.g. MediaPipe's landmarks) so a detector that
    already produced pose does not have to recompute it downstream."""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    extra: Optional[dict] = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)


def iou(a: Detection, b) -> float:
    """Intersection over union. `b` may be a Detection or an (x1,y1,x2,y2)
    tuple, so ground-truth boxes stored as plain lists work directly."""
    bx = b.as_xyxy() if isinstance(b, Detection) else tuple(b)
    ix1, iy1 = max(a.x1, bx[0]), max(a.y1, bx[1])
    ix2, iy2 = min(a.x2, bx[2]), min(a.y2, bx[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_b = max(0.0, bx[2] - bx[0]) * max(0.0, bx[3] - bx[1])
    union = a.area + area_b - inter
    return inter / union if union > 0 else 0.0


class PersonDetector:
    """Interface. `name` must uniquely identify the detector AND its
    configuration, because resolution is part of what is being benchmarked."""
    name = "base"

    def detect(self, frame_bgr) -> List[Detection]:
        raise NotImplementedError

    def detect_batch(self, frames_bgr) -> List[List[Detection]]:
        return [self.detect(f) if f is not None else [] for f in frames_bgr]

    def close(self):
        pass


class MediaPipeDetector(PersonDetector):
    """The CURRENT pipeline's candidate generator, wrapped as a detector.

    Deliberately reuses zero_storage_pipeline's shared PoseLandmarker rather
    than building its own, so the benchmark measures the detector the
    pipeline actually runs -- same model file, same num_poses, same options.
    """

    def __init__(self, max_poses: Optional[int] = None):
        self.name = "mediapipe_heavy"
        self._max_poses = max_poses

    def detect(self, frame_bgr) -> List[Detection]:
        import cv2
        import zero_storage_pipeline as zsp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with zsp._DETECTOR_LOCK:
            candidates = zsp._detect_candidates(zsp._get_shared_detector_locked(), [rgb])
        h, w = frame_bgr.shape[:2]

        out = []
        for pose in candidates[0]:
            xs = [pose[i].x * w for i in _BODY_LANDMARKS if i < len(pose)]
            ys = [pose[i].y * h for i in _BODY_LANDMARKS if i < len(pose)]
            if not xs:
                continue
            # MediaPipe has no per-person detection score; the mean landmark
            # visibility is the closest honest analogue. It is NOT calibrated
            # against YOLO's objectness and must not be compared to it as if
            # it were -- it is recorded so the field is never silently empty.
            vis = [getattr(pose[i], "visibility", 0.0) for i in _BODY_LANDMARKS if i < len(pose)]
            out.append(Detection(min(xs), min(ys), max(xs), max(ys),
                                 float(np.mean(vis)) if vis else 0.0,
                                 extra={"landmarks": pose,
                                        "conf_is_visibility": True}))
        return out


class YoloDetector(PersonDetector):
    """Ultralytics YOLO restricted to the person class.

    conf is deliberately LOW by default. This benchmark's primary metric is
    batsman candidate recall: a distant batsman that scores 0.15 is a
    recovered candidate for downstream scoring to reason about, whereas a
    default 0.25 threshold discards it before selection ever runs. False
    positives are measured separately rather than suppressed up front.
    """

    def __init__(self, weights="yolo11m.pt", imgsz=960, conf=0.10, device=0):
        from ultralytics import YOLO
        self.weights, self.imgsz, self.conf = weights, imgsz, conf
        self.device = device
        self.name = f"{weights.replace('.pt', '')}@{imgsz}"
        self.model = YOLO(weights)
        self.model.to(device if isinstance(device, str) else f"cuda:{device}"
                      if _cuda_available() else "cpu")

    def detect(self, frame_bgr) -> List[Detection]:
        return self.detect_batch([frame_bgr])[0]

    def detect_batch(self, frames_bgr) -> List[List[Detection]]:
        present = [(i, f) for i, f in enumerate(frames_bgr) if f is not None]
        results = [[] for _ in frames_bgr]
        if not present:
            return results
        preds = self.model.predict(
            [f for _, f in present], imgsz=self.imgsz, classes=[0],
            conf=self.conf, verbose=False, device=self.model.device)
        for (idx, _), r in zip(present, preds):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            confs = b.conf.cpu().numpy()
            results[idx] = [Detection(float(x1), float(y1), float(x2), float(y2), float(c))
                            for (x1, y1, x2, y2), c in zip(xyxy, confs)]
        return results


class YoloxDetector(PersonDetector):
    """YOLOX-HumanArt via rtmlib/ONNXRuntime — the MMPose family's detector.

    WHY YOLOX AND NOT RTMDet: the Phase-2 shortlist named RTMDet, but
    rtmlib does not publish an RTMDet PERSON model — its only RTMDet ONNX is
    for hands, and `rtmdet-nano-person.zip` 404s on both the openmmlab and
    HuggingFace mirrors. The genuine mmdet route needs mmcv, which has no
    prebuilt wheel for torch 2.11 + cu128 on Windows and would have to be
    compiled. YOLOX-HumanArt is what MMPose's own top-down RTMPose pipelines
    actually ship as the person detector, so it is the faithful
    stand-in for "the MMPose-ecosystem detector" and a genuinely different
    architecture from YOLO11.

    RUNTIME IS NOT COMPARABLE with the YOLO numbers: onnxruntime here is the
    CPU build (no CUDAExecutionProvider available), while YOLO runs on CUDA.
    Recall is comparable; seconds-per-frame is not, and is labelled as such.
    """

    _MODELS = {
        "tiny": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                 "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"),
        "m": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
              "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"),
        "x": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
              "yolox_x_8xb8-300e_humanart-a39d44ed.zip"),
    }

    def __init__(self, imgsz=640, device=None, variant="m"):
        from rtmlib import YOLOX
        import onnxruntime as ort

        self.cuda = any("CUDA" in p for p in ort.get_available_providers())
        self.runtime_comparable = self.cuda
        self.name = f"yolox-{variant}@{imgsz}"
        self.imgsz = imgsz
        self.model = YOLOX(onnx_model=self._MODELS[variant],
                           model_input_size=(imgsz, imgsz),
                           backend="onnxruntime",
                           device=device or ("cuda" if self.cuda else "cpu"))

    def detect(self, frame_bgr) -> List[Detection]:
        boxes = self.model(frame_bgr)
        out = []
        for b in boxes:
            vals = [float(v) for v in list(b)]
            x1, y1, x2, y2 = vals[:4]
            # rtmlib's detector returns bboxes only (no score column in some
            # versions); default to 1.0 rather than inventing a calibration.
            conf = vals[4] if len(vals) > 4 else 1.0
            out.append(Detection(x1, y1, x2, y2, conf))
        return out


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def build_detector(spec: str) -> PersonDetector:
    """Resolve a benchmark spec string to a detector.

    Specs: 'mediapipe', 'yolo11n@640', 'yolo11m@960', 'yolo11x@1280',
    'rtmdet@640'. Keeping configuration in the name means a results table can
    never accidentally attribute one configuration's numbers to another.
    """
    if spec == "mediapipe":
        return MediaPipeDetector()
    if spec.startswith("yolox"):
        head, _, size = spec.partition("@")
        variant = head.split("-")[1] if "-" in head else "m"
        return YoloxDetector(imgsz=int(size or 640), variant=variant)
    if spec.startswith("yolo"):
        model, _, size = spec.partition("@")
        return YoloDetector(weights=f"{model}.pt", imgsz=int(size or 960))
    raise ValueError(f"unknown detector spec {spec!r}")
