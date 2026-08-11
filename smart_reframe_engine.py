#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smart_reframe_engine.py — EnterpriseSmartReframer

A standalone, dependency-isolated Smart-Reframe engine that converts a
horizontal (or any aspect) source video into an intelligent 9:16 vertical
crop track, mirroring the "auto-reframe" behavior of tools like Opus Clip.

This module is intentionally self-contained: importing it never touches the
host application's state, and every heavy dependency (`cv2`, `mediapipe`,
`numpy`) is imported lazily / defensively so the module can be imported even
on machines that only have OpenCV, or neither.

Public integration surface (used by RenderDummy.py):

    from smart_reframe_engine import generate_smart_track, ENGINE_AVAILABLE

    track, faces_found, meta = generate_smart_track(video_path, duration)
    # track   -> [[t, cx, cy], ...]   cx/cy normalized 0..1 face-center points
    # faces_found -> bool
    # meta    -> dict of diagnostics (engine used, scene cuts, speaker swaps..)

It can also be run completely standalone from the CLI to render a finished
9:16 MP4 with audio preserved — see `if __name__ == "__main__"` at the
bottom.

ARCHITECTURE
------------
 1. FaceDetectorBackend      — MediaPipe Face Detection primary, Haar cascade
                                fallback if MediaPipe isn't installed or errors.
 1b. PersonDetectorBackend   — YOLOv8 person-detection tier (ultralytics),
                                completing the dual-model CV stack: supplies a
                                head-region pseudo-face fallback when the face
                                detector finds nobody, and full body extents
                                used by the group (two-shot) framing decision.
                                Fully optional — skipped if `ultralytics` isn't
                                installed or a model can't be loaded.
 2. MouthMotionEstimator     — MediaPipe Face Mesh lip-landmark delta tracking,
                                used to score which on-screen face is actively
                                talking (skipped gracefully if unavailable).
 3. IOUTracker               — assigns persistent integer track IDs to faces
                                across sampled frames via IoU + centroid
                                fallback matching (a lightweight multi-object
                                state machine — no scipy dependency).
 4. SceneCutDetector         — cv2.calcHist correlation between consecutive
                                sampled frames to flag hard cuts.
 5. OpticalFlowDriftEstimator— cv2.calcOpticalFlowFarneback dense flow, used
                                to keep tracking a region when face detection
                                drops out (fast head turns, profile shots).
 6. KalmanAxis1D             — true constant-velocity 1D Kalman filter
                                (state = [position, velocity]) used per-axis
                                to estimate the hidden "true" target position.
 7. DeadZoneInertialSmoother — wraps a KalmanAxis1D pair (x, y) and adds a
                                dead-zone (micro-jitter lock) plus S-curve
                                inertial pan easing when the subject leaves
                                the dead-zone, or an instant snap on a hard
                                scene cut / speaker swap.
 8. EnterpriseSmartReframer  — orchestrates the two-pass pipeline:
      Pass 1 (analyze):  sample frames, detect faces, track IDs, score
                          "importance" per track (size + centrality + mouth
                          motion), flag scene cuts.
      Pass 2 (build):    walk samples in order, pick the active-speaker
                          track per sample, feed it through the Kalman +
                          dead-zone + inertial smoother, reset state cleanly
                          on hard cuts / speaker swaps, and fall back to
                          optical-flow drift / rolling-window (last 10
                          frames) velocity coasting when detection is
                          momentarily lost. Also evaluates, per sample,
                          whether two active speakers should be co-framed
                          in a widened two-shot, producing a companion
                          `zoom_track` alongside the cx/cy `track`.
 9. render_video()           — optional full standalone renderer: a
                                dedicated reader thread decodes frames into a
                                bounded queue.Queue while the main thread
                                applies the crop/zoom (auto-zoom fallback
                                with cv2.INTER_LANCZOS4 rescale) and streams
                                to ffmpeg for H.264 encoding, then remuxes
                                the original audio.

Every class degrades gracefully: if `cv2` is entirely missing, the whole
module is unusable and `ENGINE_AVAILABLE` is False. If `mediapipe` is
missing, multi-speaker mouth-motion scoring and the primary face detector
are skipped and the engine automatically falls back to Haar cascades with
size/centrality-only scoring — everything else (Kalman smoothing, dead-zone,
scene-cut segmentation, optical-flow fallback) still runs in full.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

try:
    from ultralytics import YOLO as _UltralyticsYOLO
    YOLO_OK = True
except Exception:
    YOLO_OK = False

ENGINE_AVAILABLE = CV2_OK and NUMPY_OK


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SmartReframeConfig:
    """All tunable parameters for the engine in one place."""

    # --- sampling ---
    sample_every: float = 0.30          # seconds between analyzed frames
    max_samples: int = 900              # hard cap so huge videos don't hang

    # --- crop composition (Matte Boundary Clamping) ---
    headroom_frac: float = 0.15         # extra buffer above the highest face
    side_cushion_frac: float = 0.40     # extra horizontal cushion around face
    target_aspect: float = 9.0 / 16.0   # output crop aspect ratio (w/h)
    min_face_frac_of_height: float = 0.16   # auto-zoom-out trigger (face too big)
    max_face_frac_of_height: float = 0.55   # auto-zoom-in trigger (face too small)

    # --- Kalman filter (per axis) ---
    kalman_process_var: float = 6.0     # process noise (how much we trust motion model)
    kalman_measurement_var: float = 40.0  # measurement noise (how much we trust detections)

    # --- dead-zone / inertial panning ---
    dead_zone_frac: float = 0.065       # +/- fraction of width/height: locked, no pan
    inertial_ease_frames: int = 10      # frames (in sample-space) to complete an S-curve pan
    max_pan_speed_frac: float = 0.35    # max fraction of width a pan may cover per second

    # --- multi-speaker importance scoring weights ---
    w_size: float = 0.40
    w_centrality: float = 0.25
    w_mouth_motion: float = 0.35
    speaker_switch_hysteresis: float = 0.12   # new speaker must beat current by this margin
    speaker_switch_min_hold_s: float = 0.9    # minimum time to hold a speaker before re-eval

    # --- scene cut detection ---
    scene_cut_hist_correlation_floor: float = 0.55  # below this correlation => hard cut
    scene_cut_min_gap_s: float = 0.5                # ignore cuts closer together than this

    # --- fault tolerance ---
    optical_flow_max_frames: int = 6    # how many consecutive samples optical flow may bridge
    velocity_coast_max_frames: int = 4  # then coast on last known velocity before giving up

    # --- detector ---
    min_detection_confidence: float = 0.5
    haar_scale_factor: float = 1.15
    haar_min_neighbors: int = 5

    # --- dual-model CV stack (YOLO person layer, item 1 of the spec) ---
    enable_yolo: bool = True                 # try YOLO person detector if ultralytics is installed
    yolo_model_name: str = "yolov8n.pt"      # nano model: fast enough to run per-sample on CPU
    yolo_person_conf: float = 0.35           # min confidence to accept a YOLO "person" box
    yolo_class_id_person: int = 0            # COCO class id for "person" in the standard YOLOv8 weights

    # --- multi-speaker group (two-shot) framing ---
    enable_group_framing: bool = True
    group_frame_gap_frac: float = 0.55       # max horizontal gap (fraction of frame width) between
                                              # two active speakers' centers to still frame both together
    group_frame_mouth_activity_floor: float = 0.15   # both must be at least this "talking" to co-frame
    group_frame_max_zoom_out: float = 1.6    # cap on how far the group shot may zoom out (1.0 = no zoom)
    group_frame_ease_frames: int = 8         # frames to ease the zoom level in/out (avoid a zoom snap)

    # --- rolling-window velocity extrapolation (item 5: last-N-frame coast) ---
    velocity_window_frames: int = 10         # rolling window size for velocity-coast extrapolation

    # --- async I/O pipeline ---
    reader_queue_size: int = 32              # bounded queue depth between the reader thread and encoder


# ═══════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FaceObservation:
    """A single detected face in a single sampled frame, in pixel space."""
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0
    mouth_motion: float = 0.0     # 0..1, delta lip-distance vs. previous sample for this track
    track_id: int = -1

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def iou(self, other: "FaceObservation") -> float:
        ax1, ay1, ax2, ay2 = self.x, self.y, self.x + self.w, self.y + self.h
        bx1, by1, bx2, by2 = other.x, other.y, other.x + other.w, other.y + other.h
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """Persistent multi-object tracking state for one on-screen person."""
    track_id: int
    last_obs: FaceObservation
    last_seen_sample_idx: int
    hits: int = 1
    prev_mouth_landmarks: Optional[Any] = None
    importance_ema: float = 0.0


@dataclass
class Sample:
    """Everything gathered for one analyzed frame index during Pass 1."""
    t: float
    frame_idx: int
    faces: List[FaceObservation] = field(default_factory=list)
    scene_cut: bool = False
    frame_gray_small: Optional[Any] = None   # kept only for optical-flow bridging
    person_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    # full-body (x, y, w, h) boxes from the YOLO person layer — used both as a
    # fallback tier when the face detector finds nothing, and as the input to
    # the multi-speaker "group framing" decision in Pass 2.


# ═══════════════════════════════════════════════════════════════════════
# 1D Kalman filter — constant velocity model
# ═══════════════════════════════════════════════════════════════════════

class KalmanAxis1D:
    """A true 1D Kalman filter tracking [position, velocity] hidden state.

    State transition matrix F = [[1, dt], [0, 1]]
    Measurement matrix       H = [[1, 0]]
    Process noise covariance Q (scaled by dt)
    Measurement noise cov.   R
    """

    def __init__(self, process_var: float, measurement_var: float, initial_pos: float = 0.0):
        self.process_var = process_var
        self.measurement_var = measurement_var
        # state: [position, velocity]
        self.x = [float(initial_pos), 0.0]
        # error covariance matrix (2x2), start with high uncertainty
        self.P = [[100.0, 0.0], [0.0, 100.0]]
        self.initialized = False

    def reset(self, pos: float) -> None:
        """Hard reset used on scene cuts / speaker swaps — clears the filter
        state queue instead of trying to smoothly interpolate through a cut."""
        self.x = [float(pos), 0.0]
        self.P = [[100.0, 0.0], [0.0, 100.0]]
        self.initialized = True

    def predict(self, dt: float) -> float:
        F = [[1.0, dt], [0.0, 1.0]]
        x0, x1 = self.x
        nx0 = F[0][0] * x0 + F[0][1] * x1
        nx1 = F[1][0] * x0 + F[1][1] * x1
        self.x = [nx0, nx1]

        # Q scaled by dt for a discretized constant-velocity model
        q = self.process_var * max(dt, 1e-3)
        Q = [[q * dt * dt / 3.0, q * dt / 2.0], [q * dt / 2.0, q]]

        P = self.P
        # P = F P F^T + Q
        FP = [
            [F[0][0] * P[0][0] + F[0][1] * P[1][0], F[0][0] * P[0][1] + F[0][1] * P[1][1]],
            [F[1][0] * P[0][0] + F[1][1] * P[1][0], F[1][0] * P[0][1] + F[1][1] * P[1][1]],
        ]
        FPFt = [
            [FP[0][0] * F[0][0] + FP[0][1] * F[0][1], FP[0][0] * F[1][0] + FP[0][1] * F[1][1]],
            [FP[1][0] * F[0][0] + FP[1][1] * F[0][1], FP[1][0] * F[1][0] + FP[1][1] * F[1][1]],
        ]
        self.P = [
            [FPFt[0][0] + Q[0][0], FPFt[0][1] + Q[0][1]],
            [FPFt[1][0] + Q[1][0], FPFt[1][1] + Q[1][1]],
        ]
        return self.x[0]

    def update(self, measurement: float) -> float:
        if not self.initialized:
            self.reset(measurement)
            return self.x[0]
        H = [1.0, 0.0]
        P = self.P
        R = self.measurement_var
        # innovation covariance S = H P H^T + R
        S = H[0] * P[0][0] * H[0] + H[0] * P[0][1] * H[1] + H[1] * P[1][0] * H[0] + H[1] * P[1][1] * H[1] + R
        if S <= 1e-9:
            S = 1e-9
        # Kalman gain K = P H^T / S
        K0 = (P[0][0] * H[0] + P[0][1] * H[1]) / S
        K1 = (P[1][0] * H[0] + P[1][1] * H[1]) / S
        y = measurement - (H[0] * self.x[0] + H[1] * self.x[1])  # innovation
        self.x = [self.x[0] + K0 * y, self.x[1] + K1 * y]
        # P = (I - K H) P
        self.P = [
            [P[0][0] - K0 * H[0] * P[0][0] - K0 * H[1] * P[1][0],
             P[0][1] - K0 * H[0] * P[0][1] - K0 * H[1] * P[1][1]],
            [P[1][0] - K1 * H[0] * P[0][0] - K1 * H[1] * P[1][0],
             P[1][1] - K1 * H[0] * P[0][1] - K1 * H[1] * P[1][1]],
        ]
        return self.x[0]

    @property
    def velocity(self) -> float:
        return self.x[1]

    @property
    def position(self) -> float:
        return self.x[0]


def _smoothstep(t: float) -> float:
    """Classic S-curve (Hermite smoothstep) ease used for inertial panning —
    accelerates out of the dead-zone and decelerates into the new target,
    mimicking a hydraulic tripod pan instead of a linear/robotic move."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class DeadZoneInertialSmoother:
    """Combines a KalmanAxis1D for X and Y with dead-zone locking and
    S-curve inertial panning bounds. This is the "camera operator" layer
    sitting on top of the raw Kalman position estimate."""

    def __init__(self, cfg: SmartReframeConfig, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.kx = KalmanAxis1D(cfg.kalman_process_var, cfg.kalman_measurement_var, frame_w / 2.0)
        self.ky = KalmanAxis1D(cfg.kalman_process_var, cfg.kalman_measurement_var, frame_h / 2.0)
        self.output_x = frame_w / 2.0
        self.output_y = frame_h / 2.0
        self.pan_active = False
        self.pan_start_x = self.output_x
        self.pan_start_y = self.output_y
        self.pan_target_x = self.output_x
        self.pan_target_y = self.output_y
        self.pan_progress = 0.0
        self._initialized = False

    def hard_reset(self, x: float, y: float) -> None:
        """Used on scene cuts or speaker swaps: clear filter queues and jump
        straight to the new position (a clean cut, not an aggressive pan)."""
        self.kx.reset(x)
        self.ky.reset(y)
        self.output_x = x
        self.output_y = y
        self.pan_active = False
        self.pan_progress = 0.0

    def step(self, measured_x: Optional[float], measured_y: Optional[float], dt: float) -> Tuple[float, float]:
        """Advance the smoother by one sample. measured_* may be None if
        detection was lost (predict-only Kalman step continues coasting)."""
        dt = max(dt, 1e-3)
        self.kx.predict(dt)
        self.ky.predict(dt)
        if measured_x is not None:
            self.kx.update(measured_x)
        if measured_y is not None:
            self.ky.update(measured_y)

        target_x, target_y = self.kx.position, self.ky.position

        if not self._initialized:
            self.output_x, self.output_y = target_x, target_y
            self._initialized = True
            return self.output_x, self.output_y

        dead_zone_px_x = self.cfg.dead_zone_frac * self.frame_w
        dead_zone_px_y = self.cfg.dead_zone_frac * self.frame_h
        dx = target_x - self.output_x
        dy = target_y - self.output_y

        outside_dead_zone = abs(dx) > dead_zone_px_x or abs(dy) > dead_zone_px_y

        if not outside_dead_zone and not self.pan_active:
            # Micro-movement inside the dead-zone: stay perfectly locked.
            return self.output_x, self.output_y

        if outside_dead_zone and not self.pan_active:
            # Just broke out of the dead-zone: start a fresh inertial pan.
            self.pan_active = True
            self.pan_start_x, self.pan_start_y = self.output_x, self.output_y
            self.pan_progress = 0.0

        # Continuously refresh the target in case the Kalman estimate keeps
        # drifting while we're mid-pan (keeps the pan honest / adaptive).
        self.pan_target_x, self.pan_target_y = target_x, target_y

        step_frac = 1.0 / max(1, self.cfg.inertial_ease_frames)
        self.pan_progress = min(1.0, self.pan_progress + step_frac)
        eased = _smoothstep(self.pan_progress)

        new_x = self.pan_start_x + (self.pan_target_x - self.pan_start_x) * eased
        new_y = self.pan_start_y + (self.pan_target_y - self.pan_start_y) * eased

        # Clamp max pan speed so a bad detection spike can't whip-pan the camera.
        max_step_x = self.cfg.max_pan_speed_frac * self.frame_w * dt
        max_step_y = self.cfg.max_pan_speed_frac * self.frame_h * dt
        new_x = self.output_x + max(-max_step_x, min(max_step_x, new_x - self.output_x))
        new_y = self.output_y + max(-max_step_y, min(max_step_y, new_y - self.output_y))

        self.output_x, self.output_y = new_x, new_y

        if self.pan_progress >= 1.0:
            self.pan_active = False

        return self.output_x, self.output_y


# ═══════════════════════════════════════════════════════════════════════
# Face detection backends
# ═══════════════════════════════════════════════════════════════════════

class FaceDetectorBackend:
    """MediaPipe Face Detection primary, Haar cascade automatic fallback.
    Exposes a single `.detect(bgr_frame) -> List[FaceObservation]` method
    regardless of which backend is actually active."""

    def __init__(self, cfg: SmartReframeConfig):
        self.cfg = cfg
        self.backend = "none"
        self._mp_detector = None
        self._haar = None
        if MEDIAPIPE_OK:
            try:
                self._mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=cfg.min_detection_confidence
                )
                self.backend = "mediapipe"
            except Exception:
                self._mp_detector = None
        if self._mp_detector is None and CV2_OK:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            try:
                self._haar = cv2.CascadeClassifier(cascade_path)
                if self._haar.empty():
                    self._haar = None
                else:
                    self.backend = "haar"
            except Exception:
                self._haar = None

    @property
    def available(self) -> bool:
        return self._mp_detector is not None or self._haar is not None

    def detect(self, frame_bgr) -> List[FaceObservation]:
        h, w = frame_bgr.shape[:2]
        if self._mp_detector is not None:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = self._mp_detector.process(rgb)
                out = []
                if result.detections:
                    for det in result.detections:
                        bb = det.location_data.relative_bounding_box
                        fx, fy = bb.xmin * w, bb.ymin * h
                        fw, fh = bb.width * w, bb.height * h
                        # Clamp to frame bounds — MediaPipe can return
                        # slightly out-of-bounds boxes near the edges.
                        fx = max(0.0, min(fx, w - 1))
                        fy = max(0.0, min(fy, h - 1))
                        fw = max(1.0, min(fw, w - fx))
                        fh = max(1.0, min(fh, h - fy))
                        score = det.score[0] if det.score else 0.75
                        out.append(FaceObservation(fx, fy, fw, fh, confidence=float(score)))
                return out
            except Exception:
                pass  # fall through to Haar if MediaPipe throws mid-stream
        if self._haar is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            min_size = (max(20, int(w * 0.06)), max(20, int(h * 0.06)))
            faces = self._haar.detectMultiScale(
                gray, scaleFactor=self.cfg.haar_scale_factor,
                minNeighbors=self.cfg.haar_min_neighbors, minSize=min_size,
            )
            return [FaceObservation(float(x), float(y), float(fw), float(fh), confidence=0.6)
                    for (x, y, fw, fh) in faces]
        return []


class PersonDetectorBackend:
    """Secondary detection tier: a YOLOv8 person detector, run alongside
    `FaceDetectorBackend`. This is the second half of the "dual-model CV
    stack" — MediaPipe/Haar gives precise face boxes when a face is visible
    square-on, while YOLO's person boxes stay reliable when someone has
    turned away, is partially occluded, or is far enough from the lens that
    the face detector's confidence collapses. It is also the sole source of
    body-extent information used for multi-speaker group (two-shot) framing,
    since a face box alone can't tell us how much a second person's body
    would need to fit inside the crop.

    Entirely optional: if `ultralytics` isn't installed, or model download
    fails (no network, etc.), `.available` is False and the engine silently
    proceeds on the face-only tier exactly as before.
    """

    def __init__(self, cfg: SmartReframeConfig):
        self.cfg = cfg
        self._model = None
        if cfg.enable_yolo and YOLO_OK:
            try:
                self._model = _UltralyticsYOLO(cfg.yolo_model_name)
            except Exception:
                self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, frame_bgr) -> List[Tuple[float, float, float, float]]:
        """Returns a list of (x, y, w, h) person boxes in pixel space,
        filtered to the COCO "person" class and the configured confidence
        floor. Never raises — any inference failure degrades to an empty
        list so the rest of the pipeline keeps running untouched."""
        if self._model is None:
            return []
        try:
            results = self._model.predict(
                frame_bgr, classes=[self.cfg.yolo_class_id_person],
                conf=self.cfg.yolo_person_conf, verbose=False,
            )
        except Exception:
            return []
        boxes: List[Tuple[float, float, float, float]] = []
        h, w = frame_bgr.shape[:2]
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                try:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                except Exception:
                    continue
                x1 = max(0.0, min(x1, w - 1))
                y1 = max(0.0, min(y1, h - 1))
                x2 = max(x1 + 1.0, min(x2, w))
                y2 = max(y1 + 1.0, min(y2, h))
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

    @staticmethod
    def head_region_from_person_box(box: Tuple[float, float, float, float]) -> FaceObservation:
        """Synthesizes a pseudo-face box from the top ~22% of a YOLO person
        box (a reasonable head/shoulders proportion for a standing or seated
        subject), used as the Pass-1 fallback tier when the primary face
        detector finds nobody but a person is clearly present on screen."""
        x, y, w, h = box
        head_h = h * 0.22
        head_w = w * 0.55
        head_x = x + (w - head_w) / 2.0
        return FaceObservation(head_x, y, head_w, head_h, confidence=0.4)


class MouthMotionEstimator:
    """Uses MediaPipe Face Mesh lip landmarks to compute a 0..1 "how much
    did the mouth move since last sample" score per tracked face. This is
    the active-conversational-state signal used for multi-speaker anchor
    switching. Fully optional — engine works without it."""

    # Inner + outer lip landmark indices (MediaPipe Face Mesh topology)
    _UPPER_LIP = 13
    _LOWER_LIP = 14
    _LEFT_MOUTH = 61
    _RIGHT_MOUTH = 291

    def __init__(self, cfg: SmartReframeConfig):
        self.cfg = cfg
        self._mesh = None
        if MEDIAPIPE_OK:
            try:
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False, max_num_faces=6, refine_landmarks=False,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5,
                )
            except Exception:
                self._mesh = None

    @property
    def available(self) -> bool:
        return self._mesh is not None

    def landmarks_for_face(self, frame_bgr, face: FaceObservation):
        """Crops loosely around the face box and runs the mesh on just that
        region for speed, returning normalized lip landmark points, or None."""
        if self._mesh is None:
            return None
        h, w = frame_bgr.shape[:2]
        pad = 0.35
        x0 = max(0, int(face.x - face.w * pad))
        y0 = max(0, int(face.y - face.h * pad))
        x1 = min(w, int(face.x + face.w * (1 + pad)))
        y1 = min(h, int(face.y + face.h * (1 + pad)))
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame_bgr[y0:y1, x0:x1]
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self._mesh.process(rgb)
        except Exception:
            return None
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        try:
            upper = lm[self._UPPER_LIP]
            lower = lm[self._LOWER_LIP]
            left = lm[self._LEFT_MOUTH]
            right = lm[self._RIGHT_MOUTH]
        except IndexError:
            return None
        mouth_w = max(1e-4, math.hypot((right.x - left.x) * (x1 - x0), (right.y - left.y) * (y1 - y0)))
        mouth_open = math.hypot((lower.x - upper.x) * (x1 - x0), (lower.y - upper.y) * (y1 - y0))
        return mouth_open / mouth_w  # normalized so it's scale-invariant


# ═══════════════════════════════════════════════════════════════════════
# Multi-object tracker (IOU-based state machine)
# ═══════════════════════════════════════════════════════════════════════

class IOUTracker:
    """A lightweight multi-object tracker: matches new detections to
    existing tracks via IoU (falling back to nearest-centroid when IoU is
    zero for all pairs, e.g. fast motion between sparse samples), assigns
    persistent track IDs, and prunes stale tracks."""

    def __init__(self, iou_match_floor: float = 0.15, max_missed_samples: int = 3):
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.iou_match_floor = iou_match_floor
        self.max_missed_samples = max_missed_samples

    def update(self, sample_idx: int, detections: List[FaceObservation]) -> List[FaceObservation]:
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        matches: List[Tuple[int, int]] = []  # (track_id, det_idx)

        # Greedy best-IoU matching (no scipy dependency required).
        pairs = []
        for tid in unmatched_tracks:
            for di in unmatched_dets:
                score = self.tracks[tid].last_obs.iou(detections[di])
                if score > self.iou_match_floor:
                    pairs.append((score, tid, di))
        pairs.sort(key=lambda p: -p[0])
        used_tracks, used_dets = set(), set()
        for score, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(di)
            matches.append((tid, di))

        # Centroid-distance fallback for anything IoU couldn't match
        # (handles sparse sampling where a face moved further than its own
        # box size between samples).
        remaining_tracks = [t for t in unmatched_tracks if t not in used_tracks]
        remaining_dets = [d for d in unmatched_dets if d not in used_dets]
        if remaining_tracks and remaining_dets:
            cpairs = []
            for tid in remaining_tracks:
                lo = self.tracks[tid].last_obs
                for di in remaining_dets:
                    d = detections[di]
                    dist = math.hypot(lo.cx - d.cx, lo.cy - d.cy)
                    diag = math.hypot(lo.w, lo.h) + 1e-6
                    if dist < diag * 1.5:
                        cpairs.append((dist, tid, di))
            cpairs.sort(key=lambda p: p[0])
            for dist, tid, di in cpairs:
                if tid in used_tracks or di in used_dets:
                    continue
                used_tracks.add(tid)
                used_dets.add(di)
                matches.append((tid, di))

        for tid, di in matches:
            det = detections[di]
            det.track_id = tid
            tr = self.tracks[tid]
            tr.last_obs = det
            tr.last_seen_sample_idx = sample_idx
            tr.hits += 1

        for di in range(len(detections)):
            if di not in used_dets:
                det = detections[di]
                tid = self._next_id
                self._next_id += 1
                det.track_id = tid
                self.tracks[tid] = Track(track_id=tid, last_obs=det, last_seen_sample_idx=sample_idx)

        # Prune tracks that haven't been seen in a while.
        stale = [tid for tid, tr in self.tracks.items()
                 if sample_idx - tr.last_seen_sample_idx > self.max_missed_samples]
        for tid in stale:
            del self.tracks[tid]

        return detections


# ═══════════════════════════════════════════════════════════════════════
# Scene-cut detection
# ═══════════════════════════════════════════════════════════════════════

class SceneCutDetector:
    """Histogram-correlation based hard-cut detector (two-pass look-ahead:
    consulted for every consecutive sampled-frame pair during Pass 1)."""

    def __init__(self, cfg: SmartReframeConfig):
        self.cfg = cfg
        self._prev_hist = None
        self._last_cut_t = -1e9

    def check(self, frame_bgr, t: float) -> bool:
        hist = cv2.calcHist([frame_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        is_cut = False
        if self._prev_hist is not None:
            corr = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr < self.cfg.scene_cut_hist_correlation_floor and \
               (t - self._last_cut_t) > self.cfg.scene_cut_min_gap_s:
                is_cut = True
                self._last_cut_t = t
        self._prev_hist = hist
        return is_cut


# ═══════════════════════════════════════════════════════════════════════
# Optical-flow drift fallback
# ═══════════════════════════════════════════════════════════════════════

class OpticalFlowDriftEstimator:
    """When face detection fails on a sample (profile turn, motion blur,
    occlusion), estimate how the last-known face region drifted using dense
    Farneback optical flow between the last-good frame and the current one."""

    def estimate_drift(self, prev_gray, curr_gray, region: Tuple[float, float, float, float]
                        ) -> Optional[Tuple[float, float]]:
        if prev_gray is None or curr_gray is None:
            return None
        x, y, w, h = region
        h_img, w_img = prev_gray.shape[:2]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w_img, int(x + w)), min(h_img, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return None
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )
        except Exception:
            return None
        region_flow = flow[y0:y1, x0:x1]
        if region_flow.size == 0:
            return None
        mean_dx = float(region_flow[..., 0].mean())
        mean_dy = float(region_flow[..., 1].mean())
        return mean_dx, mean_dy


# ═══════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════

class EnterpriseSmartReframer:
    """Two-pass Smart-Reframe engine: analyzes a source video and produces
    a smoothed, scene-cut-aware, multi-speaker-aware 9:16 crop track."""

    def __init__(self, config: Optional[SmartReframeConfig] = None):
        if not ENGINE_AVAILABLE:
            raise ImportError(
                "smart_reframe_engine needs 'opencv-python' (and optionally "
                "'mediapipe' for multi-speaker + mouth-motion scoring) and 'numpy'. "
                "Install with: pip install opencv-python-headless numpy mediapipe"
            )
        self.cfg = config or SmartReframeConfig()
        self.detector = FaceDetectorBackend(self.cfg)
        self.person_detector = PersonDetectorBackend(self.cfg)
        self.mouth_estimator = MouthMotionEstimator(self.cfg)
        self.scene_cut_detector = SceneCutDetector(self.cfg)
        self.flow_estimator = OpticalFlowDriftEstimator()
        self.meta: Dict[str, Any] = {}

    # ---------------------------------------------------------- Pass 1 ---
    def _pass_one_analyze(self, cap, fps: float, frame_w: int, frame_h: int, duration: float
                           ) -> List[Sample]:
        step_frames = max(1, int(round(fps * self.cfg.sample_every)))
        tracker = IOUTracker()
        samples: List[Sample] = []
        frame_idx = 0
        taken = 0
        analysis_scale = 480.0 / max(frame_w, 1)  # downscale big frames for the histogram/flow pass
        analysis_scale = min(1.0, analysis_scale) if analysis_scale > 0 else 1.0

        while taken < self.cfg.max_samples:
            t = frame_idx / fps
            if t > duration:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            faces = self.detector.detect(frame)

            # --- Dual-model fallback tier: YOLO person boxes ---
            # Always collected (cheap relative to the sample interval) since
            # Pass 2's group-framing decision needs body extents, not just
            # face boxes. Also used to synthesize a pseudo-face the moment
            # the primary face detector comes up empty but a person is
            # clearly present (profile turn, partial occlusion, distance).
            person_boxes: List[Tuple[float, float, float, float]] = []
            if self.person_detector.available:
                person_boxes = self.person_detector.detect(frame)
                if not faces and person_boxes:
                    # Pick the largest person box as the most likely active
                    # subject and synthesize a head-region pseudo-face so the
                    # rest of the pipeline (tracking, scoring, smoothing)
                    # never has to special-case "face missing but person seen".
                    largest = max(person_boxes, key=lambda b: b[2] * b[3])
                    pseudo = PersonDetectorBackend.head_region_from_person_box(largest)
                    faces = [pseudo]

            faces = tracker.update(taken, faces)

            # Mouth-motion scoring per matched track (skipped if mediapipe unavailable).
            if self.mouth_estimator.available:
                for f in faces:
                    tr = tracker.tracks.get(f.track_id)
                    lip_ratio = self.mouth_estimator.landmarks_for_face(frame, f)
                    if lip_ratio is not None and tr is not None:
                        if tr.prev_mouth_landmarks is not None:
                            delta = abs(lip_ratio - tr.prev_mouth_landmarks)
                            f.mouth_motion = max(0.0, min(1.0, delta * 6.0))
                        tr.prev_mouth_landmarks = lip_ratio

            small = cv2.resize(frame, (0, 0), fx=analysis_scale, fy=analysis_scale) \
                if analysis_scale < 1.0 else frame
            is_cut = self.scene_cut_detector.check(small, t)
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            samples.append(Sample(t=t, frame_idx=frame_idx, faces=faces,
                                   scene_cut=is_cut, frame_gray_small=gray_small,
                                   person_boxes=person_boxes))
            frame_idx += step_frames
            taken += 1

        return samples

    # -------------------------------------------------- Speaker scoring ---
    def _importance_score(self, face: FaceObservation, frame_w: int, frame_h: int) -> float:
        cfg = self.cfg
        size_score = min(1.0, face.area / (frame_w * frame_h * 0.35))
        center_dist = math.hypot((face.cx - frame_w / 2.0) / (frame_w / 2.0),
                                  (face.cy - frame_h / 2.0) / (frame_h / 2.0))
        centrality_score = max(0.0, 1.0 - center_dist)
        return (cfg.w_size * size_score
                + cfg.w_centrality * centrality_score
                + cfg.w_mouth_motion * face.mouth_motion)

    # ---------------------------------------------------------- Pass 2 ---
    def _pass_two_build_track(self, samples: List[Sample], frame_w: int, frame_h: int
                               ) -> Tuple[List[List[float]], bool, Dict[str, Any]]:
        cfg = self.cfg
        smoother = DeadZoneInertialSmoother(cfg, frame_w, frame_h)
        track: List[List[float]] = []

        current_speaker_id = None
        speaker_locked_until_t = -1e9
        last_good_region: Optional[Tuple[float, float, float, float]] = None
        last_good_gray = None
        misses_in_a_row = 0
        # Rolling window of the last N per-axis velocity samples (item 5 of
        # the spec: "continuous velocity extrapolation vector derived from
        # the last 10 frames"), rather than a single last-observed delta —
        # this smooths out one noisy detection right before tracking drops.
        velocity_window: deque = deque(maxlen=max(1, self.cfg.velocity_window_frames))
        any_face_found = False
        scene_cuts = 0
        speaker_swaps = 0

        # --- Group (two-shot) framing state ---
        zoom_track: List[List[float]] = []       # [[t, zoom_factor], ...] companion to `track`
        group_zoom_current = 1.0
        group_zoom_target = 1.0
        group_zoom_progress = 1.0                # 1.0 = fully eased to target already

        for i, s in enumerate(samples):
            candidates = s.faces
            best_face = None
            if candidates:
                scored = [(self._importance_score(f, frame_w, frame_h), f) for f in candidates]
                scored.sort(key=lambda p: -p[0])
                top_score, top_face = scored[0]

                if current_speaker_id is None:
                    best_face = top_face
                    current_speaker_id = top_face.track_id
                else:
                    incumbent = next((f for f in candidates if f.track_id == current_speaker_id), None)
                    if incumbent is None:
                        best_face = top_face
                        current_speaker_id = top_face.track_id
                        speaker_swaps += 1
                    else:
                        incumbent_score = self._importance_score(incumbent, frame_w, frame_h)
                        can_switch = s.t >= speaker_locked_until_t
                        if (can_switch and top_face.track_id != current_speaker_id
                                and top_score - incumbent_score > cfg.speaker_switch_hysteresis):
                            best_face = top_face
                            current_speaker_id = top_face.track_id
                            speaker_locked_until_t = s.t + cfg.speaker_switch_min_hold_s
                            speaker_swaps += 1
                        else:
                            best_face = incumbent

            hard_cut = s.scene_cut
            if hard_cut:
                scene_cuts += 1

            if best_face is not None:
                any_face_found = True
                misses_in_a_row = 0
                last_good_region = (best_face.x, best_face.y, best_face.w, best_face.h)
                last_good_gray = s.frame_gray_small
                measured_x, measured_y = self._compute_target_center(best_face, frame_w, frame_h)
                if i > 0:
                    prev_out = (smoother.output_x, smoother.output_y)
                    dt_est = max(1e-3, s.t - samples[i - 1].t)
                    velocity_window.append(
                        ((measured_x - prev_out[0]) / dt_est, (measured_y - prev_out[1]) / dt_est)
                    )
            else:
                # --- Fault tolerance chain: optical flow, then velocity coast ---
                measured_x = measured_y = None
                if last_good_region is not None and misses_in_a_row < cfg.optical_flow_max_frames:
                    drift = self.flow_estimator.estimate_drift(last_good_gray, s.frame_gray_small, last_good_region)
                    if drift is not None:
                        dx, dy = drift
                        lx, ly, lw, lh = last_good_region
                        lx, ly = lx + dx, ly + dy
                        last_good_region = (lx, ly, lw, lh)
                        cx = (lx + lw / 2.0) / frame_w
                        cy = (ly + lh / 2.0) / frame_h
                        measured_x, measured_y = cx * frame_w, cy * frame_h
                        last_good_gray = s.frame_gray_small
                elif misses_in_a_row < cfg.optical_flow_max_frames + cfg.velocity_coast_max_frames:
                    # Predictive velocity coast: keep panning at the average
                    # speed observed over the last N frames (rolling window)
                    # until tracking recovers, instead of freezing or trusting
                    # a single possibly-noisy last delta.
                    dt_est = self.cfg.sample_every
                    if velocity_window:
                        avg_vx = sum(v[0] for v in velocity_window) / len(velocity_window)
                        avg_vy = sum(v[1] for v in velocity_window) / len(velocity_window)
                    else:
                        avg_vx = avg_vy = 0.0
                    measured_x = smoother.output_x + avg_vx * dt_est
                    measured_y = smoother.output_y + avg_vy * dt_est
                misses_in_a_row += 1

            dt = self.cfg.sample_every if i == 0 else max(1e-3, s.t - samples[i - 1].t)

            if hard_cut and measured_x is not None:
                # Clean jump-cut: clear Kalman queues and snap instead of panning.
                smoother.hard_reset(measured_x, measured_y)
            else:
                smoother.step(measured_x, measured_y, dt)

            out_cx = smoother.output_x / frame_w
            out_cy = smoother.output_y / frame_h
            out_cx = max(0.0, min(1.0, out_cx))
            out_cy = max(0.0, min(1.0, out_cy))
            track.append([round(s.t, 3), round(out_cx, 4), round(out_cy, 4)])

            # --- Multi-speaker group (two-shot) framing (item 2 / item 6) ---
            # If two people are both actively talking and close enough
            # together, widen (zoom out) the crop so both stay in frame
            # instead of whip-panning between them on every speaker swap.
            new_target = self._group_frame_zoom_target(candidates, s.person_boxes, frame_w, frame_h) \
                if cfg.enable_group_framing else 1.0
            if abs(new_target - group_zoom_target) > 1e-3:
                # Target changed (group formed/dissolved): start a fresh ease
                # from wherever the zoom currently sits, rather than snapping.
                group_zoom_target = new_target
                group_zoom_progress = 0.0
            ease_step = 1.0 / max(1, cfg.group_frame_ease_frames)
            group_zoom_progress = min(1.0, group_zoom_progress + ease_step)
            eased = _smoothstep(group_zoom_progress)
            group_zoom_current = group_zoom_current + (group_zoom_target - group_zoom_current) * eased
            zoom_track.append([round(s.t, 3), round(group_zoom_current, 4)])

        meta = {
            "engine": "EnterpriseSmartReframer",
            "detector_backend": self.detector.backend,
            "person_detector_available": self.person_detector.available,
            "mouth_motion_available": self.mouth_estimator.available,
            "samples_analyzed": len(samples),
            "scene_cuts_detected": scene_cuts,
            "speaker_swaps": speaker_swaps,
            "zoom_track": zoom_track,
        }
        return track, any_face_found, meta

    def _group_frame_zoom_target(self, candidates: List[FaceObservation],
                                  person_boxes: List[Tuple[float, float, float, float]],
                                  frame_w: int, frame_h: int) -> float:
        """Decides whether the current sample should widen to a two-shot
        and, if so, how far to zoom out (as a multiplier passed to
        `crop_box_for_point`, where >1.0 means "zoom in" per that method's
        convention — so here we return the *inverse* zoom, i.e. a value
        < 1.0 to widen the effective crop, clamped to
        `1.0 / group_frame_max_zoom_out` at the widest).

        Two faces qualify as a "group" moment when both currently show
        mouth-motion above the activity floor (both are mid-conversation,
        not one listening silently) and their horizontal centers are within
        `group_frame_gap_frac` of the frame width apart. When they qualify,
        the returned zoom is chosen so the horizontal span covering both
        faces (plus the standard side cushion) fits inside the crop width.
        """
        cfg = self.cfg
        active = [f for f in candidates if f.mouth_motion >= cfg.group_frame_mouth_activity_floor]
        if len(active) < 2:
            return 1.0
        active.sort(key=lambda f: -f.mouth_motion)
        a, b = active[0], active[1]
        gap = abs(a.cx - b.cx) / max(1.0, frame_w)
        if gap > cfg.group_frame_gap_frac:
            return 1.0

        # Widen enough that a bounding span covering both faces (plus a
        # cushion so neither is cropped tight at the frame edge) fits within
        # the target 9:16 crop's width at the current zoom.
        left = min(a.x, b.x)
        right = max(a.x + a.w, b.x + b.w)
        span = (right - left) * (1.0 + cfg.side_cushion_frac * 0.5)
        crop_h = frame_h
        crop_w_at_1x = crop_h * cfg.target_aspect
        needed_zoom_out = span / max(1.0, crop_w_at_1x)
        needed_zoom_out = max(1.0, min(cfg.group_frame_max_zoom_out, needed_zoom_out))
        # crop_box_for_point treats zoom>1 as "zoom in" (smaller crop), so a
        # widen request is expressed as its reciprocal.
        return 1.0 / needed_zoom_out

    def _compute_target_center(self, face: FaceObservation, frame_w: int, frame_h: int) -> Tuple[float, float]:
        """Applies the headroom/side-cushion composition bias so the crop
        anchor sits at a cinematic point relative to the face, not dead-on
        the geometric center (e.g. a little above center for headroom)."""
        cfg = self.cfg
        cx = face.cx
        cy = face.y + face.h * (0.5 - cfg.headroom_frac * 0.5)
        cy = max(face.h * 0.15, min(frame_h - face.h * 0.15, cy))
        return cx, cy

    # ------------------------------------------------------- Public API ---
    def generate_track(self, video_path, duration: Optional[float] = None
                        ) -> Tuple[List[List[float]], bool, Dict[str, Any]]:
        """Runs the full two-pass pipeline and returns
        (track=[[t,cx,cy],...], faces_found, meta_diagnostics)."""
        video_path = str(video_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for analysis: {video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            if frame_w <= 0 or frame_h <= 0:
                raise RuntimeError("Source video reports zero-area frames — cannot analyze.")
            if duration is None:
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                duration = (frame_count / fps) if frame_count > 0 else 0.0
            if not duration or duration <= 0:
                raise RuntimeError("Could not determine clip duration for analysis.")

            samples = self._pass_one_analyze(cap, fps, frame_w, frame_h, duration)
            if not samples:
                return [], False, {"engine": "EnterpriseSmartReframer", "samples_analyzed": 0}
            track, faces_found, meta = self._pass_two_build_track(samples, frame_w, frame_h)
            self.meta = meta
            return track, faces_found, meta
        finally:
            cap.release()

    # --------------------------------------------------- Crop box maths ---
    def crop_box_for_point(self, cx_frac: float, cy_frac: float, frame_w: int, frame_h: int,
                            zoom: float = 1.0) -> Tuple[int, int, int, int]:
        """Given a normalized (cx, cy) anchor, returns a clamped
        (x, y, w, h) pixel crop box at the configured 9:16 target aspect,
        applying the side-cushion / auto-zoom fallback rules.

        `zoom` follows the convention used throughout this engine:
          - zoom == 1.0 : standard single-subject 9:16 crop (full source height).
          - zoom  > 1.0 : digital zoom-IN (subject too small / too close a
                          fallback wants tighter framing) — shrinks target_h
                          below frame_h.
          - zoom  < 1.0 : digital zoom-OUT, driven by group (two-shot)
                          framing when two active speakers need to share the
                          crop. Since the source's vertical extent is already
                          fully used at zoom==1.0, "zooming out" here widens
                          target_w beyond the normal side-cushion, up to the
                          full frame width. The resulting box is intentionally
                          wider than a strict 9:16 crop; the final resize to
                          the fixed 9:16 output canvas in `render_video`
                          absorbs the difference, which is the same
                          trade-off professional auto-reframe tools make for
                          brief two-shot moments rather than hard-cutting
                          between speakers.
        """
        cfg = self.cfg
        target_h = frame_h / max(zoom, 1.0)
        target_w = target_h * cfg.target_aspect
        if target_w > frame_w:
            target_w = float(frame_w)
            target_h = target_w / cfg.target_aspect
        cushion = 1.0 + cfg.side_cushion_frac
        target_w = min(frame_w, target_w * cushion) if target_w * cushion <= frame_w else target_w

        if zoom < 1.0:
            # Widen for a group two-shot: scale target_w by 1/zoom (capped to
            # the full frame width) while target_h stays put — see docstring.
            widened_w = target_w / max(zoom, 1e-3)
            target_w = min(float(frame_w), widened_w)

        cx_px = cx_frac * frame_w
        cy_px = cy_frac * frame_h
        x = cx_px - target_w / 2.0
        y = cy_px - target_h / 2.0
        x = max(0.0, min(frame_w - target_w, x))
        y = max(0.0, min(frame_h - target_h, y))
        return int(round(x)), int(round(y)), int(round(target_w)), int(round(target_h))

    # ------------------------------------------------- Standalone render ---
    def render_video(self, input_path, output_path, on_progress: Optional[Callable[[int, int], None]] = None
                      ) -> Dict[str, Any]:
        """Fully standalone renderer: analyzes the video, then re-encodes it
        to a 9:16 vertical MP4 following the smart crop track, piping raw
        frames to ffmpeg on a background thread and remuxing the original
        audio track back in afterward. Returns a diagnostics dict."""
        input_path = str(input_path)
        output_path = str(output_path)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        ffmpeg_bin = _resolve_ffmpeg()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = (frame_count / fps) if frame_count > 0 else 0.0
        cap.release()
        if frame_w <= 0 or frame_h <= 0 or duration <= 0:
            raise RuntimeError("Invalid source video dimensions or duration.")

        track, faces_found, meta = self.generate_track(input_path, duration)
        out_h = 1920
        out_w = int(out_h * self.cfg.target_aspect)

        # Build a fast lookup: for any timestamp, interpolate cx/cy from track.
        track_pts = track if track else [[0.0, 0.5, 0.5]]

        def interp(t: float) -> Tuple[float, float]:
            if t <= track_pts[0][0]:
                return track_pts[0][1], track_pts[0][2]
            if t >= track_pts[-1][0]:
                return track_pts[-1][1], track_pts[-1][2]
            lo, hi = 0, len(track_pts) - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if track_pts[mid][0] <= t:
                    lo = mid
                else:
                    hi = mid
            t0, cx0, cy0 = track_pts[lo]
            t1, cx1, cy1 = track_pts[hi]
            if t1 <= t0:
                return cx0, cy0
            f = (t - t0) / (t1 - t0)
            return cx0 + (cx1 - cx0) * f, cy0 + (cy1 - cy0) * f

        # Same binary-search interpolation for the group-framing zoom track
        # (item 2/6: dynamic zoom for simultaneous multi-speaker scenes).
        zoom_pts = meta.get("zoom_track") or [[0.0, 1.0]]

        def interp_zoom(t: float) -> float:
            if t <= zoom_pts[0][0]:
                return zoom_pts[0][1]
            if t >= zoom_pts[-1][0]:
                return zoom_pts[-1][1]
            lo, hi = 0, len(zoom_pts) - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if zoom_pts[mid][0] <= t:
                    lo = mid
                else:
                    hi = mid
            t0, z0 = zoom_pts[lo]
            t1, z1 = zoom_pts[hi]
            if t1 <= t0:
                return z0
            f = (t - t0) / (t1 - t0)
            return z0 + (z1 - z0) * f

        cap = cv2.VideoCapture(input_path)
        ffmpeg_cmd = [
            ffmpeg_bin, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps:.5f}",
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-f", "mp4", output_path + ".video_only.mp4",
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stderr_buf = []

        def _drain_stderr():
            try:
                for line in iter(proc.stderr.readline, b""):
                    stderr_buf.append(line)
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        # --- Async multi-threaded I/O pipeline (item 7 of the spec) ---
        # A dedicated reader thread does nothing but decode frames from
        # OpenCV as fast as the source allows and hands them off through a
        # bounded, thread-safe queue.Queue. The main thread is freed up to
        # spend all of its time on crop-box math, the Lanczos resize, and
        # feeding ffmpeg's stdin — the two stages overlap instead of the
        # decode stalling the encode (or vice versa) every single frame.
        frame_queue: "queue.Queue" = queue.Queue(maxsize=max(2, self.cfg.reader_queue_size))
        _SENTINEL = object()
        reader_error: List[BaseException] = []

        def _reader_worker():
            try:
                idx = 0
                while True:
                    ok, frm = cap.read()
                    if not ok:
                        break
                    frame_queue.put((idx, frm))
                    idx += 1
            except BaseException as exc:  # surface decode errors to the main thread
                reader_error.append(exc)
            finally:
                frame_queue.put(_SENTINEL)

        reader_thread = threading.Thread(target=_reader_worker, daemon=True, name="frame-reader")
        reader_thread.start()

        frame_idx = 0
        try:
            while True:
                item = frame_queue.get()
                if item is _SENTINEL:
                    break
                idx, frame = item
                t = idx / fps
                cx, cy = interp(t)
                zoom = interp_zoom(t)
                x, y, w, h = self.crop_box_for_point(cx, cy, frame_w, frame_h, zoom=zoom)
                cropped = frame[y:y + h, x:x + w]
                resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                try:
                    proc.stdin.write(resized.tobytes())
                except (BrokenPipeError, OSError):
                    break
                frame_idx += 1
                if on_progress:
                    on_progress(frame_idx, frame_count)
            if reader_error:
                raise reader_error[0]
        finally:
            cap.release()
            reader_thread.join(timeout=5)
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.wait()
            stderr_thread.join(timeout=5)

        if proc.returncode != 0:
            raise RuntimeError("ffmpeg video encode failed: " + b"".join(stderr_buf[-40:]).decode("utf-8", "ignore"))

        # Remux original audio (if any) back onto the freshly encoded video.
        video_only = output_path + ".video_only.mp4"
        mux_cmd = [
            ffmpeg_bin, "-y", "-i", video_only, "-i", input_path,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0?",
            "-c:a", "aac", "-b:a", "160k", "-shortest",
            "-movflags", "+faststart", output_path,
        ]
        mux = subprocess.run(mux_cmd, capture_output=True)
        try:
            os.remove(video_only)
        except OSError:
            pass
        if mux.returncode != 0:
            raise RuntimeError("ffmpeg audio remux failed: " + mux.stderr.decode("utf-8", "ignore")[-4000:])

        return {
            "output_path": output_path,
            "frames_rendered": frame_idx,
            "duration": duration,
            "faces_found": faces_found,
            **meta,
        }


def _resolve_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError("Could not locate an ffmpeg binary (checked imageio_ffmpeg and PATH).")


# ═══════════════════════════════════════════════════════════════════════
# Integration-friendly module-level function (what RenderDummy.py calls)
# ═══════════════════════════════════════════════════════════════════════

def generate_smart_track(video_path, duration: Optional[float] = None,
                          config: Optional[SmartReframeConfig] = None
                          ) -> Tuple[List[List[float]], bool, Dict[str, Any]]:
    """Convenience wrapper: builds an EnterpriseSmartReframer, runs the
    two-pass pipeline, and returns (track, faces_found, meta). Raises
    ImportError if cv2/numpy aren't installed, so callers can catch that
    specifically and fall back to a simpler detector."""
    engine = EnterpriseSmartReframer(config)
    return engine.generate_track(video_path, duration)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="smart_reframe_engine",
        description="EnterpriseSmartReframer — auto-reframe 16:9 video to 9:16 following the active speaker.",
    )
    p.add_argument("input", help="Path to the source video file.")
    p.add_argument("output", help="Path to write the rendered 9:16 MP4.")
    p.add_argument("--sample-every", type=float, default=0.30, help="Seconds between analyzed frames (default 0.30).")
    p.add_argument("--dead-zone", type=float, default=0.065, help="Dead-zone fraction of frame width/height (default 0.065).")
    p.add_argument("--side-cushion", type=float, default=0.40, help="Side cushion fraction around the face (default 0.40).")
    return p


def main() -> int:
    if not ENGINE_AVAILABLE:
        print("ERROR: this module requires opencv-python and numpy. "
              "Install with: pip install opencv-python-headless numpy mediapipe", file=sys.stderr)
        return 1
    args = _build_arg_parser().parse_args()
    cfg = SmartReframeConfig(
        sample_every=args.sample_every,
        dead_zone_frac=args.dead_zone,
        side_cushion_frac=args.side_cushion,
    )
    engine = EnterpriseSmartReframer(cfg)

    try:
        from tqdm import tqdm
        pbar = {"bar": None}

        def progress(done, total):
            if pbar["bar"] is None:
                pbar["bar"] = tqdm(total=total or None, unit="frame", desc="Rendering")
            pbar["bar"].update(1)

        result = engine.render_video(args.input, args.output, on_progress=progress)
        if pbar["bar"] is not None:
            pbar["bar"].close()
    except ImportError:
        # tqdm not installed — fall back to a plain periodic print.
        last_print = [0.0]

        def progress(done, total):
            now = time.time()
            if now - last_print[0] > 1.0 or done == total:
                pct = (done / total * 100.0) if total else 0.0
                print(f"\rRendering... {done}/{total or '?'} ({pct:5.1f}%)", end="", flush=True)
                last_print[0] = now

        result = engine.render_video(args.input, args.output, on_progress=progress)
        print()

    print("Done.")
    print(f"  Output:            {result['output_path']}")
    print(f"  Frames rendered:   {result['frames_rendered']}")
    print(f"  Duration:          {result['duration']:.2f}s")
    print(f"  Faces found:       {result['faces_found']}")
    print(f"  Detector backend:  {result.get('detector_backend')}")
    print(f"  YOLO person layer: {'active' if result.get('person_detector_available') else 'unavailable'}")
    print(f"  Scene cuts:        {result.get('scene_cuts_detected')}")
    print(f"  Speaker swaps:     {result.get('speaker_swaps')}")
    zt = result.get("zoom_track") or []
    group_frames = sum(1 for _, z in zt if z < 0.999)
    print(f"  Group two-shot frames: {group_frames}/{len(zt)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())