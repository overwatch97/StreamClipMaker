import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from multimodal_utils import FFmpegFrameExtractor, ProgressReporter, aggregate_window_series, clip01, get_window_bounds, summarize_percentiles

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DEFAULT_LANDMARKER_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

def get_mediapipe_model_path():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(DEFAULT_LANDMARKER_PATH):
        print(f"Downloading MediaPipe Face Landmarker model to {DEFAULT_LANDMARKER_PATH}...", flush=True)
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, DEFAULT_LANDMARKER_PATH)
    return DEFAULT_LANDMARKER_PATH

def require_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe is required for multimodal emotion scoring. Install it via 'pip install mediapipe'.") from exc
    return mp

def get_face_landmarker(device="cpu"):
    mp = require_mediapipe()
    model_path = get_mediapipe_model_path()
    
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # MediaPipe Python GPU delegate is often tricky on Windows without custom builds, 
    # but we pass it if requested and let it fallback internally.
    delegate = mp.tasks.BaseOptions.Delegate.GPU if device == "gpu" else mp.tasks.BaseOptions.Delegate.CPU
    
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path, delegate=delegate),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return FaceLandmarker.create_from_options(options), mp

def _crop_combined_facecam(frame):
    height, width = frame.shape[:2]
    x0 = int(width * 0.75)
    y0 = int(height * 0.66)
    return frame[y0:height, x0:width]

def _dist(p1, p2):
    return np.hypot(p1.x - p2.x, p1.y - p2.y)

def calculate_heuristics(landmarks):
    if not landmarks:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    lm = landmarks[0] # First face
    
    # Eye Aspect Ratio (EAR)
    l_v1 = _dist(lm[160], lm[144])
    l_v2 = _dist(lm[158], lm[153])
    l_h  = _dist(lm[33], lm[133])
    ear_left = (l_v1 + l_v2) / (2.0 * l_h + 1e-6)
    
    r_v1 = _dist(lm[385], lm[380])
    r_v2 = _dist(lm[387], lm[373])
    r_h  = _dist(lm[362], lm[263])
    ear_right = (r_v1 + r_v2) / (2.0 * r_h + 1e-6)
    ear = (ear_left + ear_right) / 2.0
    
    # Mouth Aspect Ratio (MAR)
    m_v = _dist(lm[13], lm[14])
    m_h = _dist(lm[78], lm[308])
    mar = m_v / (m_h + 1e-6)
    
    # Head Pose (simplified as distance of nose from center of face bounding box)
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    nose = lm[1]
    pose_mag = np.hypot(nose.x - cx, nose.y - cy)
    
    # Absolute center of nose for engagement delta
    nose_x = nose.x
    nose_y = nose.y
    
    return ear, mar, pose_mag, nose_x, nose_y

def moving_average(data, window_size=5):
    if len(data) < window_size:
        return np.array(data)
    padded = np.pad(data, (window_size//2, window_size - 1 - window_size//2), mode='edge')
    return np.convolve(padded, np.ones(window_size)/window_size, mode='valid')

def min_max_normalize(data):
    data_arr = np.array(data, dtype=float)
    if len(data_arr) == 0:
        return data_arr
    d_min = np.min(data_arr)
    d_max = np.max(data_arr)
    if d_max - d_min <= 1e-6:
        return np.zeros_like(data_arr)
    return (data_arr - d_min) / (d_max - d_min)

def extract_facial_emotion_series(
    video_path,
    facecam_path=None,
    sample_fps=2.0,
    *,
    device="cpu",
    strict_device=False,
):
    source_path = facecam_path if facecam_path else video_path
    fallback_crop = facecam_path is None

    requested_device = str(device)
    actual_device = requested_device
    fallback_error = None

    try:
        landmarker, mp = get_face_landmarker(device=requested_device)
    except Exception as exc:
        if requested_device == "gpu" and not strict_device:
            fallback_error = str(exc)
            actual_device = "cpu"
            landmarker, mp = get_face_landmarker(device="cpu")
        else:
            raise

    extractor = FFmpegFrameExtractor(source_path, sample_fps, target_width=640 if not fallback_crop else 1280, device=actual_device)
    duration = float(extractor.info.get("duration") or 0)
    total_samples = int(duration * sample_fps) if duration > 0 else 0

    print(f"    Emotion Analysis: Analyzing {total_samples} samples via MediaPipe...", flush=True)

    times = []
    raw_ears = []
    raw_mars = []
    nose_xs = []
    nose_ys = []
    
    reporter = ProgressReporter(total_samples, label="Emotion")
    sample_index = 0

    mp_image_class = mp.Image
    mp_image_format = mp.ImageFormat.SRGB

    for frame in extractor:
        sample_index += 1
        reporter.update()
        reporter.report()

        sample_time = sample_index / sample_fps
        times.append(float(sample_time))

        analysis_frame = _crop_combined_facecam(frame) if fallback_crop else frame
        if analysis_frame.size == 0:
            raw_ears.append(0.0)
            raw_mars.append(0.0)
            nose_xs.append(0.5)
            nose_ys.append(0.5)
            continue

        rgb_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image_class(image_format=mp_image_format, data=rgb_frame)
        
        try:
            result = landmarker.detect(mp_img)
            ear, mar, pose_mag, nx, ny = calculate_heuristics(result.face_landmarks)
            raw_ears.append(ear)
            raw_mars.append(mar)
            nose_xs.append(nx)
            nose_ys.append(ny)
        except Exception:
            raw_ears.append(0.0)
            raw_mars.append(0.0)
            nose_xs.append(0.5)
            nose_ys.append(0.5)

    extractor.close()

    # Engagement: delta of nose position
    nose_xs = np.array(nose_xs)
    nose_ys = np.array(nose_ys)
    dx = np.diff(nose_xs, prepend=nose_xs[0])
    dy = np.diff(nose_ys, prepend=nose_ys[0])
    raw_engagement = np.hypot(dx, dy)

    # 1. Temporal Smoothing
    smooth_ear = moving_average(raw_ears, window_size=3)
    smooth_mar = moving_average(raw_mars, window_size=3)
    smooth_engagement = moving_average(raw_engagement, window_size=3)

    # 2. Per-Video Normalization [0, 1]
    norm_ear = min_max_normalize(smooth_ear)
    norm_mar = min_max_normalize(smooth_mar)
    norm_eng = min_max_normalize(smooth_engagement)

    # 3. Derive overall emotion score (max of surprise or reaction intensity)
    emotion_score_norm = np.maximum(norm_ear, norm_mar)

    return {
        "times": np.asarray(times, dtype=float),
        "emotion_score_norm": emotion_score_norm,
        "surprise_level": norm_ear,
        "engagement_level": norm_eng,
        "reaction_level": norm_mar,
        "metadata": {
            "sample_fps": float(sample_fps),
            "fallback_crop": bool(fallback_crop),
            "requested_device": requested_device,
            "actual_device": actual_device,
            "fallback_error": fallback_error,
        },
    }

def score_emotion_windows(face_bundle, audio_bundle, windows):
    results = []
    face_times = face_bundle.get("times", [])
    
    for window in windows:
        start_time, end_time = get_window_bounds(window)

        emotion_score = aggregate_window_series(
            face_times,
            face_bundle.get("emotion_score_norm", []),
            start_time,
            end_time,
            reducer="quantile",
            quantile=0.75,
        )
        surprise = aggregate_window_series(
            face_times,
            face_bundle.get("surprise_level", []),
            start_time,
            end_time,
            reducer="quantile",
            quantile=0.75,
        )
        engagement = aggregate_window_series(
            face_times,
            face_bundle.get("engagement_level", []),
            start_time,
            end_time,
            reducer="quantile",
            quantile=0.75,
        )
        reaction = aggregate_window_series(
            face_times,
            face_bundle.get("reaction_level", []),
            start_time,
            end_time,
            reducer="quantile",
            quantile=0.75,
        )

        results.append(
            {
                "score": emotion_score, # Required by base pipeline
                "features": {
                    "surprise_level": surprise,
                    "engagement_level": engagement,
                    "reaction_level": reaction,
                },
            }
        )

    return results, face_bundle.get("metadata", {})
