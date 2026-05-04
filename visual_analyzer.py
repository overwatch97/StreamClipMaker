import os
from pathlib import Path

import cv2
import numpy as np

from multimodal_utils import FFmpegFrameExtractor, ProgressReporter, aggregate_window_series, clip01, get_window_bounds, summarize_percentiles

_YOLO_MODEL = None

def require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics YOLO is required for multimodal visual scoring.") from exc
    return YOLO

def get_bytetrack_config():
    import ultralytics
    root = Path(ultralytics.__file__).resolve().parent
    tracker_path = root / "cfg" / "trackers" / "bytetrack.yaml"
    if not tracker_path.exists():
        raise RuntimeError(f"ByteTrack config not found at {tracker_path}")
    return str(tracker_path)

def get_yolo_model(model_name="yolo11n.pt"):
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        YOLO = require_ultralytics()
        _YOLO_MODEL = YOLO(model_name)
    return _YOLO_MODEL

def _track_frame(model, frame, tracker_path, device):
    kwargs = {
        "source": frame,
        "persist": True,
        "tracker": tracker_path,
        "verbose": False,
    }
    if device:
        kwargs["device"] = "cuda:0" if device == "gpu" else "cpu"
    try:
        return model.track(**kwargs)
    except TypeError:
        kwargs.pop("device", None)
        return model.track(**kwargs)

# ── Semantic Mapping ─────────────────────────────────────────────────────────

NAME_SEMANTIC_MAP = {
    "weapon": "combat", "gun": "combat", "knife": "combat", "sword": "combat", "enemy": "combat",
    "horse": "travel", "car": "travel", "vehicle": "travel", "bicycle": "travel", "motorcycle": "travel",
    "airplane": "travel", "bus": "travel", "train": "travel", "truck": "travel", "boat": "travel",
    "person": "interaction", "npc": "interaction"
}

def _map_scene_type(boxes, names_dict):
    if boxes is None or len(boxes) == 0:
        return "neutral", 0.0, 0.0
    
    cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
    
    best_scene = "neutral"
    best_conf = 0.0
    
    for cls_id, conf in zip(cls_ids, confs):
        name = names_dict.get(cls_id, "").lower()
        semantic = NAME_SEMANTIC_MAP.get(name, "neutral")
            
        if semantic != "neutral" and conf > best_conf:
            best_scene = semantic
            best_conf = float(conf)
            
    activity_level = float(len(boxes))
    return best_scene, best_conf, activity_level

# ── Signal Processing ────────────────────────────────────────────────────────

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

def extract_visual_feature_series(video_path, sample_fps=6.0, *, device="cpu", strict_device=False):
    tracker_path = get_bytetrack_config()
    model = get_yolo_model()
    names_dict = model.names
    
    requested_device = str(device)
    actual_device = requested_device
    fallback_error = None

    extractor = FFmpegFrameExtractor(video_path, sample_fps, target_width=320, device=actual_device)
    duration = float(extractor.info.get("duration") or 0)
    total_samples = int(duration * sample_fps) if duration > 0 else 0

    print(f"    Visual Analysis: Decoding via GPU ({actual_device}) + Analyzing {total_samples} samples...", flush=True)

    times = []
    raw_motion_abs = []
    scene_confidences = []
    activity_levels = []
    scene_types = []

    prev_gray = None
    sample_index = 0
    reporter = ProgressReporter(total_samples, label="Visual")

    batch_size = 16
    batch_frames = []
    batch_data = [] 

    def flush_batch():
        if not batch_frames:
            return
        
        try:
            results = _track_frame(model, batch_frames, tracker_path, actual_device)
        except Exception as exc:
            if actual_device == "gpu" and not strict_device:
                results = _track_frame(model, batch_frames, tracker_path, "cpu")
            else:
                raise
        
        for i, result in enumerate(results):
            current_time, motion_abs = batch_data[i]
            boxes = getattr(result, "boxes", None)
            
            scene_type, scene_conf, activity = _map_scene_type(boxes, names_dict)
            
            times.append(float(current_time))
            raw_motion_abs.append(motion_abs)
            scene_types.append(scene_type)
            scene_confidences.append(scene_conf)
            activity_levels.append(activity)

        batch_frames.clear()
        batch_data.clear()

    for analysis_frame in extractor:
        sample_index += 1
        reporter.update()
        reporter.report()

        current_time = sample_index / sample_fps
        gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)

        motion_abs = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_abs = float(np.mean(diff))

        batch_frames.append(analysis_frame)
        batch_data.append((current_time, motion_abs))

        if len(batch_frames) >= batch_size:
            flush_batch()

        prev_gray = gray

    flush_batch()
    extractor.close()

    # Calculate Motion Delta (Change in Motion)
    raw_motion_abs = np.array(raw_motion_abs)
    motion_delta = np.abs(np.diff(raw_motion_abs, prepend=raw_motion_abs[0]))

    # Temporal Smoothing
    smooth_motion_delta = moving_average(motion_delta, window_size=5)
    smooth_scene_conf = moving_average(scene_confidences, window_size=5)
    smooth_activity = moving_average(activity_levels, window_size=5)

    # Per-Video Normalization
    norm_motion_delta = min_max_normalize(smooth_motion_delta)
    norm_scene_conf = min_max_normalize(smooth_scene_conf)
    norm_activity = min_max_normalize(smooth_activity)

    return {
        "times": np.asarray(times, dtype=float),
        "motion_delta_norm": norm_motion_delta,
        "scene_confidence": norm_scene_conf,
        "activity_level": norm_activity,
        "scene_types": scene_types,
        "metadata": {
            "sample_fps": float(sample_fps),
            "requested_device": requested_device,
            "actual_device": actual_device,
            "fallback_error": fallback_error,
        },
    }

def score_visual_windows(visual_bundle, windows):
    results = []
    times = visual_bundle.get("times", [])
    scene_types = visual_bundle.get("scene_types", [])

    for window in windows:
        start_time, end_time = get_window_bounds(window)
        
        # We need to find the dominant scene type in the window
        window_indices = np.where((times >= start_time) & (times <= end_time))[0]
        dominant_scene = "neutral"
        if len(window_indices) > 0:
            window_scenes = [scene_types[i] for i in window_indices]
            # Get most frequent scene that is not 'neutral'
            non_neutral = [s for s in window_scenes if s != "neutral"]
            if non_neutral:
                from collections import Counter
                dominant_scene = Counter(non_neutral).most_common(1)[0][0]

        motion_delta = aggregate_window_series(
            times, visual_bundle.get("motion_delta_norm", []), start_time, end_time, reducer="quantile", quantile=0.75
        )
        scene_conf = aggregate_window_series(
            times, visual_bundle.get("scene_confidence", []), start_time, end_time, reducer="quantile", quantile=0.75
        )
        activity = aggregate_window_series(
            times, visual_bundle.get("activity_level", []), start_time, end_time, reducer="mean"
        )

        visual_score = clip01((0.5 * motion_delta) + (0.3 * scene_conf) + (0.2 * activity))

        results.append(
            {
                "score": visual_score,
                "features": {
                    "motion_delta_norm": motion_delta,
                    "scene_confidence": scene_conf,
                    "activity_level": activity,
                    "scene_type": dominant_scene
                },
            }
        )

    return results, visual_bundle.get("metadata", {})
