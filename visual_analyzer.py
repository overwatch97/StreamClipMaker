from pathlib import Path

import cv2
import numpy as np

from multimodal_utils import FFmpegFrameExtractor, ProgressReporter, aggregate_window_series, clip01, get_window_bounds, robust_normalize, summarize_percentiles


_YOLO_MODEL = None


def require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics YOLO is required for multimodal visual scoring. Install it in the StreamClipMaker venv."
        ) from exc
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


def _resize_for_analysis(frame, width=320):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    new_height = int(h * (width / w))
    return cv2.resize(frame, (width, new_height))


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


def extract_visual_feature_series(video_path, sample_fps=6.0, *, device="cpu", strict_device=False):
    """
    Runs a lightweight frame sampler over gameplay and tracks objects with YOLO11 + ByteTrack.
    We keep only compact per-frame signals so later window scoring stays cheap.
    """
    tracker_path = get_bytetrack_config()
    model = get_yolo_model()
    requested_device = str(device)
    actual_device = requested_device
    fallback_error = None

    extractor = FFmpegFrameExtractor(video_path, sample_fps, target_width=320, device=actual_device)
    duration = float(extractor.info.get("duration") or 0)
    total_samples = int(duration * sample_fps) if duration > 0 else 0

    print(f"    Visual Analysis: Decoding via GPU ({actual_device}) + Analyzing {total_samples} samples...", flush=True)

    times = []
    motion_values = []
    scene_values = []
    speed_values = []
    confidence_values = []

    prev_gray = None
    prev_hist = None
    prev_centers = {}
    sample_index = 0

    reporter = ProgressReporter(total_samples, label="Visual")

    batch_size = 16
    batch_frames = []
    batch_data = [] # Stores (current_time, motion_score, scene_score, hist) for each frame in the batch

    def flush_batch():
        nonlocal prev_centers
        if not batch_frames:
            return
        
        try:
            results = _track_frame(model, batch_frames, tracker_path, actual_device)
        except Exception as exc:
            if actual_device == "gpu" and not strict_device:
                # We don't want to fail the whole pipe, but fallback is tricky for a batch.
                # For simplicity, we just try to rerun the whole batch on CPU.
                results = _track_frame(model, batch_frames, tracker_path, "cpu")
            else:
                raise
        
        for i, result in enumerate(results):
            analysis_frame = batch_frames[i]
            current_time, motion_score, scene_score, current_hist = batch_data[i]
            
            boxes = getattr(result, "boxes", None)
            mean_confidence = 0.0
            subject_speed = 0.0
            current_centers = {}

            if boxes is not None and len(boxes) > 0:
                confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.asarray([], dtype=float)
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.asarray([], dtype=int)
                coords = boxes.xyxy.cpu().numpy()
                mean_confidence = float(np.mean(confidences)) if confidences.size else 0.0

                height, width = analysis_frame.shape[:2]
                diagonal = max(np.hypot(width, height), 1.0)
                delta_t = max(1.0 / sample_fps, 1e-6)

                for idx, box in enumerate(coords):
                    center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=float)
                    track_id = int(ids[idx]) if idx < len(ids) else idx
                    current_centers[track_id] = center
                    if track_id in prev_centers:
                        pixel_speed = np.linalg.norm(center - prev_centers[track_id]) / delta_t
                        subject_speed = max(subject_speed, float(pixel_speed / diagonal))
            
            times.append(float(current_time))
            motion_values.append(motion_score)
            scene_values.append(scene_score)
            speed_values.append(subject_speed)
            confidence_values.append(mean_confidence)
            prev_centers = current_centers

        batch_frames.clear()
        batch_data.clear()

    for analysis_frame in extractor:
        sample_index += 1
        reporter.update()
        reporter.report()

        current_time = sample_index / sample_fps
        gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)

        motion_score = 0.0
        scene_score = 0.0
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_gray is not None:
            # OPTIMIZATION: Use absdiff for motion instead of Farneback CPU bottleneck
            # Farneback is thousands of times slower than simple absdiff for basic motion intensity.
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = float(np.mean(diff))
            
            frame_delta = float(np.mean(diff) / 255.0)
            hist_corr = cv2.compareHist(prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_CORREL)
            hist_delta = float(1.0 - max(-1.0, min(1.0, hist_corr)))
            scene_score = (0.5 * frame_delta) + (0.5 * hist_delta)

        batch_frames.append(analysis_frame)
        batch_data.append((current_time, motion_score, scene_score, hist))

        if len(batch_frames) >= batch_size:
            flush_batch()

        prev_gray = gray
        prev_hist = hist

    flush_batch()

    extractor.close()

    return {
        "times": np.asarray(times, dtype=float),
        "motion_norm": robust_normalize(motion_values, lower_percentile=50.0, upper_percentile=97.0),
        "scene_change_norm": robust_normalize(scene_values, lower_percentile=50.0, upper_percentile=97.0),
        "subject_speed_norm": robust_normalize(speed_values, lower_percentile=50.0, upper_percentile=97.0),
        "tracking_confidence_norm": robust_normalize(confidence_values, lower_percentile=30.0, upper_percentile=95.0),
        "metadata": {
            "sample_fps": float(sample_fps),
            "motion_percentiles": summarize_percentiles(motion_values),
            "scene_change_percentiles": summarize_percentiles(scene_values),
            "subject_speed_percentiles": summarize_percentiles(speed_values),
            "tracking_confidence_percentiles": summarize_percentiles(confidence_values),
            "requested_device": requested_device,
            "actual_device": actual_device,
            "fallback_error": fallback_error,
        },
    }


def score_visual_windows(visual_bundle, windows):
    results = []
    times = visual_bundle.get("times", [])

    for window in windows:
        start_time, end_time = get_window_bounds(window)
        motion_intensity = aggregate_window_series(
            times,
            visual_bundle.get("motion_norm", []),
            start_time,
            end_time,
            reducer="quantile",
            quantile=0.75,
        )
        scene_change = aggregate_window_series(
            times,
            visual_bundle.get("scene_change_norm", []),
            start_time,
            end_time,
            reducer="quantile",
            quantile=0.75,
        )
        subject_speed = aggregate_window_series(
            times,
            visual_bundle.get("subject_speed_norm", []),
            start_time,
            end_time,
            reducer="quantile",
            quantile=0.75,
        )
        tracking_confidence = aggregate_window_series(
            times,
            visual_bundle.get("tracking_confidence_norm", []),
            start_time,
            end_time,
            reducer="mean",
        )

        visual_score = clip01(
            (0.35 * motion_intensity)
            + (0.25 * scene_change)
            + (0.25 * subject_speed)
            + (0.15 * tracking_confidence)
        )

        results.append(
            {
                "score": visual_score,
                "features": {
                    "motion_intensity": motion_intensity,
                    "scene_change": scene_change,
                    "subject_speed": subject_speed,
                    "tracking_confidence": tracking_confidence,
                },
            }
        )

    return results, visual_bundle.get("metadata", {})
