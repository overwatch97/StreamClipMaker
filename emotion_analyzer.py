import os

import cv2
import numpy as np

from multimodal_utils import FFmpegFrameExtractor, ProgressReporter, aggregate_window_series, clip01, get_window_bounds, robust_normalize, summarize_percentiles


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EMOTION_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "emotion-ferplus-8.onnx")
EMOTION_LABELS = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)

_FACE_DETECTOR = None
_SESSION_CACHE = {}


def require_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for multimodal emotion scoring.") from exc
    return ort


def get_face_detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            raise RuntimeError(f"OpenCV face cascade not found at {cascade_path}")
        _FACE_DETECTOR = detector
    return _FACE_DETECTOR


def _providers_for_device(device):
    if device == "gpu":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def get_emotion_session(device="cpu", model_path=DEFAULT_EMOTION_MODEL_PATH):
    ort = require_onnxruntime()
    resolved_model_path = os.path.abspath(model_path)
    if not os.path.exists(resolved_model_path):
        raise RuntimeError(f"Emotion ONNX model not found: {resolved_model_path}")

    cache_key = (resolved_model_path, str(device))
    if cache_key not in _SESSION_CACHE:
        session = ort.InferenceSession(resolved_model_path, providers=_providers_for_device(device))
        _SESSION_CACHE[cache_key] = session
    return _SESSION_CACHE[cache_key]


def _crop_combined_facecam(frame):
    """
    Reuses the app's existing assumption that a combined facecam often lives in the lower-right corner.
    This fallback is less reliable than a dedicated facecam input, so later scoring applies a confidence penalty.
    """
    height, width = frame.shape[:2]
    x0 = int(width * 0.75)
    y0 = int(height * 0.66)
    return frame[y0:height, x0:width]


def _largest_face_region(detector, gray_frame):
    boxes = detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
    if boxes is None or len(boxes) == 0:
        return None
    x, y, w, h = max(boxes, key=lambda item: item[2] * item[3])
    return int(x), int(y), int(w), int(h)


def _softmax(scores):
    scores = np.asarray(scores, dtype=np.float32)
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    denom = float(np.sum(exp_scores))
    if denom <= 0.0:
        return np.zeros_like(scores, dtype=np.float32)
    return exp_scores / denom


def _score_face_batch(session, regions):
    if not regions:
        return []
        
    input_name = session.get_inputs()[0].name
    results = []

    for res in regions:
        # Preprocess single region
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        input_tensor = resized.astype(np.float32)[np.newaxis, np.newaxis, :, :] # Shape [1, 1, 64, 64]
        
        # Inference with fixed batch size [1, 1, 64, 64]
        outputs = session.run(None, {input_name: input_tensor})
        
        probabilities = _softmax(outputs[0][0])
        emotion_scores = dict(zip(EMOTION_LABELS, probabilities.tolist()))
        score = max(float(emotion_scores.get("happiness", 0.0)), float(emotion_scores.get("surprise", 0.0)))
        results.append(score)
        
    return results


def _score_face_frame(session, frame):
    # Compatibility fallback for single frame
    return _score_face_batch(session, [frame])[0]


def extract_facial_emotion_series(
    video_path,
    facecam_path=None,
    sample_fps=2.0,
    *,
    device="cpu",
    strict_device=False,
    model_path=DEFAULT_EMOTION_MODEL_PATH,
):
    source_path = facecam_path if facecam_path else video_path
    fallback_crop = facecam_path is None
    reliability_multiplier = 0.6 if fallback_crop else 1.0

    requested_device = str(device)
    actual_device = requested_device
    fallback_error = None

    try:
        session = get_emotion_session(device=requested_device, model_path=model_path)
    except Exception as exc:
        if requested_device == "gpu" and not strict_device:
            fallback_error = str(exc)
            actual_device = "cpu"
            session = get_emotion_session(device="cpu", model_path=model_path)
        else:
            raise

    detector = get_face_detector()
    extractor = FFmpegFrameExtractor(source_path, sample_fps, target_width=640 if not fallback_crop else 1280, device=actual_device)
    duration = float(extractor.info.get("duration") or 0)
    total_samples = int(duration * sample_fps) if duration > 0 else 0

    print(f"    Emotion Analysis: Decoding via GPU ({actual_device}) + Analyzing {total_samples} samples...", flush=True)

    times = []
    facial_values = []
    sample_index = 0
    
    batch_frames = []
    batch_times = []
    batch_size = 32

    reporter = ProgressReporter(total_samples, label="Emotion")

    for frame in extractor:
        sample_index += 1
        reporter.update()
        reporter.report()

        sample_time = sample_index / sample_fps

        analysis_frame = _crop_combined_facecam(frame) if fallback_crop else frame
        if analysis_frame.size == 0:
            times.append(float(sample_time))
            facial_values.append(0.0)
            continue

        if analysis_frame.shape[1] > 320:
            scale = 320.0 / analysis_frame.shape[1]
            analysis_frame = cv2.resize(
                analysis_frame,
                (320, max(1, int(analysis_frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
        face_box = _largest_face_region(detector, gray)

        if face_box is not None:
            x, y, w, h = face_box
            face_region = analysis_frame[y : y + h, x : x + w]
            batch_frames.append(face_region)
            batch_times.append(float(sample_time))
        else:
            times.append(float(sample_time))
            facial_values.append(0.0)

        if len(batch_frames) >= batch_size:
            try:
                batch_results = _score_face_batch(session, batch_frames)
                for t, s in zip(batch_times, batch_results):
                    times.append(t)
                    facial_values.append(clip01(s * reliability_multiplier))
            except Exception as exc:
                if actual_device == "gpu" and not strict_device:
                    fallback_error = str(exc)
                    actual_device = "cpu"
                    session = get_emotion_session(device="cpu", model_path=model_path)
                    batch_results = _score_face_batch(session, batch_frames)
                    for t, s in zip(batch_times, batch_results):
                        times.append(t)
                        facial_values.append(clip01(s * reliability_multiplier))
                else:
                    extractor.close()
                    raise
            batch_frames = []
            batch_times = []

    # Final batch
    if batch_frames:
        batch_results = _score_face_batch(session, batch_frames)
        for t, s in zip(batch_times, batch_results):
            times.append(t)
            facial_values.append(clip01(s * reliability_multiplier))

    extractor.close()

    providers = []
    if hasattr(session, "get_providers"):
        providers = list(session.get_providers())

    return {
        "times": np.asarray(times, dtype=float),
        "facial_expression_norm": robust_normalize(facial_values, lower_percentile=40.0, upper_percentile=95.0),
        "metadata": {
            "sample_fps": float(sample_fps),
            "fallback_crop": bool(fallback_crop),
            "reliability_multiplier": float(reliability_multiplier),
            "facial_expression_percentiles": summarize_percentiles(facial_values),
            "requested_device": requested_device,
            "actual_device": actual_device,
            "providers": providers,
            "fallback_error": fallback_error,
            "model_path": os.path.abspath(model_path),
        },
    }


def score_emotion_windows(face_bundle, audio_bundle, windows):
    results = []
    face_times = face_bundle.get("times", [])
    audio_times = audio_bundle.get("times", [])

    for window in windows:
        start_time, end_time = get_window_bounds(window)

        facial_expression = aggregate_window_series(
            face_times,
            face_bundle.get("facial_expression_norm", []),
            start_time,
            end_time,
            reducer="quantile",
            quantile=0.75,
        )
        pitch_arousal = aggregate_window_series(
            audio_times,
            audio_bundle.get("pitch_arousal_norm", []),
            start_time,
            end_time,
            reducer="quantile",
            quantile=0.75,
        )
        rms_lift = aggregate_window_series(
            audio_times,
            audio_bundle.get("rms_lift_norm", []),
            start_time,
            end_time,
            reducer="quantile",
            quantile=0.75,
        )

        vocal_arousal = clip01((0.55 * pitch_arousal) + (0.45 * rms_lift))
        emotion_score = clip01((0.60 * facial_expression) + (0.40 * vocal_arousal))

        results.append(
            {
                "score": emotion_score,
                "features": {
                    "facial_expression": facial_expression,
                    "vocal_arousal": vocal_arousal,
                    "pitch_arousal": pitch_arousal,
                    "rms_lift": rms_lift,
                },
            }
        )

    metadata = {
        **face_bundle.get("metadata", {}),
        "voice_feature_source": "audio_bundle",
    }
    return results, metadata
