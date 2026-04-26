import numpy as np

from multimodal_utils import aggregate_window_series, clip01, get_window_bounds, robust_normalize, summarize_percentiles


def require_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "librosa is required for multimodal audio scoring. Install it in the StreamClipMaker venv."
        ) from exc
    return librosa


def extract_audio_feature_series(audio_path):
    """
    Precomputes stream-level audio features once so window scoring only performs cheap aggregation.
    Features are normalized with robust percentiles to stay stable across very loud or very quiet streams.
    """
    librosa = require_librosa()

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    if y.size == 0:
        return {
            "times": np.asarray([], dtype=float),
            "rms_norm": np.asarray([], dtype=float),
            "sound_change_norm": np.asarray([], dtype=float),
            "pitch_variation_norm": np.asarray([], dtype=float),
            "pitch_arousal_norm": np.asarray([], dtype=float),
            "rms_lift_norm": np.asarray([], dtype=float),
            "duration": 0.0,
            "metadata": {},
        }

    frame_length = 2048
    hop_length = 512

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    pitch = librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    rms_delta = np.abs(np.diff(rms, prepend=rms[0]))
    sound_change = np.maximum(rms_delta, onset_strength)

    finite_pitch = np.isfinite(pitch) & (pitch > 0)
    if finite_pitch.any():
        pitch_filled = pitch.copy()
        pitch_filled[~finite_pitch] = float(np.median(pitch[finite_pitch]))
    else:
        pitch_filled = np.zeros_like(pitch, dtype=float)

    log_pitch = np.log1p(pitch_filled)
    pitch_variation = np.abs(np.diff(log_pitch, prepend=log_pitch[0]))
    pitch_baseline = float(np.median(pitch_filled[finite_pitch])) if finite_pitch.any() else 0.0
    pitch_arousal = np.maximum(0.0, pitch_filled - pitch_baseline)
    rms_baseline = float(np.median(rms))
    rms_lift = np.maximum(0.0, rms - rms_baseline)

    bundle = {
        "times": times,
        "rms_norm": robust_normalize(rms, lower_percentile=55.0, upper_percentile=97.0),
        "sound_change_norm": robust_normalize(sound_change, lower_percentile=50.0, upper_percentile=97.0),
        "pitch_variation_norm": robust_normalize(pitch_variation, lower_percentile=50.0, upper_percentile=97.0),
        "pitch_arousal_norm": robust_normalize(pitch_arousal, lower_percentile=50.0, upper_percentile=97.0),
        "rms_lift_norm": robust_normalize(rms_lift, lower_percentile=50.0, upper_percentile=97.0),
        "duration": float(len(y) / sr),
        "metadata": {
            "sample_rate": int(sr),
            "frame_length": int(frame_length),
            "hop_length": int(hop_length),
            "rms_percentiles": summarize_percentiles(rms),
            "sound_change_percentiles": summarize_percentiles(sound_change),
            "pitch_variation_percentiles": summarize_percentiles(pitch_variation),
        },
    }
    return bundle


def score_audio_windows(audio_bundle, windows):
    results = []
    times = audio_bundle.get("times", [])

    for window in windows:
        start_time, end_time = get_window_bounds(window)
        rms_spike = aggregate_window_series(
            times,
            audio_bundle.get("rms_norm", []),
            start_time,
            end_time,
            reducer="quantile",
            quantile=0.80,
        )
        sound_change = aggregate_window_series(
            times,
            audio_bundle.get("sound_change_norm", []),
            start_time,
            end_time,
            reducer="quantile",
            quantile=0.80,
        )
        pitch_variation = aggregate_window_series(
            times,
            audio_bundle.get("pitch_variation_norm", []),
            start_time,
            end_time,
            reducer="quantile",
            quantile=0.75,
        )

        audio_score = clip01((0.45 * rms_spike) + (0.35 * sound_change) + (0.20 * pitch_variation))
        results.append(
            {
                "score": audio_score,
                "features": {
                    "rms_spike": rms_spike,
                    "sound_change": sound_change,
                    "pitch_variation": pitch_variation,
                },
            }
        )

    return results, audio_bundle.get("metadata", {})
