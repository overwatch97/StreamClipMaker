import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from audio_analyzer import extract_audio_feature_series, score_audio_windows
from emotion_analyzer import extract_facial_emotion_series, score_emotion_windows
from hardware import detect_capabilities, plan_hardware, record_stage_metric
from multimodal_utils import clip01, infer_stream_duration
from speech_analyzer import analyze_speech_windows
from visual_analyzer import extract_visual_feature_series, score_visual_windows


@dataclass(frozen=True)
class SegmentWindow:
    index: int
    start_time: float
    end_time: float

    @property
    def peak_time(self):
        return (self.start_time + self.end_time) / 2.0


@dataclass
class SegmentScores:
    speech: float = 0.0
    audio: float = 0.0
    visual: float = 0.0
    emotion: float = 0.0
    momentum: float = 0.0
    total: float = 0.0

    def as_dict(self, digits=4):
        return {
            "speech": round(float(self.speech), digits),
            "audio": round(float(self.audio), digits),
            "visual": round(float(self.visual), digits),
            "emotion": round(float(self.emotion), digits),
            "momentum": round(float(self.momentum), digits),
            "total": round(float(self.total), digits),
        }


@dataclass
class SegmentResult:
    window: SegmentWindow
    text: str
    scores: SegmentScores
    speech_features: dict = field(default_factory=dict)
    audio_features: dict = field(default_factory=dict)
    visual_features: dict = field(default_factory=dict)
    emotion_features: dict = field(default_factory=dict)

    def to_window_json(self):
        return {
            "start_time": round(float(self.window.start_time), 4),
            "end_time": round(float(self.window.end_time), 4),
            "scores": self.scores.as_dict(),
        }


def dominant_modality(scores):
    modalities = [
        ("speech", float(scores.speech)),
        ("audio", float(scores.audio)),
        ("visual", float(scores.visual)),
        ("emotion", float(scores.emotion)),
    ]
    modalities.sort(key=lambda item: item[1], reverse=True)
    if len(modalities) > 1 and (modalities[0][1] - modalities[1][1]) <= 0.05:
        return "mixed"
    return modalities[0][0]


def build_reason(scores):
    descriptors = {
        "speech": "excited speech",
        "audio": "a strong audio spike",
        "visual": "fast visual motion",
        "emotion": "a clear reaction beat",
    }
    ranked = sorted(
        [
            ("speech", float(scores.speech)),
            ("audio", float(scores.audio)),
            ("visual", float(scores.visual)),
            ("emotion", float(scores.emotion)),
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    primary, secondary = ranked[0], ranked[1]
    parts = [descriptors[primary[0]]]
    if secondary[1] >= 0.45:
        parts.append(descriptors[secondary[0]])

    reason = "Driven by " + " and ".join(parts)
    if scores.momentum >= 0.5:
        reason += "; the window builds into an even bigger peak"
    return reason


def compute_momentum_scores(base_scores):
    momentum_scores = []
    for idx, base_score in enumerate(base_scores):
        future_scores = base_scores[idx + 1 : idx + 4]
        future_peak = max(future_scores) if future_scores else base_score
        momentum = clip01((future_peak - base_score) / 0.25)
        momentum_scores.append(momentum)
    return momentum_scores


def build_segment_results(windows, speech_rows, audio_rows, visual_rows, emotion_rows):
    base_scores = []
    for index in range(len(windows)):
        base_score = clip01(
            (0.25 * speech_rows[index]["score"])
            + (0.20 * audio_rows[index]["score"])
            + (0.20 * visual_rows[index]["score"])
            + (0.25 * emotion_rows[index]["score"])
        )
        base_scores.append(base_score)

    momentum_scores = compute_momentum_scores(base_scores)

    results = []
    for index, window in enumerate(windows):
        total_score = clip01(base_scores[index] + (0.10 * momentum_scores[index]))
        segment_scores = SegmentScores(
            speech=speech_rows[index]["score"],
            audio=audio_rows[index]["score"],
            visual=visual_rows[index]["score"],
            emotion=emotion_rows[index]["score"],
            momentum=momentum_scores[index],
            total=total_score,
        )
        results.append(
            SegmentResult(
                window=window,
                text=speech_rows[index]["text"],
                scores=segment_scores,
                speech_features=speech_rows[index]["features"],
                audio_features=audio_rows[index]["features"],
                visual_features=visual_rows[index]["features"],
                emotion_features=emotion_rows[index]["features"],
            )
        )
    return results


def _input_minutes(duration_secs):
    return max(float(duration_secs or 0.0) / 60.0, 1e-6)


def _record_device_result(stage, requested_device, actual_device, elapsed_seconds, duration_secs, metadata, hardware_profile):
    fallback_error = metadata.get("fallback_error")
    if requested_device == "gpu" and actual_device != "gpu" and fallback_error:
        record_stage_metric(
            stage,
            "gpu",
            _input_minutes(duration_secs),
            elapsed_seconds,
            success=False,
            error=fallback_error,
            profile_path=hardware_profile,
        )

    record_stage_metric(
        stage,
        actual_device,
        _input_minutes(duration_secs),
        elapsed_seconds,
        success=True,
        profile_path=hardware_profile,
    )


def _run_visual_stage(video_path, device, strict_device, duration_secs, hardware_profile):
    started = time.perf_counter()
    bundle = extract_visual_feature_series(
        video_path,
        device=device,
        strict_device=strict_device,
    )
    elapsed = time.perf_counter() - started
    metadata = bundle.get("metadata", {})
    requested_device = metadata.get("requested_device", device)
    actual_device = metadata.get("actual_device", requested_device)
    _record_device_result("visual", requested_device, actual_device, elapsed, duration_secs, metadata, hardware_profile)
    return bundle, elapsed


def _run_emotion_stage(video_path, facecam_path, device, strict_device, duration_secs, hardware_profile):
    started = time.perf_counter()
    bundle = extract_facial_emotion_series(
        video_path,
        facecam_path=facecam_path,
        device=device,
        strict_device=strict_device,
    )
    elapsed = time.perf_counter() - started
    metadata = bundle.get("metadata", {})
    requested_device = metadata.get("requested_device", device)
    actual_device = metadata.get("actual_device", requested_device)
    _record_device_result("emotion", requested_device, actual_device, elapsed, duration_secs, metadata, hardware_profile)
    return bundle, elapsed


def run_multimodal_scoring(
    video_path,
    audio_path,
    transcript_data,
    windows,
    facecam_path=None,
    *,
    hardware_mode="auto",
    visual_device="auto",
    emotion_device="auto",
    hardware_profile=None,
    hardware_plan=None,
    capabilities=None,
):
    capabilities = capabilities or detect_capabilities()
    hardware_plan = hardware_plan or plan_hardware(
        hardware_mode=hardware_mode,
        stage_overrides={"visual": visual_device, "emotion": emotion_device},
        capabilities=capabilities,
        profile_path=hardware_profile,
        stages=("visual", "emotion"),
    )

    stream_duration = infer_stream_duration(video_path=video_path, transcript_data=transcript_data)
    visual_policy = hardware_plan.stage_device("visual")
    emotion_policy = hardware_plan.stage_device("emotion")
    visual_strict = hardware_plan.stage_strict("visual")
    emotion_strict = hardware_plan.stage_strict("emotion")

    with ThreadPoolExecutor(max_workers=4) as executor:
        speech_future = executor.submit(analyze_speech_windows, transcript_data, windows)
        audio_future = executor.submit(extract_audio_feature_series, audio_path)
        visual_future = executor.submit(
            _run_visual_stage,
            video_path,
            visual_policy,
            visual_strict,
            stream_duration,
            hardware_profile,
        )
        emotion_future = executor.submit(
            _run_emotion_stage,
            video_path,
            facecam_path,
            emotion_policy,
            emotion_strict,
            stream_duration,
            hardware_profile,
        )

        speech_rows, speech_meta = speech_future.result()
        audio_bundle = audio_future.result()
        audio_rows, audio_meta = score_audio_windows(audio_bundle, windows)
        visual_bundle, visual_elapsed = visual_future.result()
        visual_rows, visual_meta = score_visual_windows(visual_bundle, windows)
        face_bundle, emotion_elapsed = emotion_future.result()

    emotion_rows, emotion_meta = score_emotion_windows(face_bundle, audio_bundle, windows)

    segment_results = build_segment_results(windows, speech_rows, audio_rows, visual_rows, emotion_rows)
    metadata = {
        "duration_secs": infer_stream_duration(
            video_path=video_path,
            audio_duration=audio_bundle.get("duration"),
            transcript_data=transcript_data,
        ),
        "audio": audio_meta,
        "visual": {
            **visual_meta,
            "elapsed_seconds": float(visual_elapsed),
            "planned_device": visual_policy,
            "strict_device": bool(visual_strict),
        },
        "emotion": {
            **emotion_meta,
            "elapsed_seconds": float(emotion_elapsed),
            "planned_device": emotion_policy,
            "strict_device": bool(emotion_strict),
        },
        "speech": speech_meta,
        "hardware": {
            "visual": {
                "planned_device": visual_policy,
                "actual_device": visual_meta.get("actual_device", visual_policy),
                "strict_device": bool(visual_strict),
            },
            "emotion": {
                "planned_device": emotion_policy,
                "actual_device": emotion_meta.get("actual_device", emotion_policy),
                "strict_device": bool(emotion_strict),
            },
        },
    }
    return segment_results, metadata
