import json
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

import audio_analyzer
import emotion_analyzer
import hardware
import llm_selector
import scoring_engine
import speech_analyzer
import visual_analyzer
from multimodal_utils import robust_normalize
from scoring_engine import SegmentResult, SegmentScores, SegmentWindow, compute_momentum_scores
from segment_ranker import select_top_highlights


def _build_transcript_segment(start, end, text, repeat=1):
    tokens = (text.split() or ["..."]) * max(1, repeat)
    duration = max(end - start, 1e-6)
    step = duration / len(tokens)
    words = []
    for index, token in enumerate(tokens):
        word_start = start + (index * step)
        word_end = min(end, word_start + step)
        words.append({"word": token, "start": word_start, "end": word_end})
    return {"start": start, "end": end, "text": " ".join(tokens), "words": words}


def _write_wav(path, samples, sample_rate=16000):
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _make_segment(start_time, end_time, total, *, speech=0.0, audio=0.0, visual=0.0, emotion=0.0, momentum=0.0, text=""):
    return SegmentResult(
        window=SegmentWindow(index=int(start_time * 10), start_time=float(start_time), end_time=float(end_time)),
        text=text or f"text {start_time}-{end_time}",
        scores=SegmentScores(
            speech=float(speech),
            audio=float(audio),
            visual=float(visual),
            emotion=float(emotion),
            momentum=float(momentum),
            total=float(total),
        ),
    )


class SpeechAnalyzerTests(unittest.TestCase):
    def test_transcript_scoring_prefers_hype_keywords_punctuation_and_rate_spikes(self):
        transcript = [
            _build_transcript_segment(0.0, 4.0, "just warming up", repeat=1),
            _build_transcript_segment(
                4.0,
                8.0,
                "OMG no way! let's go! clip that! holy bro wait what insane crazy!",
                repeat=2,
            ),
            _build_transcript_segment(8.0, 12.0, "cooling down", repeat=1),
        ]
        windows = [
            SegmentWindow(index=0, start_time=0.0, end_time=4.0),
            SegmentWindow(index=1, start_time=4.0, end_time=8.0),
            SegmentWindow(index=2, start_time=8.0, end_time=12.0),
        ]

        results, metadata = speech_analyzer.analyze_speech_windows(transcript, windows)

        self.assertEqual(len(results), 3)
        self.assertEqual(metadata["windows_scored"], 3)
        self.assertTrue(all(0.0 <= row["score"] <= 1.0 for row in results))
        self.assertGreater(results[1]["features"]["exclamation_density"], 0.0)
        self.assertGreater(results[1]["features"]["keyword_intensity"], results[0]["features"]["keyword_intensity"])
        self.assertGreater(results[1]["features"]["speech_rate_spike"], 0.0)
        self.assertGreater(results[1]["score"], results[0]["score"])
        self.assertGreater(results[1]["score"], results[2]["score"])


class AudioAnalyzerTests(unittest.TestCase):
    def test_audio_scoring_detects_spikes_sound_change_and_pitch_variation(self):
        bundle = {
            "times": np.asarray([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype=float),
            "rms_norm": np.asarray([0.05, 0.04, 0.05, 0.04, 0.85, 0.90, 0.88, 0.82], dtype=float),
            "sound_change_norm": np.asarray([0.02, 0.03, 0.02, 0.03, 0.70, 0.95, 0.65, 0.60], dtype=float),
            "pitch_variation_norm": np.asarray([0.01, 0.01, 0.02, 0.01, 0.75, 0.85, 0.80, 0.78], dtype=float),
            "duration": 8.0,
            "metadata": {
                "rms_percentiles": {"p50": 0.435, "p75": 0.835, "p95": 0.9},
                "sound_change_percentiles": {"p50": 0.315, "p75": 0.675, "p95": 0.95},
                "pitch_variation_percentiles": {"p50": 0.385, "p75": 0.785, "p95": 0.85},
            },
        }
        windows = [
            SegmentWindow(index=0, start_time=0.0, end_time=4.0),
            SegmentWindow(index=1, start_time=4.0, end_time=8.0),
        ]
        results, metadata = audio_analyzer.score_audio_windows(bundle, windows)

        self.assertEqual(bundle["duration"], 8.0)
        self.assertIn("rms_percentiles", metadata)
        self.assertTrue(all(0.0 <= row["score"] <= 1.0 for row in results))
        self.assertTrue(all(0.0 <= row["features"]["rms_spike"] <= 1.0 for row in results))
        self.assertTrue(all(0.0 <= row["features"]["sound_change"] <= 1.0 for row in results))
        self.assertTrue(all(0.0 <= row["features"]["pitch_variation"] <= 1.0 for row in results))
        self.assertGreater(results[1]["features"]["rms_spike"], results[0]["features"]["rms_spike"])
        self.assertGreater(results[1]["features"]["sound_change"], results[0]["features"]["sound_change"])
        self.assertGreater(results[1]["features"]["pitch_variation"], results[0]["features"]["pitch_variation"])
        self.assertGreater(results[1]["score"], results[0]["score"])


class FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class FakeBoxes:
    def __init__(self, xyxy, conf, ids):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.id = FakeTensor(ids)

    def __len__(self):
        return len(self.xyxy.numpy())


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeTrackerModel:
    def __init__(self, frames):
        self._frames = list(frames)
        self._index = 0

    def track(self, source, persist=True, tracker=None, verbose=False):
        boxes = self._frames[self._index]
        self._index += 1
        return [FakeResult(boxes)]


class FakeVideoCapture:
    def __init__(self, frames, fps):
        self._frames = [frame.copy() for frame in frames]
        self._fps = fps
        self._index = 0

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame.copy()

    def get(self, prop):
        if prop == visual_analyzer.cv2.CAP_PROP_FPS:
            return self._fps
        if prop == visual_analyzer.cv2.CAP_PROP_FRAME_COUNT:
            return len(self._frames)
        return 0

    def release(self):
        return None


class FakeFaceDetector:
    def __init__(self, boxes_per_frame):
        self._boxes = list(boxes_per_frame)
        self._index = 0

    def detectMultiScale(self, gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24)):
        if self._index >= len(self._boxes):
            return np.asarray([], dtype=np.int32)
        boxes = self._boxes[self._index]
        self._index += 1
        return np.asarray(boxes, dtype=np.int32)


class FakeEmotionInput:
    def __init__(self, name="input"):
        self.name = name


class FakeEmotionSession:
    def __init__(self, outputs, providers=None):
        self._outputs = list(outputs)
        self._index = 0
        self._providers = list(providers or ["CPUExecutionProvider"])

    def get_inputs(self):
        return [FakeEmotionInput("Input3")]

    def get_providers(self):
        return list(self._providers)

    def run(self, _outputs, _inputs):
        if self._index >= len(self._outputs):
            result = self._outputs[-1]
        else:
            result = self._outputs[self._index]
            self._index += 1
        if isinstance(result, Exception):
            raise result
        return [np.asarray([result], dtype=np.float32)]


class VisualAnalyzerTests(unittest.TestCase):
    def test_visual_scoring_uses_motion_scene_change_and_tracking_speed(self):
        frames = []
        frame0 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame0[8:18, 8:18] = 255
        frames.append(frame0)

        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame1[20:32, 18:30] = 255
        frames.append(frame1)

        frame2 = np.full((64, 64, 3), 90, dtype=np.uint8)
        frame2[10:26, 30:46] = 255
        frames.append(frame2)

        frame3 = np.full((64, 64, 3), 200, dtype=np.uint8)
        frame3[26:42, 36:52] = 20
        frames.append(frame3)

        tracked_boxes = [
            FakeBoxes([[8, 8, 24, 24]], [0.55], [1]),
            FakeBoxes([[20, 10, 36, 26]], [0.80], [1]),
            FakeBoxes([[32, 18, 48, 34]], [0.92], [1]),
            FakeBoxes([[40, 26, 56, 42]], [0.97], [1]),
        ]

        with mock.patch("visual_analyzer.cv2.VideoCapture", return_value=FakeVideoCapture(frames, fps=1.0)):
            with mock.patch("visual_analyzer.get_bytetrack_config", return_value="bytetrack.yaml"):
                with mock.patch("visual_analyzer.get_yolo_model", return_value=FakeTrackerModel(tracked_boxes)):
                    bundle = visual_analyzer.extract_visual_feature_series("synthetic.mp4", sample_fps=1.0)

        windows = [
            SegmentWindow(index=0, start_time=0.0, end_time=2.0),
            SegmentWindow(index=1, start_time=1.0, end_time=3.0),
        ]
        results, metadata = visual_analyzer.score_visual_windows(bundle, windows)

        self.assertEqual(metadata["sample_fps"], 1.0)
        self.assertEqual(len(bundle["times"]), 4)
        self.assertGreater(float(np.max(bundle["motion_norm"])), 0.0)
        self.assertGreater(float(np.max(bundle["subject_speed_norm"])), 0.0)
        self.assertGreater(float(np.max(bundle["tracking_confidence_norm"])), 0.0)
        self.assertGreater(metadata["scene_change_percentiles"]["p95"], 0.0)
        self.assertTrue(all(0.0 <= row["score"] <= 1.0 for row in results))
        self.assertGreater(max(row["features"]["motion_intensity"] for row in results), 0.0)
        self.assertGreater(max(row["features"]["scene_change"] for row in results), 0.0)
        self.assertGreater(max(row["features"]["subject_speed"] for row in results), 0.0)


class EmotionAnalyzerTests(unittest.TestCase):
    def _emotion_capture(self, frame_count=3):
        frames = [np.full((96, 96, 3), 32 + (index * 24), dtype=np.uint8) for index in range(frame_count)]
        return FakeVideoCapture(frames, fps=1.0)

    def test_facecam_and_fallback_metadata_are_reported(self):
        facecam_session = FakeEmotionSession(
            outputs=[
                [0.0, 3.5, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            providers=["CPUExecutionProvider"],
        )

        with mock.patch("emotion_analyzer.get_face_detector", return_value=FakeFaceDetector([[(8, 8, 40, 40)]] * 3)):
            with mock.patch("emotion_analyzer.get_emotion_session", return_value=facecam_session):
                with mock.patch("emotion_analyzer.cv2.VideoCapture", return_value=self._emotion_capture()):
                    facecam_bundle = emotion_analyzer.extract_facial_emotion_series(
                        "combined.mp4",
                        facecam_path="facecam.mp4",
                        sample_fps=1.0,
                    )

        fallback_session = FakeEmotionSession(
            outputs=[
                [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            providers=["CPUExecutionProvider"],
        )

        with mock.patch("emotion_analyzer.get_face_detector", return_value=FakeFaceDetector([[(8, 8, 40, 40)]] * 3)):
            with mock.patch("emotion_analyzer.get_emotion_session", return_value=fallback_session):
                with mock.patch("emotion_analyzer.cv2.VideoCapture", return_value=self._emotion_capture()):
                    fallback_bundle = emotion_analyzer.extract_facial_emotion_series(
                        "combined.mp4",
                        facecam_path=None,
                        sample_fps=1.0,
                    )

        self.assertFalse(facecam_bundle["metadata"]["fallback_crop"])
        self.assertEqual(facecam_bundle["metadata"]["reliability_multiplier"], 1.0)
        self.assertTrue(np.all(facecam_bundle["facial_expression_norm"] >= 0.0))
        self.assertTrue(np.all(facecam_bundle["facial_expression_norm"] <= 1.0))
        self.assertGreater(float(np.max(facecam_bundle["facial_expression_norm"])), 0.0)
        self.assertEqual(facecam_bundle["metadata"]["actual_device"], "cpu")

        self.assertTrue(fallback_bundle["metadata"]["fallback_crop"])
        self.assertEqual(fallback_bundle["metadata"]["reliability_multiplier"], 0.6)
        self.assertGreater(float(np.max(fallback_bundle["facial_expression_norm"])), 0.0)
        self.assertEqual(fallback_bundle["metadata"]["actual_device"], "cpu")

    def test_no_face_detected_stays_zero_and_voice_can_dominate(self):
        with mock.patch("emotion_analyzer.get_face_detector", return_value=FakeFaceDetector([[], []])):
            with mock.patch(
                "emotion_analyzer.get_emotion_session",
                return_value=FakeEmotionSession(outputs=[[0.0] * 8], providers=["CPUExecutionProvider"]),
            ):
                with mock.patch("emotion_analyzer.cv2.VideoCapture", return_value=self._emotion_capture(frame_count=2)):
                    face_bundle = emotion_analyzer.extract_facial_emotion_series(
                        "combined.mp4",
                        facecam_path=None,
                        sample_fps=1.0,
                    )

        self.assertTrue(np.allclose(face_bundle["facial_expression_norm"], 0.0))

        windows = [
            SegmentWindow(index=0, start_time=0.0, end_time=2.0),
            SegmentWindow(index=1, start_time=2.0, end_time=4.0),
        ]
        voiced_audio = {
            "times": np.asarray([0.5, 1.5, 2.5, 3.5], dtype=float),
            "pitch_arousal_norm": np.asarray([0.0, 0.0, 1.0, 1.0], dtype=float),
            "rms_lift_norm": np.asarray([0.0, 0.0, 1.0, 1.0], dtype=float),
        }

        results, metadata = emotion_analyzer.score_emotion_windows(face_bundle, voiced_audio, windows)

        self.assertEqual(metadata["voice_feature_source"], "audio_bundle")
        self.assertEqual(results[0]["features"]["facial_expression"], 0.0)
        self.assertEqual(results[1]["features"]["facial_expression"], 0.0)
        self.assertEqual(results[0]["score"], 0.0)
        self.assertGreater(results[1]["features"]["vocal_arousal"], 0.0)
        self.assertGreater(results[1]["score"], results[0]["score"])

    def test_gpu_provider_falls_back_to_cpu_when_session_setup_fails(self):
        detector = FakeFaceDetector([[(8, 8, 40, 40)]])
        cpu_session = FakeEmotionSession(outputs=[[0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        def fake_session(device="cpu", model_path=None):
            if device == "gpu":
                raise RuntimeError("CUDA provider unavailable")
            return cpu_session

        with mock.patch("emotion_analyzer.get_face_detector", return_value=detector):
            with mock.patch("emotion_analyzer.get_emotion_session", side_effect=fake_session):
                with mock.patch("emotion_analyzer.cv2.VideoCapture", return_value=self._emotion_capture(frame_count=1)):
                    bundle = emotion_analyzer.extract_facial_emotion_series(
                        "combined.mp4",
                        facecam_path="facecam.mp4",
                        sample_fps=1.0,
                        device="gpu",
                        strict_device=False,
                    )

        self.assertEqual(bundle["metadata"]["requested_device"], "gpu")
        self.assertEqual(bundle["metadata"]["actual_device"], "cpu")
        self.assertIn("CUDA provider unavailable", bundle["metadata"]["fallback_error"])


class RankingTests(unittest.TestCase):
    def test_flat_normalization_returns_zero_signal(self):
        normalized = robust_normalize([0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(normalized, 0.0))

    def test_momentum_scoring_looks_forward_three_windows(self):
        momentum = compute_momentum_scores([0.2, 0.5, 0.7, 0.4])
        self.assertEqual(len(momentum), 4)
        self.assertAlmostEqual(momentum[0], 1.0)
        self.assertAlmostEqual(momentum[1], 0.8)
        self.assertAlmostEqual(momentum[2], 0.0)
        self.assertAlmostEqual(momentum[3], 0.0)

    def test_ranker_preserves_diversity_and_merges_adjacent_peaks(self):
        transcript = [
            _build_transcript_segment(0.0, 4.0, "intro"),
            _build_transcript_segment(4.0, 8.0, "peak one"),
            _build_transcript_segment(5.5, 9.5, "peak two"),
            _build_transcript_segment(16.0, 20.0, "peak three"),
        ]
        segments = [
            _make_segment(0.0, 4.0, 0.25, speech=0.25, text="intro"),
            _make_segment(4.0, 8.0, 0.91, speech=0.91, text="peak one"),
            _make_segment(5.5, 9.5, 0.89, audio=0.89, text="peak two"),
            _make_segment(16.0, 20.0, 0.88, visual=0.88, text="peak three"),
        ]

        highlights = select_top_highlights(
            segment_results=segments,
            transcript_data=transcript,
            video_duration=24.0,
            top_k=3,
            min_separation_secs=1.0,
        )

        self.assertEqual(len(highlights), 2)
        merged = next(item for item in highlights if item["source_window_start"] == 4.0)
        self.assertEqual(merged["category"], "speech")
        self.assertLessEqual(merged["start"], 4.0)
        self.assertGreaterEqual(merged["end"], 9.5)
        self.assertIn("Driven by", merged["reason"])

    def test_ranker_keeps_spaced_out_top_peaks(self):
        transcript = [
            _build_transcript_segment(0.0, 4.0, "A"),
            _build_transcript_segment(1.0, 5.0, "B"),
            _build_transcript_segment(12.0, 16.0, "C"),
            _build_transcript_segment(24.0, 28.0, "D"),
        ]
        segments = [
            _make_segment(0.0, 4.0, 0.95, speech=0.95, text="A"),
            _make_segment(1.0, 5.0, 0.93, audio=0.93, text="B"),
            _make_segment(12.0, 16.0, 0.92, visual=0.92, text="C"),
            _make_segment(24.0, 28.0, 0.91, emotion=0.91, text="D"),
        ]

        highlights = select_top_highlights(
            segment_results=segments,
            transcript_data=transcript,
            video_duration=32.0,
            top_k=3,
            min_separation_secs=6.0,
        )

        peak_times = sorted(item["peak_time"] for item in highlights)
        self.assertEqual(len(highlights), 3)
        self.assertEqual(peak_times, [2.0, 14.0, 26.0])

    def test_highlight_regression_stays_close_to_legacy_ranges(self):
        transcript = []
        legacy_segments = []
        candidate_segments = []
        peak_starts = [0.0, 12.0, 24.0, 36.0, 48.0, 60.0]

        for index, start in enumerate(peak_starts):
            end = start + 4.0
            transcript.append(_build_transcript_segment(start, end, f"peak {index}", repeat=2))
            legacy_segments.append(
                _make_segment(
                    start,
                    end,
                    0.92 - (index * 0.01),
                    speech=0.55,
                    audio=0.40,
                    visual=0.45,
                    emotion=0.70 - (index * 0.02),
                    text=f"peak {index}",
                )
            )
            candidate_segments.append(
                _make_segment(
                    start,
                    end,
                    0.89 - (index * 0.01),
                    speech=0.55,
                    audio=0.40,
                    visual=0.45,
                    emotion=0.62 - (index * 0.02),
                    text=f"peak {index}",
                )
            )

        legacy = select_top_highlights(
            segment_results=legacy_segments,
            transcript_data=transcript,
            video_duration=72.0,
            top_k=6,
            min_separation_secs=6.0,
        )
        candidate = select_top_highlights(
            segment_results=candidate_segments,
            transcript_data=transcript,
            video_duration=72.0,
            top_k=6,
            min_separation_secs=6.0,
        )

        matches = 0
        for legacy_item in legacy:
            for candidate_item in candidate:
                if abs(legacy_item["start"] - candidate_item["start"]) <= 2.0 and abs(legacy_item["end"] - candidate_item["end"]) <= 2.0:
                    matches += 1
                    break

        self.assertGreaterEqual(matches, 5)


class SchedulingTests(unittest.TestCase):
    def test_gpu_visual_and_gpu_emotion_do_not_overlap(self):
        transcript = [_build_transcript_segment(0.0, 4.0, "one")]
        windows = [SegmentWindow(index=0, start_time=0.0, end_time=4.0)]
        call_log = []
        caps = hardware.HardwareCapabilities(
            torch_cuda_available=True,
            torch_device_name="Test GPU",
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            onnx_cuda_available=True,
        )

        def fake_visual(*args, **kwargs):
            call_log.append(("visual_start", time.perf_counter()))
            time.sleep(0.15)
            call_log.append(("visual_end", time.perf_counter()))
            return (
                {"times": np.asarray([0.5]), "motion_norm": np.asarray([0.1]), "scene_change_norm": np.asarray([0.1]), "subject_speed_norm": np.asarray([0.1]), "tracking_confidence_norm": np.asarray([0.1]), "metadata": {"actual_device": "gpu"}},
                0.15,
            )

        def fake_emotion(*args, **kwargs):
            call_log.append(("emotion_start", time.perf_counter()))
            time.sleep(0.15)
            call_log.append(("emotion_end", time.perf_counter()))
            return ({"times": np.asarray([0.5]), "facial_expression_norm": np.asarray([0.1]), "metadata": {"actual_device": "gpu"}}, 0.15)

        with mock.patch("scoring_engine.detect_capabilities", return_value=caps):
            with mock.patch("scoring_engine.extract_audio_feature_series", return_value={"times": np.asarray([0.5]), "duration": 4.0, "pitch_arousal_norm": np.asarray([0.1]), "rms_lift_norm": np.asarray([0.1]), "rms_norm": np.asarray([0.1]), "sound_change_norm": np.asarray([0.1]), "pitch_variation_norm": np.asarray([0.1]), "metadata": {}}):
                with mock.patch("scoring_engine.analyze_speech_windows", return_value=([{"text": "one", "score": 0.1, "features": {}}], {"windows_scored": 1})):
                    with mock.patch("scoring_engine.score_audio_windows", return_value=([{"score": 0.1, "features": {}}], {})):
                        with mock.patch("scoring_engine.score_visual_windows", return_value=([{"score": 0.1, "features": {}}], {"actual_device": "gpu"})):
                            with mock.patch("scoring_engine.score_emotion_windows", return_value=([{"score": 0.1, "features": {}}], {"actual_device": "gpu", "voice_feature_source": "audio_bundle"})):
                                with mock.patch("scoring_engine._run_visual_stage", side_effect=fake_visual):
                                    with mock.patch("scoring_engine._run_emotion_stage", side_effect=fake_emotion):
                                        scoring_engine.run_multimodal_scoring(
                                            "video.mp4",
                                            "audio.wav",
                                            transcript,
                                            windows,
                                            hardware_mode="auto",
                                            visual_device="gpu",
                                            emotion_device="gpu",
                                        )

        visual_end = next(ts for name, ts in call_log if name == "visual_end")
        emotion_start = next(ts for name, ts in call_log if name == "emotion_start")
        self.assertGreaterEqual(emotion_start, visual_end)

    def test_cpu_emotion_can_overlap_gpu_visual(self):
        transcript = [_build_transcript_segment(0.0, 4.0, "one")]
        windows = [SegmentWindow(index=0, start_time=0.0, end_time=4.0)]
        call_log = []
        caps = hardware.HardwareCapabilities(
            torch_cuda_available=True,
            torch_device_name="Test GPU",
            onnx_providers=["CPUExecutionProvider"],
            onnx_cuda_available=False,
        )

        def fake_visual(*args, **kwargs):
            call_log.append(("visual_start", time.perf_counter()))
            time.sleep(0.15)
            call_log.append(("visual_end", time.perf_counter()))
            return (
                {"times": np.asarray([0.5]), "motion_norm": np.asarray([0.1]), "scene_change_norm": np.asarray([0.1]), "subject_speed_norm": np.asarray([0.1]), "tracking_confidence_norm": np.asarray([0.1]), "metadata": {"actual_device": "gpu"}},
                0.15,
            )

        def fake_emotion(*args, **kwargs):
            call_log.append(("emotion_start", time.perf_counter()))
            time.sleep(0.15)
            call_log.append(("emotion_end", time.perf_counter()))
            return ({"times": np.asarray([0.5]), "facial_expression_norm": np.asarray([0.1]), "metadata": {"actual_device": "cpu"}}, 0.15)

        with mock.patch("scoring_engine.detect_capabilities", return_value=caps):
            with mock.patch("scoring_engine.extract_audio_feature_series", return_value={"times": np.asarray([0.5]), "duration": 4.0, "pitch_arousal_norm": np.asarray([0.1]), "rms_lift_norm": np.asarray([0.1]), "rms_norm": np.asarray([0.1]), "sound_change_norm": np.asarray([0.1]), "pitch_variation_norm": np.asarray([0.1]), "metadata": {}}):
                with mock.patch("scoring_engine.analyze_speech_windows", return_value=([{"text": "one", "score": 0.1, "features": {}}], {"windows_scored": 1})):
                    with mock.patch("scoring_engine.score_audio_windows", return_value=([{"score": 0.1, "features": {}}], {})):
                        with mock.patch("scoring_engine.score_visual_windows", return_value=([{"score": 0.1, "features": {}}], {"actual_device": "gpu"})):
                            with mock.patch("scoring_engine.score_emotion_windows", return_value=([{"score": 0.1, "features": {}}], {"actual_device": "cpu", "voice_feature_source": "audio_bundle"})):
                                with mock.patch("scoring_engine._run_visual_stage", side_effect=fake_visual):
                                    with mock.patch("scoring_engine._run_emotion_stage", side_effect=fake_emotion):
                                        scoring_engine.run_multimodal_scoring(
                                            "video.mp4",
                                            "audio.wav",
                                            transcript,
                                            windows,
                                            hardware_mode="auto",
                                            visual_device="gpu",
                                            emotion_device="cpu",
                                        )

        visual_end = next(ts for name, ts in call_log if name == "visual_end")
        emotion_start = next(ts for name, ts in call_log if name == "emotion_start")
        self.assertLess(emotion_start, visual_end)


class SelectorSmokeTests(unittest.TestCase):
    def test_selector_writes_segments_and_best_moments_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            transcript_path = tmp / "transcript.json"
            audio_path = tmp / "audio.wav"
            video_path = tmp / "video.mp4"
            output_path = tmp / "best_moments.json"

            transcript_path.write_text("[]", encoding="utf-8")
            audio_path.write_bytes(b"wav")
            video_path.write_bytes(b"video")

            segments = [
                {
                    "start_time": 0.0,
                    "end_time": 4.0,
                    "scores": {
                        "speech": 0.2,
                        "audio": 0.3,
                        "visual": 0.4,
                        "emotion": 0.5,
                        "momentum": 0.1,
                        "total": 0.6,
                    },
                }
            ]
            highlights = [
                {
                    "start": 1.0,
                    "end": 10.0,
                    "peak_time": 4.5,
                    "category": "mixed",
                    "score": 88,
                    "reason": "Driven by excited speech and a clear reaction beat",
                    "text": "clip text",
                    "source_window_start": 2.0,
                    "source_window_end": 6.0,
                    "scores": {
                        "speech": 0.8,
                        "audio": 0.5,
                        "visual": 0.4,
                        "emotion": 0.75,
                        "momentum": 0.2,
                        "total": 0.88,
                    },
                }
            ]
            metadata = {"segments_scored": 1, "highlights_selected": 1}

            with mock.patch("llm_selector.analyze_multimodal_highlights", return_value=(segments, highlights, metadata)):
                returned = llm_selector.find_best_clipping_moments(
                    str(transcript_path),
                    str(audio_path),
                    video_path=str(video_path),
                    use_vision=True,
                    output_path=str(output_path),
                )

            segments_path = tmp / "best_moments_segments.json"
            self.assertEqual(returned, highlights)
            self.assertTrue(output_path.exists())
            self.assertTrue(segments_path.exists())

            written_segments = json.loads(segments_path.read_text(encoding="utf-8"))
            written_highlights = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written_segments, segments)
            self.assertEqual(written_highlights, highlights)
            self.assertIsInstance(written_highlights[0]["score"], int)
            self.assertIsInstance(written_highlights[0]["scores"]["total"], float)


if __name__ == "__main__":
    unittest.main()
