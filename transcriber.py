import json
import os
import sys
import time
import wave

from faster_whisper import WhisperModel
from anyascii import anyascii

from hardware import build_preflight_lines, detect_capabilities, plan_hardware, record_stage_metric
from multimodal_utils import ProgressReporter


# Attempt to load newly installed nvidia DLLs from site-packages to fix missing CUDA DLLs on Windows.
try:
    for path in sys.path:
        nvidia_path = os.path.join(path, "nvidia")
        if os.path.exists(nvidia_path):
            for lib in os.listdir(nvidia_path):
                bin_path = os.path.join(nvidia_path, lib, "bin")
                if os.path.exists(bin_path):
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
                    if hasattr(os, "add_dll_directory"):
                        os.add_dll_directory(bin_path)
except Exception:
    pass


def _probe_wav_duration(audio_path):
    try:
        with wave.open(audio_path, "rb") as handle:
            frame_rate = float(handle.getframerate() or 0.0)
            frame_count = float(handle.getnframes() or 0.0)
            if frame_rate > 0 and frame_count > 0:
                return frame_count / frame_rate
    except Exception:
        return 0.0
    return 0.0


def _load_model(model_size, planned_device, strict_device):
    whisper_device = "cuda" if planned_device == "gpu" else "cpu"
    compute_type = "float16" if planned_device == "gpu" else "int8"
    try:
        model = WhisperModel(model_size, device=whisper_device, compute_type=compute_type)
        return model, planned_device
    except Exception as exc:
        if planned_device == "gpu" and not strict_device:
            print(f"Transcription GPU setup failed. Falling back to CPU... ({exc})", flush=True)
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            return model, "cpu"
        raise


def transcribe_audio(
    audio_path,
    model_size="base",
    hardware_mode="auto",
    device_policy=None,
    output_path="transcript.json",
    hardware_profile=None,
    hinglish=False,
):
    capabilities = detect_capabilities()
    hardware_plan = plan_hardware(
        hardware_mode=hardware_mode,
        stage_overrides={"transcribe": device_policy},
        capabilities=capabilities,
        profile_path=hardware_profile,
        stages=("transcribe",),
    )
    planned_device = hardware_plan.stage_device("transcribe")
    strict_device = hardware_plan.stage_strict("transcribe")

    print("Hardware plan for transcription:", flush=True)
    for line in build_preflight_lines(hardware_plan):
        print(f"  {line}", flush=True)

    print(f"Loading Whisper model '{model_size}' on planned device {planned_device}...", flush=True)
    model, actual_device = _load_model(model_size, planned_device, strict_device)
    if planned_device != actual_device:
        print(f"Transcription actual device: {actual_device} (planned {planned_device})", flush=True)
    else:
        print(f"Transcription actual device: {actual_device}", flush=True)

    print(f"Transcribing {audio_path}...", flush=True)
    started = time.perf_counter()
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability), flush=True)

    do_hinglish = hinglish and info.language == "hi"
    if do_hinglish:
        print("Hindi detected. Hinglish Romanization is ENABLED.", flush=True)

    duration_secs = _probe_wav_duration(audio_path)
    reporter = ProgressReporter(duration_secs or 1, label="Transcribe")

    results = []
    for segment in segments:
        reporter.current_item = min(segment.end, duration_secs) if duration_secs else segment.end
        reporter.report()
        
        text = segment.text.strip()
        if do_hinglish:
            text = anyascii(text)

        words = []
        for w in getattr(segment, "words", None) or []:
            word_str = w.word
            if do_hinglish:
                word_str = anyascii(word_str)
            words.append({
                "word": word_str,
                "start": w.start,
                "end": w.end,
                "probability": w.probability,
            })

        results.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": text,
                "words": words,
            }
        )

    elapsed = time.perf_counter() - started
    duration_secs = _probe_wav_duration(audio_path)
    input_minutes = max(float(duration_secs) / 60.0, 1e-6)

    if planned_device == "gpu" and actual_device != "gpu":
        record_stage_metric(
            "transcribe",
            "gpu",
            input_minutes,
            elapsed,
            success=False,
            error="runtime fallback to CPU",
            profile_path=hardware_profile,
        )
    record_stage_metric(
        "transcribe",
        actual_device,
        input_minutes,
        elapsed,
        success=True,
        profile_path=hardware_profile,
    )

    print("Transcription complete.", flush=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"Transcript saved to {output_path}", flush=True)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone Whisper Transcriber")
    parser.add_argument("audio", help="Path to input audio wav file")
    parser.add_argument("--output", default="transcript.json", help="Path to save output JSON transcript")
    parser.add_argument("--model", default="base", help="Whisper model size (base, small, medium, etc.)")
    parser.add_argument("--hardware-mode", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "cuda"), default=None)
    parser.add_argument("--hardware-profile", default=None, help="Path to the shared hardware profile JSON")
    parser.add_argument("--hinglish", action="store_true", help="Romanize Hindi transcription")
    args = parser.parse_args()

    try:
        transcribe_audio(
            args.audio,
            model_size=args.model,
            hardware_mode=args.hardware_mode,
            device_policy=args.device,
            output_path=args.output,
            hardware_profile=args.hardware_profile,
            hinglish=args.hinglish,
        )
        sys.exit(0)
    except Exception as exc:
        print(f"\nERROR Transcriber Error: {exc}")
        sys.exit(1)
