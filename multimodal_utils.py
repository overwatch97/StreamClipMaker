import json
import math
import subprocess
import time
from typing import Iterable

import cv2
import numpy as np


def clip01(value):
    return float(max(0.0, min(1.0, value)))


def safe_div(numerator, denominator, default=0.0):
    if not denominator:
        return default
    return numerator / denominator


def robust_normalize(values, lower_percentile=50.0, upper_percentile=95.0):
    """
    Maps a numeric series into [0, 1] using robust percentiles instead of raw min/max.
    This keeps one loud spike from flattening the rest of the stream.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr, dtype=float)

    finite = arr[finite_mask]
    lo = float(np.percentile(finite, lower_percentile))
    hi = float(np.percentile(finite, upper_percentile))

    if math.isclose(lo, hi):
        # A flat stream has no relative spikes, so treat it as zero-signal.
        return np.zeros_like(arr, dtype=float)

    norm = np.zeros_like(arr, dtype=float)
    norm[finite_mask] = (finite - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


def positive_zscore(value, mean, std, scale=2.0):
    if std <= 1e-6:
        return 0.0
    return clip01(max(0.0, (value - mean) / (std * scale)))


def get_window_bounds(window):
    if hasattr(window, "start_time") and hasattr(window, "end_time"):
        return float(window.start_time), float(window.end_time)
    return float(window["start_time"]), float(window["end_time"])


def aggregate_window_series(times, values, start_time, end_time, reducer="mean", default=0.0, quantile=0.75):
    times_arr = np.asarray(times, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if times_arr.size == 0 or values_arr.size == 0:
        return float(default)

    mask = (times_arr >= start_time) & (times_arr < end_time)
    if not np.any(mask):
        return float(default)

    window_values = values_arr[mask]
    if reducer == "mean":
        return float(np.mean(window_values))
    if reducer == "max":
        return float(np.max(window_values))
    if reducer == "quantile":
        return float(np.quantile(window_values, quantile))
    raise ValueError(f"Unsupported reducer: {reducer}")


def flatten_transcript_words(transcript_data):
    words = []
    for segment in transcript_data:
        for word in segment.get("words", []):
            token = str(word.get("word", "")).strip()
            if not token:
                continue
            words.append(
                {
                    "word": token,
                    "start": float(word.get("start", 0.0)),
                    "end": float(word.get("end", 0.0)),
                    "lower": token.lower(),
                }
            )
    return words


def collect_transcript_text(transcript_data, start_time, end_time):
    overlapping = []
    for segment in transcript_data:
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", 0.0))
        if seg_end > start_time and seg_start < end_time:
            text = str(segment.get("text", "")).strip()
            if text:
                overlapping.append(text)
    return " ".join(overlapping).strip()


def infer_stream_duration(video_path=None, audio_duration=None, transcript_data=None):
    duration_candidates = []
    if audio_duration:
        duration_candidates.append(float(audio_duration))

    if transcript_data:
        transcript_ends = [float(segment.get("end", 0.0)) for segment in transcript_data if "end" in segment]
        if transcript_ends:
            duration_candidates.append(max(transcript_ends))

    if video_path:
        cap = cv2.VideoCapture(video_path)
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if fps > 0 and frame_count > 0:
                duration_candidates.append(frame_count / fps)
        finally:
            cap.release()

    if not duration_candidates:
        return 0.0
    return float(max(duration_candidates))


def summarize_percentiles(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"p50": 0.0, "p75": 0.0, "p95": 0.0}
    return {
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
    }


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def truncate_text(text, limit=320):
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def mean_or_default(values: Iterable[float], default=0.0):
    values = list(values)
    if not values:
        return float(default)
    return float(sum(values) / len(values))


class ProgressReporter:
    def __init__(self, total_items, label="Processing"):
        self.total_items = total_items
        self.label = label
        self.start_time = time.perf_counter()
        self.current_item = 0
        self.last_report_time = 0

    def update(self, count=1):
        self.current_item += count

    def report(self, force=False, frequency=2.0):
        """Prints progress to stdout if 'frequency' seconds have passed or if 'force' is True."""
        now = time.perf_counter()
        if not force and (now - self.last_report_time) < frequency:
            return

        self.last_report_time = now
        elapsed = now - self.start_time
        
        if self.total_items > 0:
            percent = (self.current_item / self.total_items) * 100
            rate = self.current_item / elapsed if elapsed > 0 else 0
            remaining_items = self.total_items - self.current_item
            eta = remaining_items / rate if rate > 0 else 0
            
            report_line = (
                f"      [{self.label}] {self.current_item}/{self.total_items} "
                f"({percent:.1f}%) | "
                f"Taken: {self.format_seconds(elapsed)} | "
                f"Remaining: {self.format_seconds(eta)}"
            )
        else:
            report_line = (
                f"      [{self.label}] {self.current_item}/? "
                f"(N/A%) | "
                f"Taken: {self.format_seconds(elapsed)} | "
                f"Remaining: N/A"
            )
        print(report_line, flush=True)

    @staticmethod
    def format_seconds(seconds):
        if seconds < 0:
            seconds = 0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"


def get_video_stream_info(video_path):
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,duration,nb_frames:format=duration",
            "-of", "json", video_path
        ]
        res = json.loads(subprocess.check_output(cmd).decode())
        
        info = {}
        if res.get("streams"):
            info = res["streams"][0]
        
        # Fallback to format duration if stream duration is missing
        if not info.get("duration") and res.get("format", {}).get("duration"):
            info["duration"] = res["format"]["duration"]
            
        # Fallback to calculating duration from nb_frames and avg_frame_rate
        if not info.get("duration") and info.get("nb_frames") and info.get("avg_frame_rate"):
            try:
                num, den = map(float, info["avg_frame_rate"].split("/"))
                if num > 0 and den > 0:
                    fps = num / den
                    info["duration"] = str(float(info["nb_frames"]) / fps)
            except Exception:
                pass
                
        return info
    except Exception:
        return {}


class FFmpegFrameExtractor:
    def __init__(self, video_path, sample_fps, target_width=320, device="cpu"):
        self.video_path = video_path
        self.sample_fps = sample_fps
        self.target_width = target_width
        self.device = device
        self.info = get_video_stream_info(video_path)
        
        self.native_w = int(self.info.get("width", 1920))
        self.native_h = int(self.info.get("height", 1080))
        self.codec = self.info.get("codec_name", "h264")
        
        # Scale to target width while preserving aspect ratio
        self.out_w = target_width
        self.out_h = int(self.native_h * (target_width / self.native_w))
        # Ensure even dimensions for ffmpeg filters if needed, 
        # but for rawvideo bgr24 it's less critical.
        
        self.process = self._launch()

    def _launch(self):
        # We try to use hardware acceleration if device="gpu"
        # However, we only use cuvid if it's h264 or hevc.
        input_args = []
        if self.device == "gpu":
            if self.codec == "h264":
                input_args = ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
            elif self.codec in ("hevc", "h265"):
                input_args = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
            else:
                input_args = ["-hwaccel", "cuda"]

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error"
        ] + input_args + [
            "-i", self.video_path,
            "-vf", f"fps={self.sample_fps},scale={self.out_w}:{self.out_h}",
            "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

    def __iter__(self):
        frame_size = self.out_w * self.out_h * 3
        while True:
            in_bytes = self.process.stdout.read(frame_size)
            if not in_bytes or len(in_bytes) < frame_size:
                break
            
            frame = np.frombuffer(in_bytes, np.uint8).reshape((self.out_h, self.out_w, 3))
            yield frame

    def close(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except:
                self.process.kill()
