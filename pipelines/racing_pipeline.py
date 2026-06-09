"""
Racing-game highlight detection pipeline.

Detects five core racing event types — drift, speed_burst, crash, near_miss,
and overtake — by analysing multimodal signals from the unified TimelineFrame
stream.  Uses sliding-window analysis with numpy-based signal smoothing and a
temporal-momentum bonus so events that build over time outscore isolated spikes.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

from phase3_types import EventMoment, GameProfile
from core.cache.timeline_frame import TimelineFrame
from pipelines.base_pipeline import BaseGenrePipeline
from pipelines.registry import PipelineRegistry

logger = logging.getLogger(__name__)

# ── Thresholds & constants ──────────────────────────────────────────────────
_WINDOW_SIZE: int = 4                 # seconds — sliding window width (3–5 range)
_SMOOTHING_KERNEL: int = 3            # Gaussian-ish uniform kernel width

# Drift
_DRIFT_LATERAL_THRESHOLD: float = 0.45
_DRIFT_MIN_DURATION: float = 2.0      # sustained lateral flow ≥ 2 s

# Speed burst
_SPEED_BURST_THRESHOLD: float = 0.6   # normalised speed
_SPEED_ACCEL_THRESHOLD: float = 0.25  # Δ speed / Δ t

# Crash
_CRASH_SHAKE_THRESHOLD: float = 0.55
_CRASH_MOTION_THRESHOLD: float = 0.50
_CRASH_AUDIO_THRESHOLD: float = 0.50
_CRASH_AFTERMATH_SECS: int = 3        # frames to check for lingering shake

# Near miss
_NEAR_MISS_SHAKE_THRESHOLD: float = 0.40
_NEAR_MISS_RESOLVE_SECS: int = 2      # must resolve within this many frames

# Overtake
_OVERTAKE_POSITION_DELTA: float = 0.15  # tracked-object x-shift threshold

# Event priorities
_PRIORITIES: Dict[str, int] = {
    "crash": 10,
    "near_miss": 9,
    "drift": 7,
    "overtake": 6,
    "speed_burst": 5,
}

# Pacing buffers for racing events (generous context for replays)
_PRE_CONTEXT_BUFFER: float = 5.0
_POST_PAYOFF_BUFFER: float = 3.0

# Overlap merge threshold (seconds) — events closer than this are merged
_OVERLAP_MERGE_GAP: float = 1.0

# Tuple type returned by each sub-detector
_DetectionTuple = Tuple[float, float, float, float, str]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _smooth(signal: np.ndarray, kernel_size: int = _SMOOTHING_KERNEL) -> np.ndarray:
    """Apply a simple uniform (box) smoothing kernel to *signal*."""
    if len(signal) < kernel_size:
        return signal
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(signal, kernel, mode="same")


def _temporal_momentum(scores: np.ndarray, centre: int, radius: int = 2) -> float:
    """
    Compute a momentum bonus for index *centre* by looking at how strongly
    the signal builds in the neighbourhood [centre-radius … centre+radius].
    A rising slope into the peak scores higher than an isolated spike.

    Returns a multiplier in [1.0, 1.5].
    """
    lo = max(0, centre - radius)
    hi = min(len(scores), centre + radius + 1)
    window = scores[lo:hi]
    if len(window) < 2:
        return 1.0
    # Fraction of neighbours above the median
    med = float(np.median(window))
    if med <= 0:
        return 1.0
    above_ratio = float(np.mean(window >= med * 0.6))
    return 1.0 + 0.5 * min(above_ratio, 1.0)


# ── Pipeline ────────────────────────────────────────────────────────────────

class RacingPipeline(BaseGenrePipeline):
    """
    Racing / sim-racing / arcade-racing highlight-detection pipeline.

    Analyses the timeline with a sliding window of ~4 s, running five parallel
    sub-detectors for drift, speed burst, crash, near miss, and overtake.
    Results are deduplicated, scored with temporal momentum, and returned as
    :class:`EventMoment` instances.
    """

    # ── BaseGenrePipeline interface ─────────────────────────────────────

    @property
    def name(self) -> str:
        return "RacingPipeline"

    def detect(
        self,
        timeline: List[TimelineFrame],
        transcript_data: Optional[List[Dict]] = None,
        profile: Optional[GameProfile] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> List[EventMoment]:
        if not timeline:
            logger.warning("RacingPipeline.detect() called with empty timeline")
            return []

        logger.info(
            "RacingPipeline: analysing %d frames (%.1f–%.1f s)",
            len(timeline),
            timeline[0].timestamp,
            timeline[-1].timestamp,
        )

        # Run all sub-detectors
        detections: List[_DetectionTuple] = []
        detections.extend(self._detect_drift(timeline))
        detections.extend(self._detect_speed_burst(timeline))
        detections.extend(self._detect_crash(timeline))
        detections.extend(self._detect_near_miss(timeline))
        detections.extend(self._detect_overtake(timeline))

        logger.info(
            "RacingPipeline: raw sub-detections — drift=%d  speed_burst=%d  crash=%d  near_miss=%d  overtake=%d",
            sum(1 for d in detections if d[4] == "drift"),
            sum(1 for d in detections if d[4] == "speed_burst"),
            sum(1 for d in detections if d[4] == "crash"),
            sum(1 for d in detections if d[4] == "near_miss"),
            sum(1 for d in detections if d[4] == "overtake"),
        )

        # Deduplicate overlapping detections (prefer higher-priority events)
        detections = self._deduplicate(detections)

        # Build EventMoment objects
        moments = self._build_moments(detections, transcript_data)
        logger.info("RacingPipeline: returning %d events after deduplication", len(moments))
        return moments

    # ── Sub-detectors ───────────────────────────────────────────────────

    def _detect_drift(
        self, timeline: List[TimelineFrame]
    ) -> List[_DetectionTuple]:
        """
        High lateral optical-flow ratio sustained over ≥ 2 seconds.
        """
        results: List[_DetectionTuple] = []
        lateral = np.array(
            [f.optical_flow.get("lateral_ratio", 0.0) for f in timeline],
            dtype=np.float64,
        )
        lateral_smooth = _smooth(lateral)
        n = len(timeline)

        in_drift = False
        drift_start_idx = 0

        for i in range(n):
            above = lateral_smooth[i] >= _DRIFT_LATERAL_THRESHOLD
            if above and not in_drift:
                in_drift = True
                drift_start_idx = i
            elif not above and in_drift:
                in_drift = False
                duration = timeline[i].timestamp - timeline[drift_start_idx].timestamp
                if duration >= _DRIFT_MIN_DURATION:
                    seg = lateral_smooth[drift_start_idx:i]
                    peak_local = int(np.argmax(seg))
                    peak_idx = drift_start_idx + peak_local
                    raw_score = float(np.mean(seg))
                    momentum = _temporal_momentum(lateral_smooth, peak_idx)
                    score = min(raw_score * momentum, 1.0)
                    results.append((
                        timeline[drift_start_idx].timestamp,
                        timeline[i].timestamp,
                        timeline[peak_idx].timestamp,
                        score,
                        "drift",
                    ))

        # Handle drift that extends to end of timeline
        if in_drift:
            duration = timeline[-1].timestamp - timeline[drift_start_idx].timestamp
            if duration >= _DRIFT_MIN_DURATION:
                seg = lateral_smooth[drift_start_idx:]
                peak_local = int(np.argmax(seg))
                peak_idx = drift_start_idx + peak_local
                raw_score = float(np.mean(seg))
                momentum = _temporal_momentum(lateral_smooth, peak_idx)
                score = min(raw_score * momentum, 1.0)
                results.append((
                    timeline[drift_start_idx].timestamp,
                    timeline[-1].timestamp,
                    timeline[peak_idx].timestamp,
                    score,
                    "drift",
                ))

        return results

    def _detect_speed_burst(
        self, timeline: List[TimelineFrame]
    ) -> List[_DetectionTuple]:
        """
        High vehicle_speed_estimate with sudden acceleration (speed derivative).
        """
        results: List[_DetectionTuple] = []
        speed = np.array(
            [f.vehicle_speed_estimate for f in timeline], dtype=np.float64
        )
        if len(speed) < 2:
            return results

        speed_smooth = _smooth(speed)
        accel = np.gradient(speed_smooth)
        n = len(timeline)

        i = 0
        while i < n:
            if (
                speed_smooth[i] >= _SPEED_BURST_THRESHOLD
                and accel[i] >= _SPEED_ACCEL_THRESHOLD
            ):
                start_idx = i
                # Extend window while speed stays high
                j = i + 1
                while j < n and speed_smooth[j] >= _SPEED_BURST_THRESHOLD * 0.7:
                    j += 1
                end_idx = min(j, n - 1)

                seg_speed = speed_smooth[start_idx : end_idx + 1]
                seg_accel = accel[start_idx : end_idx + 1]
                peak_local = int(np.argmax(seg_speed))
                peak_idx = start_idx + peak_local

                raw_score = float(
                    0.6 * np.max(seg_speed) + 0.4 * np.clip(np.max(seg_accel), 0, 1)
                )
                momentum = _temporal_momentum(speed_smooth, peak_idx)
                score = min(raw_score * momentum, 1.0)

                results.append((
                    timeline[start_idx].timestamp,
                    timeline[end_idx].timestamp,
                    timeline[peak_idx].timestamp,
                    score,
                    "speed_burst",
                ))
                i = end_idx + 1
            else:
                i += 1

        return results

    def _detect_crash(
        self, timeline: List[TimelineFrame]
    ) -> List[_DetectionTuple]:
        """
        Sudden high camera_shake + motion_energy spike + audio_energy spike,
        with lingering shake aftermath.
        """
        results: List[_DetectionTuple] = []
        shake = np.array([f.camera_shake for f in timeline], dtype=np.float64)
        motion = np.array([f.motion_energy for f in timeline], dtype=np.float64)
        audio = np.array([f.audio_energy for f in timeline], dtype=np.float64)
        n = len(timeline)

        shake_smooth = _smooth(shake)
        motion_smooth = _smooth(motion)
        audio_smooth = _smooth(audio)

        i = 0
        while i < n:
            if (
                shake_smooth[i] >= _CRASH_SHAKE_THRESHOLD
                and motion_smooth[i] >= _CRASH_MOTION_THRESHOLD
                and audio_smooth[i] >= _CRASH_AUDIO_THRESHOLD
            ):
                # Check for aftermath: shake stays elevated for _CRASH_AFTERMATH_SECS
                aftermath_end = min(i + _CRASH_AFTERMATH_SECS, n - 1)
                aftermath_shake = shake_smooth[i : aftermath_end + 1]
                has_aftermath = bool(
                    np.mean(aftermath_shake) >= _CRASH_SHAKE_THRESHOLD * 0.4
                )

                if has_aftermath:
                    start_idx = max(0, i - 1)
                    end_idx = aftermath_end

                    combined = (
                        0.4 * shake_smooth[start_idx : end_idx + 1]
                        + 0.3 * motion_smooth[start_idx : end_idx + 1]
                        + 0.3 * audio_smooth[start_idx : end_idx + 1]
                    )
                    peak_local = int(np.argmax(combined))
                    peak_idx = start_idx + peak_local
                    raw_score = float(np.max(combined))
                    momentum = _temporal_momentum(shake_smooth, peak_idx)
                    score = min(raw_score * momentum, 1.0)

                    results.append((
                        timeline[start_idx].timestamp,
                        timeline[end_idx].timestamp,
                        timeline[peak_idx].timestamp,
                        score,
                        "crash",
                    ))
                    i = end_idx + 1
                else:
                    i += 1
            else:
                i += 1

        return results

    def _detect_near_miss(
        self, timeline: List[TimelineFrame]
    ) -> List[_DetectionTuple]:
        """
        Brief camera_shake spike that resolves quickly (no crash aftermath).
        """
        results: List[_DetectionTuple] = []
        shake = np.array([f.camera_shake for f in timeline], dtype=np.float64)
        motion = np.array([f.motion_energy for f in timeline], dtype=np.float64)
        audio = np.array([f.audio_energy for f in timeline], dtype=np.float64)
        n = len(timeline)

        shake_smooth = _smooth(shake)

        i = 0
        while i < n:
            if shake_smooth[i] >= _NEAR_MISS_SHAKE_THRESHOLD:
                # Check that the shake resolves within _NEAR_MISS_RESOLVE_SECS
                resolve_end = min(i + _NEAR_MISS_RESOLVE_SECS, n - 1)
                resolved = bool(shake_smooth[resolve_end] < _NEAR_MISS_SHAKE_THRESHOLD * 0.5)

                # Exclude events that also qualify as crashes
                is_crash = bool(
                    motion[i] >= _CRASH_MOTION_THRESHOLD
                    and audio[i] >= _CRASH_AUDIO_THRESHOLD
                    and shake_smooth[i] >= _CRASH_SHAKE_THRESHOLD
                )

                if resolved and not is_crash:
                    start_idx = max(0, i - 1)
                    end_idx = resolve_end
                    peak_idx = start_idx + int(
                        np.argmax(shake_smooth[start_idx : end_idx + 1])
                    )
                    raw_score = float(shake_smooth[peak_idx])
                    momentum = _temporal_momentum(shake_smooth, peak_idx)
                    score = min(raw_score * momentum, 1.0)

                    results.append((
                        timeline[start_idx].timestamp,
                        timeline[end_idx].timestamp,
                        timeline[peak_idx].timestamp,
                        score,
                        "near_miss",
                    ))
                    i = end_idx + 1
                else:
                    i += 1
            else:
                i += 1

        return results

    def _detect_overtake(
        self, timeline: List[TimelineFrame]
    ) -> List[_DetectionTuple]:
        """
        Vehicle objects appearing and being passed — detected via tracked_objects
        position changes over a sliding window.
        """
        results: List[_DetectionTuple] = []
        n = len(timeline)
        if n < _WINDOW_SIZE:
            return results

        for i in range(n - _WINDOW_SIZE + 1):
            window = timeline[i : i + _WINDOW_SIZE]
            overtake_score = self._score_overtake_window(window)
            if overtake_score > 0.0:
                start_t = window[0].timestamp
                end_t = window[-1].timestamp
                # Peak is at the frame with highest individual overtake contribution
                per_frame_scores = []
                for k in range(1, len(window)):
                    per_frame_scores.append(
                        self._pairwise_overtake_score(window[k - 1], window[k])
                    )
                if per_frame_scores:
                    peak_offset = int(np.argmax(per_frame_scores)) + 1
                else:
                    peak_offset = len(window) // 2
                peak_t = window[peak_offset].timestamp

                results.append((start_t, end_t, peak_t, overtake_score, "overtake"))

        # Merge overlapping overtake detections
        results = self._merge_adjacent(results)
        return results

    # ── Overtake helpers ────────────────────────────────────────────────

    @staticmethod
    def _pairwise_overtake_score(
        prev: TimelineFrame, curr: TimelineFrame
    ) -> float:
        """
        Score a single frame-pair for overtake evidence based on tracked
        objects shifting backward (i.e., player passing them).
        """
        if not prev.tracked_objects or not curr.tracked_objects:
            return 0.0

        prev_positions = {
            obj.get("id", idx): obj.get("x", 0.5)
            for idx, obj in enumerate(prev.tracked_objects)
            if obj.get("class", "").lower() in ("car", "vehicle", "kart", "bike")
        }
        curr_positions = {
            obj.get("id", idx): obj.get("x", 0.5)
            for idx, obj in enumerate(curr.tracked_objects)
            if obj.get("class", "").lower() in ("car", "vehicle", "kart", "bike")
        }

        total_delta = 0.0
        count = 0
        for obj_id, prev_x in prev_positions.items():
            if obj_id in curr_positions:
                delta = prev_x - curr_positions[obj_id]  # positive = passed
                if delta > _OVERTAKE_POSITION_DELTA:
                    total_delta += delta
                    count += 1

        if count == 0:
            return 0.0
        return min(total_delta / count, 1.0)

    def _score_overtake_window(self, window: List[TimelineFrame]) -> float:
        """
        Aggregate overtake evidence across a full sliding window.
        """
        if len(window) < 2:
            return 0.0

        scores = []
        for k in range(1, len(window)):
            scores.append(self._pairwise_overtake_score(window[k - 1], window[k]))

        arr = np.array(scores, dtype=np.float64)
        if float(np.max(arr)) < 0.1:
            return 0.0

        raw = float(np.mean(arr[arr > 0])) if np.any(arr > 0) else 0.0
        centre = int(np.argmax(arr))
        momentum = _temporal_momentum(arr, centre)
        return min(raw * momentum, 1.0)

    # ── Deduplication & merge ───────────────────────────────────────────

    @staticmethod
    def _merge_adjacent(
        detections: List[_DetectionTuple],
    ) -> List[_DetectionTuple]:
        """Merge detections of the same type that overlap or are very close."""
        if not detections:
            return detections
        detections = sorted(detections, key=lambda d: d[0])
        merged: List[_DetectionTuple] = [detections[0]]
        for det in detections[1:]:
            prev = merged[-1]
            # Same event type and overlapping / within merge gap
            if det[4] == prev[4] and det[0] <= prev[1] + _OVERLAP_MERGE_GAP:
                # Keep the higher score and wider time span
                best_score = max(prev[3], det[3])
                peak = prev[2] if prev[3] >= det[3] else det[2]
                merged[-1] = (
                    min(prev[0], det[0]),
                    max(prev[1], det[1]),
                    peak,
                    best_score,
                    prev[4],
                )
            else:
                merged.append(det)
        return merged

    def _deduplicate(
        self, detections: List[_DetectionTuple]
    ) -> List[_DetectionTuple]:
        """
        Remove lower-priority events that overlap temporally with a
        higher-priority event of a different type.
        """
        if not detections:
            return detections

        # Sort by priority descending, then by score descending
        detections = sorted(
            detections,
            key=lambda d: (_PRIORITIES.get(d[4], 0), d[3]),
            reverse=True,
        )

        kept: List[_DetectionTuple] = []
        for det in detections:
            overlaps_higher = False
            for existing in kept:
                if _PRIORITIES.get(existing[4], 0) > _PRIORITIES.get(det[4], 0):
                    # Check temporal overlap
                    if det[0] < existing[1] and det[1] > existing[0]:
                        overlaps_higher = True
                        break
            if not overlaps_higher:
                kept.append(det)

        # Re-sort chronologically
        kept.sort(key=lambda d: d[0])
        return kept

    # ── EventMoment construction ────────────────────────────────────────

    def _build_moments(
        self,
        detections: List[_DetectionTuple],
        transcript_data: Optional[List[Dict]],
    ) -> List[EventMoment]:
        """Convert raw detection tuples into fully-populated EventMoment objects."""
        moments: List[EventMoment] = []
        for start, end, peak_time, score, event_type in detections:
            duration = end - start
            transcript_text = self._find_transcript(start, end, transcript_data)

            moment = EventMoment(
                event_type=event_type,
                start=start,
                end=end,
                peak_time=peak_time,
                duration=duration,
                final_score=score,
                surprise_score=score if event_type in ("crash", "near_miss") else score * 0.5,
                conflict_score=score if event_type in ("crash", "overtake") else score * 0.3,
                payoff_score=score * 0.8,
                priority=_PRIORITIES.get(event_type, 5),
                scene_type="racing",
                features={
                    "event_type": event_type,
                    "detection_pipeline": self.name,
                },
                transcript=transcript_text,
                event_confidence=score,
                pre_context_buffer=_PRE_CONTEXT_BUFFER,
                post_payoff_buffer=_POST_PAYOFF_BUFFER,
                peak_prominence=score * 0.6,
                label=f"Racing {event_type.replace('_', ' ').title()}",
                short_title=event_type.replace("_", " ").title(),
                hook_style="action",
            )
            moments.append(moment)

        return moments

    @staticmethod
    def _find_transcript(
        start: float,
        end: float,
        transcript_data: Optional[List[Dict]],
    ) -> str:
        """
        Extract transcript text overlapping the [start, end] window.
        Each transcript entry is expected to carry 'start', 'end', and 'text' keys.
        """
        if not transcript_data:
            return ""
        segments: List[str] = []
        for seg in transcript_data:
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            if seg_start < end and seg_end > start:
                text = seg.get("text", "").strip()
                if text:
                    segments.append(text)
        return " ".join(segments)


# ── Plugin registration ─────────────────────────────────────────────────────
PipelineRegistry.register("racing", RacingPipeline)
PipelineRegistry.register("sim-racing", RacingPipeline)
PipelineRegistry.register("arcade-racing", RacingPipeline)
