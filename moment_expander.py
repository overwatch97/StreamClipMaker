"""
moment_expander.py — Arc-Aware Clip Boundary Expansion
========================================================
Converts an ArcRegion (from arc_detector.py) or a legacy DetectedEvent
into a HighlightCandidate with precise clip boundaries.

Shape-aware expansion:
  SPIKE     — peak ± tight buffer (5s pre, 8s post)
  TENSION   — use full arc start→end (the whole arc IS the story)
  COMEDY    — 3s setup before spike onset + 6s post-peak
  DRAMA     — full arc, snapped to sentence boundaries
  TRIUMPH   — 10s before arc end (show the struggle) + 6s post-peak
  DISCOVERY — 2s before arc + 12s after onset
"""

import logging
from typing import List, Dict, Optional

import numpy as np

from phase3_types import (
    ArcRegion, ArcShape, ARC_DURATION_RULES,
    GameProfile, TimelineSecond, DetectedEvent, HighlightCandidate,
)
import editing_brain

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Arc-Region Expansion (primary path)
# ─────────────────────────────────────────────────────────────────────────────

def expand_arc(
    arc: ArcRegion,
    timeline: List[TimelineSecond],
    profile: Optional[GameProfile],
    transcript_data: List[Dict],
    stream_duration: float = 0.0,
) -> HighlightCandidate:
    """
    Converts an ArcRegion into a HighlightCandidate with shape-aware boundaries.
    """
    shape   = arc.shape_type
    min_dur, max_dur = ARC_DURATION_RULES.get(shape.value, (5, 60))

    # Shape-specific boundary strategy
    if shape == ArcShape.SPIKE:
        start, end = _expand_spike(arc, timeline, stream_duration)

    elif shape == ArcShape.TENSION:
        # The entire arc is the clip — setup and resolution both matter
        start, end = arc.start, arc.end
        # Safety cap at max_dur, keeping the peak centered
        if (end - start) > max_dur:
            half = max_dur / 2.0
            start = max(0.0, arc.peak_time - half * 0.45)
            end   = min(stream_duration or arc.end, arc.peak_time + half * 0.55)

    elif shape == ArcShape.COMEDY:
        # 3s of setup before onset + punchline plays out fully
        start = max(0.0, arc.start - 3.0)
        end   = min(stream_duration or arc.end, arc.peak_time + 6.0)

    elif shape == ArcShape.DRAMA:
        # Dialogue must be complete — use full arc
        start, end = arc.start, arc.end
        if (end - start) > max_dur:
            end = start + max_dur

    elif shape == ArcShape.TRIUMPH:
        # Show the last phase of struggle + the win
        start = max(0.0, arc.end - min(40.0, arc.duration * 0.65))
        end   = min(stream_duration or arc.end, arc.peak_time + 6.0)
        end   = max(end, arc.end)  # Always include the arc end (the payoff)

    elif shape == ArcShape.DISCOVERY:
        # Brief setup → the reveal
        start = max(0.0, arc.start - 2.0)
        end   = min(stream_duration or arc.end, arc.start + min(arc.duration + 8.0, max_dur))

    else:
        start, end = arc.start, arc.end

    # Minimum duration floor
    if (end - start) < min_dur:
        end = start + min_dur

    # Snap to natural speech boundaries
    start = editing_brain.find_nearest_word_boundary(start, transcript_data, search_window=2.0)
    end   = editing_brain.find_nearest_word_boundary(end,   transcript_data, search_window=2.5)

    # Clamp to stream
    if stream_duration > 0:
        start = max(0.0, min(start, stream_duration - min_dur))
        end   = min(stream_duration, end)

    # Build a synthetic DetectedEvent for the HighlightCandidate anchor
    anchor = DetectedEvent(
        event_type=shape.value,
        timestamp=arc.peak_time,
        score=arc.quality_score,
        evidence={
            "peak_audio":   arc.peak_audio,
            "peak_motion":  arc.peak_motion,
            "peak_emotion": arc.peak_emotion,
            "peak_speech":  arc.peak_speech,
            "end_composite": arc.end_composite,
        },
    )

    profile_id      = profile.game_id      if profile else "generic"
    profile_version = profile.version      if profile else "1.0.0"
    game_id         = profile.game_id      if profile else "generic"

    candidate = HighlightCandidate(
        start=round(float(start), 3),
        end=round(float(end), 3),
        anchor_event=anchor,
        score=arc.quality_score,
        category=arc.label or shape.value,
        reason=arc.label or f"{shape.value} arc at {arc.start:.1f}s",
        game_id=game_id,
        events=[anchor],
        profile_id=profile_id,
        profile_version=profile_version,
        evidence=anchor.evidence,
        text=arc.transcript,
        rank_score=arc.quality_score,
    )

    # Bake hook/title into clipper JSON via text field
    if arc.hook_sentence:
        candidate.text = f"[Hook: {arc.hook_sentence}] {candidate.text}".strip()

    return candidate


# ─────────────────────────────────────────────────────────────────────────────
# SPIKE-specific expander
# ─────────────────────────────────────────────────────────────────────────────

def _expand_spike(
    arc: ArcRegion,
    timeline: List[TimelineSecond],
    stream_duration: float,
) -> tuple:
    """For SPIKE arcs: peak ± dynamically sized buffer based on surrounding signal."""
    peak = arc.peak_time

    # Look backward for a speech spike (setup / "watch this" moment)
    start = max(0.0, peak - 5.0)
    for sec in reversed([s for s in timeline if (peak - 6.0) <= s.timestamp < peak]):
        if sec.speech_score > 0.40 or sec.fused_score > 0.50:
            start = min(start, sec.timestamp - 0.5)
            break

    # Look forward for an emotion spike (the reaction)
    end = min(stream_duration or (peak + 8.0), peak + 8.0)
    for sec in [s for s in timeline if peak < s.timestamp <= (peak + 10.0)]:
        if sec.emotion_score > 0.40 or sec.speech_score > 0.35:
            end = max(end, sec.timestamp + 0.5)
            break

    return max(0.0, start), end


# ─────────────────────────────────────────────────────────────────────────────
# Legacy DetectedEvent expansion (kept for any old call-sites)
# ─────────────────────────────────────────────────────────────────────────────

def expand(
    event: DetectedEvent,
    timeline: List[TimelineSecond],
    profile: GameProfile,
    transcript_data: List[Dict],
) -> HighlightCandidate:
    """
    Legacy expander for DetectedEvent objects.
    Wraps the event as a synthetic SPIKE arc and delegates to expand_arc.
    """
    # Build a minimal ArcRegion from the DetectedEvent
    arc = ArcRegion(
        shape_type=ArcShape.SPIKE,
        start=max(0.0, event.timestamp - 5.0),
        end=event.timestamp + 8.0,
        peak_time=event.timestamp,
        quality_score=event.score,
        composite_values=[event.score],
        transcript="",
        label=event.event_type,
    )
    return expand_arc(arc, timeline, profile, transcript_data)


def merge_overlapping(
    candidates: List[HighlightCandidate],
    rules: Dict,
) -> List[HighlightCandidate]:
    """
    Merges adjacent highlight candidates that are within merge_threshold seconds.
    """
    if not candidates:
        return []

    sorted_cands = sorted(candidates, key=lambda c: c.start)
    merged = [sorted_cands[0]]
    merge_threshold = rules.get("merge_threshold", 5.0)

    for current in sorted_cands[1:]:
        last = merged[-1]
        gap  = current.start - last.end
        if gap <= merge_threshold:
            last.end   = max(last.end, current.end)
            last.score = max(last.score, current.score)
            if hasattr(last, "events") and last.events:
                last.events.extend(current.events or [])
            last.reason = f"{last.reason} + merged"
        else:
            merged.append(current)

    return merged
