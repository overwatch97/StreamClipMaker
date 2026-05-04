"""
editing_brain.py — Human Editor Logic
======================================
Polishes arc boundaries and validates clips before they go to the renderer.

KEY CHANGE from previous version:
  - The hard-coded "travel rejection" rules (visual > 0.3 and emotion < 0.2
    → reject) are REMOVED. ArcDetector handles this: boring flat travel never
    rises above the elevation threshold and never becomes an arc.
  - validate_clip_logic() now uses arc-shape-aware duration rules from
    ARC_DURATION_RULES instead of a single fixed window.
  - The payoff check is shape-aware: tension arcs MUST resolve (signal drops),
    spike arcs just need a strong peak.
"""

import logging
from typing import List, Dict, Optional

from phase3_types import ARC_DURATION_RULES

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary Snapping
# ─────────────────────────────────────────────────────────────────────────────

def find_nearest_word_boundary(
    timestamp: float,
    transcript_data: List[Dict],
    search_window: float = 4.0,
) -> float:
    """
    Snaps a timestamp to the nearest natural speech boundary
    (sentence-end punctuation > long pause > any word gap).
    Falls back to original timestamp if no transcript words are nearby.
    """
    all_words = []
    for segment in transcript_data:
        all_words.extend(segment.get("words", []))

    if not all_words:
        return timestamp

    near_words = [w for w in all_words if abs(w["start"] - timestamp) < search_window]
    if not near_words:
        return timestamp

    near_words.sort(key=lambda x: x["start"])

    best_boundary = timestamp
    min_dist = float("inf")

    for i in range(len(near_words) - 1):
        w1 = near_words[i]
        w2 = near_words[i + 1]

        gap_start  = w1["end"]
        gap_end    = w2["start"]
        gap_center = (gap_start + gap_end) / 2.0
        gap_dur    = gap_end - gap_start
        dist       = abs(gap_center - timestamp)

        text = w1.get("text", w1.get("word", "")).strip()
        has_punctuation = text.endswith((".", "!", "?", ","))

        effective_dist = dist
        if has_punctuation:
            effective_dist *= 0.1    # Strongly prefer sentence ends
        elif gap_dur > 0.4:
            effective_dist *= 0.3    # Prefer natural pauses

        if effective_dist < min_dist:
            min_dist = effective_dist
            best_boundary = gap_center

    return best_boundary


# ─────────────────────────────────────────────────────────────────────────────
# Score Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_avg_score(timeline, start_t: float, end_t: float) -> float:
    if not timeline:
        return 1.0
    scores = []
    for t in timeline:
        if start_t <= t.timestamp <= end_t:
            meta = getattr(t, "metadata", {})
            if meta and "scores" in meta:
                scores.append(meta["scores"]["total"])
            elif hasattr(t, "fused_score") and t.fused_score > 0:
                scores.append(t.fused_score)
    return sum(scores) / len(scores) if scores else 1.0


def get_payoff_score(timeline, start_t: float, end_t: float) -> float:
    if not timeline:
        return 1.0
    duration = end_t - start_t
    payoff_start = end_t - (duration * 0.25)
    scores = []
    for t in timeline:
        if payoff_start <= t.timestamp <= end_t:
            scores.append((t.emotion_score + t.speech_score) / 2.0)
    return sum(scores) / len(scores) if scores else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Arc-Aware Clip Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_clip_logic(
    arc: Dict,
    transcript_data: List[Dict],
    profile=None,
    timeline=None,
) -> bool:
    """
    Validates whether a clip passes quality criteria.

    Rules are shape-aware:
      - Each ArcShape has its own min/max duration window
      - Tension arcs must resolve (end signal drops)
      - All arcs must have a minimum hook clarity
      - The hard travel rejection rule (visual > 0.3 → reject) is GONE

    Args:
        arc: clipper-json dict with keys: start, end, shape_type, score, evidence, etc.
        transcript_data: Whisper transcript segments
        profile: GameProfile object (optional — not used for hard rules anymore)
        timeline: list of TimelineSecond objects (optional, for score lookups)
    """
    start_time = arc.get("start", 0.0)
    end_time   = arc.get("end", 0.0)
    duration   = end_time - start_time
    shape_type = arc.get("shape_type", "combat")
    peak_time  = arc.get("peak_time", start_time + 2.0)

    # ── 1. Duration check (shape-aware) ─────────────────────────────────────
    min_dur, max_dur = ARC_DURATION_RULES.get(shape_type, (5, 60))
    if duration < min_dur or duration > max_dur:
        logger.debug(
            f"  Rejected [{shape_type}] at {start_time:.1f}s — "
            f"duration {duration:.1f}s outside [{min_dur}, {max_dur}]"
        )
        return False

    # ── 2. Quality score gate ────────────────────────────────────────────────
    score = arc.get("score", 0)
    if score < 15:   # quality_score < 0.15 expressed as 0-100 int
        logger.debug(f"  Rejected [{shape_type}] at {start_time:.1f}s — quality too low ({score})")
        return False

    # ── 3. Hook clarity check ────────────────────────────────────────────────
    if timeline:
        clarity_score = get_avg_score(timeline, start_time, start_time + 2.0)
        peak_offset   = peak_time - start_time
        peak_score_n  = score / 100.0
        if clarity_score < 0.12:
            # Hook Forgiveness: allow if the peak lands very early and is very strong
            if not (peak_offset <= 2.5 and peak_score_n >= 0.75):
                logger.debug(
                    f"  Rejected [{shape_type}] at {start_time:.1f}s — "
                    f"confusing hook (clarity={clarity_score:.3f})"
                )
                return False

    # ── 4. Tension arc resolution check ─────────────────────────────────────
    # A TENSION arc that is still at peak signal at the end hasn't resolved yet.
    # Don't clip it — the story isn't finished.
    if shape_type == "travel":
        evidence    = arc.get("evidence", {})
        end_comp    = evidence.get("end_composite", None)
        peak_audio  = evidence.get("peak_audio", 0.5)
        peak_motion = evidence.get("peak_motion", 0.5)
        peak_sig    = (peak_audio + peak_motion) / 2.0
        if end_comp is not None and peak_sig > 0.1:
            resolution_ratio = end_comp / (peak_sig + 0.01)
            if resolution_ratio > 0.85:
                logger.debug(
                    f"  Rejected [travel] at {start_time:.1f}s — "
                    f"no resolution (end_comp={end_comp:.3f} / peak={peak_sig:.3f})"
                )
                return False

    # ── 5. Payoff strength check ─────────────────────────────────────────────
    # Tension arcs don't need a speech/emotion payoff — their payoff is the resolution drop,
    # which we already checked above.
    if timeline and shape_type != "travel":
        payoff = get_payoff_score(timeline, start_time, end_time)
        payoff_threshold = 0.20 if shape_type in (
            "neutral", "surprise"
        ) else 0.30
        if payoff < payoff_threshold:
            logger.debug(
                f"  Rejected [{shape_type}] at {start_time:.1f}s — "
                f"weak payoff ({payoff:.3f} < {payoff_threshold})"
            )
            return False


    return True


# ─────────────────────────────────────────────────────────────────────────────
# Hook Trimmer
# ─────────────────────────────────────────────────────────────────────────────

def fix_clip_hook(arc: Dict, transcript_data: List[Dict]) -> Dict:
    """
    Trims the start of SPIKE arcs so the action lands within the first 2.5s.
    TENSION / DRAMA / TRIUMPH arcs are left untouched — their setup IS the value.
    """
    shape_type = arc.get("shape_type", "combat")

    # Only aggressively trim spike arcs — other shapes need their build-up
    if shape_type not in ("combat", "reaction"):
        return arc

    start_time = arc["start"]
    peak_time  = arc.get("peak_time", start_time + 2.0)

    if (peak_time - start_time) > 2.5:
        new_start = peak_time - 1.8
        arc["start"] = find_nearest_word_boundary(
            new_start, transcript_data, search_window=1.0
        )

    return arc


# ─────────────────────────────────────────────────────────────────────────────
# Main Polish Pass
# ─────────────────────────────────────────────────────────────────────────────

def refine_clips_for_social(
    arcs: List[Dict],
    transcript_data: List[Dict],
    profile=None,
    timeline=None,
) -> List[Dict]:
    """
    Final polish pass before clips go to the renderer.

    Steps:
      1. Snap boundaries to natural speech boundaries
      2. Fix hooks on SPIKE/COMEDY arcs (trim fat before the action)
      3. Validate with shape-aware rules
      4. Return only passing clips
    """
    refined = []
    for arc in arcs:
        shape_type = arc.get("shape_type", "combat")

        # 1. Natural boundary snapping
        #    Tension/Drama/Triumph: wider snap window to preserve setup/payoff
        snap_window = 3.0 if shape_type in (
            "travel", "neutral", "combat"
        ) else 1.5

        arc["start"] = find_nearest_word_boundary(
            arc["start"], transcript_data, search_window=snap_window
        )
        arc["end"] = find_nearest_word_boundary(
            arc["end"], transcript_data, search_window=snap_window
        )

        # 2. Hook fix (spike / comedy only)
        arc = fix_clip_hook(arc, transcript_data)

        # 3. Validate
        if validate_clip_logic(arc, transcript_data, profile=profile, timeline=timeline):
            refined.append(arc)
        else:
            logger.info(
                f"  Clip rejected by editing_brain: [{shape_type}] "
                f"{arc.get('start', 0):.1f}s – {arc.get('end', 0):.1f}s"
            )

    logger.info(f"editing_brain: {len(refined)}/{len(arcs)} clips passed validation.")
    return refined
