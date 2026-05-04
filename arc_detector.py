"""
arc_detector.py — Dynamic Signal-Shape Intelligence Engine
===========================================================
Replaces the rule-based event detection system.

Instead of asking "does this second match a predefined event?", this module
reads the *mathematical shape* of the composite signal curve over time and
identifies what kind of moment it is — regardless of game or genre.

The six arc shapes (SPIKE, TENSION, COMEDY, DRAMA, TRIUMPH, DISCOVERY) cover
every type of compelling gaming moment a human editor would pick.
"""

import logging
import statistics
from typing import List, Dict, Optional, Tuple

from phase3_types import (
    ArcShape, ArcRegion, ARC_DURATION_RULES,
    TimelineSecond, GameProfile
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal Weights (used to build the composite curve)
# These are the default weights when no profile overrides are provided.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "audio":   0.30,
    "motion":  0.30,
    "emotion": 0.20,
    "speech":  0.20,
}


def _get_weights(profile: Optional[GameProfile]) -> Dict[str, float]:
    """Merge profile score_weights with defaults."""
    if not profile or not profile.score_weights:
        return DEFAULT_WEIGHTS
    w = {
        "audio":   profile.score_weights.get("audio",   DEFAULT_WEIGHTS["audio"]),
        "motion":  profile.score_weights.get("visual",  DEFAULT_WEIGHTS["motion"]),
        "emotion": profile.score_weights.get("emotion", DEFAULT_WEIGHTS["emotion"]),
        "speech":  profile.score_weights.get("speech",  DEFAULT_WEIGHTS["speech"]),
    }
    # Normalise so weights always sum to 1.0
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Arc Detector
# ─────────────────────────────────────────────────────────────────────────────

class ArcDetector:
    """
    Detects narrative arc regions from the per-second signal timeline.

    Pipeline:
        1. Build composite signal curve from 4 modalities
        2. Compute global baseline (median + std) — the "normal" level
        3. Find elevated regions (contiguous segments above threshold)
        4. Classify each region's *shape* (SPIKE / TENSION / COMEDY / DRAMA / TRIUMPH / DISCOVERY)
        5. Score each arc for clip quality
        6. Filter by duration and quality gates
    """

    # Elevation threshold: how many standard deviations above median counts as "elevated"
    ELEVATION_SIGMA = 0.45

    # Minimum elevation above median (absolute floor, sigma-independent)
    MIN_ELEVATION_ABOVE_MEDIAN = 0.08

    # Minimum seconds a region must be elevated to be considered an arc
    MIN_REGION_DURATION = 5.0

    def detect(
        self,
        timeline: List[TimelineSecond],
        profile: Optional[GameProfile] = None,
        transcript_data: Optional[List[Dict]] = None,
    ) -> List[ArcRegion]:
        """
        Main entry point. Returns detected ArcRegions sorted by quality (best first).
        """
        if not timeline:
            return []

        weights = _get_weights(profile)
        ts, composite = self._build_composite(timeline, weights)

        if not composite:
            return []

        baseline_median, baseline_std = self._compute_baseline(composite)
        threshold = baseline_median + max(
            self.ELEVATION_SIGMA * baseline_std,
            self.MIN_ELEVATION_ABOVE_MEDIAN
        )

        logger.info(
            f"ArcDetector: baseline median={baseline_median:.3f}, "
            f"std={baseline_std:.3f}, threshold={threshold:.3f}"
        )

        raw_regions = self._find_elevated_regions(ts, composite, threshold)
        arcs = []
        for region_ts, region_comp in raw_regions:
            arc = self._classify_and_score(
                region_ts, region_comp, timeline, profile, transcript_data
            )
            if arc is not None:
                arcs.append(arc)

        # Deduplicate arcs that substantially overlap
        arcs = self._deduplicate(arcs)

        # Sort best quality first
        arcs.sort(key=lambda a: a.quality_score, reverse=True)

        logger.info(f"ArcDetector: {len(arcs)} arcs detected from {len(timeline)} seconds of timeline.")
        return arcs

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Build composite curve
    # ─────────────────────────────────────────────────────────────────────────

    def _build_composite(
        self,
        timeline: List[TimelineSecond],
        weights: Dict[str, float],
    ) -> Tuple[List[float], List[float]]:
        """Returns (timestamps, composite_values) aligned lists."""
        ts_list = []
        comp_list = []
        for sec in timeline:
            if sec.is_ignore_state:
                # Ignore states (loading screens etc.) are clamped to 0
                ts_list.append(sec.timestamp)
                comp_list.append(0.0)
                continue
            val = (
                weights["audio"]   * sec.audio_score   +
                weights["motion"]  * sec.visual_score  +
                weights["emotion"] * sec.emotion_score +
                weights["speech"]  * sec.speech_score
            )
            ts_list.append(sec.timestamp)
            comp_list.append(min(max(val, 0.0), 1.0))
        return ts_list, comp_list

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Baseline
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_baseline(self, composite: List[float]) -> Tuple[float, float]:
        if not composite:
            return 0.0, 0.0
        sorted_vals = sorted(composite)
        # Use 30th percentile as the "quiet" baseline.
        # The median is skewed when a long exciting arc occupies >50% of the stream.
        p30_idx = max(0, int(len(sorted_vals) * 0.30) - 1)
        baseline = sorted_vals[p30_idx]
        # Std is still computed over the full distribution for sensitivity
        std = statistics.pstdev(composite) if len(composite) > 1 else 0.05
        return baseline, max(std, 0.01)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Find elevated regions (contiguous above threshold)
    # ─────────────────────────────────────────────────────────────────────────

    def _find_elevated_regions(
        self,
        ts: List[float],
        comp: List[float],
        threshold: float,
    ) -> List[Tuple[List[float], List[float]]]:
        """
        Groups contiguous seconds where composite >= threshold into regions.
        A gap of ≤ 3 seconds (3 samples) is bridged to avoid splitting one arc
        into many small fragments.
        """
        BRIDGE_GAP = 6  # samples — bridge gaps up to 6s to avoid splitting one arc

        regions = []
        in_region = False
        region_ts: List[float] = []
        region_comp: List[float] = []
        below_count = 0

        for i, (t, v) in enumerate(zip(ts, comp)):
            if v >= threshold:
                if not in_region:
                    in_region = True
                    region_ts = []
                    region_comp = []
                    below_count = 0
                region_ts.append(t)
                region_comp.append(v)
                below_count = 0
            else:
                if in_region:
                    below_count += 1
                    region_ts.append(t)
                    region_comp.append(v)
                    if below_count > BRIDGE_GAP:
                        # End of region — trim the trailing below-threshold samples
                        trim_idx = len(region_ts) - below_count
                        final_ts = region_ts[:trim_idx]
                        final_comp = region_comp[:trim_idx]
                        dur = final_ts[-1] - final_ts[0] if len(final_ts) > 1 else 0
                        if dur >= self.MIN_REGION_DURATION:
                            regions.append((final_ts, final_comp))
                        in_region = False
                        region_ts = []
                        region_comp = []
                        below_count = 0

        # Close any open region at end of stream
        if in_region and region_ts:
            if below_count > 0:
                trim_idx = len(region_ts) - below_count
                region_ts = region_ts[:trim_idx]
                region_comp = region_comp[:trim_idx]
            dur = region_ts[-1] - region_ts[0] if len(region_ts) > 1 else 0
            if dur >= self.MIN_REGION_DURATION:
                regions.append((region_ts, region_comp))

        return regions

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — Classify shape + Step 5 — Score quality
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_and_score(
        self,
        region_ts: List[float],
        region_comp: List[float],
        timeline: List[TimelineSecond],
        profile: Optional[GameProfile],
        transcript_data: Optional[List[Dict]],
    ) -> Optional[ArcRegion]:
        """
        Classifies a region into an ArcShape and scores its clip quality.
        Returns None if the region fails basic quality gates.
        """
        if len(region_ts) < 2:
            return None

        duration = region_ts[-1] - region_ts[0]
        peak_idx = region_comp.index(max(region_comp))
        peak_time = region_ts[peak_idx]
        peak_val = region_comp[peak_idx]
        end_val = region_comp[-1]
        start_val = region_comp[0]
        mean_val = statistics.mean(region_comp)

        # Pull per-modality peaks for enrichment
        region_secs = [
            s for s in timeline
            if region_ts[0] <= s.timestamp <= region_ts[-1]
        ]
        peak_audio   = max((s.audio_score   for s in region_secs), default=0.0)
        peak_motion  = max((s.visual_score  for s in region_secs), default=0.0)
        peak_emotion = max((s.emotion_score for s in region_secs), default=0.0)
        peak_speech  = max((s.speech_score  for s in region_secs), default=0.0)

        # Derivatives: slope of first and last third
        third = max(1, len(region_comp) // 3)
        rise_slope  = self._slope(region_comp[:third])
        fall_slope  = self._slope(region_comp[-third:])
        mid_variance = statistics.pvariance(region_comp[third:-third]) if len(region_comp) > 6 else 0.0

        # For tension resolution, look at the signal shortly AFTER the arc ends
        post_region_secs = [
            s for s in timeline
            if region_ts[-1] < s.timestamp <= region_ts[-1] + 4.0
        ]
        weights = _get_weights(profile)
        post_vals = [
            weights["audio"]*s.audio_score + weights["motion"]*s.visual_score + 
            weights["emotion"]*s.emotion_score + weights["speech"]*s.speech_score
            for s in post_region_secs
        ]
        resolved_val = min(post_vals) if post_vals else region_comp[-1]

        # Silence before region (for COMEDY detection)
        secs_before_region = [
            s for s in timeline
            if (region_ts[0] - 30) <= s.timestamp < region_ts[0]
        ]
        pre_region_mean = statistics.mean(
            [s.audio_score * 0.5 + s.visual_score * 0.5 for s in secs_before_region]
        ) if secs_before_region else 0.5

        # ── Shape Classification ──────────────────────────────────────────────
        shape = self._classify_shape(
            duration=duration,
            rise_slope=rise_slope,
            fall_slope=fall_slope,
            mid_variance=mid_variance,
            peak_val=peak_val,
            start_val=start_val,
            end_val=end_val,
            mean_val=mean_val,
            peak_speech=peak_speech,
            peak_emotion=peak_emotion,
            peak_motion=peak_motion,
            peak_audio=peak_audio,
            pre_region_mean=pre_region_mean,
            resolved_val=resolved_val,
        )

        # ── Duration Gate ────────────────────────────────────────────────────
        min_dur, max_dur = ARC_DURATION_RULES.get(shape.value, (5, 90))
        if duration < min_dur:
            return None  # Too short for this shape

        # Clip the arc to max duration if needed (keep the peak centered)
        clipped_start = region_ts[0]
        clipped_end   = region_ts[-1]
        if duration > max_dur:
            clipped_start, clipped_end = self._clip_to_max(
                region_ts, region_comp, peak_time, max_dur
            )

        # ── Quality Score ────────────────────────────────────────────────────
        quality = self._score_quality(
            shape=shape,
            peak_val=peak_val,
            mean_val=mean_val,
            end_val=end_val,
            duration=min(duration, max_dur),
            peak_emotion=peak_emotion,
            peak_speech=peak_speech,
        )

        if quality < 0.20:
            return None

        # ── Transcript ───────────────────────────────────────────────────────
        transcript = ""
        if transcript_data:
            transcript = _collect_transcript_in_range(
                transcript_data, clipped_start, clipped_end
            )

        return ArcRegion(
            shape_type=shape,
            start=round(clipped_start, 3),
            end=round(clipped_end, 3),
            peak_time=round(peak_time, 3),
            quality_score=round(quality, 4),
            composite_values=region_comp,
            peak_audio=round(peak_audio, 3),
            peak_motion=round(peak_motion, 3),
            peak_emotion=round(peak_emotion, 3),
            peak_speech=round(peak_speech, 3),
            end_composite=round(resolved_val, 3),
            transcript=transcript,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Shape classification rules
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_shape(
        self,
        duration: float,
        rise_slope: float,
        fall_slope: float,
        mid_variance: float,
        peak_val: float,
        start_val: float,
        end_val: float,
        mean_val: float,
        peak_speech: float,
        peak_emotion: float,
        peak_motion: float,
        peak_audio: float,
        pre_region_mean: float,
        resolved_val: float,
    ) -> ArcShape:
        """
        Classifies an elevated region into one of 6 arc shapes using
        purely mathematical signal features — no game-specific knowledge.

        Decision tree (in priority order):
          DRAMA     — dominant speech+emotion, low motion
          COMEDY    — came from silence, fast spike
          TRIUMPH   — high mid-variance (oscillating) + strong peak at end
          TENSION   — long, gradual rise, sustained plateau, notable drop at end
          DISCOVERY — steep simultaneous shift across multiple modalities
          SPIKE     — default: short/sharp
        """
        # DRAMA: speech and emotion dominate, motion is secondary
        is_speech_dominant = peak_speech > 0.55 and peak_emotion > 0.40
        is_low_motion      = peak_motion < 0.45
        if is_speech_dominant and is_low_motion and duration >= 10:
            return ArcShape.DRAMA

        # COMEDY: long silence before → sudden surprise spike
        came_from_silence = pre_region_mean < 0.18
        fast_onset        = rise_slope > 0.06  # rises steeply at start
        if came_from_silence and fast_onset and duration <= 25:
            return ArcShape.COMEDY

        # TRIUMPH: high mid-section variance (struggle) + peak at the end
        mid_struggle      = mid_variance > 0.012
        peak_at_end       = end_val >= (peak_val * 0.70) and end_val > mean_val
        if mid_struggle and peak_at_end and duration >= 15:
            return ArcShape.TRIUMPH

        # TENSION: long, gradual or moderate rise → sustained → notable drop at end
        # allow slight negative slope for oscillating signals
        gradual_rise      = rise_slope > -0.02 and rise_slope < 0.08
        sustained_plateau = mean_val > 0.28
        drops_at_end      = resolved_val < (peak_val * 0.85)
        if gradual_rise and sustained_plateau and drops_at_end and duration >= 20:
            return ArcShape.TENSION

        # DISCOVERY: steep onset AND all major modalities contribute (cross-modal shift)
        cross_modal       = (peak_audio > 0.30 and peak_motion > 0.30 and
                             (peak_speech > 0.25 or peak_emotion > 0.25))
        steep_onset       = rise_slope > 0.04
        if steep_onset and cross_modal and 8 <= duration <= 45:
            return ArcShape.DISCOVERY

        # Default: SPIKE
        return ArcShape.SPIKE

    # ─────────────────────────────────────────────────────────────────────────
    # Quality scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _score_quality(
        self,
        shape: ArcShape,
        peak_val: float,
        mean_val: float,
        end_val: float,
        duration: float,
        peak_emotion: float,
        peak_speech: float,
    ) -> float:
        """
        Scores arc quality 0.0–1.0.  Shape-specific rules reward what makes
        each arc type compelling to an audience.
        """
        # Base: how far above baseline is the peak?
        intensity = min(peak_val * 1.2, 1.0)

        if shape == ArcShape.SPIKE:
            # Pure intensity — the spike itself is the moment
            quality = intensity * 0.8 + peak_emotion * 0.2

        elif shape == ArcShape.TENSION:
            # Great tension arcs have: sustained high mean + a clear drop at end (resolution)
            resolution_bonus = max(0.0, (peak_val - end_val) / (peak_val + 0.01))
            quality = mean_val * 0.5 + resolution_bonus * 0.35 + intensity * 0.15

        elif shape == ArcShape.COMEDY:
            # Comedy = surprise. Reward sharpness of spike relative to arc mean
            surprise = min((peak_val - mean_val) / (mean_val + 0.01), 2.0) / 2.0
            quality = surprise * 0.6 + peak_emotion * 0.3 + intensity * 0.1

        elif shape == ArcShape.DRAMA:
            # Drama needs sustained emotional weight and coherent speech
            quality = peak_emotion * 0.4 + peak_speech * 0.4 + mean_val * 0.2

        elif shape == ArcShape.TRIUMPH:
            # Triumph: the peak at the end is the payoff — reward high end_val
            end_bonus = min(end_val * 1.5, 1.0)
            quality = end_bonus * 0.5 + mean_val * 0.3 + intensity * 0.2

        elif shape == ArcShape.DISCOVERY:
            # Discovery: cross-modal coherence is the signal — reward mean + intensity
            quality = intensity * 0.5 + mean_val * 0.35 + peak_emotion * 0.15

        else:
            quality = intensity * 0.7 + mean_val * 0.3

        # Duration bonus: prefer arcs that are well within their shape's ideal range
        min_dur, max_dur = ARC_DURATION_RULES.get(shape.value, (5, 60))
        ideal_dur = (min_dur + max_dur) / 2.0
        dur_factor = 1.0 - min(abs(duration - ideal_dur) / ideal_dur, 0.4)
        quality = quality * (0.85 + 0.15 * dur_factor)

        return min(max(quality, 0.0), 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _slope(values: List[float]) -> float:
        """Simple linear slope of a list of values (rise per sample)."""
        n = len(values)
        if n < 2:
            return 0.0
        return (values[-1] - values[0]) / max(n - 1, 1)

    @staticmethod
    def _clip_to_max(
        region_ts: List[float],
        region_comp: List[float],
        peak_time: float,
        max_dur: float,
    ) -> Tuple[float, float]:
        """
        When a region exceeds max_dur, clip it symmetrically around the peak.
        Biases slightly toward including more *post-peak* content (the payoff).
        """
        half = max_dur / 2.0
        pre_budget  = half * 0.45
        post_budget = half * 0.55
        start = max(region_ts[0], peak_time - pre_budget)
        end   = min(region_ts[-1], peak_time + post_budget)
        # If post-budget overflows, shift extra to pre
        if (end - start) < max_dur:
            shortfall = max_dur - (end - start)
            start = max(region_ts[0], start - shortfall)
        return round(start, 3), round(end, 3)

    def _deduplicate(self, arcs: List[ArcRegion]) -> List[ArcRegion]:
        """Remove arcs that are >80% overlapping with a higher-quality arc."""
        arcs_sorted = sorted(arcs, key=lambda a: a.quality_score, reverse=True)
        kept: List[ArcRegion] = []
        for arc in arcs_sorted:
            overlap = False
            for existing in kept:
                inter_start = max(arc.start, existing.start)
                inter_end   = min(arc.end,   existing.end)
                inter_dur   = max(0.0, inter_end - inter_start)
                arc_dur     = max(arc.duration, 0.01)
                if inter_dur / arc_dur > 0.80:
                    overlap = True
                    break
            if not overlap:
                kept.append(arc)
        return kept


# ─────────────────────────────────────────────────────────────────────────────
# Utility: collect transcript text within a time range
# ─────────────────────────────────────────────────────────────────────────────

def _collect_transcript_in_range(
    transcript_data: List[Dict],
    start: float,
    end: float,
) -> str:
    """Gathers spoken words from the transcript within [start, end]."""
    words = []
    for segment in transcript_data:
        seg_start = segment.get("start", 0.0)
        seg_end   = segment.get("end", 0.0)
        if seg_end < start or seg_start > end:
            continue
        # Word-level if available
        if "words" in segment:
            for w in segment["words"]:
                ws = w.get("start", seg_start)
                if start <= ws <= end:
                    words.append(w.get("word", w.get("text", "")).strip())
        else:
            words.append(segment.get("text", "").strip())
    return " ".join(w for w in words if w)
