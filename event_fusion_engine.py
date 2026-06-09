import logging
import statistics
from typing import List, Dict, Optional, Tuple
import numpy as np

from phase3_types import EventMoment, TimelineSecond, EVENT_PRIORITIES
from pacing_profiles import get_pacing_profile, resolve_pacing_style

logger = logging.getLogger(__name__)


class RejectionStats:
    """Tracks why moments were rejected — for diagnostics and threshold tuning."""

    def __init__(self):
        self.total_raw = 0
        self.rejected_duration = 0
        self.rejected_prominence = 0
        self.rejected_flat_curve = 0
        self.rejected_intensity = 0
        self.rejected_surprise_payoff = 0
        self.rejected_std_dev = 0
        self.rejected_continuity = 0  # tracked in hook_analyzer
        self.final_valid = 0
        self.peak_score_seen = 0.0
        self.best_rejected = None  # (score, reason) of the closest-to-threshold reject

    def record_peak(self, score: float):
        if score > self.peak_score_seen:
            self.peak_score_seen = score

    def record_reject(self, score: float, reason: str):
        if self.best_rejected is None or score > self.best_rejected[0]:
            self.best_rejected = (score, reason)

    def print_summary(self, pacing_style: str, thresholds: dict):
        total_rejected = (
            self.rejected_duration + self.rejected_prominence +
            self.rejected_flat_curve + self.rejected_intensity +
            self.rejected_surprise_payoff + self.rejected_std_dev
        )
        print(f"\n  -- Fusion Engine Rejection Report (pacing={pacing_style}) --", flush=True)
        print(f"  Raw segments evaluated   : {self.total_raw}", flush=True)
        print(f"  Rejected (duration)      : {self.rejected_duration}", flush=True)
        print(f"  Rejected (prominence)    : {self.rejected_prominence}  [threshold: {thresholds['min_prominence']:.2f}]", flush=True)
        print(f"  Rejected (flat curve)    : {self.rejected_flat_curve}", flush=True)
        print(f"  Rejected (intensity)     : {self.rejected_intensity}  [threshold: {thresholds['min_intensity']:.2f}]", flush=True)
        print(f"  Rejected (surp/payoff)   : {self.rejected_surprise_payoff}  [threshold: {thresholds['min_surprise_payoff']:.2f}]", flush=True)
        print(f"  Rejected (std_dev flat)  : {self.rejected_std_dev}", flush=True)
        print(f"  Final valid events       : {self.final_valid}", flush=True)
        print(f"  Peak score in stream     : {self.peak_score_seen:.4f}", flush=True)
        if self.best_rejected:
            print(f"  Closest rejected event   : score={self.best_rejected[0]:.4f}  reason='{self.best_rejected[1]}'", flush=True)
        print(f"  -----------------------------------------------------", flush=True)


class EventFusionEngine:
    """
    Detects highlight moments from a multimodal timeline using adaptive
    pacing-aware thresholds. Profiles are resolved from the game's genre
    so FPS games stay strict while cinematic games use relaxed detection.
    """

    # Structural constants — not tuned per pacing style
    MIN_EVENT_DURATION = 3.0
    MAX_EVENT_DURATION = 15.0
    GAP_MERGE_THRESHOLD = 1.5

    # Human-feel pacing buffers — range values, interpolated per profile
    # PRE_CONTEXT: how far before the peak we start the clip (gives viewer setup)
    # POST_PAYOFF: how long after the peak we hold (lets aftermath breathe)
    PRE_CONTEXT_BUFFER_FPS       = 1.5   # tight — FPS is fast-cut
    PRE_CONTEXT_BUFFER_CINEMATIC = 2.5   # wide — cinematic needs setup window
    PRE_CONTEXT_BUFFER_BALANCED  = 2.0   # middle ground
    POST_PAYOFF_BUFFER_FPS       = 1.0   # short — keep FPS snappy
    POST_PAYOFF_BUFFER_CINEMATIC = 2.0   # long — preserve laugh/silence/aftermath
    POST_PAYOFF_BUFFER_BALANCED  = 1.5   # middle ground

    # Legacy aliases kept for fallback compatibility
    PRE_BUFFER  = 2.0
    POST_BUFFER = 1.5

    def __init__(self):
        self._stats = RejectionStats()
        self._pacing_style = "balanced"
        self._thresholds = get_pacing_profile("balanced")
        self._pre_buffer  = self.PRE_CONTEXT_BUFFER_BALANCED
        self._post_buffer = self.POST_PAYOFF_BUFFER_BALANCED
        self._is_racing = False
        self._racing_weights = {}
        self._racing_clip_rules = {}

    def _load_thresholds(self, genre: str, pacing_style_override: str = None, profile=None, is_racing: bool = False):
        """Resolve pacing style from genre or explicit override, then load thresholds."""
        self._pacing_style = resolve_pacing_style(genre, pacing_style_override)
        self._thresholds = dict(get_pacing_profile(self._pacing_style))
        # Set pacing-aware clip buffers for human-feel editing
        self._pre_buffer, self._post_buffer = self._resolve_pacing_buffers(self._pacing_style)
        if is_racing:
            thresholds = getattr(profile, "thresholds", {}) or {}
            clip_rules = getattr(profile, "clip_rules", {}) or {}
            weights = getattr(profile, "weights", {}) or {}
            self._racing_weights = weights
            self._racing_clip_rules = clip_rules

            # Racing thresholds are intentionally relaxed to fix over-rejection first.
            # Some early false positives are acceptable; later tuning can reduce noise.
            self._thresholds["start_threshold"] = thresholds.get("start_threshold", 0.38)
            self._thresholds["min_intensity"] = thresholds.get("intensity_threshold", 0.38)
            self._thresholds["min_prominence"] = thresholds.get("prominence_threshold", 0.04) * 0.3
            self._thresholds["min_surprise_payoff"] = thresholds.get("surprise_payoff_threshold", 0.20)
            self._thresholds["end_threshold"] = min(
                self._thresholds.get("end_threshold", 0.28),
                self._thresholds["start_threshold"] * 0.75,
            )
            self._pre_buffer = float(clip_rules.get("pre_context", 8.0))
            self._post_buffer = float(clip_rules.get("post_payoff", 10.0))
        print(
            f"  [EventFusion] Pacing style: '{self._pacing_style}' "
            f"(start>={self._thresholds['start_threshold']:.2f}, "
            f"intensity>={self._thresholds['min_intensity']:.2f}, "
            f"pre_ctx={self._pre_buffer:.1f}s, post_payoff={self._post_buffer:.1f}s)",
            flush=True,
        )
        if is_racing:
            self._print_racing_thresholds()

    def _resolve_pacing_buffers(self, pacing_style: str) -> Tuple[float, float]:
        """
        Return (pre_context_secs, post_payoff_secs) for the given pacing style.
        Cinematic → wider buffers for setup + aftermath.
        FPS → tighter buffers to preserve snappy cut feel.
        """
        mapping = {
            "fps":       (self.PRE_CONTEXT_BUFFER_FPS,       self.POST_PAYOFF_BUFFER_FPS),
            "cinematic": (self.PRE_CONTEXT_BUFFER_CINEMATIC, self.POST_PAYOFF_BUFFER_CINEMATIC),
            "balanced":  (self.PRE_CONTEXT_BUFFER_BALANCED,  self.POST_PAYOFF_BUFFER_BALANCED),
        }
        return mapping.get(pacing_style, (self.PRE_CONTEXT_BUFFER_BALANCED, self.POST_PAYOFF_BUFFER_BALANCED))

    def _print_racing_thresholds(self):
        weights = self._racing_weights or {}
        print("\n  [Racing Thresholds]", flush=True)
        print(f"  Pacing Profile     : {self._pacing_style}", flush=True)
        print(f"  Intensity          : {self._thresholds['min_intensity']:.3f}", flush=True)
        print(f"  Prominence         : {self._thresholds['min_prominence']:.3f}", flush=True)
        print(f"  Surprise/Payoff    : {self._thresholds['min_surprise_payoff']:.3f}", flush=True)
        print(f"  Momentum Weight    : {weights.get('momentum', 0.65):.3f}", flush=True)
        print(f"  Momentum Cap       : {weights.get('momentum_bonus_cap', 0.25):.3f}", flush=True)
        print(f"  Persistence Cap    : {weights.get('persistence_bonus_cap', 0.15):.3f}", flush=True)

    def _print_racing_decision(self, decision: str, reason: str, peak_time: float, feat: dict,
                               peak_score: float, peak_prominence: float, event_type: str = ""):
        print("\n  [Racing Decision]", flush=True)
        print(f"  t={peak_time:.1f}s", flush=True)
        print(f"  Decision           : {decision}", flush=True)
        if event_type:
            print(f"  Event              : {event_type}", flush=True)
        print(f"  Reason             : {reason}", flush=True)
        print(f"  Final Score        : {peak_score:.3f}", flush=True)
        print(f"  Base               : {feat.get('racing_score_base', 0.0):.3f}", flush=True)
        print(f"  Rolling Motion     : {feat.get('rolling_motion', 0.0):.3f}", flush=True)
        print(f"  Momentum Bonus     : {feat.get('momentum_bonus', 0.0):.3f}", flush=True)
        print(f"  Persistence Bonus  : {feat.get('persistence_bonus', 0.0):.3f}", flush=True)
        print(f"  Prominence         : {peak_prominence:.3f}", flush=True)
        print(
            f"  Closest Signals    : motion={feat.get('motion_delta', 0.0):.3f} "
            f"audio={feat.get('audio_peak', 0.0):.3f} "
            f"rolling_motion={feat.get('rolling_motion', 0.0):.3f}",
            flush=True,
        )
        print(f"  Tags               : {', '.join(feat.get('tags', [])) or 'none'}", flush=True)

    def detect(
        self,
        timeline: List[TimelineSecond],
        transcript_data: List[Dict] = None,
        genre: str = "general",
        pacing_style: str = None,
        profile=None,
    ) -> List[EventMoment]:
        if not timeline:
            return []

        self._stats = RejectionStats()
        is_racing = profile is not None and profile.genre == "racing"
        self._is_racing = is_racing
        self._load_thresholds(genre, pacing_style, profile=profile, is_racing=is_racing)
        t = self._thresholds  # shorthand

        times = []
        scores = []
        features_list = []

        # ── 1. Build per-window fusion scores ────────────────────────────────
        rolling_emotions = []
        motion_scores = []
        consecutive_high_motion_frames = 0
        racing_weights = getattr(profile, "weights", {}) if is_racing else {}
        for sec in timeline:
            meta = sec.metadata
            a_feat = meta.get("audio_features", {})
            e_feat = meta.get("emotion_features", {})
            v_feat = meta.get("visual_features", {})
            s_feat = meta.get("speech_features", {})

            audio_peak    = a_feat.get("audio_peak_norm", 0.0)
            emotion_score = e_feat.get("emotion_score_norm", 0.0) if e_feat else 0.0
            surprise_level= e_feat.get("surprise_level", 0.0) if e_feat else 0.0
            motion_delta  = v_feat.get("motion_delta_norm", 0.0)
            lateral_flow  = v_feat.get("lateral_flow_norm", v_feat.get("lateral_flow", 0.0))
            scene_conf    = v_feat.get("scene_confidence", 0.0)
            scene_type    = v_feat.get("scene_type", "neutral")
            speech_energy = s_feat.get("speech_energy_norm", 0.0)
            keyword_weight= s_feat.get("keyword_weight", 0.2)
            motion_scores.append(motion_delta)

            # Face Presence Boost
            rolling_emotions.append(emotion_score)
            if len(rolling_emotions) > 5:
                rolling_emotions.pop(0)
            local_emotion_avg = sum(rolling_emotions) / len(rolling_emotions)
            emotion_delta = emotion_score - local_emotion_avg
            engagement_level = e_feat.get("engagement_level", 0.0) if e_feat else 0.0
            face_visible = engagement_level > 0.01 or emotion_score > 0.05
            boost = 1.15 if (face_visible and emotion_score > 0.6 and emotion_delta > 0.15) else 1.0

            if not is_racing:
                # Protected legacy path: non-racing genres keep the original FPS/action formula.
                base_score = (
                    (audio_peak    * 0.30) +
                    (emotion_score * 0.25) +
                    (motion_delta  * 0.15) +
                    (scene_conf    * 0.10) +
                    (speech_energy * 0.10) +
                    (keyword_weight* 0.10)
                )
                final_score = min(base_score * boost, 1.0)
                rolling_motion = 0.0
                momentum_bonus = 0.0
                persistence_bonus = 0.0
                tags = []
            else:
                motion_weight = racing_weights.get("motion", 0.55)
                audio_weight = racing_weights.get("audio", 0.20)
                speech_weight = racing_weights.get("speech", 0.05)
                momentum_weight = racing_weights.get("momentum", 0.65)
                momentum_cap = racing_weights.get("momentum_bonus_cap", 0.25)
                persistence_cap = racing_weights.get("persistence_bonus_cap", 0.15)

                # Racing energy is often sustained. Bonuses are capped so smooth driving
                # improves survival without overpowering the rest of the score.
                base_score = (
                    (motion_delta * motion_weight) +
                    (audio_peak * audio_weight) +
                    (speech_energy * speech_weight) +
                    (emotion_score * 0.05) +
                    (scene_conf * 0.05)
                )
                rolling_motion = float(np.mean(motion_scores[max(0, len(motion_scores) - 11):]))
                momentum_bonus = min(rolling_motion * momentum_weight, momentum_cap)
                if motion_delta > 0.45:
                    consecutive_high_motion_frames += 1
                else:
                    consecutive_high_motion_frames = 0
                persistence_bonus = min(consecutive_high_motion_frames / 20.0, 1.0) * persistence_cap
                tags = []
                if rolling_motion > 0.5 and consecutive_high_motion_frames >= 8:
                    tags.append("sequence_candidate")
                final_score = min(base_score + momentum_bonus + persistence_bonus, 1.0)

            self._stats.record_peak(final_score)

            times.append(sec.timestamp)
            scores.append(final_score)
            features_list.append({
                "audio_peak":     audio_peak,
                "emotion_score":  emotion_score,
                "surprise_level": surprise_level,
                "motion_delta":   motion_delta,
                "lateral_flow":    lateral_flow,
                "scene_type":     scene_type,
                "speech_energy":  speech_energy,
                "keyword_weight": keyword_weight,
                "rolling_motion": rolling_motion,
                "momentum_bonus": momentum_bonus,
                "persistence_bonus": persistence_bonus,
                "consecutive_high_motion_frames": consecutive_high_motion_frames,
                "racing_score_base": base_score,
                "tags": tags,
            })

        # ── 2. Adaptive Percentile-Based Dynamic Threshold ────────────────────
        # The start_threshold from the profile is the *floor*.
        # We additionally compute a percentile-based threshold from the
        # actual stream scores and take whichever is lower — this means
        # a naturally quiet stream will still produce the best moments.
        score_array = np.array(scores)
        percentile_threshold = float(np.percentile(score_array, t["percentile_floor"]))
        effective_start = min(t["start_threshold"], percentile_threshold)
        effective_end   = min(t["end_threshold"],   effective_start * 0.75)

        logger.debug(
            f"Profile start={t['start_threshold']:.3f}, "
            f"P{t['percentile_floor']}={percentile_threshold:.3f} → "
            f"effective_start={effective_start:.3f}"
        )

        # ── 3. Hysteresis Segmentation ────────────────────────────────────────
        raw_events = self._segment_hysteresis(times, scores, effective_start, effective_end)

        # ── 4. Merge nearby events ────────────────────────────────────────────
        merged_events = self._merge_events(times, raw_events)

        # ── 5. Split multi-peak events ────────────────────────────────────────
        merged_events = self._split_multi_peak_events(scores, merged_events)

        # ── 6. Validate & Build Moments ───────────────────────────────────────
        moments = []
        for (start_idx, end_idx) in merged_events:
            self._stats.total_raw += 1
            ev_times = times[start_idx:end_idx + 1]
            if not ev_times:
                continue

            dur = ev_times[-1] - ev_times[0]
            if dur < self.MIN_EVENT_DURATION:
                self._stats.rejected_duration += 1
                continue

            max_event_duration = float(self._racing_clip_rules.get("max_duration", self.MAX_EVENT_DURATION)) if is_racing else self.MAX_EVENT_DURATION
            if dur > max_event_duration:
                split_idx = start_idx + int((max_event_duration / dur) * len(ev_times))
                end_idx   = min(split_idx, len(times) - 1)
                ev_times  = times[start_idx:end_idx + 1]

            start_time = max(0.0, ev_times[0] - self._pre_buffer)
            end_time   = ev_times[-1] + self._post_buffer

            moment = self._build_moment(
                start_idx, end_idx, start_time, end_time,
                times, scores, features_list, transcript_data,
            )
            if moment:
                moments.append(moment)
                self._stats.final_valid += 1

        # ── 7. Apply silence boost ────────────────────────────────────────────
        self._apply_silence_boost(moments, timeline)

        # ── 8. Top-N Fallback (NEVER return 0) ───────────────────────────────
        if not moments:
            print(
                f"\n  ⚠️  No moments survived quality filtering. "
                f"Activating top-{t['fallback_n']} raw segment fallback...",
                flush=True,
            )
            moments = self._top_n_fallback(
                times, scores, features_list, transcript_data,
                n=t["fallback_n"],
            )
            if moments:
                print(f"  ✅  Fallback recovered {len(moments)} moment(s).", flush=True)

        # ── 9. Print rejection diagnostics ───────────────────────────────────
        self._stats.print_summary(self._pacing_style, t)

        return sorted(moments, key=lambda m: m.final_score, reverse=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Hysteresis Segmentation
    # ─────────────────────────────────────────────────────────────────────────

    def _segment_hysteresis(self, times, scores, start_threshold, end_threshold):
        events = []
        in_event = False
        start_idx = 0

        for i, score in enumerate(scores):
            if not in_event and score >= start_threshold:
                in_event = True
                start_idx = i
            elif in_event and score < end_threshold:
                # Require sustained drop (3 consecutive frames below end_threshold)
                if i + 2 < len(scores) and scores[i+1] < end_threshold and scores[i+2] < end_threshold:
                    events.append((start_idx, i))
                    in_event = False
                elif i + 2 >= len(scores):
                    events.append((start_idx, i))
                    in_event = False

        if in_event:
            events.append((start_idx, len(scores) - 1))

        return events

    # ─────────────────────────────────────────────────────────────────────────
    # Merge Events
    # ─────────────────────────────────────────────────────────────────────────

    def _merge_events(self, times, events):
        if not events:
            return []
        merged = [events[0]]
        for curr in events[1:]:
            prev = merged[-1]
            gap_seconds = times[curr[0]] - times[prev[1]]
            if gap_seconds <= self.GAP_MERGE_THRESHOLD:
                merged[-1] = (prev[0], curr[1])
            else:
                merged.append(curr)
        return merged

    # ─────────────────────────────────────────────────────────────────────────
    # Split Multi-Peak Events
    # ─────────────────────────────────────────────────────────────────────────

    def _split_multi_peak_events(self, scores, events):
        new_events = []
        for start, end in events:
            ev_scores = scores[start:end + 1]
            if len(ev_scores) < 5:
                new_events.append((start, end))
                continue

            peaks = [
                i for i in range(1, len(ev_scores) - 1)
                if ev_scores[i] > ev_scores[i-1] and ev_scores[i] > ev_scores[i+1]
            ]

            if len(peaks) > 1:
                primary_peak = max(peaks, key=lambda p: ev_scores[p])
                primary_val  = ev_scores[primary_peak]

                split_points = []
                for p in peaks:
                    if p == primary_peak:
                        continue
                    if ev_scores[p] >= 0.8 * primary_val:
                        p1, p2 = min(p, primary_peak), max(p, primary_peak)
                        if p2 - p1 > 2:
                            valley_idx = p1 + int(np.argmin(ev_scores[p1:p2 + 1]))
                            if ev_scores[valley_idx] < 0.8 * min(ev_scores[p1], ev_scores[p2]):
                                split_points.append(start + valley_idx)

                if split_points:
                    split_points = sorted(set(split_points))
                    last_split = start
                    for sp in split_points:
                        if sp > last_split:
                            new_events.append((last_split, sp))
                        last_split = sp + 1
                    if end >= last_split:
                        new_events.append((last_split, end))
                else:
                    new_events.append((start, end))
            else:
                new_events.append((start, end))

        return new_events

    # ─────────────────────────────────────────────────────────────────────────
    # Build Moment
    # ─────────────────────────────────────────────────────────────────────────

    def _build_moment(self, start_idx, end_idx, start_time, end_time,
                      times, scores, features_list, transcript_data):
        t = self._thresholds
        ev_scores   = scores[start_idx:end_idx + 1]
        ev_features = features_list[start_idx:end_idx + 1]

        local_peak_idx  = int(np.argmax(ev_scores))
        peak_time       = times[start_idx + local_peak_idx]
        peak_score      = ev_scores[local_peak_idx]

        local_average   = float(np.mean(ev_scores))
        peak_prominence = peak_score - local_average

        pre_peak  = ev_scores[:local_peak_idx]
        post_peak = ev_scores[local_peak_idx + 1:]
        rise = (peak_score - pre_peak[0])  if len(pre_peak)  > 0 else 0.0
        drop = (peak_score - post_peak[-1]) if len(post_peak) > 0 else 0.0

        self._stats.record_peak(peak_score)
        peak_feat = ev_features[local_peak_idx]
        end_feat  = ev_features[-1]
        sustained_racing_momentum = (
            self._is_racing and
            peak_feat.get("rolling_motion", 0.0) >= 0.45 and
            peak_feat.get("consecutive_high_motion_frames", 0) >= 5
        )

        # ── Guard: Peak Prominence ────────────────────────────────────────────
        if peak_prominence < t["min_prominence"]:
            # Racing highlights are often smooth momentum arcs, not FPS-like spikes.
            # Strong sustained motion can survive low prominence in racing mode only.
            if not sustained_racing_momentum:
                self._stats.rejected_prominence += 1
                reason = f"low prominence ({peak_prominence:.3f}<{t['min_prominence']:.3f})"
                self._stats.record_reject(peak_score, reason)
                logger.debug(f"Rejected: weak prominence {peak_prominence:.3f} < {t['min_prominence']:.3f}")
                if self._is_racing:
                    self._print_racing_decision("rejected", reason, peak_time, peak_feat, peak_score, peak_prominence)
                return None

        high_frames = sum(1 for s in ev_scores if s > local_average + 0.05)
        if high_frames < 2 and peak_prominence < (t["min_prominence"] * 1.5):
            if not sustained_racing_momentum:
                self._stats.rejected_prominence += 1
                reason = f"brief spike (frames>{local_average+0.05:.3f}={high_frames})"
                self._stats.record_reject(peak_score, reason)
                logger.debug(f"Rejected: too brief (high_frames={high_frames})")
                if self._is_racing:
                    self._print_racing_decision("rejected", reason, peak_time, peak_feat, peak_score, peak_prominence)
                return None

        # ── Guard: Flat Curve ─────────────────────────────────────────────────
        if rise < 0.05 and drop < 0.05:
            if not sustained_racing_momentum:
                self._stats.rejected_flat_curve += 1
                reason = f"flat curve (rise={rise:.3f}, drop={drop:.3f})"
                self._stats.record_reject(peak_score, reason)
                logger.debug(f"Rejected: flat curve rise={rise:.3f} drop={drop:.3f}")
                if self._is_racing:
                    self._print_racing_decision("rejected", reason, peak_time, peak_feat, peak_score, peak_prominence)
                return None

        # ── Guard: Std-Dev Flatness ───────────────────────────────────────────
        std_dev = np.std(ev_scores) if len(ev_scores) > 1 else 0.0
        if std_dev < 0.04 and peak_score < (t["min_intensity"] * 1.1):
            self._stats.rejected_std_dev += 1
            self._stats.record_reject(peak_score, f"flat stddev ({std_dev:.3f})")
            logger.debug(f"Rejected: flat stddev {std_dev:.3f}, peak {peak_score:.3f}")
            return None

        surprise_score = peak_feat["motion_delta"] + peak_feat["audio_peak"] + peak_feat["surprise_level"]
        is_combat      = peak_feat["scene_type"] == "combat"
        conflict_score = (1.0 if is_combat else 0.0) + peak_feat["speech_energy"]

        post_peak_scores = ev_scores[local_peak_idx:]
        drop_slope = 0.0
        if len(post_peak_scores) > 2:
            drop_slope = post_peak_scores[0] - post_peak_scores[-1]
        audio_drop  = peak_feat["audio_peak"]  - end_feat["audio_peak"]
        motion_drop = peak_feat["motion_delta"] - end_feat["motion_delta"]
        payoff_score= peak_score + max(0.0, drop_slope) + max(0.0, audio_drop) + max(0.0, motion_drop)

        # ── Guard: Minimum Intensity ──────────────────────────────────────────
        if peak_score < t["min_intensity"]:
            self._stats.rejected_intensity += 1
            reason = f"intensity ({peak_score:.3f}<{t['min_intensity']:.3f})"
            self._stats.record_reject(peak_score, reason)
            logger.debug(f"Rejected: intensity {peak_score:.3f} < {t['min_intensity']:.3f}")
            if self._is_racing:
                self._print_racing_decision("rejected", reason, peak_time, peak_feat, peak_score, peak_prominence)
            return None

        # ── Guard: Surprise / Payoff ──────────────────────────────────────────
        if surprise_score < t["min_surprise_payoff"] and payoff_score < t["min_surprise_payoff"]:
            self._stats.rejected_surprise_payoff += 1
            reason = (
                f"weak surprise ({surprise_score:.3f}) & payoff ({payoff_score:.3f}) "
                f"< {t['min_surprise_payoff']:.3f}"
            )
            self._stats.record_reject(
                peak_score,
                reason,
            )
            logger.debug(f"Rejected: surprise={surprise_score:.3f} payoff={payoff_score:.3f}")
            if self._is_racing:
                self._print_racing_decision("rejected", reason, peak_time, peak_feat, peak_score, peak_prominence)
            return None

        # ── Event Type ────────────────────────────────────────────────────────
        if self._is_racing:
            sudden_motion_drop = motion_drop > 0.35 and peak_feat["motion_delta"] > 0.45
            impact_like_audio_motion = peak_feat["motion_delta"] > 0.65 and peak_feat["audio_peak"] > 0.50
            if sudden_motion_drop or impact_like_audio_motion:
                event_type = "CRASH"
            elif peak_feat.get("lateral_flow", 0.0) > 0.4:
                event_type = "DRIFT"
            elif peak_feat["motion_delta"] > 0.5 and peak_feat["audio_peak"] > 0.4:
                event_type = "HIGH_SPEED"
            elif peak_feat.get("rolling_motion", 0.0) > 0.5:
                event_type = "SPEED_BURST"
            else:
                event_type = "RACING_MOMENT"
        else:
            event_type = "neutral"
            if surprise_score > 1.8:
                event_type = "surprise"
            elif conflict_score > 1.2:
                event_type = "combat"
            elif peak_feat["emotion_score"] > 0.5:
                event_type = "reaction"
            elif peak_feat["scene_type"] == "travel":
                event_type = "travel"

        priority = EVENT_PRIORITIES.get(event_type, 1)
        if self._is_racing:
            self._print_racing_decision(
                "survived", "passed racing thresholds", peak_time,
                peak_feat, peak_score, peak_prominence, event_type
            )

        # ── Confidence ────────────────────────────────────────────────────────
        variance = float(np.var(ev_scores))
        signal_consistency = max(0.0, 1.0 - (variance * 5))
        active_modalities = sum(
            1 for feat, threshold in [
                ("audio_peak", 0.35), ("motion_delta", 0.35),
                ("emotion_score", 0.35), ("speech_energy", 0.35),
            ]
            if peak_feat[feat] > threshold
        )
        modality_agreement = active_modalities / 4.0
        event_confidence = min(
            1.0,
            (signal_consistency * 0.3) +
            (modality_agreement * 0.4) +
            (min(peak_prominence * 2, 1.0) * 0.3),
        )

        # ── Adaptive Payoff Extension ─────────────────────────────────────────
        from payoff_detector import evaluate_resolution, is_sentence_incomplete
        
        MAX_PAYOFF_EXTENSION = 4.0
        original_end_time = end_time
        max_end_time = end_time + MAX_PAYOFF_EXTENSION
        
        payoff_detected = False
        resolution_score = 0.0
        ending_reason = "fixed_buffer"
        
        current_end_time = original_end_time
        step_size = 1.0
        
        while current_end_time <= max_end_time:
            # Safely get index for the current timeline second
            current_idx = min(len(times) - 1, int(current_end_time))
            if current_idx < 0: current_idx = 0
            
            current_features = features_list[current_idx]
            current_audio = current_features.get("audio_peak", 0.0)
            current_motion = current_features.get("motion_delta", 0.0)
            
            transcript_so_far = ""
            if transcript_data:
                from arc_detector import _collect_transcript_in_range
                transcript_so_far = _collect_transcript_in_range(transcript_data, start_time, current_end_time)
                
            payoff_detected, res_score, reason = evaluate_resolution(
                peak_features=peak_feat,
                current_end_features=current_features,
                post_peak_audio=current_audio,
                post_peak_motion=current_motion,
                transcript=transcript_so_far
            )
            
            if payoff_detected:
                resolution_score = res_score
                ending_reason = reason
                break
                
            current_end_time += step_size
            
        final_end_time = min(current_end_time, max_end_time)
        extension_used = final_end_time - original_end_time
        
        transcript = ""
        if transcript_data:
            from arc_detector import _collect_transcript_in_range
            transcript = _collect_transcript_in_range(transcript_data, start_time, final_end_time)
            
        transcript_sentence_incomplete = is_sentence_incomplete(transcript)

        return EventMoment(
            event_type=event_type,
            start=start_time,
            end=final_end_time,
            peak_time=peak_time,
            duration=final_end_time - start_time,
            final_score=peak_score,
            surprise_score=surprise_score,
            conflict_score=conflict_score,
            payoff_score=payoff_score,
            priority=priority,
            scene_type=peak_feat["scene_type"],
            features=peak_feat,
            transcript=transcript,
            event_confidence=event_confidence,
            pre_context_buffer=self._pre_buffer,
            post_payoff_buffer=self._post_buffer,
            peak_prominence=peak_prominence,
            resolution_score=resolution_score,
            payoff_detected=payoff_detected,
            ending_reason=ending_reason,
            ending_extension_used=extension_used,
            transcript_sentence_incomplete=transcript_sentence_incomplete
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Top-N Fallback (ensures system NEVER returns 0 clips)
    # ─────────────────────────────────────────────────────────────────────────

    def _top_n_fallback(self, times, scores, features_list, transcript_data, n=2):
        """
        Select the top-N highest scoring windows as emergency clip candidates.
        These bypass quality guards entirely — they are the best of what exists.
        A minimum separation of 60s is enforced to avoid clustering.
        """
        MIN_SEPARATION = 60.0
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        selected = []
        used_times = []

        for idx, score in indexed:
            if len(selected) >= n:
                break
            t_sec = times[idx]
            if any(abs(t_sec - used) < MIN_SEPARATION for used in used_times):
                continue

            feat = features_list[idx]
            surprise_score = feat["motion_delta"] + feat["audio_peak"] + feat["surprise_level"]
            conflict_score = feat["speech_energy"]
            payoff_score   = score

            start_time = max(0.0, t_sec - self._pre_buffer - 3.0)
            end_time   = t_sec + self._post_buffer + 3.0

            transcript = ""
            if transcript_data:
                from arc_detector import _collect_transcript_in_range
                transcript = _collect_transcript_in_range(transcript_data, start_time, end_time)

            fallback_event_type = "neutral"
            if self._is_racing:
                if feat.get("lateral_flow", 0.0) > 0.4:
                    fallback_event_type = "DRIFT"
                elif feat["motion_delta"] > 0.5 and feat["audio_peak"] > 0.4:
                    fallback_event_type = "HIGH_SPEED"
                elif feat.get("rolling_motion", 0.0) > 0.5:
                    fallback_event_type = "SPEED_BURST"
                else:
                    fallback_event_type = "RACING_MOMENT"

            moment = EventMoment(
                event_type=fallback_event_type,
                start=start_time,
                end=end_time,
                peak_time=t_sec,
                duration=end_time - start_time,
                final_score=score,
                surprise_score=surprise_score,
                conflict_score=conflict_score,
                payoff_score=payoff_score,
                priority=EVENT_PRIORITIES.get(fallback_event_type, 1),
                scene_type=feat.get("scene_type", "neutral"),
                features=feat,
                transcript=transcript,
                event_confidence=0.3,  # Low confidence — fallback only
                pre_context_buffer=self._pre_buffer,
                post_payoff_buffer=self._post_buffer,
                peak_prominence=0.0,   # unknown in fallback path
                resolution_score=0.0,
                payoff_detected=False,
                ending_reason="fallback",
                ending_extension_used=0.0,
                transcript_sentence_incomplete=False
            )
            selected.append(moment)
            used_times.append(t_sec)
            print(
                f"    [Fallback] Clip at t={t_sec:.1f}s  score={score:.4f}  "
                f"(audio={feat['audio_peak']:.3f}, motion={feat['motion_delta']:.3f}, "
                f"speech={feat['speech_energy']:.3f})",
                flush=True,
            )

        return selected

    # ─────────────────────────────────────────────────────────────────────────
    # Silence Boost
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_silence_boost(self, moments, timeline):
        for m in moments:
            post_peak = [
                s for s in timeline
                if m.peak_time < s.timestamp <= m.peak_time + 3.0
            ]
            if post_peak:
                avg_audio = np.mean([
                    s.metadata.get("audio_features", {}).get("audio_peak_norm", 0.0)
                    for s in post_peak
                ])
                if avg_audio < 0.2:
                    m.payoff_score *= 1.25

    def get_stats(self) -> RejectionStats:
        return self._stats
