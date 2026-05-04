import logging
import statistics
from typing import List, Dict, Optional, Tuple
import numpy as np

from phase3_types import EventMoment, TimelineSecond, EVENT_PRIORITIES

logger = logging.getLogger(__name__)

class EventFusionEngine:
    START_THRESHOLD = 0.7
    END_THRESHOLD = 0.5
    MIN_EVENT_DURATION = 3.0
    MAX_EVENT_DURATION = 15.0
    GAP_MERGE_THRESHOLD = 1.5
    PRE_BUFFER = 0.5
    POST_BUFFER = 1.0

    def detect(self, timeline: List[TimelineSecond], transcript_data: List[Dict] = None) -> List[EventMoment]:
        if not timeline:
            return []

        times = []
        scores = []
        features_list = []

        # 1. Calculate fusion scores
        for sec in timeline:
            meta = sec.metadata
            a_feat = meta.get("audio_features", {})
            e_feat = meta.get("emotion_features", {})
            v_feat = meta.get("visual_features", {})
            s_feat = meta.get("speech_features", {})

            audio_peak = a_feat.get("audio_peak_norm", 0.0)
            emotion_score = e_feat.get("emotion_score_norm", 0.0)
            surprise_level = e_feat.get("surprise_level", 0.0)
            motion_delta = v_feat.get("motion_delta_norm", 0.0)
            scene_conf = v_feat.get("scene_confidence", 0.0)
            scene_type = v_feat.get("scene_type", "neutral")
            speech_energy = s_feat.get("speech_energy_norm", 0.0)
            keyword_weight = s_feat.get("keyword_weight", 0.2)
            
            # Face Presence Boost
            engagement_level = e_feat.get("engagement_level", 0.0)
            face_visible = engagement_level > 0.01 or emotion_score > 0.05
            boost = 1.15 if (face_visible and emotion_score > 0.4) else 1.0

            # Base formula
            base_score = (
                (audio_peak * 0.30) +
                (emotion_score * 0.25) +
                (motion_delta * 0.15) +
                (scene_conf * 0.10) +
                (speech_energy * 0.10) +
                (keyword_weight * 0.10)
            )
            
            final_score = min(base_score * boost, 1.0)
            
            times.append(sec.timestamp)
            scores.append(final_score)
            features_list.append({
                "audio_peak": audio_peak,
                "emotion_score": emotion_score,
                "surprise_level": surprise_level,
                "motion_delta": motion_delta,
                "scene_type": scene_type,
                "speech_energy": speech_energy,
                "keyword_weight": keyword_weight,
            })

        # 2. Hysteresis Segmentation
        raw_events = self._segment_hysteresis(times, scores)

        # 3. Merge Events
        merged_events = self._merge_events(times, raw_events)

        # 4. Enforce Duration Constraints & Build Moments
        moments = []
        for (start_idx, end_idx) in merged_events:
            ev_times = times[start_idx:end_idx+1]
            if not ev_times: continue
            
            dur = ev_times[-1] - ev_times[0]
            if dur < self.MIN_EVENT_DURATION:
                continue
                
            if dur > self.MAX_EVENT_DURATION:
                # Split event roughly to max duration
                split_idx = start_idx + int((self.MAX_EVENT_DURATION / dur) * len(ev_times))
                end_idx = min(split_idx, len(times) - 1)
                ev_times = times[start_idx:end_idx+1]

            start_time = max(0.0, ev_times[0] - self.PRE_BUFFER)
            end_time = ev_times[-1] + self.POST_BUFFER
            
            moment = self._build_moment(
                start_idx, end_idx, start_time, end_time, 
                times, scores, features_list, transcript_data
            )
            
            if moment:
                moments.append(moment)

        # Apply Silence-Aware Scoring Boost
        self._apply_silence_boost(moments, timeline)

        return sorted(moments, key=lambda m: m.final_score, reverse=True)

    def _segment_hysteresis(self, times, scores):
        events = []
        in_event = False
        start_idx = 0
        
        for i, score in enumerate(scores):
            if not in_event and score >= self.START_THRESHOLD:
                in_event = True
                start_idx = i
            elif in_event and score < self.END_THRESHOLD:
                # Sustained drop requirement (check next few frames if possible)
                if i + 2 < len(scores) and scores[i+1] < self.END_THRESHOLD and scores[i+2] < self.END_THRESHOLD:
                    events.append((start_idx, i))
                    in_event = False
                elif i + 2 >= len(scores):
                    events.append((start_idx, i))
                    in_event = False
                    
        if in_event:
            events.append((start_idx, len(scores)-1))
            
        return events

    def _merge_events(self, times, events):
        if not events: return []
        merged = [events[0]]
        for curr in events[1:]:
            prev = merged[-1]
            gap_seconds = times[curr[0]] - times[prev[1]]
            if gap_seconds <= self.GAP_MERGE_THRESHOLD:
                merged[-1] = (prev[0], curr[1])
            else:
                merged.append(curr)
        return merged

    def _build_moment(self, start_idx, end_idx, start_time, end_time, times, scores, features_list, transcript_data):
        ev_scores = scores[start_idx:end_idx+1]
        ev_features = features_list[start_idx:end_idx+1]
        
        # Peak Detection Inside Event
        local_peak_idx = int(np.argmax(ev_scores))
        peak_time = times[start_idx + local_peak_idx]
        peak_score = ev_scores[local_peak_idx]
        
        peak_feat = ev_features[local_peak_idx]
        end_feat = ev_features[-1]
        
        # Sub-scores
        surprise_score = peak_feat["motion_delta"] + peak_feat["audio_peak"] + peak_feat["surprise_level"]
        
        is_combat = peak_feat["scene_type"] == "combat"
        conflict_score = (1.0 if is_combat else 0.0) + peak_feat["speech_energy"]
        
        # Payoff Score
        post_peak_scores = ev_scores[local_peak_idx:]
        drop_slope = 0.0
        if len(post_peak_scores) > 2:
            drop_slope = post_peak_scores[0] - post_peak_scores[-1]
            
        audio_drop = peak_feat["audio_peak"] - end_feat["audio_peak"]
        motion_drop = peak_feat["motion_delta"] - end_feat["motion_delta"]
        payoff_score = peak_score + max(0.0, drop_slope) + max(0.0, audio_drop) + max(0.0, motion_drop)
        
        # Event Type Resolution
        event_type = "neutral"
        if surprise_score > 1.8:
            event_type = "surprise"
        elif conflict_score > 1.2:
            event_type = "combat"
        elif peak_feat["emotion_score"] > 0.6:
            event_type = "reaction"
        elif peak_feat["scene_type"] == "travel":
            event_type = "travel"
            
        priority = EVENT_PRIORITIES.get(event_type, 1)
        
        # Clip Quality Guard
        std_dev = np.std(ev_scores) if len(ev_scores) > 1 else 0.0
        if std_dev < 0.05 and peak_score < 0.8:
            logger.debug(f"Event rejected: Flat intensity curve (std={std_dev:.3f}, peak={peak_score:.3f})")
            return None 
            
        # Strict Filtering Rules (Intensity > 0.7 AND (Surprise > 0.6 OR Payoff > 0.6))
        if peak_score < 0.7:
            logger.debug(f"Event rejected: Intensity too low ({peak_score:.3f})")
            return None
        if surprise_score < 0.6 and payoff_score < 0.6:
            logger.debug(f"Event rejected: Weak surprise ({surprise_score:.3f}) and weak payoff ({payoff_score:.3f})")
            return None

        transcript = ""
        if transcript_data:
            from arc_detector import _collect_transcript_in_range
            transcript = _collect_transcript_in_range(transcript_data, start_time, end_time)

        return EventMoment(
            event_type=event_type,
            start=start_time,
            end=end_time,
            peak_time=peak_time,
            duration=end_time - start_time,
            final_score=peak_score,
            surprise_score=surprise_score,
            conflict_score=conflict_score,
            payoff_score=payoff_score,
            priority=priority,
            scene_type=peak_feat["scene_type"],
            features=peak_feat,
            transcript=transcript
        )
        
    def _apply_silence_boost(self, moments, timeline):
        for m in moments:
            post_peak_secs = [s for s in timeline if m.peak_time < s.timestamp <= m.peak_time + 3.0]
            if post_peak_secs:
                avg_audio = np.mean([s.metadata.get("audio_features", {}).get("audio_peak_norm", 0.0) for s in post_peak_secs])
                if avg_audio < 0.2: 
                    m.payoff_score *= 1.25 # Boost retention/payoff
