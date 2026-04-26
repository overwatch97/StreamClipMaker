import numpy as np
from typing import List, Dict, Optional, Tuple
from phase3_types import GameProfile, TimelineSecond, DetectedEvent, HighlightCandidate
import editing_brain

def expand(event: DetectedEvent, timeline: List[TimelineSecond], profile: GameProfile, 
           transcript_data: List[Dict]) -> HighlightCandidate:
    """
    Expands a clip boundary from a detected event peak using human-editor logic:
    CAUSE (Hook) -> ACTION (Main Event) -> RESULT (Payoff/Reaction).
    """
    rules = profile.context_rules
    max_cause_lookback = 5.0 # Max seconds to look back for a hook
    max_result_lookahead = 10.0 # Max seconds to look forward for a payoff
    min_dur = rules.get("min_clip_duration", 5.0) # Social shorts can be very short
    max_dur = rules.get("max_clip_duration", 60.0)

    timestamps = [s.timestamp for s in timeline]
    try:
        peak_idx = timestamps.index(event.timestamp)
    except ValueError:
        peak_idx = int(np.argmin([abs(t - event.timestamp) for t in timestamps]))

    # --- 1. Find CAUSE (Hook / Setup) ---
    # Look backwards for a spike in speech (someone starting to talk loudly) or visual motion.
    start_idx = peak_idx
    cause_found = False
    while start_idx > 0:
        current = timeline[start_idx - 1]
        if current.is_ignore_state:
            break
        
        elapsed = event.timestamp - current.timestamp
        if elapsed > max_cause_lookback:
            break
            
        # Hook criteria: A sudden speech spike or high overall score *before* the action hits.
        if current.speech_score > 0.4 or current.fused_score > 0.5:
            start_idx -= 1
            cause_found = True
        elif not cause_found and elapsed <= 1.0:
            # Micro-setup: If we haven't found a strong cause, at least step back a tiny bit
            start_idx -= 1
        else:
            break

    # --- 2. Find RESULT (Payoff / Reaction) ---
    # Look forwards for emotion spikes (laugh/scream) or speech resolving.
    end_idx = peak_idx
    result_found = False
    while end_idx < len(timeline) - 1:
        current = timeline[end_idx + 1]
        if current.is_ignore_state:
            break
            
        elapsed = current.timestamp - event.timestamp
        if elapsed > max_result_lookahead:
            break
            
        # Reaction criteria: High emotion score (facecam) or prolonged speech.
        if current.emotion_score > 0.4 or current.speech_score > 0.3:
            end_idx += 1
            result_found = True
        elif not result_found and elapsed <= 2.0:
            # Let the action breathe a little before cutting if no immediate reaction
            end_idx += 1
        else:
            break

    # Snap to transcript boundaries using semantic sentence boundaries
    start_time = timeline[start_idx].timestamp
    end_time = timeline[end_idx].timestamp
    
    # Precise snapping: Hook must snap cleanly to start of sentence. 
    # Result must snap cleanly to end of sentence.
    start_time = editing_brain.find_nearest_word_boundary(start_time, transcript_data, search_window=2.0)
    end_time = editing_brain.find_nearest_word_boundary(end_time, transcript_data, search_window=3.0)

    # Force minimum duration if it's too short (prefer expanding forward)
    duration = end_time - start_time
    if duration < min_dur:
        end_time = start_time + min_dur

    if duration > max_dur:
        # Emergency truncation
        end_time = start_time + max_dur

    return HighlightCandidate(
        start=round(float(start_time), 3),
        end=round(float(end_time), 3),
        anchor_event=event,
        score=event.score,
        category="game_highlight",
        reason=f"Action at {event.timestamp}s with Cause/Result arc.",
        game_id=profile.game_id,
        events=[event],
        profile_id=profile.game_id,
        profile_version=profile.version,
        evidence=event.evidence
    )

def merge_overlapping(candidates: List[HighlightCandidate], rules: Dict) -> List[HighlightCandidate]:
    """
    Merges candidates that overlap or are extremely close to each other,
    creating continuous story arcs for multi-kills or prolonged events.
    """
    if not candidates:
        return []
    
    sorted_cands = sorted(candidates, key=lambda c: c.start)
    merged = [sorted_cands[0]]
    merge_threshold = rules.get("merge_threshold", 5.0) # Tighter merge threshold for human-editing
    
    for current in sorted_cands[1:]:
        last = merged[-1]
        
        gap = current.start - last.end
        if gap <= merge_threshold:
            new_end = max(last.end, current.end)
            new_score = max(last.score, current.score)
            
            combined_events = []
            if hasattr(last, "events") and last.events:
                combined_events.extend(last.events)
            if hasattr(current, "events") and current.events:
                combined_events.extend(current.events)
            
            last.end = new_end
            last.score = new_score
            last.events = combined_events
            last.reason = f"{last.reason} + merged subsequent event"
        else:
            merged.append(current)
            
    return merged

