import numpy as np
from typing import List, Dict, Optional, Tuple
from phase3_types import GameProfile, TimelineSecond, DetectedEvent, HighlightCandidate
import editing_brain

def expand(event: DetectedEvent, timeline: List[TimelineSecond], profile: GameProfile, 
           transcript_data: List[Dict]) -> HighlightCandidate:
    """
    Expands a clip boundary from a detected event peak.
    Uses context_rules for margins, stops on ignore_states, and snaps to transcript boundaries.
    """
    rules = profile.context_rules
    buffer_start = rules.get("expansion_buffer_start", 8.0)
    buffer_end = rules.get("expansion_buffer_end", 8.0)
    min_dur = rules.get("min_clip_duration", 15.0)
    max_dur = rules.get("max_clip_duration", 60.0)

    # Find the index in the timeline
    timestamps = [s.timestamp for s in timeline]
    try:
        peak_idx = timestamps.index(event.timestamp)
    except ValueError:
        # Fallback to closest
        peak_idx = int(np.argmin([abs(t - event.timestamp) for t in timestamps]))

    # Grow backward for setup
    start_idx = peak_idx
    while start_idx > 0:
        current = timeline[start_idx - 1]
        if current.is_ignore_state:
            break
        
        elapsed = event.timestamp - current.timestamp
        if elapsed > buffer_start + 20.0: # Hard cap on setup
            break
            
        # Keep growing if score is high (action happens before the peak)
        # OR within a tiny wind-up window (max 1.5s) to ensure the hook starts instantly.
        # We ignore the large 'buffer_start' for blind backward expansion to prevent 0.15s swipe-away rate.
        windup_window = min(1.5, buffer_start)
        if current.fused_score > 0.3 or elapsed <= windup_window:
            start_idx -= 1
        else:
            break

    # Grow forward for resolution
    end_idx = peak_idx
    while end_idx < len(timeline) - 1:
        current = timeline[end_idx + 1]
        if current.is_ignore_state:
            break
            
        elapsed = current.timestamp - event.timestamp
        if elapsed > buffer_end + 30.0: # Hard cap on resolution
            break
            
        if current.fused_score > 0.3 or elapsed <= buffer_end:
            end_idx += 1
        else:
            break

    # Snap to transcript boundaries using semantic sentence boundaries
    start_time = timeline[start_idx].timestamp
    end_time = timeline[end_idx].timestamp
    
    start_time = editing_brain.find_nearest_word_boundary(start_time, transcript_data, search_window=2.0)
    end_time = editing_brain.find_nearest_word_boundary(end_time, transcript_data, search_window=3.0)

    # Final Clamp
    duration = end_time - start_time
    if duration < min_dur:
        # Expand FORWARD to reach min_dur to preserve the immediate Hook-First start.
        # Only expand backward slightly (max 0.5s) to soften the cut cut if possible.
        diff = min_dur - duration
        start_padding = min(0.5, start_time)
        start_time = start_time - start_padding
        end_time = end_time + (diff + start_padding)
    elif duration > max_dur:
        # Truncate to max_dur around peak
        start_time = max(start_time, event.timestamp - (max_dur * 0.4))
        end_time = start_time + max_dur

    return HighlightCandidate(
        start=round(float(start_time), 3),
        end=round(float(end_time), 3),
        anchor_event=event,
        score=event.score,
        category="game_highlight",
        reason=f"Detected {event.event_type} event",
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
    
    # Sort by start time chronological order
    sorted_cands = sorted(candidates, key=lambda c: c.start)
    merged = [sorted_cands[0]]
    merge_threshold = rules.get("merge_threshold", 8.0)
    
    for current in sorted_cands[1:]:
        last = merged[-1]
        
        # If the gap between the end of the last clip and the start of this one is less than the threshold
        # (or they overlap), merge them.
        gap = current.start - last.end
        if gap <= merge_threshold:
            # Combine them
            new_end = max(last.end, current.end)
            new_score = max(last.score, current.score)
            
            # Create a combined event list safely
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
