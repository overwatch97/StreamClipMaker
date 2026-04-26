import json
import os
from typing import List, Dict, Optional

def find_nearest_word_boundary(timestamp: float, transcript_data: List[Dict], search_window=4.0) -> float:
    """
    Finds the nearest semantic boundary (punctuation or large pause) within a search window 
    to avoid cutting mid-sentence. Falls back to nearest word gap if no obvious pause is found.
    """
    all_words = []
    for segment in transcript_data:
        all_words.extend(segment.get("words", []))
    
    if not all_words:
        return timestamp
        
    # Find words near the timestamp
    near_words = [w for w in all_words if abs(w["start"] - timestamp) < search_window]
    if not near_words:
        return timestamp
        
    near_words.sort(key=lambda x: x["start"])
    
    best_boundary = timestamp
    min_dist = float('inf')
    
    # Priority 1: Punctuation (end of sentence)
    # Priority 2: Natural pause (gap > 0.4s)
    # Priority 3: Any gap
    for i in range(len(near_words) - 1):
        w1 = near_words[i]
        w2 = near_words[i+1]
        
        gap_start = w1["end"]
        gap_end = w2["start"]
        gap_center = (gap_start + gap_end) / 2.0
        gap_duration = gap_end - gap_start
        
        dist = abs(gap_center - timestamp)
        
        # Check if w1 ends with punctuation
        text = w1.get("text", w1.get("word", "")).strip()
        has_punctuation = text.endswith(('.', '!', '?', ','))
        
        # Apply weighting to distance so we heavily prefer punctuation/pauses
        effective_dist = dist
        if has_punctuation:
            effective_dist *= 0.1  # Huge preference for sentence end
        elif gap_duration > 0.4:
            effective_dist *= 0.3  # Strong preference for natural pause
            
        if effective_dist < min_dist:
            min_dist = effective_dist
            best_boundary = gap_center
            
    return best_boundary

def apply_hook_tease(arc: Dict, tease_duration=3.0) -> Dict:
    """
    Implements 'Hook Placement' by anchoring back to the true multimodal peak_time.
    """
    # Use the peak_time passed from the moment_expander!
    peak_time = arc.get("peak_time", arc["start"] + 2.0)
    
    # Position the teaser to start slightly before the peak payload hits
    arc["hook_time"] = max(arc["start"], peak_time - 1.0)
    
    return arc

def refine_clips_for_social(arcs: List[Dict], transcript_data: List[Dict]) -> List[Dict]:
    """
    Polishes the narrative arcs into final clipping instructions without destroying context bounds.
    """
    refined = []
    for arc in arcs:
        # 1. Natural timing adjustment (snap to sentence boundaries / pauses)
        arc["start"] = find_nearest_word_boundary(arc["start"], transcript_data)
        arc["end"] = find_nearest_word_boundary(arc["end"], transcript_data)
        
        # (Removed: hardcoded 15-second expansion bug. The logic in moment_expander.py already 
        # respects game-specific profile durations dynamically, we don't want to break it here).
            
        # 2. Add 'Hook' context anchored to the peak event
        arc = apply_hook_tease(arc)
        
        refined.append(arc)
    return refined
