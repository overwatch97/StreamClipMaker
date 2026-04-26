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

def validate_clip_logic(arc: Dict, transcript_data: List[Dict]) -> bool:
    """
    Validates if a clip passes the Human-Editor criteria.
    Criteria:
    1. Hook length: Is the setup too long before the action?
    2. Understandable without sound: Does the peak have high visual motion/action if speech is low?
    3. Has Payoff: Does it naturally resolve? (We check duration after peak).
    """
    start_time = arc["start"]
    peak_time = arc.get("peak_time", start_time + 2.0)
    end_time = arc["end"]
    
    # Rule 1: Hook must appear quickly. Setup should not exceed 4 seconds.
    # Note: moment_expander already tries to keep it short, but this is the final gate.
    if (peak_time - start_time) > 4.5:
        # We will attempt to trim it in the calling function, but for now it's invalid
        return False
        
    # Rule 2: Understandable without sound. 
    # If it's just someone talking quietly with no visual action, it's a bad gaming short.
    # We check the arc's metadata or score. Since we only have the dict here, 
    # we assume clips with score < 60 that don't have a strong visual component are risky.
    score = arc.get("score", 0)
    reason = arc.get("reason", "").lower()
    
    is_visual_event = any(k in reason for k in ["kill", "visual", "motion", "action", "combat"])
    
    if score < 65 and not is_visual_event:
        # If it's a low score and not inherently visual, it might be boring without sound.
        return False
        
    # Rule 3: Payoff length. Must have at least 1-2 seconds after the action peak.
    if (end_time - peak_time) < 1.0:
        return False
        
    return True

def fix_clip_hook(arc: Dict, transcript_data: List[Dict]) -> Dict:
    """
    Aggressively trims the start of the clip so the hook appears within 2 seconds of the video starting.
    """
    start_time = arc["start"]
    peak_time = arc.get("peak_time", start_time + 2.0)
    
    if (peak_time - start_time) > 2.0:
        # Trim it down to a micro-setup (1.5s before peak)
        new_start = peak_time - 1.5
        arc["start"] = find_nearest_word_boundary(new_start, transcript_data, search_window=1.0)
        
    return arc

def refine_clips_for_social(arcs: List[Dict], transcript_data: List[Dict]) -> List[Dict]:
    """
    Polishes the narrative arcs into final clipping instructions using Human Editor logic.
    Rejects or fixes clips that don't pass the narrative rules.
    """
    refined = []
    for arc in arcs:
        # 1. Natural timing adjustment (snap to sentence boundaries / pauses)
        arc["start"] = find_nearest_word_boundary(arc["start"], transcript_data, search_window=1.5)
        arc["end"] = find_nearest_word_boundary(arc["end"], transcript_data, search_window=2.0)
        
        # 2. Fix long hooks (trim fat before the action)
        arc = fix_clip_hook(arc, transcript_data)
        
        # 3. Validate
        if validate_clip_logic(arc, transcript_data):
            refined.append(arc)
        else:
            # Clip was rejected by Human Editor logic (e.g. boring, too long setup, no payoff)
            continue
            
    return refined
