"""
payoff_detector.py — Continuity & Emotional Payoff Completion System
====================================================================
Analyzes post-peak timeline and transcript data to determine if an
event has emotionally resolved (e.g., laughter, silence, combat ended)
so clips do not end abruptly.
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def is_sentence_incomplete(transcript: str) -> bool:
    """Returns True if the transcript ends mid-sentence."""
    if not transcript:
        return False
        
    text = transcript.strip()
    if not text:
        return False
        
    # Check for terminal punctuation
    terminal_punctuation = ('.', '!', '?', '"', '”', ']', ')')
    
    if not text.endswith(terminal_punctuation):
        return True
        
    return False

def evaluate_resolution(
    peak_features: Dict[str, Any], 
    current_end_features: Dict[str, Any], 
    post_peak_audio: float, 
    post_peak_motion: float,
    transcript: str = ""
) -> Tuple[bool, float, str]:
    """
    Evaluates if the emotional moment has concluded at the current end time.
    
    Returns:
        payoff_detected (bool): Whether the moment has concluded.
        resolution_score (float): 0.0-1.0 score of how finished it feels.
        ending_reason (str): The primary reason the ending was chosen.
    """
    resolution_score = 0.0
    ending_reason = "fixed_buffer"
    
    peak_audio = float(peak_features.get("audio_peak", 0.0))
    peak_motion = float(peak_features.get("motion_delta", 0.0))
    peak_emotion = float(peak_features.get("emotion_score", 0.0))
    
    current_audio = float(current_end_features.get("audio_peak", 0.0))
    current_motion = float(current_end_features.get("motion_delta", 0.0))
    current_emotion = float(current_end_features.get("emotion_score", 0.0))
    
    # 1. Silence After Peak (Weight: 0.25)
    # We compare the current audio to the peak audio. If it's significantly lower, it's silent.
    audio_drop = peak_audio - current_audio
    if audio_drop > 0.25 or current_audio < 0.20:
        resolution_score += 0.25
        ending_reason = "silence_after_peak"
        
    # 2. Motion Cooldown (Weight: 0.20)
    motion_drop = peak_motion - current_motion
    if motion_drop > 0.25:
        resolution_score += 0.20
        if ending_reason == "fixed_buffer":
            ending_reason = "motion_cooldown"
            
    # 3. Combat End (Weight: 0.10)
    if peak_features.get("scene_type") == "combat" and current_end_features.get("scene_type") != "combat":
        resolution_score += 0.10
        if resolution_score > 0.2:
            ending_reason = "combat_end"
            
    # 4. Emotion Release (Weight: 0.15)
    if peak_emotion > 0.5 and current_emotion < 0.3:
        resolution_score += 0.15
        if resolution_score > 0.2:
            ending_reason = "emotion_release"
            
    # 5. Reaction Dialogue (Weight: 0.30)
    sentence_incomplete = False
    if transcript:
        sentence_incomplete = is_sentence_incomplete(transcript)
        if sentence_incomplete:
            # Strong penalty if they are still mid-sentence
            resolution_score -= 0.50
            ending_reason = "sentence_incomplete"
        else:
            # Sentence completes
            resolution_score += 0.30
            if resolution_score > 0.4:
                ending_reason = "reaction_dialogue"
                
    payoff_detected = resolution_score >= 0.35 and not sentence_incomplete
    
    if not payoff_detected and ending_reason != "sentence_incomplete":
        ending_reason = "intensity_drop" if resolution_score > 0.1 else "fixed_buffer"
        
    return payoff_detected, max(0.0, resolution_score), ending_reason
