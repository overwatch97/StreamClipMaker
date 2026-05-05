import logging
import numpy as np
from typing import List

from phase3_types import TimelineSecond
from event_fusion_engine import EventFusionEngine

logging.basicConfig(level=logging.DEBUG)

def generate_synthetic_timeline() -> List[TimelineSecond]:
    """Generates a noisy, multi-modal signal testing edge cases."""
    timeline = []
    
    for t in range(70):
        # Base noise
        audio_peak = np.random.uniform(0.0, 0.1)
        emotion_score = np.random.uniform(0.0, 0.1)
        surprise_level = 0.0
        motion_delta = np.random.uniform(0.0, 0.1)
        scene_conf = 0.5
        scene_type = "neutral"
        speech_energy = np.random.uniform(0.0, 0.1)
        keyword_weight = 0.2

        # --- Test Case 1: Prominence Rejection (t=5 to 10) ---
        if 5 <= t <= 10:
            audio_peak = 0.8
            motion_delta = 0.8
            emotion_score = 0.8
            speech_energy = 0.8
            
        # --- Test Case 2: Multi-Peak Split (t=15 to 30) ---
        elif 15 <= t <= 30:
            scene_type = "combat"
            if t == 18: # Primary peak
                audio_peak = 0.95
                motion_delta = 0.95
                emotion_score = 0.9
                speech_energy = 0.9
            elif t in [17, 19]: 
                audio_peak = 0.85
                motion_delta = 0.85
                emotion_score = 0.85
                speech_energy = 0.85
            elif 21 <= t <= 24: # Valley
                audio_peak = 0.3
                motion_delta = 0.3
                emotion_score = 0.3
                speech_energy = 0.3
            elif t == 27: # Secondary peak
                audio_peak = 0.85
                motion_delta = 0.85
                emotion_score = 0.8
                speech_energy = 0.8
            elif t in [26, 28]:
                audio_peak = 0.75
                motion_delta = 0.75
                emotion_score = 0.75
                speech_energy = 0.75
            else:
                audio_peak = 0.5
                motion_delta = 0.5
                emotion_score = 0.4

        # --- Test Case 3: Good Reaction (t=40 to 48) ---
        elif 40 <= t <= 48:
            scene_type = "neutral"
            if t == 44: # Peak
                audio_peak = 0.95
                emotion_score = 0.95
                surprise_level = 0.95
                speech_energy = 0.95
                motion_delta = 0.8
            elif t in [43, 45]:
                audio_peak = 0.85
                emotion_score = 0.85
                speech_energy = 0.85
                motion_delta = 0.7
            elif t in [42, 43, 46, 47]: # Should cross 0.7 with boost or just stay above 0.5
                audio_peak = 0.8
                emotion_score = 0.8
                speech_energy = 0.8
                motion_delta = 0.6
            elif t > 47: # Silence Payoff
                audio_peak = 0.05
            else:
                audio_peak = 0.4
                emotion_score = 0.4

        # --- Test Case 4: Flat Curve Rejection (t=55 to 65) ---
        elif 55 <= t <= 65:
            audio_peak = 0.85
            motion_delta = 0.85
            emotion_score = 0.85
            speech_energy = 0.85
            
        metadata = {
            "audio_features": {"audio_peak_norm": audio_peak},
            "emotion_features": {"emotion_score_norm": emotion_score, "surprise_level": surprise_level, "engagement_level": 0.5},
            "visual_features": {"motion_delta_norm": motion_delta, "scene_confidence": scene_conf, "scene_type": scene_type},
            "speech_features": {"speech_energy_norm": speech_energy, "keyword_weight": keyword_weight},
        }
        
        timeline.append(TimelineSecond(timestamp=float(t), metadata=metadata))
        
    return timeline

def test_fusion_engine():
    print("Generating synthetic timeline...")
    timeline = generate_synthetic_timeline()
    
    print("Running Event Fusion Engine...")
    engine = EventFusionEngine()
    
    # We can also attach a logging handler to see the logs in stdout
    logging.getLogger("event_fusion_engine").setLevel(logging.DEBUG)
    
    events = engine.detect(timeline)
    
    print(f"\nDetected {len(events)} valid events:")
    for i, e in enumerate(events):
        print(f"\nEvent {i+1}:")
        print(f"  Type:       {e.event_type.upper()}")
        print(f"  Time:       {e.start:.1f}s - {e.end:.1f}s (Peak: {e.peak_time:.1f}s)")
        print(f"  Duration:   {e.duration:.1f}s")
        print(f"  Confidence: {e.event_confidence:.3f}")
        print(f"  Scores:     Final={e.final_score:.3f}, Surprise={e.surprise_score:.3f}, Payoff={e.payoff_score:.3f}")

if __name__ == "__main__":
    test_fusion_engine()
