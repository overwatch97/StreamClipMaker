import logging
import numpy as np
from typing import List

from phase3_types import TimelineSecond
from event_fusion_engine import EventFusionEngine

logging.basicConfig(level=logging.DEBUG)

def generate_synthetic_timeline() -> List[TimelineSecond]:
    """Generates a noisy, multi-modal signal simulating a gaming highlight."""
    timeline = []
    
    for t in range(40):
        # Base noise
        audio_peak = np.random.uniform(0.0, 0.2)
        emotion_score = np.random.uniform(0.0, 0.1)
        surprise_level = np.random.uniform(0.0, 0.1)
        motion_delta = np.random.uniform(0.0, 0.1)
        scene_conf = 0.5
        scene_type = "neutral"
        speech_energy = np.random.uniform(0.0, 0.2)
        keyword_weight = 0.2

        # 1. Travel Phase (0-10s)
        if t < 10:
            scene_type = "travel"
            motion_delta = np.random.uniform(0.1, 0.3)

        # 2. Combat Engagement (15-20s) - Action Peak
        elif 15 <= t <= 20:
            scene_type = "combat"
            audio_peak = np.random.uniform(0.6, 0.9)
            motion_delta = np.random.uniform(0.6, 0.9)
            speech_energy = np.random.uniform(0.4, 0.7)
            emotion_score = np.random.uniform(0.3, 0.6)

        # 3. Post-Combat Reaction (22-26s) - Surprise Peak
        elif 22 <= t <= 26:
            scene_type = "neutral"
            audio_peak = np.random.uniform(0.4, 0.6)
            emotion_score = np.random.uniform(0.8, 1.0)
            surprise_level = np.random.uniform(0.8, 1.0)
            speech_energy = np.random.uniform(0.7, 1.0)
            keyword_weight = 1.0 # e.g. "OMG WHAT"

        # 4. Silent Payoff (27-30s)
        elif 27 <= t <= 30:
            audio_peak = 0.05 # Silence
            emotion_score = 0.1

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
    
    events = engine.detect(timeline)
    
    print(f"\nDetected {len(events)} valid events:")
    for i, e in enumerate(events):
        print(f"\nEvent {i+1}:")
        print(f"  Type:     {e.event_type.upper()}")
        print(f"  Time:     {e.start:.1f}s - {e.end:.1f}s (Peak: {e.peak_time:.1f}s)")
        print(f"  Duration: {e.duration:.1f}s")
        print(f"  Scores:   Final={e.final_score:.3f}, Surprise={e.surprise_score:.3f}, Payoff={e.payoff_score:.3f}, Priority={e.priority}")

if __name__ == "__main__":
    test_fusion_engine()
