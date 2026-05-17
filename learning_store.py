"""
learning_store.py
Stores winning patterns for future optimization using a simple JSON file.
Incorporates Active Learning, Confidence Tracking, Recency Weighting, and Failure Recovery.
"""
import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

STORE_PATH = "learning_patterns.json"

def _load_store() -> Dict[str, Any]:
    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_store(data: Dict[str, Any]):
    with open(STORE_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def update_learning_store(winner: Dict[str, Any]):
    """
    Updates the learning store with the best pattern for the specific event type.
    Includes Stability & Adaptation Controls.
    """
    if not winner:
        return
        
    store = _load_store()
    event_type = winner.get("event_type", "neutral")
    winner_hook_type = winner.get("hook_type", "neutral")
    winner_caption_style = winner.get("caption_style", "dynamic")
    top_score = winner.get("final_score", 0.0)
    
    current_pattern = store.get(event_type, {})
    
    # Initialize if missing
    if not current_pattern:
        store[event_type] = {
            "best_hook_type": winner_hook_type,
            "best_caption_style": winner_caption_style,
            "confidence": 0.2,
            "sample_count": 1,
            "historical_score": top_score,
            "recovery_mode": False
        }
        _save_store(store)
        logger.info(f"Initialized learning store for event type: {event_type}")
        return

    # Update sample count
    current_pattern["sample_count"] = current_pattern.get("sample_count", 0) + 1

    # Recency Weighting (Trend Adaptation)
    historical = current_pattern.get("historical_score", 0.0)
    # Score = (recent_score * 0.7) + (historical_score * 0.3)
    new_historical = (top_score * 0.7) + (historical * 0.3)
    current_pattern["historical_score"] = round(new_historical, 3)

    # Failure Detection & Recovery
    # If the winning score is exceptionally low, all variants performed poorly
    performance_threshold = 0.25 # Arbitrary threshold for "poor performance"
    if top_score < performance_threshold:
        logger.warning(f"Failure Recovery triggered for {event_type}! Top score was only {top_score:.2f}")
        current_pattern["recovery_mode"] = True
        current_pattern["confidence"] = max(0.0, current_pattern.get("confidence", 0.0) * 0.5) # Reduce bias temporarily
    else:
        current_pattern["recovery_mode"] = False

    # Update confidence
    confidence = current_pattern.get("confidence", 0.0)
    if current_pattern.get("best_hook_type") == winner_hook_type:
        confidence = min(1.0, confidence + 0.1) # Increase if it wins repeatedly
    else:
        confidence -= 0.15 # Decrease faster if it loses
        
    if confidence <= 0.0:
        # Paradigm shift
        current_pattern["best_hook_type"] = winner_hook_type
        current_pattern["best_caption_style"] = winner_caption_style
        confidence = 0.2
        logger.info(f"Learning Shift for {event_type}: New best hook type is {winner_hook_type}")
        
    # Anti-Overconfidence Cap enforcement happens dynamically during generation, 
    # but we store the raw confidence capped at 1.0 here.
    current_pattern["confidence"] = round(confidence, 2)
    store[event_type] = current_pattern
    
    _save_store(store)
    logger.info(f"Updated learning store for {event_type} (Confidence: {confidence:.2f}, Samples: {current_pattern['sample_count']})")

def get_best_pattern(event_type: str) -> Dict[str, Any]:
    store = _load_store()
    return store.get(event_type, {})
