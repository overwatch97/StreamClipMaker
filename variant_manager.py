"""
variant_manager.py
Generates variants per event with Active Learning Bias Weighting and Stability Controls.
"""
import uuid
import logging
import random
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def classify_hook_type(hook: str) -> str:
    """Heuristically classify a hook into a type."""
    text = hook.lower()
    if any(w in text for w in ["wait", "what", "ready", "guard", "😳"]):
        return "surprise"
    if any(w in text for w in ["wrong", "fight", "mistake", "💀"]):
        return "conflict"
    if any(w in text for w in ["bro", "😭", "lost it"]):
        return "reaction"
    if any(w in text for w in ["sense", "fail", "😂"]):
        return "humor"
    return "neutral"

def generate_variants(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    from hook_generator import HookGenerator
    from caption_generator import CaptionGenerator
    from learning_store import get_best_pattern
    
    hg = HookGenerator()
    cg = CaptionGenerator()
    
    event_type = event.get("event_type", "neutral")
    
    # 1. Load Learning Patterns
    learning = get_best_pattern(event_type)
    
    preferred_hook_type = learning.get("best_hook_type", None)
    preferred_caption_style = learning.get("best_caption_style", None)
    confidence = float(learning.get("confidence", 0.0))
    sample_count = int(learning.get("sample_count", 0))
    recovery_mode = bool(learning.get("recovery_mode", False))
    
    # Minimum Data Threshold & Failure Recovery Check
    if sample_count < 5 or recovery_mode:
        bias_weight = 1.0 # Disable bias, pure exploration
        logger.info(f"Bias disabled for {event_type} (Samples: {sample_count}, Recovery: {recovery_mode})")
    else:
        # Adaptive Bias Strength
        bias_weight = min(1.5, 1.0 + (confidence * 0.5))
        logger.info(f"Loaded learning for {event_type}: Hook={preferred_hook_type}, Cap={preferred_caption_style}, Bias={bias_weight:.2f}")

    event_data = {
        "event_type": event_type,
        "emotion": "excitement",
        "intensity": event.get("intensity", 5),
        "peak_time": event.get("peak_time", 2.0),
        "duration": event.get("duration", 10.0),
        "transcript": event.get("transcript", ""),
        "context": "A compelling moment."
    }
    
    if preferred_caption_style and bias_weight > 1.0:
        event_data["context"] += f" Style preference: {preferred_caption_style}."

    raw_hooks = hg.generate_hooks(event_data)
    captions = cg.generate_captions(event_data)

    scored_hooks = []
    for hook in raw_hooks:
        score = random.random()
        h_type = classify_hook_type(hook)
        
        # Apply bias weighting
        if preferred_hook_type and h_type == preferred_hook_type:
            score *= bias_weight
            
        scored_hooks.append({"hook": hook, "type": h_type, "score": score})
        
    scored_hooks.sort(key=lambda x: x["score"], reverse=True)
    
    # Variant Diversity Guard
    # Ensure we don't just pick two identical hook types if we have options
    selected_hooks = [scored_hooks[0]]
    for h in scored_hooks[1:]:
        if len(selected_hooks) >= 2:
            break
        
        # Ensure at least 1 exploratory variant if possible
        if h["type"] != selected_hooks[0]["type"]:
            selected_hooks.append(h)
            
    # Fallback if all hooks were the same type
    if len(selected_hooks) < 2 and len(scored_hooks) >= 2:
        selected_hooks.append(scored_hooks[1])

    variants = []
    for h in selected_hooks:
        variants.append({
            "variant_id": str(uuid.uuid4()),
            "event_type": event_type,
            "hook": h["hook"],
            "hook_type": h["type"],
            "caption_style": preferred_caption_style or "dynamic",
            "captions": captions
        })

    logger.info(f"Generated {len(variants)} variants (Top Bias Score: {selected_hooks[0]['score']:.2f})")
    return variants
