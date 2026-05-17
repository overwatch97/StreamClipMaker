"""
batch_renderer.py
Renders multiple variants for a single event sequentially.
"""
import logging
import json
from typing import List, Dict, Any
from editing_engine import render_short

logger = logging.getLogger(__name__)

def render_variants(video_path: str, event: Dict[str, Any], variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    outputs = []

    for v in variants:
        logger.info(f"Rendering variant {v['variant_id']}")
        
        input_data = {
            "video_path": video_path,
            "event": event,
            "hook": v["hook"],
            "captions": v["captions"],
            "event_id": v["variant_id"] 
        }
        
        try:
            result_json = render_short(input_data)
            result = json.loads(result_json)
            
            outputs.append({
                "variant_id": v["variant_id"],
                "output_path": result.get("output_path"),
                "event_type": v.get("event_type", "neutral"),
                "hook": v["hook"],
                "caption_style": "dynamic",
                "parent_event_id": event.get("id", "unknown_event")
            })
        except Exception as e:
            logger.error(f"Failed to render variant {v['variant_id']}: {e}")

    return outputs
