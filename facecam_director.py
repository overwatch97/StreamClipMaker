import os
import json
from typing import List, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_intersection(box1, box2):
    # box format: [x_min, y_min, x_max, y_max]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    return intersection_area / box1_area

def get_facecam_layout(emotion_score: float, profile: Dict = None):
    """
    Returns the FFmpeg overlay coordinates and scale based on emotion score and
    spatial deadzones rules.
    """
    if profile is None:
        # Fallback to generic-fps
        try:
            with open(os.path.join(SCRIPT_DIR, "game_profiles", "generic-fps.json")) as f:
                profile = json.load(f)
        except Exception:
            profile = {}

    # Scale up for high emotion (Pop-off)
    scale = 0.35 if emotion_score > 85 else 0.28
    
    # In 9:16, width is 1.0, height is 1.0 (normalized relative to bounding box)
    cam_h_norm = scale * (9/16)
    cam_w_norm = scale
    
    # Default to Bottom Right
    default_x = 1.0 - cam_w_norm - 0.02 # 20px padding approx
    default_y = 1.0 - cam_h_norm - 0.01

    fc_box = [default_x, default_y, default_x + cam_w_norm, default_y + cam_h_norm]
    
    x_expr = "W-w-20"
    y_expr = "H-h-20"
    
    if "spatial_rules" in profile:
        rules = profile["spatial_rules"]
        dead_zones = rules.get("dead_zones", [])
        safe_zones = rules.get("safe_zones", [])
        
        intersected = False
        for dz in dead_zones:
            if "x_min" in dz:
                dz_box = [dz["x_min"], dz["y_min"], dz["x_max"], dz["y_max"]]
                if check_intersection(fc_box, dz_box) > 0.10: # >10% intersection
                    intersected = True
                    break
        
        if intersected and safe_zones:
            sz = safe_zones[0]
            x_expr = f"(W-w)*{sz.get('x', 0.5)}"
            y_expr = f"(H-h)*{sz.get('y', 0.1)}"
            
    return {
        "ffmpeg_overlay": f"overlay={x_expr}:{y_expr}",
        "scale": scale
    }
