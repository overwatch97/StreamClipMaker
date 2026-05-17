"""
winner_selector.py
Selects the best-performing variant based on performance scores.
"""
import logging
from typing import List, Dict, Any

# Import calculate_score from where we placed it (we will append to the existing scoring_engine.py)
from scoring_engine import calculate_variant_score

logger = logging.getLogger(__name__)

def select_best(variants_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not variants_data:
        return None
    
    best_variant = max(variants_data, key=calculate_variant_score)
    top_score = calculate_variant_score(best_variant)
    best_variant['final_score'] = top_score
    logger.info(f"Selected winner: {best_variant['variant_id']} with score {top_score:.2f}")
    return best_variant
