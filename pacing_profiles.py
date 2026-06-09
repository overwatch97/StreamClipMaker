"""
pacing_profiles.py — Adaptive Pacing Profile Definitions
=========================================================
Maps game genres/styles to detection sensitivity profiles.
This allows the Event Fusion Engine to adapt its thresholds
without globally weakening quality filters.

Pacing styles:
  - "fps"       : Fast-paced shooters (Overwatch, DOOM, Valorant)
  - "cinematic" : Narrative/atmospheric games (RDR2, Cyberpunk, GTA story)
  - "balanced"  : Middle-ground (general, RPG, open-world action)

Each profile defines:
  - start_threshold     : Score needed to BEGIN an event region
  - end_threshold       : Score needed to END an event region (hysteresis)
  - min_prominence      : Minimum peak prominence vs. local average
  - min_intensity       : Absolute minimum peak score
  - min_surprise_payoff : Minimum combined surprise/payoff score
  - percentile_floor    : Percentile of stream scores used as dynamic threshold
  - fallback_n          : How many top segments to select if 0 clips found
"""

from typing import Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# Core Pacing Profiles
# ─────────────────────────────────────────────────────────────────────────────

PACING_PROFILES: Dict[str, Dict[str, Any]] = {

    # Fast-paced competitive shooters: high motion, loud audio, sharp spikes
    "fps": {
        "start_threshold":      0.70,
        "end_threshold":        0.50,
        "min_prominence":       0.15,
        "min_intensity":        0.70,
        "min_surprise_payoff":  0.60,
        "percentile_floor":     85,
        "fallback_n":           2,
        "description": "Tuned for fast-paced FPS games with rapid intensity spikes",
    },

    # Narrative/cinematic games: emotional ramps, dialogue, slower pacing
    "cinematic": {
        "start_threshold":      0.50,
        "end_threshold":        0.35,
        "min_prominence":       0.08,
        "min_intensity":        0.45,
        "min_surprise_payoff":  0.38,
        "percentile_floor":     75,
        "fallback_n":           3,
        "description": "Tuned for story-driven cinematic games with slower narrative pacing",
    },

    # Balanced: open-world action, RPGs, mixed content
    "balanced": {
        "start_threshold":      0.60,
        "end_threshold":        0.42,
        "min_prominence":       0.12,
        "min_intensity":        0.55,
        "min_surprise_payoff":  0.50,
        "percentile_floor":     80,
        "fallback_n":           2,
        "description": "Balanced profile for RPGs and mixed open-world games",
    },

    # Racing games: sustained speed and drift chains form smoother curves than FPS spikes.
    "racing": {
        "start_threshold":      0.38,
        "end_threshold":        0.28,
        "min_prominence":       0.04,
        "min_intensity":        0.38,
        "min_surprise_payoff":  0.20,
        "percentile_floor":     65,
        "fallback_n":           4,
        "description": "Tuned for racing games with sustained momentum and cinematic pacing",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Genre → Pacing Style Mapping
# ─────────────────────────────────────────────────────────────────────────────

GENRE_TO_PACING: Dict[str, str] = {
    # FPS / competitive
    "fps":                    "fps",
    "tactical-shooter":       "fps",
    "battle-royale":          "fps",
    "hero-shooter":           "fps",
    "arena-shooter":          "fps",

    # Racing
    "racing":                 "racing",

    # Cinematic / narrative
    "open-world-adventure":   "cinematic",
    "narrative":              "cinematic",
    "story-driven":           "cinematic",
    "rpg-story":              "cinematic",
    "adventure":              "cinematic",
    "survival-horror":        "cinematic",
    "cinematic":              "cinematic",

    # Balanced
    "open-world":             "balanced",
    "rpg":                    "balanced",
    "action-rpg":             "balanced",
    "moba":                   "balanced",
    "strategy":               "balanced",
    "general":                "balanced",
}


def get_pacing_profile(pacing_style: str) -> Dict[str, Any]:
    """
    Return a pacing profile dict by style name.
    Falls back to 'balanced' if the style is unknown.
    """
    return PACING_PROFILES.get(pacing_style, PACING_PROFILES["balanced"])


def resolve_pacing_style(genre: str, explicit_style: str = None) -> str:
    """
    Resolve the pacing style from an explicit override or genre mapping.

    Args:
        genre:          The game's genre string (from game profile JSON).
        explicit_style: An explicit 'pacing_style' field from the game profile JSON.
                        If set, it takes priority over genre mapping.

    Returns:
        A pacing style key: "fps", "cinematic", or "balanced".
    """
    if explicit_style and explicit_style in PACING_PROFILES:
        return explicit_style
    return GENRE_TO_PACING.get(genre, "balanced")
