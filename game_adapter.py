"""
game_adapter.py — Semantic Enrichment Layer
=============================================
This module no longer SCORES or FILTERS moments.
That job belongs to arc_detector.py.

GameAdapter's new role:
  • Load game profile (optional — detection works without one)
  • Enrich ArcRegion objects with game-specific labels and CLIP prompts
  • Build dynamic CLIP visual prompts from transcript context
  • Expose profile metadata (weights, rules) for arc_detector to consume

The old hard-coded genre rules (travel rejection multipliers, spike-only
detection) are REMOVED. Arc shape detection handles those naturally —
flat signals never rise above threshold and never become arcs.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

from phase3_types import EventMoment, ArcShape, GameProfile

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Default CLIP prompt templates per arc shape
# These are used when a game profile doesn't provide its own templates.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CLIP_PROMPT_TEMPLATES: Dict[str, str] = {
    ArcShape.SPIKE.value:     "an intense action moment in a video game with explosions or combat: {context}",
    ArcShape.TENSION.value:   "a tense suspenseful chase or stealth sequence in a video game: {context}",
    ArcShape.COMEDY.value:    "a funny unexpected moment in a video game causing laughter: {context}",
    ArcShape.DRAMA.value:     "an emotional dramatic cutscene or conversation in a video game: {context}",
    ArcShape.TRIUMPH.value:   "a triumphant victory moment in a video game after a difficult challenge: {context}",
    ArcShape.DISCOVERY.value: "a player discovering something surprising or new in a video game: {context}",
}

# Human-readable arc labels per shape (used in GUI + LLM prompts)
DEFAULT_ARC_LABELS: Dict[str, str] = {
    ArcShape.SPIKE.value:     "Action Moment",
    ArcShape.TENSION.value:   "Suspense Arc",
    ArcShape.COMEDY.value:    "Comedy Beat",
    ArcShape.DRAMA.value:     "Dramatic Scene",
    ArcShape.TRIUMPH.value:   "Triumph Moment",
    ArcShape.DISCOVERY.value: "Discovery Moment",
}


class GameAdapter:
    """
    Enriches ArcRegion objects with game-specific semantic context.
    Loads a game profile if available, falls back to generic defaults.
    """

    def __init__(self, profiles_dir: str = "game_profiles"):
        self.profiles_dir = profiles_dir
        self.profile_data: Dict = {}
        self.profile: Optional[GameProfile] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Profile Loading
    # ─────────────────────────────────────────────────────────────────────────

    def load_profile(self, game_id: str) -> Dict:
        """Loads a game profile JSON. Falls back to generic if not found."""
        profile_path = os.path.join(self.profiles_dir, f"{game_id}.json")
        if not os.path.exists(profile_path):
            profile_path = os.path.join(self.profiles_dir, "generic.json")

        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    self.profile_data = json.load(f)
                    self.profile = GameProfile.from_json(self.profile_data)
                logger.info(f"GameAdapter: loaded profile '{game_id}' from {profile_path}")
            except Exception as e:
                logger.warning(f"GameAdapter: failed to load profile '{game_id}': {e}")
                self.profile_data = {}
                self.profile = None
        else:
            logger.warning(f"GameAdapter: no profile found for '{game_id}', using bare defaults.")
            self.profile_data = {}
            self.profile = None

        return self.profile_data

    def get_profile(self) -> Optional[GameProfile]:
        return self.profile

    # ─────────────────────────────────────────────────────────────────────────
    # Arc Enrichment (the main public API)
    # ─────────────────────────────────────────────────────────────────────────

    def enrich_arcs(self, arcs: List[EventMoment]) -> List[EventMoment]:
        """
        Adds label and CLIP prompt to each EventMoment.
        Does NOT modify scores or filter arcs.
        """
        for arc in arcs:
            arc.label      = self._label_arc(arc)
            arc.clip_prompt = self._build_clip_prompt(arc)
        return arcs

    def enrich_arc(self, arc: EventMoment) -> EventMoment:
        """Single-arc version of enrich_arcs."""
        arc.label      = self._label_arc(arc)
        arc.clip_prompt = self._build_clip_prompt(arc)
        return arc

    # ─────────────────────────────────────────────────────────────────────────
    # Label Generation
    # ─────────────────────────────────────────────────────────────────────────

    def _label_arc(self, arc: EventMoment) -> str:
        """
        Generates a human-readable label for the arc.
        Uses game-profile label overrides when available, otherwise uses
        the transcript to produce a richer description.
        """
        shape_key = arc.event_type

        # Profile may define custom labels per shape
        profile_labels = self.profile_data.get("arc_labels", {})
        if shape_key in profile_labels:
            return profile_labels[shape_key]

        # Build a context-aware label from transcript keywords
        context_nouns = self._extract_context_words(arc.transcript, max_words=3)
        base_label = DEFAULT_ARC_LABELS.get(shape_key, "Highlight")

        if context_nouns:
            return f"{base_label} — {context_nouns}"
        return base_label

    # ─────────────────────────────────────────────────────────────────────────
    # CLIP Prompt Generation
    # ─────────────────────────────────────────────────────────────────────────

    def _build_clip_prompt(self, arc: EventMoment) -> str:
        """
        Builds a CLIP-compatible visual description of the arc.

        Priority:
          1. Game profile template (e.g., RDR2 knows about cowboys/lawmen)
          2. Default template filled with transcript nouns
          3. Bare default if no transcript
        """
        shape_key = arc.event_type

        # 1. Profile-specific templates
        profile_templates = self.profile_data.get("clip_prompt_templates", {})
        template = profile_templates.get(shape_key, DEFAULT_CLIP_PROMPT_TEMPLATES.get(shape_key, ""))

        # 2. Fill {context} placeholder with transcript-derived nouns
        context = self._extract_context_words(arc.transcript, max_words=5)
        game_name = self.profile_data.get("game_name", "")

        if "{context}" in template:
            context_str = context if context else game_name
            prompt = template.format(context=context_str)
        else:
            prompt = template

        # Add game name prefix if available and not already present
        if game_name and game_name.lower() not in prompt.lower():
            prompt = f"[{game_name}] {prompt}"

        return prompt or f"a compelling {shape_key} moment in a video game"

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_context_words(transcript: str, max_words: int = 4) -> str:
        """
        Extracts the most meaningful content words from a transcript snippet.
        Simple heuristic: take longer words (≥5 chars) that aren't stop-words.
        """
        if not transcript:
            return ""

        STOP_WORDS = {
            "about", "after", "again", "before", "could", "every", "going",
            "gonna", "their", "there", "these", "thing", "think", "those",
            "through", "under", "until", "where", "which", "while", "would",
            "right", "gonna", "really", "there", "they're", "that", "this",
            "with", "just", "yeah", "okay", "like", "know", "have",
        }

        words = re.findall(r"[a-zA-Z']+", transcript.lower())
        content = [
            w for w in words
            if len(w) >= 5 and w not in STOP_WORDS
        ]

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for w in content:
            if w not in seen:
                seen.add(w)
                unique.append(w)

        return ", ".join(unique[:max_words])

    # ─────────────────────────────────────────────────────────────────────────
    # Legacy compatibility shims
    # These keep old call-sites working during the transition.
    # They are no-ops — all scoring now happens in arc_detector.py.
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate(self, windows, audio_rows, visual_rows, emotion_rows, speech_rows):
        """No-op shim — calibration is now implicit in ArcDetector baseline."""
        pass

    def detect_events(self, *args, **kwargs) -> List[str]:
        """No-op shim — event detection replaced by ArcDetector."""
        return []

    def evaluate(self, *args, **kwargs) -> float:
        """No-op shim — relevance scoring replaced by ArcDetector quality score."""
        return 1.0

    def compute_two_layer_score(self, audio_val, visual_val, emotion_val, speech_val, events) -> float:
        """
        Shim for scoring_engine.build_segment_results().
        Returns a simple weighted sum so legacy code paths don't break.
        """
        weights = self.profile_data.get("score_weights", {})
        a = weights.get("audio",   0.25)
        v = weights.get("visual",  0.25)
        e = weights.get("emotion", 0.25)
        s = weights.get("speech",  0.25)
        total = a + v + e + s or 1.0
        raw = (a * audio_val + v * visual_val + e * emotion_val + s * speech_val) / total
        return min(max(raw, 0.0), 1.0)
