"""
narrative_engine.py — LLM Storyteller
=======================================
The LLM's new job: given a detected arc, generate a compelling hook sentence
and short title. It is NOT used for classification or filtering.

Arc shape detection (arc_detector.py) already decided WHAT the moment is.
The LLM now decides HOW to frame it for an audience.

If Ollama is unavailable, shape-aware fallback templates are used so
clips are always produced — just without bespoke hooks.
"""

import json
import logging
import subprocess
import time
from typing import List, Optional, Dict

from phase3_types import EventMoment

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback hook templates (used when LLM is unavailable)
# Indexed by ArcShape.value → list of template strings.
# The best template is chosen based on available transcript content.
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_HOOKS: Dict[str, List[str]] = {
    "combat": [
        "This moment changed everything...",
        "Nobody saw this coming.",
        "Watch what happens next.",
    ],
    "travel": [
        "No horse. No gun. Just survival instinct.",
        "The longest few minutes of the stream.",
        "Can they escape? Watch to find out.",
    ],
    "reaction": [
        "This was NOT supposed to happen 😂",
        "I genuinely could not believe this.",
        "The game had other plans...",
    ],
    "surprise": [
        "Wait... what is THAT?",
        "Nobody talks about this hidden detail.",
        "Found something the devs didn't want you to see.",
    ],
    "neutral": [
        "This scene hit different.",
        "The story gets real here.",
        "You need to hear this.",
    ],
}

FALLBACK_TITLES: Dict[str, str] = {
    "combat":   "Insane Action Moment",
    "travel":   "Tense Chase No One Expected",
    "reaction": "Funniest Thing That Happened",
    "neutral":  "Emotional Story Beat",
    "surprise": "Hidden Secret Discovered",
}


# ─────────────────────────────────────────────────────────────────────────────
# NarrativeEngine
# ─────────────────────────────────────────────────────────────────────────────

class NarrativeEngine:
    """
    Generates hooks, titles, and share-worthiness explanations for detected arcs.

    Workflow:
      1. Try Ollama (local LLM) for each arc — rich, context-aware hooks
      2. Fall back to shape-aware template hooks if Ollama is unavailable
    """

    def __init__(self, model: str = "llama3:8b"):
        self.model = model
        self.available = False
        self._check_ollama()

    def _check_ollama(self):
        """Quick connectivity check + optional server start."""
        import requests
        url = "http://localhost:11434/api/tags"
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                self.available = True
                return
        except Exception:
            pass

        logger.info("Ollama not detected — attempting to start 'ollama serve'...")
        try:
            subprocess.run(["ollama", "--version"], check=True, capture_output=True)
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            for _ in range(5):
                time.sleep(2)
                try:
                    r = requests.get(url, timeout=1)
                    if r.status_code == 200:
                        logger.info("Ollama started successfully.")
                        self.available = True
                        return
                except Exception:
                    continue
        except Exception:
            pass

        logger.info("Ollama unavailable — NarrativeEngine will use fallback templates.")
        self.available = False

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def enrich_arcs(self, arcs: List[EventMoment], game_name: str = "") -> List[EventMoment]:
        """
        Adds hook_sentence and short_title to each ArcRegion.
        Returns the same list, mutated in place.
        """
        for arc in arcs:
            self._enrich_single(arc, game_name)
        return arcs

    # ─────────────────────────────────────────────────────────────────────────
    # Single Arc Enrichment
    # ─────────────────────────────────────────────────────────────────────────

    def _enrich_single(self, arc: EventMoment, game_name: str = "") -> EventMoment:
        if self.available and arc.transcript.strip():
            try:
                result = self._query_ollama_for_arc(arc, game_name)
                if result:
                    arc.hook_sentence = result.get("hook", "")
                    arc.short_title   = result.get("title", "")
                    return arc
            except Exception as e:
                logger.warning(f"NarrativeEngine: LLM enrichment failed for arc at {arc.start:.1f}s: {e}")

        # Fallback
        arc.hook_sentence = self._fallback_hook(arc)
        arc.short_title   = self._fallback_title(arc)
        return arc

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Query
    # ─────────────────────────────────────────────────────────────────────────

    def _query_ollama_for_arc(self, arc: EventMoment, game_name: str) -> Optional[Dict]:
        """
        Queries Ollama with a structured prompt that describes the arc shape
        and transcript, and asks for a hook + title in JSON format.
        """
        import requests

        shape_descriptions = {
            "combat":   "a sudden intense action spike (kill, explosion, or dramatic event)",
            "travel":   "a sustained tension arc — rising danger, chase, evasion, or stealth sequence",
            "reaction": "a comedy beat or strong reaction — an unexpected, funny, or absurd moment",
            "neutral":  "a dramatic scene — emotional dialogue, story revelation, or character moment",
            "surprise": "a discovery moment — finding something surprising, hidden, or new",
        }

        shape_desc = shape_descriptions.get(arc.event_type, "a compelling gaming moment")
        game_context = f"Game: {game_name}. " if game_name else ""
        transcript_snippet = arc.transcript[:300].strip() if arc.transcript else "(no dialogue)"

        prompt = (
            f"You are a YouTube Shorts editor who writes viral hooks for gaming clips.\n\n"
            f"{game_context}"
            f"A {arc.duration:.0f}-second clip was detected. It is: {shape_desc}.\n"
            f"What the player said during this moment: \"{transcript_snippet}\"\n\n"
            f"Write:\n"
            f"1. hook — A single sentence (max 12 words) to open the Short. "
            f"Make it create curiosity or urgency. No clickbait. Be authentic.\n"
            f"2. title — A YouTube Shorts title (max 8 words). Punchy, specific, not generic.\n\n"
            f"Respond ONLY with valid JSON: {{\"hook\": \"...\", \"title\": \"...\"}}"
        )

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=25,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()

            # Extract JSON even if surrounded by other text
            match_start = raw.find("{")
            match_end   = raw.rfind("}") + 1
            if match_start != -1 and match_end > match_start:
                parsed = json.loads(raw[match_start:match_end])
                # Validate expected keys
                if "hook" in parsed and "title" in parsed:
                    return parsed

        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                self.available = False
                logger.warning("Ollama connection lost during narrative enrichment.")
            raise

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Fallback template hooks
    # ─────────────────────────────────────────────────────────────────────────

    def _fallback_hook(self, arc: EventMoment) -> str:
        """
        Picks the best fallback hook for this arc.
        If there's useful transcript content, tries to incorporate it.
        """
        shape_key = arc.event_type
        templates = FALLBACK_HOOKS.get(shape_key, ["Watch this moment."])

        # If transcript has content, try to make the hook slightly more specific
        if arc.transcript.strip():
            words = arc.transcript.strip().split()
            if len(words) >= 4:
                # Use first 6 words of transcript as raw context
                snippet = " ".join(words[:6]).rstrip(".,!?")
                # Only use it if it seems like natural speech (not just noise)
                if len(snippet) > 10:
                    return templates[0]  # Still use template — transcript becomes context in title

        # Rotate based on arc start time (deterministic variety)
        idx = int(arc.start / 60) % len(templates)
        return templates[idx]

    def _fallback_title(self, arc: EventMoment) -> str:
        """Shape-aware fallback title."""
        shape_key = arc.event_type
        base_title = FALLBACK_TITLES.get(shape_key, "Epic Gaming Moment")

        # Add approximate timestamp for context
        minutes = int(arc.start // 60)
        if minutes > 0:
            return f"{base_title} (at {minutes}min)"
        return base_title


# ─────────────────────────────────────────────────────────────────────────────
# Legacy compatibility — old scene-based API
# Kept so existing imports in other modules don't break.
# ─────────────────────────────────────────────────────────────────────────────

def detect_scenes(segment_results, threshold=0.3, max_scene_duration=60.0):
    """
    Legacy shim. The new pipeline uses ArcDetector instead of scene grouping.
    Returns an empty list — hook_analyzer.py now calls ArcDetector directly.
    """
    logger.debug("detect_scenes() called — this is a legacy shim. Use ArcDetector.")
    return []


def stitch_narrative_arcs(scenes, top_k=5, max_arc_duration=90.0):
    """
    Legacy shim. Arc stitching is now handled by ArcDetector + story_builder.
    Returns empty list.
    """
    logger.debug("stitch_narrative_arcs() called — this is a legacy shim.")
    return []


class NarrativeAI(NarrativeEngine):
    """
    Legacy alias for NarrativeEngine. Kept for backwards compatibility
    with any code that still imports NarrativeAI.
    """
    pass
