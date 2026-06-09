"""Narrative Arc Builder — maps event sequences to a five-stage narrative
arc model for cinematic clip construction.

Each genre defines its own arc template with distinct timing offsets around
the detected peak, producing the timing boundaries that downstream editing
stages use to place hooks, build-ups, climaxes, recovery beats, and
resolutions.
"""

import logging
from typing import List, Dict, Optional

from core.cache.timeline_frame import TimelineFrame
from phase3_types import EventMoment

logger = logging.getLogger(__name__)


class NarrativeArcBuilder:
    """Build five-stage narrative arcs from detected event sequences.

    Supported genres
    ----------------
    * **fps** — fast, tight arcs centred on a combat peak.
    * **racing** — wider arcs with more lead-in and extended recovery.

    Each template defines five abstract stages whose concrete timestamps
    are computed relative to the event's ``peak_time``.
    """

    TEMPLATES: Dict[str, List[str]] = {
        "fps": [
            "search",       # Stage 1 — calm before the storm
            "engage",       # Stage 2 — build-up / approach
            "climax",       # Stage 3 — peak action
            "reaction",     # Stage 4 — immediate aftermath / streamer reaction
            "resolution",   # Stage 5 — wind-down / result screen
        ],
        "racing": [
            "acceleration", # Stage 1 — speed ramp-up
            "drift_or_chase",  # Stage 2 — aggressive manoeuvre
            "climax",       # Stage 3 — overtake / crash / finish
            "recovery",     # Stage 4 — stabilise / slow-down
            "reaction",     # Stage 5 — streamer reaction
        ],
    }

    # Timing offsets (in seconds) relative to peak_time for each stage
    # boundary.  Order: hook_start, buildup_start, climax, recovery,
    # resolution.
    _OFFSETS: Dict[str, List[float]] = {
        "fps":    [-4.0, -2.0, 0.0, 3.0, 6.0],
        "racing": [-10.0, -5.0, 0.0, 5.0, 10.0],
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_arc(
        self,
        sequence: EventMoment,
        timeline: List[TimelineFrame],
        genre: str,
    ) -> Dict[str, float]:
        """Compute the five-stage timing boundaries for *sequence*.

        Parameters
        ----------
        sequence:
            The event moment (or merged sequence) whose ``peak_time``
            anchors the arc.
        timeline:
            Full list of ``TimelineFrame`` objects for the source video.
            Used for future energy-aware boundary adjustments (currently
            reserved).
        genre:
            One of the supported genre keys (``"fps"``, ``"racing"``).
            Falls back to ``"fps"`` if the genre is unrecognised.

        Returns
        -------
        Dict[str, float]
            A dictionary with the keys ``hook_start``, ``buildup_start``,
            ``climax``, ``recovery``, and ``resolution``, each mapped to
            an absolute timestamp (seconds).
        """
        if genre not in self._OFFSETS:
            logger.warning(
                "Unknown genre '%s' — defaulting to 'fps' arc template.",
                genre,
            )
            genre = "fps"

        offsets = self._OFFSETS[genre]
        peak = sequence.peak_time

        boundaries: Dict[str, float] = {
            "hook_start":     max(peak + offsets[0], 0.0),
            "buildup_start":  max(peak + offsets[1], 0.0),
            "climax":         peak + offsets[2],
            "recovery":       peak + offsets[3],
            "resolution":     peak + offsets[4],
        }

        # Clamp hook_start to not exceed the event's own start
        if boundaries["hook_start"] < sequence.start:
            boundaries["hook_start"] = sequence.start

        # If a timeline is available, clamp resolution to video length
        if timeline:
            video_end = timeline[-1].timestamp
            boundaries["resolution"] = min(boundaries["resolution"], video_end)

        logger.debug(
            "NarrativeArc [%s] peak=%.1f → hook=%.1f build=%.1f climax=%.1f "
            "recovery=%.1f resolution=%.1f",
            genre,
            peak,
            boundaries["hook_start"],
            boundaries["buildup_start"],
            boundaries["climax"],
            boundaries["recovery"],
            boundaries["resolution"],
        )
        return boundaries

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def get_template(self, genre: str) -> List[str]:
        """Return the stage names for *genre*, defaulting to ``"fps"``."""
        return self.TEMPLATES.get(genre, self.TEMPLATES["fps"])

    def stage_at(
        self,
        timestamp: float,
        boundaries: Dict[str, float],
    ) -> Optional[str]:
        """Return the narrative stage active at *timestamp*, or ``None``
        if the timestamp falls outside the arc.

        The stage ordering is: hook → buildup → climax → recovery →
        resolution.
        """
        stages = [
            ("hook",       boundaries["hook_start"],    boundaries["buildup_start"]),
            ("buildup",    boundaries["buildup_start"],  boundaries["climax"]),
            ("climax",     boundaries["climax"],         boundaries["recovery"]),
            ("recovery",   boundaries["recovery"],       boundaries["resolution"]),
            ("resolution", boundaries["resolution"],     boundaries["resolution"] + 2.0),
        ]
        for name, stage_start, stage_end in stages:
            if stage_start <= timestamp < stage_end:
                return name
        return None
