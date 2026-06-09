"""Sequence Assembler — converts isolated event spikes into continuous
cinematic sequences by stitching nearby compatible events together.

Multiple events separated by short temporal gaps are merged into a single
sequence.  Merged events receive a combo multiplier on their maximum score
and a unified label derived from the highest-priority constituent.
"""

import logging
from typing import List, Dict, Any

from phase3_types import EventMoment, EVENT_PRIORITIES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compatibility groups — events within the same group can be merged.
# ---------------------------------------------------------------------------
_COMPAT_GROUPS: Dict[str, str] = {
    "combat":   "action",
    "surprise": "action",
    "racing":   "racing",
    "drift":    "racing",
    "travel":   "racing",
}

# "reaction" is universally compatible with every other type.
_UNIVERSAL_TYPES = {"reaction"}


class SequenceAssembler:
    """Merge nearby, compatible ``EventMoment`` instances into longer
    cinematic sequences."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def assemble(
        self,
        events: List[EventMoment],
        gap_threshold: float = 6.0,
    ) -> List[EventMoment]:
        """Assemble isolated events into continuous sequences.

        Parameters
        ----------
        events:
            Unsorted list of detected event moments.
        gap_threshold:
            Maximum gap (seconds) between the end of one event and the
            start of the next for them to be considered part of the same
            sequence.

        Returns
        -------
        List[EventMoment]
            A (potentially shorter) list where adjacent compatible events
            have been merged.  Each merged event carries a 1.10× combo
            multiplier on the maximum constituent score.
        """
        if not events:
            return []

        sorted_events = sorted(events, key=lambda e: e.start)
        merged: List[EventMoment] = []
        current_group: List[EventMoment] = [sorted_events[0]]

        for ev in sorted_events[1:]:
            prev = current_group[-1]
            gap = ev.start - prev.end

            if gap <= gap_threshold and self._are_compatible(prev.event_type, ev.event_type):
                current_group.append(ev)
            else:
                merged.append(self._merge_group(current_group))
                current_group = [ev]

        # Flush the last group
        merged.append(self._merge_group(current_group))

        logger.info(
            "SequenceAssembler: %d events → %d sequences (gap ≤ %.1fs)",
            len(events),
            len(merged),
            gap_threshold,
        )
        return merged

    # ------------------------------------------------------------------
    # Compatibility check
    # ------------------------------------------------------------------
    def _are_compatible(self, type_a: str, type_b: str) -> bool:
        """Return ``True`` when *type_a* and *type_b* may be merged.

        Rules
        -----
        * Any type is compatible with ``reaction``.
        * Two types sharing the same compatibility group are compatible.
        """
        if type_a in _UNIVERSAL_TYPES or type_b in _UNIVERSAL_TYPES:
            return True

        group_a = _COMPAT_GROUPS.get(type_a)
        group_b = _COMPAT_GROUPS.get(type_b)

        if group_a is not None and group_a == group_b:
            return True

        return False

    # ------------------------------------------------------------------
    # Label selection
    # ------------------------------------------------------------------
    def _determine_sequence_label(self, type_a: str, type_b: str) -> str:
        """Pick the higher-priority label from two event types.

        Falls back to ``type_a`` when both have equal or unknown priority.
        """
        prio_a = EVENT_PRIORITIES.get(type_a, 0)
        prio_b = EVENT_PRIORITIES.get(type_b, 0)
        return type_a if prio_a >= prio_b else type_b

    # ------------------------------------------------------------------
    # Internal merge helper
    # ------------------------------------------------------------------
    def _merge_group(self, group: List[EventMoment]) -> EventMoment:
        """Collapse a list of compatible events into one ``EventMoment``.

        The merged event spans from the earliest start to the latest end
        and inherits the peak time / score from the strongest constituent.
        A 1.10× combo multiplier is applied when more than one event is
        present.
        """
        if len(group) == 1:
            return group[0]

        combo_multiplier = 1.10

        # Find the constituent with the highest final score
        best = max(group, key=lambda e: e.final_score)
        boosted_score = min(best.final_score * combo_multiplier, 1.0)

        # Derive the unified label across the group
        label = group[0].event_type
        for ev in group[1:]:
            label = self._determine_sequence_label(label, ev.event_type)

        merged_event = EventMoment(
            event_type=label,
            start=group[0].start,
            end=group[-1].end,
            peak_time=best.peak_time,
            duration=group[-1].end - group[0].start,
            final_score=boosted_score,
            surprise_score=max(e.surprise_score for e in group),
            conflict_score=max(e.conflict_score for e in group),
            payoff_score=max(e.payoff_score for e in group),
            priority=EVENT_PRIORITIES.get(label, 1),
            scene_type=best.scene_type,
            features=best.features,
            transcript=" ".join(e.transcript for e in group if e.transcript),
            hook_sentence=best.hook_sentence,
            short_title=best.short_title,
            label=label,
            clip_prompt=best.clip_prompt,
            event_confidence=best.event_confidence,
        )

        logger.debug(
            "Merged %d events [%.1f–%.1f] → '%s' (score %.3f → %.3f)",
            len(group),
            merged_event.start,
            merged_event.end,
            label,
            best.final_score,
            boosted_score,
        )
        return merged_event
