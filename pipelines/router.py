"""
Pipeline Router — selects the appropriate genre pipeline based on game profile
and automated signal analysis of the timeline.

The router implements a two-tier selection strategy:
  1. **Profile override** — if a GameProfile is supplied with a known genre,
     the corresponding registered pipeline is returned immediately.
  2. **Automated detection** — when no profile is available (or the profile
     genre is "general"), lightweight signal heuristics decide between the
     available pipelines.  FPS/action is the safe fallback.
"""

from typing import List, Dict, Tuple, Optional
import logging

from core.cache.timeline_frame import TimelineFrame
from phase3_types import GameProfile, EventMoment
from pipelines.base_pipeline import BaseGenrePipeline
from pipelines.registry import PipelineRegistry

# Trigger pipeline self-registration
import pipelines.fps_action_pipeline  # noqa: F401
import pipelines.racing_pipeline      # noqa: F401

logger = logging.getLogger(__name__)

# ── Genre-to-registry-key mapping ──────────────────────────────────────────
_GENRE_PIPELINE_MAP: Dict[str, str] = {
    "fps":              "fps_action",
    "tactical-shooter": "fps_action",
    "battle-royale":    "fps_action",
    "racing":           "racing",
}


class PipelineRouter:
    """Routes incoming timeline data to the correct genre-specific pipeline.

    Parameters
    ----------
    confidence_threshold : float
        Minimum confidence required for the auto-detected genre to be
        selected over the FPS fallback.  Default ``0.45``.
    """

    def __init__(self, confidence_threshold: float = 0.45) -> None:
        self.threshold = confidence_threshold

    # ── Primary API ────────────────────────────────────────────────────────

    def route(
        self,
        timeline: List[TimelineFrame],
        profile: Optional[GameProfile] = None,
    ) -> Tuple[BaseGenrePipeline, float]:
        """Select a pipeline for *timeline*, optionally guided by *profile*.

        Returns
        -------
        tuple[BaseGenrePipeline, float]
            The instantiated pipeline and the selection confidence (1.0 when
            the choice was forced by the game profile).
        """

        # 1. Profile override takes absolute priority
        if profile is not None and profile.genre != "general":
            registry_key = _GENRE_PIPELINE_MAP.get(profile.genre)
            if registry_key is not None:
                pipeline_cls = PipelineRegistry.get(registry_key)
                if pipeline_cls is not None:
                    logger.info(
                        "Profile override: genre '%s' → pipeline '%s'",
                        profile.genre,
                        registry_key,
                    )
                    return pipeline_cls(), 1.0

            # Unknown genre in profile — warn and fall through to detection
            logger.warning(
                "Profile genre '%s' has no registered pipeline; "
                "falling back to auto-detection.",
                profile.genre,
            )

        # 2. Automated signal-based detection
        fps_conf, racing_conf = self._compute_confidences(timeline)
        logger.debug(
            "Auto-detection confidences: fps=%.3f  racing=%.3f",
            fps_conf,
            racing_conf,
        )

        if racing_conf > self.threshold and racing_conf > fps_conf:
            pipeline_cls = PipelineRegistry.get("racing")
            if pipeline_cls is not None:
                logger.info(
                    "Auto-detected racing pipeline (confidence=%.2f)",
                    racing_conf,
                )
                return pipeline_cls(), racing_conf

        if fps_conf > self.threshold:
            pipeline_cls = PipelineRegistry.get("fps_action")
            if pipeline_cls is not None:
                logger.info(
                    "Auto-detected FPS pipeline (confidence=%.2f)",
                    fps_conf,
                )
                return pipeline_cls(), fps_conf

        # 3. Fallback — FPS pipeline is the safe default
        fallback_cls = PipelineRegistry.get("fps_action")
        if fallback_cls is None:
            raise RuntimeError(
                "No 'fps_action' pipeline registered — cannot provide a "
                "fallback.  Ensure fps_action_pipeline is importable."
            )
        fallback_conf = max(fps_conf, 0.01)
        logger.info(
            "Using FPS pipeline as safe fallback (confidence=%.2f)",
            fallback_conf,
        )
        return fallback_cls(), fallback_conf

    # ── Signal heuristics ──────────────────────────────────────────────────

    def _compute_confidences(
        self, timeline: List[TimelineFrame]
    ) -> Tuple[float, float]:
        """Derive genre confidences from lightweight signal heuristics.

        Racing indicators
        -----------------
        * Sustained high lateral optical-flow ratio (steering input).
        * Positive ``vehicle_speed_estimate`` across many frames.
        * Elevated ``camera_shake`` without combat scene labels.

        FPS indicators
        --------------
        * ``tracked_objects`` present with ``scene_type`` set to "combat".
        * High ``motion_energy`` spikes (gunfight / ability usage).
        * Low lateral optical-flow ratio (forward-looking camera).

        Returns
        -------
        tuple[float, float]
            ``(fps_confidence, racing_confidence)`` each clamped to 0.0–1.0.
        """
        if not timeline:
            return 0.0, 0.0

        total = len(timeline)

        # ── Accumulate per-frame evidence ──────────────────────────────────
        fps_votes: float = 0.0
        racing_votes: float = 0.0

        high_motion_count: int = 0
        combat_scene_count: int = 0
        lateral_flow_sum: float = 0.0
        speed_positive_count: int = 0
        camera_shake_sum: float = 0.0

        for frame in timeline:
            # FPS evidence
            if frame.scene_type == "combat":
                combat_scene_count += 1
            if frame.tracked_objects:
                for obj in frame.tracked_objects:
                    if obj.get("scene_type") == "combat":
                        fps_votes += 0.3
            if frame.motion_energy > 0.6:
                high_motion_count += 1

            # Racing evidence
            lateral_ratio = frame.optical_flow.get("lateral_ratio", 0.0)
            lateral_flow_sum += lateral_ratio
            if frame.vehicle_speed_estimate > 0.0:
                speed_positive_count += 1
            camera_shake_sum += frame.camera_shake

        # ── Normalise votes ────────────────────────────────────────────────
        # FPS confidence components
        combat_ratio = combat_scene_count / total
        high_motion_ratio = high_motion_count / total
        fps_signal = (
            0.45 * combat_ratio
            + 0.35 * high_motion_ratio
            + 0.20 * min(fps_votes / max(total, 1), 1.0)
        )

        # Racing confidence components
        avg_lateral = lateral_flow_sum / total
        speed_ratio = speed_positive_count / total
        avg_shake = camera_shake_sum / total

        racing_signal = (
            0.35 * min(avg_lateral / 0.5, 1.0)   # lateral flow saturation at 0.5
            + 0.40 * speed_ratio
            + 0.25 * min(avg_shake / 0.4, 1.0)   # shake saturation at 0.4
        )

        # Clamp to [0.0, 1.0]
        fps_conf = max(0.0, min(1.0, fps_signal))
        racing_conf = max(0.0, min(1.0, racing_signal))

        return fps_conf, racing_conf

    # ── Convenience entry-point ────────────────────────────────────────────

    def detect_and_merge(
        self,
        timeline: List[TimelineFrame],
        profile: Optional[GameProfile] = None,
        transcript_data: Optional[List[Dict]] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> List[EventMoment]:
        """Route to the best pipeline and run its detection pass.

        This is a convenience wrapper that combines :meth:`route` and the
        pipeline's ``detect()`` method in a single call.

        Returns
        -------
        list[EventMoment]
            Detected event highlights from the selected pipeline.
        """
        pipeline, confidence = self.route(timeline, profile)
        logger.info(
            "Routed to %s (confidence=%.2f)", pipeline.name, confidence
        )
        return pipeline.detect(
            timeline,
            transcript_data=transcript_data,
            profile=profile,
            video_path=video_path,
            audio_path=audio_path,
        )
