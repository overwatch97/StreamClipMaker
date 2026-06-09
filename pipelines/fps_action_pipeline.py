from typing import List, Dict, Optional
from phase3_types import EventMoment, TimelineSecond, GameProfile
from event_fusion_engine import EventFusionEngine
from core.cache.timeline_frame import TimelineFrame
from pipelines.base_pipeline import BaseGenrePipeline
from pipelines.registry import PipelineRegistry

class FPSActionPipeline(BaseGenrePipeline):
    """
    FPS / Action Action highlight pipeline.
    Encapsulates and freezes the legacy EventFusionEngine logic to ensure no regression.
    """
    @property
    def name(self) -> str:
        return "FPSActionPipeline"

    def detect(
        self,
        timeline: List[TimelineFrame],
        transcript_data: Optional[List[Dict]] = None,
        profile: Optional[GameProfile] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> List[EventMoment]:
        # 1. Translate TimelineFrames to legacy TimelineSeconds
        legacy_timeline: List[TimelineSecond] = []
        for f in timeline:
            sec = TimelineSecond(
                timestamp=f.timestamp,
                speech_score=f.speech_emotion,
                audio_score=f.audio_energy,
                visual_score=f.motion_energy,
                emotion_score=f.facecam_motion,
                metadata={
                    "audio_features": {"audio_peak_norm": f.audio_energy},
                    "emotion_features": {
                        "emotion_score_norm": f.facecam_motion,
                        "surprise_level": f.metadata.get("surprise_level", 0.0),
                        "engagement_level": f.facecam_motion,
                        "reaction_level": f.metadata.get("reaction_level", 0.0)
                    },
                    "visual_features": {
                        "motion_delta_norm": f.motion_energy,
                        "scene_confidence": f.metadata.get("scene_confidence", 0.0),
                        "scene_type": f.scene_type
                    },
                    "speech_features": {
                        "speech_energy_norm": f.speech_emotion,
                        "keyword_weight": 0.2
                    }
                }
            )
            sec.fused_score = f.motion_energy  # Fallback for baseline checks
            legacy_timeline.append(sec)

        # 2. Invoke frozen EventFusionEngine logic
        engine = EventFusionEngine()
        genre = profile.genre if profile else "fps"
        pacing_style = profile.context_rules.get("pacing_style") if profile and profile.context_rules else None
        
        return engine.detect(
            timeline=legacy_timeline,
            transcript_data=transcript_data,
            genre=genre,
            pacing_style=pacing_style,
        )

# Register to the plugin system
PipelineRegistry.register("fps", FPSActionPipeline)
PipelineRegistry.register("fps_action", FPSActionPipeline)
PipelineRegistry.register("battle-royale", FPSActionPipeline)
PipelineRegistry.register("tactical-shooter", FPSActionPipeline)
