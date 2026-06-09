from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from phase3_types import EventMoment
from core.cache.timeline_frame import TimelineFrame
from phase3_types import GameProfile

class BaseGenrePipeline(ABC):
    """
    Abstract Base Class for all genre-specific clipping pipelines.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def detect(
        self,
        timeline: List[TimelineFrame],
        transcript_data: Optional[List[Dict]] = None,
        profile: Optional[GameProfile] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> List[EventMoment]:
        """
        Executes event detection and scoring logic on the timeline.
        Returns:
            List[EventMoment]: A list of detected event highlights.
        """
        pass
