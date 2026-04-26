from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class GameProfile:
    game_id: str
    game_name: str
    genre: str
    priority_events: List[str]
    ignore_states: List[str]
    context_rules: Dict[str, float]
    score_weights: Dict[str, float]
    event_rules: List[Dict[str, Any]]
    spatial_rules: Dict[str, Any] = field(default_factory=dict)
    audio_weights: Optional[Dict[str, float]] = None
    transitions: Optional[Dict[str, Any]] = None
    version: str = "1.0.0"

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return cls(
            game_id=data.get("game_id", "generic"),
            game_name=data.get("game_name", "Generic Game"),
            genre=data.get("genre", "general"),
            priority_events=data.get("priority_events", []),
            ignore_states=data.get("ignore_states", []),
            context_rules=data.get("context_rules", {}),
            score_weights=data.get("score_weights", {}),
            event_rules=data.get("event_rules", []),
            spatial_rules=data.get("spatial_rules", {}),
            audio_weights=data.get("audio_weights"),
            transitions=data.get("transitions"),
            version=data.get("version", "1.0.0")
        )

@dataclass
class TimelineSecond:
    timestamp: float
    speech_score: float = 0.0
    audio_score: float = 0.0
    visual_score: float = 0.0
    emotion_score: float = 0.0
    detect_scores: Dict[str, float] = field(default_factory=dict)
    is_ignore_state: bool = False
    ignore_reason: Optional[str] = None
    fused_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectedEvent:
    event_type: str
    timestamp: float
    score: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    duration: float = 1.0

@dataclass
class HighlightCandidate:
    start: float
    end: float
    anchor_event: DetectedEvent
    score: float
    category: str
    reason: str
    game_id: str
    events: List[DetectedEvent] = field(default_factory=dict)
    profile_id: str = ""
    profile_version: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def to_clipper_json(self):
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "peak_time": round(self.anchor_event.timestamp, 3),
            "category": self.category,
            "score": int(self.score * 100),
            "reason": self.reason,
            "text": self.text,
            "game": self.game_id,
            "events": [e.event_type for e in self.events],
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "evidence": self.evidence
        }
