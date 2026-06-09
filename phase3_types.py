from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Arc Shape Types  (NEW — Dynamic Intelligence Layer)
# ─────────────────────────────────────────────────────────────────────────────

class ArcShape(Enum):
    """
    The six universal shapes that describe any compelling gaming moment.
    Detected from the mathematical curve of the composite signal — game-agnostic.
    """
    SPIKE     = "spike"      # Sudden sharp peak — kill, explosion, jump scare
    TENSION   = "tension"    # Slow rise → sustained plateau → sudden drop — chase, stealth, standoff
    COMEDY    = "comedy"     # Long quiet → unexpected multi-signal spike — fail, funny NPC
    DRAMA     = "drama"      # High speech + emotion sustained, low motion — dialogue, cutscene
    TRIUMPH   = "triumph"    # Oscillating struggle → sustained high at end — boss kill, clutch win
    DISCOVERY = "discovery"  # All signals shift character simultaneously — new area, plot reveal

EVENT_PRIORITIES = {
    "surprise": 10,
    "reaction": 9,
    "combat": 7,
    "CRASH": 8,
    "DRIFT": 6,
    "HIGH_SPEED": 6,
    "SPEED_BURST": 5,
    "RACING_MOMENT": 4,
    "travel": 3,
    "neutral": 1
}

@dataclass
class EventMoment:
    event_type: str
    start: float
    end: float
    peak_time: float
    duration: float
    final_score: float
    surprise_score: float
    conflict_score: float
    payoff_score: float
    priority: int
    scene_type: str
    features: Dict[str, Any] = field(default_factory=dict)
    transcript: str = ""
    hook_sentence: str = ""
    short_title: str = ""
    label: str = ""
    clip_prompt: str = ""
    event_confidence: float = 0.0
    # Phase 1 pacing fields — stamped by EventFusionEngine per pacing profile
    pre_context_buffer: float = 2.0   # seconds of setup before the action peak
    post_payoff_buffer: float = 1.5   # seconds of aftermath after the peak region
    peak_prominence: float = 0.0      # peak score minus local average (for slow-mo gate)
    hook_style: str = "contextual"    # set by HookGenerator; used in debug metadata
    # Continuity & Payoff Debug fields
    resolution_score: float = 0.0
    payoff_detected: bool = False
    ending_reason: str = "fixed_buffer"
    ending_extension_used: float = 0.0
    transcript_sentence_incomplete: bool = False

    def to_clipper_json(self) -> Dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "peak_time": round(self.peak_time, 3),
            "event_type": self.event_type,
            "category": self.event_type.capitalize(),
            "score": int(self.final_score * 100),
            "rank_score": round(self.final_score, 4),
            "reason": f"{self.event_type.capitalize()} Moment",
            "text": self.transcript,
            "hook": self.hook_sentence,
            "title": self.short_title,
            "events": [self.event_type],
            # Top-level pacing fields consumed directly by editing_engine
            "pre_context_buffer":  round(self.pre_context_buffer, 2),
            "post_payoff_buffer":  round(self.post_payoff_buffer, 2),
            "peak_prominence":     round(self.peak_prominence, 3),
            "hook_style":          self.hook_style,
            "features":            self.features,
            "evidence": {
                "surprise_score":    round(self.surprise_score, 3),
                "conflict_score":    round(self.conflict_score, 3),
                "payoff_score":      round(self.payoff_score, 3),
                "event_confidence":  round(self.event_confidence, 3),
                "peak_prominence":   round(self.peak_prominence, 3),
                "pre_context_buffer":round(self.pre_context_buffer, 2),
                "post_payoff_buffer":round(self.post_payoff_buffer, 2),
                "resolution_score":  round(self.resolution_score, 3),
                "payoff_detected":   self.payoff_detected,
                "ending_reason":     self.ending_reason,
                "ending_extension_used": round(self.ending_extension_used, 2),
                "transcript_sentence_incomplete": self.transcript_sentence_incomplete
            }
        }


# Clip duration rules per arc shape (min_secs, max_secs)
ARC_DURATION_RULES: Dict[str, tuple] = {
    "combat":     (5,  35),
    "travel":     (20, 90),
    "reaction":   (6,  30),
    "neutral":    (10, 75),
    "surprise":   (8,  45),
}

# Human-readable labels for GUI display
ARC_SHAPE_LABELS: Dict[str, str] = {
    "combat":     "Action Moment",
    "travel":     "Suspense Arc",
    "reaction":   "Comedy Beat",
    "neutral":    "Story Beat",
    "surprise":   "Discovery Moment",
}


@dataclass
class ArcRegion:
    """
    A contiguous region of the timeline where signals are elevated above baseline.
    The shape_type describes the mathematical pattern of that elevation.
    """
    shape_type: ArcShape
    start: float                     # seconds
    end: float                       # seconds
    peak_time: float                 # timestamp of maximum composite signal
    quality_score: float             # 0.0-1.0  overall arc quality
    composite_values: List[float]    # per-second composite signal within the arc
    peak_audio: float = 0.0
    peak_motion: float = 0.0
    peak_emotion: float = 0.0
    peak_speech: float = 0.0
    end_composite: float = 0.0       # composite value at arc end (for tension resolution check)
    transcript: str = ""
    label: str = ""                  # enriched by GameAdapter
    clip_prompt: str = ""            # enriched by GameAdapter (for CLIP visual scoring)
    hook_sentence: str = ""          # generated by NarrativeEngine LLM
    short_title: str = ""            # generated by NarrativeEngine LLM

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_clipper_json(self) -> Dict:
        min_dur, max_dur = ARC_DURATION_RULES.get(self.shape_type.value, (5, 60))
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "peak_time": round(self.peak_time, 3),
            "shape_type": self.shape_type.value,
            "category": ARC_SHAPE_LABELS.get(self.shape_type.value, "Highlight"),
            "score": int(self.quality_score * 100),
            "rank_score": round(self.quality_score, 4),
            "reason": self.label or ARC_SHAPE_LABELS.get(self.shape_type.value, "Arc"),
            "text": self.transcript,
            "hook": self.hook_sentence,
            "title": self.short_title,
            "events": [self.shape_type.value],
            "evidence": {
                "peak_audio": round(self.peak_audio, 3),
                "peak_motion": round(self.peak_motion, 3),
                "peak_emotion": round(self.peak_emotion, 3),
                "peak_speech": round(self.peak_speech, 3),
                "end_composite": round(self.end_composite, 3),
            }
        }


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
    thresholds: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    clip_rules: Dict[str, float] = field(default_factory=dict)
    pacing_style: Optional[str] = None
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
            thresholds=data.get("thresholds", {}),
            weights=data.get("weights", {}),
            clip_rules=data.get("clip_rules", {}),
            pacing_style=data.get("pacing_style"),
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
    rank_score: float = 0.0  # Blended score used for final ranking

    def to_clipper_json(self):
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "peak_time": round(self.anchor_event.timestamp, 3),
            "event_type": self.anchor_event.event_type,
            "category": self.category,
            "score": int(self.score * 100),
            "rank_score": round(self.rank_score, 4),
            "reason": self.reason,
            "text": self.text,
            "game": self.game_id,
            "events": [e.event_type for e in self.events],
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "evidence": self.evidence
        }
