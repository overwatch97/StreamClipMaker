from event_fusion_engine import EventFusionEngine
import editing_brain
from moment_expander import expand_arc
from phase3_types import GameProfile, TimelineSecond


def _racing_profile():
    return GameProfile.from_json({
        "game_id": "generic-racing",
        "game_name": "Generic Racing Game",
        "genre": "racing",
        "pacing_style": "racing",
        "priority_events": [],
        "ignore_states": [],
        "context_rules": {"pacing_style": "racing"},
        "score_weights": {},
        "event_rules": [],
        "thresholds": {
            "start_threshold": 0.38,
            "intensity_threshold": 0.38,
            "prominence_threshold": 0.04,
            "surprise_payoff_threshold": 0.20,
        },
        "weights": {
            "motion": 0.55,
            "audio": 0.20,
            "speech": 0.05,
            "momentum": 0.85,
            "momentum_bonus_cap": 0.25,
            "persistence_bonus_cap": 0.15,
        },
        "clip_rules": {
            "min_duration": 20,
            "max_duration": 60,
            "pre_context": 8.0,
            "post_payoff": 10.0,
        },
    })


def _timeline():
    rows = []
    for t in range(80):
        audio = 0.05
        motion = 0.06
        lateral = 0.0
        if 20 <= t <= 45:
            motion = 0.55
            audio = 0.42
        if 30 <= t <= 36:
            lateral = 0.45
        rows.append(TimelineSecond(
            timestamp=float(t),
            metadata={
                "audio_features": {"audio_peak_norm": audio},
                "emotion_features": {"emotion_score_norm": 0.04, "surprise_level": 0.0},
                "visual_features": {
                    "motion_delta_norm": motion,
                    "lateral_flow_norm": lateral,
                    "scene_confidence": 0.1,
                    "scene_type": "neutral",
                },
                "speech_features": {"speech_energy_norm": 0.02, "keyword_weight": 0.0},
            },
        ))
    return rows


def test_racing_survival():
    profile = _racing_profile()
    timeline = _timeline()
    engine = EventFusionEngine()
    events = engine.detect(
        timeline=timeline,
        transcript_data=[],
        genre=profile.genre,
        pacing_style=profile.pacing_style,
        profile=profile,
    )
    assert events, "expected sustained racing motion to survive filtering"
    assert events[0].event_type != "neutral", "expected racing label instead of neutral"

    candidate = expand_arc(events[0], timeline, profile, [], stream_duration=80.0)
    assert candidate.end - candidate.start >= 20.0, "expected racing clip duration floor"
    highlights = editing_brain.refine_clips_for_social(
        [candidate.to_clipper_json()],
        [],
        profile=profile,
        timeline=timeline,
    )
    assert highlights, "expected racing clip to survive EditingBrain validation"
    assert "sequence_candidate" in highlights[0].get("evidence", {}).get("tags", [])


if __name__ == "__main__":
    test_racing_survival()
    print("Racing survival smoke test passed.")
