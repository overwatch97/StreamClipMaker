import logging

from multimodal_utils import infer_stream_duration, load_json
from scoring_engine import run_multimodal_scoring
from segment_ranker import generate_sliding_windows
import editing_brain

logger = logging.getLogger(__name__)

def analyze_multimodal_highlights(
    video_path,
    audio_path,
    transcript_path,
    facecam_path=None,
    window_secs=4.0,
    stride_secs=1.0,
    top_k=6,
    min_separation_secs=6.0,
    hardware_mode="auto",
    visual_device="auto",
    emotion_device="auto",
    hardware_profile=None,
    hardware_plan=None,
    capabilities=None,
    # New Game-Aware Fields
    game=None,
    game_profile=None,
    profile_source="local",
    detect_device=None,
):
    """
    Game-Aware Phase 3 pipeline.
    Flow: Base Signals -> Profile Resolution -> Game Detection -> Moment Expansion -> Story Building.
    """
    # Lazy imports — deferred so module load stays fast
    import cv2
    from PIL import Image
    from phase3_types import TimelineSecond
    from profile_resolver import ProfileResolver
    from game_aware_detector import GameAwareDetector
    import moment_expander
    import story_builder

    transcript_data = load_json(transcript_path)
    stream_duration = infer_stream_duration(video_path=video_path, transcript_data=transcript_data)

    # 1. Profile Resolution
    resolver = ProfileResolver()
    allow_online = (profile_source == "cloud")
    profile = resolver.resolve(game_id=game, custom_path=game_profile, allow_online=allow_online)
    print(f"Using Game Profile: {profile.game_name} (v{profile.version})", flush=True)

    # 2. Base Multimodal Signal Extraction
    # We still use the sliding windows for base signals (speech, audio, etc.)
    windows = generate_sliding_windows(stream_duration, window_secs=window_secs, stride_secs=stride_secs)
    segment_results, base_metadata = run_multimodal_scoring(
        video_path=video_path,
        audio_path=audio_path,
        transcript_data=transcript_data,
        windows=windows,
        facecam_path=facecam_path,
        hardware_mode=hardware_mode,
        visual_device=visual_device,
        emotion_device=emotion_device,
        hardware_profile=hardware_profile,
        hardware_plan=hardware_plan,
        capabilities=capabilities,
    )

    # Convert SegmentResults to TimelineSeconds (per-second ledger)
    timeline = []
    for res in segment_results:
        timeline.append(TimelineSecond(
            timestamp=res.window.peak_time,
            speech_score=res.scores.speech,
            audio_score=res.scores.audio,
            visual_score=res.scores.visual,
            emotion_score=res.scores.emotion,
            metadata=res.to_window_json()
        ))

    # 3. Game-Aware Event Detection
    print(f"Running Phase 3: Game-Aware Event Detection for {profile.game_id}...", flush=True)
    
    # Extract frames for CLIP processing (at 1s intervals)
    frame_samples = _extract_frame_samples(video_path, timeline)
    
    detector = GameAwareDetector(device=detect_device or "auto")
    events = detector.detect(timeline, profile, frame_samples)
    print(f"Detected {len(events)} game events.", flush=True)

    # 4. Moment Expansion
    print("Running Phase 3: Moment-Centric Expansion...", flush=True)
    candidates = []
    # Filter for priority events based on profile
    priority_events = [e for e in events if e.event_type in profile.priority_events]
    # If no priority events, use all events as candidates
    search_set = priority_events if priority_events else events
    
    # Filter out redundant events (too close together)
    unique_anchors = []
    if search_set:
        search_set.sort(key=lambda x: x.score, reverse=True)
        for evt in search_set:
            if not any(abs(evt.timestamp - u.timestamp) < min_separation_secs for u in unique_anchors):
                unique_anchors.append(evt)
    
    for anchor in unique_anchors:
        candidate = moment_expander.expand(anchor, timeline, profile, transcript_data)
        candidates.append(candidate)

    # 5. Story Building (Merging windows)
    print("Running Phase 3: Story Building (Merging windows)...", flush=True)
    merge_threshold = profile.context_rules.get("merge_threshold", 8.0)
    highlights_objs = story_builder.build(candidates, gap_threshold=merge_threshold)
    
    # Convert back to legacy-compatible JSON list
    highlights = [h.to_clipper_json() for h in highlights_objs]
    
    # Limit to top_k and sort by score
    highlights = sorted(highlights, key=lambda x: x["score"], reverse=True)[:top_k]

    # --- Phase 4: Editing Brain ---
    print("Running Phase 4: Editing Brain (Boundary Refinement)...", flush=True)
    highlights = editing_brain.refine_clips_for_social(highlights, transcript_data)

    # Prepare outputs
    segments = [t.__dict__ for t in timeline] # Richer per-second ledger

    metadata = {
        **base_metadata,
        "game": profile.game_id,
        "profile_version": profile.version,
        "events_detected": len(events),
        "highlights_selected": len(highlights),
    }

    return segments, highlights, metadata

def _extract_frame_samples(video_path, timeline, max_samples=300):
    """
    Extracts frames from the video at timestamps in the timeline.
    Limits to 300 samples to keep memory/processing sane.
    """
    import cv2
    from PIL import Image

    samples = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return samples

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sample at most 1 frame per 2 seconds if video is very long
    step = 1
    if len(timeline) > max_samples:
        step = len(timeline) // max_samples

    for i in range(0, len(timeline), step):
        t_sec = timeline[i].timestamp
        frame_idx = int(t_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert to PIL Image for CLIP
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            samples[t_sec] = img
    
    cap.release()
    return samples
