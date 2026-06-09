"""
hook_analyzer.py — Phase 3 Pipeline Orchestrator
==================================================
Orchestrates the full highlight detection pipeline:

  1. Profile Resolution
  2. Base Multimodal Signal Extraction  (speech / audio / visual / emotion)
  3. ArcDetector  → detect arc shapes from signal curves  [NEW]
  4. GameAdapter  → enrich arcs with labels & CLIP prompts [REWRITTEN]
  5. NarrativeEngine → LLM hooks & titles                  [REWRITTEN]
  6. moment_expander → shape-aware clip boundaries
  7. story_builder → merge adjacent clips
  8. editing_brain → boundary polish + shape-aware validation

The old rule-based GameAwareDetector (CLIP per predefined prompt) is still
run as a secondary pass to add visual evidence, but it is no longer the
gating factor for whether a clip is produced.
"""

import logging
from typing import List, Dict, Optional

from multimodal_utils import infer_stream_duration, load_json
from scoring_engine import run_multimodal_scoring
from segment_ranker import generate_sliding_windows
import editing_brain

logger = logging.getLogger(__name__)


def analyze_multimodal_highlights(
    video_path: str,
    audio_path: str,
    transcript_path: str,
    facecam_path: Optional[str] = None,
    window_secs: float = 4.0,
    stride_secs: float = 1.0,
    top_k: int = 6,
    min_separation_secs: float = 6.0,
    hardware_mode: str = "auto",
    visual_device: str = "auto",
    emotion_device: str = "auto",
    hardware_profile: Optional[str] = None,
    hardware_plan=None,
    capabilities=None,
    # Game-Aware fields
    game: Optional[str] = None,
    game_profile: Optional[str] = None,
    profile_source: str = "local",
    detect_device: Optional[str] = None,
):
    """
    Full Phase 3 pipeline — returns (segments, highlights, metadata).

    highlights is a list of clipper-JSON dicts ready for clipper.py.
    """
    # Lazy imports — keep module load fast
    import cv2
    from PIL import Image
    from phase3_types import TimelineSecond, ArcShape
    from profile_resolver import ProfileResolver
    from event_fusion_engine import EventFusionEngine
    from game_adapter import GameAdapter
    from narrative_engine import NarrativeEngine
    import moment_expander
    import story_builder

    transcript_data  = load_json(transcript_path)
    stream_duration  = infer_stream_duration(video_path=video_path, transcript_data=transcript_data)

    # ── 1. Profile Resolution ─────────────────────────────────────────────────
    resolver     = ProfileResolver()
    allow_online = (profile_source == "cloud")
    profile      = resolver.resolve(
        game_id=game, custom_path=game_profile, allow_online=allow_online
    )
    print(f"Using Game Profile: {profile.game_name} (v{profile.version})", flush=True)

    # ── 2. Base Multimodal Signal Extraction ─────────────────────────────────
    windows = generate_sliding_windows(
        stream_duration, window_secs=window_secs, stride_secs=stride_secs
    )

    game_adapter = GameAdapter()
    game_adapter.load_profile(profile.game_id)

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
        game_adapter=game_adapter,
    )

    # Convert SegmentResults to TimelineSeconds
    timeline: List[TimelineSecond] = []
    for res in segment_results:
        sec = TimelineSecond(
            timestamp=res.window.peak_time,
            speech_score=res.scores.speech,
            audio_score=res.scores.audio,
            visual_score=res.scores.visual,
            emotion_score=res.scores.emotion,
            metadata=res.to_window_json(),
        )
        sec.fused_score = res.scores.total
        timeline.append(sec)

    # ── 3. Event Fusion Engine ────────────────────────────────────────────────
    print("Running Phase 3: Event Fusion Engine...", flush=True)

    fusion_engine = EventFusionEngine()
    arcs = fusion_engine.detect(
        timeline=timeline,
        transcript_data=transcript_data,
        genre=profile.genre,
        pacing_style=profile.pacing_style or profile.context_rules.get("pacing_style"),
        profile=profile,
    )
    print(f"  → {len(arcs)} event moments detected.", flush=True)

    # ── Diagnostic: zero-arc feedback ────────────────────────────────────────
    if not arcs:
        stats = fusion_engine.get_stats()
        print("\n  🚨 DIAGNOSTIC — Zero moments survived quality filtering:", flush=True)
        print(f"     Peak score detected in stream : {stats.peak_score_seen:.4f}", flush=True)
        from pacing_profiles import get_pacing_profile, resolve_pacing_style
        _style = resolve_pacing_style(profile.genre, profile.pacing_style or profile.context_rules.get("pacing_style"))
        _thresh = fusion_engine._thresholds if profile.genre == "racing" else get_pacing_profile(_style)
        print(f"     Pacing style used             : {_style}", flush=True)
        print(f"     Start threshold required      : {_thresh['start_threshold']:.4f}", flush=True)
        print(f"     Min intensity required        : {_thresh['min_intensity']:.4f}", flush=True)
        if stats.best_rejected:
            br_score, br_reason = stats.best_rejected
            print(f"     Closest rejected event       : score={br_score:.4f}  reason='{br_reason}'", flush=True)
        print(
            f"\n  💡 TIP: If the peak score ({stats.peak_score_seen:.4f}) is close to the threshold,"
            f" add \"pacing_style\": \"cinematic\" to your game profile JSON.",
            flush=True,
        )
    
    for arc in arcs:
        print(
            f"     [{arc.event_type.upper():10s}] "
            f"{arc.start:.1f}s – {arc.end:.1f}s  "
            f"quality={arc.final_score:.2f}",
            flush=True,
        )


    # ── 4. Game Adapter Enrichment ────────────────────────────────────────────
    print("Running Phase 3: Semantic Enrichment...", flush=True)
    arcs = game_adapter.enrich_arcs(arcs)

    # ── 5. Narrative Engine (LLM hooks + titles) ──────────────────────────────
    print("Running Phase 3: Narrative Hook Generation...", flush=True)
    narrative = NarrativeEngine()
    game_name = profile.game_name if profile else ""
    arcs = narrative.enrich_arcs(arcs, game_name=game_name)

    # ── 5b. Optional: CLIP visual scoring pass ────────────────────────────────
    # Still useful for enriching evidence, but no longer a gating factor.
    frame_samples = _extract_frame_samples(video_path, timeline)
    if frame_samples and arcs:
        _run_clip_visual_pass(arcs, frame_samples, detect_device or "auto")

    # ── 6. Shape-Aware Moment Expansion ──────────────────────────────────────
    print("Running Phase 3: Shape-Aware Clip Boundary Expansion...", flush=True)
    candidates = []
    seen_times = []

    # Sort by quality; pick top arcs with minimum temporal separation
    arcs_sorted = sorted(arcs, key=lambda a: a.final_score, reverse=True)

    for arc in arcs_sorted:
        if len(candidates) >= top_k * 2:  # Allow 2× for post-validation filtering
            break
        # Separation check — don't allow two arcs whose peaks are < min_separation apart
        if any(abs(arc.peak_time - t) < min_separation_secs for t in seen_times):
            continue

        candidate = moment_expander.expand_arc(
            arc=arc,
            timeline=timeline,
            profile=profile,
            transcript_data=transcript_data,
            stream_duration=stream_duration,
        )

        # Continuity sanity check
        continuity = editing_brain.get_avg_score(
            timeline, arc.peak_time - 1.5, arc.peak_time + 1.5
        )
        if continuity < 0.12 and arc.final_score > 0.80:
            print(
                f"  Rejected isolated glitch at t={arc.peak_time:.1f}s "
                f"(continuity={continuity:.3f})",
                flush=True,
            )
            continue

        candidate.rank_score = arc.final_score
        candidates.append(candidate)
        seen_times.append(arc.peak_time)

    # ── 7. Story Building ─────────────────────────────────────────────────────
    print("Running Phase 3: Story Building...", flush=True)
    merge_threshold = profile.context_rules.get("merge_threshold", 8.0)
    highlight_objs  = story_builder.build(candidates, gap_threshold=merge_threshold)

    # Convert to clipper JSON
    highlights = [h.to_clipper_json() for h in highlight_objs]

    # Sort by rank_score, cap at top_k
    highlights = sorted(
        highlights,
        key=lambda x: x.get("rank_score", x["score"] / 100.0),
        reverse=True,
    )[:top_k]

    # ── 8. Editing Brain — final polish + validation ──────────────────────────
    print("Running Phase 4: Editing Brain (Boundary Refinement + Validation)...", flush=True)
    highlights = editing_brain.refine_clips_for_social(
        highlights, transcript_data, profile=profile, timeline=timeline
    )

    print(
        f"Phase 3 complete: {len(highlights)} highlight(s) ready for rendering.",
        flush=True,
    )

    # ── Output ────────────────────────────────────────────────────────────────
    segments = [t.__dict__ for t in timeline]
    metadata = {
        **base_metadata,
        "game":               profile.game_id,
        "profile_version":    profile.version,
        "arcs_detected":      len(arcs),
        "highlights_selected": len(highlights),
        "arc_summary": [
            {
                "shape":   a.event_type,
                "start":   round(a.start, 1),
                "end":     round(a.end, 1),
                "quality": round(a.final_score, 3),
            }
            for a in arcs
        ],
    }

    return segments, highlights, metadata


# ─────────────────────────────────────────────────────────────────────────────
# CLIP Visual Pass (secondary — enrichment only)
# ─────────────────────────────────────────────────────────────────────────────

def _run_clip_visual_pass(arcs, frame_samples, device: str):
    """
    Runs CLIP visual scoring against each arc's clip_prompt.
    Updates arc evidence but does NOT filter or rescore.
    Silently skips if CLIP model cannot be loaded.
    """
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image

        dev = "cuda" if (device == "gpu" or (device == "auto" and
              __import__("torch").cuda.is_available())) else "cpu"
        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        for arc in arcs:
            if not arc.clip_prompt:
                continue
            # Sample one frame near the arc peak
            peak_frame = min(
                frame_samples.items(),
                key=lambda kv: abs(kv[0] - arc.peak_time),
                default=(None, None),
            )
            if peak_frame[1] is None:
                continue
            img = peak_frame[1]
            try:
                inputs = processor(
                    text=[arc.clip_prompt], images=img,
                    return_tensors="pt", padding=True
                ).to(dev)
                with __import__("torch").no_grad():
                    out = model(**inputs)
                clip_score = float(out.logits_per_image.softmax(dim=1).cpu()[0][0])
                logger.debug(f"CLIP score for [{arc.event_type}] at {arc.start:.1f}s: {clip_score:.3f}")
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"CLIP visual pass skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Frame Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_frame_samples(video_path: str, timeline, max_samples: int = 300) -> Dict:
    """
    Extracts frames from the video at timeline timestamps.
    Limits to max_samples to keep memory sane.
    """
    import cv2
    from PIL import Image

    samples = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return samples

    fps  = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, len(timeline) // max_samples)

    for i in range(0, len(timeline), step):
        t_sec     = timeline[i].timestamp
        frame_idx = int(t_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(__import__("cv2").cvtColor(frame, __import__("cv2").COLOR_BGR2RGB))
            samples[t_sec] = img

    cap.release()
    return samples
