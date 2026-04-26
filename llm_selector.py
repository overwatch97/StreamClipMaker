import argparse
import json
import os

from hardware import build_preflight_lines, detect_capabilities, plan_hardware
from hook_analyzer import analyze_multimodal_highlights


def _require_existing_path(path, label):
    if not path:
        raise ValueError(f"{label} is required for multimodal highlight scoring.")

    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _default_segments_output(output_path):
    stem, _ext = os.path.splitext(os.path.abspath(output_path))
    return f"{stem}_segments.json"


def _write_json(path, payload):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def find_best_clipping_moments(
    transcript_path,
    audio_path=None,
    video_path=None,
    facecam_path=None,
    use_vision=False,
    output_path="best_moments.json",
    window_secs=4.0,
    stride_secs=1.0,
    top_k=6,
    segments_output=None,
    min_separation_secs=6.0,
    hardware_mode="auto",
    visual_device=None,
    emotion_device=None,
    hardware_profile=None,
    # New Game-Aware Fields
    game=None,
    game_profile=None,
    profile_source="local",
    detect_device=None,
):
    """
    Compatibility entrypoint for StreamClipMaker's selection stage.
    The old Ollama-first flow is retired; this now runs deterministic multimodal scoring.
    """
    transcript_path = _require_existing_path(transcript_path, "Transcript JSON")
    audio_path = _require_existing_path(audio_path, "Audio WAV")
    video_path = _require_existing_path(video_path, "Source video")
    if facecam_path:
        facecam_path = _require_existing_path(facecam_path, "Facecam video")

    output_path = os.path.abspath(output_path)
    segments_output = os.path.abspath(segments_output or _default_segments_output(output_path))

    if use_vision:
        print(
            "Warning: --vision is deprecated and ignored. Multimodal visual scoring is always enabled.",
            flush=True,
        )

    print(f"Loading transcript from {transcript_path}", flush=True)
    print(
        f"Scoring full stream with {window_secs:.1f}s windows at {stride_secs:.1f}s stride...",
        flush=True,
    )

    capabilities = detect_capabilities()
    hardware_plan = plan_hardware(
        hardware_mode=hardware_mode,
        stage_overrides={
            "visual": visual_device, 
            "emotion": emotion_device,
            "game_detect": detect_device
        },
        capabilities=capabilities,
        profile_path=hardware_profile,
        stages=("visual", "emotion", "game_detect"),
    )
    print("Hardware plan for multimodal scoring:", flush=True)
    for line in build_preflight_lines(hardware_plan):
        print(f"  {line}", flush=True)

    segments, highlights, metadata = analyze_multimodal_highlights(
        video_path=video_path,
        audio_path=audio_path,
        transcript_path=transcript_path,
        facecam_path=facecam_path,
        window_secs=window_secs,
        stride_secs=stride_secs,
        top_k=top_k,
        min_separation_secs=min_separation_secs,
        hardware_mode=hardware_mode,
        visual_device=visual_device,
        emotion_device=emotion_device,
        hardware_profile=hardware_profile,
        hardware_plan=hardware_plan,
        capabilities=capabilities,
        # New args
        game=game,
        game_profile=game_profile,
        profile_source=profile_source,
        detect_device=detect_device
    )

    _write_json(segments_output, segments)
    print(f"Wrote per-second score ledger to {segments_output}", flush=True)

    _write_json(output_path, highlights)
    print(f"Wrote {len(highlights)} selected highlights to {output_path}", flush=True)

    print(
        (
            f"Scored {metadata.get('segments_scored', len(segments))} windows and selected "
            f"{metadata.get('highlights_selected', len(highlights))} highlights."
        ),
        flush=True,
    )

    return highlights


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Standalone multimodal highlight selector")
    parser.add_argument("transcript", help="Path to transcript JSON")
    parser.add_argument("audio", help="Path to extracted WAV audio")
    parser.add_argument("--video", help="Path to source gameplay video")
    parser.add_argument("--facecam", help="Path to separate facecam video")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Deprecated compatibility flag. Multimodal visual scoring is always enabled.",
    )
    parser.add_argument("--output", default="best_moments.json", help="Path to save highlight JSON")
    parser.add_argument(
        "--segments-output",
        default=None,
        help="Path to save the full per-window segment ledger JSON",
    )
    parser.add_argument("--window-secs", type=float, default=4.0, help="Sliding window length in seconds")
    parser.add_argument("--stride-secs", type=float, default=1.0, help="Sliding window stride in seconds")
    parser.add_argument("--top-k", type=int, default=6, help="Number of highlight peaks to keep before merge")
    parser.add_argument("--hardware-mode", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--visual-device", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--emotion-device", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--detect-device", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--hardware-profile", default=None, help="Path to the shared hardware profile JSON")
    
    # New Game-Aware args
    parser.add_argument("--game", default=None, help="Game ID (e.g., overwatch-2)")
    parser.add_argument("--game-profile", default=None, help="Path to custom game profile JSON")
    parser.add_argument("--profile-source", choices=("local", "cloud"), default="local")
    
    return parser


if __name__ == "__main__":
    import sys
    import traceback

    args = build_arg_parser().parse_args()

    try:
        find_best_clipping_moments(
            args.transcript,
            args.audio,
            video_path=args.video,
            facecam_path=args.facecam,
            use_vision=args.vision,
            output_path=args.output,
            window_secs=args.window_secs,
            stride_secs=args.stride_secs,
            top_k=args.top_k,
            segments_output=args.segments_output,
            hardware_mode=args.hardware_mode,
            visual_device=args.visual_device,
            emotion_device=args.emotion_device,
            hardware_profile=args.hardware_profile,
            # New passed-through args
            game=args.game,
            game_profile=args.game_profile,
            profile_source=args.profile_source,
            detect_device=args.detect_device
        )
        raise SystemExit(0)
    except Exception:
        print(f"\n[ERROR] Selection Error:\n{traceback.format_exc()}")
        raise SystemExit(1)
