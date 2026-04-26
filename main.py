import argparse
import json
import os
import subprocess
import sys
import time

from audio_extractor import extract_audio
from hardware import build_preflight_lines, default_profile_path, detect_capabilities, plan_hardware
from runtime_env import build_runtime_env


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIBER_SCRIPT = os.path.join(SCRIPT_DIR, "transcriber.py")
LLM_SELECTOR_SCRIPT = os.path.join(SCRIPT_DIR, "llm_selector.py")
CLIPPER_SCRIPT = os.path.join(SCRIPT_DIR, "clipper.py")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="StreamClipMaker Auto-Shorts Pipeline")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--output", default=None, help="Output directory for clips")
    parser.add_argument("--facecam", action="store_true", help="Enable Split-Screen Facecam layout")
    parser.add_argument(
        "--watermark",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable watermark image",
    )
    parser.add_argument("--channel", default="Overwatch-live", help="Channel name for watermark")
    parser.add_argument("--watermark-src", default=None, help="Path to custom watermark image")
    parser.add_argument(
        "--hook-badge",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable/Disable hook score badge",
    )
    parser.add_argument("--facecam_src", default=None, help="Separate facecam recording video file")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Deprecated compatibility flag. Multimodal visual scoring is always enabled.",
    )
    parser.add_argument("--spotlight", action="store_true", help="Enable AI character tracking/reframing")
    parser.add_argument("--hardware-mode", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--transcribe-device", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--visual-device", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--emotion-device", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--encode-device", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--spotlight-device", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--detect-device", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--short", default=True, action=argparse.BooleanOptionalAction, help="Enable short video generation")
    parser.add_argument("--long", default=True, action=argparse.BooleanOptionalAction, help="Enable long video generation")
    parser.add_argument("--short-res", default="1080x1920", help="Resolution for short video (WxH)")
    parser.add_argument("--long-res", default="source", help="Resolution for long video (WxH or 'source')")
    parser.add_argument("--hardware-profile", default=None, help="Path to the shared hardware profile JSON")
    parser.add_argument("--subtitles", default=True, action=argparse.BooleanOptionalAction, help="Burn-in subtitles")
    parser.add_argument("--music", default=True, action=argparse.BooleanOptionalAction, help="Add royalty-free background music")
    parser.add_argument("--hinglish", action="store_true", help="Romanize Hindi transcription")
    
    # New Game-Aware args
    parser.add_argument("--game", default=None, help="Game ID (e.g., overwatch-2)")
    parser.add_argument("--game-profile", default=None, help="Path to custom game profile JSON")
    parser.add_argument("--profile-source", choices=("local", "cloud"), default="local")
    
    return parser


def resolve_output_dir(cli_output):
    if cli_output:
        return cli_output

    print("\nWhere would you like to save the final clips?")
    print("Press Enter to use the default 'output_clips' folder.")
    user_output_dir = input("Folder path: ").strip()
    if not user_output_dir:
        user_output_dir = "output_clips"
    return user_output_dir


def main(
    video_path,
    output_dir="output_clips",
    facecam=False,
    watermark=True,
    channel="Overwatch-live",
    watermark_src=None,
    hook_badge=False,
    facecam_src=None,
    use_vision=False,
    use_spotlight=False,
    hardware_mode="auto",
    transcribe_device=None,
    visual_device=None,
    emotion_device=None,
    encode_device=None,
    spotlight_device=None,
    detect_device=None,
    hardware_profile=None,
    gen_short=True,
    gen_long=True,
    short_res="1080x1920",
    long_res="source",
    subtitles=True,
    music=True,
    hinglish=False,
    # New Game-Aware args
    game=None,
    game_profile=None,
    profile_source="local",
):
    video_path = os.path.abspath(video_path)
    output_dir = os.path.abspath(output_dir)
    if facecam_src:
        facecam_src = os.path.abspath(facecam_src)

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    safe_name = "".join(c for c in video_name if c.isalnum() or c in (" ", "-", "_")).strip()[:60]
    safe_name = safe_name.replace(" ", "_")

    # New Directory-based Cache
    cache_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "cache", safe_name))
    os.makedirs(cache_dir, exist_ok=True)

    audio_path = os.path.join(cache_dir, "audio.wav")
    transcript_path = os.path.join(cache_dir, "transcript.json")
    best_moments_path = os.path.join(cache_dir, "moments.json")
    segments_path = os.path.join(cache_dir, "segments.json")
    events_path = os.path.join(cache_dir, "events.json") # New
    resolved_profile_path = os.path.join(cache_dir, "resolved_profile.json") # New

    print(f"=== Starting Auto-Shorts Pipeline for {video_path} ===", flush=True)
    print(f"    Cache Dir: {cache_dir}", flush=True)
    start_time = time.time()
    runtime_env = build_runtime_env()
    hardware_profile = os.path.abspath(hardware_profile or default_profile_path())

    capabilities = detect_capabilities()
    hardware_plan = plan_hardware(
        hardware_mode=hardware_mode,
        stage_overrides={
            "transcribe": transcribe_device,
            "visual": visual_device,
            "emotion": emotion_device,
            "game_detect": detect_device,
            "encode": encode_device,
            "spotlight": spotlight_device,
        },
        capabilities=capabilities,
        profile_path=hardware_profile,
    )

    print("\n--- HARDWARE PREFLIGHT ---", flush=True)
    for line in build_preflight_lines(hardware_plan):
        print(f"    {line}", flush=True)

    print("\n--- PHASE 1: AUDIO EXTRACTION ---", flush=True)
    if not os.path.exists(audio_path):
        extract_audio(video_path, output_audio_path=audio_path)
    else:
        print("Audio already extracted. Skipping...", flush=True)

    print("\n--- PHASE 2: AI TRANSCRIPTION ---", flush=True)
    if not os.path.exists(transcript_path):
        try:
            transcriber_cmd = [
                sys.executable,
                TRANSCRIBER_SCRIPT,
                audio_path,
                "--output",
                transcript_path,
                "--model",
                "large-v3",
                "--hardware-mode",
                hardware_mode,
                "--hardware-profile",
                hardware_profile,
            ]
            if transcribe_device is not None:
                transcriber_cmd += ["--device", transcribe_device]
            if hinglish:
                transcriber_cmd.append("--hinglish")

            subprocess.run(
                transcriber_cmd,
                check=True,
                cwd=SCRIPT_DIR,
                env=runtime_env,
            )
        except subprocess.CalledProcessError:
            if os.path.exists(transcript_path):
                print(
                    "\n⚠️ Phase 2 had a minor exit-cleanup error, but the transcript was saved successfully. Continuing...",
                    flush=True,
                )
            else:
                print("\n❌ Phase 2 Failed. Skipping rest of pipeline.", flush=True)
                return
    else:
        print("Transcript already generated. Skipping...", flush=True)

    print("\n--- PHASE 3: MULTIMODAL HIGHLIGHT RANKING ---", flush=True)
    if not os.path.exists(best_moments_path):
        try:
            llm_cmd = [
                sys.executable,
                LLM_SELECTOR_SCRIPT,
                transcript_path,
                audio_path,
                "--output",
                best_moments_path,
                "--segments-output",
                segments_path,
                "--video",
                video_path,
                "--hardware-mode",
                hardware_mode,
                "--hardware-profile",
                hardware_profile,
            ]
            if visual_device is not None:
                llm_cmd += ["--visual-device", visual_device]
            if emotion_device is not None:
                llm_cmd += ["--emotion-device", emotion_device]
            if detect_device is not None:
                llm_cmd += ["--detect-device", detect_device]
            if facecam_src:
                llm_cmd += ["--facecam", facecam_src]
            if use_vision:
                llm_cmd.append("--vision")
            
            # Pass Game-Aware args
            if game:
                llm_cmd += ["--game", game]
            if game_profile:
                llm_cmd += ["--game-profile", game_profile]
            if profile_source:
                llm_cmd += ["--profile-source", profile_source]

            subprocess.run(llm_cmd, check=True, cwd=SCRIPT_DIR, env=runtime_env)
        except subprocess.CalledProcessError:
            if os.path.exists(best_moments_path):
                print(
                    "\n⚠️ Phase 3 had a minor exit-cleanup error, but moments were saved successfully. Continuing...",
                    flush=True,
                )
            else:
                print("\n❌ Phase 3 Failed (AI connection or logic error). Skipping rest of pipeline.", flush=True)
                return
    else:
        print("Moments already selected. Skipping...", flush=True)

    if os.path.exists(best_moments_path):
        with open(best_moments_path, "r", encoding="utf-8") as f:
            moments = json.load(f)
        if not moments:
            print("\n⏹  No exciting moments were found in this stream. Try a different video!", flush=True)
            import shutil
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            return
    else:
        print("\n❌ Error: Moments file not found after Phase 3.", flush=True)
        return

    print("\n--- PHASE 4: FINAL VIDEO CLIPPING ---", flush=True)
    try:
        clipper_cmd = [
            sys.executable,
            CLIPPER_SCRIPT,
            video_path,
            best_moments_path,
            "--output",
            output_dir,
            "--transcript",
            transcript_path,
            "--channel",
            channel,
            "--hardware-mode",
            hardware_mode,
            "--hardware-profile",
            hardware_profile,
            "--short" if gen_short else "--no-short",
            "--long" if gen_long else "--no-long",
            "--short-res",
            short_res,
            "--long-res",
            long_res,
            "--subtitles" if subtitles else "--no-subtitles",
            "--music" if music else "--no-music",
        ]
        if encode_device is not None:
            clipper_cmd += ["--encode-device", encode_device]
        if spotlight_device is not None:
            clipper_cmd += ["--spotlight-device", spotlight_device]
        if visual_device is not None:
            clipper_cmd += ["--visual-device", visual_device]
        if facecam:
            clipper_cmd.append("--facecam")
        if watermark:
            clipper_cmd.append("--watermark")
        else:
            clipper_cmd.append("--no-watermark")
        
        if watermark_src:
            clipper_cmd += ["--watermark-src", watermark_src]
            
        if hook_badge:
            clipper_cmd.append("--hook-badge")
        else:
            clipper_cmd.append("--no-hook-badge")

        if facecam_src:
            clipper_cmd += ["--facecam_src", facecam_src]
        if use_spotlight:
            clipper_cmd.append("--spotlight")

        subprocess.run(clipper_cmd, check=True, cwd=SCRIPT_DIR, env=runtime_env)
    except subprocess.CalledProcessError:
        if os.path.exists(output_dir) and any(f.endswith(".mp4") for f in os.listdir(output_dir)):
            print("\n⚠️ Phase 4 finished with some errors, but some clips were successfully created.", flush=True)
        else:
            print("\n❌ Phase 4 Failed completely. No video clips were generated.", flush=True)
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Phase 4 Failed with an unexpected error: {e}", flush=True)
        sys.exit(1)

    mp4_clips = [f for f in os.listdir(output_dir) if f.endswith(".mp4")] if os.path.exists(output_dir) else []
    if not mp4_clips:
        print("\n❌ Pipeline Error: Subtitles were generated, but no MP4 video files were found in the output folder.")
        print("    This usually means FFmpeg failed during rendering. Check the logs above for details.")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n=== Pipeline Complete in {elapsed / 60:.1f} minutes! ===", flush=True)
    print(f"Check the '{output_dir}' directory for your new YouTube Shorts.", flush=True)

    print("\n--- CLEANUP: Removing temporary cache directory ---")
    import shutil
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Deleted: {cache_dir}")
    print("Cleanup complete!")


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    user_output_dir = resolve_output_dir(args.output)
    main(
        args.video,
        user_output_dir,
        facecam=args.facecam,
        watermark=args.watermark,
        channel=args.channel,
        watermark_src=args.watermark_src,
        hook_badge=args.hook_badge,
        facecam_src=args.facecam_src,
        use_vision=args.vision,
        use_spotlight=args.spotlight,
        hardware_mode=args.hardware_mode,
        transcribe_device=args.transcribe_device,
        visual_device=args.visual_device,
        emotion_device=args.emotion_device,
        encode_device=args.encode_device,
        spotlight_device=args.spotlight_device,
        detect_device=args.detect_device,
        hardware_profile=args.hardware_profile,
        gen_short=args.short,
        gen_long=args.long,
        short_res=args.short_res,
        long_res=args.long_res,
        subtitles=args.subtitles,
        music=args.music,
        hinglish=args.hinglish,
        game=args.game,
        game_profile=args.game_profile,
        profile_source=args.profile_source,
    )
