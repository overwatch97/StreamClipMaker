import argparse
import json
import os
import subprocess
import time

import cv2
import numpy as np
from ultralytics import YOLO

from hardware import build_preflight_lines, detect_capabilities, plan_hardware, record_stage_metric
from subtitler import generate_ass_subtitle
from multimodal_utils import ProgressReporter

import narrative_engine
import editing_brain
import audio_director


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHANNEL_NAME = "Overwatch-live"
WATERMARK_IMAGE = os.path.join(SCRIPT_DIR, "watermark.png")
DEFAULT_PRO_SETTINGS = {
    "use_facecam": False,
    "use_watermark": True,
    "channel_name": DEFAULT_CHANNEL_NAME,
    "facecam_corner": "bottom-right",
}

_yolo_model = None


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Standalone Video Clipper")
    parser.add_argument("video", help="Path to source video")
    parser.add_argument("moments", help="Path to best_moments.json")
    parser.add_argument("--output", default="output_clips", help="Directory to save clips")
    parser.add_argument("--transcript", default="transcript.json", help="Path to transcript.json")
    parser.add_argument("--facecam", action="store_true", help="Enable Split-Screen Facecam layout")
    parser.add_argument(
        "--watermark",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable watermark overlay",
    )
    parser.add_argument("--watermark-src", default=None, help="Path to custom watermark image file")
    parser.add_argument(
        "--hook-badge",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable Hook Score badge overlay",
    )
    parser.add_argument("--channel", default=DEFAULT_CHANNEL_NAME, help="Channel name for watermark")
    parser.add_argument("--facecam_src", default=None, help="Separate facecam recording video file")
    parser.add_argument("--spotlight", action="store_true", help="Enable AI character tracking/reframing")
    parser.add_argument("--hardware-mode", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--encode-device", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--spotlight-device", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--visual-device", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--hardware-profile", default=None, help="Path to the shared hardware profile JSON")
    parser.add_argument("--short", default=True, action=argparse.BooleanOptionalAction, help="Enable short video generation")
    parser.add_argument("--long", default=True, action=argparse.BooleanOptionalAction, help="Enable long video generation")
    parser.add_argument("--short-res", default="1080x1920", help="Resolution for short video (WxH)")
    parser.add_argument("--long-res", default="source", help="Resolution for long video (WxH or 'source')")
    parser.add_argument("--subtitles", default=True, action=argparse.BooleanOptionalAction, help="Show/Hide subtitles")
    parser.add_argument("--music", default=True, action=argparse.BooleanOptionalAction, help="Add background music")
    return parser


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolo11n.pt")
    return _yolo_model


def format_time_hhmmss(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def escape_ffmpeg_path(path):
    """
    Robustly escape Windows paths for FFmpeg filters.
    See: https://ffmpeg.org/ffmpeg-filters.html#Notes-on-filtergraph-escaping
    """
    p = os.path.abspath(path).replace("\\", "/")
    # Colons in Windows paths (C:/...) must be escaped as C\\:/
    p = p.replace(":", "\\:")
    # Special characters like # and , can break filter parsing
    p = p.replace("#", "\\#")
    p = p.replace(",", "\\,")
    p = p.replace("[", "\\[")
    p = p.replace("]", "\\]")
    p = p.replace("'", "'\\\\''")
    return p


def escape_ffmpeg_text(text):
    escaped = str(text).replace("\\", r"\\")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    escaped = escaped.replace("%", r"\%")
    return escaped


def format_time_simple(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}{m:02d}{s:02d}"


def build_branding_filter(filter_script, v_final_label, safe_font_path, channel_name, use_watermark, logo_label=None, apply_grading=True):
    channel_name = (channel_name or DEFAULT_CHANNEL_NAME).strip() or DEFAULT_CHANNEL_NAME
    safe_channel_name = escape_ffmpeg_text(channel_name)

    # Cinematic Auto-Color-Grade (Subtle contrast and saturation boost)
    grading = "eq=contrast=1.05:brightness=0.01:saturation=1.1, curves=preset=lighter" if apply_grading else "copy"
    
    if use_watermark and logo_label:
        return (
            f"{filter_script}; "
            f"{v_final_label}{grading}[graded]; "
            f"{logo_label}scale=250:-1[logo_scaled]; "
            f"[graded][logo_scaled]overlay=W-w-50:H-h-185[brand]; "
            f"[brand]drawtext=text='{safe_channel_name}':fontcolor=white:fontsize=28:"
            f"x=W-tw-50:y=H-275:box=1:boxcolor=black@0.5:boxborderw=12:fontfile='{safe_font_path}',"
            f"format=yuv420p[v]"
        )

    if use_watermark:
        return (
            f"{filter_script}; "
            f"{v_final_label}{grading}[graded]; "
            f"[graded]drawtext=text='{safe_channel_name}':fontcolor=white:fontsize=32:"
            f"x=W-tw-50:y=H-180:box=1:boxcolor=black@0.5:boxborderw=12:fontfile='{safe_font_path}',"
            f"format=yuv420p[v]"
        )

    return f"{filter_script}; {v_final_label}{grading}[v]"


def cleanup_temp_files(paths):
    for temp_path in paths:
        if not temp_path:
            continue
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError as exc:
            print(f"Warning: failed to remove temporary file {temp_path}: {exc}")


def create_extended_highlight(video_path, output_dir, long_filename, ext_start, ext_duration, *, long_res="source", use_watermark=True, watermark_src=None, channel_name=None, use_hook_badge=False):
    video_path = os.path.abspath(video_path)
    long_filename = os.path.basename(long_filename)
    
    cmd = [
        "ffmpeg",
        "-ss",
        str(ext_start),
        "-i",
        video_path,
        "-t",
        str(ext_duration),
    ]

    logo_path = watermark_src if watermark_src and os.path.exists(watermark_src) else WATERMARK_IMAGE
    has_logo = use_watermark and os.path.exists(logo_path)
    if has_logo:
        cmd += ["-i", logo_path]

    # Build filter complex for long version
    filter_parts = []
    v_label = "[0:v]"

    # Apply resolution if specified
    if long_res and long_res.lower() != "source":
        try:
            w, h = long_res.lower().split("x")
            filter_parts.append(f"{v_label}scale={w}:{h}[res]")
            v_label = "[res]"
        except Exception:
            print(f"Warning: invalid long resolution format '{long_res}', ignoring.")

    # Branding
    safe_font_path = "C\\:/Windows/Fonts/arial.ttf"
    logo_input_label = "[1:v]" if has_logo else None
    
    branding_script = build_branding_filter(
        "" if not filter_parts else filter_parts[0],
        v_label if not filter_parts else "[res]",
        safe_font_path,
        channel_name,
        use_watermark,
        logo_label=logo_input_label
    )
    
    # build_branding_filter returns a script like "filter; [v]..."
    # If filter_parts was empty, it might start with "; ". We cleanup.
    if branding_script.startswith("; "):
        branding_script = branding_script[2:]

    cmd += ["-filter_complex", branding_script, "-map", "[v]", "-map", "0:a?"]

    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "12",
        "-c:a",
        "aac",
        "-y",
        long_filename,
    ]

    attempts = [
        (
            "Clean High-Quality Encode",
            cmd
        ),
        (
            "High Quality Encode (CRF 14 Fallback)",
            # Simple rebuild for fallback
            cmd[:cmd.index("-crf")+1] + ["14"] + cmd[cmd.index("-crf")+2:]
        ),
    ]

    for index, (label, cmd) in enumerate(attempts):
        try:
            print(f"> Running FFmpeg (Long Highlight): {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=output_dir, env=_ffmpeg_env())
            print(f"Successfully created {long_filename} ({label})")
            return True
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.decode("utf-8", errors="replace")
            if index == 0:
                print(f"Long version primary encode failed for {long_filename}, trying fallback...")
            else:
                print(f"Warning: failed to create long version for {long_filename}. Skipping.")
                print(f"--- Long Version Error (last 800 chars) ---\n...{err[-800:]}")
    return False


def _ffmpeg_env():
    env = os.environ.copy()
    env["FONTCONFIG_FILE"] = "nul"
    env["FONTCONFIG_PATH"] = "nul"
    return env


def _run_detector(model, frame, device):
    kwargs = {"verbose": False, "classes": [0]}
    if device:
        kwargs["device"] = "cuda:0" if device == "gpu" else "cpu"
    try:
        return model(frame, **kwargs)[0]
    except TypeError:
        kwargs.pop("device", None)
        return model(frame, **kwargs)[0]


def _record_profile_event(stage, device, duration_secs, elapsed_seconds, hardware_profile, *, success=True, error=None):
    record_stage_metric(
        stage,
        device,
        max(float(duration_secs or 0.0) / 60.0, 1e-6),
        elapsed_seconds,
        success=success,
        error=error,
        profile_path=hardware_profile,
    )


def analyze_spotlight_path(video_path, start_time, duration, *, device="cpu", strict_device=False, target_w=1080, target_h=1920):
    model = get_yolo_model()
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)

    if fps <= 0.0 or width <= 0.0 or height <= 0.0:
        cap.release()
        return [], device, "invalid video geometry"

    scale = target_h / height
    canvas_w = target_w
    scaled_w = width * scale
    default_x = (canvas_w - scaled_w) / 2

    path = []
    current_x = default_x
    sample_rate = max(1, int(fps / 5))
    actual_device = str(device)
    fallback_error = None

    reporter = ProgressReporter(len(range(0, int(duration * fps), sample_rate)), label="Spotlight")

    for frame_idx in range(0, int(duration * fps), sample_rate):
        reporter.update()
        reporter.report()
        ts = start_time + (frame_idx / fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = _run_detector(model, frame, actual_device)
        except Exception as exc:
            if actual_device == "gpu" and not strict_device:
                fallback_error = str(exc)
                actual_device = "cpu"
                results = _run_detector(model, frame, actual_device)
            else:
                cap.release()
                raise

        target_x = default_x
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            center_dist = np.abs((boxes[:, 0] + boxes[:, 2]) / 2 - width / 2)
            best_idx = np.argmin(center_dist)

            box = boxes[best_idx]
            box_center_x = (box[0] + box[2]) / 2
            target_x = (target_w / 2) - (box_center_x * scale)
            target_x = min(0, max(target_x, canvas_w - scaled_w))

        alpha = 0.15
        current_x = current_x * (1 - alpha) + target_x * alpha
        path.append((ts, current_x))

    cap.release()
    return path, actual_device, fallback_error


def create_clips(
    video_path,
    best_moments_path="best_moments.json",
    output_dir="output_clips",
    transcript_path="transcript.json",
    pro_settings=None,
    facecam_video_path=None,
    use_spotlight=False,
    *,
    hardware_mode="auto",
    encode_device=None,
    spotlight_device=None,
    visual_device=None,
    hardware_profile=None,
    gen_short=True,
    gen_long=True,
    short_res="1080x1920",
    long_res="source",
    watermark_src=None,
    use_hook_badge=True,
    use_subtitles=True,
    use_music=True,
):
    pro_settings = {**DEFAULT_PRO_SETTINGS, **(pro_settings or {})}

    video_path = os.path.abspath(video_path)
    best_moments_path = os.path.abspath(best_moments_path)
    transcript_path = os.path.abspath(transcript_path)
    output_dir = os.path.abspath(output_dir)
    if facecam_video_path:
        facecam_video_path = os.path.abspath(facecam_video_path)

    os.makedirs(output_dir, exist_ok=True)

    capabilities = detect_capabilities()
    hardware_plan = plan_hardware(
        hardware_mode=hardware_mode,
        stage_overrides={
            "visual": visual_device,
            "encode": encode_device,
            "spotlight": spotlight_device,
        },
        capabilities=capabilities,
        profile_path=hardware_profile,
        stages=("visual", "encode", "spotlight"),
    )

    print("Hardware plan for clipping:", flush=True)
    for line in build_preflight_lines(hardware_plan):
        print(f"  {line}", flush=True)

    encode_policy = hardware_plan.stage_device("encode")
    encode_strict = hardware_plan.stage_strict("encode")
    spotlight_policy = hardware_plan.stage_device("spotlight")
    spotlight_strict = hardware_plan.stage_strict("spotlight")

    print(f"Loading moments from {best_moments_path}...")
    with open(best_moments_path, "r", encoding="utf-8") as handle:
        moments = json.load(handle)

    if not moments:
        print("No moments to clip!")
        return

    print(f"Creating {len(moments)} clips...")
    
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    safe_name = "".join(c for c in video_filename if c.isalnum() or c in (" ", "-", "_")).strip()[:60]
    safe_name = safe_name.replace(" ", "_")

    use_watermark = pro_settings.get("use_watermark", True)
    channel_name = pro_settings.get("channel_name", DEFAULT_CHANNEL_NAME)
    logo_path = watermark_src if watermark_src and os.path.exists(watermark_src) else WATERMARK_IMAGE
    
    if use_watermark and not os.path.exists(logo_path):
        print(f"Warning: watermark image not found at {logo_path}. Falling back to text-only channel branding.")

    try:
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        total_video_duration = float(subprocess.check_output(probe_cmd).decode().strip())
    except Exception as exc:
        print(f"Warning: could not detect total video duration. Bounds checking may be limited. ({exc})")
        total_video_duration = 99999999.0

    # Defensively check if facecam file has an audio track to prevent FFmpeg dropping into Safe Mode
    facecam_has_audio = False
    if facecam_video_path and os.path.exists(facecam_video_path):
        try:
            probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", facecam_video_path]
            output = subprocess.check_output(probe_cmd).decode().strip()
            facecam_has_audio = "audio" in output
        except Exception:
            pass


    clips_created = 0
    reporter = ProgressReporter(len(moments), label="Clipping")

    for i, moment in enumerate(moments):
        reporter.update()
        reporter.report(force=True)
        clip_temp_files = []
        try:
            start_time = float(moment["start"])
            end_time = float(moment["end"])
            duration = end_time - start_time

            if duration < 10:
                diff = 10 - duration
                # Enforce Hook-First logic: Do NOT pad backward and destroy the hook point.
                # Pad entirely forward into the resolution to hit the minimum duration.
                end_time = end_time + diff
                duration = 10

            # Safety Hard Cap: Truncate clips that somehow exceeded reasonable social lengths
            # This prevents 1.5-hour rogue clips from timing out the pipeline.
            MAX_CLIP_DURATION = 120.0
            if duration > MAX_CLIP_DURATION:
                print(f"Warning: Clip {i+1} duration ({duration:.1f}s) exceeds safety limit. Truncating to {MAX_CLIP_DURATION}s.")
                duration = MAX_CLIP_DURATION
                end_time = start_time + duration

            category = moment.get("category", "good_scene")
            safe_category = "".join(c for c in category if c.isalnum() or c in (" ", "-", "_")).strip()[:30]
            safe_category = safe_category.replace(" ", "_")
            score = moment.get("score", 0)

            f_start = format_time_simple(start_time)
            f_end = format_time_simple(end_time)

            output_file = os.path.join(output_dir, f"clip_{i+1:03d}_{score}V_{safe_category}_{f_start}-{f_end}.mp4")
            ass_file = os.path.join(output_dir, f"clip_{i+1:03d}_subs.ass")

            generate_ass_subtitle(transcript_path, start_time, end_time, ass_file)

            # --- Phase 4: Editing Brain Integration ---
            # Apply 'Natural Clips' and 'Hook Placement'
            # We assume moments might have been pre-processed by narrative_engine
            # but we double check here.
            
            print(f"Clipping segment {i+1}: Start {start_time:.2f}s, Duration {duration:.2f}s -> {output_file}")
            
            # --- Phase 5: Audio Director integration ---
            audio_path = os.path.abspath(f"cache_{safe_name}_audio.wav")
            music_manifest = audio_director.get_royalty_free_music_manifest(SCRIPT_DIR)
            selected_music = music_manifest[0]["path"] if music_manifest else None
            
            safe_ass_path = escape_ffmpeg_path(ass_file)
            output_filename = os.path.basename(output_file)

            if score >= 90:
                badge_color = "0x00CC44"
            elif score >= 65:
                badge_color = "0xF39C12"
            else:
                badge_color = "0xE74C3C"

            badge_text = escape_ffmpeg_text(f"HOOK  {score}/100")

            # Dynamically resolve Windows font path for portability
            windir = os.environ.get("WINDIR", "C:/Windows")
            clean_windir = windir.replace("\\", "/")
            safe_font_path = f"{clean_windir}/Fonts/arial.ttf".replace(":", "\\:")
            safe_badge_color = badge_color.replace("0x", "#")

            # Parse resolution
            try:
                target_w, target_h = map(int, short_res.lower().split("x"))
            except Exception:
                print(f"Warning: invalid short resolution format '{short_res}', using 1080x1920.")
                target_w, target_h = 1080, 1920

            if pro_settings.get("use_facecam", False):
                import facecam_director
                fc_layout = facecam_director.get_facecam_layout(score)
                cam_w = int(target_w * fc_layout["scale"])
                cam_h = cam_w  # Tight 1:1 Square Crop
                r = 25 # Corner radius for rounded corners
                
                cam_src = "[1:v]" if facecam_video_path else "[0:v]"
                
                # Build highly styled professional layout (Crop to 1:1 -> BoxBorder -> RoundedCorners -> DropShadow)
                if not facecam_video_path:
                    # If derived from combined source, crop inner quadrant first
                    cam_pipe = f"{cam_src}crop=iw/4:ih/3:iw*0.75:ih*0.66,crop='min(iw,ih):min(iw,ih):(iw-min(iw,ih))/2:(ih-min(iw,ih))/2',scale={cam_w}:{cam_h},format=rgba[cam_sq]; "
                else:
                    cam_pipe = f"{cam_src}crop='min(iw,ih):min(iw,ih):(iw-min(iw,ih))/2:(ih-min(iw,ih))/2',scale={cam_w}:{cam_h},format=rgba[cam_sq]; "
                
                # Apply 3px white inner border
                cam_pipe += f"[cam_sq]drawbox=x=0:y=0:w=iw:h=ih:color=white@0.65:t=3[cam_brd]; "
                
                # Apply Rounded Corner Alpha Mask via GEQ
                cam_pipe += f"[cam_brd]geq=lum='p(X,Y)':a='if(gt(abs(W/2-X),W/2-{r})*gt(abs(H/2-Y),H/2-{r}),if(gt(hypot(abs(W/2-X)-(W/2-{r}),abs(H/2-Y)-(H/2-{r})),{r}),0,alpha(X,Y)),alpha(X,Y))'[cam_rnd]; "
                
                # Apply Soft Drop Shadow and Overlay Back Together
                cam_pipe += f"[cam_rnd]split=2[cfg][cbg]; [cbg]colorchannelmixer=rr=0:gg=0:bb=0:aa=0.7,boxblur=15:10,pad=iw+30:ih+30:15:15:color=black@0[shadow]; [shadow][cfg]overlay=15:15[cam]; "


                if use_subtitles:
                    sub_part = f"[base]subtitles='{safe_ass_path}':force_style='Alignment=2,MarginV=100'[sub]; "
                    v_label_after_sub = "[sub]"
                else:
                    sub_part = ""
                    v_label_after_sub = "[base]"

                badge_part = ""
                if use_hook_badge:
                    badge_part = (
                        f"{v_label_after_sub}drawtext=text='{badge_text}':fontcolor=white:fontsize=42:x=50:y=60:"
                        f"box=1:boxcolor={safe_badge_color}:boxborderw=18:fontfile='{safe_font_path}'[badge]; "
                    )
                    v_label_after_badge = "[badge]"
                else:
                    v_label_after_badge = v_label_after_sub

                filter_script = (
                    f"{cam_pipe}"
                    f"[0:v]crop='trunc(ih*{target_w}/{target_h}/2)*2':ih,scale={target_w}:{target_h}[game]; "
                    f"[game][cam]{fc_layout['ffmpeg_overlay']}[base]; "
                    f"{sub_part}"
                    f"{badge_part}"
                    f"color=c=yellow:s={target_w}x14[pb]; {v_label_after_badge}[pb]overlay=x=-W+(t/{duration})*W:y=H-14[pb_over]"
                )
                v_final_label = "[pb_over]"
            elif use_spotlight:
                print(f"Running SmartSpotlight on clip {start_time:.1f}s with planned device {spotlight_policy}...", flush=True)
                spotlight_started = time.perf_counter()
                spotlight_result = analyze_spotlight_path(
                    video_path,
                    start_time,
                    duration,
                    device=spotlight_policy,
                    strict_device=spotlight_strict,
                    target_w=target_w,
                    target_h=target_h,
                )
                spotlight_elapsed = time.perf_counter() - spotlight_started

                if isinstance(spotlight_result, tuple) and len(spotlight_result) == 3:
                    camera_path, actual_spotlight_device, spotlight_error = spotlight_result
                else:
                    camera_path = spotlight_result
                    actual_spotlight_device = spotlight_policy
                    spotlight_error = None

                if spotlight_policy == "gpu" and actual_spotlight_device != "gpu":
                    _record_profile_event(
                        "spotlight",
                        "gpu",
                        duration,
                        spotlight_elapsed,
                        hardware_profile,
                        success=False,
                        error=spotlight_error or "runtime fallback to CPU",
                    )
                _record_profile_event(
                    "spotlight",
                    actual_spotlight_device,
                    duration,
                    spotlight_elapsed,
                    hardware_profile,
                    success=True,
                )

                cmd_file = os.path.join(output_dir, f"spotlight_{i+1:03d}_{int(start_time)}.txt")
                with open(cmd_file, "w", encoding="utf-8") as handle:
                    for ts, x in camera_path:
                        handle.write(f"{ts - start_time:.3f} overlay x {x};\n")
                clip_temp_files.append(cmd_file)

                safe_cmd_path = escape_ffmpeg_path(cmd_file)
                if use_subtitles:
                    sub_part = f"[base]subtitles='{safe_ass_path}':force_style='Alignment=2,MarginV=150'[sub]; "
                    v_label_after_sub = "[sub]"
                else:
                    sub_part = ""
                    v_label_after_sub = "[base]"

                badge_part = ""
                if use_hook_badge:
                    badge_part = (
                        f"{v_label_after_sub}drawtext=text='{badge_text}':fontcolor=white:fontsize=42:x=50:y=60:"
                        f"box=1:boxcolor={safe_badge_color}:boxborderw=18:fontfile='{safe_font_path}'[badge]; "
                    )
                    v_label_after_badge = "[badge]"
                else:
                    v_label_after_badge = v_label_after_sub

                filter_script = (
                    f"color=c=black:s={target_w}x{target_h}:d={duration}[bg]; "
                    f"[0:v]scale=-1:{target_h}[vid]; "
                    f"[bg][vid]overlay=x=0:y=0[base_raw]; [base_raw]sendcmd=f='{safe_cmd_path}'[base]; "
                    f"{sub_part}"
                    f"{badge_part}"
                    f"{v_label_after_badge}drawbox=y=ih-20:color=white@0.3:width=iw:height=10[bar_bg]; "
                    f"[bar_bg]drawbox=y=ih-20:color=white@0.8:width=iw*(t/{duration}):height=10:t=fill[outv]"
                )
                v_final_label = "[outv]"
            else:
                if use_subtitles:
                    sub_part = f"[base]subtitles='{safe_ass_path}'[sub]; "
                    v_label_after_sub = "[sub]"
                else:
                    sub_part = ""
                    v_label_after_sub = "[base]"

                badge_part = ""
                if use_hook_badge:
                    badge_part = (
                        f"{v_label_after_sub}drawtext=text='{badge_text}':fontcolor=white:fontsize=42:x=50:y=60:"
                        f"box=1:boxcolor={safe_badge_color}:boxborderw=18:fontfile='{safe_font_path}'[badge]; "
                    )
                    v_label_after_badge = "[badge]"
                else:
                    v_label_after_badge = v_label_after_sub

                filter_script = (
                    f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=20:10[bg]; "
                    f"[0:v]scale={target_w}:-1[fg]; "
                    f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base]; "
                    f"{sub_part}"
                    f"{badge_part}"
                    f"color=c=yellow:s={target_w}x14[pb]; {v_label_after_badge}[pb]overlay=x=-W+(t/{duration})*W:y=H-14[pb_over]"
                )
                v_final_label = "[pb_over]"

            has_watermark_img = use_watermark and os.path.exists(WATERMARK_IMAGE)
            logo_input_label = None
            if has_watermark_img:
                # Logo will be the last input.
                # Inputs: 0:video, [1:facecam], [last:logo]
                if facecam_video_path:
                    logo_input_label = "[2:v]"
                else:
                    logo_input_label = "[1:v]"

            vf_full = build_branding_filter(
                filter_script,
                v_final_label,
                safe_font_path,
                channel_name,
                use_watermark,
                logo_label=logo_input_label,
            )

            def try_ffmpeg(encoder_name, use_filters=True, music_path=None):
                cmd = [
                    "ffmpeg",
                    "-ss", str(start_time),
                    "-t", str(duration),
                    "-i", video_path,
                ]

                if facecam_video_path:
                    cmd += ["-ss", str(start_time), "-t", str(duration), "-i", facecam_video_path]

                has_logo = use_watermark and os.path.exists(logo_path)
                if has_logo:
                    cmd += ["-i", logo_path]
                
                if music_path:
                    cmd += ["-stream_loop", "-1", "-i", music_path]

                if use_filters:
                    comp = vf_full
                    # Audio Pipeline
                    # Audio Pipeline
                    # Calculate index for music input: game(0) + facecam(1/0) + logo(1/0)
                    music_idx = 1 + (1 if facecam_video_path else 0) + (1 if has_logo else 0)
                    music_label = f"[{music_idx}:a]" if music_path else None
                    
                    # Ducking logic: find speech in this window
                    # For now, simplistic ducking:
                    ducking_levels = [{"start": 0, "end": duration, "is_speech": True}] 
                    
                    a_comp = audio_director.generate_audio_mix_filter(
                        "[0:a]", 
                        music_label, 
                        ducking_levels, 
                        duration,
                        facecam_audio_label="[1:a]" if facecam_has_audio else None
                    )
                    
                    # Professional loop transitions: 0.3s audio fade in/out to mask popping, video remains loopable
                    comp += f"; {a_comp}; [aout]afade=t=in:st=0:d=0.3,afade=t=out:st={duration-0.3}:d=0.3[aout_final]"
                    cmd += ["-filter_complex", comp, "-map", "[v]", "-map", "[aout_final]"]
                else:
                    v_comp = f"[0:v]crop='trunc(ih*{target_w}/{target_h}/2)*2':ih,scale={target_w}:{target_h}[v]"
                    cmd += ["-filter_complex", v_comp, "-map", "[v]", "-map", "0:a?"]

                # Force Constant Frame Rate (CFR) to prevent 'corrupted' video on VFR sources
                cmd += ["-vsync", "cfr", "-r", "60"]

                if encoder_name == "h264_nvenc":
                    cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "20", "-tune", "hq"]
                else:
                    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]

                cmd += ["-t", str(duration), "-c:a", "aac", "-y", output_filename]

                mode_label = f"{encoder_name}" + (" + Music" if music_path else "")
                print(f"> Running FFmpeg ({mode_label}): {' '.join(cmd)}")

                # Limit FFmpeg runtime to 15 minutes to prevent 10-hour hangs
                return subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    cwd=output_dir,
                    env=_ffmpeg_env(),
                    timeout=900, 
                )

            success = False
            if gen_short:
                planned_gpu = encode_policy == "gpu"
                if planned_gpu:
                    encode_started = time.perf_counter()
                    try:
                        try_ffmpeg("h264_nvenc", use_filters=True, music_path=selected_music if use_music else None)
                        encode_elapsed = time.perf_counter() - encode_started
                        _record_profile_event("encode", "gpu", duration, encode_elapsed, hardware_profile, success=True)
                        print(f"Successfully created {output_file} (using GPU)")
                        success = True
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as gpu_exc:
                        encode_elapsed = time.perf_counter() - encode_started
                        err_bytes = gpu_exc.stderr or b""
                        gpu_err = err_bytes.decode("utf-8", errors="replace")
                        if isinstance(gpu_exc, subprocess.TimeoutExpired):
                            gpu_err += "\n[TIMED OUT]"
                        _record_profile_event(
                            "encode",
                            "gpu",
                            duration,
                            encode_elapsed,
                            hardware_profile,
                            success=False,
                            error=gpu_err[-800:],
                        )
                        if encode_strict:
                            print(f"FATAL: GPU encoding failed for clip {i+1} in strict GPU mode.")
                            print(f"--- GPU Error (last 800 chars) ---\n...{gpu_err[-800:]}")
                            raise
                        print(f"GPU encoding failed for clip {i+1}, trying CPU fallback...")
                        cpu_started = time.perf_counter()
                        try:
                            try_ffmpeg("libx264", use_filters=True, music_path=selected_music if use_music else None)
                            cpu_elapsed = time.perf_counter() - cpu_started
                            _record_profile_event("encode", "cpu", duration, cpu_elapsed, hardware_profile, success=True)
                            print(f"Successfully created {output_file} (using CPU)")
                            success = True
                        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as cpu_exc:
                            cpu_elapsed = time.perf_counter() - cpu_started
                            err_bytes = cpu_exc.stderr or b""
                            cpu_err = err_bytes.decode("utf-8", errors="replace")
                            if isinstance(cpu_exc, subprocess.TimeoutExpired):
                                cpu_err += "\n[TIMED OUT]"
                            _record_profile_event(
                                "encode",
                                "cpu",
                                duration,
                                cpu_elapsed,
                                hardware_profile,
                                success=False,
                                error=cpu_err[-800:],
                            )
                            print(f"Warning: full rendering failed for clip {i+1}. Attempting SAFE MODE (No Subtitles)...")
                            safe_started = time.perf_counter()
                            try:
                                try_ffmpeg("libx264", use_filters=False)
                                safe_elapsed = time.perf_counter() - safe_started
                                _record_profile_event("encode", "cpu", duration, safe_elapsed, hardware_profile, success=True)
                                print(f"Successfully created {output_file} (SAFE MODE - NO SUBTITLES)")
                                success = True
                            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as safe_exc:
                                safe_elapsed = time.perf_counter() - safe_started
                                err_bytes = safe_exc.stderr or b""
                                safe_err = err_bytes.decode("utf-8", errors="replace")
                                if isinstance(safe_exc, subprocess.TimeoutExpired):
                                    safe_err += "\n[TIMED OUT]"
                                _record_profile_event(
                                    "encode",
                                    "cpu",
                                    duration,
                                    safe_elapsed,
                                    hardware_profile,
                                    success=False,
                                    error=safe_err[-800:],
                                )
                                print(f"FATAL: clip {i+1} failed even in Safe Mode.")
                                print(f"--- GPU Error (last 800 chars) ---\n...{gpu_err[-800:]}")
                                print(f"--- CPU Error (last 800 chars) ---\n...{cpu_err[-800:]}")
                                print(f"--- Safe Mode Error (last 800 chars) ---\n...{safe_err[-800:]}")
                else:
                    cpu_started = time.perf_counter()
                    try:
                        try_ffmpeg("libx264", use_filters=True, music_path=selected_music if use_music else None)
                        cpu_elapsed = time.perf_counter() - cpu_started
                        _record_profile_event("encode", "cpu", duration, cpu_elapsed, hardware_profile, success=True)
                        print(f"Successfully created {output_file} (using CPU)")
                        success = True
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as cpu_exc:
                        cpu_elapsed = time.perf_counter() - cpu_started
                        err_bytes = cpu_exc.stderr or b""
                        cpu_err = err_bytes.decode("utf-8", errors="replace")
                        if isinstance(cpu_exc, subprocess.TimeoutExpired):
                            cpu_err += "\n[TIMED OUT]"
                        _record_profile_event(
                            "encode",
                            "cpu",
                            duration,
                            cpu_elapsed,
                            hardware_profile,
                            success=False,
                            error=cpu_err[-800:],
                        )
                        print(f"Warning: full rendering failed for clip {i+1}. Attempting SAFE MODE (No Subtitles)...")
                        safe_started = time.perf_counter()
                        try:
                            try_ffmpeg("libx264", use_filters=False)
                            safe_elapsed = time.perf_counter() - safe_started
                            _record_profile_event("encode", "cpu", duration, safe_elapsed, hardware_profile, success=True)
                            print(f"Successfully created {output_file} (SAFE MODE - NO SUBTITLES)")
                            success = True
                        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as safe_exc:
                            safe_elapsed = time.perf_counter() - safe_started
                            err_bytes = safe_exc.stderr or b""
                            safe_err = err_bytes.decode("utf-8", errors="replace")
                            if isinstance(safe_exc, subprocess.TimeoutExpired):
                                safe_err += "\n[TIMED OUT]"
                            _record_profile_event(
                                "encode",
                                "cpu",
                                duration,
                                safe_elapsed,
                                hardware_profile,
                                success=False,
                                error=safe_err[-800:],
                            )
                            print(f"FATAL: clip {i+1} failed even in Safe Mode.")
                            print(f"--- CPU Error (last 800 chars) ---\n...{cpu_err[-800:]}")
                            print(f"--- Safe Mode Error (last 800 chars) ---\n...{safe_err[-800:]}")
            else:
                print(f"Skipping short video generation for segment {i+1}.")
                success = True # Skip short rendering

            if success:
                clips_created += 1
                desc_file = os.path.join(output_dir, f"clip_{i+1:03d}_{score}V_{safe_category}_desc.txt")
                with open(desc_file, "w", encoding="utf-8") as handle:
                    handle.write(
                        f"Category: {category}\n"
                        f"Virality Hook Score: {score}/100\n"
                        f"Reason for clip: {moment.get('reason')}\n\n"
                        f"Transcript Snippet:\n{moment.get('text')}\n"
                    )

                long_duration_target = 600
                ext_start = max(0, start_time - (long_duration_target / 2))
                ext_end = min(total_video_duration, ext_start + long_duration_target)

                if ext_end == total_video_duration:
                    ext_start = max(0, total_video_duration - long_duration_target)

                f_ext_start = format_time_simple(ext_start)
                f_ext_end = format_time_simple(ext_end)
                long_filename = f"clip_{i+1:03d}_{score}V_{safe_category}_LONG_{f_ext_start}-{f_ext_end}.mp4"
                ext_duration = ext_end - ext_start

                if gen_long:
                    print(
                        f"Creating Extended Highlight: {format_time_hhmmss(ext_start)} "
                        f"to {format_time_hhmmss(ext_end)} -> {long_filename}"
                    )
                    create_extended_highlight(video_path, output_dir, long_filename, ext_start, ext_duration, long_res=long_res, use_watermark=use_watermark, watermark_src=watermark_src, channel_name=channel_name, use_hook_badge=use_hook_badge)
        finally:
            cleanup_temp_files(clip_temp_files)

    if clips_created == 0:
        raise RuntimeError("No clips were successfully created. Check FFmpeg errors above.")


def run_cli(argv=None):
    args = build_arg_parser().parse_args(argv)
    pro_settings = {
        "use_facecam": args.facecam,
        "use_watermark": args.watermark,
        "channel_name": args.channel,
    }
    create_clips(
        args.video,
        best_moments_path=args.moments,
        output_dir=args.output,
        transcript_path=args.transcript,
        pro_settings=pro_settings,
        facecam_video_path=args.facecam_src,
        use_spotlight=args.spotlight,
        hardware_mode=args.hardware_mode,
        encode_device=args.encode_device,
        spotlight_device=args.spotlight_device,
        visual_device=args.visual_device,
        hardware_profile=args.hardware_profile,
        gen_short=args.short,
        gen_long=args.long,
        short_res=args.short_res,
        long_res=args.long_res,
        watermark_src=args.watermark_src,
        use_hook_badge=args.hook_badge,
        use_subtitles=args.subtitles,
        use_music=args.music,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_cli())
    except Exception as exc:
        print(f"\nClipper Error: {exc}")
        raise SystemExit(1)
