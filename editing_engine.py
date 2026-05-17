"""
editing_engine.py — Short-Form Video Rendering Engine
=====================================================
Transforms selected clips into engaging short-form videos with:
- Vertical 1080x1920 layout with blurred background
- Smooth zoom easing (gradual accel → peak → release) for human-feel pacing
- Smart slow motion: combat/impact events only, max 0.8s, no stacking
- Payoff detection: extends clip ~1s when silence/pause follows peak
- Hook & Caption overlays with fade animations
- Trimming to exact pacing-profile-aware boundaries
- Robust stability, validation, and fallback layers
- Debug metadata: pre_context_used, post_payoff_used, slow_motion_triggered, hook_style
"""

import json
import logging
import os
import subprocess
import random
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Primary and Fallback Font logic
PRIMARY_FONT = "C\\:/Windows/Fonts/impact.ttf" if os.name == 'nt' else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FALLBACK_FONT = "Arial" if os.name == 'nt' else "Sans"

def get_safe_font() -> str:
    """Returns the primary font if it exists on the system, otherwise returns the fallback."""
    # Check if the primary font file exists (stripping the FFmpeg-style path escaping)
    clean_path = PRIMARY_FONT.replace('C\\:', 'C:').replace('\\:', ':')
    if os.path.exists(clean_path):
        return PRIMARY_FONT
    return FALLBACK_FONT

def sanitize_text(text: str, max_len: int = 40) -> str:
    """Removes breaking characters and enforces length limits to prevent overflow."""
    if not text:
        return ""
    # Remove quotes and backslashes
    clean = text.replace("'", "").replace('"', '').replace('\\', '').strip()
    # Caption Overflow Guard: Truncate if necessary
    if len(clean) > max_len:
        clean = clean[:max_len-3] + "..."
    return clean


# Legacy detect_payoff removed; logic is now in payoff_detector.py and evaluated during fusion.

def validate_video(file_path: str) -> bool:
    """Validates that a video file is playable and has correct resolution."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return False
        
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 3:
            width = int(lines[0])
            height = int(lines[1])
            # Basic resolution check
            if width != 1080 or height != 1920:
                return False
            return True
    except Exception:
        pass
    return False

class EditingEngine:
    def __init__(self, output_dir: str = "output", preview_mode: bool = False, seed: int = None):
        self.output_dir = output_dir
        self.preview_mode = preview_mode
        self.max_size_mb = 15.0  # Target max file size for platform compatibility
        self._slow_motion_triggered = False  # set per-render by _build_filtergraph
        if seed is not None:
            random.seed(seed)
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def _trim_video(self, input_path: str, start: float, end: float, temp_out: str):
        duration = end - start
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-c:a", "aac",
            temp_out
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _build_filtergraph(self, event: Dict[str, Any], hook: str, captions: list, peak_time_rel: float, clip_duration: float) -> str:
        filters = []
        font_to_use = get_safe_font()
        
        # 1. Background
        res_h = 960 if self.preview_mode else 1920
        res_w = 540 if self.preview_mode else 1080
        
        if self.preview_mode:
            filters.append(f"[0:v]scale={res_w}:{res_h}:force_original_aspect_ratio=increase,crop={res_w}:{res_h},boxblur=10:10[bg]")
        else:
            filters.append(f"[0:v]scale={res_w}:{res_h}:force_original_aspect_ratio=increase,crop={res_w}:{res_h},boxblur=20:20[bg]")

        # 2. Foreground Motion Engine — smooth easing instead of instant zoom jump
        intensity  = event.get("intensity", 5)
        event_type = event.get("event_type", "neutral")
        peak_prominence = float(event.get("peak_prominence", 0.0))

        # --- Slow-motion gate ------------------------------------------------
        # Only fires for high-intensity combat/impact moments with a clear peak.
        # Max window 0.8s; never stacks across multiple events.
        _SLOWMO_ELIGIBLE_TYPES = {"combat", "surprise", "reaction"}
        _SLOWMO_INTENSITY_FLOOR = 7          # intensity must be ≥ 7/10
        _SLOWMO_PROMINENCE_FLOOR = 0.15      # peak must be clearly above local avg
        _SLOWMO_MAX_SECS = 0.8

        use_slow_motion = (
            not self.preview_mode
            and intensity >= _SLOWMO_INTENSITY_FLOOR
            and peak_prominence >= _SLOWMO_PROMINENCE_FLOOR
            and event_type in _SLOWMO_ELIGIBLE_TYPES
        )
        self._slow_motion_triggered = use_slow_motion  # saved for debug metadata

        # --- Zoom expression — gradual accel → peak → gentle release ----------
        if intensity <= 4:
            # Low-energy clip: very subtle drift, no jump
            z_expr = f"1.02 + 0.03*(time/{max(clip_duration, 0.1)})"
            shake_x, shake_y = "0", "0"
        else:
            # Smooth easing in three phases:
            #   Phase 1 (0 → peak): gradual zoom-in  (cubic ease-in feel via squared ramp)
            #   Phase 2 (peak → peak+0.5): hold at max zoom
            #   Phase 3 (peak+0.5 → end): slow release back toward 1.0
            pt = peak_time_rel
            peak_z   = 1.18           # max zoom level — intentional but not jarring
            settle_z = 1.05           # where zoom settles after peak
            hold_end = pt + 0.5       # how long to hold peak zoom

            # Phase 1: ease-in using a squared ramp for natural acceleration feel
            phase1 = f"1.0 + ({peak_z}-1.0)*pow(time/{max(pt, 0.01)}, 2)"
            # Phase 2: hold
            phase2 = f"{peak_z}"
            # Phase 3: ease-out linearly toward settle_z
            release_dur = max(clip_duration - hold_end, 0.1)
            phase3 = f"{peak_z} - ({peak_z}-{settle_z})*((time-{hold_end})/{release_dur})"

            z_expr = (
                f"if(lt(time,{pt}), {phase1}, "
                f"if(lt(time,{hold_end}), {phase2}, "
                f"max({settle_z}, {phase3})))"
            )

            # Shake: only for very high intensity, and softened vs before
            shake_x, shake_y = "0", "0"
            if intensity > 8 and not self.preview_mode:
                shake_dur = 0.35   # slightly shorter for cleaner feel
                shake_x = f"if(between(time,{pt},{pt}+{shake_dur}), (random(1)-0.5)*14, 0)"
                shake_y = f"if(between(time,{pt},{pt}+{shake_dur}), (random(1)-0.5)*14, 0)"

        x_expr = f"iw/2-(iw/zoom)/2 + {shake_x}"
        y_expr = f"ih/2-(ih/zoom)/2 + {shake_y}"

        if self.preview_mode and intensity <= 4:
            filters.append(f"[0:v]scale=-1:{res_h}[fg_zoomed];[bg][fg_zoomed]overlay=(W-w)/2:(H-h)/2[base_layout]")
        else:
            slowmo_filter = ""
            if use_slow_motion:
                # Insert setpts slow-down capped to _SLOWMO_MAX_SECS after peak
                slowmo_filter = (
                    f"[fg_scaled]setpts=if(between(t,{peak_time_rel},{peak_time_rel}+{_SLOWMO_MAX_SECS}),"
                    f"2.0*PTS,PTS)[fg_slowed];"
                )
                slo_src = "[fg_slowed]"
            else:
                slo_src = "[fg_scaled]"

            filters.append(
                f"[0:v]scale=-1:{res_h}[fg_scaled];"
                + slowmo_filter +
                f"{slo_src}zoompan=z='{z_expr}':d=1:x='{x_expr}':y='{y_expr}':fps=30[fg_zoomed];"
                f"[bg][fg_zoomed]overlay=(W-w)/2:(H-h)/2[base_layout]"
            )

        current_layer = "[base_layout]"
        
        # 3. Hook Overlay (Hook Visibility Enhancement: Black box for readability)
        hook_text = sanitize_text(hook, max_len=30) or "This did not go to plan…"
        fs_hook = 48 if self.preview_mode else 72
        hook_filter = (
            f"{current_layer}drawtext=text='{hook_text}':fontfile='{font_to_use}':"
            f"fontcolor=white:bordercolor=black:borderw=4:fontsize={fs_hook}:"
            f"box=1:boxcolor=black@0.4:boxborderw=10:" # Enhancement: Added semi-transparent box
            f"x=(w-text_w)/2:y=(h-text_h)*0.15:enable='between(t,0,2)'[with_hook]"
        )
        filters.append(hook_filter)
        current_layer = "[with_hook]"

        # 4. Captions Overlay
        fs_cap = 42 if self.preview_mode else 64
        if not captions:
            captions = [{"time": peak_time_rel, "text": "Wait for it..."}]
            
        for i, cap in enumerate(captions):
            text = sanitize_text(cap.get("text", ""), max_len=35)
            start_t = max(0.0, min(float(cap.get("time", 0.0)), clip_duration - 1.0))
            
            if i + 1 < len(captions):
                next_t = float(captions[i+1].get("time", start_t + 2.0))
                end_t = max(start_t + 0.5, min(next_t, clip_duration))
            else:
                end_t = min(start_t + 2.0, clip_duration)

            alpha_expr = (
                f"if(lt(t,{start_t}),0,if(lt(t,{start_t}+0.15),(t-{start_t})/0.15,"
                f"if(lt(t,{end_t}-0.15),1,if(lt(t,{end_t}),1-(t-({end_t}-0.15))/0.15,0))))"
            )

            out_layer = f"[cap_{i}]" if i < len(captions)-1 else "[v_out]"
            cap_filter = (
                f"{current_layer}drawtext=text='{text}':fontfile='{font_to_use}':"
                f"fontcolor=white:bordercolor=black:borderw=3:fontsize={fs_cap}:"
                f"x=(w-text_w)/2:y=(h-text_h)*0.85:alpha='{alpha_expr}'{out_layer}"
            )
            filters.append(cap_filter)
            current_layer = out_layer

        if not filters or "[v_out]" not in filters[-1]:
             filters.append(f"{current_layer}copy[v_out]")

        return ";".join(filters)

    def render_short(self, input_data: Dict[str, Any]) -> str:
        """Main execution function with render time tracking and file size control."""
        render_start_wall = time.time()  # Render Time Tracking
        self._slow_motion_triggered = False  # reset per render

        video_path = input_data.get("video_path")
        event      = input_data.get("event", {})
        hook       = input_data.get("hook", "This did not go to plan\u2026")
        captions   = input_data.get("captions", [])
        event_id   = input_data.get("event_id", f"evt_{int(time.time())}")
        hook_style = input_data.get("hook_style", "contextual")
        fallback_used = False

        if not video_path or not os.path.exists(video_path):
            return json.dumps({"error": "Input video missing"})

        # ── Timing extraction ─────────────────────────────────────────────────
        # The start and end timestamps from the fusion engine ALREADY include
        # the pacing buffers and the dynamic payoff extensions.
        # We enforce a hard length cap to prevent runaway clips.
        trim_start    = float(event.get("start", 0))
        trim_end      = float(event.get("end", trim_start + 10))
        raw_peak      = float(event.get("peak_time", trim_start + 2))

        pre_context_buf  = float(event.get("pre_context_buffer",  2.0))
        post_payoff_buf  = float(event.get("post_payoff_buffer",  1.5))
        
        evidence = event.get("evidence", {})
        payoff_detected = evidence.get("payoff_detected", False)
        resolution_score = evidence.get("resolution_score", 0.0)
        ending_extension_used = evidence.get("ending_extension_used", 0.0)
        ending_reason = evidence.get("ending_reason", "fixed_buffer")

        clip_duration = trim_end - trim_start
        MAX_TOTAL_DURATION = 18.0
        if clip_duration > MAX_TOTAL_DURATION:
            trim_end = trim_start + MAX_TOTAL_DURATION
            clip_duration = MAX_TOTAL_DURATION

        clip_duration = max(7.0, clip_duration)
        peak_rel      = max(0.0, min(raw_peak - trim_start, clip_duration))

        base_name = os.path.basename(video_path).split('.')[0]
        temp_trimmed = os.path.join(self.output_dir, f"temp_{base_name}.mp4")
        final_output = os.path.join(self.output_dir, f"short_{base_name}.mp4")

        try:
            # 1. Trimming
            self._trim_video(video_path, trim_start, trim_start + clip_duration, temp_trimmed)

            # 2. Filtering
            filtergraph = self._build_filtergraph(event, hook, captions, peak_rel, clip_duration)
            crf, preset = ("28", "ultrafast") if self.preview_mode else ("22", "fast")

            cmd = [
                "ffmpeg", "-y", "-i", temp_trimmed, "-filter_complex", filtergraph,
                "-map", "[v_out]", "-map", "0:a", "-c:v", "libx264", "-preset", preset, "-crf", crf,
                "-c:a", "aac", "-b:a", "192k", final_output
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Output File Size Control
            file_size_mb = os.path.getsize(final_output) / (1024 * 1024)
            if file_size_mb > self.max_size_mb and not self.preview_mode:
                logger.info(f"File size too large ({file_size_mb:.1f}MB). Re-encoding with bitrate limit.")
                cmd_reencode = [
                    "ffmpeg", "-y", "-i", final_output, "-c:v", "libx264", "-b:v", "4M", 
                    "-maxrate", "5M", "-bufsize", "10M", "-c:a", "copy", 
                    final_output.replace(".mp4", "_optimized.mp4")
                ]
                subprocess.run(cmd_reencode, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.replace(final_output.replace(".mp4", "_optimized.mp4"), final_output)
                file_size_mb = os.path.getsize(final_output) / (1024 * 1024)

            # Validation
            if not self.preview_mode and not validate_video(final_output):
                logger.warning("Validation failed. Returning raw trim.")
                os.replace(temp_trimmed, final_output)
                fallback_used = True

        except Exception as e:
            logger.error(f"Render Error: {e}")
            fallback_used = True
            if os.path.exists(temp_trimmed):
                os.rename(temp_trimmed, final_output)

        finally:
            if os.path.exists(temp_trimmed):
                try: os.remove(temp_trimmed)
                except: pass

        render_time = time.time() - render_start_wall  # Calculate Render Duration

        # ── Debug Review Metadata ─────────────────────────────────────────────
        # Emitted with every render so manual reviewers and the learning system
        # can trace exactly which pacing decisions were applied to each clip.
        result_meta = {
            "event_id":              event_id,
            "output_path":           final_output,
            "duration":              round(clip_duration, 2),
            "render_time":           round(render_time, 2),
            "file_size_mb":          round(os.path.getsize(final_output) / (1024 * 1024), 2),
            "fallback_used":         fallback_used,
            # ── Phase 1 pacing debug fields ──────────────────────────────────
            "pre_context_used":      round(pre_context_buf, 2),
            "post_payoff_used":      round(post_payoff_buf, 2),
            "slow_motion_triggered": self._slow_motion_triggered,
            "hook_style":            hook_style,
            # ── Payoff Completion Debug ──────────────────────────────────────
            "payoff_detected":       payoff_detected,
            "resolution_score":      round(resolution_score, 3),
            "ending_extension_used": round(ending_extension_used, 2),
            "ending_reason":         ending_reason,
        }

        logger.info(f"Render Complete: {json.dumps(result_meta, indent=2)}")
        return json.dumps(result_meta)

def render_short(input_data: Dict[str, Any], preview_mode: bool = False) -> str:
    engine = EditingEngine(preview_mode=preview_mode)
    return engine.render_short(input_data)
