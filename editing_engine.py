"""
editing_engine.py — Short-Form Video Rendering Engine
=====================================================
Transforms selected clips into engaging short-form videos with:
- Vertical 1080x1920 layout with blurred background
- Zoom & Shake motion dynamics around peak moments
- Hook & Caption overlays with fade animations
- Trimming to exact event boundaries
- Robust stability, validation, and fallback layers
"""

import json
import logging
import os
import subprocess
import random
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_FONT = "Arial"
if os.name == 'nt':
    DEFAULT_FONT = "C\\:/Windows/Fonts/impact.ttf"

def sanitize_text(text: str) -> str:
    """Removes characters that could break FFmpeg text rendering."""
    if not text:
        return ""
    # Remove quotes and backslashes to avoid filtergraph syntax errors
    return text.replace("'", "").replace('"', '').replace('\\', '').strip()

def validate_video(file_path: str, expected_duration: float = None) -> bool:
    """Validates that a video file is playable, has correct resolution, and duration."""
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
            duration = float(lines[2])
            
            # Check resolution
            if width != 1080 or height != 1920:
                logger.warning(f"Validation failed: Invalid resolution {width}x{height}")
                return False
                
            # Check duration limits roughly
            if duration < 6.5 or duration > 15.5:
                logger.warning(f"Validation failed: Duration {duration}s out of bounds")
                return False
                
            return True
    except Exception as e:
        logger.warning(f"Validation failed during ffprobe: {e}")
        
    return False

class EditingEngine:
    def __init__(self, output_dir: str = "output", preview_mode: bool = False, seed: int = None):
        self.output_dir = output_dir
        self.preview_mode = preview_mode
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
        
        # 1. Background (Blurred and Scaled)
        if self.preview_mode:
            filters.append("[0:v]scale=540:960:force_original_aspect_ratio=increase,crop=540:960,boxblur=10:10[bg]")
        else:
            filters.append("[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:20[bg]")

        # 2. Foreground Motion Engine (Zoom + Shake)
        intensity = event.get("intensity", 5)
        
        # Minimum Motion Guarantee
        if intensity <= 4:
            # Subtle zoom
            z_expr = f"1.03 + 0.02*(time/{clip_duration})"
            shake_x = "0"
            shake_y = "0"
        else:
            # Normal zoom logic
            z_expr = (
                f"if(lt(time,{peak_time_rel}), 1.0+0.1*(time/{peak_time_rel}), "
                f"if(lt(time,{peak_time_rel}+0.3), 1.2, "
                f"max(1.0, 1.2-0.2*(time-{peak_time_rel}-0.3))))"
            )
            
            shake_x = "0"
            shake_y = "0"
            if intensity > 8 and not self.preview_mode:
                shake_x = f"if(between(time,{peak_time_rel},{peak_time_rel}+0.4), (random(1)-0.5)*20, 0)"
                shake_y = f"if(between(time,{peak_time_rel},{peak_time_rel}+0.4), (random(1)-0.5)*20, 0)"

        x_expr = f"iw/2-(iw/zoom)/2 + {shake_x}"
        y_expr = f"ih/2-(ih/zoom)/2 + {shake_y}"

        res_h = 960 if self.preview_mode else 1920
        res_w = 540 if self.preview_mode else 1080
        
        # Prevent zoompan heavy processing in preview if low intensity
        if self.preview_mode and intensity <= 4:
            filters.append(
                f"[0:v]scale=-1:{res_h}[fg_zoomed];"
                f"[bg][fg_zoomed]overlay=(W-w)/2:(H-h)/2[base_layout]"
            )
        else:
            filters.append(
                f"[0:v]scale=-1:{res_h}[fg_scaled];"
                f"[fg_scaled]zoompan=z='{z_expr}':d=1:x='{x_expr}':y='{y_expr}':fps=30[fg_zoomed];"
                f"[bg][fg_zoomed]overlay=(W-w)/2:(H-h)/2[base_layout]"
            )

        current_layer = "[base_layout]"
        
        # 3. Hook Overlay (0-2s)
        hook = sanitize_text(hook) or "THIS WAS CRAZY 😳"
        
        fs_hook = 48 if self.preview_mode else 72
        hook_filter = (
            f"{current_layer}drawtext=text='{hook}':fontfile='{DEFAULT_FONT}':"
            f"fontcolor=white:bordercolor=black:borderw=4:fontsize={fs_hook}:"
            f"x=(w-text_w)/2:y=(h-text_h)*0.15:enable='between(t,0,2)'[with_hook]"
        )
        filters.append(hook_filter)
        current_layer = "[with_hook]"

        # 4. Captions Overlay
        fs_cap = 42 if self.preview_mode else 64
        
        if not captions:
            captions = [{"time": peak_time_rel, "text": "Wait for it..."}]
            
        for i, cap in enumerate(captions):
            text = sanitize_text(cap.get("text", "Wait..."))
            
            # Time Alignment Safety
            start_t = float(cap.get("time", 0.0))
            start_t = max(0.0, min(start_t, clip_duration - 1.0))
            
            if i + 1 < len(captions):
                next_t = float(captions[i+1].get("time", start_t + 2.0))
                end_t = max(start_t + 0.5, min(next_t, clip_duration))
            else:
                end_t = min(start_t + 2.0, clip_duration)

            alpha_expr = (
                f"if(lt(t,{start_t}),0,"
                f"if(lt(t,{start_t}+0.15),(t-{start_t})/0.15,"
                f"if(lt(t,{end_t}-0.15),1,"
                f"if(lt(t,{end_t}),1-(t-({end_t}-0.15))/0.15,0))))"
            )

            out_layer = f"[cap_{i}]" if i < len(captions)-1 else "[v_out]"
            cap_filter = (
                f"{current_layer}drawtext=text='{text}':fontfile='{DEFAULT_FONT}':"
                f"fontcolor=white:bordercolor=black:borderw=3:fontsize={fs_cap}:"
                f"x=(w-text_w)/2:y=(h-text_h)*0.85:alpha='{alpha_expr}'{out_layer}"
            )
            filters.append(cap_filter)
            current_layer = out_layer

        if len(captions) == 0:
            filters.append(f"{current_layer}copy[v_out]")

        return ";".join(filters)

    def render_short(self, input_data: Dict[str, Any]) -> str:
        video_path = input_data.get("video_path")
        event = input_data.get("event", {})
        hook = input_data.get("hook")
        captions = input_data.get("captions", [])
        event_id = input_data.get("event_id", f"evt_{int(time.time())}")
        fallback_used = False

        if not hook:
            hook = "THIS WAS CRAZY 😳"
            fallback_used = True
            
        if not captions:
            captions = [{"time": 0.5, "text": "Wait for it..."}]
            fallback_used = True

        if not video_path or not os.path.exists(video_path):
            logger.error(f"Render failed: Input video not found {video_path}")
            return json.dumps({"error": "Input video not found"})

        raw_start = float(event.get("start", 0))
        raw_end = float(event.get("end", raw_start + 10))
        raw_peak = float(event.get("peak_time", raw_start + 2))

        trim_start = max(0.0, raw_start - 1.5)
        trim_end = raw_end + 0.5
        
        duration = trim_end - trim_start
        if duration < 7.0:
            trim_end = trim_start + 7.0
        elif duration > 15.0:
            trim_end = trim_start + 15.0
            
        clip_duration = trim_end - trim_start
        peak_rel = max(0.0, min(raw_peak - trim_start, clip_duration))

        base_name = os.path.basename(video_path).split('.')[0]
        temp_trimmed = os.path.join(self.output_dir, f"temp_{base_name}.mp4")
        final_output = os.path.join(self.output_dir, f"short_{base_name}.mp4")

        # Logging start
        logger.info(
            f"--- Starting Render ---\n"
            f"Event ID: {event_id}\n"
            f"Selected Hook: {hook}\n"
            f"Caption Count: {len(captions)}\n"
            f"Duration: {clip_duration:.2f}s\n"
            f"Effects: Preview={self.preview_mode}, Intensity={event.get('intensity', 5)}\n"
            f"Fallback Used: {fallback_used}\n"
            f"-----------------------"
        )

        try:
            self._trim_video(video_path, trim_start, trim_end, temp_trimmed)

            filtergraph = self._build_filtergraph(event, hook, captions, peak_rel, clip_duration)

            crf = "28" if self.preview_mode else "22"
            preset = "ultrafast" if self.preview_mode else "fast"

            cmd = [
                "ffmpeg", "-y",
                "-i", temp_trimmed,
                "-filter_complex", filtergraph,
                "-map", "[v_out]",
                "-map", "0:a",
                "-c:v", "libx264", "-preset", preset, "-crf", crf,
                "-c:a", "aac", "-b:a", "192k",
                final_output
            ]

            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Validation
            if not self.preview_mode and not validate_video(final_output, clip_duration):
                logger.warning("Output validation failed! Returning trimmed clip as fallback.")
                fallback_used = True
                fallback_cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_trimmed,
                    "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
                    "-c:v", "libx264", "-preset", "fast",
                    "-c:a", "aac",
                    final_output
                ]
                subprocess.run(fallback_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        except Exception as e:
            logger.error(f"Render engine crashed: {e}. Falling back to raw trim.")
            fallback_used = True
            try:
                if os.path.exists(temp_trimmed):
                    os.rename(temp_trimmed, final_output)
            except Exception as inner_e:
                logger.error(f"Total pipeline failure: {inner_e}")
                return json.dumps({"error": "Total rendering failure"})
        finally:
            if os.path.exists(temp_trimmed) and os.path.exists(final_output) and final_output != temp_trimmed:
                try:
                    os.remove(temp_trimmed)
                except Exception:
                    pass

        logger.info(f"Render successful. Output path: {final_output}")
        return json.dumps({
            "output_path": final_output,
            "fallback_used": fallback_used,
            "duration": clip_duration
        })

def render_short(input_data: Dict[str, Any], preview_mode: bool = False, seed: int = None) -> str:
    engine = EditingEngine(preview_mode=preview_mode, seed=seed)
    return engine.render_short(input_data)
