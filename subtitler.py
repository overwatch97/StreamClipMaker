import json
import os

def format_time_ass(seconds):
    """Formats time as h:mm:ss.cs required for ASS subtitles."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def generate_ass_subtitle(transcript_path, clip_start, clip_end, output_ass_path):
    """
    Reads the global transcript and extracts the precise words spoken during the clip.
    Generates a gaming-themed vertical .ass subtitle file.
    """
    print(f"Generating subtitles for clip: {clip_start}s to {clip_end}s...")
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    words_in_clip = []
    for segment in data:
        for w in segment.get("words", []):
            if w["end"] > clip_start and w["start"] < clip_end:
                # adjust times relative to the clip start
                rel_start = max(0.0, w["start"] - clip_start)
                rel_end = w["end"] - clip_start
                word = w["word"].strip()
                if word:
                    words_in_clip.append({
                        "word": word,
                        "start": rel_start,
                        "end": rel_end
                    })
                    
    # Professional "Pop-in" Subtitles
    # Instead of chunks, we create a short event for every word with an animation transform
    ass_dialogue_lines = []
    for i, w in enumerate(words_in_clip):
        word_text = w["word"].strip().upper()
        if not word_text:
            continue
            
        w_start = w["start"]
        w_end = w["end"]
        
        # Ensure minimal duration for visibility
        if w_end - w_start < 0.2:
            w_end = w_start + 0.2
            
        # Optional: Peek at next word to fill gap
        if i + 1 < len(words_in_clip):
            next_start = words_in_clip[i+1]["start"]
            if next_start > w_end and next_start - w_end < 0.3:
                w_end = next_start
        
        # ASS Animation Tag: Scale from 80% to 110% then back to 100%
        # \t(start, end, acc, tags)
        # We also change color to Yellow {\c&H00FFFF&} for the active word
        animation = r"{\fscx80\fscy80\t(0,100,1,\fscx120\fscy120)\t(100,200,1,\fscx100\fscy100)}"
        color_active = r"{\c&H00FFFF&}" # Yellow spotlight
        
        line = f"Dialogue: 0,{format_time_ass(w_start)},{format_time_ass(w_end)},Default,,0,0,0,,{animation}{color_active}{word_text}"
        ass_dialogue_lines.append(line)

    ass_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 1",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,110,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,12,6,5,10,10,10,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]
    
    ass_lines.extend(ass_dialogue_lines)
        
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_ass_path), exist_ok=True)
    
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ass_lines))
        
    print(f"Subtitles generated successfully at {output_ass_path}")
    return output_ass_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 4:
        generate_ass_subtitle(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
