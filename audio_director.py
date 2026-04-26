import os
from typing import List, Dict, Optional

def detect_beat_grid(audio_path) -> List[float]:
    """
    Detects beat timestamps in the gameplay audio to allow music syncing.
    """
    try:
        import librosa
        y, sr = librosa.load(audio_path, duration=60) # Only sample first min for tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        return beat_times.tolist()
    except Exception:
        return []

def generate_audio_mix_filter(
    game_audio_label: str,
    music_audio_label: Optional[str],
    ducking_levels: List[Dict],
    duration: float,
    facecam_audio_label: Optional[str] = None
) -> str:
    """
    Builds a complex FFmpeg filter string for professional mixing.
    - Ducking: Music volume drops when speech is detected.
    - Normalization: -14 LUFS target.
    """
    filter_graph = ""
    
    # 1. Process base vocals (Facecam + Gameplay if facecam exists, else just Gameplay)
    if facecam_audio_label:
        # Boost facecam vocals (1.5) and slightly lower game audio (0.7)
        filter_graph += f"{facecam_audio_label}volume=1.5[facev]; "
        filter_graph += f"{game_audio_label}volume=0.7[gamea]; "
        filter_graph += f"[facev][gamea]amix=inputs=2:dropout_transition=2[vocals]; "
    else:
        # Fallback to just normalized game audio as the vocal source
        filter_graph += f"{game_audio_label}volume=1.2[vocals]; "

    # 2. Add Music if available
    if not music_audio_label:
        return f"{filter_graph}[vocals]loudnorm=I=-14:LRA=11:tp=-1[aout]"
        
    # Example logic: Duck music to 0.15 during speech (from ducking_levels)
    duck_conditions = []
    for level in ducking_levels:
        if level["is_speech"]:
            duck_conditions.append(f"between(t,{level['start']},{level['end']})")
            
    if duck_conditions:
        duck_filter = f"volume='if({'+'.join(duck_conditions)}, 0.15, 0.45)':eval=frame"
    else:
        duck_filter = "volume=0.45"
        
    filter_graph += (
        f"{music_audio_label}{duck_filter}[bg_music]; "
        f"[vocals][bg_music]amix=inputs=2:duration=first:dropout_transition=2[mixed]; "
        f"[mixed]loudnorm=I=-14:LRA=11:tp=-1[aout]"
    )
    return filter_graph

def get_royalty_free_music_manifest(assets_dir: str) -> List[Dict]:
    """
    Scans assets/music for available tracks.
    """
    music_dir = os.path.join(assets_dir, "music")
    if not os.path.exists(music_dir):
        return []
        
    tracks = []
    for f in os.listdir(music_dir):
        if f.endswith(('.mp3', '.wav', '.m4a')):
            tracks.append({
                "path": os.path.join(music_dir, f),
                "name": f,
                "mood": "dynamic" # Placeholder for future AI mood detection
            })
    return tracks
