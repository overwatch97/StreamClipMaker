import subprocess
import os

def extract_audio(video_path, output_audio_path="temp_audio.wav"):
    """
    Extracts mono, 16kHz audio from a video using FFmpeg.
    This format is ideal for whisper transcription and loudness analysis.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    print(f"Extracting audio from {video_path}...")
    
    # We use subprocess to call FFmpeg directly
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",          # Disable video
        "-ac", "1",     # Mono audio
        "-ar", "16000", # 16 kHz sample rate (good for whisper)
        "-y",           # Overwrite output file
        output_audio_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Audio successfully extracted to {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        extract_audio(sys.argv[1])
