import numpy as np
import logging

logger = logging.getLogger(__name__)

def require_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required for StreamClipMaker/pAi audio analysis. Install it in the venv.") from exc
    return librosa

class AudioSignalExtractor:
    """
    Extracts stream-wide audio feature series using librosa, caching arrays internally
    and exposing timestamps slice retrieval.
    """
    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self.librosa = require_librosa()
        self.is_loaded = False
        
        # Extracted signal arrays
        self.times = np.array([])
        self.rms_norm = np.array([])
        self.sound_change_norm = np.array([])
        self.pitch_arousal_norm = np.array([])
        self.screech_score_norm = np.array([])  # Tire screech detection
        self.duration = 0.0

    def load_and_analyze(self):
        """Loads and precomputes all audio curves for the VOD."""
        if self.is_loaded:
            return
            
        logger.info(f"AudioSignalExtractor: Precomputing audio signals from {self.audio_path}...")
        y, sr = self.librosa.load(self.audio_path, sr=None, mono=True)
        if y.size == 0:
            self.is_loaded = True
            return

        self.duration = float(len(y) / sr)
        frame_length = 2048
        hop_length = 512

        # 1. Compute RMS energy
        rms = self.librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        self.times = self.librosa.times_like(rms, sr=sr, hop_length=hop_length)

        # 2. Compute Onset Strength & Sound Change
        onset_strength = self.librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        rms_delta = np.abs(np.diff(rms, prepend=rms[0]))
        sound_change = np.maximum(rms_delta, onset_strength)

        # 3. Pitch Tracking
        pitch = self.librosa.yin(
            y,
            fmin=self.librosa.note_to_hz("C2"),
            fmax=self.librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        
        finite_pitch = np.isfinite(pitch) & (pitch > 0)
        if finite_pitch.any():
            pitch_filled = pitch.copy()
            pitch_filled[~finite_pitch] = float(np.median(pitch[finite_pitch]))
        else:
            pitch_filled = np.zeros_like(pitch, dtype=float)
            
        pitch_baseline = float(np.median(pitch_filled[finite_pitch])) if finite_pitch.any() else 0.0
        pitch_arousal = np.maximum(0.0, pitch_filled - pitch_baseline)

        # 4. Tire Screech Detection (Sustained energy in high frequency band 2kHz - 5kHz)
        # Compute Spectrogram to measure energy in specific bands
        stft = np.abs(self.librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
        freqs = self.librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        
        # Indices corresponding to 2kHz - 5kHz
        screech_band_idx = np.where((freqs >= 2000) & (freqs <= 5000))[0]
        if len(screech_band_idx) > 0:
            screech_energy = np.mean(stft[screech_band_idx, :], axis=0)
        else:
            screech_energy = np.zeros_like(rms)

        # Normalization helper
        def robust_normalize(data, lower_percentile=50.0, upper_percentile=97.0):
            data_arr = np.array(data, dtype=float)
            if len(data_arr) == 0:
                return data_arr
            d_min = float(np.percentile(data_arr, lower_percentile))
            d_max = float(np.percentile(data_arr, upper_percentile))
            if d_max - d_min <= 1e-6:
                return np.zeros_like(data_arr)
            return np.clip((data_arr - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)

        # Normalize features
        self.rms_norm = robust_normalize(rms, 55.0, 97.0)
        self.sound_change_norm = robust_normalize(sound_change, 50.0, 97.0)
        self.pitch_arousal_norm = robust_normalize(pitch_arousal, 50.0, 97.0)
        self.screech_score_norm = robust_normalize(screech_energy, 60.0, 98.0)

        self.is_loaded = True
        logger.info("AudioSignalExtractor: Precomputation complete.")

    def get_features_at(self, timestamp: float, window_size: float = 1.0) -> dict:
        """Returns aggregated features for the window around timestamp."""
        if not self.is_loaded:
            self.load_and_analyze()

        if len(self.times) == 0:
            return {"rms": 0.0, "sound_change": 0.0, "pitch_arousal": 0.0, "screech": 0.0}

        start = max(0.0, timestamp - window_size / 2.0)
        end = timestamp + window_size / 2.0
        
        indices = np.where((self.times >= start) & (self.times <= end))[0]
        if len(indices) == 0:
            # Fallback to closest frame
            closest = np.argmin(np.abs(self.times - timestamp))
            indices = [closest]

        return {
            "rms": float(np.max(self.rms_norm[indices])),
            "sound_change": float(np.max(self.sound_change_norm[indices])),
            "pitch_arousal": float(np.mean(self.pitch_arousal_norm[indices])),
            "screech": float(np.max(self.screech_score_norm[indices]))
        }
