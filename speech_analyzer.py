import re
import numpy as np

from multimodal_utils import clip01, collect_transcript_text, flatten_transcript_words, get_window_bounds

HYPE_PHRASES = {
    # Shock (1.0)
    "what": 1.0,
    "wtf": 1.0,
    "omg": 1.0,
    
    # Reaction (0.8)
    "no way": 0.8,
    "holy": 0.8,
    "crazy": 0.8,
    "insane": 0.8,
    "bro": 0.8,
    "let's go": 0.8,
    "clip that": 0.8,
    "wait": 0.8,
}

def _keyword_weight(text):
    lower = text.lower()
    best_weight = 0.2  # Normal generic speech baseline
    
    for phrase, weight in HYPE_PHRASES.items():
        if phrase in lower:
            best_weight = max(best_weight, weight)
            
    # All-caps words often show excited delivery
    caps_hits = sum(1 for token in re.findall(r"\b[A-Z]{3,}\b", text) if token.isalpha())
    if caps_hits > 0:
        best_weight = max(best_weight, 0.8)
        
    return float(best_weight)

def _calculate_pauses(words):
    pause_duration = 0.0
    if len(words) < 2:
        return 0.0
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i-1]["end"]
        if gap > 0.5:
            pause_duration += gap
    return pause_duration

def min_max_normalize(data):
    data_arr = np.array(data, dtype=float)
    if len(data_arr) == 0:
        return data_arr
    d_min = np.min(data_arr)
    d_max = np.max(data_arr)
    if d_max - d_min <= 1e-6:
        return np.zeros_like(data_arr)
    return (data_arr - d_min) / (d_max - d_min)

def analyze_speech_windows(transcript_data, windows):
    words = flatten_transcript_words(transcript_data)
    window_rows = []

    for window in windows:
        start_time, end_time = get_window_bounds(window)
        text = collect_transcript_text(transcript_data, start_time, end_time)
        overlapping_words = [w for w in words if w["end"] > start_time and w["start"] < end_time]
        word_count = len(overlapping_words)
        duration = max(end_time - start_time, 1e-6)
        
        # Speaking speed (words per second)
        wps = word_count / duration
        pauses = _calculate_pauses(overlapping_words)
        kw_weight = _keyword_weight(text) if text.strip() else 0.0

        window_rows.append(
            {
                "text": text,
                "wps": wps,
                "pauses": pauses,
                "kw_weight": kw_weight,
            }
        )

    # Per-video normalization for speech energy (wps)
    wps_values = [r["wps"] for r in window_rows]
    wps_norm = min_max_normalize(wps_values)

    results = []
    for i, row in enumerate(window_rows):
        # We also maintain 'score' for legacy pipeline compatibility, but the fusion engine
        # will primarily use the raw normalized features.
        speech_energy_norm = float(wps_norm[i])
        kw_weight = row["kw_weight"]
        
        # Legacy speech score blend
        legacy_score = clip01((0.5 * speech_energy_norm) + (0.5 * kw_weight))

        results.append(
            {
                "text": row["text"],
                "score": legacy_score,
                "features": {
                    "speech_energy_norm": speech_energy_norm,
                    "keyword_weight": kw_weight,
                    "pause_duration": float(row["pauses"]),
                },
            }
        )

    metadata = {
        "windows_scored": len(results),
    }
    return results, metadata
