import re
from statistics import mean, pstdev

from multimodal_utils import clip01, collect_transcript_text, flatten_transcript_words, get_window_bounds, positive_zscore


HYPE_PHRASES = {
    "omg": 1.0,
    "no way": 1.0,
    "what": 0.5,
    "insane": 1.0,
    "crazy": 0.9,
    "let's go": 1.0,
    "clip that": 1.0,
    "holy": 0.8,
    "bro": 0.6,
    "wait": 0.5,
}


def _keyword_intensity(text):
    lower = text.lower()
    keyword_hits = sum(lower.count(phrase) * weight for phrase, weight in HYPE_PHRASES.items())

    # All-caps words often show excited delivery in gaming transcripts.
    caps_hits = sum(1 for token in re.findall(r"\b[A-Z]{3,}\b", text) if token.isalpha())
    return clip01((keyword_hits + 0.5 * caps_hits) / 3.0)


def _punctuation_intensity(text, word_count):
    excitement_marks = text.count("!") + 0.5 * text.count("?")
    return clip01(excitement_marks / max(1.0, word_count / 4.0))


def analyze_speech_windows(transcript_data, windows):
    """
    Scores every sliding window from transcript-derived excitement signals.
    We combine punctuation spikes, hype-keyword intensity, and WPM lift over the stream baseline.
    """
    words = flatten_transcript_words(transcript_data)
    window_rows = []

    for window in windows:
        start_time, end_time = get_window_bounds(window)
        text = collect_transcript_text(transcript_data, start_time, end_time)
        overlapping_words = [word for word in words if word["end"] > start_time and word["start"] < end_time]
        word_count = len(overlapping_words)
        duration = max(end_time - start_time, 1e-6)
        wpm = (word_count * 60.0) / duration
        window_rows.append(
            {
                "text": text,
                "word_count": word_count,
                "wpm": wpm,
            }
        )

    wpm_values = [row["wpm"] for row in window_rows]
    wpm_mean = mean(wpm_values) if wpm_values else 0.0
    wpm_std = pstdev(wpm_values) if len(wpm_values) > 1 else 0.0

    results = []
    for row in window_rows:
        text = row["text"]
        word_count = row["word_count"]
        punctuation_intensity = _punctuation_intensity(text, word_count)
        keyword_intensity = _keyword_intensity(text)
        speech_rate_spike = positive_zscore(row["wpm"], wpm_mean, wpm_std)

        speech_score = clip01(
            (0.35 * punctuation_intensity)
            + (0.35 * keyword_intensity)
            + (0.30 * speech_rate_spike)
        )

        results.append(
            {
                "text": text,
                "score": speech_score,
                "features": {
                    "exclamation_density": punctuation_intensity,
                    "keyword_intensity": keyword_intensity,
                    "speech_rate_spike": speech_rate_spike,
                    "wpm": row["wpm"],
                    "word_count": float(word_count),
                },
            }
        )

    metadata = {
        "wpm_mean": float(wpm_mean),
        "wpm_std": float(wpm_std),
        "windows_scored": len(results),
    }
    return results, metadata
