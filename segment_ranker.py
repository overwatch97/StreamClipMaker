from scoring_engine import SegmentWindow, build_reason, dominant_modality
from multimodal_utils import clip01, collect_transcript_text, truncate_text


def generate_sliding_windows(duration_secs, window_secs=4.0, stride_secs=1.0):
    if duration_secs <= 0:
        return []

    window_secs = float(window_secs)
    stride_secs = float(stride_secs)
    if duration_secs <= window_secs:
        return [SegmentWindow(index=0, start_time=0.0, end_time=float(duration_secs))]

    windows = []
    index = 0
    start_time = 0.0
    last_start = max(0.0, duration_secs - window_secs)

    while start_time <= last_start + 1e-6:
        end_time = min(duration_secs, start_time + window_secs)
        windows.append(SegmentWindow(index=index, start_time=float(start_time), end_time=float(end_time)))
        index += 1
        start_time += stride_secs

    if windows and abs(windows[-1].start_time - last_start) > 1e-6:
        windows.append(SegmentWindow(index=index, start_time=float(last_start), end_time=float(duration_secs)))

    return windows


def _shape_highlight_range(group, all_segments, video_duration):
    anchor = max(group, key=lambda segment: segment.scores.total)
    peak_time = anchor.window.peak_time

    start_time = peak_time - 3.0
    end_time = peak_time + 3.5

    left_neighborhood = [
        segment for segment in all_segments if (peak_time - 1.5) <= segment.window.peak_time < peak_time
    ]
    right_neighborhood = [
        segment for segment in all_segments if peak_time < segment.window.peak_time <= (peak_time + 1.5)
    ]
    if left_neighborhood:
        valley = min(left_neighborhood, key=lambda segment: segment.scores.total)
        start_time = min(start_time, valley.window.start_time)
    if right_neighborhood:
        valley = min(right_neighborhood, key=lambda segment: segment.scores.total)
        end_time = max(end_time, valley.window.end_time)

    start_time = min(start_time, min(segment.window.start_time for segment in group))
    end_time = max(end_time, max(segment.window.end_time for segment in group))

    start_time = max(0.0, start_time)
    end_time = min(float(video_duration), end_time)

    duration = end_time - start_time
    if duration < 8.0:
        deficit = 8.0 - duration
        start_time = max(0.0, start_time - (deficit / 2.0))
        end_time = min(float(video_duration), end_time + (deficit / 2.0))

    if (end_time - start_time) > 14.0:
        start_time = max(0.0, peak_time - 6.5)
        end_time = min(float(video_duration), peak_time + 7.5)

    if (end_time - start_time) < 8.0:
        # Final clamp if we were limited by stream boundaries.
        end_time = min(float(video_duration), start_time + 8.0)
        start_time = max(0.0, end_time - 8.0)

    return round(start_time, 4), round(end_time, 4)


def _overlaps(existing_ranges, start_time, end_time):
    for existing_start, existing_end in existing_ranges:
        if end_time > existing_start and start_time < existing_end:
            return True
    return False


def select_top_highlights(segment_results, transcript_data, video_duration, top_k=6, min_separation_secs=6.0):
    ranked = sorted(segment_results, key=lambda segment: segment.scores.total, reverse=True)

    chosen_peaks = []
    for candidate in ranked:
        if all(abs(candidate.window.peak_time - existing.window.peak_time) >= min_separation_secs for existing in chosen_peaks):
            chosen_peaks.append(candidate)
        if len(chosen_peaks) >= top_k:
            break

    chosen_peaks.sort(key=lambda segment: segment.window.peak_time)

    merged_groups = []
    for candidate in chosen_peaks:
        if not merged_groups:
            merged_groups.append([candidate])
            continue

        previous_group = merged_groups[-1]
        previous_anchor = max(previous_group, key=lambda segment: segment.scores.total)
        merged_duration = candidate.window.end_time - previous_group[0].window.start_time
        if (candidate.window.peak_time - previous_anchor.window.peak_time) <= 2.0 and merged_duration <= 14.0:
            previous_group.append(candidate)
        else:
            merged_groups.append([candidate])

    candidate_highlights = []
    for group in merged_groups:
        anchor = max(group, key=lambda segment: segment.scores.total)
        start_time, end_time = _shape_highlight_range(group, segment_results, video_duration)
        highlight_text = truncate_text(collect_transcript_text(transcript_data, start_time, end_time) or anchor.text, 400)
        candidate_highlights.append(
            {
                "start": start_time,
                "end": end_time,
                "peak_time": round(anchor.window.peak_time, 4),
                "category": dominant_modality(anchor.scores),
                "score": int(round(anchor.scores.total * 100)),
                "reason": build_reason(anchor.scores),
                "text": highlight_text,
                "source_window_start": round(anchor.window.start_time, 4),
                "source_window_end": round(anchor.window.end_time, 4),
                "scores": anchor.scores.as_dict(),
            }
        )

    candidate_highlights.sort(key=lambda item: item["score"], reverse=True)

    accepted = []
    accepted_ranges = []
    for highlight in candidate_highlights:
        if _overlaps(accepted_ranges, highlight["start"], highlight["end"]):
            continue
        accepted.append(highlight)
        accepted_ranges.append((highlight["start"], highlight["end"]))

    return accepted
