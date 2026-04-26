from typing import List
from phase3_types import HighlightCandidate

def build(candidates: List[HighlightCandidate], gap_threshold: float = 3.0) -> List[HighlightCandidate]:
    """
    Merges adjacent highlight candidates if they overlap or are within the gap threshold.
    Always emits contiguous windows.
    """
    if not candidates:
        return []

    # Sort by start time
    candidates.sort(key=lambda x: x.start)
    
    merged: List[HighlightCandidate] = []
    if not candidates:
        return merged

    current = candidates[0]

    for next_c in candidates[1:]:
        # If they overlap or are very close
        if next_c.start <= current.end + gap_threshold:
            # Merge
            new_end = max(current.end, next_c.end)
            
            # Update current candidate
            current.end = new_end
            current.score = max(current.score, next_c.score)
            current.events.extend([e for e in next_c.events if e not in current.events])
            # Merge reasons/evidence if needed
            current.reason = f"Combined Story: {current.reason} + {next_c.reason}"
        else:
            merged.append(current)
            current = next_c

    merged.append(current)
    return merged
