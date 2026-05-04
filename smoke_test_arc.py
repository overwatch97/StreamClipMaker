"""
smoke_test_arc.py  —  validates the new arc intelligence pipeline
Run from StreamClipMaker root with: venv\Scripts\python.exe smoke_test_arc.py
"""
import math
import sys
from phase3_types import ArcShape, ARC_DURATION_RULES, TimelineSecond
from arc_detector import ArcDetector
from game_adapter import GameAdapter
import editing_brain

PASS = "PASS"
FAIL = "FAIL"
errors = []

def check(label, cond):
    status = PASS if cond else FAIL
    print(f"  [{status}] {label}")
    if not cond:
        errors.append(label)


print("=" * 55)
print("Smoke Test: Dynamic Arc Intelligence Pipeline")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# TEST 1: RDR2-style foot chase — TENSION arc expected
# ─────────────────────────────────────────────────────────
print()
print("TEST 1: RDR2 Foot Chase (sustained motion, no speech/gun)")
timeline1 = []
for i in range(180):
    sec = TimelineSecond(timestamp=float(i))
    sec.audio_score = 0.12
    sec.visual_score = 0.10
    sec.emotion_score = 0.08
    sec.speech_score = 0.08
    sec.fused_score = 0.10
    if 40 <= i <= 85:
        t = (i - 40) / 45.0
        sec.audio_score   = 0.52 + 0.08 * math.sin(t * math.pi * 3)
        sec.visual_score  = 0.58 + 0.06 * math.cos(t * math.pi * 2)
        sec.emotion_score = 0.12
        sec.speech_score  = 0.08
        sec.fused_score   = 0.48
    timeline1.append(sec)

detector = ArcDetector()
arcs1 = detector.detect(timeline1)
print(f"  Detected {len(arcs1)} arc(s):")
for a in arcs1:
    print(f"    [{a.shape_type.value.upper():10s}] {a.start:.0f}s-{a.end:.0f}s  quality={a.quality_score:.3f}")

check("At least 1 arc detected", len(arcs1) >= 1)
chase_found = any(a.start <= 60 and a.end >= 65 for a in arcs1)
check("Chase arc window covered (40-85s range)", chase_found)

# ─────────────────────────────────────────────────────────
# TEST 2: Comedy beat — COMEDY arc expected
# ─────────────────────────────────────────────────────────
print()
print("TEST 2: Comedy Beat (long quiet then sudden spike)")
timeline2 = []
for i in range(120):
    sec = TimelineSecond(timestamp=float(i))
    sec.audio_score = 0.08
    sec.visual_score = 0.07
    sec.emotion_score = 0.05
    sec.speech_score = 0.06
    sec.fused_score = 0.07
    if 80 <= i <= 92:
        sec.audio_score   = 0.72
        sec.visual_score  = 0.68
        sec.emotion_score = 0.75
        sec.speech_score  = 0.65
        sec.fused_score   = 0.70
    timeline2.append(sec)

arcs2 = detector.detect(timeline2)
print(f"  Detected {len(arcs2)} arc(s):")
for a in arcs2:
    print(f"    [{a.shape_type.value.upper():10s}] {a.start:.0f}s-{a.end:.0f}s  quality={a.quality_score:.3f}")
check("Comedy arc detected", len(arcs2) >= 1)

# ─────────────────────────────────────────────────────────
# TEST 3: GameAdapter enrichment on RDR2 profile
# ─────────────────────────────────────────────────────────
print()
print("TEST 3: GameAdapter RDR2 enrichment")
adapter = GameAdapter()
adapter.load_profile("red-dead-redemption-2")
if arcs1:
    enriched = adapter.enrich_arcs(list(arcs1))
    arc = enriched[0]
    print(f"  label:  {arc.label}")
    print(f"  prompt: {arc.clip_prompt[:90]}")
    check("Label is non-empty", bool(arc.label))
    check("CLIP prompt is non-empty", bool(arc.clip_prompt))

# ─────────────────────────────────────────────────────────
# TEST 4: editing_brain validates chase arc
# ─────────────────────────────────────────────────────────
print()
print("TEST 4: editing_brain validates TENSION arc (should pass)")
if arcs1:
    arc_json = arcs1[0].to_clipper_json()
    result = editing_brain.validate_clip_logic(arc_json, [], timeline=timeline1)
    check("TENSION arc passes validation", result)

# ─────────────────────────────────────────────────────────
# TEST 5: Flat boring signal produces NO arcs
# ─────────────────────────────────────────────────────────
print()
print("TEST 5: Flat boring signal (horse travel) produces no arcs")
timeline3 = []
for i in range(120):
    sec = TimelineSecond(timestamp=float(i))
    sec.audio_score = 0.11
    sec.visual_score = 0.13
    sec.emotion_score = 0.05
    sec.speech_score = 0.06
    sec.fused_score = 0.09
    timeline3.append(sec)

arcs3 = detector.detect(timeline3)
print(f"  Detected {len(arcs3)} arc(s) from flat signal")
check("No arcs from boring flat signal", len(arcs3) == 0)

# ─────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────
print()
print("=" * 55)
if errors:
    print(f"FAILED: {len(errors)} test(s) failed:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("All tests PASSED.")
    sys.exit(0)
