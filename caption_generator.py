"""
caption_generator.py — Dynamic LLM Caption Generation
=====================================================
Implements the Dynamic Caption Generator prompt logic for Short-Form Video.
Generates time-synced, emotionally expressive captions based on event data.
"""

import json
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

CAPTION_GENERATOR_PROMPT = """# 🚀 TASK: Dynamic Caption Generator (Short-Form Video)

## 🎯 Objective

Generate **high-engagement, time-synced captions** for a short video clip based on detected event data.

Captions MUST:

* Increase **watch time and retention**
* Reinforce **emotion and reaction**
* Be **short, punchy, and dynamic**
* Feel like a **live reaction**, not subtitles

---

# 🧠 INPUT FORMAT

You will receive structured event data:

{input_json}

---

# ⚙️ TASK

Generate a **timeline of 3–5 captions** placed strategically across the clip.

Each caption must:

* Be **2–6 words max**
* Be emotionally expressive
* Match the moment timing
* Avoid full sentences

---

# ⏱ TIMING RULES (VERY IMPORTANT)

## 1. Start (0–1 sec)
👉 Build curiosity
Examples:
* "Wait for it..."
* "Something's off..."
* "This felt wrong..."

## 2. Pre-Peak (~peak_time - 1 sec)
👉 Build tension
Examples:
* "Oh no..."
* "Here it comes..."
* "Bad idea..."

## 3. Peak (peak_time)
👉 Strong reaction (MOST IMPORTANT)
Examples:
* "BRO WHAT 😭"
* "NO WAY 💀"
* "WHAT JUST HAPPENED 😳"

## 4. Post-Peak (~peak_time + 1 sec)
👉 Payoff / reaction
Examples:
* "I'm done 💀"
* "That's insane"
* "No chance..."

## 5. Optional Ending
👉 Reinforce emotion
Examples:
* "This is wild"
* "I can't believe it"

---

# 🔥 CAPTION STYLE RULES

* Use **caps for impact** (but not always)
* Max **1 emoji per caption**
* Keep it **raw and relatable**
* No punctuation-heavy sentences
* Avoid subtitles like:
  ❌ "He shoots the enemy"
  ❌ "Then this happens"

---

# 🧠 EVENT-BASED STYLE ADJUSTMENT

## Surprise:
* Shock-heavy
* "WHAT", "NO WAY"

## Combat:
* Tension + dominance
* "BIG mistake", "He's done"

## Funny:
* Humor + exaggeration
* "I'm done 😂", "Ain't no way"

## Reaction:
* Personal emotion
* "BRO...", "WHY 😭"

---

# ⚠️ CONSTRAINTS

* ❌ No long sentences
* ❌ No narration
* ❌ No hashtags
* ❌ No repeating same caption
* ❌ No generic filler ("nice", "cool")

---

# 🧾 OUTPUT FORMAT

Respond ONLY with valid JSON matching this format:
{
  "captions": [
    {"time": 0.2, "text": "Wait for it..."},
    {"time": 2.0, "text": "Oh no..."},
    {"time": 3.5, "text": "BRO WHAT 😭"},
    {"time": 4.8, "text": "I'm done 💀"}
  ]
}

---

# 🎯 PRIORITY LOGIC

Captions should optimize for:
👉 Emotion > Timing > Accuracy

User should feel:
👉 "I want to see what happens next"

---

# 🚀 FINAL GOAL

Captions should:

* Keep viewers watching till the end
* Amplify emotion
* Make the clip feel alive
"""

class CaptionGenerator:
    def __init__(self, model: str = "llama3:8b", endpoint: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.endpoint = endpoint

    def generate_captions(self, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates 3-5 time-synced captions based on structured event data.
        
        Expected event_data format:
        {
          "event_type": "reaction | combat | surprise | funny | fail",
          "emotion": "shock | anger | excitement | confusion | none",
          "intensity": 0-10,
          "peak_time": float,
          "duration": float,
          "transcript": "optional spoken words",
          "context": "short description of what happens"
        }
        
        Returns a list of dicts like: [{"time": 0.2, "text": "..."}, ...]
        """
        input_json = json.dumps(event_data, indent=2)
        prompt = CAPTION_GENERATOR_PROMPT.replace("{input_json}", input_json)

        try:
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=30,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()

            # Extract JSON from the response
            match_start = raw.find("{")
            match_end = raw.rfind("}") + 1
            if match_start != -1 and match_end > match_start:
                parsed = json.loads(raw[match_start:match_end])
                if "captions" in parsed and isinstance(parsed["captions"], list):
                    return parsed["captions"]
            
            logger.warning(f"Failed to parse captions from LLM response: {raw}")
            
        except Exception as e:
            logger.error(f"CaptionGenerator: Error generating captions: {e}")

        # Fallback
        return [
            {"time": 0.5, "text": "Wait for it..."},
            {"time": max(1.0, event_data.get("peak_time", 2.0)), "text": "BRO NO WAY 💀"}
        ]

    def enrich_arc_with_captions(self, arc: Any, context: str = "") -> Any:
        """
        Helper to map an existing EventMoment/ArcRegion to the expected event_data
        format and attach dynamic captions.
        """
        # Map arc shape to event_type
        shape = getattr(arc, "event_type", getattr(arc, "shape_type", "neutral"))
        if hasattr(shape, "value"):
            shape = shape.value
            
        event_type_mapping = {
            "spike": "surprise",
            "tension": "combat",
            "comedy": "funny",
            "drama": "reaction",
            "triumph": "combat",
            "discovery": "surprise"
        }
        mapped_type = event_type_mapping.get(shape, "reaction")
        
        transcript = getattr(arc, "transcript", "")
        duration = getattr(arc, "duration", 10.0)
        
        # Calculate local peak time relative to clip start if possible
        # Otherwise use absolute peak_time
        start_time = getattr(arc, "start", 0.0)
        abs_peak = getattr(arc, "peak_time", start_time + duration / 2)
        local_peak_time = max(0.1, abs_peak - start_time)

        event_data = {
            "event_type": mapped_type,
            "emotion": "shock" if mapped_type in ["surprise", "funny"] else "excitement",
            "intensity": min(10, int(getattr(arc, "final_score", 0.5) * 10)),
            "peak_time": round(local_peak_time, 2),
            "duration": round(duration, 2),
            "transcript": transcript,
            "context": context or "An intense gaming moment."
        }

        captions = self.generate_captions(event_data)
        
        # Save dynamic captions to the arc object
        if captions:
            setattr(arc, "dynamic_captions", captions)
            
        return arc
