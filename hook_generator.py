"""
hook_generator.py — Advanced LLM Hook Generation
================================================
Implements the Hook Generator prompt logic for Short-Form Video.
Generates scroll-stopping hooks based on detected event data.
"""

import json
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

HOOK_GENERATOR_PROMPT = """# 🚀 TASK: Hook Generator (Short-Form Video)

## 🎯 Objective

Generate a **scroll-stopping hook** for a short video based on detected event data.

The hook MUST:

* Grab attention in the **first 1 second**
* Create **curiosity, tension, or emotion**
* Be **short, punchy, and high-energy**
* Match the **actual moment** (no clickbait mismatch)

---

# 🧠 INPUT FORMAT

You will receive structured event data:

{input_json}

---

# ⚙️ TASK

Generate **3 hook variations**.

Each hook must:

* Be **5–10 words max**
* Use **simple, emotional language**
* Be optimized for **YouTube Shorts / Reels / TikTok**
* Avoid generic phrases like:

  * "watch this"
  * "check this out"

---

# 🔥 HOOK STYLE RULES

## 1. Surprise-Based Events
Use: shock, disbelief
Examples: "I WAS NOT READY FOR THIS 😳", "THIS CAUGHT ME OFF GUARD..."

## 2. Combat / Conflict Events
Use: tension, dominance, mistake
Examples: "They picked the WRONG guy 💀", "This fight went very wrong..."

## 3. Reaction / Emotional Events
Use: personal reaction, relatability
Examples: "BRO WHAT JUST HAPPENED 😭", "I LOST IT HERE 😂"

## 4. Funny / Fail Events
Use: humor, exaggeration
Examples: "This made no sense 😂", "Biggest fail ever 💀"

---

# ⚠️ CONSTRAINTS

* ❌ No long sentences
* ❌ No explanation-style text
* ❌ No hashtags
* ❌ No emojis spam (max 1–2)
* ❌ Must match actual event (no fake hype)

---

# 🎯 SCORING LOGIC (IMPORTANT)

Prioritize:
* Curiosity > Accuracy > Style

Hook should make user think:
👉 "Wait... what happened?"

---

# 🧾 OUTPUT FORMAT

Respond ONLY with valid JSON matching this format:
{
  "hooks": [
    "HOOK OPTION 1",
    "HOOK OPTION 2",
    "HOOK OPTION 3"
  ]
}

---

# 🚀 FINAL GOAL

Hooks should:
👉 Stop scrolling instantly
👉 Match the moment perfectly
👉 Increase retention from second 0
"""

class HookGenerator:
    def __init__(self, model: str = "llama3:8b", endpoint: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.endpoint = endpoint

    def generate_hooks(self, event_data: Dict[str, Any]) -> List[str]:
        """
        Generates 3 hook variations based on structured event data.
        
        Expected event_data format:
        {
          "event_type": "reaction | combat | surprise | funny | fail",
          "emotion": "shock | anger | excitement | confusion | none",
          "intensity": 0-10,
          "surprise_score": 0-10,
          "conflict_score": 0-10,
          "payoff_score": 0-10,
          "context": "short description of what happens"
        }
        """
        input_json = json.dumps(event_data, indent=2)
        prompt = HOOK_GENERATOR_PROMPT.replace("{input_json}", input_json)

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
                if "hooks" in parsed and isinstance(parsed["hooks"], list):
                    return parsed["hooks"]
            
            logger.warning(f"Failed to parse hooks from LLM response: {raw}")
            
        except Exception as e:
            logger.error(f"HookGenerator: Error generating hooks: {e}")

        # Fallback
        return ["Wait for it..."]

    def enrich_arc_with_hooks(self, arc: Any, context: str = "") -> Any:
        """
        Helper to map an existing EventMoment/ArcRegion to the expected event_data
        format and populate its hook variations.
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
        
        # Build context from transcript
        transcript = getattr(arc, "transcript", "")
        if transcript and context:
            full_context = f"{context}. Speech: '{transcript}'"
        else:
            full_context = context or transcript or "A compelling gaming moment."

        event_data = {
            "event_type": mapped_type,
            "emotion": "excitement", # Simplification, could be dynamically mapped
            "intensity": min(10, int(getattr(arc, "final_score", 0.5) * 10)),
            "surprise_score": min(10, int(getattr(arc, "surprise_score", 0) * 10)),
            "conflict_score": min(10, int(getattr(arc, "conflict_score", 0) * 10)),
            "payoff_score": min(10, int(getattr(arc, "payoff_score", 0) * 10)),
            "context": full_context
        }

        hooks = self.generate_hooks(event_data)
        
        # Save hooks to the arc object
        if hooks:
            setattr(arc, "hooks", hooks)
            setattr(arc, "hook_sentence", hooks[0])  # Set first hook as primary
            
        return arc
