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
* Reference **context, tension, or expectation** — NOT just raw emotion
* Be **short, conversational, and specific to this moment**
* Feel like it was written by a **human editor**, not an AI hype generator

---

# 🧠 INPUT FORMAT

You will receive structured event data including event type, intensity, and context:

{input_json}

The `context` field is the most important — use it to make the hook SPECIFIC.

---

# ⚙️ TASK

Generate **3 hook variations** at different narrative styles.

Each hook must:

* Be **5–12 words max**
* Reference the **setup, tension, or expectation** of this moment
* Feel like a **human thought** — conversational, not shouty
* Be optimized for **YouTube Shorts / Reels / TikTok** retention

---

# 🔥 HOOK STYLE RULES

## Priority Order (CRITICAL):

```
Curiosity + Context  >  Emotion  >  Generic Shock
```

The goal is NOT to make the viewer feel an emotion.
The goal is to make the viewer NEED to see what happens next.

---

## ✅ GOOD Hook Styles (USE THESE):

### 1. Expectation Subversion
Set up what SHOULD have happened, then imply it didn't.
Examples:
- "I thought this guy was friendly…"
- "This was supposed to be an easy fight"
- "I had no idea what I was walking into"

### 2. Narrative Tension
Describe the setup — NOT the payoff.
Examples:
- "This fight escalated WAY too fast"
- "Something felt wrong the moment I entered"
- "They had no idea I was watching"

### 3. Reluctant Reveal
Imply regret, surprise, or inevitability.
Examples:
- "I should've just walked away…"
- "This is where it all went wrong"
- "I didn't see that coming at all"

### 4. Understatement (powerful when used correctly)
Describe the moment with deliberate calm.
Examples:
- "Things got complicated"
- "That escalated quickly"
- "This did not go to plan"

---

## ❌ BAD Hook Styles (NEVER USE THESE):

- "THIS WAS INSANE 😳"
- "BRO WHAT 😭"
- "BROOOOO 😭😭"
- "YOU WON'T BELIEVE THIS"
- "THIS WAS CRAZY 😳"
- "OMG 😱"
- Any all-caps single-emotion shout
- Any hook that could apply to ANY video (not specific to this moment)

---

# ⚠️ CONSTRAINTS

* ❌ No hashtags
* ❌ Max 1 emoji (optional — only if it adds meaning)
* ❌ No all-caps shouting
* ❌ Must reference the actual context field, not just the event_type
* ✅ Prefer lowercase or title case — feels more human
* ✅ Trailing ellipsis (…) is encouraged for tension

---

# 🎯 SCORING LOGIC (IMPORTANT)

The best hook makes the viewer think:
👉 "Wait — what happened before this?"
👉 "Why did that happen?"
👉 "I need to see how this plays out"

NOT:
👉 "wow that sounds hype" (generic)

---

# 🧾 OUTPUT FORMAT

Respond ONLY with valid JSON matching this format:
{
  "hooks": [
    "HOOK OPTION 1",
    "HOOK OPTION 2",
    "HOOK OPTION 3"
  ],
  "hook_style": "contextual"
}

The `hook_style` field must always be "contextual".

---

# 🚀 FINAL GOAL

Hooks should:
👉 Stop scrolling by creating CURIOSITY, not just shock
👉 Feel like a human editor wrote them
👉 Be specific to THIS moment — not any moment
"""

class HookGenerator:
    def __init__(self, model: str = "llama3:8b", endpoint: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.endpoint = endpoint

    def generate_hooks(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates 3 hook variations based on structured event data.
        Returns a dict with 'hooks' (list) and 'hook_style' (str) keys.

        Expected event_data format:
        {
          "event_type": "reaction | combat | surprise | funny | fail",
          "emotion": "shock | anger | excitement | confusion | none",
          "intensity": 0-10,
          "surprise_score": 0-10,
          "conflict_score": 0-10,
          "payoff_score": 0-10,
          "context": "short description of what happens"  ← most important field
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
                    return {
                        "hooks": parsed["hooks"],
                        "hook_style": parsed.get("hook_style", "contextual"),
                    }

            logger.warning(f"Failed to parse hooks from LLM response: {raw}")

        except Exception as e:
            logger.error(f"HookGenerator: Error generating hooks: {e}")

        # Fallback — minimal generic, but still better than shouting
        return {"hooks": ["This did not go to plan…"], "hook_style": "fallback"}

    def enrich_arc_with_hooks(self, arc: Any, context: str = "") -> Any:
        """
        Helper to map an existing EventMoment/ArcRegion to the expected event_data
        format and populate its hook variations.
        Passes rich context so the LLM can generate specific, curiosity-driven hooks.
        """
        # Map arc shape to event_type
        shape = getattr(arc, "event_type", getattr(arc, "shape_type", "neutral"))
        if hasattr(shape, "value"):
            shape = shape.value

        event_type_mapping = {
            "spike":     "surprise",
            "tension":   "combat",
            "comedy":    "funny",
            "drama":     "reaction",
            "triumph":   "combat",
            "discovery": "surprise",
        }
        mapped_type = event_type_mapping.get(shape, "reaction")

        # Build the richest possible context string for the LLM
        # Context specificity is the single biggest driver of hook quality.
        transcript = getattr(arc, "transcript", "")
        label      = getattr(arc, "label", "")
        scene_type = getattr(arc, "scene_type", "")

        context_parts = []
        if context:
            context_parts.append(context)
        if label:
            context_parts.append(f"moment label: {label}")
        if scene_type and scene_type not in ("neutral", ""):
            context_parts.append(f"scene type: {scene_type}")
        if transcript:
            context_parts.append(f"streamer said: '{transcript[:120]}'")
        full_context = ". ".join(context_parts) if context_parts else "A compelling gaming moment."

        # Derive emotion from features if available
        features = getattr(arc, "features", {})
        emotion_score = features.get("emotion_score", 0.0)
        if emotion_score > 0.7:
            emotion = "shock"
        elif emotion_score > 0.4:
            emotion = "excitement"
        else:
            emotion = "focus"

        event_data = {
            "event_type":    mapped_type,
            "emotion":       emotion,
            "intensity":     min(10, int(getattr(arc, "final_score", 0.5) * 10)),
            "surprise_score":min(10, int(getattr(arc, "surprise_score", 0) * 10)),
            "conflict_score":min(10, int(getattr(arc, "conflict_score", 0) * 10)),
            "payoff_score":  min(10, int(getattr(arc, "payoff_score", 0) * 10)),
            "context":       full_context,
        }

        result = self.generate_hooks(event_data)
        hooks      = result.get("hooks", ["This did not go to plan…"])
        hook_style = result.get("hook_style", "contextual")

        # Save hooks and metadata to the arc object
        if hooks:
            setattr(arc, "hooks",        hooks)
            setattr(arc, "hook_sentence", hooks[0])   # primary hook
            setattr(arc, "hook_style",   hook_style)  # for debug metadata

        return arc
