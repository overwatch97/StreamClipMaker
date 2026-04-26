import logging
import torch
import numpy as np
from typing import List, Dict, Any
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from phase3_types import GameProfile, TimelineSecond, DetectedEvent

logger = logging.getLogger(__name__)

class GameAwareDetector:
    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        logger.info(f"Initializing GameAwareDetector on {self.device}")
        
        # Load CLIP for zero-shot visual event detection
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            self.clip_model = None

        # Placeholder for YAMNet-style audio tagging
        # In a full release, this would load a pretrained audio classifier.
        self.audio_classifier = None 

    def _resolve_device(self, device: str) -> str:
        if device == "gpu": return "cuda"
        if device == "cpu": return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def detect(self, timeline: List[TimelineSecond], profile: GameProfile, 
               frame_samples: Dict[float, Image.Image]) -> List[DetectedEvent]:
        """
        Runs per-second detection of game events defined in the profile.
        Fuses visual (CLIP/YOLO), audio (Tagging), and manual profile weights.
        """
        events = []
        
        # Extract event prompts from profile
        visual_prompts = [rule["visual_prompt"] for rule in profile.event_rules if "visual_prompt" in rule]
        if not visual_prompts:
            visual_prompts = ["a high energy gaming moment"]

        # 1. Batch Visual Detection (CLIP)
        clip_scores = self._score_visual_events(frame_samples, visual_prompts)

        # 2. Iterate through timeline and fuse
        for second in timeline:
            ts = second.timestamp
            
            # Check for ignore states first
            if self._is_ignore_state(second, profile):
                second.is_ignore_state = True
                second.ignore_reason = "Profile ignore state detected"
                continue

            # Event Detection Logic
            for rule in profile.event_rules:
                event_name = rule["event"]
                min_score = rule.get("min_score", 0.5)
                
                # Visual Evidence
                v_prompt = rule.get("visual_prompt")
                v_score = clip_scores.get(ts, {}).get(v_prompt, 0.0) if v_prompt else 0.0
                
                # Audio Evidence (using base audio/speech signals + tagging)
                a_tag = rule.get("audio_tag")
                # Simple heuristic: high audio spike + tag similarity (tagged as 1.0 for now if spike is high)
                a_score = second.audio_score if a_tag else 0.0
                
                # Fusion
                # Weighting: 60% visual, 40% audio for specific events
                fused_event_score = (v_score * 0.6) + (a_score * 0.4)
                
                if fused_event_score >= min_score:
                    event = DetectedEvent(
                        event_type=event_name,
                        timestamp=ts,
                        score=fused_event_score,
                        evidence={
                            "visual_clip": v_score,
                            "audio_spike": a_score,
                            "visual_motion": second.visual_score,
                            "emotion": second.emotion_score
                        }
                    )
                    events.append(event)
                    second.detect_scores[event_name] = fused_event_score

            # Final Fused Score for the Second
            # Combines base multimodal signals with detected event peaks
            base_fused = (
                (second.speech_score * profile.score_weights.get("speech", 0.25)) +
                (second.audio_score * profile.score_weights.get("audio", 0.25)) +
                (second.visual_score * profile.score_weights.get("visual", 0.25)) +
                (second.emotion_score * profile.score_weights.get("emotion", 0.25))
            )
            
            event_bonus = max(second.detect_scores.values()) if second.detect_scores else 0.0
            second.fused_score = np.clip(base_fused + (event_bonus * 0.3), 0.0, 1.0)

        return events

    def _score_visual_events(self, frame_samples: Dict[float, Image.Image], 
                             prompts: List[str]) -> Dict[float, Dict[str, float]]:
        """
        Uses CLIP to score frames against event prompts in batch.
        """
        if not self.clip_model or not frame_samples:
            return {}

        results = {}
        # Simple implementation for now: process one by one or in small batches
        # In a production environment, we'd batch these for GPU efficiency.
        for ts, img in frame_samples.items():
            try:
                inputs = self.clip_processor(text=prompts, images=img, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                
                # Softmax across prompts to get relative scores
                probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
                results[ts] = {p: float(s) for p, s in zip(prompts, probs)}
            except Exception as e:
                logger.warning(f"Failed to score frame at {ts}: {e}")
                results[ts] = {}
        
        return results

    def _is_ignore_state(self, second: TimelineSecond, profile: GameProfile) -> bool:
        # Check against simple ignore state triggers
        # e.g., very low visual motion + very low audio for 3+ seconds (loading)
        # OR specific detections like "Loading Screen" text in future versions
        if not profile.ignore_states:
            return False
            
        # Hardcoded heuristics for generic ignore states
        if second.visual_score < 0.05 and second.audio_score < 0.05:
            return True
            
        return False
