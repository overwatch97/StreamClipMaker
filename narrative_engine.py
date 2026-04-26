import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from multimodal_utils import collect_transcript_text

@dataclass
class Scene:
    start_time: float
    end_time: float
    avg_score: float
    dominant_modality: str
    transcript: str
    windows: List[int] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    summary: str = ""

def detect_scenes(segment_results, threshold=0.3, max_scene_duration=60.0) -> List[Scene]:
    """
    Groups contiguous windows into Scenes based on score stability and modality.
    Includes a max_scene_duration safety cap to ensure segments are auto-cut properly.
    """
    scenes = []
    if not segment_results:
        return scenes

    current_windows = [segment_results[0]]
    
    for i in range(1, len(segment_results)):
        prev = segment_results[i-1]
        curr = segment_results[i]
        
        # Simple heuristic: if score drops significantly or modality changes, split scene
        score_diff = abs(curr.scores.total - prev.scores.total)
        
        # Calculate current scene duration
        current_duration = curr.window.end_time - current_windows[0].window.start_time
        
        if score_diff > threshold or current_duration > max_scene_duration:
            # Finalize current scene
            scenes.append(_build_scene(current_windows))
            current_windows = [curr]
        else:
            current_windows.append(curr)
            
    if current_windows:
        scenes.append(_build_scene(current_windows))
        
    return scenes

def _build_scene(windows) -> Scene:
    start = windows[0].window.start_time
    end = windows[-1].window.end_time
    avg_score = sum(w.scores.total for w in windows) / len(windows)
    
    # Simple transcript collection
    text = " ".join([w.text for w in windows if w.text])
    
    return Scene(
        start_time=start,
        end_time=end,
        avg_score=avg_score,
        dominant_modality="mixed", # TODO: logic from scoring_engine
        transcript=text,
        windows=[w.window.index for w in windows]
    )

class NarrativeAI:
    def __init__(self, model="llama3:8b"):
        self.model = model
        self.available = False
        self._check_and_start_server()

    def _check_and_start_server(self):
        """
        One-time check for Ollama connectivity.
        Attempts to start the server if it's in the path but not running.
        """
        import requests
        url = "http://localhost:11434/api/tags" # Lightweight check
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                self.available = True
                return
        except Exception:
            pass

        # If not reachable, try to start it
        print("Ollama server not detected. Attempting to start 'ollama serve'...", flush=True)
        try:
            # Check if ollama is in path
            subprocess.run(["ollama", "--version"], check=True, capture_output=True)
            # Start in background (using START on windows)
            # Detached process to avoid hanging the pipeline
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            
            # Wait a few seconds for it to boot
            import time
            for _ in range(5):
                time.sleep(2)
                try:
                    res = requests.get(url, timeout=1)
                    if res.status_code == 200:
                        print("Ollama server started successfully.", flush=True)
                        self.available = True
                        return
                except Exception:
                    continue
        except Exception:
            print("Warning: Could not start Ollama automatically. Narrative AI will use score-based fallback.", flush=True)
        
        self.available = False

    def tag_scenes(self, scenes: List[Scene]) -> List[Scene]:
        """
        Uses Ollama to tag scenes as Setup, Action, or Reaction.
        """
        if not self.available:
            for scene in scenes:
                scene.tags = ['action'] if scene.avg_score > 0.6 else ['setup']
            return scenes

        for scene in scenes:
            prompt = (
                f"Analyze this gaming video scene transcript and categorize its narrative role.\n"
                f"Transcript: \"{scene.transcript}\"\n\n"
                f"Rules:\n"
                f"1. Tag as 'setup' if it's building tension or explaining context.\n"
                f"2. Tag as 'action' if it's a climax, kill, or high-energy moment.\n"
                f"3. Tag as 'combat' if it's an intense fighting or battle scene.\n"
                f"4. Tag as 'reaction' if it's a laugh, scream, or post-moment comment.\n"
                f"Output ONLY the tag (one word)."
            )
            
            try:
                tag = self._query_ollama(prompt).strip().lower()
                if tag in ['setup', 'action', 'combat', 'reaction']:
                    scene.tags = [tag]
                else:
                    scene.tags = ['action'] if scene.avg_score > 0.6 else ['setup']
            except Exception:
                scene.tags = ['action'] if scene.avg_score > 0.6 else ['setup']
                
        return scenes

    def _query_ollama(self, prompt: str) -> str:
        if not self.available:
            return "action"

        import requests
        try:
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("response", "action")
        except Exception as e:
            # We already checked connectivity once, but if it fails mid-run, we mark as unavailable.
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                self.available = False
                print(f"Ollama connection lost during tagging: {e}", flush=True)
            return "action"

def stitch_narrative_arcs(scenes: List[Scene], top_k=5, max_arc_duration=90.0) -> List[Dict]:
    """
    Takes high-scoring 'action' scenes and attaches nearby 'setup' and 'reaction' scenes.
    Ensures final arc doesn't exceed social-media friendly durations.
    """
    arcs = []
    action_scenes = [s for s in scenes if any(t in s.tags for t in ('action', 'combat'))]
    action_scenes.sort(key=lambda s: s.avg_score, reverse=True)
    
    selected_actions = action_scenes[:top_k]
    
    for action in selected_actions:
        arc_scenes = [action]
        is_combat = 'combat' in action.tags
        target_max_duration = 35.0 if is_combat else max_arc_duration
        
        # Look for setup before
        idx = scenes.index(action)
        if idx > 0:
            prev = scenes[idx-1]
            if 'setup' in prev.tags or prev.avg_score > 0.3:
                # Only add if it doesn't blow the duration cap
                if (action.end_time - prev.start_time) <= target_max_duration:
                    arc_scenes.insert(0, prev)
                
        # Look for reaction after
        if idx < len(scenes) - 1:
            nxt = scenes[idx+1]
            if 'reaction' in nxt.tags or nxt.avg_score > 0.4:
                # Only add if it doesn't blow the duration cap
                if (nxt.end_time - arc_scenes[0].start_time) <= target_max_duration:
                    arc_scenes.append(nxt)
                
        start = arc_scenes[0].start_time
        end = arc_scenes[-1].end_time
        
        # Safety Truncation: if still too long (extreme cases), cap at the Action beat
        if (end - start) > target_max_duration:
            end = start + target_max_duration
        
        # Final story metadata
        arcs.append({
            "start": start,
            "end": end,
            "score": int(action.avg_score * 100),
            "category": "narrative_highlight",
            "reason": f"Action beat at {action.start_time:.1f}s with connected setup/reaction.",
            "text": " ".join([s.transcript for s in arc_scenes])
        })
        
    return arcs
