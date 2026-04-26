import os
import json
import logging
from typing import Optional, Dict, Any
from phase3_types import GameProfile

logger = logging.getLogger(__name__)

class ProfileResolver:
    def __init__(self, bundled_dir: str = "game_profiles", cache_dir: str = "cache/profiles"):
        self.bundled_dir = bundled_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def resolve(self, game_id: Optional[str] = None, custom_path: Optional[str] = None, 
                allow_online: bool = False) -> GameProfile:
        """
        Resolves a game ID or path to a GameProfile object.
        Precedence: Custom Path > Bundled JSON > Cloud/Cache > Generic Default.
        """
        # 1. Custom Path
        if custom_path and os.path.exists(custom_path):
            logger.info(f"Loading custom profile from {custom_path}")
            return self._load_file(custom_path)

        # 2. Bundled JSON
        if game_id:
            bundled_path = os.path.join(self.bundled_dir, f"{game_id}.json")
            if os.path.exists(bundled_path):
                logger.info(f"Loading bundled profile for {game_id}")
                return self._load_file(bundled_path)

        # 3. Cloud/Cache (if allowed)
        if game_id and allow_online:
            cached_path = os.path.join(self.cache_dir, f"{game_id}.json")
            if os.path.exists(cached_path):
                logger.info(f"Loading cached profile for {game_id}")
                return self._load_file(cached_path)
            
            # Simulated Cloud Generation
            profile_data = self._generate_cloud_profile(game_id)
            if profile_data:
                self._save_cache(game_id, profile_data)
                return GameProfile.from_json(profile_data)

        # 4. Fallback to Generic
        generic_path = os.path.join(self.bundled_dir, "generic.json")
        if os.path.exists(generic_path):
            logger.info("Falling back to generic profile")
            return self._load_file(generic_path)

        # 5. Last resort hardcoded default
        logger.warning("No profiles found. Using hardcoded generic default.")
        return GameProfile.from_json({
            "game_id": "generic",
            "game_name": "Generic Default",
            "genre": "general",
            "priority_events": ["spike"],
            "ignore_states": [],
            "context_rules": {"min_clip_duration": 10, "max_clip_duration": 40},
            "score_weights": {"speech": 0.25, "audio": 0.25, "visual": 0.25, "emotion": 0.25},
            "event_rules": []
        })

    def _load_file(self, path: str) -> GameProfile:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return GameProfile.from_json(data)

    def _save_cache(self, game_id: str, data: Dict[str, Any]):
        path = os.path.join(self.cache_dir, f"{game_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _generate_cloud_profile(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Placeholder for optional cloud profile generation.
        In a real implementation, this would query a metadata API.
        """
        logger.info(f"Simulating cloud generation for {game_id}")
        # For now, just return a generic template with the game_id
        return {
            "game_id": game_id,
            "game_name": game_id.replace("-", " ").title(),
            "genre": "generated",
            "priority_events": ["spike", "excitement"],
            "ignore_states": ["loading", "menu"],
            "context_rules": {"min_clip_duration": 10, "max_clip_duration": 40},
            "score_weights": {"speech": 0.25, "audio": 0.25, "visual": 0.25, "emotion": 0.25},
            "event_rules": []
        }
