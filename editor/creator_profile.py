"""Creator Profile — personalised style profiles that control editing
behaviour such as zoom strength, clip length constraints, pacing feel,
subtitle rendering, and audio ducking.

Profiles are serialised to / from JSON so creators can save, share, and
tweak their preferred editing aesthetic.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class CreatorStyleProfile:
    """Encapsulates a creator's stylistic preferences for clip editing.

    Attributes
    ----------
    profile_name:
        Human-readable identifier for the profile.
    zoom_strength:
        Scaling multiplier applied to dynamic reframing zoom.
    min_clip_length:
        Minimum allowable clip duration in seconds.
    max_clip_length:
        Maximum allowable clip duration in seconds.
    reaction_bias:
        Multiplier boosting facecam / reaction priority.
    music_ducking_db:
        Background music ducking level in decibels (negative values
        attenuate).
    subtitle_style:
        Visual style preset for subtitles.  One of ``"dynamic_pop"``,
        ``"cinematic_bottom"``, or ``"clean"``.
    pacing_feel:
        Overall clip pacing.  One of ``"energetic"``, ``"cinematic"``,
        or ``"chill"``.
    """

    profile_name: str = "default"
    zoom_strength: float = 1.0
    min_clip_length: float = 8.0
    max_clip_length: float = 45.0
    reaction_bias: float = 1.0
    music_ducking_db: float = -6.0
    subtitle_style: str = "dynamic_pop"
    pacing_feel: str = "energetic"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "CreatorStyleProfile":
        """Deserialise a profile from a JSON file at *path*.

        Parameters
        ----------
        path:
            Absolute or relative path to the ``.json`` file.

        Returns
        -------
        CreatorStyleProfile
            The loaded profile instance.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        json.JSONDecodeError
            If the file contains invalid JSON.
        """
        logger.info("Loading creator profile from '%s'.", path)

        with open(path, "r", encoding="utf-8") as fh:
            data: Dict[str, Any] = json.load(fh)

        # Only accept keys that are valid dataclass fields
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}

        profile = cls(**filtered)
        logger.debug("Loaded profile: %s", profile)
        return profile

    def save(self, path: str) -> None:
        """Serialise this profile to a JSON file at *path*.

        Parent directories are created automatically if they do not
        exist.

        Parameters
        ----------
        path:
            Destination file path.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

        logger.info("Saved creator profile '%s' to '%s'.", self.profile_name, path)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def default(cls) -> "CreatorStyleProfile":
        """Return a profile with all default values."""
        return cls()
