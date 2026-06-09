"""Dynamic Reframer — intelligent cropping and camera control for
vertical (9:16) short-form content.

Computes per-frame focus points based on tracked objects, genre
heuristics, and optical flow, then derives a smooth crop trajectory
that avoids jarring camera jumps while keeping the action centred.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from editor.creator_profile import CreatorStyleProfile

logger = logging.getLogger(__name__)


class DynamicReframer:
    """Real-time virtual-camera system for vertical reframing.

    Parameters
    ----------
    genre:
        Game genre — ``"fps"`` or ``"racing"``.  Determines focus-point
        heuristics.
    creator_profile:
        Optional :class:`CreatorStyleProfile` controlling zoom strength
        and other stylistic overrides.
    """

    def __init__(
        self,
        genre: str = "fps",
        creator_profile: Optional[CreatorStyleProfile] = None,
    ) -> None:
        self.genre = genre
        self.profile = creator_profile or CreatorStyleProfile.default()

        logger.debug(
            "DynamicReframer initialised — genre=%s, zoom_strength=%.2f",
            self.genre,
            self.profile.zoom_strength,
        )

    # ------------------------------------------------------------------
    # Focus-point computation
    # ------------------------------------------------------------------
    def compute_focus_point(
        self,
        frame_width: int,
        frame_height: int,
        tracked_objects: List[Dict[str, Any]],
        genre: str,
        optical_flow: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """Determine the ideal focus point for a single frame.

        Parameters
        ----------
        frame_width, frame_height:
            Dimensions of the source frame in pixels.
        tracked_objects:
            List of YOLO-style bounding-box dicts.  Each dict is
            expected to carry at least ``{"x": float, "y": float,
            "w": float, "h": float, "label": str, "confidence": float}``.
        genre:
            ``"fps"`` or ``"racing"`` — selects different spatial biases.
        optical_flow:
            Optional dict ``{"dx": float, "dy": float}`` describing the
            dominant motion vector in normalised coordinates.

        Returns
        -------
        Tuple[float, float]
            ``(focus_x, focus_y)`` normalised to ``[0, 1]``.
        """
        if not tracked_objects:
            # Fallback: centre of frame (with slight upward bias for FPS)
            default_y = 0.40 if genre == "fps" else 0.50
            return (0.50, default_y)

        if genre == "fps":
            focus_x, focus_y = self._focus_fps(
                frame_width, frame_height, tracked_objects,
            )
        elif genre == "racing":
            focus_x, focus_y = self._focus_racing(
                frame_width, frame_height, tracked_objects, optical_flow,
            )
        else:
            focus_x, focus_y = self._focus_generic(
                frame_width, frame_height, tracked_objects,
            )

        # Clamp to valid range
        focus_x = max(0.0, min(1.0, focus_x))
        focus_y = max(0.0, min(1.0, focus_y))

        return (focus_x, focus_y)

    # ------------------------------------------------------------------
    # Crop-region computation
    # ------------------------------------------------------------------
    def compute_crop_region(
        self,
        frame_width: int,
        frame_height: int,
        focus_x: float,
        focus_y: float,
        target_aspect: float = 9.0 / 16.0,
        zoom_factor: float = 1.0,
    ) -> Tuple[int, int, int, int]:
        """Calculate the pixel crop rectangle for a vertical short.

        Parameters
        ----------
        frame_width, frame_height:
            Source frame dimensions.
        focus_x, focus_y:
            Normalised focus point (``0–1``).
        target_aspect:
            Width-to-height ratio of the output (default ``9/16``).
        zoom_factor:
            Additional zoom multiplier (``>1`` zooms in).

        Returns
        -------
        Tuple[int, int, int, int]
            ``(x1, y1, x2, y2)`` pixel coordinates of the crop region.
        """
        effective_zoom = zoom_factor * self.profile.zoom_strength

        # Desired crop size
        crop_h = int(frame_height / effective_zoom)
        crop_w = int(crop_h * target_aspect)

        # Clamp crop to source dimensions
        crop_w = min(crop_w, frame_width)
        crop_h = min(crop_h, frame_height)

        # Centre the crop on the focus point
        cx = int(focus_x * frame_width)
        cy = int(focus_y * frame_height)

        x1 = cx - crop_w // 2
        y1 = cy - crop_h // 2

        # Shift crop region inside frame boundaries
        x1 = max(0, min(x1, frame_width - crop_w))
        y1 = max(0, min(y1, frame_height - crop_h))

        x2 = x1 + crop_w
        y2 = y1 + crop_h

        return (x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # Trajectory smoothing
    # ------------------------------------------------------------------
    def smooth_trajectory(
        self,
        focus_points: List[Tuple[float, float]],
        smoothing_window: int = 15,
    ) -> List[Tuple[float, float]]:
        """Apply exponential moving average to a sequence of focus points.

        Prevents jarring camera jumps between consecutive frames.

        Parameters
        ----------
        focus_points:
            Ordered list of ``(focus_x, focus_y)`` tuples.
        smoothing_window:
            Controls the smoothing strength.  Higher values yield
            smoother (slower) camera movement.  The EMA alpha is
            computed as ``2 / (smoothing_window + 1)``.

        Returns
        -------
        List[Tuple[float, float]]
            Smoothed focus points of the same length as the input.
        """
        if not focus_points:
            return []

        alpha = 2.0 / (smoothing_window + 1)
        smoothed: List[Tuple[float, float]] = [focus_points[0]]

        for i in range(1, len(focus_points)):
            prev_x, prev_y = smoothed[-1]
            cur_x, cur_y = focus_points[i]

            sx = alpha * cur_x + (1.0 - alpha) * prev_x
            sy = alpha * cur_y + (1.0 - alpha) * prev_y
            smoothed.append((sx, sy))

        logger.debug(
            "Smoothed %d focus points (window=%d, α=%.3f).",
            len(focus_points),
            smoothing_window,
            alpha,
        )
        return smoothed

    # ------------------------------------------------------------------
    # Genre-specific focus strategies (private)
    # ------------------------------------------------------------------
    def _focus_fps(
        self,
        frame_width: int,
        frame_height: int,
        tracked_objects: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        """FPS focus: tight lock on crosshair / enemy centre with rapid
        panning.  Prioritises high-confidence 'person' detections near
        the crosshair region.
        """
        crosshair_x = 0.50
        crosshair_y = 0.45  # slightly above centre

        best_obj = None
        best_score = -1.0

        for obj in tracked_objects:
            cx = (obj.get("x", 0.0) + obj.get("w", 0.0) / 2) / frame_width
            cy = (obj.get("y", 0.0) + obj.get("h", 0.0) / 2) / frame_height
            conf = obj.get("confidence", 0.0)

            # Proximity to crosshair (closer → higher score)
            dist = ((cx - crosshair_x) ** 2 + (cy - crosshair_y) ** 2) ** 0.5
            score = conf * max(0.0, 1.0 - dist)

            if score > best_score:
                best_score = score
                best_obj = obj

        if best_obj is not None:
            fx = (best_obj["x"] + best_obj["w"] / 2) / frame_width
            fy = (best_obj["y"] + best_obj["h"] / 2) / frame_height
            return (fx, fy)

        return (crosshair_x, crosshair_y)

    def _focus_racing(
        self,
        frame_width: int,
        frame_height: int,
        tracked_objects: List[Dict[str, Any]],
        optical_flow: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """Racing focus: wider view preserving horizon.  Tracks the
        largest vehicle detection and applies drift lead-space based on
        optical flow direction.
        """
        # Find the largest bounding-box area (assumed to be the player car)
        best_obj = None
        best_area = 0.0

        for obj in tracked_objects:
            area = obj.get("w", 0.0) * obj.get("h", 0.0)
            if area > best_area:
                best_area = area
                best_obj = obj

        if best_obj is not None:
            fx = (best_obj["x"] + best_obj["w"] / 2) / frame_width
            fy = (best_obj["y"] + best_obj["h"] / 2) / frame_height
        else:
            fx, fy = 0.50, 0.50

        # Apply drift lead-space: shift focus in the direction of motion
        if optical_flow:
            lead_strength = 0.08
            fx += optical_flow.get("dx", 0.0) * lead_strength
            fy += optical_flow.get("dy", 0.0) * lead_strength

        # Keep horizon visible — bias toward upper third
        fy = min(fy, 0.55)

        return (fx, fy)

    def _focus_generic(
        self,
        frame_width: int,
        frame_height: int,
        tracked_objects: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        """Generic focus: weighted centroid of all tracked objects."""
        total_weight = 0.0
        wx_sum = 0.0
        wy_sum = 0.0

        for obj in tracked_objects:
            conf = obj.get("confidence", 0.5)
            cx = (obj.get("x", 0.0) + obj.get("w", 0.0) / 2) / frame_width
            cy = (obj.get("y", 0.0) + obj.get("h", 0.0) / 2) / frame_height

            wx_sum += cx * conf
            wy_sum += cy * conf
            total_weight += conf

        if total_weight > 0:
            return (wx_sum / total_weight, wy_sum / total_weight)

        return (0.50, 0.50)
