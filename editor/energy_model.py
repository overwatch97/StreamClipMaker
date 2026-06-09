"""Energy Model — computes a rolling excitement energy curve over
timeline frames by fusing motion, audio, and facecam signals.

The resulting 1-D energy array drives downstream decisions such as
peak detection, pacing adjustments, and slow-motion gating.
"""

import logging
from typing import List

import numpy as np

from core.cache.timeline_frame import TimelineFrame

logger = logging.getLogger(__name__)

# 5-tap Gaussian-style smoothing kernel (sums to 1.0)
_SMOOTH_KERNEL = np.array([0.05, 0.15, 0.60, 0.15, 0.05], dtype=np.float64)


class EnergyTimeline:
    """Fused excitement energy curve derived from multimodal timeline
    frames.

    Parameters
    ----------
    motion_weight:
        Contribution weight for the motion energy channel.
    audio_weight:
        Contribution weight for the audio energy channel.
    emotion_weight:
        Contribution weight for the facecam/emotion channel.
    """

    def __init__(
        self,
        motion_weight: float = 0.35,
        audio_weight: float = 0.35,
        emotion_weight: float = 0.30,
    ) -> None:
        self.motion_weight = motion_weight
        self.audio_weight = audio_weight
        self.emotion_weight = emotion_weight

        logger.debug(
            "EnergyTimeline weights: motion=%.2f audio=%.2f emotion=%.2f",
            self.motion_weight,
            self.audio_weight,
            self.emotion_weight,
        )

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def compute_timeline_energy(
        self,
        frames: List[TimelineFrame],
    ) -> np.ndarray:
        """Compute a smoothed energy curve from *frames*.

        For each frame the raw score is::

            score = motion_energy × w₁ + audio_energy × w₂ + facecam_motion × w₃

        After scoring, a 5-tap Gaussian kernel is convolved over the
        curve and the result is clipped to ``[0, 1]``.

        Parameters
        ----------
        frames:
            Chronologically ordered ``TimelineFrame`` objects.

        Returns
        -------
        np.ndarray
            1-D float64 array of length ``len(frames)`` with values in
            ``[0.0, 1.0]``.
        """
        if not frames:
            logger.warning("compute_timeline_energy called with empty frame list.")
            return np.array([], dtype=np.float64)

        n = len(frames)
        raw = np.empty(n, dtype=np.float64)

        for i, f in enumerate(frames):
            raw[i] = (
                f.motion_energy * self.motion_weight
                + f.audio_energy * self.audio_weight
                + f.facecam_motion * self.emotion_weight
            )

        # Apply 5-tap Gaussian smoothing (mode='same' preserves length)
        if n >= len(_SMOOTH_KERNEL):
            smoothed = np.convolve(raw, _SMOOTH_KERNEL, mode="same")
        else:
            # Sequence is shorter than the kernel — skip smoothing
            smoothed = raw.copy()

        smoothed = np.clip(smoothed, 0.0, 1.0)

        logger.info(
            "EnergyTimeline: %d frames → energy range [%.3f, %.3f]",
            n,
            float(smoothed.min()),
            float(smoothed.max()),
        )
        return smoothed

    # ------------------------------------------------------------------
    # Point queries
    # ------------------------------------------------------------------
    def get_energy_at(
        self,
        energy_curve: np.ndarray,
        timestamp: float,
        fps: float = 1.0,
    ) -> float:
        """Look up the energy value at a specific *timestamp*.

        Parameters
        ----------
        energy_curve:
            Pre-computed energy array (from :meth:`compute_timeline_energy`).
        timestamp:
            Time in seconds to query.
        fps:
            Frames per second of the energy curve.  Defaults to ``1.0``
            (one entry per second).

        Returns
        -------
        float
            Energy value in ``[0.0, 1.0]``, or ``0.0`` if the timestamp
            falls outside the curve.
        """
        if energy_curve.size == 0:
            return 0.0

        idx = int(round(timestamp * fps))
        if idx < 0 or idx >= len(energy_curve):
            return 0.0

        return float(energy_curve[idx])

    # ------------------------------------------------------------------
    # Peak detection
    # ------------------------------------------------------------------
    def find_peaks(
        self,
        energy_curve: np.ndarray,
        threshold: float = 0.5,
    ) -> List[int]:
        """Find indices of local maxima that exceed *threshold*.

        A sample is considered a local maximum when it is strictly
        greater than both its immediate neighbours and its value is at
        least *threshold*.

        Parameters
        ----------
        energy_curve:
            Pre-computed energy array.
        threshold:
            Minimum energy level for a peak to be reported.

        Returns
        -------
        List[int]
            Sorted list of frame indices where peaks were detected.
        """
        if energy_curve.size < 3:
            # Need at least 3 points for a local-max check
            return [
                int(i) for i in range(len(energy_curve))
                if energy_curve[i] >= threshold
            ]

        peaks: List[int] = []
        for i in range(1, len(energy_curve) - 1):
            val = energy_curve[i]
            if (
                val >= threshold
                and val > energy_curve[i - 1]
                and val > energy_curve[i + 1]
            ):
                peaks.append(i)

        logger.debug(
            "find_peaks: %d peaks above threshold %.2f in %d-frame curve.",
            len(peaks),
            threshold,
            len(energy_curve),
        )
        return peaks
