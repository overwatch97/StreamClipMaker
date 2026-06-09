import cv2
import numpy as np
import logging
from emotion_analyzer import get_face_landmarker, calculate_heuristics, _crop_combined_facecam

logger = logging.getLogger(__name__)

class FacecamSignalExtractor:
    """
    Analyzes facecam frames to detect face presence, surprise (EAR), screaming/speaking (MAR),
    and horizontal/vertical movement (engagement).
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.landmarker = None
        self.mp = None
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            self.landmarker, self.mp = get_face_landmarker(device=self.device)
        except Exception as e:
            logger.warning(f"FacecamSignalExtractor: Failed to initialize MediaPipe Face Landmarker on {self.device}: {e}")

    def analyze_frame(self, frame: np.ndarray, is_combined: bool = True) -> dict:
        """
        Analyzes a single facecam frame and extracts heuristics.
        """
        if self.landmarker is None or frame is None or frame.size == 0:
            return {"emotion_score": 0.0, "surprise": 0.0, "reaction": 0.0, "nose_x": 0.5, "nose_y": 0.5, "visible": False}

        # Crop if facecam is overlayed on the bottom right of the gameplay frame
        analysis_frame = _crop_combined_facecam(frame) if is_combined else frame
        if analysis_frame.size == 0:
            return {"emotion_score": 0.0, "surprise": 0.0, "reaction": 0.0, "nose_x": 0.5, "nose_y": 0.5, "visible": False}

        rgb_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            result = self.landmarker.detect(mp_img)
            if not result.face_landmarks:
                return {"emotion_score": 0.0, "surprise": 0.0, "reaction": 0.0, "nose_x": 0.5, "nose_y": 0.5, "visible": False}
                
            ear, mar, pose_mag, nx, ny = calculate_heuristics(result.face_landmarks)
            
            # Simple normalization mapping
            # EAR threshold: typical value range ~[0.15, 0.40]
            # MAR threshold: typical value range ~[0.05, 0.70]
            surprise = float(np.clip((ear - 0.15) / 0.25, 0.0, 1.0))
            reaction = float(np.clip((mar - 0.05) / 0.60, 0.0, 1.0))
            emotion_score = max(surprise, reaction)
            
            return {
                "emotion_score": emotion_score,
                "surprise": surprise,
                "reaction": reaction,
                "nose_x": nx,
                "nose_y": ny,
                "visible": True
            }
        except Exception as e:
            logger.debug(f"FacecamSignalExtractor: detect error: {e}")
            return {"emotion_score": 0.0, "surprise": 0.0, "reaction": 0.0, "nose_x": 0.5, "nose_y": 0.5, "visible": False}
