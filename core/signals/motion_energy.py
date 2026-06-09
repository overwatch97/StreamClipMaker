import cv2
import numpy as np

class MotionEnergyDetector:
    """
    Computes global motion energy via frame differencing.
    """
    def __init__(self, blur_ksize: int = 5):
        self.blur_ksize = blur_ksize

    def compute_energy(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """
        Computes absolute frame difference and returns the mean intensity.
        """
        if prev_frame is None or curr_frame is None:
            return 0.0
            
        # Optional blur to reduce camera sensor noise
        prev_blur = cv2.GaussianBlur(prev_frame, (self.blur_ksize, self.blur_ksize), 0)
        curr_blur = cv2.GaussianBlur(curr_frame, (self.blur_ksize, self.blur_ksize), 0)
        
        diff = cv2.absdiff(prev_blur, curr_blur)
        energy = float(np.mean(diff))
        
        # Scale to standard range [0, 1] based on standard threshold
        normalized_energy = min(energy / 25.0, 1.0)
        return normalized_energy
