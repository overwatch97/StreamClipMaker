import cv2
import numpy as np

class OpticalFlowTracker:
    """
    Computes dense Farneback optical flow to track vertical and horizontal motion vectors.
    """
    def __init__(self, pyr_scale: float = 0.5, levels: int = 3, winsize: int = 15,
                 iterations: int = 3, poly_n: int = 5, poly_sigma: float = 1.2):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma

    def compute_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> dict:
        """
        Calculates flow between two grayscale frames.
        Returns horizontal motion, vertical motion, and lateral ratio (sliding indicator).
        """
        if prev_gray is None or curr_gray is None:
            return {"dx": 0.0, "dy": 0.0, "lateral_ratio": 0.0}

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            self.pyr_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma, 0
        )
        
        # Split into horizontal (x) and vertical (y) flow maps
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        
        mean_dx = float(np.mean(np.abs(flow_x)))
        mean_dy = float(np.mean(np.abs(flow_y)))
        
        # Lateral ratio is higher when there is more horizontal flow relative to vertical flow (e.g. during a drift)
        total_flow = mean_dx + mean_dy
        lateral_ratio = float(mean_dx / (total_flow + 1e-6)) if total_flow > 0.1 else 0.0
        
        return {
            "dx": mean_dx,
            "dy": mean_dy,
            "lateral_ratio": lateral_ratio
        }
