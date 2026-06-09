import os
import time
import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

from core.cache.timeline_frame import TimelineFrame
from core.cache.storage import TimelineCache

from core.signals.optical_flow import OpticalFlowTracker
from core.signals.motion_energy import MotionEnergyDetector
from core.signals.audio_analysis import AudioSignalExtractor
from core.signals.facecam_analysis import FacecamSignalExtractor

from visual_analyzer import get_yolo_model, get_bytetrack_config, _map_scene_type, require_ultralytics
from speech_analyzer import analyze_speech_windows
from multimodal_utils import FFmpegFrameExtractor, ProgressReporter

logger = logging.getLogger(__name__)

class SharedSignalExtractor:
    """
    Orchestrates the extraction of multimodal signals from gameplay video and audio.
    Caches results to disk to prevent duplicate runs of expensive operations.
    """
    def __init__(self, hardware_mode: str = "auto", device: str = "cpu"):
        self.cache = TimelineCache()
        self.device = "cuda" if (device == "gpu" or (device == "auto" and cv2.ocl.haveOpenCL())) else "cpu"
        self.flow_tracker = OpticalFlowTracker()
        self.motion_detector = MotionEnergyDetector()

    def extract(self, video_path: str, audio_path: str, transcript_data: List[Dict],
                facecam_path: Optional[str] = None, sample_fps: float = 2.0) -> List[TimelineFrame]:
        """
        Extracts all multimodal signals for the video.
        Checks disk cache first to ensure zero-redundancy.
        """
        # 1. Try loading from cache
        cached_timeline = self.cache.load(video_path)
        if cached_timeline:
            logger.info("SharedSignalExtractor: Timeline loaded successfully from cache.")
            return cached_timeline

        logger.info("SharedSignalExtractor: Cache miss. Extracting signals from video/audio...")
        start_time = time.time()

        # 2. Extract Audio features
        audio_extractor = AudioSignalExtractor(audio_path)
        audio_extractor.load_and_analyze()

        # 3. Initialize Facecam extractor if path is provided
        facecam_extractor = None
        if facecam_path and os.path.exists(facecam_path):
            facecam_extractor = FacecamSignalExtractor(device=self.device)

        # 4. Setup YOLO Tracker
        yolo_model = get_yolo_model()
        names_dict = yolo_model.names
        tracker_path = get_bytetrack_config()

        # 5. Extract video frames sequentially
        # Decodes via GPU/FFmpeg helper
        video_extractor = FFmpegFrameExtractor(video_path, sample_fps, target_width=320, device=self.device)
        duration = float(video_extractor.info.get("duration") or 0)
        total_samples = int(duration * sample_fps) if duration > 0 else 0

        frames: List[TimelineFrame] = []
        prev_gray = None
        sample_index = 0
        
        reporter = ProgressReporter(total_samples, label="Signals Extraction")

        # Track nose position for facecam delta
        last_nose_x, last_nose_y = 0.5, 0.5

        # Speech features
        # Generate generic windows aligned with sample stamps to score speech energy
        from segment_ranker import generate_sliding_windows
        windows = generate_sliding_windows(duration, window_secs=1.0, stride_secs=1.0)
        speech_rows, _ = analyze_speech_windows(transcript_data, windows)

        batch_size = 16
        batch_frames = []
        batch_data = []

        def flush_yolo_batch():
            if not batch_frames:
                return
            
            # Execute batch YOLO tracking on GPU/CPU
            kwargs = {"source": batch_frames, "persist": True, "tracker": tracker_path, "verbose": False}
            try:
                results = yolo_model.track(**kwargs)
            except Exception:
                kwargs.pop("device", None)
                results = yolo_model.track(**kwargs)

            for idx, result in enumerate(results):
                t_sec, motion, flow_data, audio_feats, facecam_feats = batch_data[idx]
                boxes = getattr(result, "boxes", None)
                scene_type, scene_conf, activity = _map_scene_type(boxes, names_dict)

                # Format YOLO detected objects
                tracked_objs = []
                if boxes is not None and boxes.xyxy is not None:
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()
                    bboxes = boxes.xyxy.cpu().numpy()
                    for c_id, conf, bbox in zip(cls_ids, confs, bboxes):
                        tracked_objs.append({
                            "class_id": int(c_id),
                            "name": names_dict.get(c_id, ""),
                            "bbox": [float(val) for val in bbox],
                            "conf": float(conf)
                        })

                # Calculate camera shake (high frequency high vertical flow)
                camera_shake = float(np.clip(flow_data["dy"] / 5.0, 0.0, 1.0))

                # Estimate vehicle speed: based on vehicle object scale & flow
                vehicle_speed = 0.0
                if scene_type == "travel" and flow_data["dx"] > 0.5:
                    vehicle_speed = float(np.clip(flow_data["dx"] * 1.5, 0.0, 1.0))

                # Facecam movement delta (nose delta)
                facecam_motion = 0.0
                if facecam_feats.get("visible", False):
                    nonlocal last_nose_x, last_nose_y
                    dx = facecam_feats["nose_x"] - last_nose_x
                    dy = facecam_feats["nose_y"] - last_nose_y
                    facecam_motion = float(np.clip(np.hypot(dx, dy) * 10.0, 0.0, 1.0))
                    last_nose_x = facecam_feats["nose_x"]
                    last_nose_y = facecam_feats["nose_y"]

                # Extract speech energy
                speech_energy = 0.0
                sec_idx = int(t_sec)
                if sec_idx < len(speech_rows):
                    speech_energy = float(speech_rows[sec_idx].get("score", 0.0))

                frame = TimelineFrame(
                    timestamp=t_sec,
                    motion_energy=motion,
                    optical_flow=flow_data,
                    audio_energy=audio_feats["rms"],
                    pitch_arousal=audio_feats["pitch_arousal"],
                    speech_emotion=speech_energy,
                    tracked_objects=tracked_objs,
                    scene_type=scene_type,
                    facecam_motion=facecam_motion,
                    vehicle_speed_estimate=vehicle_speed,
                    camera_shake=camera_shake,
                    event_candidates={},
                    metadata={
                        "audio_screech": audio_feats["screech"],
                        "scene_confidence": scene_conf,
                        "surprise_level": facecam_feats.get("surprise", 0.0),
                        "reaction_level": facecam_feats.get("reaction", 0.0)
                    }
                )
                frames.append(frame)

            batch_frames.clear()
            batch_data.clear()

        # Iterate through frames
        for frame in video_extractor:
            sample_index += 1
            reporter.update()
            reporter.report()

            t_sec = sample_index / sample_fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Heuristics
            motion = self.motion_detector.compute_energy(prev_gray, gray)
            flow_data = self.flow_tracker.compute_flow(prev_gray, gray)

            # Retrieve precomputed audio features
            audio_feats = audio_extractor.get_features_at(t_sec, window_size=1.0)

            # Extract facecam landmarks
            facecam_feats = {"emotion_score": 0.0, "surprise": 0.0, "reaction": 0.0, "nose_x": 0.5, "nose_y": 0.5, "visible": False}
            if facecam_extractor:
                # Capture frame from facecam video at target timestamp
                # For simplicity, we just pass the frame if combined, or query facecam file
                facecam_feats = facecam_extractor.analyze_frame(frame, is_combined=True)

            batch_frames.append(frame)
            batch_data.append((t_sec, motion, flow_data, audio_feats, facecam_feats))

            if len(batch_frames) >= batch_size:
                flush_yolo_batch()

            prev_gray = gray

        flush_yolo_batch()
        video_extractor.close()

        # Save to disk cache
        self.cache.save(video_path, frames)
        elapsed = time.time() - start_time
        logger.info(f"SharedSignalExtractor: Extraction complete in {elapsed:.1f}s. Saved to cache.")

        return frames
