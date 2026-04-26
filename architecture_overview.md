# StreamClipMaker Architecture Overview

StreamClipMaker is a professional-grade, locally-run AI production pipeline designed to transform long-form streams into retention-optimized YouTube Shorts. Its architecture is divided into three main layers: the **User Interface**, the **Orchestration Layer**, and the **AI/ML Analysis Pipeline**.

---

## 1. The Interface Layer (`gui.py`)
Built using `customtkinter` with a premium dark-mode aesthetic, this layer serves as the control center.
*   **Coordinate Management**: It gathers essential paths (Source Video, Facecam, Output Folder) and Pro settings (Watermarks, SmartSpotlight).
*   **Live Feedback**: It uses a multi-threaded system to stream console logs in real-time and updates a phased-based progress bar.
*   **Hardware Control**: Allows you to force specific stages to CPU/GPU or use the "Auto" mode which relies on the Hardware Layer's intelligence.

---

## 2. The Hardware & Orchestration Layer (`main.py` & `hardware.py`)
This is the "brain" that manages the 4-phase lifecycle of a project.
*   **Preflight System**: Before a single pixel is processed, `hardware.py` detects your system's capabilities (CUDA cores, NVENC encoders, ONNX providers).
*   **Hardware Profile (`hardware_profile.json`)**: It actually *learns* from your PC. It tracks how long each task takes on CPU vs. GPU and automatically picks the fastest successful path for future clips.
*   **Caching System**: It uses a fingerprinting method to save intermediate files (transcripts, extracted audio). If you rerun a stream, it skips the heavy AI analysis and goes straight to clipping.

---

## 3. The 4-Phase AI Pipeline

### Phase 1: Signal Extraction (`audio_extractor.py`)
The pipeline isolation begins here. It extracts a high-fidelity WAV file from the source. It separates the "hearing" from the "seeing" to allow parallel AI analysis in the next steps.

### Phase 2: Cognitive Transcription (`transcriber.py`)
Uses **Faster-Whisper** to generate word-level timestamped transcripts. This mapping exactly *when* every word is spoken is critical for the animated subtitles in the final video.

### Phase 3: Multi-modal Ranking (`scoring_engine.py` & `llm_selector.py`)
Calculates a "Virality Score" by combining four distinct AI signals:
1.  **Speech Tone**: Detecting excitement or high-energy keywords in the transcript.
2.  **Audio Spikes**: Finding loud moments (screams, gameplay explosions).
3.  **Visual Motion**: Detecting fast-paced action or scene changes.
4.  **Emotion AI**: Using facial recognition (on facecam) to detect genuine reactions like joy, surprise, or anger.
*   **Momentum Logic**: It doesn't just look for "peaks"—it looks for "build-ups" where excitement increases over 10-15 seconds.

### Phase 4: Cinematic Production (`clipper.py` & `subtitler.py`)
Once the best moments are found, they are sent to the rendering engine:
*   **SmartSpotlight**: A YOLO-based AI tracks characters in gameplay and dynamically crops them into a vertical 9:16 frame.
*   **Complex Overlay Engine**: Uses FFmpeg `filter_complex` to merge game, facecam, animated word-level subtitles, watermarks, and dynamic progress bars.
*   **Encoder Optimization**: Uses `h264_nvenc` for hardware-accelerated encoding on NVIDIA GPUs.

---

## Technical Stack Summary

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Logic Orchestrator** | Python 3.10+ subprocess | Isolates heavy AI tasks to prevent memory crashes. |
| **Vision Tracking** | YOLOv8 / YOLOv11 | AI Reframing (SmartSpotlight). |
| **Speech Engine** | Faster-Whisper | Timestamped transcription. |
| **Render Engine** | FFmpeg (Filter Complex) | Layout merging, Subtitles, and Watermarks. |
| **GUI Framework** | CustomTkinter | Desktop application interface. |
| **Emotion Analysis** | ONNX Runtime | Facial reaction detection. |
| **Scoring Logic** | Custom Weighted Multi-modal | Selecting high-retention clips. |
