# 🎬 StreamClipMaker

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FFmpeg](https://img.shields.io/badge/Render-FFmpeg-orange.svg)](https://ffmpeg.org/)
[![AI-Powered](https://img.shields.io/badge/AI-Multimodal-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**StreamClipMaker** is a professional-grade, locally-run AI production pipeline designed to transform long-form gaming streams into retention-optimized vertical videos (YouTube Shorts, TikToks, Reels). 

Unlike simple clipping tools, StreamClipMaker uses **Multimodal AI** to understand the narrative and emotional context of your gameplay, ensuring every clip captures the most high-energy and viral-worthy moments.

---

## 🚀 Key Features

- **🧠 Multimodal Scoring Engine**: Ranks moments based on a combination of speech tone, audio spikes, visual motion, and facial emotion analysis.
- **🤖 SmartSpotlight™**: Uses YOLO-based AI to track characters and gameplay elements, automatically reframing horizontal content into a 9:16 vertical layout.
- **🎭 Facial Emotion AI**: Analyzes your facecam in real-time to detect genuine reactions like joy, surprise, or intense focus.
- **💬 Animated Subtitles**: Generates word-level, timestamped transcripts with Faster-Whisper and renders them as dynamic, high-engagement captions.
- **⚡ Hardware Accelerated**: Optimized for NVIDIA GPUs using `h264_nvenc` and ONNX Runtime for lightning-fast AI analysis and rendering.
- **🛠️ Professional GUI**: A premium dark-mode interface built with `CustomTkinter` for full control over your production workflow.

---

## 🏗️ Architecture: The 4-Phase Pipeline

StreamClipMaker operates in four distinct phases to ensure maximum quality and reliability:

1.  **Signal Extraction**: Isolates audio from video to allow parallel AI processing.
2.  **Cognitive Transcription**: Generates high-accuracy transcripts using Faster-Whisper.
3.  **Multimodal Ranking**: Calculates a "Virality Score" by merging signals from 4 different AI models.
4.  **Cinematic Production**: Orchestrates complex FFmpeg filter graphs to merge gameplay, facecam, watermarks, and subtitles into a final export.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Vision Tracking** | YOLOv8 / YOLOv11 |
| **Speech Engine** | Faster-Whisper |
| **Emotion Analysis** | ONNX Runtime |
| **Render Engine** | FFmpeg (Filter Complex) |
| **GUI Framework** | CustomTkinter |
| **Hardware Orchestration** | Custom Weighted Multi-modal Logic |

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/overwatch97/StreamClipMaker.git
   cd StreamClipMaker
   ```

2. **Run the setup script:**
   ```bash
   ./Initial_Setup.bat
   ```
   *This will create a virtual environment and install all necessary dependencies (FFmpeg, CUDA drivers, etc. may be required separately).*

3. **Launch the application:**
   ```bash
   ./Launch_GUI.bat
   ```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ for creators who want to automate their highlight workflow without sacrificing quality.*
