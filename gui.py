import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import subprocess
import threading
import os
import sys
import json
import time
from datetime import datetime
from hardware import build_preflight_lines, detect_capabilities, plan_hardware
from runtime_env import build_runtime_env

# ─────────────────────────────────────────────
# Theme & Appearance
# ─────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT      = "#3B8EEA"
BG_DARK     = "#0F0F13"
BG_PANEL    = "#1A1A24"
BG_CARD     = "#22222F"
TEXT_MAIN   = "#EAEAEA"
TEXT_DIM    = "#7A7A9A"
SUCCESS_CLR = "#2ECC71"
WARN_CLR    = "#F39C12"
ERR_CLR     = "#E74C3C"
FONT_HEAD   = ("Segoe UI", 22, "bold")
FONT_SUB    = ("Segoe UI", 12)
FONT_MONO   = ("Consolas", 11)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = os.path.join(SCRIPT_DIR, "venv", "Scripts", "python.exe")
MAIN_PY    = os.path.join(SCRIPT_DIR, "main.py")


def build_pipeline_command(
    video_path,
    output_dir,
    *,
    use_facecam=False,
    use_watermark=True,
    use_vision=False,
    use_spotlight=False,
    channel_name="Overwatch-live",
    facecam_src=None,
    logo_src=None,
    show_hook=False,
    hardware_mode="auto",
    transcribe_device=None,
    visual_device=None,
    emotion_device=None,
    encode_device=None,
    spotlight_device=None,
    gen_short=True,
    gen_long=True,
    short_res="1080x1920",
    long_res="source",
    use_subtitles=True,
    use_hinglish=False,
    game=None,
    game_profile=None,
    profile_source="local",
    detect_device=None,
):
    cmd = [PYTHON_EXE, "-u", MAIN_PY, video_path, "--output", output_dir]
    if use_facecam:
        cmd.append("--facecam")
    if use_watermark:
        cmd.append("--watermark")
    else:
        cmd.append("--no-watermark")
    
    if logo_src:
        cmd += ["--watermark-src", logo_src]
    
    if show_hook:
        cmd.append("--hook-badge")
    else:
        cmd.append("--no-hook-badge")

    if use_vision:
        cmd.append("--vision")
    if use_spotlight:
        cmd.append("--spotlight")
    cmd += ["--channel", channel_name, "--hardware-mode", hardware_mode]
    if transcribe_device is not None:
        cmd += ["--transcribe-device", transcribe_device]
    if visual_device is not None:
        cmd += ["--visual-device", visual_device]
    if emotion_device is not None:
        cmd += ["--emotion-device", emotion_device]
    if encode_device is not None:
        cmd += ["--encode-device", encode_device]
    if spotlight_device is not None:
        cmd += ["--spotlight-device", spotlight_device]
    if not gen_short:
        cmd.append("--no-short")
    if not gen_long:
        cmd.append("--no-long")
    cmd += ["--short-res", short_res, "--long-res", long_res]
    if use_subtitles:
        cmd.append("--subtitles")
    else:
        cmd.append("--no-subtitles")
    if use_hinglish:
        cmd.append("--hinglish")
    if game:
        cmd += ["--game", game]
    if game_profile:
        cmd += ["--game-profile", game_profile]
    if profile_source:
        cmd += ["--profile-source", profile_source]
    if detect_device:
        cmd += ["--detect-device", detect_device]
    if facecam_src:
        cmd += ["--facecam_src", facecam_src]
    return cmd


def terminate_process_tree(process):
    if not process or process.poll() is not None:
        return

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            process.terminate()
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass

# ─────────────────────────────────────────────
# Helper: detect phase from log line
# ─────────────────────────────────────────────
PHASES = {
    "PHASE 1": ("🎵  Audio Extraction",  0.05),
    "PHASE 2": ("🤖  AI Transcription",  0.30),
    "PHASE 3": ("🧠  Highlight Ranking",   0.60),
    "PHASE 4": ("✂️  Video Clipping",     0.85),
    "Pipeline Complete": ("✅  Done!",    1.00),
}

def detect_phase(line):
    for key, val in PHASES.items():
        if key in line:
            return val
    return None


# ─────────────────────────────────────────────
# Main App Window
# ─────────────────────────────────────────────
class StreamClipApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("StreamClipMaker — Auto Shorts Studio")
        self.geometry("980x760")
        self.minsize(780, 620)
        self.configure(fg_color=BG_DARK)
        self.resizable(True, True)

        self._process   = None
        self._running   = False
        self._video_path = tk.StringVar()
        self._facecam_path = tk.StringVar()
        self._output_dir = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Desktop", "StreamClips"))
        
        # Pro Settings Vars
        self._use_facecam = tk.BooleanVar(value=False)
        self._use_watermark = tk.BooleanVar(value=True)
        self._logo_path = tk.StringVar(value="")
        self._show_hook_badge = tk.BooleanVar(value=False)
        self._use_vision = tk.BooleanVar(value=False)
        self._use_spotlight = tk.BooleanVar(value=True)
        self._channel_name = tk.StringVar(value="Overwatch-live")
        self._hardware_mode = tk.StringVar(value="auto")
        self._transcribe_device = tk.StringVar(value="auto")
        self._visual_device = tk.StringVar(value="auto")
        self._emotion_device = tk.StringVar(value="auto")
        self._encode_device = tk.StringVar(value="auto")
        self._spotlight_device = tk.StringVar(value="auto")
        self._hardware_summary = tk.StringVar(value="Hardware summary loading...")
        
        # Video Export Settings
        self._gen_short = tk.BooleanVar(value=True)
        self._gen_long = tk.BooleanVar(value=True)
        self._short_res = tk.StringVar(value="1080x1920")
        self._long_res = tk.StringVar(value="source")
        self._burn_subtitles = tk.BooleanVar(value=True)
        self._use_hinglish = tk.BooleanVar(value=False)
        
        # Game-Aware Phase 3 Vars
        self._game_id = tk.StringVar(value="generic")
        self._game_profile_path = tk.StringVar(value="")
        self._profile_source = tk.StringVar(value="local")
        self._detect_device = tk.StringVar(value="auto")
        
        self._hardware_advanced_visible = False
        self._queue = []
        self._current_job_id = None
        self._queue_file = os.path.join(SCRIPT_DIR, "streamclip_queue.json")

        self._build_ui()
        self._load_app_settings()
        self._load_queue()
        # Launch hardware preflight in background so the window appears instantly
        threading.Thread(target=self._refresh_hardware_summary, daemon=True).start()

    # ──────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──────────────────────────────
        header = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="🎬  StreamClipMaker",
            font=FONT_HEAD,
            text_color=TEXT_MAIN
        ).pack(side="left", padx=24, pady=14)

        ctk.CTkLabel(
            header,
            text="Auto YouTube Shorts Studio  •  GPU Powered",
            font=FONT_SUB,
            text_color=TEXT_DIM
        ).pack(side="left", padx=4, pady=14)

        # ── Main body (Scrollable) ───────────────
        body = ctk.CTkScrollableFrame(self, fg_color=BG_DARK, corner_radius=0)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        # ── Input Card ──────────────────────────
        card = ctk.CTkFrame(body, fg_color=BG_CARD, corner_radius=14)
        card.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        card.columnconfigure(1, weight=1)

        # Video file row
        ctk.CTkLabel(card, text="🎮  Video File", font=("Segoe UI", 12, "bold"),
                     text_color=TEXT_MAIN).grid(row=0, column=0, padx=18, pady=(18,4), sticky="w")

        self._video_entry = ctk.CTkEntry(
            card, textvariable=self._video_path,
            placeholder_text="Click Browse or drag & drop a .mp4 file...",
            height=40, corner_radius=8, font=("Segoe UI", 12, "bold"),
            fg_color=BG_PANEL, border_color=ACCENT, text_color=SUCCESS_CLR
        )
        self._video_entry.grid(row=1, column=0, columnspan=2, padx=18, pady=(0,6), sticky="ew")

        ctk.CTkButton(
            card, text="Browse Video", width=140, height=36,
            corner_radius=8, font=("Segoe UI", 12, "bold"),
            fg_color=ACCENT, hover_color="#2568c8",
            command=self._browse_video
        ).grid(row=2, column=0, padx=18, pady=(0,14), sticky="w")

        # Output folder row
        ctk.CTkLabel(card, text="📁  Output Folder", font=("Segoe UI", 12, "bold"),
                     text_color=TEXT_MAIN).grid(row=3, column=0, padx=18, pady=(4,4), sticky="w")

        self._output_entry = ctk.CTkEntry(
            card, textvariable=self._output_dir,
            height=40, corner_radius=8, font=("Segoe UI", 12, "bold"),
            fg_color=BG_PANEL, border_color="#444466", text_color=SUCCESS_CLR
        )
        self._output_entry.grid(row=4, column=0, columnspan=2, padx=18, pady=(0,6), sticky="ew")

        ctk.CTkButton(
            card, text="Browse Folder", width=140, height=36,
            corner_radius=8, font=("Segoe UI", 12, "bold"),
            fg_color="#333355", hover_color="#444477",
            command=self._browse_output
        ).grid(row=5, column=0, padx=18, pady=(0,18), sticky="w")

        # Optional Facecam row
        ctk.CTkLabel(card, text="📹  Facecam File (Optional / Separate Recording)", 
                     font=("Segoe UI", 12, "bold"),
                     text_color="#F1C40F").grid(row=6, column=0, padx=18, pady=(4,4), sticky="w")

        self._facecam_entry = ctk.CTkEntry(
            card, textvariable=self._facecam_path,
            placeholder_text="Select your separate facecam recording (leave blank if combined)...",
            height=40, corner_radius=8, font=("Segoe UI", 11, "bold"),
            fg_color="#12121A", border_color="#444466", text_color=SUCCESS_CLR
        )
        self._facecam_entry.grid(row=7, column=0, columnspan=2, padx=18, pady=(0,6), sticky="ew")

        ctk.CTkButton(
            card, text="Browse Facecam", width=140, height=36,
            corner_radius=8, font=("Segoe UI", 12, "bold"),
            fg_color="#A3810F", hover_color="#8B6D0D",
            command=self._browse_facecam
        ).grid(row=8, column=0, padx=18, pady=(0,18), sticky="w")

        # ── Pro Edition Settings Card ────────────
        pro_card = ctk.CTkFrame(body, fg_color=BG_CARD, corner_radius=14)
        pro_card.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(14, 0))
        pro_card.columnconfigure(0, weight=1)

        ctk.CTkLabel(pro_card, text="💎  Pro Edition Settings", font=("Segoe UI", 13, "bold"),
                     text_color="#F1C40F").grid(row=0, column=0, padx=18, pady=(18, 10), sticky="w")

        ctk.CTkCheckBox(pro_card, text="Enable Split-Screen (Facecam)", 
                         variable=self._use_facecam, font=FONT_SUB,
                         fg_color=ACCENT, hover_color="#2568c8").grid(row=1, column=0, padx=18, pady=8, sticky="w")
        
        ctk.CTkLabel(pro_card, text="Split Mode puts facecam atop gameplay.", 
                     font=("Segoe UI", 10), text_color=TEXT_DIM).grid(row=2, column=0, padx=42, pady=(0, 8), sticky="w")

        ctk.CTkCheckBox(pro_card, text="Burn-in Logo Watermark", 
                         variable=self._use_watermark, font=FONT_SUB,
                         fg_color=ACCENT, hover_color="#2568c8").grid(row=3, column=0, padx=18, pady=8, sticky="w")

        ctk.CTkLabel(pro_card, text="Channel Name (for Watermark):", 
                     font=("Segoe UI", 11, "bold"), text_color=TEXT_MAIN).grid(row=4, column=0, padx=18, pady=(12, 4), sticky="w")

        self._channel_entry = ctk.CTkEntry(
            pro_card, textvariable=self._channel_name,
            height=36, corner_radius=8, font=FONT_SUB,
            fg_color=BG_PANEL, border_color="#444466", text_color=TEXT_MAIN
        )
        self._channel_entry.grid(row=5, column=0, padx=18, pady=(0, 6), sticky="ew")

        ctk.CTkLabel(pro_card, text="Logo Image (Optional):", 
                     font=("Segoe UI", 11, "bold"), text_color=TEXT_MAIN).grid(row=6, column=0, padx=18, pady=(12, 4), sticky="w")

        logo_row = ctk.CTkFrame(pro_card, fg_color="transparent")
        logo_row.grid(row=7, column=0, padx=18, pady=(0, 18), sticky="ew")
        logo_row.columnconfigure(0, weight=1)

        self._logo_entry = ctk.CTkEntry(
            logo_row, textvariable=self._logo_path,
            placeholder_text="Leave blank for watermark.png...",
            height=32, corner_radius=8, font=("Segoe UI", 10),
            fg_color=BG_PANEL, border_color="#444466", text_color=TEXT_MAIN
        )
        self._logo_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            logo_row, text="Browse Logo", width=100, height=32,
            corner_radius=8, font=("Segoe UI", 10, "bold"),
            fg_color="#333355", hover_color="#444477",
            command=self._browse_logo
        ).grid(row=0, column=1)

        ctk.CTkCheckBox(pro_card, text="Show Hook Score Badge", 
                         variable=self._show_hook_badge, font=FONT_SUB,
                         fg_color=ACCENT, hover_color="#2568c8").grid(row=8, column=0, padx=18, pady=8, sticky="w")
        
        ctk.CTkLabel(pro_card, text="Displays virality hook score at the top left.", 
                     font=("Segoe UI", 10), text_color=TEXT_DIM).grid(row=9, column=0, padx=42, pady=(0, 8), sticky="w")

        ctk.CTkCheckBox(pro_card, text="✨ Burn-in Subtitles", 
                         variable=self._burn_subtitles, font=FONT_SUB,
                         fg_color=SUCCESS_CLR, hover_color="#27ae60").grid(row=10, column=0, padx=18, pady=8, sticky="w")
        
        ctk.CTkCheckBox(pro_card, text="🇮🇳 Romanize Hindi (Hinglish)", 
                         variable=self._use_hinglish, font=FONT_SUB,
                         fg_color="#F1C40F", hover_color="#d4ac0d").grid(row=11, column=0, padx=18, pady=8, sticky="w")
        
        ctk.CTkCheckBox(pro_card, text="Pass legacy --vision flag (ignored)", 
                         variable=self._use_vision, font=FONT_SUB,
                         fg_color=ACCENT, hover_color="#2568c8").grid(row=12, column=0, padx=18, pady=8, sticky="w")
        
        ctk.CTkLabel(pro_card, text="Multimodal visual scoring always runs now.", 
                     font=("Segoe UI", 10), text_color=TEXT_DIM).grid(row=13, column=0, padx=42, pady=(0, 8), sticky="w")

        ctk.CTkCheckBox(pro_card, text="🎬 Enable SmartSpotlight AI", 
                         variable=self._use_spotlight, font=FONT_SUB,
                         fg_color=ACCENT, hover_color="#2568c8").grid(row=14, column=0, padx=18, pady=8, sticky="w")
        
        ctk.CTkLabel(pro_card, text="AI tracks characters in vertical 9:16 view.", 
                     font=("Segoe UI", 10), text_color=TEXT_DIM).grid(row=15, column=0, padx=42, pady=(0, 8), sticky="w")

        # --- New Game Detection Section ---
        ctk.CTkLabel(pro_card, text="🎮  Game Detection (Phase 3)", font=("Segoe UI", 12, "bold"),
                     text_color="#F1C40F").grid(row=15, column=0, padx=18, pady=(12, 4), sticky="w")
        
        game_frame = ctk.CTkFrame(pro_card, fg_color=BG_PANEL, corner_radius=10)
        game_frame.grid(row=16, column=0, padx=18, pady=(0, 12), sticky="ew")
        game_frame.columnconfigure(0, weight=1)

        # Get available profiles
        profiles_dir = os.path.join(SCRIPT_DIR, "game_profiles")
        available_profiles = ["generic"]
        if os.path.exists(profiles_dir):
            available_profiles = [f.replace(".json", "") for f in os.listdir(profiles_dir) if f.endswith(".json")]

        self._game_menu = ctk.CTkOptionMenu(
            game_frame,
            values=available_profiles,
            variable=self._game_id,
            fg_color="#12121A",
            button_color=ACCENT,
            button_hover_color="#2568c8",
        )
        self._game_menu.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")

        self._profile_src_btn = ctk.CTkSegmentedButton(
            game_frame,
            values=["local", "cloud"],
            variable=self._profile_source,
            fg_color="#12121A",
            selected_color=ACCENT,
        )
        self._profile_src_btn.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")

        # Shifted export/hardware rows up/down as needed. 
        # (Row 17 -> Export, 18-22 -> Hardware)
        ctk.CTkLabel(pro_card, text="📦  Video Export Settings", font=("Segoe UI", 12, "bold"),
                     text_color=TEXT_MAIN).grid(row=17, column=0, padx=18, pady=(12, 4), sticky="w")

        exp_frame = ctk.CTkFrame(pro_card, fg_color=BG_PANEL, corner_radius=10)
        exp_frame.grid(row=18, column=0, padx=18, pady=(0, 12), sticky="ew")
        exp_frame.columnconfigure(1, weight=1)

        ctk.CTkCheckBox(exp_frame, text="Generate Shorts", variable=self._gen_short, font=FONT_SUB,
                        fg_color=ACCENT).grid(row=0, column=0, padx=12, pady=(12, 4), sticky="w")
        ctk.CTkEntry(exp_frame, textvariable=self._short_res, width=100, height=28, font=FONT_SUB,
                     fg_color=BG_CARD).grid(row=0, column=1, padx=12, pady=(12, 4), sticky="e")

        ctk.CTkCheckBox(exp_frame, text="Generate Long Highlights", variable=self._gen_long, font=FONT_SUB,
                        fg_color=ACCENT).grid(row=1, column=0, padx=12, pady=(4, 12), sticky="w")
        ctk.CTkEntry(exp_frame, textvariable=self._long_res, width=100, height=28, font=FONT_SUB,
                     fg_color=BG_CARD).grid(row=1, column=1, padx=12, pady=(4, 12), sticky="e")

        # ── Hardware Settings ──────────
        ctk.CTkLabel(pro_card, text="Hardware", font=("Segoe UI", 11, "bold"),
                     text_color=TEXT_MAIN).grid(row=19, column=0, padx=18, pady=(12, 4), sticky="w")

        self._hardware_mode_menu = ctk.CTkOptionMenu(
            pro_card,
            values=["auto", "cpu", "gpu"],
            variable=self._hardware_mode,
            command=lambda _value: self._refresh_hardware_summary(),
            fg_color=BG_PANEL,
            button_color=ACCENT,
            button_hover_color="#2568c8",
        )
        self._hardware_mode_menu.grid(row=20, column=0, padx=18, pady=(0, 8), sticky="ew")

        self._hardware_summary_label = ctk.CTkLabel(
            pro_card,
            textvariable=self._hardware_summary,
            justify="left",
            anchor="w",
            wraplength=250,
            font=("Segoe UI", 10),
            text_color=TEXT_DIM,
        )
        self._hardware_summary_label.grid(row=21, column=0, padx=18, pady=(0, 8), sticky="ew")

        self._hardware_toggle_btn = ctk.CTkButton(
            pro_card,
            text="Advanced Hardware Options",
            height=32,
            corner_radius=8,
            font=("Segoe UI", 11, "bold"),
            fg_color="#2A2A3A",
            hover_color="#3A3A52",
            command=self._toggle_hardware_advanced,
        )
        self._hardware_toggle_btn.grid(row=22, column=0, padx=18, pady=(0, 8), sticky="ew")

        self._hardware_advanced = ctk.CTkFrame(pro_card, fg_color=BG_PANEL, corner_radius=10)
        self._hardware_advanced.grid(row=23, column=0, padx=18, pady=(0, 12), sticky="ew")
        self._hardware_advanced.columnconfigure(1, weight=1)

        self._transcribe_menu = self._build_hardware_menu(self._hardware_advanced, 0, "Transcribe", self._transcribe_device)
        self._visual_menu = self._build_hardware_menu(self._hardware_advanced, 1, "Visual", self._visual_device)
        self._emotion_menu = self._build_hardware_menu(self._hardware_advanced, 2, "Emotion", self._emotion_device)
        self._detect_menu = self._build_hardware_menu(self._hardware_advanced, 3, "Detector", self._detect_device)
        self._encode_menu = self._build_hardware_menu(self._hardware_advanced, 4, "Encode", self._encode_device)
        self._spotlight_menu = self._build_hardware_menu(self._hardware_advanced, 5, "Spotlight", self._spotlight_device)
        self._hardware_advanced.grid_remove()

        # ── Job Queue Card ───────────────────────
        queue_card = ctk.CTkFrame(body, fg_color=BG_CARD, corner_radius=14)
        queue_card.grid(row=2, column=1, sticky="nsew", padx=(14, 0), pady=(0, 14))
        queue_card.columnconfigure(0, weight=1)
        queue_card.rowconfigure(1, weight=1)

        queue_header = ctk.CTkFrame(queue_card, fg_color="transparent")
        queue_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        queue_header.columnconfigure(1, weight=1)

        ctk.CTkLabel(queue_header, text="📋  Job Queue", font=("Segoe UI", 13, "bold"),
                     text_color=ACCENT).grid(row=0, column=0, sticky="w")

        queue_btn_row = ctk.CTkFrame(queue_header, fg_color="transparent")
        queue_btn_row.grid(row=0, column=1, sticky="e")

        self._process_queue_btn = ctk.CTkButton(
            queue_btn_row, text="▶ Process Queue", width=130, height=28,
            corner_radius=8, font=("Segoe UI", 11, "bold"),
            fg_color=SUCCESS_CLR, hover_color="#27ae60",
            command=self._process_queue
        )
        self._process_queue_btn.pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            queue_btn_row, text="🗑 Clear Done", width=100, height=28,
            corner_radius=8, font=("Segoe UI", 11, "bold"),
            fg_color="#2A2A3A", hover_color="#4A2222",
            command=self._clear_completed_jobs
        ).pack(side="left")

        self._queue_list = ctk.CTkScrollableFrame(queue_card, fg_color=BG_PANEL, corner_radius=8, height=180)
        self._queue_list.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

        prog_frame = ctk.CTkFrame(body, fg_color=BG_CARD, corner_radius=14)
        prog_frame.grid(row=1, column=0, sticky="nsew", pady=(0,14))
        prog_frame.columnconfigure(0, weight=1)
        prog_frame.rowconfigure(1, weight=1)

        # Top bar inside progress frame
        top_bar = ctk.CTkFrame(prog_frame, fg_color="transparent")
        top_bar.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 4))
        top_bar.columnconfigure(1, weight=1)

        ctk.CTkLabel(top_bar, text="⚡  Live Progress", font=("Segoe UI", 13, "bold"),
                     text_color=TEXT_MAIN).grid(row=0, column=0, sticky="w")

        self._phase_label = ctk.CTkLabel(top_bar, text="Idle", font=FONT_SUB,
                                          text_color=TEXT_DIM)
        self._phase_label.grid(row=0, column=1, sticky="e")

        self._progress = ctk.CTkProgressBar(prog_frame, height=8, corner_radius=4,
                                             progress_color=ACCENT, fg_color=BG_PANEL)
        self._progress.set(0)
        self._progress.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))

        # Phase step indicators
        steps_frame = ctk.CTkFrame(prog_frame, fg_color="transparent")
        steps_frame.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 10))

        self._step_labels = {}
        steps = [("🎵", "Audio"), ("🤖", "Transcribe"), ("🧠", "Rank"), ("✂️", "Clip")]
        for i, (icon, name) in enumerate(steps):
            lbl = ctk.CTkLabel(steps_frame, text=f"{icon} {name}",
                                font=("Segoe UI", 11), text_color=TEXT_DIM)
            lbl.grid(row=0, column=i, padx=12, sticky="w")
            self._step_labels[name] = lbl

        # Console output
        self._console = ctk.CTkTextbox(
            prog_frame,
            font=FONT_MONO,
            fg_color="#0A0A12",
            text_color="#C8E6C9",
            corner_radius=8,
            wrap="word",
            state="disabled"
        )
        self._console.grid(row=3, column=0, sticky="nsew", padx=18, pady=(0, 14))
        prog_frame.rowconfigure(3, weight=1)

        # Console color tags via underlying tkinter widget
        self._console._textbox.tag_config("phase",   foreground="#3B8EEA", font=("Consolas", 11, "bold"))
        self._console._textbox.tag_config("success", foreground="#2ECC71", font=("Consolas", 11, "bold"))
        self._console._textbox.tag_config("warn",    foreground="#F39C12")
        self._console._textbox.tag_config("error",   foreground="#E74C3C", font=("Consolas", 11, "bold"))
        self._console._textbox.tag_config("moment",  foreground="#A78BFA", font=("Consolas", 11, "bold"))

        # ── Bottom Action Bar ────────────────────
        action_bar = ctk.CTkFrame(body, fg_color="transparent")
        action_bar.grid(row=2, column=0, sticky="ew")
        action_bar.columnconfigure(0, weight=1)

        self._start_btn = ctk.CTkButton(
            action_bar, text="▶  Add to Queue",
            height=50, corner_radius=12,
            font=("Segoe UI", 15, "bold"),
            fg_color=ACCENT, hover_color="#2568c8",
            command=self._start_pipeline
        )
        self._start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self._stop_btn = ctk.CTkButton(
            action_bar, text="⏹  Stop",
            height=50, corner_radius=12, width=120,
            font=("Segoe UI", 13, "bold"),
            fg_color="#3A1C1C", hover_color=ERR_CLR,
            command=self._stop_pipeline,
            state="disabled"
        )
        self._stop_btn.grid(row=0, column=1, padx=(0, 10))

        ctk.CTkButton(
            action_bar, text="🗑  Clear Cache",
            height=50, corner_radius=12, width=150,
            font=("Segoe UI", 13, "bold"),
            fg_color="#2A2A3A", hover_color="#4A2222",
            command=self._clear_cache
        ).grid(row=0, column=2)

        # Status bar
        self._status_var = tk.StringVar(value="Ready — select a video to get started!")
        ctk.CTkLabel(body, textvariable=self._status_var,
                     font=("Segoe UI", 11), text_color=TEXT_DIM).grid(
            row=3, column=0, sticky="w", pady=(8, 0))

    # ──────────────────────────────────────────
    # Browse Handlers
    # ──────────────────────────────────────────
    def _build_hardware_menu(self, parent, row, label, variable):
        ctk.CTkLabel(
            parent,
            text=label,
            font=("Segoe UI", 10, "bold"),
            text_color=TEXT_MAIN,
        ).grid(row=row, column=0, padx=(12, 8), pady=6, sticky="w")
        menu = ctk.CTkOptionMenu(
            parent,
            values=["auto", "cpu", "gpu"],
            variable=variable,
            command=lambda _value: self._refresh_hardware_summary(),
            fg_color="#12121A",
            button_color=ACCENT,
            button_hover_color="#2568c8",
        )
        menu.grid(row=row, column=1, padx=(0, 12), pady=6, sticky="ew")
        return menu

    def _toggle_hardware_advanced(self):
        self._hardware_advanced_visible = not self._hardware_advanced_visible
        if self._hardware_advanced_visible:
            self._hardware_advanced.grid()
            self._hardware_toggle_btn.configure(text="Hide Advanced Hardware Options")
        else:
            self._hardware_advanced.grid_remove()
            self._hardware_toggle_btn.configure(text="Advanced Hardware Options")

    def _refresh_hardware_summary(self):
        self._hardware_summary.set("Probing hardware capabilities... 🔎")
        
        def _probe_task():
            try:
                capabilities = detect_capabilities()
                plan = plan_hardware(
                    hardware_mode=self._hardware_mode.get(),
                    stage_overrides={
                        "transcribe": self._hardware_override_value(self._transcribe_device.get()),
                        "visual": self._hardware_override_value(self._visual_device.get()),
                        "emotion": self._hardware_override_value(self._emotion_device.get()),
                        "game_detect": self._hardware_override_value(self._detect_device.get()),
                        "encode": self._hardware_override_value(self._encode_device.get()),
                        "spotlight": self._hardware_override_value(self._spotlight_device.get()),
                    },
                    capabilities=capabilities,
                )
                result = "\n".join(build_preflight_lines(plan))
                # Update UI on main thread
                self.after(0, lambda: self._hardware_summary.set(result))
            except Exception as exc:
                self.after(0, lambda: self._hardware_summary.set(f"Hardware preflight failed: {exc}"))
        
        # If we are already in a background thread (from init), just run the probe.
        # Otherwise, spin up a new thread.
        import threading
        if threading.current_thread() is threading.main_thread():
            threading.Thread(target=_probe_task, daemon=True).start()
        else:
            _probe_task()

    def _hardware_override_value(self, value):
        value = (value or "").strip().lower()
        return None if value == "auto" else value

    # ──────────────────────────────────────────
    # App Settings Persistence
    # ──────────────────────────────────────────
    def _get_pref_file(self):
        return os.path.join(SCRIPT_DIR, "streamclip_prefs.json")

    def _save_app_settings(self):
        prefs = {
            "video_path": self._video_path.get(),
            "facecam_path": self._facecam_path.get(),
            "output_dir": self._output_dir.get(),
            "logo_path": self._logo_path.get(),
            "channel_name": self._channel_name.get(),
            "use_facecam": self._use_facecam.get(),
            "use_watermark": self._use_watermark.get(),
            "show_hook_badge": self._show_hook_badge.get(),
            "use_spotlight": self._use_spotlight.get(),
            "gen_short": self._gen_short.get(),
            "gen_long": self._gen_long.get(),
            "short_res": self._short_res.get(),
            "long_res": self._long_res.get(),
            "burn_subtitles": self._burn_subtitles.get(),
            "use_hinglish": self._use_hinglish.get(),
            "hardware_mode": self._hardware_mode.get(),
            "game_id": self._game_id.get(),
            "profile_source": self._profile_source.get(),
            "detect_device": self._detect_device.get(),
        }
        try:
            with open(self._get_pref_file(), "w", encoding="utf-8") as f:
                json.dump(prefs, f, indent=2)
        except Exception as e:
            print(f"Error saving preferences: {e}")

    def _load_app_settings(self):
        pref_file = self._get_pref_file()
        if not os.path.exists(pref_file):
            return
        try:
            with open(pref_file, "r", encoding="utf-8") as f:
                prefs = json.load(f)
                
            if "video_path" in prefs: self._video_path.set(prefs["video_path"])
            if "facecam_path" in prefs: self._facecam_path.set(prefs["facecam_path"])
            if "output_dir" in prefs: self._output_dir.set(prefs["output_dir"])
            if "logo_path" in prefs: self._logo_path.set(prefs["logo_path"])
            if "channel_name" in prefs: self._channel_name.set(prefs["channel_name"])
            if "use_facecam" in prefs: self._use_facecam.set(prefs["use_facecam"])
            if "use_watermark" in prefs: self._use_watermark.set(prefs["use_watermark"])
            if "show_hook_badge" in prefs: self._show_hook_badge.set(prefs["show_hook_badge"])
            if "use_spotlight" in prefs: self._use_spotlight.set(prefs["use_spotlight"])
            if "gen_short" in prefs: self._gen_short.set(prefs["gen_short"])
            if "gen_long" in prefs: self._gen_long.set(prefs["gen_long"])
            if "short_res" in prefs: self._short_res.set(prefs["short_res"])
            if "long_res" in prefs: self._long_res.set(prefs["long_res"])
            if "burn_subtitles" in prefs: self._burn_subtitles.set(prefs["burn_subtitles"])
            if "use_hinglish" in prefs: self._use_hinglish.set(prefs["use_hinglish"])
            if "hardware_mode" in prefs: self._hardware_mode.set(prefs["hardware_mode"])
            if "game_id" in prefs: self._game_id.set(prefs["game_id"])
            if "profile_source" in prefs: self._profile_source.set(prefs["profile_source"])
            if "detect_device" in prefs: self._detect_device.set(prefs["detect_device"])
            
            # Refresh hardware labels based on loaded mode
            self._refresh_hardware_summary()
        except Exception as e:
            print(f"Error loading preferences: {e}")

    def _browse_video(self):
        initial = os.path.dirname(self._video_path.get()) if self._video_path.get() and os.path.exists(os.path.dirname(self._video_path.get())) else None
        path = filedialog.askopenfilename(
            initialdir=initial,
            title="Select your stream video",
            filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.webm"), ("All files", "*.*")]
        )
        if path:
            self._video_path.set(path)
            self._status_var.set(f"Video selected: {os.path.basename(path)}")
            self._save_app_settings()

    def _browse_output(self):
        initial = self._output_dir.get() if self._output_dir.get() and os.path.exists(self._output_dir.get()) else None
        path = filedialog.askdirectory(initialdir=initial, title="Select Output Folder for Clips")
        if path:
            self._output_dir.set(path)
            self._save_app_settings()

    def _browse_facecam(self):
        initial = os.path.dirname(self._facecam_path.get()) if self._facecam_path.get() and os.path.exists(os.path.dirname(self._facecam_path.get())) else None
        path = filedialog.askopenfilename(
            initialdir=initial,
            title="Select your separate Facecam recording",
            filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.webm"), ("All files", "*.*")]
        )
        if path:
            self._facecam_path.set(path)
            self._save_app_settings()

    def _browse_logo(self):
        initial = os.path.dirname(self._logo_path.get()) if self._logo_path.get() and os.path.exists(os.path.dirname(self._logo_path.get())) else None
        path = filedialog.askopenfilename(
            initialdir=initial,
            title="Select Logo Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")]
        )
        if path:
            self._logo_path.set(path)
            self._save_app_settings()

    # ──────────────────────────────────────────
    # Queue Management & Persistence
    # ──────────────────────────────────────────
    def _get_job_snapshot(self):
        return {
            "id": f"job_{int(time.time() * 1000)}",
            "video_path": self._video_path.get().strip(),
            "facecam_path": self._facecam_path.get().strip(),
            "output_dir": self._output_dir.get().strip(),
            "use_facecam": self._use_facecam.get(),
            "use_watermark": self._use_watermark.get(),
            "logo_path": self._logo_path.get().strip(),
            "show_hook_badge": self._show_hook_badge.get(),
            "use_vision": self._use_vision.get(),
            "use_spotlight": self._use_spotlight.get(),
            "channel_name": self._channel_name.get(),
            "hardware_mode": self._hardware_mode.get(),
            "transcribe_device": self._transcribe_device.get(),
            "visual_device": self._visual_device.get(),
            "emotion_device": self._emotion_device.get(),
            "encode_device": self._encode_device.get(),
            "spotlight_device": self._spotlight_device.get(),
            "gen_short": self._gen_short.get(),
            "gen_long": self._gen_long.get(),
            "short_res": self._short_res.get(),
            "long_res": self._long_res.get(),
            "burn_subtitles": self._burn_subtitles.get(),
            "use_hinglish": self._use_hinglish.get(),
            "game_id": self._game_id.get(),
            "profile_source": self._profile_source.get(),
            "detect_device": self._detect_device.get(),
            "status": "Pending",
            "progress": 0,
        }

    def _save_queue(self):
        try:
            # We only save Pending or Failed jobs to keep it clean, 
            # or just save everything and let the loader filter.
            with open(self._queue_file, "w", encoding="utf-8") as f:
                json.dump(self._queue, f, indent=2)
        except Exception as e:
            print(f"Error saving queue: {e}")

    def _load_queue(self):
        if not os.path.exists(self._queue_file):
            return
        try:
            with open(self._queue_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Clean up statuses on load: Running -> Pending if it was interrupted
                    for job in data:
                        if job.get("status") == "Running":
                            job["status"] = "Pending"
                    self._queue = data
                    self._refresh_queue_ui()
                    
                    # If any pending jobs exist, maybe auto-start? 
                    # For safety, let's wait for user to click "Add/Process".
        except Exception as e:
            print(f"Error loading queue: {e}")

    def _refresh_queue_ui(self):
        # Clear current list
        for child in self._queue_list.winfo_children():
            child.destroy()
            
        for i, job in enumerate(self._queue):
            self._build_queue_entry(job, i)

    def _build_queue_entry(self, job, index):
        entry = ctk.CTkFrame(self._queue_list, fg_color="#1A1A2A" if job["status"] != "Running" else "#2A2A4A", corner_radius=8)
        entry.pack(fill="x", pady=4, padx=8)
        
        status_colors = {"Pending": TEXT_DIM, "Running": ACCENT, "Done": SUCCESS_CLR, "Failed": ERR_CLR}
        status_icons = {"Pending": "⏳", "Running": "🚀", "Done": "✅", "Failed": "❌"}
        
        icon = status_icons.get(job["status"], "•")
        lbl_status = ctk.CTkLabel(entry, text=f"{icon} {job['status']}", font=("Segoe UI", 10, "bold"), 
                                  text_color=status_colors.get(job["status"], TEXT_MAIN))
        lbl_status.pack(side="left", padx=10)
        
        name = os.path.basename(job["video_path"])
        lbl_name = ctk.CTkLabel(entry, text=name, font=("Segoe UI", 11), text_color=TEXT_MAIN, anchor="w")
        lbl_name.pack(side="left", fill="x", expand=True, padx=5)
        
        if job["status"] == "Pending":
            btn_remove = ctk.CTkButton(entry, text="✕", width=30, height=24, fg_color="transparent", 
                                       hover_color="#AA3333", command=lambda idx=index: self._remove_from_queue(idx))
            btn_remove.pack(side="right", padx=5)

    def _remove_from_queue(self, index):
        if 0 <= index < len(self._queue):
            if self._queue[index]["status"] == "Running":
                self._stop_pipeline()
            self._queue.pop(index)
            self._save_queue()
            self._refresh_queue_ui()

    # ──────────────────────────────────────────
    # Pipeline Control
    # ──────────────────────────────────────────
    def _process_queue(self):
        """Start processing all pending jobs in the queue."""
        if self._running:
            self._log("⚠  Queue is already running!\n", "warn")
            return
        pending = [j for j in self._queue if j["status"] == "Pending"]
        if not pending:
            self._log("ℹ  No pending jobs in the queue. Add jobs first.\n", "warn")
            return
        self._log(f"🚀  Starting queue — {len(pending)} job(s) pending...\n")
        self._start_next_job()

    def _clear_completed_jobs(self):
        """Remove Done and Failed jobs from the queue list."""
        before = len(self._queue)
        self._queue = [j for j in self._queue if j["status"] not in ("Done", "Failed")]
        removed = before - len(self._queue)
        self._save_queue()
        self._refresh_queue_ui()
        if removed:
            self._log(f"🗑  Removed {removed} completed/failed job(s) from queue.\n", "warn")
        else:
            self._log("ℹ  No completed jobs to clear.\n")

    def _start_pipeline(self):
        video = self._video_path.get().strip()
        output = self._output_dir.get().strip()

        if not video:
            self._log("⚠  Please select a video file first, then click \"Add to Queue\".\n"
                      "   To start pending jobs, click ▶ Process Queue.\n", "warn")
            return

        if not os.path.exists(video):
            self._log(f"⚠  Video file not found:\n   {video}\n", "error")
            return
        if not output:
            self._log("⚠  Please select an output folder!\n", "error")
            return

        job = self._get_job_snapshot()
        self._queue.append(job)
        self._save_queue()
        self._save_app_settings()
        self._refresh_queue_ui()
        self._log(f"📦  Added to queue: {os.path.basename(video)}\n"
                  f"   Click ▶ Process Queue to start, or it will auto-start if idle.\n")

        # Auto-start if nothing is running
        if not self._running:
            self._start_next_job()

    def _start_next_job(self):
        # Find first pending job
        next_job = None
        for job in self._queue:
            if job["status"] == "Pending":
                next_job = job
                break
        
        if not next_job:
            self._running = False
            self._reset_ui()
            return

        self._clear_console()
        self._running = True
        self._current_job_id = next_job["id"]
        next_job["status"] = "Running"
        self._save_queue()
        self._refresh_queue_ui()

        self._start_btn.configure(text="➕  Add Another to Queue")
        self._stop_btn.configure(state="normal")
        self._process_queue_btn.configure(state="disabled")
        self._progress.set(0)
        self._phase_label.configure(text=f"Starting: {os.path.basename(next_job['video_path'])}", text_color=ACCENT)
        self._status_var.set(f"Processing: {os.path.basename(next_job['video_path'])}")

        thread = threading.Thread(target=self._run_pipeline, args=(next_job,), daemon=True)
        thread.start()

    def _stop_pipeline(self):
        if self._process:
            terminate_process_tree(self._process)
        self._running = False
        self._log("\n⏹  Pipeline stopped by user.\n", "error")
        self._reset_ui()

    def _run_pipeline(self, job):
        env = build_runtime_env()

        video_path = job["video_path"]
        output_dir = job["output_dir"]
        facecam_src = job["facecam_path"]
        
        cmd = build_pipeline_command(
            video_path,
            output_dir,
            use_facecam=job["use_facecam"],
            use_watermark=job["use_watermark"],
            use_vision=job["use_vision"],
            use_spotlight=job["use_spotlight"],
            channel_name=job["channel_name"],
            facecam_src=facecam_src or None,
            logo_src=job["logo_path"] or None,
            show_hook=job["show_hook_badge"],
            hardware_mode=job["hardware_mode"],
            transcribe_device=self._hardware_override_value(job["transcribe_device"]),
            visual_device=self._hardware_override_value(job["visual_device"]),
            emotion_device=self._hardware_override_value(job["emotion_device"]),
            encode_device=self._hardware_override_value(job["encode_device"]),
            spotlight_device=self._hardware_override_value(job["spotlight_device"]),
            gen_short=job["gen_short"],
            gen_long=job["gen_long"],
            short_res=job["short_res"],
            long_res=job["long_res"],
            use_subtitles=job["burn_subtitles"],
            use_hinglish=job["use_hinglish"],
            game=job.get("game_id", "generic"),
            profile_source=job.get("profile_source", "local"),
            detect_device=self._hardware_override_value(job.get("detect_device", "auto")),
        )

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=SCRIPT_DIR,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0,
            )
            p = self._process

            for line in p.stdout:
                if not self._running:
                    break
                self.after(0, self._handle_line, line)

            p.wait()
            exit_code = p.returncode

        except Exception as e:
            self.after(0, self._log, f"\n❌  Error launching pipeline:\n{e}\n", "error")
            exit_code = -1

        self.after(0, self._pipeline_finished, exit_code)

    def _pipeline_finished(self, exit_code=0):
        # Update current job status
        current_job = None
        for job in self._queue:
            if job["id"] == self._current_job_id:
                current_job = job
                break
        
        if current_job:
            if exit_code == 0:
                current_job["status"] = "Done"
                self._log(f"\n✅  Job Complete: {os.path.basename(current_job['video_path'])}\n", "success")
            else:
                current_job["status"] = "Failed"
                self._log(f"\n❌  Job Failed (code {exit_code}): {os.path.basename(current_job['video_path'])}\n", "error")
        
        self._save_queue()
        self._refresh_queue_ui()
        
        # Move to next job if still running
        if self._running:
            self._start_next_job()
        else:
            self._reset_ui()

    def _reset_ui(self):
        self._running = False
        self._process = None
        self._start_btn.configure(state="normal", text="▶  Add to Queue")
        self._stop_btn.configure(state="disabled")
        self._process_queue_btn.configure(state="normal")

    def _clear_cache(self):
        """Scans the script directory for cache_* files and deletes them."""
        if self._running:
            self._log("⚠  Cannot clear cache while pipeline is running!\n", "warn")
            return

        cache_dirs = [
            d for d in os.listdir(os.path.join(SCRIPT_DIR, "cache"))
            if os.path.isdir(os.path.join(SCRIPT_DIR, "cache", d))
        ] if os.path.exists(os.path.join(SCRIPT_DIR, "cache")) else []

        if not cache_dirs:
            self._log("✅  No cache directories found — already clean!\n", "success")
            self._status_var.set("Cache is already empty.")
            return

        # Calculate total size
        total_bytes = 0
        file_count = 0
        for dname in cache_dirs:
            dpath = os.path.join(SCRIPT_DIR, "cache", dname)
            for root, dirs, files in os.walk(dpath):
                for f in files:
                    total_bytes += os.path.getsize(os.path.join(root, f))
                    file_count += 1
        
        total_mb = total_bytes / (1024 * 1024)

        # Confirm with the user
        confirm = tk.messagebox.askyesno(
            title="Clear Cache",
            message=f"Found {len(cache_dirs)} stream cache folder(s) with {file_count} file(s) totalling {total_mb:.1f} MB.\n\nDelete all of them?"
        )
        if not confirm:
            return

        import shutil
        self._log(f"\n🗑  Clearing {len(cache_dirs)} cache directory(s)...\n", "warn")
        for dname in cache_dirs:
            dpath = os.path.join(SCRIPT_DIR, "cache", dname)
            try:
                shutil.rmtree(dpath)
                self._log(f"   Deleted: {dname}/\n", "warn")
            except Exception as e:
                self._log(f"   Error deleting {dname}: {e}\n", "error")

        self._log(f"✅  Done! {total_mb:.1f} MB freed.\n", "success")
        self._status_var.set(f"Cache cleared — {total_mb:.1f} MB freed!")

    # ──────────────────────────────────────────
    # Console Logging
    # ──────────────────────────────────────────
    def _handle_line(self, line):
        line = line.rstrip()
        if not line:
            return

        # Detect phase changes for progress bar
        phase_info = detect_phase(line)
        if phase_info:
            label, progress = phase_info
            self._progress.set(progress)
            self._phase_label.configure(text=label, text_color=ACCENT)
            self._update_steps(label)

        # Choose colour tag
        tag = None
        low = line.lower()
        if "--- phase" in low or "===" in low:
            tag = "phase"
        elif "+++ found" in low or "successfully created" in low or "complete" in low:
            tag = "success"
        elif "error" in low or "failed" in low or "traceback" in low:
            tag = "error"
        elif "warning" in low or "skipping" in low:
            tag = "warn"
        elif "exciting moment" in low:
            tag = "moment"

        self._log(line + "\n", tag)

    def _log(self, text, tag=None):
        self._console.configure(state="normal")
        if tag:
            self._console._textbox.insert("end", text, tag)
        else:
            self._console._textbox.insert("end", text)
        self._console._textbox.see("end")
        self._console.configure(state="disabled")

    def _clear_console(self):
        self._console.configure(state="normal")
        self._console.delete("1.0", "end")
        self._console.configure(state="disabled")

    def _update_steps(self, phase_label):
        mapping = {
            "Audio":      ["Audio"],
            "Transcri":   ["Audio", "Transcribe"],
            "Rank":       ["Audio", "Transcribe", "Rank"],
            "Clip":       ["Audio", "Transcribe", "Rank", "Clip"],
            "Done":       ["Audio", "Transcribe", "Rank", "Clip"],
        }
        for key, active_steps in mapping.items():
            if key in phase_label:
                for name, lbl in self._step_labels.items():
                    if name in active_steps:
                        lbl.configure(text_color=SUCCESS_CLR)
                    else:
                        lbl.configure(text_color=TEXT_DIM)
                break


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = StreamClipApp()
    app.mainloop()
