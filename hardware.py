import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROFILE_PATH = os.path.join(SCRIPT_DIR, "hardware_profile.json")
HARDWARE_STAGES = ("transcribe", "visual", "emotion", "game_detect", "encode", "spotlight")
DEVICE_POLICIES = {"auto", "cpu", "gpu"}


@dataclass
class StageMetrics:
    avg_seconds_per_input_minute: Optional[float] = None
    samples: int = 0
    last_success: Optional[str] = None
    last_error: Optional[str] = None

    @classmethod
    def from_dict(cls, payload):
        payload = payload or {}
        avg_value = payload.get("avg_seconds_per_input_minute")
        if avg_value is not None:
            avg_value = float(avg_value)
        return cls(
            avg_seconds_per_input_minute=avg_value,
            samples=int(payload.get("samples", 0) or 0),
            last_success=payload.get("last_success"),
            last_error=payload.get("last_error"),
        )

    def has_success_history(self):
        return self.avg_seconds_per_input_minute is not None and self.samples > 0 and bool(self.last_success)

    def to_dict(self):
        return {
            "avg_seconds_per_input_minute": self.avg_seconds_per_input_minute,
            "samples": int(self.samples),
            "last_success": self.last_success,
            "last_error": self.last_error,
        }


@dataclass
class HardwareCapabilities:
    platform_system: str = field(default_factory=platform.system)
    platform_release: str = field(default_factory=platform.release)
    torch_cuda_available: bool = False
    torch_device_name: Optional[str] = None
    whisper_gpu_available: bool = False
    whisper_gpu_reason: str = "CUDA transcription unavailable"
    onnx_providers: Iterable[str] = field(default_factory=list)
    onnx_cuda_available: bool = False
    onnx_gpu_reason: str = "CUDAExecutionProvider unavailable"
    nvenc_available: bool = False
    nvenc_reason: str = "NVENC unavailable"

    def gpu_available_for(self, stage):
        stage = str(stage)
        if stage in ("visual", "spotlight", "game_detect"):
            return bool(self.torch_cuda_available)
        if stage == "transcribe":
            return bool(self.whisper_gpu_available)
        if stage == "emotion":
            return bool(self.onnx_cuda_available)
        if stage == "encode":
            return bool(self.nvenc_available)
        return False

    def gpu_reason_for(self, stage):
        stage = str(stage)
        if stage in ("visual", "spotlight", "game_detect"):
            if self.torch_cuda_available:
                return f"GPU available ({self.torch_device_name or 'CUDA'})"
            return "CPU-only torch build"
        if stage == "transcribe":
            return self.whisper_gpu_reason
        if stage == "emotion":
            return self.onnx_gpu_reason
        if stage == "encode":
            return self.nvenc_reason
        return "CPU-only stage"


@dataclass
class HardwarePlan:
    hardware_mode: str
    stages: Dict[str, str]
    reasons: Dict[str, str]
    strict_stages: Dict[str, bool]
    capabilities: HardwareCapabilities

    def stage_device(self, stage):
        return self.stages[str(stage)]

    def stage_reason(self, stage):
        return self.reasons[str(stage)]

    def stage_strict(self, stage):
        return bool(self.strict_stages.get(str(stage), False))


def normalize_device_policy(value, *, default="auto"):
    value = default if value is None else str(value).strip().lower()
    if value == "cuda":
        value = "gpu"
    if value not in DEVICE_POLICIES:
        raise ValueError(f"Unsupported device policy: {value}")
    return value


def default_profile_path():
    return DEFAULT_PROFILE_PATH


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_hardware_profile(profile_path=None):
    profile_path = os.path.abspath(profile_path or DEFAULT_PROFILE_PATH)
    if not os.path.exists(profile_path):
        return {}
    try:
        with open(profile_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def save_hardware_profile(profile, profile_path=None):
    profile_path = os.path.abspath(profile_path or DEFAULT_PROFILE_PATH)
    output_dir = os.path.dirname(profile_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, sort_keys=True)


def get_stage_metrics(profile, stage, device):
    stage = str(stage)
    device = str(device)
    return StageMetrics.from_dict((profile or {}).get(stage, {}).get(device, {}))


def record_stage_metric(
    stage,
    device,
    input_minutes,
    elapsed_seconds,
    *,
    success=True,
    error=None,
    profile_path=None,
):
    stage = str(stage)
    device = str(device)
    profile = load_hardware_profile(profile_path)
    stage_metrics = get_stage_metrics(profile, stage, device)

    if success:
        normalized = float(elapsed_seconds) / max(float(input_minutes or 0.0), 1e-6)
        if stage_metrics.avg_seconds_per_input_minute is None or stage_metrics.samples <= 0:
            stage_metrics.avg_seconds_per_input_minute = normalized
        else:
            total = stage_metrics.avg_seconds_per_input_minute * stage_metrics.samples
            stage_metrics.avg_seconds_per_input_minute = (total + normalized) / (stage_metrics.samples + 1)
        stage_metrics.samples += 1
        stage_metrics.last_success = _now_iso()
        if error is None:
            stage_metrics.last_error = None
    else:
        stage_metrics.last_error = str(error or "unknown error")

    profile.setdefault(stage, {})
    profile[stage][device] = stage_metrics.to_dict()
    save_hardware_profile(profile, profile_path)
    return stage_metrics


def detect_capabilities(ffmpeg_bin="ffmpeg"):
    capabilities = HardwareCapabilities()

    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            capabilities.torch_cuda_available = True
            capabilities.torch_device_name = torch.cuda.get_device_name(0)
    except Exception:
        capabilities.torch_cuda_available = False

    try:
        import ctranslate2

        cuda_count = int(ctranslate2.get_cuda_device_count())
        if cuda_count > 0:
            capabilities.whisper_gpu_available = True
            capabilities.whisper_gpu_reason = f"CUDA transcription available ({cuda_count} device(s))"
        else:
            capabilities.whisper_gpu_reason = "ctranslate2 reports no CUDA devices"
    except Exception as exc:
        capabilities.whisper_gpu_reason = f"ctranslate2 unavailable: {exc}"

    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
        capabilities.onnx_providers = providers
        if "CUDAExecutionProvider" in providers:
            capabilities.onnx_cuda_available = True
            capabilities.onnx_gpu_reason = "CUDAExecutionProvider available"
        else:
            providers_text = ", ".join(providers) if providers else "none"
            capabilities.onnx_gpu_reason = f"CUDAExecutionProvider unavailable (providers: {providers_text})"
    except Exception as exc:
        capabilities.onnx_providers = []
        capabilities.onnx_gpu_reason = f"onnxruntime unavailable: {exc}"

    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        encoder_text = "\n".join([result.stdout or "", result.stderr or ""])
        if "h264_nvenc" in encoder_text:
            capabilities.nvenc_available = True
            capabilities.nvenc_reason = "NVENC available"
        else:
            capabilities.nvenc_reason = "FFmpeg build does not expose h264_nvenc"
    except Exception as exc:
        capabilities.nvenc_reason = f"NVENC probe failed: {exc}"

    return capabilities


def _available_devices_for_stage(stage, capabilities):
    devices = ["cpu"]
    if capabilities.gpu_available_for(stage):
        devices.append("gpu")
    return devices


def _preferred_auto_device(stage, capabilities, resolved_stages):
    if stage == "spotlight" and resolved_stages.get("visual") == "gpu" and capabilities.gpu_available_for("spotlight"):
        return "gpu"
    if stage == "spotlight" and resolved_stages.get("visual") == "cpu":
        return "cpu"
    if capabilities.gpu_available_for(stage):
        return "gpu"
    return "cpu"


def _resolve_auto_device(stage, capabilities, profile, resolved_stages):
    available_devices = _available_devices_for_stage(stage, capabilities)
    gpu_metrics = get_stage_metrics(profile, stage, "gpu")
    cpu_metrics = get_stage_metrics(profile, stage, "cpu")
    candidates = []

    if "gpu" in available_devices and gpu_metrics.has_success_history():
        candidates.append(("gpu", gpu_metrics.avg_seconds_per_input_minute))
    if "cpu" in available_devices and cpu_metrics.has_success_history():
        candidates.append(("cpu", cpu_metrics.avg_seconds_per_input_minute))

    if candidates:
        selected, selected_speed = min(candidates, key=lambda item: item[1])
        return selected, f"profile favors {selected.upper()} ({selected_speed:.2f}s/input-minute)"

    selected = _preferred_auto_device(stage, capabilities, resolved_stages)
    if selected == "gpu":
        if stage == "spotlight" and resolved_stages.get("visual") == "gpu":
            return "gpu", "following visual device; no performance history yet"
        return "gpu", "preferring GPU by default; no performance history yet"

    if capabilities.gpu_available_for(stage):
        return "cpu", "preferring CPU from linked stage; no performance history yet"
    return "cpu", f"GPU unavailable; {capabilities.gpu_reason_for(stage)}"


def _resolve_stage(stage, policy, capabilities, profile, resolved_stages):
    policy = normalize_device_policy(policy)
    if policy == "cpu":
        return "cpu", "forced CPU", False

    if policy == "gpu":
        if not capabilities.gpu_available_for(stage):
            reason = capabilities.gpu_reason_for(stage)
            
            # Category-specific fix advice
            if stage == "emotion":
                fix_cmd = "pip uninstall onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu==1.20.1"
                category = "ONNX Runtime GPU"
            elif stage == "transcribe":
                fix_cmd = "pip install ctranslate2"
                category = "CTranslate2"
            elif stage == "encode":
                fix_cmd = "Ensure NVIDIA Drivers are updated and FFmpeg supports NVENC"
                category = "FFmpeg NVENC"
            else:
                fix_cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
                category = "PyTorch CUDA"

            raise RuntimeError(
                f"[{stage.upper()}] requires GPU, but {reason}.\n\n"
                f"Category: {category}\n"
                f"Suggested Fix: {fix_cmd}"
            )
        return "gpu", "forced GPU", True

    selected, reason = _resolve_auto_device(stage, capabilities, profile, resolved_stages)
    return selected, reason, False


def plan_hardware(
    hardware_mode="auto",
    stage_overrides=None,
    *,
    capabilities=None,
    profile_path=None,
    stages=None,
):
    hardware_mode = normalize_device_policy(hardware_mode)
    capabilities = capabilities or detect_capabilities()
    profile = load_hardware_profile(profile_path)
    stages = tuple(stages or HARDWARE_STAGES)
    stage_overrides = dict(stage_overrides or {})

    resolved = {}
    reasons = {}
    strict_stages = {}

    for stage in stages:
        stage_override = stage_overrides.get(stage)
        stage_policy = normalize_device_policy(hardware_mode if stage_override is None else stage_override)
        selected, reason, strict = _resolve_stage(stage, stage_policy, capabilities, profile, resolved)
        resolved[stage] = selected
        reasons[stage] = reason
        strict_stages[stage] = strict

    return HardwarePlan(
        hardware_mode=hardware_mode,
        stages=resolved,
        reasons=reasons,
        strict_stages=strict_stages,
        capabilities=capabilities,
    )


def build_preflight_lines(plan):
    lines = []
    for stage in HARDWARE_STAGES:
        if stage not in plan.stages:
            continue
        mode = plan.stage_device(stage).upper()
        detail = plan.capabilities.gpu_reason_for(stage)
        lines.append(f"{stage}: {mode} ({plan.stage_reason(stage)}; capability: {detail})")
    return lines
