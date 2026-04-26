import json
import tempfile
import unittest
from pathlib import Path

import hardware


class HardwarePlannerTests(unittest.TestCase):
    def _caps(self, *, torch_gpu=False, whisper_gpu=False, onnx_gpu=False, nvenc=False):
        return hardware.HardwareCapabilities(
            torch_cuda_available=torch_gpu,
            torch_device_name="Test GPU" if torch_gpu else None,
            whisper_gpu_available=whisper_gpu,
            whisper_gpu_reason="Whisper GPU available" if whisper_gpu else "Whisper GPU unavailable",
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if onnx_gpu else ["CPUExecutionProvider"],
            onnx_cuda_available=onnx_gpu,
            onnx_gpu_reason="Emotion GPU available" if onnx_gpu else "Emotion GPU unavailable",
            nvenc_available=nvenc,
            nvenc_reason="NVENC available" if nvenc else "NVENC unavailable",
        )

    def test_auto_mode_prefers_fastest_successful_device_from_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "hardware_profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "visual": {
                            "cpu": {
                                "avg_seconds_per_input_minute": 24.0,
                                "samples": 2,
                                "last_success": "2026-04-10T00:00:00Z",
                                "last_error": None,
                            },
                            "gpu": {
                                "avg_seconds_per_input_minute": 10.0,
                                "samples": 2,
                                "last_success": "2026-04-10T00:00:00Z",
                                "last_error": None,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            plan = hardware.plan_hardware(
                hardware_mode="auto",
                stage_overrides={},
                capabilities=self._caps(torch_gpu=True),
                profile_path=str(profile_path),
                stages=("visual",),
            )

            self.assertEqual(plan.stage_device("visual"), "gpu")
            self.assertIn("profile favors GPU", plan.stage_reason("visual"))

    def test_global_gpu_mode_fails_fast_when_stage_has_no_gpu_capability(self):
        with self.assertRaises(RuntimeError):
            hardware.plan_hardware(
                hardware_mode="gpu",
                stage_overrides={},
                capabilities=self._caps(torch_gpu=True, whisper_gpu=True, onnx_gpu=False, nvenc=True),
                stages=("emotion",),
            )

    def test_stage_override_wins_over_global_mode(self):
        plan = hardware.plan_hardware(
            hardware_mode="gpu",
            stage_overrides={"encode": "cpu"},
            capabilities=self._caps(torch_gpu=True, whisper_gpu=True, onnx_gpu=True, nvenc=True),
            stages=("encode",),
        )

        self.assertEqual(plan.stage_device("encode"), "cpu")
        self.assertFalse(plan.stage_strict("encode"))
        self.assertEqual(plan.stage_reason("encode"), "forced CPU")


class HardwareProfileSmokeTests(unittest.TestCase):
    def test_record_stage_metric_writes_normalized_throughput(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "hardware_profile.json"

            hardware.record_stage_metric(
                "transcribe",
                "cpu",
                input_minutes=2.0,
                elapsed_seconds=30.0,
                success=True,
                profile_path=str(profile_path),
            )

            payload = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertIn("transcribe", payload)
            self.assertEqual(payload["transcribe"]["cpu"]["samples"], 1)
            self.assertAlmostEqual(payload["transcribe"]["cpu"]["avg_seconds_per_input_minute"], 15.0)
            self.assertIsNotNone(payload["transcribe"]["cpu"]["last_success"])


if __name__ == "__main__":
    unittest.main()
