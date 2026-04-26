import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import clipper
import gui
import llm_selector
import main
import runtime_env


class WatermarkCliTests(unittest.TestCase):
    def test_main_parser_defaults_watermark_on(self):
        parser = main.build_arg_parser()
        self.assertTrue(parser.parse_args(["video.mp4"]).watermark)
        self.assertFalse(parser.parse_args(["video.mp4", "--no-watermark"]).watermark)

    def test_clipper_parser_defaults_watermark_on(self):
        parser = clipper.build_arg_parser()
        self.assertTrue(parser.parse_args(["video.mp4", "moments.json"]).watermark)
        self.assertFalse(parser.parse_args(["video.mp4", "moments.json", "--no-watermark"]).watermark)


class MainRuntimeTests(unittest.TestCase):
    @mock.patch("main.plan_hardware")
    @mock.patch("main.detect_capabilities")
    @mock.patch("main.extract_audio")
    @mock.patch("main.subprocess.run")
    def test_main_uses_absolute_child_scripts_from_any_cwd(self, mock_run, mock_extract_audio, mock_detect, mock_plan):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                video_path = os.path.join(tmpdir, "input.mp4")
                Path(video_path).touch()
                output_dir = "nested_output"
                transcript_path = os.path.join(tmpdir, "cache_input_transcript.json")
                moments_path = os.path.join(tmpdir, "cache_input_moments.json")
                output_dir_abs = os.path.abspath(output_dir)

                def fake_run(cmd, check, cwd=None, **kwargs):
                    if cmd[1] == main.TRANSCRIBER_SCRIPT:
                        Path(transcript_path).write_text("[]", encoding="utf-8")
                    elif cmd[1] == main.LLM_SELECTOR_SCRIPT:
                        Path(moments_path).write_text(
                            json.dumps(
                                [
                                    {
                                        "start": 0,
                                        "end": 12,
                                        "category": "action",
                                        "score": 75,
                                        "reason": "highlight",
                                        "text": "nice shot",
                                    }
                                ]
                            ),
                            encoding="utf-8",
                        )
                    elif cmd[1] == main.CLIPPER_SCRIPT:
                        os.makedirs(output_dir_abs, exist_ok=True)
                        Path(output_dir_abs, "clip_001.mp4").touch()
                    return mock.Mock(returncode=0)

                mock_run.side_effect = fake_run
                mock_detect.return_value = mock.Mock()
                mock_plan.return_value = mock.Mock(stages={}, reasons={}, strict_stages={})

                main.main(video_path, output_dir=output_dir, watermark=False)

                commands = [call.args[0] for call in mock_run.call_args_list]
                self.assertEqual(commands[0][1], main.TRANSCRIBER_SCRIPT)
                self.assertEqual(commands[1][1], main.LLM_SELECTOR_SCRIPT)
                self.assertEqual(commands[2][1], main.CLIPPER_SCRIPT)
                self.assertIn("--segments-output", commands[1])
                self.assertIn("--no-watermark", commands[2])
                self.assertIn("--hardware-mode", commands[0])
                self.assertIn("--hardware-mode", commands[1])
                self.assertIn("--hardware-mode", commands[2])
                self.assertEqual(mock_run.call_args_list[0].kwargs["cwd"], main.SCRIPT_DIR)
                self.assertEqual(mock_run.call_args_list[1].kwargs["cwd"], main.SCRIPT_DIR)
                self.assertEqual(mock_run.call_args_list[2].kwargs["cwd"], main.SCRIPT_DIR)
                self.assertEqual(mock_run.call_args_list[0].kwargs["env"]["PYTHONIOENCODING"], "utf-8")
                self.assertEqual(mock_run.call_args_list[1].kwargs["env"]["PYTHONIOENCODING"], "utf-8")
                self.assertEqual(mock_run.call_args_list[2].kwargs["env"]["PYTHONIOENCODING"], "utf-8")
            finally:
                os.chdir(original_cwd)


class SelectorTests(unittest.TestCase):
    def test_find_best_clipping_moments_requires_audio_and_video_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / "transcript.json"
            transcript_path.write_text("[]", encoding="utf-8")

            with self.assertRaises(ValueError):
                llm_selector.find_best_clipping_moments(str(transcript_path))


class ClipperBrandingTests(unittest.TestCase):
    def test_branding_filter_uses_text_only_when_image_missing(self):
        vf = clipper.build_branding_filter(
            "color=c=black[outv]",
            "[outv]",
            "C\\:/Windows/Fonts/arial.ttf",
            "My:Channel",
            True,
            watermark_path=os.path.join(tempfile.gettempdir(), "missing-watermark.png"),
        )
        self.assertIn("drawtext=text='My\\:Channel'", vf)
        self.assertNotIn("movie='", vf)

    def test_branding_filter_uses_logo_and_channel_text_when_image_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            watermark_path = os.path.join(tmpdir, "watermark.png")
            Path(watermark_path).touch()
            vf = clipper.build_branding_filter(
                "color=c=black[outv]",
                "[outv]",
                "C\\:/Windows/Fonts/arial.ttf",
                "Channel",
                True,
                watermark_path=watermark_path,
            )
            self.assertIn("movie='", vf)
            self.assertIn("drawtext=text='Channel'", vf)


class ClipperExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.workdir = Path(self.tempdir.name)
        self.video = self.workdir / "video.mp4"
        self.video.touch()
        self.transcript = self.workdir / "transcript.json"
        self.transcript.write_text("[]", encoding="utf-8")
        self.moments = self.workdir / "moments.json"

    def _write_moments(self, payload):
        self.moments.write_text(json.dumps(payload), encoding="utf-8")

    @mock.patch("clipper.generate_ass_subtitle")
    @mock.patch("clipper.subprocess.check_output", return_value=b"1200")
    def test_create_clips_continues_when_long_render_fails(self, _mock_probe, mock_subtitles):
        self._write_moments(
            [
                {"start": 0, "end": 12, "category": "action", "score": 80, "reason": "one", "text": "first"},
                {"start": 30, "end": 45, "category": "action", "score": 82, "reason": "two", "text": "second"},
            ]
        )

        def fake_subtitles(_transcript_path, _start, _end, output_ass_path):
            Path(output_ass_path).touch()
            return output_ass_path

        mock_subtitles.side_effect = fake_subtitles

        run_calls = []

        def fake_run(cmd, check, stdout=None, stderr=None, cwd=None, env=None):
            run_calls.append((cmd, cwd))
            output_name = cmd[-1]
            if "_LONG_" in output_name and output_name.startswith("clip_001"):
                raise subprocess.CalledProcessError(1, cmd, stderr=b"long version failed")
            return mock.Mock(returncode=0)

        with mock.patch("clipper.subprocess.run", side_effect=fake_run):
            clipper.create_clips(
                str(self.video),
                best_moments_path=str(self.moments),
                output_dir=str(self.workdir / "out"),
                transcript_path=str(self.transcript),
                pro_settings={"use_watermark": False},
            )

        short_render_targets = [cmd[-1] for cmd, _ in run_calls if "_LONG_" not in cmd[-1]]
        self.assertTrue(any(name.startswith("clip_001") for name in short_render_targets))
        self.assertTrue(any(name.startswith("clip_002") for name in short_render_targets))
        self.assertTrue((self.workdir / "out" / "clip_001_80V_action_desc.txt").exists())
        self.assertTrue((self.workdir / "out" / "clip_002_82V_action_desc.txt").exists())

    @mock.patch("clipper.analyze_spotlight_path", return_value=[(5.0, 0.0), (5.5, -15.0)])
    @mock.patch("clipper.generate_ass_subtitle")
    @mock.patch("clipper.subprocess.check_output", return_value=b"1200")
    def test_create_clips_cleans_spotlight_temp_files(self, _mock_probe, mock_subtitles, _mock_spotlight):
        self._write_moments(
            [{"start": 5, "end": 18, "category": "action", "score": 90, "reason": "spotlight", "text": "focus"}]
        )

        def fake_subtitles(_transcript_path, _start, _end, output_ass_path):
            Path(output_ass_path).touch()
            return output_ass_path

        mock_subtitles.side_effect = fake_subtitles
        filter_args = []

        def fake_run(cmd, check, stdout=None, stderr=None, cwd=None, env=None):
            if "-vf" in cmd:
                filter_args.append(cmd[cmd.index("-vf") + 1])
            return mock.Mock(returncode=0)

        output_dir = self.workdir / "spotlight_out"
        with mock.patch("clipper.subprocess.run", side_effect=fake_run):
            clipper.create_clips(
                str(self.video),
                best_moments_path=str(self.moments),
                output_dir=str(output_dir),
                transcript_path=str(self.transcript),
                pro_settings={"use_watermark": False},
                use_spotlight=True,
            )

        self.assertTrue(filter_args)
        expected_cmd_path = clipper.escape_ffmpeg_path(str(output_dir / "spotlight_001_5.txt"))
        self.assertIn(expected_cmd_path, filter_args[0])
        self.assertEqual(list(output_dir.glob("spotlight_*.txt")), [])

    @mock.patch("clipper.generate_ass_subtitle")
    @mock.patch("clipper.subprocess.check_output", return_value=b"1200")
    @mock.patch("clipper.detect_capabilities")
    def test_create_clips_honors_encode_device_cpu(self, mock_detect, _mock_probe, mock_subtitles):
        self._write_moments(
            [{"start": 0, "end": 12, "category": "action", "score": 88, "reason": "cpu", "text": "clip"}]
        )

        def fake_subtitles(_transcript_path, _start, _end, output_ass_path):
            Path(output_ass_path).touch()
            return output_ass_path

        mock_subtitles.side_effect = fake_subtitles
        mock_detect.return_value = mock.Mock()
        ffmpeg_calls = []

        def fake_run(cmd, check, stdout=None, stderr=None, cwd=None, env=None):
            if cmd and cmd[0] == "ffmpeg":
                ffmpeg_calls.append(cmd)
            return mock.Mock(returncode=0)

        with mock.patch("clipper.subprocess.run", side_effect=fake_run):
            with mock.patch(
                "clipper.plan_hardware",
                return_value=mock.Mock(
                    stages={"visual": "cpu", "encode": "cpu", "spotlight": "cpu"},
                    reasons={"visual": "forced CPU", "encode": "forced CPU", "spotlight": "forced CPU"},
                    strict_stages={"visual": False, "encode": False, "spotlight": False},
                    stage_device=lambda stage: {"visual": "cpu", "encode": "cpu", "spotlight": "cpu"}[stage],
                    stage_reason=lambda stage: {"visual": "forced CPU", "encode": "forced CPU", "spotlight": "forced CPU"}[stage],
                    stage_strict=lambda stage: False,
                ),
            ):
                clipper.create_clips(
                    str(self.video),
                    best_moments_path=str(self.moments),
                    output_dir=str(self.workdir / "cpu_encode"),
                    transcript_path=str(self.transcript),
                    pro_settings={"use_watermark": False},
                    encode_device="cpu",
                )

        self.assertTrue(ffmpeg_calls)
        self.assertTrue(any("libx264" in cmd for cmd in ffmpeg_calls))
        self.assertFalse(any("h264_nvenc" in cmd for cmd in ffmpeg_calls if "_LONG_" not in cmd[-1]))

    @mock.patch("clipper.generate_ass_subtitle")
    @mock.patch("clipper.subprocess.check_output", return_value=b"1200")
    @mock.patch("clipper.detect_capabilities")
    def test_create_clips_passes_spotlight_device_to_analysis(self, mock_detect, _mock_probe, mock_subtitles):
        self._write_moments(
            [{"start": 5, "end": 18, "category": "action", "score": 90, "reason": "spotlight", "text": "focus"}]
        )

        def fake_subtitles(_transcript_path, _start, _end, output_ass_path):
            Path(output_ass_path).touch()
            return output_ass_path

        mock_subtitles.side_effect = fake_subtitles
        mock_detect.return_value = mock.Mock()

        with mock.patch("clipper.subprocess.run", return_value=mock.Mock(returncode=0)):
            with mock.patch(
                "clipper.plan_hardware",
                return_value=mock.Mock(
                    stages={"visual": "gpu", "encode": "cpu", "spotlight": "gpu"},
                    reasons={"visual": "forced GPU", "encode": "forced CPU", "spotlight": "following visual device"},
                    strict_stages={"visual": True, "encode": False, "spotlight": False},
                    stage_device=lambda stage: {"visual": "gpu", "encode": "cpu", "spotlight": "gpu"}[stage],
                    stage_reason=lambda stage: {"visual": "forced GPU", "encode": "forced CPU", "spotlight": "following visual device"}[stage],
                    stage_strict=lambda stage: {"visual": True, "encode": False, "spotlight": False}[stage],
                ),
            ):
                with mock.patch("clipper.analyze_spotlight_path", return_value=([(5.0, 0.0)], "gpu", None)) as mock_spotlight:
                    clipper.create_clips(
                        str(self.video),
                        best_moments_path=str(self.moments),
                        output_dir=str(self.workdir / "spotlight_gpu"),
                        transcript_path=str(self.transcript),
                        pro_settings={"use_watermark": False},
                        use_spotlight=True,
                        spotlight_device="gpu",
                        visual_device="gpu",
                    )

        self.assertEqual(mock_spotlight.call_args.kwargs["device"], "gpu")


class GuiHelperTests(unittest.TestCase):
    def test_runtime_env_sets_pythonioencoding(self):
        env = runtime_env.build_runtime_env()
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")

    def test_build_pipeline_command_explicitly_disables_watermark(self):
        cmd = gui.build_pipeline_command(
            "video.mp4",
            "out",
            use_watermark=False,
            use_vision=True,
            channel_name="Channel Name",
            hardware_mode="gpu",
            encode_device="cpu",
        )
        self.assertIn("--no-watermark", cmd)
        self.assertNotIn("--watermark", cmd)
        self.assertIn("--vision", cmd)
        self.assertIn("--hardware-mode", cmd)
        self.assertIn("--encode-device", cmd)

    @mock.patch("gui.subprocess.run")
    @mock.patch("gui.os.name", "nt")
    def test_terminate_process_tree_uses_taskkill_on_windows(self, mock_run):
        process = mock.Mock()
        process.poll.return_value = None
        process.pid = 4242

        gui.terminate_process_tree(process)

        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.args[0], ["taskkill", "/PID", "4242", "/T", "/F"])


if __name__ == "__main__":
    unittest.main()
