import unittest

import emotion_analyzer


class EmotionRuntimeSmokeTests(unittest.TestCase):
    def test_onnx_runtime_session_can_be_requested(self):
        session = emotion_analyzer.get_emotion_session(device="cpu")
        self.assertTrue(hasattr(session, "run"))
        self.assertEqual(session.get_inputs()[0].name, "Input3")


if __name__ == "__main__":
    unittest.main()
