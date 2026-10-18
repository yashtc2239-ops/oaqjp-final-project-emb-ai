"""Unit tests for the EmotionDetection package."""

import unittest
from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Validate each expected dominant emotion."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotions(self, mock_post: Mock) -> None:
        """The largest returned score becomes ``dominant_emotion``."""
        test_cases = {
            "I am glad this happened.": ("joy", "joy"),
            "I am sad about the outcome.": ("sadness", "sadness"),
            "I am frightened by this noise.": ("fear", "fear"),
            "I am disgusted by this behaviour.": ("disgust", "disgust"),
            "I am angry about this delay.": ("anger", "anger"),
        }
        for statement, (expected_emotion, dominant) in test_cases.items():
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = {
                "emotionPredictions": [{"emotion": {
                    "anger": 0.0, "disgust": 0.0, "fear": 0.0,
                    "joy": 0.0, "sadness": 0.0, expected_emotion: 1.0,
                }}]
            }
            mock_post.return_value = mock_response
            with self.subTest(statement=statement):
                self.assertEqual(emotion_detector(statement)["dominant_emotion"], dominant)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_blank_text_returns_none_result(self, mock_post: Mock) -> None:
        """A Watson 400 response becomes the specified blank-input result."""
        mock_post.return_value = Mock(status_code=400)
        self.assertEqual(
            emotion_detector(""),
            {"anger": None, "disgust": None, "fear": None, "joy": None,
             "sadness": None, "dominant_emotion": None},
        )


if __name__ == "__main__":
    unittest.main()
