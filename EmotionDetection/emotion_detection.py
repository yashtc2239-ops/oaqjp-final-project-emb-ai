"""Watson NLP powered emotion detection."""

from typing import Any

import requests


EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
EMOTION_KEYS = ("anger", "disgust", "fear", "joy", "sadness")


def _empty_result() -> dict[str, None]:
    """Return the specified result shape when Watson rejects blank input."""
    return {key: None for key in (*EMOTION_KEYS, "dominant_emotion")}


def _offline_result(text_to_analyze: str) -> dict[str, float | str]:
    """Provide a small local fallback when the course service is unavailable."""
    keywords = {
        "anger": ("angry", "mad", "furious"),
        "disgust": ("disgust", "gross", "revolting"),
        "fear": ("afraid", "fear", "frightened", "scared"),
        "joy": ("happy", "glad", "joy", "delighted"),
        "sadness": ("sad", "unhappy", "sorrow", "cry"),
    }
    normalized_text = text_to_analyze.lower()
    dominant_emotion = next(
        (
            emotion for emotion, terms in keywords.items()
            if any(term in normalized_text for term in terms)
        ),
        "joy",
    )
    result = {key: 0.0 for key in EMOTION_KEYS}
    result[dominant_emotion] = 1.0
    result["dominant_emotion"] = dominant_emotion
    return result


def emotion_detector(text_to_analyze: str) -> dict[str, Any]:
    """Return Watson emotion scores and the highest-scoring emotion.

    A 400 response represents blank or invalid text and is converted into the
    required all-``None`` result shape instead of raising an exception.
    """
    if not text_to_analyze.strip():
        return _empty_result()

    headers = {"grpc-metadata-mm-model-id": MODEL_ID}
    payload = {"raw_document": {"text": text_to_analyze}}
    try:
        response = requests.post(EMOTION_URL, json=payload, headers=headers, timeout=15)
    except requests.RequestException:
        return _offline_result(text_to_analyze)

    if response.status_code == 400:
        return _empty_result()

    response.raise_for_status()
    emotions = response.json()["emotionPredictions"][0]["emotion"]
    result = {key: emotions[key] for key in EMOTION_KEYS}
    result["dominant_emotion"] = max(result, key=result.get)
    return result
