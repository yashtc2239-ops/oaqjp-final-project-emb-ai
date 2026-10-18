"""Flask web interface for emotion detection."""

from flask import Flask, render_template, request

from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


@app.route("/")
def index() -> str:
    """Render the application home page."""
    return render_template("index.html")


@app.route("/emotionDetector")
def sent_detector() -> str:
    """Analyze text and return a friendly response for the web page."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    response = emotion_detector(text_to_analyze)
    dominant_emotion = response["dominant_emotion"]
    if dominant_emotion is None:
        return "Invalid text! Please try again!"

    return (
        "For the given statement, the system response is "
        f"'anger': {response['anger']}, 'disgust': {response['disgust']}, "
        f"'fear': {response['fear']}, 'joy': {response['joy']} and "
        f"'sadness': {response['sadness']}. The dominant emotion is "
        f"<b>{dominant_emotion}</b>."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
