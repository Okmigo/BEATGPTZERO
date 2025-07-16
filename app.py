# app.py: Flask API for rewriting AI-generated text
from flask import Flask, request, jsonify
import humanizer
import detector

app = Flask(__name__)

@app.route("/rewrite", methods=["POST"])
def rewrite_endpoint():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Rewrite the text and compute AI-likeness scores
    rewritten = humanizer.rewrite_text(text)
    new_score = detector.score_ai(rewritten)
    orig_score = detector.score_ai(text)

    return jsonify({
        "original_text": text,
        "original_ai_score": orig_score,
        "rewritten_text": rewritten,
        "rewritten_ai_score": new_score
    })

if __name__ == "__main__":
    # Run the app (Cloud Run will use gunicorn in production)
    app.run(host="0.0.0.0", port=8080)
