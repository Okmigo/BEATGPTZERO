from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    original = data.get("text", "")
    if not original:
        return jsonify({"error": "Missing 'text'"}), 400

    rewritten = rewrite(original)
    return jsonify({
        "original": original,
        "rewritten": rewritten,
        "bypassable": bool(rewritten)
    })

def rewrite(text):
    words = text.split()
    if len(words) < 5:
        return None  # avoid nonsense rewrites

    # Synonym-like substitution and mild scrambling
    substitutions = {
        "technology": "tech",
        "sustainability": "eco-focus",
        "data": "information",
        "innovation": "advancement",
        "global": "worldwide",
        "resources": "assets",
        "challenge": "difficulty",
    }

    new_words = [substitutions.get(w.lower(), w) for w in words]
    random.shuffle(new_words[:max(1, len(new_words)//3)])

    return " ".join(new_words)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)