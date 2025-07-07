from flask import Flask, request, jsonify
from flask_cors import CORS
from rewrite import rewrite_text

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    original = data.get("text", "")
    if not original:
        return jsonify({"error": "Missing 'text'"}), 400

    rewritten = rewrite_text(original)
    return jsonify({
        "original": original,
        "rewritten": rewritten,
        "bypassable": bool(rewritten and not rewritten.startswith("[Rewrite Error]"))
    })
