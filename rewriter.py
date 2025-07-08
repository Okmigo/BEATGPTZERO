from flask import Flask, request, jsonify
from flask_cors import CORS
from rewrite import rewrite_text
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    original = data.get("text", "")
    if not original:
        return jsonify({"error": "Missing 'text'"}), 400
    
    logger.info(f"Processing text: {original[:50]}...")
    
    rewritten = rewrite_text(original)
    bypassable = not rewritten.startswith("[Rewrite Error]")
    
    logger.info(f"Original length: {len(original)}, Rewritten length: {len(rewritten)}")
    
    return jsonify({
        "original": original,
        "rewritten": rewritten,
        "bypassable": bypassable
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
