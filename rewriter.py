from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import threading

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start model loading in background
threading.Thread(target=lambda: __import__('rewrite').load_model(), daemon=True).start()

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    original = data.get("text", "")
    if not original:
        return jsonify({"error": "Missing 'text'"}), 400
    
    logger.info(f"Processing text: {original[:50]}...")
    
    from rewrite import rewrite_text, model_loaded
    if not model_loaded:
        return jsonify({"error": "Model is still loading. Please try again in a few seconds."}), 503
    
    rewritten = rewrite_text(original)
    bypassable = not rewritten.startswith("[Rewrite Error]")
    
    logger.info(f"Original length: {len(original)}, Rewritten length: {len(rewritten)}")
    
    return jsonify({
        "original": original,
        "rewritten": rewritten,
        "bypassable": bypassable
    })

@app.route("/", methods=["GET"])
def health_check():
    from rewrite import model_loaded
    status = "ready" if model_loaded else "loading"
    return jsonify({"status": status}), 200 if model_loaded else 503

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
