from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import time
import threading
from rewrite import load_model, model_loaded, model_loading_started

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track model loading state
model_loading_started = False

@app.before_first_request
def start_model_loading():
    """Start model loading on first request if not already started"""
    global model_loading_started
    if not model_loading_started:
        logger.info("Starting model loading in background thread")
        model_loading_started = True
        threading.Thread(target=load_model, daemon=True).start()

@app.route("/analyze", methods=["POST"])
def analyze():
    # Start model loading if not already started
    start_model_loading()
    
    data = request.json
    original = data.get("text", "")
    if not original:
        return jsonify({"error": "Missing 'text'"}), 400
    
    logger.info(f"Processing text: {original[:50]}...")
    
    from rewrite import model_loaded
    if not model_loaded:
        # Estimate remaining load time
        load_time = time.time() - model_loading_started
        if load_time < 30:
            estimate = "10-20 seconds"
        elif load_time < 60:
            estimate = "5-10 seconds"
        else:
            estimate = "less than 5 seconds"
            
        return jsonify({
            "error": "Model is still loading. Please try again in a few seconds.",
            "estimated_wait": estimate
        }), 503
    
    from rewrite import rewrite_text
    rewritten = rewrite_text(original)
    bypassable = not rewritten.startswith("[Rewrite Error]")
    
    logger.info(f"Original: {len(original)} chars, Rewritten: {len(rewritten)} chars")
    
    return jsonify({
        "original": original,
        "rewritten": rewritten,
        "bypassable": bypassable
    })

@app.route("/", methods=["GET"])
def health_check():
    from rewrite import model_loaded
    if model_loaded:
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({
            "status": "loading",
            "message": "Model is initializing, please try again later"
        }), 503

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
