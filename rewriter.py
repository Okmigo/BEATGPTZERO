from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import time
import threading
import signal
from rewrite import load_model, model_loaded, model_loading_started

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Graceful shutdown handler
def handle_shutdown(signum, frame):
    logger.info("Shutdown signal received, exiting gracefully")
    os._exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)

# Track model loading state
model_loading_started = False

def start_model_loading():
    """Start model loading in background thread"""
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
    
    if not model_loaded:
        # Estimate remaining load time
        elapsed = time.time() - model_loading_started
        if elapsed < 60:
            estimate = "10-30 seconds"
        elif elapsed < 120:
            estimate = "5-10 seconds"
        else:
            estimate = "less than 5 seconds"
            
        return jsonify({
            "error": "Model is still loading. Please try again shortly.",
            "estimated_wait": estimate,
            "elapsed_seconds": round(elapsed)
        }), 503
    
    from rewrite import rewrite_text
    try:
        rewritten = rewrite_text(original)
        bypassable = not rewritten.startswith("[Rewrite Error]")
        
        logger.info(f"Original: {len(original)} chars, Rewritten: {len(rewritten)} chars")
        
        return jsonify({
            "original": original,
            "rewritten": rewritten,
            "bypassable": bypassable
        })
    except Exception as e:
        logger.error(f"Rewrite error: {str(e)}")
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500

@app.route("/", methods=["GET"])
def health_check():
    if model_loaded:
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({
            "status": "loading",
            "message": "Model is initializing"
        }), 503

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, threaded=True)
