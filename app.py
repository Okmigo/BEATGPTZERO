import logging
import threading
from flask import Flask, request, jsonify

import humanizer, detector, validator, config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global flag for model readiness
models_loaded = False

def async_model_loader():
    """Load all ML models in the background thread to avoid blocking Cloud Run startup probe."""
    global models_loaded
    try:
        logger.info("Background loading models...")
        _ = humanizer.paraphrase_tokenizer
        _ = detector.detector_model
        _ = validator.semantic_model
        models_loaded = True
        logger.info("All models loaded.")
    except Exception as e:
        logger.exception(f"Model loading failed: {e}")
        models_loaded = False

# Start background thread
threading.Thread(target=async_model_loader).start()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint used by Cloud Run."""
    if models_loaded:
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({"status": "loading"}), 503

@app.route('/rewrite', methods=['POST'])
def rewrite():
    """
    Accepts JSON payload: { "text": "..." }
    Returns: humanized output with similarity, perplexity, detection scores.
    """
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request."}), 400

    text = data['text']
    logger.info(f"Received input: {len(text)} characters")

    try:
        rewritten = humanizer.humanize_text(text)
        similarity = validator.semantic_similarity(text, rewritten)
        det_scores = detector.detect_text(rewritten)
        ppl_orig = humanizer.compute_perplexity(text)
        ppl_new = humanizer.compute_perplexity(rewritten)

        return jsonify({
            "original": text,
            "rewritten": rewritten,
            "semantic_similarity": round(similarity, 4),
            "detection_scores": det_scores,
            "perplexity_original": round(ppl_orig, 2),
            "perplexity_rewritten": round(ppl_new, 2)
        }), 200

    except Exception as e:
        logger.exception("Rewrite failed")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For local development only; use gunicorn in production
    app.run(host='0.0.0.0', port=config.PORT, debug=False)
