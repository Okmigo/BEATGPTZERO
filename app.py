import logging
import threading
from flask import Flask, request, jsonify
import json
from nltk.tokenize import sent_tokenize

import humanizer, detector, validator, config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global flag for model readiness
models_loaded = False

def async_model_loader():
    """Load all ML models in a background thread for Cloud Run startup."""
    global models_loaded
    try:
        logger.info("Background loading models...")
        _ = humanizer.paraphrase_tokenizer
        _ = detector.detector_model
        _ = validator.semantic_model
        models_loaded = True
        logger.info("All models loaded.")
    except Exception as e:
        models_loaded = False
        logger.error(f"Model loading failed: {e}")

threading.Thread(target=async_model_loader, daemon=True).start()

@app.route('/healthz', methods=['GET'])
def healthz():
    """
    Cloud Run startup probe: returns 200 only when models are loaded.
    """
    if models_loaded:
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({"status": "loading"}), 503

@app.route('/rewrite', methods=['POST'])
def rewrite():
    """
    Accepts JSON payload: { "text": "...", "user_id": "optional_user" }
    Returns: humanized output with similarity, perplexity, detection scores.
    """
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request."}), 400

    text = data['text']
    logger.info(f"Received input: {len(text)} characters")

    # Load or initialize user style profile
    user_id = data.get('user_id')
    style_data = {}
    if config.STYLE_MEMORY_ENABLED and user_id:
        try:
            with open(config.STYLE_MEMORY_FILE) as f:
                style_data = json.load(f)
        except FileNotFoundError:
            style_data = {}
        style_profile = style_data.get(user_id)
    else:
        style_profile = None

    try:
        # Iterative rewriting with detection feedback
        rewrites = []
        for _ in range(config.DETECTION_MAX_ROUNDS):
            candidate = humanizer.humanize_text(text, style_profile)
            scores = detector.detect_text(candidate)
            rewrites.append((candidate, scores))
            if scores["gpt2_score"] < config.DETECTION_THRESHOLD:
                break
        # Choose best candidate (lowest GPT-2 detection score)
        rewritten, best_scores = min(rewrites, key=lambda x: x[1]["gpt2_score"])
        det_scores = best_scores

        # Compute semantic similarity and perplexities
        similarity = validator.semantic_similarity(text, rewritten)
        ppl_orig = humanizer.compute_perplexity(text)
        ppl_new = humanizer.compute_perplexity(rewritten)

        # Update user style profile
        if config.STYLE_MEMORY_ENABLED and user_id:
            total_words = len(text.split())
            total_sents = len(sent_tokenize(text)) or 1
            if user_id not in style_data:
                style_data[user_id] = {"total_words": 0, "total_sents": 0}
            style_data[user_id]["total_words"] += total_words
            style_data[user_id]["total_sents"] += total_sents
            style_data[user_id]["avg_len"] = (style_data[user_id]["total_words"] 
                                              / style_data[user_id]["total_sents"])
            with open(config.STYLE_MEMORY_FILE, "w") as f:
                json.dump(style_data, f)

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
    # For local development; use gunicorn in production
    app.run(host='0.0.0.0', port=config.PORT, debug=False)
