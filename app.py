import logging
from flask import Flask, request, jsonify
import humanizer, detector, validator, config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Flag set when models are loaded
models_loaded = False

def load_models():
    """Load all models at startup to avoid cold-start delay."""
    global models_loaded
    if not models_loaded:
        # Trigger model loading by importing/initializing (models load on first use)
        # For example, humanizer module loads paraphraser and GPT2, detector loads RoBERTa, etc.
        logger.info("Loading models for first time...")
        # Dummy calls to ensure model instances are created
        _ = humanizer.paraphrase_tokenizer
        _ = detector.detector_model
        _ = validator.semantic_model
        models_loaded = True
        logger.info("All models loaded.")

# Call load_models() at startup
load_models()

@app.route('/health', methods=['GET'])
def health():
    """Health check: return 200 if models are loaded, else 503."""
    if models_loaded:
        return jsonify({"status": "ok"}), 200
    else:
        return jsonify({"status": "loading"}), 503

@app.route('/rewrite', methods=['POST'])
def rewrite():
    """
    Rewrite endpoint. Expects JSON { "text": "<input text>" }.
    Returns JSON with rewritten text and diagnostic info.
    """
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input, expected JSON with 'text'."}), 400

    original = data['text']
    logger.info(f"Received text of length {len(original)}")

    try:
        # Perform stylometric transformation
        rewritten = humanizer.humanize_text(original)

        # Semantic similarity check
        sim_score = validator.semantic_similarity(original, rewritten)
        logger.info(f"Semantic similarity score: {sim_score:.3f}")

        # Detection scoring
        det_scores = detector.detect_text(rewritten)
        logger.info(f"Detector scores (GPT2={det_scores['gpt2_score']:.3f}, Human={det_scores['human_score']:.3f})")

        # Compute GPT-2 perplexity for original and rewritten (optional diagnostics)
        ppl_orig = humanizer.compute_perplexity(original)
        ppl_new  = humanizer.compute_perplexity(rewritten)
        logger.info(f"Perplexity (orig: {ppl_orig:.1f}, rewritten: {ppl_new:.1f})")

        response = {
            "original": original,
            "rewritten": rewritten,
            "semantic_similarity": sim_score,
            "detection_scores": det_scores,
            "perplexity_original": ppl_orig,
            "perplexity_rewritten": ppl_new
        }
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Rewrite failed")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use Gunicorn in production; this is for local debug
    app.run(host='0.0.0.0', port=config.PORT, debug=False)
