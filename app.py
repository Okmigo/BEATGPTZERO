from flask import Flask, request, jsonify
from humanizer import rewrite_once
from validator import is_semantically_close
from detector import passes_detector
from config import MAX_REWRITE_ATTEMPTS

app = Flask(__name__)

def humanize_loop(text):
    for attempt in range(1, MAX_REWRITE_ATTEMPTS + 1):
        candidate = rewrite_once(text)
        ok_sem, sim = is_semantically_close(text, candidate)
        if not ok_sem:
            continue
        ok_det, ai_prob = passes_detector(candidate)
        if ok_det:
            return candidate, sim, ai_prob, attempt
    return None, sim, ai_prob, attempt

@app.route("/humanize", methods=["POST"])
def humanize_endpoint():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text.strip():
        return jsonify(error="Empty text"), 400

    result, sim, ai_prob, tries = humanize_loop(text)
    if result:
        return jsonify(
            humanized_text=result,
            semantic_similarity=round(sim, 3),
            ai_probability=round(ai_prob, 3),
            attempts=tries,
        ), 200
    return jsonify(error="Failed to humanize within limits",
                   semantic_similarity=round(sim,3),
                   ai_probability=round(ai_prob,3)), 422

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # for local testing: python app.py
    app.run(host="0.0.0.0", port=8080)
