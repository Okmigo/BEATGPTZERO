from flask import Flask, request, jsonify
from flask_cors import CORS
from rewrite import load_model, rewrite_text, model_loaded
import threading
import time

app = Flask(__name__)
CORS(app)

# Start model loading in background
threading.Thread(target=load_model, daemon=True).start()

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "Missing text"}), 400
    
    if not model_loaded:
        return jsonify({
            "status": "loading",
            "message": "Model is initializing",
            "wait_time": "30-60 seconds"
        }), 503
    
    result = rewrite_text(text)
    
    if result.startswith("[Error]"):
        return jsonify({
            "error": result.replace("[Error]: ", ""),
            "original": text,
            "rewritten": "",
            "bypassable": False
        }), 500
        
    return jsonify({
        "original": text,
        "rewritten": result,
        "bypassable": True
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ready" if model_loaded else "loading",
        "model": "loaded" if model_loaded else "initializing"
    }), 200 if model_loaded else 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, threaded=True)
