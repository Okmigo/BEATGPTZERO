# app.py

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Setup ---
app = FastAPI(
    title="Simple Text Humanizer",
    description="A reliable API that paraphrases text to make it more human-like.",
    version="1.0.0"
)

# --- Pydantic Models ---
class HumanizeRequest(BaseModel):
    text: str

class HumanizeResponse(BaseModel):
    humanized_text: str

# --- Load Model ---
# The model is loaded ONCE when the application starts.
# Because the files were downloaded into the Docker image, this is very fast.
try:
    logger.info("Loading paraphrasing model...")
    model_name = 'tuner007/pegasus_paraphrase'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Point to the pre-downloaded model files inside the container
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/app/models')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='/app/models').to(device)
    
    logger.info(f"Model loaded successfully on device: {device}")
    model_ready = True
except Exception as e:
    logger.error(f"FATAL: Could not load model. Error: {e}", exc_info=True)
    model_ready = False

# --- API Endpoints ---
@app.get("/healthz", status_code=200)
async def health_check():
    """Simple health check to confirm the server is running."""
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model failed to load.")
    return {"status": "ok"}

@app.post("/humanize", response_model=HumanizeResponse)
async def humanize_text_endpoint(request: HumanizeRequest):
    """Paraphrases the input text."""
    if not model_ready:
        raise HTTPException(status_code=503, detail="Service is unavailable, model not loaded.")

    if not request.text or len(request.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Input text must be at least 20 characters long.")
    
    try:
        # Tokenize and generate paraphrases
        inputs = tokenizer(request.text, return_tensors='pt', truncation=True, max_length=512).to(device)
        summary_ids = model.generate(
            inputs['input_ids'], 
            num_beams=5, 
            num_return_sequences=1, 
            early_stopping=True, 
            max_length=1024
        )
        paraphrased_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return HumanizeResponse(humanized_text=paraphrased_text)

    except Exception as e:
        logger.error(f"Error during humanization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during text processing.")
