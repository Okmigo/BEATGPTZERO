# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from datetime import datetime

# Import the new, model-based humanizer
from humanizer import TextHumanizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Definition ---
app = FastAPI(
    title="Humanizer API",
    description="Transforms AI-generated text to bypass detection using a transformer-based paraphrasing model.",
    version="2.0.0"
)

# --- Pydantic Models for API I/O ---
class HumanizeRequest(BaseModel):
    text: str

class HumanizeResponse(BaseModel):
    humanized_text: str

# --- Load Model ---
# This initializes the TextHumanizer, loading the AI model into memory.
# It's done once at startup to ensure fast responses for API calls.
try:
    humanizer = TextHumanizer()
except Exception as e:
    logger.error(f"FATAL: Could not load the TextHumanizer model. Error: {e}")
    # If the model fails to load, the app is non-functional.
    humanizer = None

# --- API Endpoints ---
@app.post("/humanize", response_model=HumanizeResponse)
async def humanize_text_endpoint(request: HumanizeRequest):
    """
    Receives raw AI-generated text and returns a humanized version.
    This endpoint targets AI detection signals like perplexity and stylometry
    by rewriting the text with a sophisticated paraphrasing model.
    """
    if not humanizer:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model is not loaded.")
    
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Input text must be at least 10 characters long.")
    
    if len(request.text) > 10000:
        raise HTTPException(status_code=413, detail="Payload too large. Maximum 10,000 characters.")

    try:
        start_time = datetime.now()
        
        # This is where the core transformation happens.
        transformed_text = humanizer.humanize(request.text)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Humanization successful. Processing time: {processing_time:.2f}ms")
        
        return HumanizeResponse(humanized_text=transformed_text)

    except Exception as e:
        logger.error(f"Error during text humanization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health", status_code=200)
async def health_check():
    """Provides a simple health check for monitoring systems."""
    return {"status": "ok" if humanizer else "degraded", "model_loaded": bool(humanizer)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
