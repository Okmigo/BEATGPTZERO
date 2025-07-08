from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from transformation_pipeline import TransformationPipeline
import os

# Initialize app with health check endpoint
app = FastAPI(
    title="BeatGPTZero API",
    description="Transforms AI-generated text to bypass detection",
    version="1.0.0"
)

# Global initialization
pipeline = None
logger = logging.getLogger("uvicorn")

class HumanizeRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline with lazy loading"""
    global pipeline
    logger.info("Initializing NLP pipeline...")
    pipeline = TransformationPipeline()
    logger.info("Service ready")

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {"status": "ok", "port": os.getenv("PORT", 3000)}

@app.post("/humanize")
async def humanize_text(request: HumanizeRequest):
    """
    Transforms AI-generated text to evade detection
    Input: {"text": "AI-generated content"}
    Output: {"humanized_text": "Stealth-optimized output"}
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty input text")
        
        if not pipeline:
            raise HTTPException(status_code=503, detail="Service initializing")
            
        transformed = pipeline.apply_transformations(request.text)
        return {"humanized_text": transformed}
    
    except Exception as e:
        logger.error(f"Transformation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Text processing error")
