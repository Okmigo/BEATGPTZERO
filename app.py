# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformation_pipeline import TransformationPipeline
import logging

app = FastAPI(
    title="BeatGPTZero API",
    description="Transforms AI-generated text to bypass detection",
    version="1.0.0"
)

pipeline = TransformationPipeline()
logger = logging.getLogger("uvicorn")

class HumanizeRequest(BaseModel):
    text: str

@app.post("/humanize")
async def humanize_text(request: HumanizeRequest):
    """
    Transforms AI-generated text to evade detection while preserving meaning
    Input: {"text": "AI-generated content"}
    Output: {"humanized_text": "Stealth-optimized output"}
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty input text")
        
        transformed = pipeline.apply_transformations(request.text)
        return {"humanized_text": transformed}
    
    except Exception as e:
        logger.error(f"Transformation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Text processing error")
