from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import logging
from datetime import datetime
import json

from rewriter import AdvancedTextRewriter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Text Transformation API",
    description="Research-grade text transformation system for linguistic analysis",
    version="1.0.0"
)

class TextTransformRequest(BaseModel):
    text: str
    aggressiveness: float = 0.7  # 0.0 to 1.0 transformation intensity
    preserve_length: bool = False
    target_style: str = "casual"  # casual, formal, academic, conversational
    
class TextTransformResponse(BaseModel):
    transformed_text: str
    transformation_summary: str
    original_length: int
    transformed_length: int
    applied_transformations: List[str]
    confidence_score: float
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# Initialize the text rewriter
rewriter = AdvancedTextRewriter()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )

@app.post("/transform", response_model=TextTransformResponse)
async def transform_text(request: TextTransformRequest):
    """
    Transform text using advanced linguistic processing techniques.
    
    This endpoint applies sophisticated text transformation algorithms
    that analyze and modify text based on linguistic patterns found
    in natural human writing.
    """
    try:
        start_time = datetime.now()
        
        # Input validation
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        if len(request.text) > 50000:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Text input too long (max 50,000 characters)")
        
        if not 0.0 <= request.aggressiveness <= 1.0:
            raise HTTPException(status_code=400, detail="Aggressiveness must be between 0.0 and 1.0")
        
        # Perform transformation
        result = rewriter.transform_text(
            text=request.text,
            aggressiveness=request.aggressiveness,
            preserve_length=request.preserve_length,
            target_style=request.target_style
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Log the transformation for monitoring
        logger.info(f"Text transformation completed in {processing_time:.2f}ms")
        
        return TextTransformResponse(
            transformed_text=result['transformed_text'],
            transformation_summary=result['transformation_summary'],
            original_length=len(request.text),
            transformed_length=len(result['transformed_text']),
            applied_transformations=result['applied_transformations'],
            confidence_score=result['confidence_score'],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error during text transformation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Advanced Text Transformation API",
        "documentation": "/docs",
        "health": "/health",
        "transform_endpoint": "/transform"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
