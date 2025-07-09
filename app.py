from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import uvicorn
from rewriter import Rewriter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HumanText API",
    description="Adversarial Humanization Service - Transform AI text to bypass detection",
    version="1.0.0"
)

# Initialize the rewriter
rewriter = Rewriter()

class HumanizeRequest(BaseModel):
    text: str = Field(..., description="The AI-generated text to humanize")
    aggressiveness: float = Field(default=0.5, ge=0.0, le=1.0, description="Transformation aggressiveness (0.0-1.0)")

class HumanizeResponse(BaseModel):
    original_text: str
    humanized_text: str
    aggressiveness: float
    transformation_summary: list[str]

@app.get("/")
async def root():
    return {"message": "HumanText API - Adversarial Humanization Service"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "humantext-api"}

@app.post("/humanize", response_model=HumanizeResponse)
async def humanize_text(request: HumanizeRequest):
    """
    Transform AI-generated text into human-like text that bypasses detection systems.
    
    Args:
        request: Contains the text to humanize and aggressiveness level
        
    Returns:
        HumanizeResponse with original text, humanized text, and transformation summary
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f"Humanizing text with aggressiveness: {request.aggressiveness}")
        
        # Process the text through the rewriter
        result = rewriter.humanize(request.text, request.aggressiveness)
        
        return HumanizeResponse(
            original_text=request.text,
            humanized_text=result["humanized_text"],
            aggressiveness=request.aggressiveness,
            transformation_summary=result["transformation_summary"]
        )
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during text processing")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
