from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging
import os
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HumanText API",
    description="Adversarial Humanization Service - Transform AI text to bypass detection",
    version="1.0.0"
)

class HumanizeRequest(BaseModel):
    text: str = Field(..., description="The AI-generated text to humanize")
    aggressiveness: float = Field(default=0.5, ge=0.0, le=1.0, description="Transformation aggressiveness (0.0-1.0)")

class HumanizeResponse(BaseModel):
    original_text: str
    humanized_text: str
    aggressiveness: float
    transformation_summary: list[str]

# Initialize rewriter lazily on first request
rewriter = None
init_lock = asyncio.Lock()

async def get_rewriter():
    global rewriter
    if rewriter is None:
        async with init_lock:
            if rewriter is None:  # Double-check locking
                logger.info("Lazy-loading rewriter models...")
                from rewriter import Rewriter
                rewriter = Rewriter()
                logger.info("Rewriter initialized successfully")
    return rewriter

@app.get("/")
async def root():
    return {"message": "HumanText API - Adversarial Humanization Service"}

@app.get("/health")
async def health_check():
    try:
        # Lightweight health check that doesn't load models
        return {"status": "ready", "service": "humantext-api"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unavailable")

@app.post("/humanize", response_model=HumanizeResponse)
async def humanize_text(request: HumanizeRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f"Humanizing text with aggressiveness: {request.aggressiveness}")
        
        # Lazy-load models on first request
        rewriter_instance = await get_rewriter()
        
        # Process the text
        result = rewriter_instance.humanize(request.text, request.aggressiveness)
        
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
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=600)
