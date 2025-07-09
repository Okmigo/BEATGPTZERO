from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from rewriter import Rewriter
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HumanText API",
    description="Adversarial Humanization Service - Transform AI text to bypass detection",
    version="1.0.0"
)

# Initialize rewriter with error handling
rewriter = None

class HumanizeRequest(BaseModel):
    text: str = Field(..., description="The AI-generated text to humanize")
    aggressiveness: float = Field(default=0.5, ge=0.0, le=1.0, description="Transformation aggressiveness (0.0-1.0)")

class HumanizeResponse(BaseModel):
    original_text: str
    humanized_text: str
    aggressiveness: float
    transformation_summary: list[str]

@app.on_event("startup")
async def startup_event():
    global rewriter
    logger.info("Starting service initialization...")
    try:
        rewriter = Rewriter()
        # Test with small text to verify models loaded
        test_result = rewriter.humanize("Test initialization", 0.1)
        logger.info("Service initialized successfully")
        logger.debug(f"Initialization test: {test_result}")
    except Exception as e:
        logger.critical(f"Initialization failed: {str(e)}")
        # Prevent deployment if critical components fail
        raise RuntimeError("Service initialization failed") from e

@app.get("/")
async def root():
    return {"message": "HumanText API - Adversarial Humanization Service"}

@app.get("/health")
async def health_check():
    if rewriter is None:
        raise HTTPException(status_code=503, detail="Service initializing")
    return {"status": "healthy", "service": "humantext-api"}

@app.post("/humanize", response_model=HumanizeResponse)
async def humanize_text(request: HumanizeRequest):
    if rewriter is None:
        raise HTTPException(status_code=503, detail="Service initializing")
    
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
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
