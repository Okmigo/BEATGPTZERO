from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging
import os
import traceback
from rewriter import Rewriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HumanText API",
    description="Adversarial Humanization Service - Transform AI text to bypass detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

class HumanizeRequest(BaseModel):
    text: str = Field(..., description="The AI-generated text to humanize")
    aggressiveness: float = Field(default=0.7, ge=0.0, le=1.0, 
                                 description="Transformation intensity (0.0-1.0)")

class HumanizeResponse(BaseModel):
    original_text: str
    humanized_text: str
    aggressiveness: float
    transformation_summary: list[str]

# Initialize rewriter during startup
rewriter = None

@app.on_event("startup")
def startup_event():
    global rewriter
    logger.info("Starting service initialization")
    
    try:
        rewriter = Rewriter(model_name="t5-base", max_retries=5)
        logger.info("Rewriter initialized successfully")
        
        # Test with a small text
        test_text = "Utilize advanced algorithms to optimize performance."
        result = rewriter.humanize(test_text, 0.7)
        logger.info(f"Test transformation: '{test_text}' → '{result['humanized_text']}'")
        
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}")
        logger.critical(traceback.format_exc())
        raise RuntimeError("Service initialization failed")

@app.get("/")
def root():
    return {
        "service": "HumanText API",
        "status": "operational" if rewriter else "initializing",
        "endpoint": "/humanize",
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "ready" if rewriter else "initializing",
        "service": "humantext-api",
        "model_loaded": bool(rewriter and rewriter._is_initialized)
    }

@app.post("/humanize", response_model=HumanizeResponse)
def humanize_text(request: HumanizeRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f"Humanization request (aggressiveness: {request.aggressiveness})")
        
        if not rewriter:
            raise HTTPException(status_code=503, detail="Service initializing")
        
        result = rewriter.humanize(request.text, request.aggressiveness)
        
        return {
            "original_text": request.text,
            "humanized_text": result["humanized_text"],
            "aggressiveness": request.aggressiveness,
            "transformation_summary": result["transformation_summary"]
        }
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Processing error: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        timeout_keep_alive=600,
        log_level="info",
        access_log=False
    )
