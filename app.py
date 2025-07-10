from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging
from rewriter import Rewriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HumanText API",
    description="Text humanization service to bypass AI detection",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None
)

class HumanizeRequest(BaseModel):
    text: str = Field(..., 
                     min_length=1, 
                     max_length=10000,
                     description="Text to humanize")
    aggressiveness: float = Field(0.7, 
                                 ge=0.0, 
                                 le=1.0,
                                 description="Transformation intensity (0.0-1.0)")

class HumanizeResponse(BaseModel):
    humanized_text: str = Field(..., description="Transformed text")
    transformation_summary: list[str] = Field(..., description="Applied transformations")

# Initialize the rewriter during startup
rewriter = Rewriter()

@app.on_event("startup")
def startup_event():
    logger.info("Service starting up")
    logger.info("Rewriter initialized successfully")

@app.get("/")
def root():
    return {
        "service": "HumanText API",
        "version": "2.0",
        "status": "operational",
        "endpoint": "/humanize",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "ready", "service": "humantext-api"}

@app.post("/humanize", response_model=HumanizeResponse)
def humanize_text(request: HumanizeRequest):
    """
    Transform AI-generated text into human-like text that bypasses detection systems
    
    - **text**: The AI-generated text to transform
    - **aggressiveness**: How aggressively to transform (0.0-1.0)
    """
    try:
        # Process the text through the rewriter
        result = rewriter.humanize(request.text, request.aggressiveness)
        
        return {
            "humanized_text": result["humanized_text"],
            "transformation_summary": result["transformation_summary"]
        }
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Text processing error"
        )

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        log_level="info"
    )
