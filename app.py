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
    description="Text humanization service",
    version="2.0.0"
)

class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    aggressiveness: float = Field(0.7, ge=0.0, le=1.0)

class HumanizeResponse(BaseModel):
    humanized_text: str
    transformation_summary: list[str]

# Initialize during startup
rewriter = Rewriter()

@app.get("/health")
def health_check():
    return {"status": "ready"}

@app.post("/humanize", response_model=HumanizeResponse)
def humanize_text(request: HumanizeRequest):
    try:
        result = rewriter.humanize(request.text, request.aggressiveness)
        return {
            "humanized_text": result["humanized_text"],
            "transformation_summary": result["transformation_summary"]
        }
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(500, "Text processing error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
