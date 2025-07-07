from fastapi import FastAPI, Request
import uvicorn
from rewriter import rewrite_text

app = FastAPI()

@app.post("/analyze")
async def analyze(request: Request):
    data = await request.json()
    input_text = data.get("text", "")
    rewritten = rewrite_text(input_text)
    return {
        "original": input_text,
        "rewritten": rewritten,
        "bypassable": bool(rewritten)
    }

if __name__ == "__main__":
    uvicorn.run("rewrite:app", host="0.0.0.0", port=3000)
