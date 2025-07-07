from fastapi import FastAPI, Request
import uvicorn
from rewriter import rewrite_text

app = FastAPI()

@app.post("/rewrite")
async def rewrite(request: Request):
    data = await request.json()
    input_text = data.get("text", "")
    rewritten = rewrite_text(input_text)
    return {"rewritten": rewritten}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3000)
