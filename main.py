from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from gpt_logic import infer_prompt, generate_variants, humanize_text

app = FastAPI()

# Health check for Cloud Run
@app.get("/healthz")
def healthz():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {"status": "running"}

# Pydantic request schemas
class TextInput(BaseModel):
    text: str

class GenerateInput(BaseModel):
    text: str
    prompt: str

class HumanizeInput(BaseModel):
    text: str
    variants: List[str]

# POST /infer — Step 1
@app.post("/infer")
def infer(input: TextInput):
    try:
        inferred = infer_prompt(input.text)
        if not inferred or "[ERROR]" in inferred:
            raise Exception("Gemini failed or returned an error.")
        return {"inferred_prompt": inferred}
    except Exception as e:
        print("❌ /infer failed:", str(e))
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

# POST /generate — Step 2
@app.post("/generate")
def generate(input: GenerateInput):
    try:
        outputs = generate_variants(input.prompt, input.text)
        if not outputs or any("[ERROR]" in o for o in outputs):
            raise Exception("Gemini failed during variant generation.")
        return {"variants": outputs}
    except Exception as e:
        print("❌ /generate failed:", str(e))
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

# POST /humanize — Step 3
@app.post("/humanize")
def humanize(input: HumanizeInput):
    try:
        result = humanize_text(input.text, input.variants)
        if not result or "[ERROR]" in result:
            raise Exception("Gemini failed during humanization.")
        return {"final_humanized_text": result}
    except Exception as e:
        print("❌ /humanize failed:", str(e))
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
