from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from gpt_logic import infer_prompt, generate_variants, humanize_text

app = FastAPI()

# Health check for Google Cloud Run
@app.get("/healthz")
def healthz():
    return {"status": "healthy"}

# Optional simple root check
@app.get("/")
def root():
    return {"status": "running"}

# Pydantic models
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
    inferred = infer_prompt(input.text)
    return {"inferred_prompt": inferred}

# POST /generate — Step 2
@app.post("/generate")
def generate(input: GenerateInput):
    outputs = generate_variants(input.prompt, input.text)
    return {"variants": outputs}

# POST /humanize — Step 3
@app.post("/humanize")
def humanize(input: HumanizeInput):
    result = humanize_text(input.text, input.variants)
    return {"final_humanized_text": result}
