from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from gpt_logic import infer_prompt, generate_variants, humanize_text

app = FastAPI()

# Models for incoming JSON requests
class TextInput(BaseModel):
    text: str

class GenerateInput(BaseModel):
    text: str
    prompt: str

class HumanizeInput(BaseModel):
    text: str
    variants: List[str]

@app.get("/")
def root():
    return {"status": "OK", "message": "beatgptzero-api is running 🎉"}

@app.post("/infer")
def infer(input: TextInput):
    inferred = infer_prompt(input.text)
    return {"inferred_prompt": inferred}

@app.post("/generate")
def generate(input: GenerateInput):
    outputs = generate_variants(input.prompt, input.text)
    return {"variants": outputs}

@app.post("/humanize")
def humanize(input: HumanizeInput):
    result = humanize_text(input.text, input.variants)
    return {"final_humanized_text": result}
