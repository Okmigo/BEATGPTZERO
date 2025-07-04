from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

GPT2_MODEL = GPT2LMHeadModel.from_pretrained("gpt2")
GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")

class TextInput(BaseModel):
    text: str

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/detect")
async def detect(input: TextInput):
    inputs = GPT2_TOKENIZER.encode(input.text, return_tensors="pt")
    with torch.no_grad():
        loss = GPT2_MODEL(inputs, labels=inputs).loss
    perplexity = torch.exp(loss).item()
    return {"perplexity": perplexity}
