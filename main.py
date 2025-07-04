from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = FastAPI()

GPT2_MODEL = GPT2LMHeadModel.from_pretrained("gpt2")
GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")

@app.get("/")
def root():
    return {"status": "ok"}
