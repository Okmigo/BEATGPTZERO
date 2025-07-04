from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
import torch
import math

app = FastAPI()

GPT2_MODEL = GPT2LMHeadModel.from_pretrained("gpt2")
GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
GPT2_MODEL.eval()

ROBERTA_MODEL = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
ROBERTA_TOKENIZER = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
ROBERTA_MODEL.eval()

if torch.cuda.is_available():
    GPT2_MODEL.cuda()
    ROBERTA_MODEL.cuda()

def calculate_perplexity(text: str) -> float:
    tokens = GPT2_TOKENIZER.encode(text, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    with torch.no_grad():
        outputs = GPT2_MODEL(tokens, labels=tokens)
        loss = outputs.loss.item()
    return math.exp(loss)

def classify_ai_text(text: str) -> float:
    tokens = ROBERTA_TOKENIZER(text, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        tokens = {k: v.cuda() for k, v in tokens.items()}
    with torch.no_grad():
        logits = ROBERTA_MODEL(**tokens).logits
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    return prob

def humanize_text(text: str) -> str:
    return text.replace("Therefore,", "So,").replace("In conclusion,", "To wrap up,").replace("Furthermore,", "Also,")

class DetectRequest(BaseModel):
    text: str
    humanize: bool = False

@app.post("/detect")
async def detect(request: DetectRequest):
    orig_text = request.text
    perplexity = calculate_perplexity(orig_text)
    ai_prob = classify_ai_text(orig_text)
    is_ai = ai_prob > 0.5

    response = {
        "original_text": orig_text,
        "perplexity_score": perplexity,
        "ai_probability": ai_prob,
        "detected_as_ai": is_ai,
        "offer_humanization": is_ai,
    }

    if is_ai and request.humanize:
        humanized = humanize_text(orig_text)
        h_perplexity = calculate_perplexity(humanized)
        h_ai_prob = classify_ai_text(humanized)
        response["humanized"] = {
            "text": humanized,
            "perplexity_score": h_perplexity,
            "ai_probability": h_ai_prob,
            "detected_as_ai": h_ai_prob > 0.5,
        }

    return response

@app.get("/ping")
def ping():
    return {"status": "ok"}
