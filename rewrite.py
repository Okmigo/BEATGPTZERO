# File: rewrite.py

from fastapi import APIRouter, Request
from pydantic import BaseModel
from .rewriter import Rewriter
import random
import re

router = APIRouter()
rewriter = Rewriter()

class RewriteRequest(BaseModel):
    text: str

class RewriteResponse(BaseModel):
    original: str
    rewritten: str
    bypassable: bool
    error: str = None

def inject_variation(text):
    text = re.sub(r"\bis not\b", "isn't", text)
    text = re.sub(r"\bdoes not\b", "doesn't", text)
    text = re.sub(r"\bcannot\b", "can't", text)
    text = re.sub(r"\bit is\b", "it's", text)
    text = re.sub(r"\bthat is\b", "that's", text)
    text = re.sub(r"\bthey are\b", "they're", text)
    if random.random() < 0.5:
        text += random.choice([
            " You know what I mean?",
            " At the end of the day, that's what matters.",
            " Right?",
            " That's the tricky part, isn't it?",
            " Anyway, it’s something to think about.",
            " Honestly, that's how it usually plays out."
        ])
    return text

def vary_sentence_length(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    output = []
    for sentence in sentences:
        if len(sentence.split()) > 18 and random.random() < 0.4:
            parts = re.split(r'[;,]', sentence)
            if len(parts) > 1:
                output.extend([p.strip().capitalize() + '.' for p in parts if p.strip()])
            else:
                output.append(sentence)
        elif len(sentence.split()) < 8 and random.random() < 0.3 and len(output) > 0:
            output[-1] += ' ' + sentence.lower()
        else:
            output.append(sentence)
    return ' '.join(output)

def reorder_clauses(text):
    clauses = re.split(r'(?<=[,;])\s+', text)
    random.shuffle(clauses)
    return ' '.join(clauses)

def add_rhetorical_devices(text):
    questions = [
        "But is it really that simple?",
        "Don't we all see this happening already?",
        "What does that say about our priorities?",
        "Is that really the best we can do?"
    ]
    if random.random() < 0.3:
        insertion = random.choice(questions)
        sentences = re.split(r'(?<=[.!?]) +', text)
        idx = random.randint(0, len(sentences))
        sentences.insert(idx, insertion)
        text = ' '.join(sentences)
    return text

@router.post("/analyze", response_model=RewriteResponse)
async def analyze(req: RewriteRequest):
    try:
        rewritten = rewriter.rewrite(req.text, decoding="sampling")
        rewritten = reorder_clauses(rewritten)
        rewritten = inject_variation(rewritten)
        rewritten = add_rhetorical_devices(rewritten)
        rewritten = vary_sentence_length(rewritten)
        return RewriteResponse(original=req.text, rewritten=rewritten, bypassable=True)
    except Exception as e:
        return RewriteResponse(original=req.text, rewritten=None, bypassable=False, error=str(e))
