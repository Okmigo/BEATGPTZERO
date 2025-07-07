# rewrite.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load paraphrasing model
model_name = "prithivida/parrot_paraphraser_on_T5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

def sample_paraphrase(text: str, max_length=256) -> str:
    input_ids = tokenizer.encode(f"paraphrase: {text} </s>", return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.9,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def inject_human_style(text: str) -> str:
    contractions = {
        "it is": "it's", "do not": "don't", "cannot": "can't",
        "you are": "you're", "they will": "they'll"
    }
    fillers = ["you know", "honestly", "anyway", "at the end of the day"]
    asides = ["but is it really?", "just saying.", "think about it."]

    for full, short in contractions.items():
        text = re.sub(rf"\b{full}\b", short, text, flags=re.IGNORECASE)

    sentences = sent_tokenize(text)
    for i in range(len(sentences)):
        if random.random() < 0.25:
            sentences[i] = f"{random.choice(fillers).capitalize()}, {sentences[i]}"
        if random.random() < 0.2:
            sentences[i] += f" {random.choice(asides)}"

    return " ".join(sentences)

def restructure_sentences(text: str) -> str:
    sentences = sent_tokenize(text)
    random.shuffle(sentences)
    return " ".join(sentences)

def rewrite_text(text: str) -> str:
    try:
        base = sample_paraphrase(text)
        shuffled = restructure_sentences(base)
        humanized = inject_human_style(shuffled)
        return humanized
    except Exception as e:
        return f"[Rewrite Error]: {e}"
