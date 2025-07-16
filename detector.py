# detector.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the OpenAI-style RoBERTa detector model
model_name = "openai-community/roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def detect(text: str) -> float:
    """
    Returns a confidence score between 0 and 1 indicating the likelihood the text is AI-generated.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    ai_prob = probs[0][1].item()  # Index 1 is AI class
    return ai_prob

def detect_ai(text: str) -> float:
    """
    Proxy function for backward compatibility.
    Returns the AI detection score for the given text.
    """
    return detect(text)
