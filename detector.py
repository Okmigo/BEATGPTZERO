import torch
import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the RoBERTa GPT-2 output detector
detector_tokenizer = AutoTokenizer.from_pretrained(config.DETECTOR_MODEL)
detector_model = AutoModelForSequenceClassification.from_pretrained(config.DETECTOR_MODEL)

def detect_text(text):
    """
    Returns: {'gpt2_score': float, 'human_score': float}.
    """
    inputs = detector_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = detector_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].tolist()
    # Index 0 = GPT-2 (AI), index 1 = human
    return {"gpt2_score": probs[0], "human_score": probs[1]}
