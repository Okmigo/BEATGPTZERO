import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from config import DETECTOR_THRESHOLD, CUDA_DEVICE

DEVICE = f"cuda:{CUDA_DEVICE}" if CUDA_DEVICE >= 0 and torch.cuda.is_available() else "cpu"
DETECT_MODEL = "openai-community/roberta-base-openai-detector"  # free GPT-2 detector[9]
_dtok = AutoTokenizer.from_pretrained(DETECT_MODEL)
_dmodel = AutoModelForSequenceClassification.from_pretrained(DETECT_MODEL).to(DEVICE)

def passes_detector(text: str) -> (bool, float):
    tokens = _dtok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    logits  = _dmodel(**tokens).logits
    probs   = softmax(logits, dim=-1).squeeze()
    ai_prob = probs[1].item()  # label index 1 = “Fake/AI-generated”
    return ai_prob <= DETECTOR_THRESHOLD, ai_prob
