# detector.py: Module to score text on how likely it is AI-generated
import config
from transformers import pipeline

# Load the RoBERTa-based AI detector (fine-tuned on GPT-2 output).
_detector = pipeline(
    "text-classification",
    model=config.DETECTOR_MODEL,
    tokenizer=config.DETECTOR_MODEL
)

def score_ai(text):
    """
    Returns the estimated AI-generation probability (0-100) for the given text.
    Uses a classifier that labels text as 'Real' (human) or 'Fake' (AI).
    We interpret 'Real' score as human-likelihood.
    """
    # Classify the text (truncate if too long to fit model)
    res = _detector(text, truncation=True, max_length=config.MAX_LENGTH)[0]
    label = res['label'].lower()
    score = res['score']
    # If label is 'real', then AI-likelihood = (1 - score); if 'fake', = score
    if label == "real":
        ai_prob = (1.0 - score) * 100.0
    else:
        ai_prob = score * 100.0
    return ai_prob
