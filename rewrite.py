from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re
import random
import nltk
from nltk.tokenize import sent_tokenize
import threading

# Download nltk punkt for sentence tokenization
nltk.download('punkt', quiet=True)

# Model loading status
model_loaded = False
load_lock = threading.Lock()
paraphraser = None

# Humanization parameters
CONTRACTIONS_MAP = {
    "it is": "it's", "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "is not": "isn't", "are not": "aren't", "was not": "wasn't", "were not": "weren't",
    "have not": "haven't", "has not": "hasn't", "had not": "hadn't", "will not": "won't",
    "would not": "wouldn't", "should not": "shouldn't", "can not": "can't", "could not": "couldn't",
    "I am": "I'm", "you are": "you're", "he is": "he's", "she is": "she's",
    "we are": "we're", "they are": "they're", "that is": "that's", "there is": "there's"
}

COLLOQUIALISMS = [
    "you know", "actually", "in fact", "basically", "to be honest", 
    "anyway", "sort of", "kind of", "I mean", "well", "right"
]

TRANSITION_WORDS = [
    "However", "Moreover", "Therefore", "Consequently", "Furthermore", 
    "Nonetheless", "Meanwhile", "Similarly", "Additionally", "Interestingly"
]

def load_model():
    """Load model in a thread-safe manner"""
    global model_loaded, paraphraser
    if model_loaded:
        return
    
    with load_loc
