import random
import re
import nltk
import torch
import numpy as np
from typing import List
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    T5ForConditionalGeneration, T5Tokenizer
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from tqdm import tqdm

# Ensure necessary NLTK resources are downloaded
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# Load models and tokenizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paraphraser
paraphrase_model = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(device)
paraphrase_tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")

# Sentence embedder
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# AI detector (Roberta-based)
detector_tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
detector_model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector").to(device)

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if "_" not in lemma.name():
                synonyms.add(lemma.name().lower())
    return list(synonyms)

def stylometric_profile(text: str) -> dict:
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    if not words:
        return {"ttr": 0, "sentence_length_mean": 0, "sentence_length_std": 0, "readability_flesch_kincaid": 0}
    sentence_lengths = [len(word_tokenize(s)) for s in sentences]
    ttr = len(set(words)) / len(words)
    mean_len = np.mean(sentence_lengths)
    std_len = np.std(sentence_lengths)
    readability = 0.39 * mean_len + 11.8 * (len(words) / len(sentences)) - 15.59
    return {
        "ttr": round(ttr, 4),
        "sentence_length_mean": round(mean_len, 2),
        "sentence_length_std": round(std_len, 2),
        "readability_flesch_kincaid": round(readability, 2)
    }

def embed(text: str) -> np.ndarray:
    return embedding_model.encode([text], convert_to_numpy=True)[0]

def cosine_similarity(vec1, vec2) -> float:
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))

def detect_ai(text: str) -> float:
    inputs = detector_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = detector_model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)
    ai_score = float(scores[0][1])
    return round(ai_score * 100, 2)

def paraphrase_sentence(sentence: str, num_return_sequences=5) -> List[str]:
    input_text = f"paraphrase: {sentence} </s>"
    input_ids = paraphrase_tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)
    outputs = paraphrase_model.generate(
        input_ids,
        max_length=256,
        num_beams=10,
        num_return_sequences=num_return_sequences,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    return list({paraphrase_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in outputs})

def rewrite(text: str) -> str:
    original_embedding = embed(text)
    sentences = sent_tokenize(text)
    rewritten = []

    for sentence in sentences:
        paraphrases = paraphrase_sentence(sentence, num_return_sequences=6)
        filtered = [s for s in paraphrases if s.lower() != sentence.lower()]
        if not filtered:
            rewritten.append(sentence)
            continue
        best = max(filtered, key=lambda s: cosine_similarity(embed(s), original_embedding))
        rewritten.append(best)

    rewritten_text = " ".join(rewritten)
    return rewritten_text
