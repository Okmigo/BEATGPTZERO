import random
import re
import nltk
import textwrap
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from detector import detect_ai
from validator import stylometric_profile

nltk.download('punkt')
nltk.download('wordnet')

# --- Helper functions ---

def synonym_swap(word):
    synsets = wn.synsets(word)
    synonyms = [lemma.name().replace("_", " ") for syn in synsets for lemma in syn.lemmas()
                if lemma.name().lower() != word.lower()]
    return random.choice(synonyms) if synonyms else word

def chunk_text(text, max_len=180):
    sentences = sent_tokenize(text)
    chunks, chunk = [], []
    curr_len = 0
    for sentence in sentences:
        if curr_len + len(sentence.split()) <= max_len:
            chunk.append(sentence)
            curr_len += len(sentence.split())
        else:
            chunks.append(" ".join(chunk))
            chunk = [sentence]
            curr_len = len(sentence.split())
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def human_variation(sentence: str) -> str:
    sentence = re.sub(r'\b(it is|it’s|there is|there are|this is)\b', '', sentence, flags=re.I)
    sentence = sentence.replace("—", "-").replace("..", ".")
    sentence = re.sub(r'\butilize\b', 'use', sentence)
    return sentence

def restructure_sentence(sentence: str) -> str:
    # Occasionally shuffle clauses if safe
    parts = re.split(r'[,;:]-?', sentence)
    if len(parts) > 1 and random.random() < 0.4:
        random.shuffle(parts)
    return ', '.join(part.strip() for part in parts)

def enrich_with_literary_tone(sentence: str) -> str:
    # Introduce human-style language unpredictability
    openings = [
        "Let’s consider this:",
        "In many ways,",
        "Oddly enough,",
        "Take this for example—",
        "To put it plainly,",
        "At its core,"
    ]
    if random.random() < 0.3:
        sentence = random.choice(openings) + " " + sentence
    if random.random() < 0.3:
        sentence = sentence.replace("and", random.choice(["and also", "and even", "as well as"]))
    return sentence

def stylize_sentence(sentence: str, user_tone: str) -> str:
    sentence = human_variation(sentence)
    sentence = restructure_sentence(sentence)
    sentence = enrich_with_literary_tone(sentence)
    if user_tone == "academic":
        sentence = sentence.replace("kind of", "somewhat").replace("a lot of", "numerous")
    return sentence

def estimate_tone(text: str) -> str:
    formal_markers = ["therefore", "however", "thus", "notably", "significant", "framework", "systemic"]
    informal_markers = ["like", "kind of", "a lot", "really", "basically", "actually"]
    f = sum(text.count(w) for w in formal_markers)
    i = sum(text.count(w) for w in informal_markers)
    return "academic" if f >= i else "conversational"

# --- Rewrite logic ---

def rewrite(text: str) -> str:
    tone = estimate_tone(text)
    chunks = chunk_text(text)
    rewritten_chunks = []

    for chunk in chunks:
        sentences = sent_tokenize(chunk)
        rewritten_sentences = []
        for s in sentences:
            styled = stylize_sentence(s, user_tone=tone)
            rewritten_sentences.append(styled)
        rewritten_chunks.append(" ".join(rewritten_sentences))

    return "\n\n".join(rewritten_chunks)

# --- Entry point for the API ---

def humanize_text(text: str) -> dict:
    original_profile = stylometric_profile(text)
    rewritten = rewrite(text)
    final_profile = stylometric_profile(rewritten)
    ai_score = detect_ai(rewritten)

    return {
        "original_text": text,
        "humanized_text": rewritten,
        "original_profile": original_profile,
        "final_profile": final_profile,
        "is_humanized": ai_score < 30,
        "perplexity": None,
        "burstiness": final_profile["sentence_length_std"],
    }
