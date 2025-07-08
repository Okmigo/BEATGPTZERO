from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re
import random
import nltk
from nltk.tokenize import sent_tokenize

# Download nltk punkt for sentence tokenization
nltk.download('punkt', quiet=True)

# Load quantized paraphrasing model
tokenizer = AutoTokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")
model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/parrot_paraphraser_on_T5")

paraphraser = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

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

def humanize_text(text: str) -> str:
    """Add human-like features to text"""
    # 1. Add contractions
    for phrase, contraction in CONTRACTIONS_MAP.items():
        text = re.sub(rf'\b{re.escape(phrase)}\b', contraction, text, flags=re.IGNORECASE)
    
    # 2. Randomly add colloquial expressions
    sentences = sent_tokenize(text)
    modified_sentences = []
    
    for sentence in sentences:
        # Randomly add colloquialism at start
        if random.random() < 0.15:  # 15% chance
            sentence = random.choice(COLLOQUIALISMS).capitalize() + ", " + sentence[0].lower() + sentence[1:]
        
        # Randomly add transition word
        if random.random() < 0.1 and len(sentence.split()) > 6:  # 10% chance for longer sentences
            sentence = random.choice(TRANSITION_WORDS) + ", " + sentence[0].lower() + sentence[1:]
            
        modified_sentences.append(sentence)
    
    text = " ".join(modified_sentences)
    
    # 3. Add rhetorical questions (5% chance)
    if random.random() < 0.05 and "?" not in text:
        question_frames = [
            "But is this really the case?",
            "What does this mean for us?",
            "How should we approach this?",
            "Why does this matter?"
        ]
        text += " " + random.choice(question_frames)
    
    return text

def restructure_sentences(text: str) -> str:
    """Vary sentence structure for more human-like flow"""
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return text
    
    # Randomly combine short sentences
    new_sentences = []
    i = 0
    while i < len(sentences):
        if i < len(sentences) - 1 and len(sentences[i].split()) < 8 and len(sentences[i+1].split()) < 12:
            combined = sentences[i] + " " + sentences[i+1].lower()
            new_sentences.append(combined)
            i += 2
        else:
            new_sentences.append(sentences[i])
            i += 1
    
    # Randomly split long sentences
    final_sentences = []
    for sentence in new_sentences:
        words = sentence.split()
        if len(words) > 25 and "," in sentence:
            parts = sentence.split(",", 1)
            final_sentences.append(parts[0] + ".")
            final_sentences.append(parts[1].strip().capitalize())
        else:
            final_sentences.append(sentence)
    
    # Randomize sentence order when it makes sense
    if len(final_sentences) > 2 and random.random() < 0.3:
        if not any(s.endswith('?') for s in final_sentences):  # Don't mess with questions
            random.shuffle(final_sentences)
    
    return " ".join(final_sentences)

def rewrite_text(text: str, num_candidates: int = 3) -> str:
    """Rewrite text with enhanced human-like features"""
    try:
        # Generate multiple candidates with diverse decoding
        candidates = []
        for _ in range(num_candidates):
            outputs = paraphraser(
                f"paraphrase: {text} </s>", 
                max_length=256,
                do_sample=True,
                top_k=120,
                top_p=0.95,
                temperature=0.9,
                num_return_sequences=1
            )
            candidates.append(outputs[0]["generated_text"])
        
        # Select candidate with highest lexical diversity
        def diversity_score(t):
            words = t.split()
            unique_words = len(set(words))
            return unique_words / len(words) if words else 0
        
        best_candidate = max(candidates, key=diversity_score)
        
        # Apply humanization and restructuring
        humanized = humanize_text(best_candidate)
        restructured = restructure_sentences(humanized)
        
        return restructured
    
    except Exception as e:
        return f"[Rewrite Error]: {e}"
