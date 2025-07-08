from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re
import random
import nltk
from nltk.tokenize import sent_tokenize
import threading
import os
import time
import gc

# Force offline mode to prevent network requests
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Download nltk punkt for sentence tokenization
nltk.download('punkt', quiet=True)

# Model loading status
model_loaded = False
model_loading_started = 0
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
    """Load model in a thread-safe manner with memory optimization"""
    global model_loaded, paraphraser, model_loading_started
    if model_loaded:
        return
    
    model_loading_started = time.time()
    print(f"🚀 Model loading started at {model_loading_started}")
    
    with load_lock:
        if model_loaded:
            return
            
        try:
            # Clear memory before loading
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            print("🔧 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")
            
            print("🔧 Loading model...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "prithivida/parrot_paraphraser_on_T5",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            print("🔧 Creating pipeline...")
            paraphraser = pipeline(
                "text2text-generation", 
                model=model, 
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            model_loaded = True
            load_time = time.time() - model_loading_started
            print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            # Fallback to CPU-only with reduced precision
            try:
                print("🔄 Attempting fallback loading...")
                paraphraser = pipeline(
                    "text2text-generation", 
                    model="prithivida/parrot_paraphraser_on_T5",
                    device=-1,
                    torch_dtype=torch.float16
                )
                model_loaded = True
                print("✅ Model loaded with fallback method")
            except Exception as fallback_e:
                print(f"❌ Fallback loading failed: {fallback_e}")

def humanize_text(text: str) -> str:
    """Add human-like features to text (memory-safe implementation)"""
    try:
        # Add colloquialisms
        if random.random() > 0.7 and len(text.split()) > 10:
            colloquial = random.choice(COLLOQUIALISMS)
            text = f"{colloquial}, {text[0].lower()}{text[1:]}"
        
        # Use contractions
        for formal, contraction in CONTRACTIONS_MAP.items():
            if formal in text:
                text = text.replace(formal, contraction)
        
        # Add transition words
        if random.random() > 0.8 and len(text.split()) > 15:
            transition = random.choice(TRANSITION_WORDS)
            sentences = sent_tokenize(text)
            if len(sentences) > 1:
                sentences[1] = f"{transition}, {sentences[1][0].lower()}{sentences[1][1:]}"
                text = ' '.join(sentences)
        
        return text
    except Exception:
        return text

def restructure_sentences(text: str) -> str:
    """Vary sentence structure for more human-like flow"""
    try:
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return text
        
        # Combine short sentences
        combined = []
        i = 0
        while i < len(sentences):
            if i < len(sentences)-1 and len(sentences[i].split()) < 5:
                combined.append(f"{sentences[i]} {sentences[i+1].lower()}")
                i += 2
            else:
                combined.append(sentences[i])
                i += 1
        
        return ' '.join(combined)
    except Exception:
        return text

def rewrite_text(text: str, num_candidates: int = 1) -> str:
    """Rewrite text with enhanced human-like features and memory optimization"""
    if not text.strip():
        return "[Rewrite Error]: Empty input"
    
    try:
        # Load model if needed
        if not model_loaded:
            load_model()
            if not model_loaded:
                return "[Rewrite Error]: Model not loaded - please try again later"
        
        # Split long text into chunks
        MAX_CHUNK_LENGTH = 300
        chunks = [text[i:i+MAX_CHUNK_LENGTH] for i in range(0, len(text), MAX_CHUNK_LENGTH)]
        results = []
        
        for chunk in chunks:
            # Generate paraphrases
            paraphrases = paraphraser(
                f"paraphrase: {chunk}",
                num_return_sequences=num_candidates,
                max_length=512,
                temperature=0.7,
                truncation=True
            )
            
            # Select best paraphrase
            candidates = [p['generated_text'] for p in paraphrases]
            best_candidate = max(candidates, key=lambda x: (len(x), len(set(x.split()))))
            
            # Enhance human-like qualities
            humanized = humanize_text(best_candidate)
            restructured = restructure_sentences(humanized)
            results.append(restructured)
            
            # Clean up memory between chunks
            del paraphrases, candidates
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return " ".join(results)
    
    except Exception as e:
        return f"[Rewrite Error]: {str(e)}"
