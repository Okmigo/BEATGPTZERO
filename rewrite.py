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
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            print("🔧 Creating pipeline...")
            paraphraser = pipeline(
                "text2text-generation", 
                model=model, 
                tokenizer=tokenizer,
                device=-1,  # Force CPU-only
                framework="pt",
                torch_dtype=torch.float16
            )
            
            # Free up resources
            del model
            gc.collect()
            
            model_loaded = True
            load_time = time.time() - model_loading_started
            print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            # Fallback to simpler approach
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
        # Use contractions
        for formal, contraction in CONTRACTIONS_MAP.items():
            if formal in text:
                text = text.replace(formal, contraction)
        
        return text
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
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        results = []
        
        for sentence in sentences:
            if len(sentence) < 10:  # Skip short sentences
                results.append(sentence)
                continue
                
            # Generate paraphrases
            paraphrases = paraphraser(
                f"paraphrase: {sentence}",
                num_return_sequences=num_candidates,
                max_length=256,
                temperature=0.7,
                truncation=True
            )
            
            # Select best paraphrase
            candidates = [p['generated_text'] for p in paraphrases]
            best_candidate = max(candidates, key=lambda x: (len(x), len(set(x.split()))))
            
            # Enhance human-like qualities
            humanized = humanize_text(best_candidate)
            results.append(humanized)
            
            # Clean up memory between sentences
            del paraphrases, candidates
            gc.collect()
        
        return " ".join(results)
    
    except Exception as e:
        return f"[Rewrite Error]: {str(e)}"
