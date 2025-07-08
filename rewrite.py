from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os
import gc
import time
import threading

# Configuration
MODEL_NAME = "prithivida/parrot_paraphraser_on_T5"
MAX_INPUT_LENGTH = 200  # Reduced from 512
NUM_CANDIDATES = 1      # Only generate 1 candidate
TEMPERATURE = 0.7       # Lower randomness for stability

# Force offline mode and optimize memory
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Global state
model_loaded = False
load_lock = threading.Lock()
paraphraser = None

def load_model():
    """Safe model loader with aggressive memory management"""
    global model_loaded, paraphraser
    
    if model_loaded: 
        return

    with load_lock:
        if model_loaded:
            return

        try:
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("🔄 Loading model (this may take 2-3 minutes)...")
            start_time = time.time()

            # Load with maximum memory optimization
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )

            # Create pipeline with CPU fallback
            paraphraser = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=-1,  # Force CPU
                framework="pt",
                torch_dtype=torch.float16
            )

            model_loaded = True
            print(f"✅ Model loaded in {time.time()-start_time:.1f}s")

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            # Emergency fallback - load with absolute minimum settings
            try:
                paraphraser = pipeline(
                    "text2text-generation",
                    model=MODEL_NAME,
                    device=-1,
                    framework="pt"
                )
                model_loaded = True
                print("⚠️  Loaded with emergency fallback")
            except Exception as fallback_e:
                print(f"❌ Fallback failed: {fallback_e}")

def rewrite_text(text: str) -> str:
    """Memory-safe text rewriting with chunking"""
    if not text.strip():
        return "[Error]: Empty input"

    if not model_loaded:
        load_model()
        if not model_loaded:
            return "[Error]: Model unavailable"

    try:
        # Process in chunks if text is long
        if len(text) > MAX_INPUT_LENGTH:
            chunks = [text[i:i+MAX_INPUT_LENGTH] for i in range(0, len(text), MAX_INPUT_LENGTH)]
            results = []
            for chunk in chunks:
                results.append(_process_chunk(chunk))
                gc.collect()
            return " ".join(results)
        return _process_chunk(text)
    except Exception as e:
        return f"[Error]: {str(e)}"

def _process_chunk(chunk: str) -> str:
    """Process a single chunk of text"""
    try:
        output = paraphraser(
            f"paraphrase: {chunk}",
            num_return_sequences=NUM_CANDIDATES,
            max_length=MAX_INPUT_LENGTH,
            temperature=TEMPERATURE,
            truncation=True
        )
        return output[0]['generated_text']
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        return "[Error]: Memory limit exceeded"
    except Exception as e:
        return f"[Error]: {str(e)}"
