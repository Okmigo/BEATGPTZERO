import logging
import os
from huggingface_hub import hf_hub_download
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(repo_id, filename, cache_dir):
    try:
        logger.info(f"Downloading {repo_id}/{filename}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True
        )
        logger.info(f"Downloaded to {path}")
        return path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

def main():
    cache_dir = os.getenv("HF_HOME", "/app/.cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download T5 model files
    t5_files = [
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer_config.json"
    ]
    for file in t5_files:
        download_model("t5-base", file, cache_dir)
    
    # Download GPT-2 model files
    gpt2_files = [
        "config.json",
        "pytorch_model.bin",
        "vocab.json",
        "merges.txt"
    ]
    for file in gpt2_files:
        download_model("gpt2", file, cache_dir)
    
    # Download spaCy model
    try:
        logger.info("Downloading spaCy en_core_web_sm")
        spacy.cli.download("en_core_web_sm")
        logger.info("spaCy model downloaded")
    except Exception as e:
        logger.error(f"spaCy download failed: {e}")
        raise

if __name__ == "__main__":
    main()
