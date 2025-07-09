import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import spacy
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Downloading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading en_core_web_sm...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model downloaded.")

    logger.info("Downloading T5 model...")
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    logger.info("T5 model downloaded.")

    logger.info("Downloading GPT-2 model...")
    text_generator = pipeline("text-generation", model="gpt2")
    logger.info("GPT-2 model downloaded.")

if __name__ == "__main__":
    main()
