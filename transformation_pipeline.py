import random
import spacy
from transformers import pipeline
from functools import lru_cache

class TransformationPipeline:
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.paraphraser = self._load_paraphraser()
        
    @lru_cache(maxsize=1)
    def _load_spacy_model(self):
        """Lazy-load Spacy model to reduce startup time"""
        return spacy.load("en_core_web_sm")
    
    @lru_cache(maxsize=1)
    def _load_paraphraser(self):
        """Lazy-load paraphrasing model"""
        return pipeline("text2text-generation", model="t5-small", max_length=512)
    
    def apply_transformations(self, text: str) -> str:
        """Apply transformation modules sequentially"""
        # Short-circuit for small inputs
        if len(text) < 25:
            return text
            
        transformations = [
            self.controlled_paraphrase,
            self.inject_burstiness,
            self.lexical_obfuscation
        ]
        for transform in transformations:
            text = transform(text)
        return text

    # ... (rest of transformation methods same as before) ...
