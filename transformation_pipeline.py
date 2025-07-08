import random
import spacy
from typing import List, Callable
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
            self.lexical_obfuscation,
            self.clause_restructuring
        ]
        for transform in transformations:
            text = transform(text)
        return text

    def controlled_paraphrase(self, text: str) -> str:
        """
        Disrupts: Stylometric patterns, n-gram distributions
        Method: Semantic-preserving rephrasing
        Approach: Transformer-based paraphrasing
        Trade-off: Minor semantic drift possible
        """
        try:
            if len(text.split()) > 300:  # Chunk long texts
                return self._chunk_paraphrase(text)
            return self.paraphraser(f"paraphrase: {text}")[0]['generated_text']
        except Exception:
            return text  # Fallback to original if paraphrasing fails

    def _chunk_paraphrase(self, text: str) -> str:
        """Handle long texts by chunking"""
        chunks = [text[i:i+300] for i in range(0, len(text), 300)]
        return " ".join([self.controlled_paraphrase(chunk) for chunk in chunks])

    def inject_burstiness(self, text: str) -> str:
        """
        Disrupts: Perplexity, burstiness metrics
        Method: Intentional entropy spikes
        Approach: Rule-based sentence manipulation
        Trade-off: Potential readability impact
        """
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        if len(sentences) < 2:
            return text  # Not enough sentences to modify
        
        modified = []
        for i, sent in enumerate(sentences):
            # Randomly apply transformations
            rand_val = random.random()
            if rand_val > 0.7 and len(sent.split()) > 8:
                modified.append(self._fragment_sentence(sent))
            elif rand_val > 0.5:
                modified.append(self._add_interjection(sent))
            elif rand_val > 0.3 and i > 0:
                modified.append(self._merge_with_previous(sent, modified))
            else:
                modified.append(sent)
        
        return " ".join(modified)

    def _fragment_sentence(self, sentence: str) -> str:
        """Split long sentences randomly"""
        if random.random() > 0.6:
            return sentence
        clauses = [clause.strip() for clause in sentence.split(",")]
        if len(clauses) > 1:
            return ". ".join(clauses[:random.randint(2, len(clauses))]) + "."
        return sentence

    def _add_interjection(self, sentence: str) -> str:
        """Insert conversational elements"""
        interjections = ["You know,", "Actually,", "Well,", "I mean,"]
        if random.random() > 0.3:
            return f"{random.choice(interjections)} {sentence}"
        return sentence

    def _merge_with_previous(self, sentence: str, modified: List[str]) -> str:
        """Combine short sentences"""
        if modified and len(sentence.split()) < 5 and len(modified[-1].split()) < 8:
            return modified.pop() + " " + sentence.lower()
        return sentence

    def lexical_obfuscation(self, text: str) -> str:
        """
        Disrupts: Token frequency, n-gram distributions
        Method: Synonym replacement & lexical variation
        Approach: Contextual word embeddings
        Trade-off: Possible unnatural phrasing
        """
        doc = self.nlp(text)
        replacements = {
            "very": ["extremely", "incredibly", "remarkably"],
            "important": ["crucial", "vital", "paramount"],
            "use": ["utilize", "employ", "leverage"],
            "show": ["demonstrate", "illustrate", "exhibit"],
            "good": ["excellent", "superb", "outstanding"]
        }
        
        new_tokens = []
        for token in doc:
            lower_text = token.text.lower()
            if lower_text in replacements and random.random() > 0.5:
                new_tokens.append(random.choice(replacements[lower_text]))
            else:
                new_tokens.append(token.text)
            new_tokens.append(token.whitespace_)
        
        return "".join(new_tokens)

    def clause_restructuring(self, text: str) -> str:
        """
        Disrupts: Syntactic uniformity, parse tree patterns
        Method: Grammatical subversion
        Approach: Dependency parse manipulation
        Trade-off: Minor grammatical errors possible
        """
        doc = self.nlp(text)
        new_sentences = []
        
        for sent in doc.sents:
            if len(sent) < 6 or random.random() > 0.5:
                new_sentences.append(sent.text)
                continue
                
            # Active/passive transformation
            if "by" in sent.text and random.random() > 0.6:
                new_sentences.append(sent.text)
            else:
                new_sentences.append(self._reorder_clauses(sent.text))
        
        return " ".join(new_sentences)

    def _reorder_clauses(self, sentence: str) -> str:
        """Change clause order randomly"""
        clauses = [clause.strip() for clause in sentence.split(",") if clause.strip()]
        if len(clauses) > 1 and random.random() > 0.4:
            random.shuffle(clauses)
            return ", ".join(clauses) + ("." if not clauses[-1].endswith(".") else "")
        return sentence

    def add_typos(self, text: str) -> str:
        """
        Disrupts: Machine-like perfection
        Method: Intentional minor errors
        Approach: Probabilistic character substitution
        Trade-off: Professionalism impact
        """
        if random.random() > 0.8:  # Only apply to 20% of texts
            return text
            
        common_typos = {
            "the": "teh",
            "and": "adn",
            "that": "taht",
            "with": "wiht",
            "this": "htis"
        }
        
        words = text.split()
        for i, word in enumerate(words):
            lower_word = word.lower()
            if lower_word in common_typos and random.random() > 0.7:
                words[i] = common_typos[lower_word]
        
        return " ".join(words)
