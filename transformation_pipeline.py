# transformation_pipeline.py
import random
import spacy
from typing import List, Callable
from transformers import pipeline

class TransformationPipeline:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.paraphraser = pipeline("text2text-generation", model="t5-small", max_length=512)
        
    def apply_transformations(self, text: str) -> str:
        """Apply transformation modules sequentially"""
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
        return self.paraphraser(f"paraphrase: {text}")[0]['generated_text']

    def inject_burstiness(self, text: str) -> str:
        """
        Disrupts: Perplexity, burstiness metrics
        Method: Intentional entropy spikes
        Approach: Rule-based sentence manipulation
        Trade-off: Potential readability impact
        """
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        # Introduce variance
        modified = []
        for i, sent in enumerate(sentences):
            if i % 3 == 0 and len(sent.split()) > 8:
                modified.append(self._fragment_sentence(sent))
            elif i % 4 == 0:
                modified.append(self._add_interjection(sent))
            else:
                modified.append(sent)
        
        return " ".join(modified)

    def _fragment_sentence(self, sentence: str) -> str:
        """Split long sentences randomly"""
        if random.random() > 0.6:
            return sentence
        clauses = [clause.strip() for clause in sentence.split(",")]
        return ". ".join(clauses[:random.randint(2, len(clauses))]) + "."

    def _add_interjection(self, sentence: str) -> str:
        """Insert conversational elements"""
        interjections = ["You know,", "Actually,", "Well,", "I mean,"]
        if random.random() > 0.7:
            return f"{random.choice(interjections)} {sentence}"
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
            "use": ["utilize", "employ", "leverage"]
        }
        
        new_tokens = []
        for token in doc:
            if token.text.lower() in replacements and random.random() > 0.6:
                new_tokens.append(random.choice(replacements[token.text.lower()]))
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
            if len(sent) < 6 or random.random() > 0.4:
                new_sentences.append(sent.text)
                continue
                
            # Active/passive transformation
            if "by" in sent.text and random.random() > 0.5:
                new_sentences.append(sent.text)
            else:
                new_sentences.append(self._reorder_clauses(sent.text))
        
        return " ".join(new_sentences)

    def _reorder_clauses(self, sentence: str) -> str:
        """Change clause order randomly"""
        clauses = [clause.strip() for clause in sentence.split(",")]
        if len(clauses) > 1 and random.random() > 0.3:
            random.shuffle(clauses)
            return ", ".join(clauses) + "."
        return sentence
