import random
import spacy
import re
from typing import List
from transformers import pipeline
from functools import lru_cache

class TransformationPipeline:
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.paraphraser = self._load_paraphraser()
        self.transition_words = [
            'however', 'meanwhile', 'consequently', 'nonetheless', 
            'subsequently', 'thereby', 'whereas', 'likewise'
        ]
        
    @lru_cache(maxsize=1)
    def _load_spacy_model(self):
        return spacy.load("en_core_web_sm")
    
    @lru_cache(maxsize=1)
    def _load_paraphraser(self):
        """Use a more robust paraphrasing model"""
        return pipeline(
            "text2text-generation", 
            model="humarin/chatgpt_paraphraser_on_T5_base",
            max_length=512
        )
    
    def apply_transformations(self, text: str) -> str:
        """Apply transformations with quality control"""
        if len(text) < 25:
            return text
            
        # Step 1: Semantic-preserving paraphrase (most important)
        text = self.controlled_paraphrase(text)
        
        # Step 2: Apply only 1-2 additional transformations
        transformations = random.sample([
            self.inject_burstiness,
            self.lexical_obfuscation,
            self.add_human_quirks
        ], k=2)
        
        for transform in transformations:
            text = transform(text)
            
        return text

    def controlled_paraphrase(self, text: str) -> str:
        """Improved paraphrasing with context preservation"""
        try:
            # Process entire paragraphs to maintain context
            paragraphs = text.split('\n\n')
            paraphrased = []
            
            for para in paragraphs:
                if len(para.split()) > 100:  # Process long paragraphs in chunks
                    chunks = self._split_into_sentences(para)
                    chunk_size = max(2, min(4, len(chunks)//2))
                    chunked_para = []
                    
                    for i in range(0, len(chunks), chunk_size):
                        chunk = " ".join(chunks[i:i+chunk_size])
                        if chunk:
                            result = self.paraphraser(
                                f"paraphrase: {chunk}",
                                max_length=512
                            )[0]['generated_text']
                            chunked_para.append(result)
                    
                    paraphrased.append(" ".join(chunked_para))
                else:
                    result = self.paraphraser(
                        f"paraphrase: {para}",
                        max_length=512
                    )[0]['generated_text']
                    paraphrased.append(result)
                    
            return '\n\n'.join(paraphrased)
        except Exception as e:
            print(f"Paraphrase failed: {str(e)}")
            return text

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text while preserving sentence boundaries"""
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def inject_burstiness(self, text: str) -> str:
        """Natural rhythm variations"""
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        if len(sentences) < 3:
            return text
            
        modified = []
        for i, sent in enumerate(sentences):
            # Only modify 20% of sentences
            if random.random() > 0.8:
                continue
                
            # Apply only one transformation per sentence
            options = [
                self._add_transition(sent),
                self._fragment_sentence(sent),
                self._merge_sentences(sent, modified) if modified else sent
            ]
            modified.append(random.choice(options))
        return " ".join(modified)

    def _add_transition(self, sentence: str) -> str:
        """Add natural transition words"""
        transitions = ["You know,", "Actually,", "Well,", "I mean,", "Interestingly,"]
        return f"{random.choice(transitions)} {sentence}" if random.random() > 0.7 else sentence

    def lexical_obfuscation(self, text: str) -> str:
        """More natural synonym replacement"""
        doc = self.nlp(text)
        replacements = {
            "very": ["extremely", "incredibly", "remarkably", "quite"],
            "important": ["crucial", "vital", "paramount", "key"],
            "use": ["utilize", "employ", "leverage", "apply"],
            "show": ["demonstrate", "illustrate", "exhibit", "reveal"],
            "good": ["excellent", "superb", "outstanding", "solid"]
        }
        
        new_tokens = []
        for token in doc:
            lower_text = token.text.lower()
            if lower_text in replacements and random.random() > 0.7:
                new_tokens.append(random.choice(replacements[lower_text]))
            else:
                new_tokens.append(token.text)
            new_tokens.append(token.whitespace_)
        
        return "".join(new_tokens)

    def add_human_quirks(self, text: str) -> str:
        """Natural human imperfections"""
        # Add contractions
        text = re.sub(r"\b(I am|you are|he is|she is|it is|we are|they are)\b", 
                      lambda m: m.group(1).split()[0][:-1] + "'" + m.group(1).split()[1][0], 
                      text, flags=re.IGNORECASE)
        
        # Add strategic typos to 0.1% of words
        words = text.split()
        typo_count = max(1, len(words) // 1000)
        for _ in range(typo_count):
            idx = random.randint(0, len(words)-1)
            word = words[idx]
            if len(word) > 3:
                typo_type = random.choice(['swap', 'omit'])
                if typo_type == 'swap' and len(word) > 2:
                    pos = random.randint(1, len(word)-2)
                    words[idx] = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
                elif typo_type == 'omit' and len(word) > 3:
                    pos = random.randint(1, len(word)-2)
                    words[idx] = word[:pos] + word[pos+1:]
        
        return " ".join(words)
