import random
import spacy
from typing import List, Dict, Tuple
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from functools import lru_cache
import re

class TransformationPipeline:
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.transition_words = [
            'however', 'meanwhile', 'consequently', 'nonetheless', 
            'subsequently', 'thereby', 'whereas', 'likewise'
        ]
        
    @lru_cache(maxsize=1)
    def _load_spacy_model(self):
        return spacy.load("en_core_web_sm")
    
    def apply_transformations(self, text: str) -> str:
        """Apply transformation modules with controlled probability"""
        if len(text) < 25:
            return text
            
        # Apply core transformations
        text = self.controlled_paraphrase(text)
        text = self.vary_sentence_structure(text)
        text = self.introduce_human_quirks(text)
        
        return text

    def controlled_paraphrase(self, text: str) -> str:
        """Semantic-preserving rephrasing with context-aware chunking"""
        try:
            # Process in coherent chunks (paragraphs)
            paragraphs = text.split('\n\n')
            paraphrased = []
            for para in paragraphs:
                if not para.strip():
                    continue
                    
                # Process paragraph in contextually complete segments
                chunks = self._split_paragraph(para)
                para_text = ""
                for chunk in chunks:
                    inputs = self.tokenizer(
                        f"paraphrase: {chunk}",
                        return_tensors="pt",
                        max_length=512,
                        truncation=True
                    )
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_length=512,
                        num_beams=5,
                        early_stopping=True
                    )
                    para_text += self.tokenizer.decode(
                        outputs[0], 
                        skip_special_tokens=True
                    ) + " "
                paraphrased.append(para_text.strip())
            return '\n\n'.join(paraphrased)
        except Exception:
            return text

    def _split_paragraph(self, text: str) -> List[str]:
        """Split text into coherent chunks (2-3 sentences)"""
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        chunks = []
        current_chunk = []
        
        for sent in sentences:
            current_chunk.append(sent)
            if len(current_chunk) >= 2 and random.random() > 0.6:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def vary_sentence_structure(self, text: str) -> str:
        """Introduce human-like sentence variation"""
        doc = self.nlp(text)
        output = []
        
        for sent in doc.sents:
            sent_text = sent.text
            
            # Apply transformations probabilistically
            if random.random() < 0.7:
                sent_text = self._strategic_contractions(sent_text)
                
            if random.random() < 0.4:
                sent_text = self._vary_transitions(sent_text)
                
            if random.random() < 0.3 and len(sent) > 15:
                sent_text = self._add_asides(sent_text)
                
            output.append(sent_text)
            
        return " ".join(output)

    def _strategic_contractions(self, text: str) -> str:
        """Add natural contractions (not overdone)"""
        replacements = {
            "it is": "it's", "do not": "don't", "is not": "isn't",
            "cannot": "can't", "I am": "I'm", "you are": "you're",
            "they are": "they're", "we have": "we've"
        }
        for full, contr in replacements.items():
            if random.random() < 0.5 and full in text.lower():
                text = re.sub(
                    re.escape(full), 
                    contr, 
                    text, 
                    flags=re.IGNORECASE
                )
        return text

    def _vary_transitions(self, text: str) -> str:
        """Diversify transition words"""
        common_transitions = [
            'however', 'therefore', 'furthermore', 
            'moreover', 'consequently'
        ]
        for trans in common_transitions:
            if trans in text.lower():
                if random.random() < 0.6:
                    replacement = random.choice(self.transition_words)
                    text = text.replace(trans, replacement, 1)
        return text

    def _add_asides(self, text: str) -> str:
        """Insert natural digressions"""
        asides = [
            " - something worth noting - ",
            ", which is interesting, ",
            " (this matters because) ",
            ", surprisingly, ",
            ", if you think about it, "
        ]
        words = text.split()
        if len(words) > 8:
            insert_pos = random.randint(3, len(words)-3)
            words.insert(insert_pos, random.choice(asides).strip())
            return " ".join(words)
        return text

    def introduce_human_quirks(self, text: str) -> str:
        """Add purposeful human imperfections"""
        # Strategic punctuation variations
        if random.random() < 0.7:
            text = text.replace(';', ',').replace('—', '-')
            
        # Occasional informal markers
        if random.random() < 0.4:
            informal = [
                "sort of", "kind of", "pretty much", 
                "you see", "actually", "basically"
            ]
            words = text.split()
            if len(words) > 15:
                insert_pos = random.randint(5, len(words)-5)
                words.insert(insert_pos, random.choice(informal))
                text = " ".join(words)
                
        # Controlled typo insertion (0.5% of words)
        words = text.split()
        typo_count = max(1, len(words) // 200)
        for _ in range(typo_count):
            idx = random.randint(0, len(words)-1)
            word = words[idx]
            if len(word) > 4:
                typo_type = random.choice(['swap', 'omit', 'double'])
                if typo_type == 'swap' and len(word) > 2:
                    pos = random.randint(1, len(word)-2)
                    words[idx] = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
                elif typo_type == 'omit' and len(word) > 3:
                    pos = random.randint(1, len(word)-2)
                    words[idx] = word[:pos] + word[pos+1:]
                elif typo_type == 'double':
                    pos = random.randint(1, len(word)-2)
                    words[idx] = word[:pos] + word[pos] + word[pos:]
        text = " ".join(words)
        
        # Sentence casing variation
        if random.random() < 0.2:
            sentences = [s.strip() for s in text.split('.') if s]
            if len(sentences) > 1:
                rand_idx = random.randint(0, len(sentences)-1)
                sentences[rand_idx] = sentences[rand_idx].lower()
                text = '. '.join(sentences) + '.' if text.endswith('.') else '. '.join(sentences)
        
        return text
