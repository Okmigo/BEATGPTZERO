import random
import re
import os
import logging
import time
from typing import List, Dict, Any, Tuple
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch

logger = logging.getLogger(__name__)

class Rewriter:
    """
    A class to perform deep, adversarial rewriting of AI-generated text to
    evade second-order AI detection systems.
    """

    def __init__(self, model_name: str = "t5-base", max_retries: int = 3):
        """
        Initializes the Rewriter with robust model loading
        """
        logger.info(f"Initializing Rewriter with model: {model_name}")
        self.model_name = model_name
        self.max_retries = max_retries
        self.nlp = None
        self.tokenizer = None
        self.model = None
        self.text_generator = None
        self._is_initialized = False
        
        # Load models immediately with retry logic
        self._initialize_models_with_retry()
        
        # Define cognitive noise patterns
        self.hedge_phrases = [
            "I think", "It seems", "Perhaps", "Maybe", "Possibly", "In my opinion",
            "From what I understand", "As far as I know", "I believe", "It appears",
            "Presumably", "Allegedly", "Supposedly", "Apparently", "Conceivably"
        ]
        
        self.fillers = [
            "you know", "well", "actually", "basically", "essentially", "sort of",
            "kind of", "like", "I mean", "to be honest", "frankly", "obviously",
            "clearly", "of course", "naturally", "admittedly"
        ]
        
        self.qualifiers = [
            "somewhat", "rather", "quite", "fairly", "relatively", "reasonably",
            "pretty", "moderately", "slightly", "particularly", "especially",
            "notably", "remarkably", "considerably", "substantially"
        ]
        
        # AI-favored words to replace
        self.ai_words = {
            "utilize": ["use", "employ", "apply", "work with"],
            "leverage": ["use", "take advantage of", "make use of", "harness"],
            "crucial": ["important", "key", "vital", "critical", "essential"],
            "robust": ["strong", "solid", "reliable", "effective", "good"],
            "comprehensive": ["complete", "thorough", "full", "extensive"],
            "facilitate": ["help", "enable", "make easier", "support"],
            "implement": ["put in place", "carry out", "execute", "do"],
            "optimize": ["improve", "enhance", "make better", "fine-tune"],
            "paradigm": ["model", "approach", "way", "framework"],
            "methodology": ["method", "approach", "way", "technique"]
        }

    def _initialize_models_with_retry(self):
        """Robust model loading with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Loading models (attempt {attempt+1}/{self.max_retries})")
                self._initialize_models()
                self._is_initialized = True
                logger.info("All models loaded successfully")
                return
            except Exception as e:
                logger.error(f"Model loading failed on attempt {attempt+1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.critical("All model loading attempts failed")
                    raise RuntimeError("Critical model loading failure") from e

    def _initialize_models(self):
        """Load all required models"""
        # Load spaCy model
        logger.info("Loading spaCy model: en_core_web_sm")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load T5 model
        logger.info("Loading T5 tokenizer and model")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Load GPT-2 pipeline
        try:
            logger.info("Loading GPT-2 text generation pipeline")
            self.text_generator = pipeline(
                "text-generation", 
                model="gpt2", 
                max_length=100,
                device=-1  # Use CPU
            )
        except Exception as e:
            logger.error(f"GPT-2 pipeline failed: {e}")
            self.text_generator = None

    def humanize(self, text: str, aggressiveness: float = 0.5) -> Dict[str, Any]:
        """Processes text with robust error handling"""
        if not self._is_initialized:
            raise RuntimeError("Rewriter not initialized")
            
        try:
            logger.info(f"Humanizing text (len: {len(text)} chars)")
            return self._humanize_text(text, aggressiveness)
        except Exception as e:
            logger.error(f"Humanization failed: {str(e)}")
            return {
                "humanized_text": text,
                "transformation_summary": [f"Humanization error: {str(e)}"]
            }

    def _humanize_text(self, text: str, aggressiveness: float) -> Dict[str, Any]:
        """Core humanization logic"""
        # Split text into manageable chunks
        chunks = self._chunk_text(text)
        all_changes = []
        processed_chunks = []
        
        for chunk in chunks:
            current_text = chunk
            chunk_changes = []
            
            # Apply transformations in order
            if aggressiveness > 0.3:
                current_text, changes = self._apply_syntactic_variety(current_text, aggressiveness)
                chunk_changes.extend(changes)
            
            if aggressiveness > 0.2:
                current_text, changes = self._modulate_lexical_choice(current_text, aggressiveness)
                chunk_changes.extend(changes)
            
            if aggressiveness > 0.1:
                current_text, changes = self._introduce_cognitive_noise(current_text, aggressiveness)
                chunk_changes.extend(changes)
            
            if aggressiveness > 0.7 and self.text_generator:
                current_text, changes = self._induce_semantic_drift(current_text, aggressiveness)
                chunk_changes.extend(changes)
            
            processed_chunks.append(current_text)
            all_changes.extend(chunk_changes)
        
        logger.info(f"Applied {len(all_changes)} transformations")
        
        return {
            "humanized_text": " ".join(processed_chunks),
            "transformation_summary": all_changes
        }

    def _chunk_text(self, text: str, max_chars: int = 1500) -> List[str]:
        """Splits text into chunks while preserving sentence boundaries"""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        try:
            doc = self.nlp(text)
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if len(current_chunk) + len(sent_text) <= max_chars:
                    current_chunk += sent_text + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent_text + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        except Exception as e:
            logger.error(f"Chunking failed: {e}. Using simple split.")
            chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        
        return chunks

    def _apply_syntactic_variety(self, text: str, aggressiveness: float) -> Tuple[str, List[str]]:
        """Alters sentence structure to break AI patterns"""
        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        except Exception as e:
            logger.error(f"Sentence parsing failed: {e}")
            return text, [f"Sentence parsing error: {str(e)}"]
        
        changes = []
        result_sentences = []
        i = 0
        
        while i < len(sentences):
            current_sent = sentences[i]
            action_prob = random.random()
            
            # Merge short sentences
            if (len(current_sent.split()) < 10 and i + 1 < len(sentences) and len(sentences[i+1].split()) < 10 and action_prob < aggressiveness * 0.3:
                merged = f"{current_sent.rstrip('.')} and {sentences[i+1].lower()}"
                result_sentences.append(merged)
                changes.append(f"Merged: '{current_sent}' + '{sentences[i+1]}'")
                i += 2
                
            # Split long sentences
            elif len(current_sent.split()) > 25 and action_prob < aggressiveness * 0.4:
                split_points = [m.start() for m in re.finditer(r',\s+(?:and|but|or|yet|so)\s+', current_sent)]
                if split_points:
                    split_point = random.choice(split_points)
                    part1 = current_sent[:split_point+1].strip()
                    part2 = current_sent[split_point+1:].strip()
                    if part2.startswith(('and ', 'but ', 'or ', 'yet ', 'so ')):
                        part2 = part2[4:].strip().capitalize()
                    result_sentences.extend([part1, part2])
                    changes.append("Split long sentence")
                else:
                    result_sentences.append(current_sent)
                i += 1
                
            # Enhance simple sentences
            elif len(current_sent.split()) < 15 and action_prob < aggressiveness * 0.2:
                if random.random() < 0.5:
                    starters = ["which", "that", "since", "because", "although", "while"]
                    enhanced = f"{current_sent.rstrip('.')}, {random.choice(starters)} is worth noting."
                    result_sentences.append(enhanced)
                    changes.append("Enhanced simple sentence")
                else:
                    result_sentences.append(current_sent)
                i += 1
                
            else:
                result_sentences.append(current_sent)
                i += 1
        
        return " ".join(result_sentences), changes

    def _introduce_cognitive_noise(self, text: str, aggressiveness: float) -> Tuple[str, List[str]]:
        """Injects human-like imperfections"""
        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        except Exception as e:
            logger.error(f"Sentence parsing failed: {e}")
            return text, [f"Sentence error: {str(e)}"]
        
        changes = []
        result_sentences = []
        
        for sentence in sentences:
            modified = sentence
            
            # Add hedging
            if random.random() < aggressiveness * 0.3:
                hedge = random.choice(self.hedge_phrases)
                modified = f"{hedge}, {sentence.lower()}"
                changes.append(f"Hedge: '{hedge}'")
            
            # Add qualifiers
            if random.random() < aggressiveness * 0.25:
                try:
                    sent_doc = self.nlp(modified)
                    for token in sent_doc:
                        if token.pos_ == "ADJ" and random.random() < 0.4:
                            qualifier = random.choice(self.qualifiers)
                            modified = modified.replace(token.text, f"{qualifier} {token.text}")
                            changes.append(f"Qualifier: '{qualifier}'")
                            break
                except:
                    pass
            
            # Add fillers
            if random.random() < aggressiveness * 0.2:
                filler = random.choice(self.fillers)
                words = modified.split()
                if len(words) > 5:
                    pos = random.randint(2, len(words)-2)
                    words.insert(pos, f"({filler})")
                    modified = " ".join(words)
                    changes.append(f"Filler: '{filler}'")
            
            result_sentences.append(modified)
        
        return " ".join(result_sentences), changes

    def _modulate_lexical_choice(self, text: str, aggressiveness: float) -> Tuple[str, List[str]]:
        """Replaces AI-typical vocabulary"""
        changes = []
        modified = text
        
        # Replace AI words
        for ai_word, alternatives in self.ai_words.items():
            pattern = rf'\b{ai_word}\b'
            for match in re.finditer(pattern, modified, re.IGNORECASE):
                if random.random() < aggressiveness * 0.4:
                    replacement = random.choice(alternatives)
                    if match.group().isupper():
                        replacement = replacement.upper()
                    elif match.group().istitle():
                        replacement = replacement.capitalize()
                    
                    modified = modified.replace(match.group(), replacement, 1)
                    changes.append(f"Replaced '{match.group()}' with '{replacement}'")
        
        # Paraphrase noun phrases
        if aggressiveness > 0.6:
            try:
                doc = self.nlp(modified)
                noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 2]
                
                for phrase in noun_phrases[:min(3, len(noun_phrases))]:
                    if random.random() < aggressiveness * 0.3:
                        input_text = f"paraphrase: {phrase}"
                        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True)
                        
                        with torch.no_grad():
                            outputs = self.model.generate(
                                inputs.input_ids,
                                max_length=32,
                                num_beams=3,
                                temperature=0.8,
                                do_sample=True
                            )
                        
                        paraphrased = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        if paraphrased and paraphrased != phrase:
                            modified = modified.replace(phrase, paraphrased, 1)
                            changes.append(f"Paraphrased: '{phrase}' → '{paraphrased}'")
            except Exception as e:
                logger.error(f"Paraphrasing failed: {e}")
        
        return modified, changes

    def _induce_semantic_drift(self, text: str, aggressiveness: float) -> Tuple[str, List[str]]:
        """Adds human-like digressions"""
        if not self.text_generator or aggressiveness < 0.7:
            return text, []
        
        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        except Exception as e:
            logger.error(f"Sentence parsing failed: {e}")
            return text, []
        
        if len(sentences) < 3:
            return text, []
        
        changes = []
        result = []
        
        for i, sent in enumerate(sentences):
            result.append(sent)
            
            if i > 0 and i < len(sentences)-1 and random.random() < aggressiveness * 0.15:
                try:
                    sent_doc = self.nlp(sent)
                    nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]
                    if nouns:
                        concept = random.choice(nouns)
                        asides = [
                            f"(which, by the way, is often misunderstood)",
                            f"(and {concept} is really interesting here)",
                            f"(though {concept} can be complicated)",
                            f"(speaking of {concept}, this varies a lot)",
                            f"(honestly, {concept} needs more attention)"
                        ]
                        aside = random.choice(asides)
                        result.append(aside)
                        changes.append(f"Added aside: '{concept}'")
                except:
                    pass
        
        return " ".join(result), changes
