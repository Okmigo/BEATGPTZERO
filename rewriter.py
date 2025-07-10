import random
import re
import os
import logging
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

    def __init__(self, model_name: str = "t5-base"):
        """
        Initializes the Rewriter, loading necessary models and resources.
        """
        logger.info(f"Initializing Rewriter with model: {model_name}")
        
        # Initialize without loading heavy models immediately
        self.model_name = model_name
        self.nlp = None
        self.tokenizer = None
        self.model = None
        self.text_generator = None
        self._is_initialized = False
        
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
        
        logger.info("Rewriter initialized without heavy models")

    def _initialize_models(self):
        """Lazy initialization of heavy models"""
        if self._is_initialized:
            return
            
        logger.info("Loading NLP models...")
        
        # Load spaCy model
        try:
            logger.info("Loading spaCy model: en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise RuntimeError("spaCy model loading failed") from e
        
        # Load T5 model for paraphrasing
        try:
            logger.info("Loading T5 tokenizer and model")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            logger.info("T5 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load T5 model: {e}")
            raise RuntimeError("T5 model loading failed") from e
        
        # Load text generation pipeline for semantic drift
        try:
            logger.info("Loading GPT-2 text generation pipeline")
            self.text_generator = pipeline(
                "text-generation", 
                model="gpt2", 
                max_length=100,
                device=-1  # Use CPU to reduce memory pressure
            )
            logger.info("GPT-2 pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to load GPT-2 pipeline: {e}")
            logger.warning("Continuing without GPT-2 semantic drift capability")
            self.text_generator = None
        
        self._is_initialized = True
        logger.info("All models loaded successfully")

    def humanize(self, text: str, aggressiveness: float = 0.5) -> Dict[str, Any]:
        """Processes text in chunks to avoid memory issues"""
        if not self._is_initialized:
            self._initialize_models()
            
        logger.info(f"Starting humanization with aggressiveness: {aggressiveness}")
        
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
            elif aggressiveness > 0.7:
                logger.warning("Semantic drift requested but GPT-2 not available")
            
            processed_chunks.append(current_text)
            all_changes.extend(chunk_changes)
        
        logger.info(f"Humanization complete. Applied {len(all_changes)} transformations")
        
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
            # Create a temporary spaCy instance if needed
            if self.nlp is None:
                temp_nlp = spacy.load("en_core_web_sm")
            else:
                temp_nlp = self.nlp
                
            doc = temp_nlp(text)
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
            # Fallback to simple splitting
            chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        
        return chunks

    def _apply_syntactic_variety(self, text: str, aggressiveness: float) -> Tuple[str, List[str]]:
        """
        Alters sentence structure to break predictable AI patterns.
        """
        try:
            if self.nlp is None:
                self._initialize_models()
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        except Exception as e:
            logger.error(f"Sentence parsing failed: {e}")
            return text, ["Sentence parsing failed - using original text"]
        
        changes = []
        result_sentences = []
        i = 0
        
        while i < len(sentences):
            current_sent = sentences[i]
            
            # Decide action based on aggressiveness
            action_prob = random.random()
            
            # Short sentences: consider merging
            if (len(current_sent.split()) < 10 and 
                i + 1 < len(sentences) and 
                len(sentences[i + 1].split()) < 10 and
                action_prob < aggressiveness * 0.3):
                
                # Merge sentences
                merged = f"{current_sent.rstrip('.')} and {sentences[i + 1].lower()}"
                result_sentences.append(merged)
                changes.append(f"Merged short sentences: '{current_sent}' + '{sentences[i + 1]}'")
                i += 2
                
            # Long sentences: consider splitting
            elif (len(current_sent.split()) > 25 and 
                  action_prob < aggressiveness * 0.4):
                
                # Split at conjunction or comma
                split_points = [m.start() for m in re.finditer(r',\s+(?:and|but|or|yet|so)\s+', current_sent)]
                if split_points:
                    split_point = random.choice(split_points)
                    part1 = current_sent[:split_point + 1].strip()
                    part2 = current_sent[split_point + 1:].strip()
                    if part2.startswith(('and ', 'but ', 'or ', 'yet ', 'so ')):
                        part2 = part2[4:].strip().capitalize()
                    result_sentences.extend([part1, part2])
                    changes.append(f"Split long sentence at conjunction")
                else:
                    result_sentences.append(current_sent)
                i += 1
                
            # Add complexity to simple sentences
            elif (len(current_sent.split()) < 15 and 
                  action_prob < aggressiveness * 0.2):
                
                # Add a dependent clause
                clause_starters = ["which", "that", "since", "because", "although", "while"]
                if random.random() < 0.5:
                    starter = random.choice(clause_starters)
                    enhanced = f"{current_sent.rstrip('.')}, {starter} is worth noting."
                    result_sentences.append(enhanced)
                    changes.append(f"Added complexity to simple sentence")
                else:
                    result_sentences.append(current_sent)
                i += 1
                
            else:
                result_sentences.append(current_sent)
                i += 1
        
        return " ".join(result_sentences), changes

    def _introduce_cognitive_noise(self, text: str, aggressiveness: float) -> Tuple[str, List[str]]:
        """
        Injects plausible human cognitive signals like hedging and fillers.
        """
        try:
            if self.nlp is None:
                self._initialize_models()
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        except Exception as e:
            logger.error(f"Sentence parsing failed: {e}")
            return text, ["Sentence parsing failed - using original text"]
        
        changes = []
        result_sentences = []
        
        for sentence in sentences:
            modified_sentence = sentence
            
            # Add hedge phrases at sentence beginning
            if random.random() < aggressiveness * 0.3:
                hedge = random.choice(self.hedge_phrases)
                modified_sentence = f"{hedge}, {sentence.lower()}"
                changes.append(f"Added hedge phrase: '{hedge}'")
            
            # Add qualifiers to adjectives
            if random.random() < aggressiveness * 0.25:
                # Find adjectives
                try:
                    sent_doc = self.nlp(modified_sentence)
                    for token in sent_doc:
                        if token.pos_ == "ADJ" and random.random() < 0.4:
                            qualifier = random.choice(self.qualifiers)
                            modified_sentence = modified_sentence.replace(token.text, f"{qualifier} {token.text}")
                            changes.append(f"Added qualifier: '{qualifier}' to adjective")
                            break
                except Exception as e:
                    logger.warning(f"Failed to add qualifiers: {e}")
            
            # Add fillers mid-sentence
            if random.random() < aggressiveness * 0.2:
                filler = random.choice(self.fillers)
                words = modified_sentence.split()
                if len(words) > 5:
                    insert_pos = random.randint(2, len(words) - 2)
                    words.insert(insert_pos, f"({filler})")
                    modified_sentence = " ".join(words)
                    changes.append(f"Added filler: '{filler}'")
            
            result_sentences.append(modified_sentence)
        
        return " ".join(result_sentences), changes

    def _modulate_lexical_choice(self, text: str, aggressiveness: float) -> Tuple[str, List[str]]:
        """
        Replaces common AI-favored words with more human-like, varied alternatives.
        """
        changes = []
        modified_text = text
        
        # Replace AI-favored words
        for ai_word, alternatives in self.ai_words.items():
            pattern = rf'\b{ai_word}\b'
            matches = re.finditer(pattern, modified_text, re.IGNORECASE)
            
            for match in matches:
                if random.random() < aggressiveness * 0.4:
                    replacement = random.choice(alternatives)
                    # Preserve case
                    if match.group().isupper():
                        replacement = replacement.upper()
                    elif match.group().istitle():
                        replacement = replacement.capitalize()
                    
                    modified_text = modified_text.replace(match.group(), replacement, 1)
                    changes.append(f"Lexical modulation: '{match.group()}' -> '{replacement}'")
        
        # Use T5 for contextual paraphrasing of phrases
        if aggressiveness > 0.6:
            # Find noun phrases to rephrase
            try:
                if self.nlp is None:
                    self._initialize_models()
                doc = self.nlp(modified_text)
                noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 2]
                
                for phrase in noun_phrases[:2]:  # Limit to avoid over-processing
                    if random.random() < aggressiveness * 0.3:
                        try:
                            # Generate paraphrase using T5
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
                                modified_text = modified_text.replace(phrase, paraphrased, 1)
                                changes.append(f"Contextual paraphrase: '{phrase}' -> '{paraphrased}'")
                        except Exception as e:
                            logger.warning(f"Error in contextual paraphrasing: {e}")
                            continue
            except Exception as e:
                logger.warning(f"Failed to parse noun phrases: {e}")
        
        return modified_text, changes

    def _induce_semantic_drift(self, text: str, aggressiveness: float) -> Tuple[str, List[str]]:
        """
        (For high aggressiveness only) Introduces a subtle, related digression.
        """
        if aggressiveness < 0.7 or not self.text_generator:
            return text, []
        
        try:
            if self.nlp is None:
                self._initialize_models()
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        except Exception as e:
            logger.error(f"Sentence parsing failed: {e}")
            return text, ["Sentence parsing failed - using original text"]
        
        changes = []
        
        # Only apply to longer texts
        if len(sentences) < 3:
            return text, []
        
        result_sentences = []
        
        for i, sentence in enumerate(sentences):
            result_sentences.append(sentence)
            
            # Add semantic drift after middle sentences
            if (i > 0 and i < len(sentences) - 1 and 
                random.random() < aggressiveness * 0.15):
                
                # Extract key concepts from sentence
                try:
                    sent_doc = self.nlp(sentence)
                    key_nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]
                    
                    if key_nouns:
                        concept = random.choice(key_nouns)
                        
                        # Generate related aside
                        aside_templates = [
                            f"(which, by the way, is something that's often misunderstood)",
                            f"(and {concept} is particularly interesting in this context)",
                            f"(though I should mention that {concept} can be quite complex)",
                            f"(speaking of {concept}, it's worth noting this varies significantly)",
                            f"(and honestly, {concept} deserves more attention than it typically gets)"
                        ]
                        
                        aside = random.choice(aside_templates)
                        result_sentences.append(aside)
                        changes.append(f"Added semantic drift: aside about '{concept}'")
                except Exception as e:
                    logger.warning(f"Failed to generate semantic drift: {e}")
        
        return " ".join(result_sentences), changes
