import random
import re
import logging
import time
import os
from typing import List, Dict, Any, Tuple
import spacy

# Disable Hugging Face warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

class Rewriter:
    """
    Robust text humanization engine with fallback mechanisms
    """
    def __init__(self):
        logger.info("Initializing Rewriter")
        self.nlp = self.load_spacy()
        self.ai_words = self.get_ai_words()
        self.hedge_phrases = self.get_hedges()
        self.fillers = self.get_fillers()
        self.qualifiers = self.get_qualifiers()
        logger.info("Rewriter initialized successfully")

    def load_spacy(self):
        """Load spaCy with multiple fallbacks"""
        try:
            return spacy.load("en_core_web_sm")
        except:
            try:
                spacy.cli.download("en_core_web_sm")
                return spacy.load("en_core_web_sm")
            except:
                logger.warning("spaCy load failed, using blank model")
                return spacy.blank("en")

    def get_ai_words(self):
        return {
            "utilize": ["use", "employ"],
            "leverage": ["use", "take advantage of"],
            "crucial": ["important", "key"],
            "robust": ["strong", "reliable"],
            "comprehensive": ["complete", "thorough"],
            "facilitate": ["help", "enable"],
            "implement": ["put in place", "carry out"],
            "optimize": ["improve", "enhance"],
            "paradigm": ["model", "approach"],
            "methodology": ["method", "way"]
        }

    def get_hedges(self):
        return [
            "I think", "It seems", "Perhaps", "Maybe",
            "In my opinion", "As far as I know"
        ]

    def get_fillers(self):
        return [
            "you know", "well", "actually", "basically",
            "I mean", "to be honest"
        ]

    def get_qualifiers(self):
        return [
            "somewhat", "rather", "quite", "fairly",
            "pretty", "slightly"
        ]

    def humanize(self, text: str, aggressiveness: float = 0.7) -> Dict[str, Any]:
        """
        Main humanization method with error resilience
        """
        try:
            return self._humanize_text(text, aggressiveness)
        except Exception as e:
            logger.error(f"Humanization failed: {str(e)}")
            return {
                "humanized_text": text,
                "transformation_summary": ["Humanization error"]
            }

    def _humanize_text(self, text: str, aggressiveness: float) -> Dict[str, Any]:
        """Core transformation pipeline"""
        changes = []
        current_text = text
        
        # Apply transformations in sequence
        current_text, c1 = self.lexical_modulation(current_text, aggressiveness)
        changes.extend(c1)
        
        current_text, c2 = self.syntactic_variation(current_text, aggressiveness)
        changes.extend(c2)
        
        current_text, c3 = self.cognitive_noise(current_text, aggressiveness)
        changes.extend(c3)
        
        return {
            "humanized_text": current_text,
            "transformation_summary": changes
        }

    def lexical_modulation(self, text: str, aggressiveness: float) -> Tuple[str, List[str]]:
        """Replace AI-typical vocabulary"""
        changes = []
        modified = text
        
        for ai_word, alternatives in self.ai_words.items():
            pattern = re.compile(rf'\b{re.escape(ai_word)}\b', re.IGNORECASE)
            for match in pattern.finditer(modified):
                if random.random() < aggressiveness * 0.6:
                    replacement = random.choice(alternatives)
                    if match.group().isupper():
                        replacement = replacement.upper()
                    elif match.group().istitle():
                        replacement = replacement.capitalize()
                    
                    modified = modified[:match.start()] + replacement + modified[match.end():]
                    changes.append(f"Replaced '{match.group()}' with '{replacement}'")
                    break  # Only replace first occurrence per word
        
        return modified, changes

    def syntactic_variation(self, text: str, aggressiveness: float) -> Tuple[str, List[str]]:
        """Alter sentence structures"""
        changes = []
        try:
            if not self.nlp.has_pipe("parser"):
                return text, []
                
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        except:
            return text, []
        
        processed = []
        
        for sent in sentences:
            words = sent.split()
            if len(words) < 25 or random.random() > aggressiveness * 0.4:
                processed.append(sent)
                continue
                
            # Try to split long sentences
            split_points = [i for i, word in enumerate(words) if word in [",", ";", "and", "but"]]
            if split_points:
                split_index = random.choice(split_points)
                part1 = " ".join(words[:split_index+1])
                part2 = " ".join(words[split_index+1:])
                processed.extend([part1, part2.capitalize()])
                changes.append("Split long sentence")
            else:
                processed.append(sent)
        
        return " ".join(processed), changes

    def cognitive_noise(self, text: str, aggressiveness: float) -> Tuple[str, List[str]]:
        """Add human-like imperfections"""
        changes = []
        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        except:
            return text, []
        
        processed = []
        
        for i, sent in enumerate(sentences):
            modified = sent
            
            # Add hedge phrases at start
            if i == 0 and random.random() < aggressiveness * 0.3:
                hedge = random.choice(self.hedge_phrases)
                modified = f"{hedge}, {sent.lower()}"
                changes.append(f"Added hedge: '{hedge}'")
            
            # Add filler words
            if random.random() < aggressiveness * 0.25:
                filler = random.choice(self.fillers)
                words = modified.split()
                if len(words) > 4:
                    pos = random.randint(1, len(words)-1)
                    words.insert(pos, f"({filler})")
                    modified = " ".join(words)
                    changes.append(f"Added filler: '{filler}'")
            
            # Add qualifiers to adjectives
            if random.random() < aggressiveness * 0.2:
                try:
                    sent_doc = self.nlp(modified)
                    for token in sent_doc:
                        if token.pos_ == "ADJ":
                            qualifier = random.choice(self.qualifiers)
                            modified = modified.replace(token.text, f"{qualifier} {token.text}", 1)
                            changes.append(f"Added qualifier: '{qualifier}'")
                            break
                except:
                    pass
            
            processed.append(modified)
        
        return " ".join(processed), changes
