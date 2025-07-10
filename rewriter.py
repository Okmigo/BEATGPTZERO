import random
import re
import logging
import spacy

logger = logging.getLogger(__name__)

class Rewriter:
    """
    Robust text humanization engine with rule-based transformations
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
        """Load spaCy with fallback to blank model"""
        try:
            return spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"spaCy load failed: {str(e)} - using blank model")
            return spacy.blank("en")

    def get_ai_words(self):
        return {
            "utilize": ["use", "employ", "apply"],
            "leverage": ["use", "take advantage of"],
            "crucial": ["important", "key", "vital"],
            "robust": ["strong", "reliable", "solid"],
            "comprehensive": ["complete", "thorough", "full"],
            "facilitate": ["help", "enable", "assist"],
            "implement": ["put in place", "carry out", "execute"],
            "optimize": ["improve", "enhance", "make better"],
            "paradigm": ["model", "approach", "framework"],
            "methodology": ["method", "approach", "technique"],
            "enhance": ["improve", "better", "strengthen"],
            "furthermore": ["also", "additionally", "what's more"],
            "consequently": ["so", "therefore", "as a result"],
            "demonstrate": ["show", "prove", "display"],
            "significant": ["important", "major", "notable"],
            "utilization": ["use", "usage", "application"],
            "optimal": ["best", "ideal", "most effective"]
        }

    def get_hedges(self):
        return [
            "I think", "It seems", "Perhaps", "Maybe", "Possibly",
            "In my opinion", "As far as I know", "I believe", 
            "It appears", "From what I understand", "To be honest",
            "Honestly", "Frankly", "Personally", "If you ask me"
        ]

    def get_fillers(self):
        return [
            "you know", "well", "actually", "basically",
            "I mean", "sort of", "kind of", "like", 
            "to be honest", "frankly", "obviously", 
            "clearly", "of course", "essentially"
        ]

    def get_qualifiers(self):
        return [
            "somewhat", "rather", "quite", "fairly",
            "pretty", "slightly", "a bit", "moderately",
            "reasonably", "relatively", "comparatively"
        ]

    def humanize(self, text: str, aggressiveness: float = 0.7) -> dict:
        """
        Transform text to appear more human-written
        """
        try:
            return self._humanize_text(text, aggressiveness)
        except Exception as e:
            logger.error(f"Humanization failed: {str(e)}")
            return {
                "humanized_text": text,
                "transformation_summary": ["Transformation error - using original text"]
            }

    def _humanize_text(self, text: str, aggressiveness: float) -> dict:
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

    def lexical_modulation(self, text: str, aggressiveness: float) -> tuple:
        """Replace AI-typical vocabulary with human alternatives"""
        changes = []
        modified = text
        
        for ai_word, alternatives in self.ai_words.items():
            pattern = re.compile(rf'\b{re.escape(ai_word)}\b', re.IGNORECASE)
            match = pattern.search(modified)
            if match and random.random() < aggressiveness * 0.7:
                replacement = random.choice(alternatives)
                
                # Preserve case
                if match.group().isupper():
                    replacement = replacement.upper()
                elif match.group().istitle():
                    replacement = replacement.capitalize()
                
                modified = modified[:match.start()] + replacement + modified[match.end():]
                changes.append(f"Lexical: '{match.group()}' → '{replacement}'")
        
        return modified, changes

    def syntactic_variation(self, text: str, aggressiveness: float) -> tuple:
        """Alter sentence structures"""
        changes = []
        processed = []
        
        try:
            # Process text in sentences
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            
            for sent in sentences:
                words = sent.split()
                
                # Skip transformation for very short sentences
                if len(words) < 5:
                    processed.append(sent)
                    continue
                
                # Randomly apply transformations based on aggressiveness
                if random.random() < aggressiveness * 0.5:
                    # Sentence splitting for long sentences
                    if len(words) > 20:
                        split_points = [i for i, word in enumerate(words) 
                                       if word in [",", ";", "and", "but", "or"]]
                        if split_points:
                            split_index = random.choice(split_points)
                            part1 = " ".join(words[:split_index+1])
                            part2 = " ".join(words[split_index+1:]).capitalize()
                            processed.extend([part1, part2])
                            changes.append("Syntactic: Split long sentence")
                            continue
                
                # Sentence merging for short consecutive sentences
                if (len(processed) > 0 and len(words) < 10 
                    and len(processed[-1].split()) < 10
                    and random.random() < aggressiveness * 0.4):
                    last_sent = processed.pop()
                    merged = f"{last_sent.rstrip('.')} and {sent.lower()}"
                    processed.append(merged)
                    changes.append("Syntactic: Merged short sentences")
                else:
                    processed.append(sent)
            
            return " ".join(processed), changes
        except Exception:
            return text, changes

    def cognitive_noise(self, text: str, aggressiveness: float) -> tuple:
        """Add human-like imperfections"""
        changes = []
        processed = []
        
        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            
            for i, sent in enumerate(sentences):
                modified = sent
                
                # Add hedge phrases at start of first sentence
                if i == 0 and random.random() < aggressiveness * 0.5:
                    hedge = random.choice(self.hedge_phrases)
                    modified = f"{hedge}, {sent.lower()}"
                    changes.append(f"Cognitive: Added hedge '{hedge}'")
                
                # Add filler words mid-sentence
                if random.random() < aggressiveness * 0.4:
                    filler = random.choice(self.fillers)
                    words = modified.split()
                    if len(words) > 4:
                        pos = random.randint(1, len(words)-1)
                        words.insert(pos, f"({filler})")
                        modified = " ".join(words)
                        changes.append(f"Cognitive: Added filler '{filler}'")
                
                # Add qualifiers to adjectives
                if random.random() < aggressiveness * 0.3:
                    # Simple adjective detection (spacy might not be available)
                    words = modified.split()
                    for j, word in enumerate(words):
                        # Simple adjective detection (ends with common suffixes)
                        if (len(word) > 4 and 
                            word.endswith(('able', 'ible', 'ful', 'ic', 'ical', 'ious', 'ous', 'ish')) 
                            and random.random() < 0.5):
                            qualifier = random.choice(self.qualifiers)
                            words[j] = f"{qualifier} {word}"
                            changes.append(f"Cognitive: Added qualifier '{qualifier}'")
                            modified = " ".join(words)
                            break
                
                processed.append(modified)
            
            return " ".join(processed), changes
        except Exception:
            return text, changes
