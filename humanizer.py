import random
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from typing import List
nltk.download('averaged_perceptron_tagger')

# Helper to introduce contractions
def apply_contractions(text):
    contractions = {
        "do not": "don't", "does not": "doesn't", "did not": "didn't",
        "is not": "isn't", "are not": "aren't", "was not": "wasn't",
        "were not": "weren't", "has not": "hasn't", "have not": "haven't",
        "had not": "hadn't", "will not": "won't", "would not": "wouldn't",
        "can not": "can't", "could not": "couldn't", "should not": "shouldn't",
        "might not": "mightn't", "must not": "mustn't"
    }
    for k, v in contractions.items():
        text = re.sub(r'\b' + k + r'\b', v, text)
    return text

# Helper to insert human-like interjections and informal phrases
def humanize_phrasing(sentence):
    openers = ["To be honest,", "Well,", "You know,", "Frankly,", "Let's be real,"]
    closers = ["if you ask me.", "and that’s saying something.", "no joke.", "just so you know.", "right?"]
    
    if random.random() < 0.2:
        sentence = random.choice(openers) + " " + sentence
    if random.random() < 0.2:
        sentence += " " + random.choice(closers)
    return sentence

# Shuffle middle sentences to simulate burstiness
def rearrange_sentences(sentences: List[str]) -> List[str]:
    if len(sentences) <= 3:
        return sentences
    intro = sentences[0]
    outro = sentences[-1]
    middle = sentences[1:-1]
    random.shuffle(middle)
    return [intro] + middle + [outro]

# Add rhetorical questions, variation, and break formality
def soften_and_variabilize(text):
    sentences = sent_tokenize(text)
    rewritten = []
    
    for s in sentences:
        if random.random() < 0.3:
            if s.strip().endswith("."):
                s = s.strip()[:-1] + "?"
        s = humanize_phrasing(s)
        rewritten.append(s)
    
    return " ".join(rewritten)

# Word-level tweaking: replace synonyms randomly (for content words only)
def inject_synonyms(text):
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    new_tokens = []

    for word, tag in tagged:
        if tag.startswith("NN") or tag.startswith("VB") or tag.startswith("JJ"):
            if random.random() < 0.15:
                syns = wordnet.synsets(word)
                lemmas = [l.name().replace('_', ' ') for s in syns for l in s.lemmas()]
                clean_lemmas = [w for w in lemmas if w.lower() != word.lower() and w.isalpha()]
                if clean_lemmas:
                    word = random.choice(clean_lemmas)
        new_tokens.append(word)
    
    return " ".join(new_tokens)

# Final rewrite function
def rewrite(text: str) -> str:
    sentences = sent_tokenize(text)
    
    # Rearrange mid-sentences
    shuffled = rearrange_sentences(sentences)
    chunked_text = " ".join(shuffled)

    # Inject rhetorical variation
    bursty = soften_and_variabilize(chunked_text)

    # Word-level tweaks (with synonyms and contractions)
    softened = inject_synonyms(bursty)
    final = apply_contractions(softened)

    return final
