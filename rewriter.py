
import random
import re

SYNONYMS = {
    "sustainable": ["eco-friendly", "green", "environmentally responsible"],
    "technology": ["tech", "innovation", "digital tools"],
    "complicated": ["complex", "intricate", "challenging"],
    "beautifully": ["elegantly", "gracefully", "nicely"],
    "future": ["tomorrow", "ahead", "next chapter"],
    "important": ["crucial", "vital", "key"],
    "systems": ["frameworks", "structures", "networks"],
    "cleaner": ["less polluting", "more eco-friendly", "lower emission"]
}

def synonym_replace(word):
    key = word.lower()
    if key in SYNONYMS:
        return random.choice(SYNONYMS[key])
    return word

def rewrite_text(input_text: str) -> str:
    words = re.split(r'(\W+)', input_text)
    rewritten_words = [synonym_replace(word) if word.isalpha() else word for word in words]
    result = ''.join(rewritten_words)

    sentences = re.split(r'(\.|\!|\?)', result)
    paired = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
    random.shuffle(paired)
    return ' '.join(paired)
