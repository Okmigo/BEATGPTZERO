# app.py
import logging
import random
import re
import nltk
from flask import Flask, request, jsonify
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

# Configuration parameters (tunable)
CONFIG = {
    "synonym_replacement_prob": 0.15,
    "sentence_restructure_prob": 0.25,
    "noise_injection_prob": 0.1,
    "sentence_merge_prob": 0.15,
    "sentence_split_prob": 0.2,
    "max_typos_per_100_words": 2,
    "filler_words": ["well", "you know", "I mean", "actually", "basically"],
    "discourse_markers": ["however", "furthermore", "nevertheless", "consequently"]
}

def log_transformation(original, transformed, transformations):
    """Log transformation metadata for iterative improvement"""
    logging.info(f"Original: {original[:50]}... | Transformed: {transformed[:50]}...")
    logging.debug(f"Applied transformations: {', '.join(transformations)}")

def calculate_text_features(text):
    """Calculate key detection metrics (simplified)"""
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return {
        "sentence_count": len(sentences),
        "avg_sentence_length": len(words)/len(sentences) if sentences else 0,
        "unique_word_ratio": len(set(words))/len(words) if words else 0
    }

def get_synonyms(word, pos_tag):
    """Get contextually relevant synonyms with POS matching"""
    pos_mapping = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'J': wordnet.ADJ, 'R': wordnet.ADV}
    simplified_pos = pos_tag[0].upper() if pos_tag else None
    wordnet_pos = pos_mapping.get(simplified_pos, wordnet.NOUN)
    
    synonyms = set()
    for syn in wordnet.synsets(word, pos=wordnet_pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower() and 3 <= len(synonym) <= 25:
                synonyms.add(synonym)
    return list(synonyms)[:3]  # Return max 3 relevant synonyms

def inject_syntactic_irregularity(sentence):
    """
    Introduce human-like syntactic variations:
    1. Randomly change sentence structure (active/passive)
    2. Add natural interruptions
    3. Vary conjunctions
    """
    transformations = []
    
    # Add parenthetical expressions
    if random.random() < 0.15 and len(sentence) > 40:
        insert_point = random.randint(int(len(sentence)*0.3), int(len(sentence)*0.7))
        parentheses_content = random.choice(["which is interesting", "as you might know", "generally speaking"])
        sentence = f"{sentence[:insert_point]} ({parentheses_content}){sentence[insert_point:]}"
        transformations.append("parenthetical")
    
    # Vary conjunctions
    conj_mapping = {"and": "but", "but": "and", "however": "nevertheless", "therefore": "consequently"}
    for original, replacement in conj_mapping.items():
        if original in sentence and random.random() < 0.4:
            sentence = sentence.replace(original, replacement, 1)
            transformations.append("conjunction_variation")
            break
    
    # Introduce left-dislocation (common in speech)
    if random.random() < 0.1 and sentence.count(',') > 1:
        parts = [p.strip() for p in sentence.split(',')]
        if len(parts) > 2 and not parts[0].endswith('?'):
            sentence = f"{parts[1]}, {parts[0]}, {', '.join(parts[2:])}"
            transformations.append("left_dislocation")
    
    return sentence, transformations

def semantic_reweighting(sentence):
    """
    Replace predictable word choices with semantically similar alternatives:
    1. Use synonym replacement with POS preservation
    2. Avoid overused AI vocabulary
    """
    transformations = []
    words = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    new_sentence = []
    
    for i, (word, tag) in enumerate(pos_tags):
        if (random.random() < CONFIG["synonym_replacement_prob"] and 
            word.isalpha() and len(word) > 3 and 
            tag.startswith(('N', 'V', 'J', 'R'))):
            
            synonyms = get_synonyms(word, tag)
            if synonyms:
                replacement = random.choice(synonyms)
                new_sentence.append(replacement)
                transformations.append(f"synonym:{word}->{replacement}")
                continue
        
        new_sentence.append(word)
    
    return ' '.join(new_sentence), transformations

def introduce_controlled_noise(text):
    """
    Add human-like noise features:
    1. Occasional filler phrases
    2. Mild punctuation variations
    3. Rare typos (strategic)
    """
    transformations = []
    sentences = sent_tokenize(text)
    
    # Add discourse markers
    if random.random() < CONFIG["noise_injection_prob"] and len(sentences) > 1:
        marker = random.choice(CONFIG["discourse_markers"])
        sentences[0] = f"{marker.capitalize()}, {sentences[0].lower()}"
        transformations.append("discourse_marker")
    
    # Introduce fillers
    if random.random() < 0.08:
        idx = random.randint(0, len(sentences)-1)
        filler = random.choice(CONFIG["filler_words"])
        sentences[idx] = f"{filler.capitalize()}, {sentences[idx].lower()}" 
        transformations.append("filler_injection")
    
    # Strategic typos (low frequency)
    word_count = sum(len(sent.split()) for sent in sentences)
    typo_count = min(CONFIG["max_typos_per_100_words"], max(1, word_count//100))
    for _ in range(typo_count):
        sent_idx = random.randint(0, len(sentences)-1)
        words = sentences[sent_idx].split()
        if len(words) > 3:
            word_idx = random.randint(0, len(words)-1)
            if len(words[word_idx]) > 4:
                # Common typo patterns: duplicate letter, vowel swap
                typo_type = random.choice(['double', 'vowel'])
                original = words[word_idx]
                if typo_type == 'double' and not any(c*2 in original for c in 'aeiou'):
                    char_idx = random.randint(1, len(original)-1)
                    words[word_idx] = original[:char_idx] + original[char_idx-1] + original[char_idx:]
                elif typo_type == 'vowel':
                    vowels = [i for i, c in enumerate(original) if c in 'aeiou']
                    if vowels:
                        pos = random.choice(vowels)
                        words[word_idx] = original[:pos] + random.choice('aeiou') + original[pos+1:]
                transformations.append(f"typo:{original}->{words[word_idx]}")
                sentences[sent_idx] = ' '.join(words)
    
    return ' '.join(sentences), transformations

def vary_sentence_structure(text):
    """
    Increase burstiness through:
    1. Strategic sentence merging
    2. Controlled sentence splitting
    3. Clause restructuring
    """
    transformations = []
    sentences = sent_tokenize(text)
    new_sentences = []
    i = 0
    
    while i < len(sentences):
        # Merge short sentences
        if (i < len(sentences)-1 and 
            random.random() < CONFIG["sentence_merge_prob"] and 
            len(sentences[i].split()) < 8 and 
            len(sentences[i+1].split()) < 10):
            
            merged = f"{sentences[i]} {sentences[i+1].lower()}"
            new_sentences.append(merged)
            transformations.append("sentence_merge")
            i += 2
            continue
        
        # Split long sentences
        current = sentences[i]
        if (random.random() < CONFIG["sentence_split_prob"] and 
            len(current.split()) > 18 and 
            ',' in current):
            
            parts = [p.strip() for p in current.split(',')]
            if len(parts) > 1:
                # Split at random comma point (preserving meaning)
                split_point = random.randint(1, len(parts)-1)
                part1 = ', '.join(parts[:split_point]) + '.'
                part2 = ' '.join(parts[split_point:]).capitalize()
                new_sentences.append(part1)
                new_sentences.append(part2)
                transformations.append("sentence_split")
                i += 1
                continue
        
        # Process normally
        new_sentences.append(current)
        i += 1
    
    return ' '.join(new_sentences), transformations

def humanize_text(text):
    """Main text transformation pipeline"""
    if not text.strip():
        return text, ["noop:empty_input"]
    
    original_features = calculate_text_features(text)
    transformations = []
    transformed = text
    
    # Apply transformation modules in randomized order
    modules = [
        ("syntactic", inject_syntactic_irregularity),
        ("semantic", semantic_reweighting),
        ("noise", introduce_controlled_noise),
        ("structure", vary_sentence_structure)
    ]
    random.shuffle(modules)
    
    for name, module in modules:
        if name == "structure":
            # Structure module processes full text
            result, mod_transforms = module(transformed)
        else:
            # Other modules process sentence-by-sentence
            sentences = sent_tokenize(transformed)
            modified = []
            for sent in sentences:
                result, sent_transforms = module(sent)
                modified.append(result)
                transformations.extend(sent_transforms)
            result = ' '.join(modified)
        
        transformed = result
        transformations.extend(mod_transforms)
    
    # Final cleanup
    transformed = re.sub(r'\s+([.,;])', r'\1', transformed)  # Fix spacing
    transformed = transformed.capitalize()  # Ensure proper capitalization
    
    # Log transformation metadata
    new_features = calculate_text_features(transformed)
    log_data = {
        "original_features": original_features,
        "transformed_features": new_features,
        "transformations": list(set(transformations))
    }
    logging.info(f"Transformation metrics: {log_data}")
    
    return transformed, transformations

@app.route('/humanize', methods=['POST'])
def humanize_endpoint():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' parameter"}), 400
    
    original_text = data['text']
    try:
        humanized, _ = humanize_text(original_text)
        return jsonify({"humanized_text": humanized})
    except Exception as e:
        logging.exception("Transformation failed")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=8080, debug=False)
