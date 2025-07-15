import re
import random
import torch
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
import nltk

import config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2TokenizerFast, GPT2LMHeadModel

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Load paraphrasing model (T5 small fine-tuned on Quora)
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(config.PARAPHRASER_MODEL)
paraphrase_tokenizer = AutoTokenizer.from_pretrained(config.PARAPHRASER_MODEL)

# Load GPT-2 model for perplexity scoring
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(config.GPT2_MODEL)
gpt2_model = GPT2LMHeadModel.from_pretrained(config.GPT2_MODEL)

def replace_synonyms(text, prob=None):
    """
    Replace words with synonyms from WordNet with given probability.
    Only replaces alphabetic tokens; keeps punctuation intact.
    """
    if prob is None:
        prob = config.SYNONYM_PROBABILITY
    tokens = re.findall(r"\w+|[^\w\s]", text)
    new_tokens = []
    for tok in tokens:
        # Only attempt for alphabetic tokens
        if tok.isalpha() and random.random() < prob:
            synsets = wn.synsets(tok)
            synonyms = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    name = lemma.name().lower().replace('_', ' ')
                    if name != tok.lower():
                        synonyms.add(name)
            if synonyms:
                # Choose a random synonym, preserve case if needed
                choice = random.choice(list(synonyms))
                # Match capitalization
                if tok[0].isupper():
                    choice = choice.capitalize()
                new_tokens.append(choice)
                continue
        new_tokens.append(tok)
    # Reconstruct text
    s = ' '.join(new_tokens)
    s = re.sub(r'\s+([.,!?;])', r'\1', s)  # fix spacing before punctuation
    return s

def chunk_text(text, max_words):
    """
    Break text into chunks of roughly max_words (by whitespace count), 
    using sentence boundaries to avoid cutting mid-sentence.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current = ""
    for sent in sentences:
        if current:
            candidate = current + " " + sent
        else:
            candidate = sent
        if len(candidate.split()) > max_words:
            if current:
                chunks.append(current.strip())
                current = sent
            else:
                # single sentence too long; force split
                chunks.append(candidate.strip())
                current = ""
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return chunks

def compute_perplexity(text):
    """
    Compute GPT-2 perplexity of the given text.
    """
    enc = gpt2_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2_model(**enc, labels=enc['input_ids'])
        loss = outputs.loss
    # Perplexity = exp(loss)
    return float(torch.exp(loss).item())

def paraphrase_chunk(chunk, num_outputs=3):
    """
    Generate multiple paraphrases for a text chunk using T5 and return the best one by GPT-2 perplexity.
    """
    # Optionally apply synonym replacement before paraphrasing
    chunk_syn = replace_synonyms(chunk)

    input_text = "paraphrase: " + chunk_syn
    inputs = paraphrase_tokenizer.encode(input_text, return_tensors='pt', truncation=True)
    max_len = min(inputs.shape[-1] + 50, 512)  # limit output length
    # Generate multiple outputs via sampling for variety
    outputs = paraphrase_model.generate(
        inputs, max_length=max_len, do_sample=True,
        top_k=50, top_p=0.9, num_return_sequences=num_outputs,
        no_repeat_ngram_size=2, early_stopping=True
    )
    paraphrases = [paraphrase_tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
    # Score each paraphrase by GPT-2 perplexity (higher is more "human-like")
    best = None
    best_score = -float('inf')
    for para in paraphrases:
        score = compute_perplexity(para)
        if score > best_score:
            best_score = score
            best = para
    return best if best is not None else chunk

def humanize_text(text):
    """
    Full pipeline: split text into chunks, paraphrase each, and rejoin.
    """
    chunks = chunk_text(text, config.CHUNK_SIZE)
    rewritten_chunks = []
    for chunk in chunks:
        try:
            best_para = paraphrase_chunk(chunk, num_outputs=config.PARAPHRASER_NUM_OUTPUTS)
            rewritten_chunks.append(best_para)
        except Exception as e:
            # If paraphrasing fails for a chunk, fall back to the original chunk
            logger = logging.getLogger(__name__)
            logger.error(f"Paraphrasing failed for chunk: {e}")
            rewritten_chunks.append(chunk)
    # Rejoin with space (sentence boundaries preserved by punctuation fix)
    return " ".join(rewritten_chunks)
