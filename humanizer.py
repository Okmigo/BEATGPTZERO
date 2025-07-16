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

# Load paraphraser (T5) and GPT-2 (for scoring)
paraphrase_tokenizer = AutoTokenizer.from_pretrained(config.PARAPHRASER_MODEL)
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(config.PARAPHRASER_MODEL)
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(config.GPT2_MODEL)
gpt2_model = GPT2LMHeadModel.from_pretrained(config.GPT2_MODEL)

def replace_synonyms(text, prob=None):
    if prob is None:
        prob = config.SYNONYM_PROBABILITY
    tokens = re.findall(r"\w+|[^\w\s]", text)
    new_tokens = []
    for tok in tokens:
        if tok.isalpha() and random.random() < prob:
            synsets = wn.synsets(tok)
            synonyms = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    name = lemma.name().lower().replace('_', ' ')
                    if name != tok.lower():
                        synonyms.add(name)
            if synonyms:
                choice = random.choice(list(synonyms))
                if tok[0].isupper():
                    choice = choice.capitalize()
                new_tokens.append(choice)
                continue
        new_tokens.append(tok)
    s = ' '.join(new_tokens)
    s = re.sub(r'\s+([.,!?;])', r'\1', s)
    return s

def chunk_text(text, max_words):
    sentences = sent_tokenize(text)
    chunks = []
    current = ""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate.split()) > max_words:
            if current:
                chunks.append(current.strip())
                current = sent
            else:
                chunks.append(candidate.strip())
                current = ""
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return chunks

def compute_perplexity(text):
    enc = gpt2_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2_model(**enc, labels=enc['input_ids'])
        loss = outputs.loss
    return float(torch.exp(loss).item())

def paraphrase_chunk(chunk, num_outputs=3):
    chunk_syn = replace_synonyms(chunk)
    input_text = "paraphrase: " + chunk_syn
    inputs = paraphrase_tokenizer.encode(input_text, return_tensors='pt', truncation=True)
    max_len = min(inputs.shape[-1] + 50, 512)
    outputs = paraphrase_model.generate(
        inputs, max_length=max_len, do_sample=True,
        top_k=50, top_p=0.9, num_return_sequences=num_outputs,
        no_repeat_ngram_size=2, early_stopping=True
    )
    paraphrases = [paraphrase_tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
    best = None
    best_score = -float('inf')
    for para in paraphrases:
        score = compute_perplexity(para)
        if score > best_score:
            best_score = score
            best = para
    return best if best is not None else chunk

def humanize_text(text, style_profile=None):
    """
    Full pipeline: split text into chunks (adapting to user style if given), paraphrase each, and rejoin.
    """
    # Adjust chunk size based on user style (average sentence length)
    chunk_size = config.CHUNK_SIZE
    if style_profile and style_profile.get('avg_len'):
        avg_len = style_profile['avg_len']
        if avg_len and avg_len < config.CHUNK_SIZE:
            chunk_size = max(50, min(config.CHUNK_SIZE, int(avg_len * 4)))
    chunks = chunk_text(text, chunk_size)
    rewritten_chunks = []
    for chunk in chunks:
        try:
            best_para = paraphrase_chunk(chunk, num_outputs=config.PARAPHRASER_NUM_OUTPUTS)
            rewritten_chunks.append(best_para)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Paraphrasing failed for chunk: {e}")
            rewritten_chunks.append(chunk)
    return " ".join(rewritten_chunks)
