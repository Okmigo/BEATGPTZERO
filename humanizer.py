import random, re
import nltk, torch
from nltk.corpus import wordnet as wn
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from config import NUM_BEAMS, NUM_CANDIDATES, CUDA_DEVICE

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

DEVICE = f"cuda:{CUDA_DEVICE}" if CUDA_DEVICE >= 0 and torch.cuda.is_available() else "cpu"
MODEL_NAME = "tuner007/pegasus_paraphrase"   # free Hugging Face checkpoint[2]
_tok = PegasusTokenizer.from_pretrained(MODEL_NAME)
_model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

_AI_BUZZ = {"leverage": "use", "harness": "tap", "tapestry": "fabric", "utilize": "use"}

def _replace_buzzwords(sent: str) -> str:
    for k, v in _AI_BUZZ.items():
        sent = re.sub(rf"\b{k}\b", v, sent, flags=re.I)
    return sent

def _random_synonym(word):
    syns = [l.name().replace('_', ' ')
            for s in wn.synsets(word) for l in s.lemmas()
            if l.name().lower() != word.lower() and word.isalpha()]
    return random.choice(syns) if syns else word

def _lexical_variation(text: str) -> str:
    tokens = nltk.word_tokenize(text)
    mutated = [ _random_synonym(t) if random.random() < 0.15 else t for t in tokens ]
    return nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(mutated)

def rewrite_once(text: str) -> str:
    # 1 – quick lexical pre-pass
    text = _replace_buzzwords(_lexical_variation(text))
    # 2 – Pegasus paraphrase
    batch = _tok([text], truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
    outs  = _model.generate(**batch,
                            num_beams=NUM_BEAMS,
                            num_return_sequences=NUM_CANDIDATES,
                            temperature=1.5,
                            max_length=512)
    candidates = _tok.batch_decode(outs, skip_special_tokens=True)
    # return one candidate randomly for diversity
    return random.choice(candidates)
