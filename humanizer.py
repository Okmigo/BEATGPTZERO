# humanizer.py: Module to rewrite AI text into human-like text
import config
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Load the paraphrasing model (T5) and sentence embedding model
# We create the pipeline with default settings; sampling is applied at call time.
_paraphraser = pipeline(
    "text2text-generation",
    model=config.PARAPHRASER_MODEL,
    tokenizer=config.PARAPHRASER_MODEL
)
# Load a sentence-similarity model (optional, ensures meaning is preserved)
_sim_model = SentenceTransformer(config.SIMILARITY_MODEL)

def rewrite_text(text):
    """
    Rewrite the given text to sound natural and human-like, preserving meaning and tone.
    Uses a text2text model with sampling for diversity.
    """
    # Prepare an instruction prompt for paraphrasing
    prompt = (
        "Rewrite the following text so it reads naturally with a human-like tone, "
        "preserving the original meaning and tone:\n\n"
        + text.strip()
    )
    # Generate paraphrase with sampling (introduce randomness)
    result = _paraphraser(
        prompt,
        max_length=config.MAX_LENGTH,
        do_sample=True,
        top_p=config.TOP_P,
        temperature=config.TEMPERATURE
    )
    rewritten = result[0]["generated_text"]

    # (Optional) Check semantic similarity to ensure meaning is retained
    try:
        orig_emb = _sim_model.encode(text, convert_to_tensor=True)
        new_emb  = _sim_model.encode(rewritten, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(orig_emb, new_emb).item()
        # If similarity is very low (<0.85), we could regenerate or warn
        if sim < 0.85:
            # Regenerate with more conservative settings (lower randomness)
            result2 = _paraphraser(
                prompt,
                max_length=config.MAX_LENGTH,
                do_sample=False  # Greedy decode for fidelity
            )
            candidate = result2[0]["generated_text"]
            # Use whichever output has higher similarity to original
            sim2 = util.pytorch_cos_sim(orig_emb, _sim_model.encode(candidate, convert_to_tensor=True)).item()
            if sim2 > sim:
                rewritten = candidate
    except Exception:
        # If embedding fails or model not available, just skip similarity check
        pass

    return rewritten
