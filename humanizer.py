import random
import re
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load models once
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")


def rewrite(text: str) -> str:
    sentences = sent_tokenize(text)
    rewritten_sentences = []

    for i, sentence in enumerate(sentences):
        # Step 1: Prompt diversity injection
        prompt_variants = [
            f"Rewrite the sentence to reflect a human academic writing style without sounding robotic: {sentence}",
            f"Reword this with natural variation while preserving its academic tone and core meaning: {sentence}",
            f"Transform this sentence into one that sounds like it was written by a human academic: {sentence}",
            f"Recast this sentence using subtle human expression and rhythm: {sentence}",
            f"Reshape the sentence to avoid formulaic AI tone: {sentence}"
        ]
        prompt = random.choice(prompt_variants)

        inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = t5_model.generate(
            inputs.input_ids,
            max_length=256,
            temperature=random.uniform(0.9, 1.3),  # Inject non-determinism
            top_p=0.95,
            do_sample=True
        )
        rewritten = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Step 2: Inject mild syntax shifts
        rewritten = tweak_syntax_subtly(rewritten)

        rewritten_sentences.append(rewritten)

    final_text = " ".join(rewritten_sentences)
    return final_text


def tweak_syntax_subtly(sentence: str) -> str:
    # Insert mid-sentence clauses or shuffle minor punctuation for human idiosyncrasy
    tweaks = [
        lambda s: re.sub(r"(\b\w{5,}\b)", r"\1,", s, count=1),  # Inject a comma
        lambda s: re.sub(r"([a-zA-Z]{4,})\.", r"\1,", s),       # Replace period with comma if stylistically safe
        lambda s: s.replace(" and ", ", and "),                # More organic list patterns
        lambda s: s.replace(" but ", ", but "),                # Softened transitions
    ]
    random.shuffle(tweaks)
    for tweak in tweaks:
        try:
            s = tweak(sentence)
            if len(s.split()) > 5:
                return s
        except:
            continue
    return sentence
