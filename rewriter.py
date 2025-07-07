from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import random
import re

tokenizer = AutoTokenizer.from_pretrained(
    "prithivida/parrot_paraphraser_on_T5",
    cache_dir="./hf_cache",
    local_files_only=False
)
model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/parrot_paraphraser_on_T5")
paraphraser = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

translator_en_fr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
translator_fr_en = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")

def rewrite_text(text: str) -> str:
    try:
        outputs = paraphraser(f"paraphrase: {text} </s>", max_length=256,
                              do_sample=True, top_k=50, top_p=0.95, temperature=0.9,
                              num_return_sequences=3)
        candidate = random.choice(outputs)["generated_text"].strip()

        translated = translator_en_fr(candidate, max_length=512)[0]['translation_text']
        back_translated = translator_fr_en(translated, max_length=512)[0]['translation_text']

        outputs2 = paraphraser(f"paraphrase: {back_translated} </s>", max_length=256,
                               do_sample=True, top_k=50, top_p=0.95, temperature=0.9)
        result = outputs2[0]["generated_text"].strip()

        contractions = {
            " do not ": " don't ", " does not ": " doesn't ",
            " cannot ": " can't ", " could not ": " couldn't ",
            " will not ": " won't ", " I am ": " I'm ",
            " we are ": " we're ", " it is ": " it's ",
            " there is ": " there's "
        }
        for key, val in contractions.items():
            result = result.replace(key, val).replace(key.title(), val.title())

        sentences = re.split(r'(?<=[.?!])\s+', result)
        new_sentences = []
        for sent in sentences:
            if len(sent) > 100 and ',' in sent:
                parts = sent.split(',', 1)
                new_sentences.append(parts[0].strip() + '.')
                new_sentences.append(parts[1].strip() + '.')
            else:
                new_sentences.append(sent)

        merged = []
        i = 0
        while i < len(new_sentences):
            if i+1 < len(new_sentences) and len(new_sentences[i].split()) < 5:
                combined = new_sentences[i].rstrip('.').strip() + ' and ' + new_sentences[i+1].strip().capitalize()
                if combined[-1] not in '.?!':
                    combined += '.'
                merged.append(combined)
                i += 2
            else:
                merged.append(new_sentences[i])
                i += 1
        result = ' '.join(merged)

        fillers = ["Anyway, ", "Interestingly, ", "You know, "]
        if random.random() < 0.3:
            result = random.choice(fillers) + result

        return result

    except Exception as e:
        return f"[Rewrite Error]: {e}"
