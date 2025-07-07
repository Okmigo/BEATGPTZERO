
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Load quantized paraphrasing model
tokenizer = AutoTokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")
model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/parrot_paraphraser_on_T5")

paraphraser = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

def rewrite_text(text: str) -> str:
    try:
        outputs = paraphraser(f"paraphrase: {text} </s>", max_length=256, do_sample=True, top_k=120, top_p=0.98, temperature=0.7)
        return outputs[0]["generated_text"]
    except Exception as e:
        return f"[Rewrite Error]: {e}"
