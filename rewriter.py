# File: rewriter.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Rewriter:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def rewrite(self, text, decoding="sampling"):
        # Prepare input with prompt
        input_text = f"paraphrase: {text} </s>"
        encoding = self.tokenizer.encode_plus(
            input_text,
            padding='max_length',
            max_length=256,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        # Decoding options
        if decoding == "sampling":
            outputs = self.model.generate(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                max_length=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.9,
                num_return_sequences=1
            )
        elif decoding == "beam":
            outputs = self.model.generate(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                max_length=256,
                num_beams=5,
                early_stopping=True
            )
        else:
            raise ValueError("Unsupported decoding strategy")

        paraphrased = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return paraphrased
