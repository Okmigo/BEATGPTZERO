# humanizer.py

import torch
import random
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TextHumanizer:
    """
    A sophisticated text humanizer using a transformer-based paraphrasing model.
    This approach evades AI detection by generating diverse, fluent, and
    contextually appropriate paraphrases, which disrupts statistical patterns
    like perplexity and token distribution that detectors rely on.
    """
    def __init__(self):
        """
        Initializes the model and tokenizer.
        This is a one-time setup process that loads the pre-trained model into memory.
        The model files were pre-downloaded into the Docker image for fast startup.
        """
        model_name = 'tuner007/pegasus_paraphrase'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the tokenizer and model from the local files within the container
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print("TextHumanizer initialized successfully on device:", self.device)

    def humanize(self, text: str, num_return_sequences: int = 3, temperature: float = 1.5) -> str:
        """
        Transforms AI-generated text to be more human-like.
        """
        if not text.strip():
            return ""

        # Use the pre-loaded NLTK 'punkt' model to split text into sentences.
        sentences = nltk.sent_tokenize(text)

        humanized_sentences = []
        for sentence in sentences:
            inputs = self.tokenizer.encode(sentence, return_tensors='pt', truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=256,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    num_beams=num_return_sequences
                )

            paraphrases = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            if paraphrases:
                selected_paraphrase = random.choice(paraphrases)
                humanized_sentences.append(selected_paraphrase)
            else:
                humanized_sentences.append(sentence)

        return ' '.join(humanized_sentences)
