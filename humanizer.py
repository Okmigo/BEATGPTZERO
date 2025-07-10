# humanizer.py

import torch
import random
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Download the 'punkt' tokenizer data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

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
        We use the 'tuner007/pegasus_paraphrase' model, which is highly effective
        for generating high-quality paraphrases.
        """
        model_name = 'tuner007/pegasus_paraphrase'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the tokenizer and model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print("TextHumanizer initialized successfully on device:", self.device)

    def humanize(self, text: str, num_return_sequences: int = 3, temperature: float = 1.5) -> str:
        """
        Transforms AI-generated text to be more human-like.

        Args:
            text (str): The AI-generated text to transform.
            num_return_sequences (int): The number of diverse paraphrases to generate for each sentence.
                                       A higher number increases variety.
            temperature (float): Controls the randomness of the output. Higher values (e.g., 1.5)
                                 produce more creative and diverse results, which is key to
                                 evading detection.

        Returns:
            str: The transformed, humanized text.
        """
        if not text.strip():
            return ""

        # 1. Disrupt AI-likeness by splitting text into sentences
        # AI detectors often analyze text as a whole. By processing sentence by
        # sentence, we break down the statistical uniformity.
        sentences = nltk.sent_tokenize(text)

        humanized_sentences = []
        for sentence in sentences:
            # 2. Encode the sentence for the model
            inputs = self.tokenizer.encode(sentence, return_tensors='pt', truncation=True).to(self.device)
            
            # 3. Generate multiple, diverse paraphrases using controlled randomness (temperature)
            # This is the core of the evasion strategy. We create multiple candidate
            # sentences that are grammatically correct but stylistically different.
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=256,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95, # Nucleus sampling
                    num_beams=num_return_sequences  # Use beam search for higher quality
                )

            # 4. Decode the generated sentences
            paraphrases = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            # 5. Introduce "burstiness" by randomly selecting one of the diverse paraphrases.
            # This variation in sentence structure is a strong signal of human writing.
            if paraphrases:
                selected_paraphrase = random.choice(paraphrases)
                humanized_sentences.append(selected_paraphrase)
            else:
                # If no paraphrase is generated, keep the original sentence
                humanized_sentences.append(sentence)

        # 6. Reconstruct the text, preserving the transformed style
        return ' '.join(humanized_sentences)
