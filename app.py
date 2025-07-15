# app.py

import os
import random
import threading
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import spacy
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import lmppl
import nltk
from nltk.corpus import wordnet
import textstat

# --- Configuration and Initialization ---

# Setup basic logging to see output in Cloud Run logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from a .env file if present
load_dotenv()

# --- Pydantic Models for API Schema ---

class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Text to analyze and humanize. A minimum of 50 words is recommended for meaningful analysis.")

class StylometricProfile(BaseModel):
    ttr: float
    sentence_length_mean: float
    sentence_length_std: float
    readability_flesch_kincaid: float

class HumanizeResponse(BaseModel):
    original_text: str
    humanized_text: str
    original_profile: StylometricProfile
    final_profile: StylometricProfile
    is_humanized: bool = Field(..., description="True if the rewritten text meets the target stylometric profile.")
    perplexity: float
    burstiness: float

# --- Singleton Model Manager for Stability ---

class ModelManager:
    """
    A thread-safe Singleton class to manage loading and access of all ML models.
    This ensures models are loaded only once at application startup, preventing memory issues.
    """
    _instance = None
    _lock = threading.Lock()
    is_ready: bool = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance.pipeline = None
        return cls._instance

    def load_models(self):
        """Loads all necessary models into memory. To be called explicitly on startup."""
        if self.is_ready:
            logger.info("Models are already loaded.")
            return

        with self._lock:
            if self.is_ready:
                return

            try:
                logger.info("Initializing ModelManager and loading models...")
                start_time = time.time()
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {self.device}")

                # 1. spaCy for linguistic analysis
                self.spacy_nlp = spacy.load("en_core_web_lg")

                # 2. Perplexity model
                self.perplexity_scorer = lmppl.LM('gpt2', device=self.device)

                # 3. Humanization model (Pegasus-based style transfer)
                humanizer_model_name = "Eemansleepdeprived/Humaneyes"
                self.humanizer_tokenizer = PegasusTokenizer.from_pretrained(humanizer_model_name)
                self.humanizer_model = PegasusForConditionalGeneration.from_pretrained(humanizer_model_name).to(self.device)

                # 4. Initialize the full V3 pipeline
                self.pipeline = HumanizationPipelineV3(
                    spacy_nlp=self.spacy_nlp,
                    perplexity_scorer=self.perplexity_scorer,
                    humanizer_tokenizer=self.humanizer_tokenizer,
                    humanizer_model=self.humanizer_model,
                    device=self.device
                )
                
                self.is_ready = True
                end_time = time.time()
                logger.info(f"All models and pipeline loaded successfully in {end_time - start_time:.2f} seconds. Service is ready.")

            except Exception as e:
                logger.error(f"FATAL: Model loading failed: {e}", exc_info=True)
                self.is_ready = False

    def get_pipeline(self):
        if not self.is_ready or not self.pipeline:
            raise RuntimeError("Pipeline is not available. Models may not have loaded correctly.")
        return self.pipeline

# --- The Advanced Humanization Pipeline V3 ---

class HumanizationPipelineV3:
    """
    A multi-stage pipeline that uses stylometry to transform AI text.
    It analyzes the text's "fingerprint" and intelligently perturbs it to match human patterns.
    """
    def __init__(self, spacy_nlp, perplexity_scorer, humanizer_tokenizer, humanizer_model, device):
        self.nlp = spacy_nlp
        self.perplexity_scorer = perplexity_scorer
        self.humanizer_tokenizer = humanizer_tokenizer
        self.humanizer_model = humanizer_model
        self.device = device

        self.TARGET_TTR = 0.55
        self.TARGET_SENT_LEN_STD = 7.0
        self.TARGET_READABILITY_MIN = 9.0
        self.TARGET_READABILITY_MAX = 14.0

    def _analyze_stylometry(self, text: str) -> StylometricProfile:
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc if token.is_alpha]
        sents = list(doc.sents)
        
        ttr = len(set(tokens)) / len(tokens) if tokens else 0.0
        
        sent_lengths = [len([tok for tok in sent if tok.is_alpha]) for sent in sents]
        sent_len_mean = np.mean(sent_lengths) if sent_lengths else 0.0
        sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0.0
        
        readability = textstat.flesch_kincaid_grade(text)
        
        return StylometricProfile(
            ttr=ttr,
            sentence_length_mean=sent_len_mean,
            sentence_length_std=sent_len_std,
            readability_flesch_kincaid=readability
        )

    def _targeted_lexical_perturbation(self, text: str, profile: StylometricProfile) -> str:
        ttr_gap = self.TARGET_TTR - profile.ttr
        replacement_rate = max(0.05, min(0.25, ttr_gap * 0.5))
        
        doc = self.nlp(text)
        tokens = list(doc)
        new_tokens = [token.text_with_ws for token in tokens]

        eligible_indices = [i for i, t in enumerate(tokens) if t.is_alpha and not t.is_stop]
        num_to_replace = int(len(eligible_indices) * replacement_rate)
        
        if num_to_replace > 0:
            indices_to_replace = random.sample(eligible_indices, num_to_replace)
            for i in indices_to_replace:
                token = tokens[i]
                synonyms = set()
                for syn in wordnet.synsets(token.lemma_):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym.lower() != token.lemma_.lower():
                            synonyms.add(synonym)
                
                if synonyms:
                    replacement = random.choice(list(synonyms))
                    if token.is_title: replacement = replacement.title()
                    elif token.is_upper: replacement = replacement.upper()
                    new_tokens[i] = replacement + token.whitespace_

        return "".join(new_tokens)

    def _targeted_structural_perturbation(self, text: str, profile: StylometricProfile) -> str:
        if profile.sentence_length_std >= self.TARGET_SENT_LEN_STD:
            return text

        sents = nltk.sent_tokenize(text)
        if len(sents) < 3:
            return text

        short_sents = sorted([(i, len(s.split())) for i, s in enumerate(sents)], key=lambda x: x[1])
        if len(short_sents) >= 2 and short_sents[0][1] < 12 and short_sents[1][1] < 12:
            i, j = sorted([short_sents[0][0], short_sents[1][0]])
            sent1 = sents[i].strip().rstrip('.')
            sent2 = sents[j].lower().strip()
            merged_sent = f"{sent1}, and {sent2}"
            
            new_sents = [s for k, s in enumerate(sents) if k not in [i, j]]
            new_sents.insert(i, merged_sent)
            return " ".join(new_sents)

        long_sents = sorted([(i, len(s.split())) for i, s in enumerate(sents)], key=lambda x: x[1], reverse=True)
        if long_sents and long_sents[0][1] > 35:
            sent_to_split_idx = long_sents[0][0]
            sent_to_split = sents[sent_to_split_idx]
            words = sent_to_split.split()
            split_point = len(words) // 2
            part1 = " ".join(words[:split_point]) + "."
            part2 = " ".join(words[split_point:]).capitalize()
            
            sents[sent_to_split_idx:sent_to_split_idx+1] = [part1, part2]
            return " ".join(sents)

        return text

    def _stylistic_transfer(self, text: str) -> str:
        try:
            input_ids = self.humanizer_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
            outputs = self.humanizer_model.generate(
                input_ids,
                max_length=int(input_ids.shape[1] * 1.5),
                num_beams=5,
                early_stopping=True,
                temperature=1.2,
                top_k=50
            )
            humanized_text = self.humanizer_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return humanized_text
        except Exception as e:
            logger.warning(f"Error during stylistic transfer: {e}")
            return text

    def _validate_output(self, text: str, profile: StylometricProfile) -> Tuple[float, float, bool]:
        burstiness = profile.sentence_length_std
        
        is_humanized = (
            profile.ttr >= self.TARGET_TTR and
            burstiness >= self.TARGET_SENT_LEN_STD and
            self.TARGET_READABILITY_MIN <= profile.readability_flesch_kincaid <= self.TARGET_READABILITY_MAX
        )
        
        try:
            perplexity = self.perplexity_scorer.get_perplexity([text.strip()]) if text.strip() else 0.0
        except Exception as e:
            logger.warning(f"Error calculating perplexity: {e}")
            perplexity = 0.0

        return perplexity, burstiness, is_humanized

    def run(self, text: str) -> Tuple:
        original_profile = self._analyze_stylometry(text)
        
        perturbed_text = text
        for _ in range(2):
            current_profile = self._analyze_stylometry(perturbed_text)
            perturbed_text = self._targeted_lexical_perturbation(perturbed_text, current_profile)
            perturbed_text = self._targeted_structural_perturbation(perturbed_text, self._analyze_stylometry(perturbed_text))

        humanized_text = self._stylistic_transfer(perturbed_text)
        
        final_profile = self._analyze_stylometry(humanized_text)
        perplexity, burstiness, is_humanized = self._validate_output(humanized_text, final_profile)

        return humanized_text, original_profile, final_profile, is_humanized, perplexity, burstiness

# --- FastAPI Application Setup ---

app = FastAPI(
    title="Humanizer V3 API",
    description="An API that rewrites text using a multi-stage stylometric pipeline to bypass advanced AI detectors.",
    version="3.0.0",
)

model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """On startup, trigger the loading of models in a background thread."""
    thread = threading.Thread(target=model_manager.load_models)
    thread.start()

@app.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint for Cloud Run. Returns OK immediately to satisfy the startup probe.
    """
    return {"status": "ok", "message": "Web server is running, models may still be loading."}

@app.post("/humanize", response_model=HumanizeResponse)
async def humanize_text_endpoint(request: HumanizeRequest):
    """
    This endpoint takes AI-generated text and processes it through the Humanizer V3 pipeline.
    """
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Service Unavailable: Models are still loading. Please try again in a moment.")

    try:
        pipeline = model_manager.get_pipeline()
        humanized_text, original_profile, final_profile, is_humanized, perplexity, burstiness = pipeline.run(request.text)

        return HumanizeResponse(
            original_text=request.text,
            humanized_text=humanized_text,
            original_profile=original_profile,
            final_profile=final_profile,
            is_humanized=is_humanized,
            perplexity=perplexity,
            burstiness=burstiness
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during humanization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred during text processing.")
