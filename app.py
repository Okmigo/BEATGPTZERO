# app.py
import asyncio
import logging
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any
from contextlib import asynccontextmanager
import time
from functools import lru_cache
import json

import numpy as np
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PegasusTokenizer, PegasusForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM, pipeline
)
from optimum.bettertransformer import BetterTransformer
import nltk
from nltk.corpus import wordnet
import textstat
from language_tool_python import LanguageTool
import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
import uvicorn

# Configure structured logging (unchanged)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Load environment variables
load_dotenv()

# Configuration
class Config:
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
    MODEL_PRECISION = os.getenv("MODEL_PRECISION", "fp16")
    PERPLEXITY_THRESHOLD = float(os.getenv("PERPLEXITY_THRESHOLD", "120.0"))
    BURSTINESS_THRESHOLD = float(os.getenv("BURSTINESS_THRESHOLD", "8.0"))
    READABILITY_THRESHOLD = float(os.getenv("READABILITY_THRESHOLD", "12.0"))
    LEXICAL_DIVERSITY_TARGET = float(os.getenv("LEXICAL_DIVERSITY_TARGET", "0.7"))
    SEMANTIC_COHERENCE_TARGET = float(os.getenv("SEMANTIC_COHERENCE_TARGET", "0.85"))

config = Config()

# Pydantic Models
class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=8000, description="Text to humanize")
    mode: str = Field(default="balanced", regex="^(fast|balanced|thorough)$")
    target_readability: Optional[float] = Field(default=None, ge=5.0, le=20.0)
    preserve_length: bool = Field(default=True)

    @validator('text')
    def validate_text_quality(cls, v):
        if len(v.split()) < 10:
            raise ValueError("Text must contain at least 10 words")
        return v

class HumanizeResponse(BaseModel):
    original_text: str
    humanized_text: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    metrics: Dict[str, float]
    bypass_probability: float = Field(..., ge=0.0, le=1.0)

# Global variables
model_manager: Optional['OptimizedModelManager'] = None
executor: Optional[ThreadPoolExecutor] = None

# Async context manager for application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, executor

    logger.info("Starting humanization service...")

    # Initialize thread pool executor
    executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

    # Initialize model manager
    model_manager = OptimizedModelManager()
    await model_manager.initialize()

    logger.info("Service initialization complete")

    yield

    logger.info("Shutting down humanization service...")

    if executor:
        executor.shutdown(wait=True)

    logger.info("Service shutdown complete")

# Optimized Model Manager (unchanged)
class OptimizedModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizers = {}
        self.processors = {}
        self.initialized = False

    async def initialize(self):
        if self.initialized:
            return

        logger.info("Initializing optimized models...")

        await asyncio.gather(
            self._load_nlp_models(),
            self._load_generation_models(),
            self._load_analysis_models()
        )

        self.initialized = True
        logger.info("Model initialization complete")

    async def _load_nlp_models(self):
        loop = asyncio.get_event_loop()

        def load_spacy():
            nlp = spacy.load("en_core_web_lg")
            nlp.disable_pipes(["ner", "textcat"])
            return nlp

        self.processors['spacy'] = await loop.run_in_executor(
            executor, load_spacy
        )

        def load_grammar_checker():
            return LanguageTool('en-US')

        self.processors['grammar'] = await loop.run_in_executor(
            executor, load_grammar_checker
        )

        def download_nltk_data():
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)

        await loop.run_in_executor(executor, download_nltk_data)

        logger.info("NLP models loaded successfully")

    async def _load_generation_models(self):
        loop = asyncio.get_event_loop()

        def load_optimized_model():
            model_name = "facebook/bart-large"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if config.MODEL_PRECISION == "fp16" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            try:
                model = BetterTransformer.transform(model)
                logger.info("BetterTransformer optimization applied")
            except Exception as e:
                logger.warning(f"BetterTransformer optimization failed: {e}")
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model)
                    logger.info("Torch compile optimization applied")
                except Exception as e:
                    logger.warning(f"Torch compile optimization failed: {e}")
            return tokenizer, model

        self.tokenizers['generator'], self.models['generator'] = await loop.run_in_executor(
            executor, load_optimized_model
        )

        logger.info("Generation models loaded successfully")

    async def _load_analysis_models(self):
        loop = asyncio.get_event_loop()

        def load_perplexity_model():
            model_name = "distilgpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if config.MODEL_PRECISION == "fp16" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer, model

        self.tokenizers['perplexity'], self.models['perplexity'] = await loop.run_in_executor(
            executor, load_perplexity_model
        )

        logger.info("Analysis models loaded successfully")

# Advanced Humanization Pipeline (unchanged)
class AdvancedHumanizationPipeline:
    def __init__(self, model_manager: OptimizedModelManager):
        self.model_manager = model_manager
        self.cache = {}

    @lru_cache(maxsize=1000)
    def _get_synonyms(self, word: str, pos: str) -> List[str]:
        synonyms = set()
        pos_map = {
            'NOUN': wordnet.NOUN,
            'VERB': wordnet.VERB,
            'ADJ': wordnet.ADJ,
            'ADV': wordnet.ADV
        }
        wn_pos = pos_map.get(pos)
        if not wn_pos:
            return []
        for synset in wordnet.synsets(word, pos=wn_pos):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower() and len(synonym) > 2:
                    synonyms.add(synonym)
        return list(synonyms)

    async def _advanced_lexical_transformation(self, text: str) -> str:
        doc = self.model_manager.processors['spacy'](text)
        tokens = []
        for token in doc:
            if (token.is_alpha and not token.is_stop and 
                token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and
                len(token.text) > 3):
                synonyms = self._get_synonyms(token.lemma_, token.pos_)
                if synonyms and random.random() < 0.2:
                    replacement = random.choice(synonyms)
                    if token.is_title:
                        replacement = replacement.title()
                    elif token.is_upper:
                        replacement = replacement.upper()
                    tokens.append(replacement + token.whitespace_)
                else:
                    tokens.append(token.text_with_ws)
            else:
                tokens.append(token.text_with_ws)
        return ''.join(tokens)

    async def _syntactic_restructuring(self, text: str) -> str:
        doc = self.model_manager.processors['spacy'](text)
        sentences = list(doc.sents)
        if len(sentences) < 2:
            return text
        restructured_sentences = []
        for sent in sentences:
            sent_text = sent.text.strip()
            if random.random() < 0.3:
                if self._is_active_voice(sent):
                    transformed = self._to_passive_voice(sent_text)
                    if transformed:
                        sent_text = transformed
                elif len(sent_text.split()) > 20:
                    split_sentences = self._split_long_sentence(sent_text)
                    restructured_sentences.extend(split_sentences)
                    continue
            restructured_sentences.append(sent_text)
        return ' '.join(restructured_sentences)

    def _is_active_voice(self, sent) -> bool:
        for token in sent:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                return True
        return False

    def _to_passive_voice(self, sentence: str) -> Optional[str]:
        if " is " in sentence or " was " in sentence:
            return None
        patterns = [
            (r'(\w+) (\w+ed) (\w+)', r'\3 was \2 by \1'),
            (r'(\w+) (\w+s) (\w+)', r'\3 is \2 by \1'),
        ]
        import re
        for pattern, replacement in patterns:
            if re.search(pattern, sentence):
                return re.sub(pattern, replacement, sentence)
        return None

    def _split_long_sentence(self, sentence: str) -> List[str]:
        conjunctions = [', and ', ', but ', ', or ', ', which ', ', that ']
        for conj in conjunctions:
            if conj in sentence:
                parts = sentence.split(conj, 1)
                if len(parts) == 2:
                    part1 = parts[0].strip() + '.'
                    part2 = parts[1].strip()
                    if not part2[0].isupper():
                        part2 = part2[0].upper() + part2[1:]
                    return [part1, part2]
        return [sentence]

    async def _semantic_enhancement(self, text: str) -> str:
        loop = asyncio.get_event_loop()
        def enhance_text():
            tokenizer = self.model_manager.tokenizers['generator']
            model = self.model_manager.models['generator']
            inputs = tokenizer(
                f"Rewrite this text to be more natural and human-like: {text}",
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=min(len(inputs.input_ids[0]) + 100, 512),
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            enhanced = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if ":" in enhanced:
                enhanced = enhanced.split(":", 1)[1].strip()
            return enhanced
        try:
            enhanced_text = await loop.run_in_executor(executor, enhance_text)
            return enhanced_text if enhanced_text else text
        except Exception as e:
            logger.warning(f"Semantic enhancement failed: {e}")
            return text

    async def _calculate_advanced_metrics(self, text: str) -> Dict[str, float]:
        loop = asyncio.get_event_loop()
        def calculate_metrics():
            metrics = {}
            metrics['perplexity'] = self._calculate_perplexity(text)
            metrics['burstiness'] = self._calculate_burstiness(text)
            metrics['readability'] = textstat.flesch_reading_ease(text)
            metrics['lexical_diversity'] = self._calculate_lexical_diversity(text)
            metrics['semantic_coherence'] = self._calculate_semantic_coherence(text)
            metrics['syntactic_complexity'] = self._calculate_syntactic_complexity(text)
            return metrics
        return await loop.run_in_executor(executor, calculate_metrics)

    def _calculate_perplexity(self, text: str) -> float:
        try:
            tokenizer = self.model_manager.tokenizers['perplexity']
            model = self.model_manager.models['perplexity']
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            return min(perplexity, 1000.0)
        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {e}")
            return 100.0

    def _calculate_burstiness(self, text: str) -> float:
        doc = self.model_manager.processors['spacy'](text)
        sentences = list(doc.sents)
        if len(sentences) < 2:
            return 0.0
        word_lengths = [len([token for token in sent if token.is_alpha]) for sent in sentences]
        char_lengths = [len(sent.text) for sent in sentences]
        word_cv = np.std(word_lengths) / np.mean(word_lengths) if np.mean(word_lengths) > 0 else 0
        char_cv = np.std(char_lengths) / np.mean(char_lengths) if np.mean(char_lengths) > 0 else 0
        burstiness = (word_cv + char_cv) / 2 * 10
        return min(burstiness, 20.0)

    def _calculate_lexical_diversity(self, text: str) -> float:
        doc = self.model_manager.processors['spacy'](text)
        words = [token.lemma_.lower() for token in doc if token.is_alpha]
        if not words:
            return 0.0
        unique_words = set(words)
        return len(unique_words) / len(words)

    def _calculate_semantic_coherence(self, text: str) -> float:
        doc = self.model_manager.processors['spacy'](text)
        sentences = list(doc.sents)
        if len(sentences) < 2:
            return 1.0
        coherence_scores = []
        for i in range(len(sentences) - 1):
            sent1 = sentences[i]
            sent2 = sentences[i + 1]
            words1 = set(token.lemma_.lower() for token in sent1 if token.is_alpha)
            words2 = set(token.lemma_.lower() for token in sent2 if token.is_alpha)
            if words1 and words2:
                overlap = len(words1 & words2) / len(words1 | words2)
                coherence_scores.append(overlap)
        return np.mean(coherence_scores) if coherence_scores else 0.5

    def _calculate_syntactic_complexity(self, text: str) -> float:
        doc = self.model_manager.processors['spacy'](text)
        total_deps = 0
        complex_deps = 0
        for token in doc:
            total_deps += 1
            if token.dep_ in ['acl', 'advcl', 'ccomp', 'xcomp', 'relcl']:
                complex_deps += 1
        complexity = complex_deps / total_deps if total_deps > 0 else 0
        return complexity * 10

    async def _calculate_bypass_probability(self, metrics: Dict[str, float]) -> float:
        scores = []
        perplexity_score = min(metrics['perplexity'] / config.PERPLEXITY_THRESHOLD, 1.0)
        scores.append(perplexity_score * 0.2)
        burstiness_score = min(metrics['burstiness'] / config.BURSTINESS_THRESHOLD, 1.0)
        scores.append(burstiness_score * 0.15)
        lexical_score = min(metrics['lexical_diversity'] / config.LEXICAL_DIVERSITY_TARGET, 1.0)
        scores.append(lexical_score * 0.25)
        semantic_score = min(metrics['semantic_coherence'] / config.SEMANTIC_COHERENCE_TARGET, 1.0)
        scores.append(semantic_score * 0.25)
        complexity_score = min(metrics['syntactic_complexity'] / 5.0, 1.0)
        scores.append(complexity_score * 0.15)
        bypass_probability = sum(scores)
        bypass_probability = 1 / (1 + np.exp(-5 * (bypass_probability - 0.5)))
        return bypass_probability

    async def process(self, text: str, mode: str = "balanced") -> Tuple[str, Dict[str, float], float]:
        start_time = time.time()
        try:
            transformed_text = await self._advanced_lexical_transformation(text)
            restructured_text = await self._syntactic_restructuring(transformed_text)
            if mode in ["balanced", "thorough"]:
                enhanced_text = await self._semantic_enhancement(restructured_text)
            else:
                enhanced_text = restructured_text
            metrics = await self._calculate_advanced_metrics(enhanced_text)
            bypass_probability = await self._calculate_bypass_probability(metrics)
            processing_time = time.time() - start_time
            metrics['processing_time'] = processing_time
            return enhanced_text, metrics, bypass_probability
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            metrics = {
                'perplexity': 50.0,
                'burstiness': 2.0,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
            return text, metrics, 0.3

# FastAPI Application
app = FastAPI(
    title="Advanced AI Text Humanizer",
    description="High-performance text humanization service with advanced AI detection bypass",
    version="3.0.0",
    lifespan=lifespan
)

@app.post("/humanize", response_model=HumanizeResponse)
async def humanize_text(request: HumanizeRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    try:
        pipeline = AdvancedHumanizationPipeline(model_manager)
        humanized_text, metrics, bypass_probability = await pipeline.process(
            request.text, request.mode
        )
        confidence_score = min(
            (metrics.get('perplexity', 50) / 150.0) * 0.4 +
            (metrics.get('burstiness', 2) / 10.0) * 0.3 +
            (metrics.get('lexical_diversity', 0.5) / 0.8) * 0.3,
            1.0
        )
        response = HumanizeResponse(
            original_text=request.text,
            humanized_text=humanized_text,
            confidence_score=confidence_score,
            processing_time=time.time() - start_time,
            metrics=metrics,
            bypass_probability=bypass_probability
        )
        return response
    except Exception as e:
        logger.error(f"Humanization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models_loaded": model_manager.initialized if model_manager else False
    }

@app.get("/metrics")
async def get_metrics():
    return {
        "device": model_manager.device if model_manager else "unknown",
        "models_loaded": len(model_manager.models) if model_manager else 0,
        "config": {
            "perplexity_threshold": config.PERPLEXITY_THRESHOLD,
            "burstiness_threshold": config.BURSTINESS_THRESHOLD,
            "batch_size": config.BATCH_SIZE
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
        reload=False,
        access_log=True
    )
