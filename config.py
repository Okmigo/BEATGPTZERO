# Model and pipeline configuration

# Paraphraser model (T5-small fine-tuned for paraphrasing)
PARAPHRASER_MODEL = "mrm8488/t5-small-finetuned-quora-for-paraphrasing"
PARAPHRASER_NUM_OUTPUTS = 3  # number of paraphrase candidates to generate

# GPT-2 model for perplexity scoring
GPT2_MODEL = "gpt2"  # or "gpt2-medium" for higher capacity (with more memory)

# Detector model (RoBERTa fine-tuned for GPT-2 output detection)
DETECTOR_MODEL = "openai-community/roberta-base-openai-detector"

# SentenceTransformer model for semantic similarity
SEMANTIC_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# Synonym replacement probability (0 = off, up to 1.0)
SYNONYM_PROBABILITY = 0.1

# Similarity threshold for validation (optional use)
SIMILARITY_THRESHOLD = 0.8

# Chunk size (in words) for splitting long text (to stay <~512 tokens)
CHUNK_SIZE = 400

# Flask server port (Cloud Run expects $PORT env)
PORT = 8080
